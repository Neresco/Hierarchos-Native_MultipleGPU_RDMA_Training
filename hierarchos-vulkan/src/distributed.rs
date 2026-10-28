//! Network RDMA transport for cross-machine gradient exchange (Phase R1).
//!
//! This module brings in self-generated `ibverbs` bindings plus a C shim for the
//! static-inline verbs (`ibv_post_send`/`ibv_post_recv`/`ibv_poll_cq`) and opaque
//! field accessors. It is the foundation for the `NetworkRdma` gradient transport
//! backend: a replica rank on a remote machine computes gradient shards and streams
//! them to the primary rank over RoCE; the primary receives and reduces them.
//!
//! Design notes:
//! - Public-header bindings generate full public ABI structs; implementation
//!   structs (ibv_qp/ibv_context/ibv_mr/ibv_cq) are opaque — the shim reaches in.
//! - Endpoint exchange (QP number + 16-byte GID) is newline-delimited JSON over a
//!   TCP control channel, with explicit flush (serde_json::to_writer does not
//!   flush a TcpStream promptly and deadlocks a full-duplex handshake).
//! - Senders bound outstanding work requests to the send-queue depth (ENOSPC guard).

mod ffi {
    #![allow(
        non_snake_case,
        non_camel_case_types,
        dead_code,
        non_upper_case_globals,
        clippy::all
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
use ffi::*;
use ffi::ibv_qp_type::IBV_QPT_RC;
use ffi::ibv_qp_state::{IBV_QPS_INIT, IBV_QPS_RTR, IBV_QPS_RTS};
use ffi::ibv_wc_status::IBV_WC_SUCCESS;
use ffi::ibv_wr_opcode::IBV_WR_SEND;

extern "C" {
    fn shim_poll_cq(cq: *mut ibv_cq, num_entries: i32, wc: *mut ibv_wc) -> i32;
    fn shim_post_send(
        qp: *mut ibv_qp,
        wr: *const ibv_send_wr,
        bad_wr: *mut *mut ibv_send_wr,
    ) -> i32;
    fn shim_post_recv(
        qp: *mut ibv_qp,
        wr: *const ibv_recv_wr,
        bad_wr: *mut *mut ibv_recv_wr,
    ) -> i32;
    fn shim_qp_num(qp: *mut ibv_qp) -> u32;
    fn shim_mr_lkey(mr: *mut ibv_mr) -> u32;
    fn shim_mr_rkey(mr: *mut ibv_mr) -> u32;
}

use std::io::{BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

/// RoCEv2 GID index (verified via perftest; commonly 3 for RoCEv2).
pub const DEFAULT_GID_INDEX: u32 = 3;
const PORT: u8 = 1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Endpoint {
    pub qpn: u32,
    pub gid: [u8; 16],
}

impl Endpoint {
    /// Newline-delimited JSON (gid as a [u8;16] array) — matches the R0 spike
    /// framing so the two are interoperable.
    fn to_json(&self) -> String {
        let mut s = serde_json::to_string(self).expect("serialize endpoint");
        s.push('\n');
        s
    }

    fn from_line(line: &str) -> anyhow::Result<Self> {
        serde_json::from_str(line.trim())
            .map_err(|e| anyhow::anyhow!("parse endpoint: {e}"))
    }
}

/// A connected RDMA link: one QP between this host and a peer, plus a CQ.
pub struct RdmaLink {
    ctx: *mut ibv_context,
    pd: *mut ibv_pd,
    send_cq: *mut ibv_cq,
    recv_cq: *mut ibv_cq,
    qp: *mut ibv_qp,
    #[allow(dead_code)]
    device_name: String,
}

unsafe impl Send for RdmaLink {}

impl RdmaLink {
    /// Open a device and create a QP, then connect it to `peer` (endpoint
    /// exchange over TCP at `control_addr`). `is_listener` side accepts; the
    /// other side connects.
    pub fn connect(
        device_name: &str,
        control_addr: SocketAddr,
        is_listener: bool,
    ) -> anyhow::Result<Self> {
        let (link, _sock) = Self::connect_with_control(device_name, control_addr, is_listener)?;
        Ok(link)
    }

    /// Same as [`connect`], but keeps the TCP control socket open after the
    /// endpoint exchange and returns it. Callers use it for per-message
    /// metadata (gradient shard headers, weight-state skeletons).
    pub fn connect_with_control(
        device_name: &str,
        control_addr: SocketAddr,
        is_listener: bool,
    ) -> anyhow::Result<(Self, TcpStream)> {
        if is_listener {
            let listener = TcpListener::bind(control_addr)?;
            return Self::connect_with_bound_listener(device_name, listener);
        }
        let sock = Self::connect_retry(control_addr, 90)?;
        Self::connect_from_socket(device_name, sock, false)
    }

    /// TCP connect with retries: peers may still be doing one-time Vulkan
    /// setup (~10 min) before they reach the RDMA control bind.
    pub(crate) fn connect_retry(addr: SocketAddr, attempts: u32) -> anyhow::Result<TcpStream> {
        let mut last_err: Option<std::io::Error> = None;
        for i in 0..attempts {
            match TcpStream::connect(addr) {
                Ok(sock) => {
                    if i > 0 {
                        eprintln!("  [rdma] control connect succeeded on attempt {}", i + 1);
                    }
                    return Ok(sock);
                }
                Err(e) => {
                    if i % 15 == 0 {
                        eprintln!("  [rdma] waiting for peer control at {addr} (attempt {}/{}: {e})", i + 1, attempts);
                    }
                    last_err = Some(e);
                    std::thread::sleep(std::time::Duration::from_secs(2));
                }
            }
        }
        Err(anyhow::anyhow!(
            "connect to {addr} failed after {attempts} attempts: {}",
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
    }

    /// Listener variant using an already-bound control listener. Lets the
    /// caller complete other setup (e.g. signal a peer) before accepting.
    pub fn connect_with_bound_listener(
        device_name: &str,
        listener: TcpListener,
    ) -> anyhow::Result<(Self, TcpStream)> {
        Self::accept_from_listener(device_name, listener)
    }

    /// Accept a control connection on an already-bound listener and complete
    /// the QP handshake (listener side writes its endpoint first).
    pub fn accept_from_listener(
        device_name: &str,
        listener: TcpListener,
    ) -> anyhow::Result<(Self, TcpStream)> {
        let (mut sock, addr) = listener.accept()?;
        eprintln!("  [rdma] control accepted from {addr}");
        Self::connect_from_socket(device_name, sock, true)
    }

    fn connect_from_socket(
        device_name: &str,
        mut sock: TcpStream,
        wrote_first: bool,
    ) -> anyhow::Result<(Self, TcpStream)> {
        let ctx = Self::open_device(device_name)?;
        let pd = unsafe { ibv_alloc_pd(ctx) };
        anyhow::ensure!(!pd.is_null(), "ibv_alloc_pd");
        let send_cq = unsafe {
            ibv_create_cq(ctx, 4096, std::ptr::null_mut(), std::ptr::null_mut(), 0)
        };
        let recv_cq = unsafe {
            ibv_create_cq(ctx, 4096, std::ptr::null_mut(), std::ptr::null_mut(), 0)
        };
        anyhow::ensure!(!send_cq.is_null() && !recv_cq.is_null(), "ibv_create_cq");
        let qp = Self::make_qp(ctx, send_cq, recv_cq, pd)?;
        let qpn = unsafe { shim_qp_num(qp) };
        let gid = Self::query_gid(ctx)?;
        let local = Endpoint { qpn, gid };

        // Endpoint exchange over TCP. The listener side writes first.
        let remote = if wrote_first {
            sock.write_all(local.to_json().as_bytes())?;
            sock.flush()?;
            let mut line = String::new();
            BufReader::new(&mut sock).read_line(&mut line)?;
            Endpoint::from_line(&line)?
        } else {
            let mut line = String::new();
            BufReader::new(&mut sock).read_line(&mut line)?;
            let remote = Endpoint::from_line(&line)?;
            sock.write_all(local.to_json().as_bytes())?;
            sock.flush()?;
            remote
        };

        Self::connect_qp(qp, &remote)?;

        Ok((
            Self {
                ctx,
                pd,
                send_cq,
                recv_cq,
                qp,
                device_name: device_name.to_owned(),
            },
            sock,
        ))
    }

    fn open_device(name: &str) -> anyhow::Result<*mut ibv_context> {
        let mut n: i32 = 0;
        let list = unsafe { ibv_get_device_list(&mut n) };
        anyhow::ensure!(n > 0, "no RDMA devices");
        let mut ctx: *mut ibv_context = std::ptr::null_mut();
        for i in 0..n {
            let dev = unsafe { *list.offset(i as isize) };
            let cname = unsafe { std::ffi::CStr::from_ptr(ibv_get_device_name(dev)) };
            if cname.to_string_lossy() == name {
                ctx = unsafe { ibv_open_device(dev) };
                break;
            }
        }
        unsafe { ibv_free_device_list(list) };
        anyhow::ensure!(!ctx.is_null(), "device {name} not found");
        Ok(ctx)
    }

    fn query_gid(ctx: *mut ibv_context) -> anyhow::Result<[u8; 16]> {
        let mut gid: ibv_gid = unsafe { std::mem::zeroed() };
        unsafe {
            ibv_query_gid(ctx, PORT, DEFAULT_GID_INDEX as i32, &mut gid);
            Ok(gid.raw)
        }
    }

    fn make_qp(
        ctx: *mut ibv_context,
        send_cq: *mut ibv_cq,
        recv_cq: *mut ibv_cq,
        pd: *mut ibv_pd,
    ) -> anyhow::Result<*mut ibv_qp> {
        unsafe {
            let mut attr: ibv_qp_init_attr = std::mem::zeroed();
            attr.send_cq = send_cq;
            attr.recv_cq = recv_cq;
            attr.srq = std::ptr::null_mut();
            attr.cap.max_send_wr = 4096;
            attr.cap.max_recv_wr = 4096;
            attr.cap.max_send_sge = 1;
            attr.cap.max_recv_sge = 1;
            attr.qp_type = IBV_QPT_RC;
            attr.sq_sig_all = 0;
            let qp = ibv_create_qp(pd, &mut attr);
            anyhow::ensure!(!qp.is_null(), "ibv_create_qp");
            Ok(qp)
        }
    }

    fn connect_qp(qp: *mut ibv_qp, remote: &Endpoint) -> anyhow::Result<()> {
        unsafe {
            let mut a: ibv_qp_attr = std::mem::zeroed();
            // INIT
            a.qp_state = IBV_QPS_INIT;
            a.port_num = PORT;
            a.pkey_index = 0;
            a.qp_access_flags =
                (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as u32;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_PKEY_INDEX.0
                    | ibv_qp_attr_mask::IBV_QP_PORT.0
                    | ibv_qp_attr_mask::IBV_QP_ACCESS_FLAGS.0) as i32,
            )?;
            // RTR
            a.qp_state = IBV_QPS_RTR;
            a.path_mtu = IBV_MTU_1024;
            a.dest_qp_num = remote.qpn;
            a.rq_psn = 0;
            a.max_dest_rd_atomic = 16;
            a.min_rnr_timer = 12;
            a.ah_attr.dlid = 0;
            a.ah_attr.sl = 0;
            a.ah_attr.src_path_bits = 0;
            a.ah_attr.is_global = 1;
            a.ah_attr.port_num = PORT;
            a.ah_attr.grh.dgid.raw.copy_from_slice(&remote.gid);
            a.ah_attr.grh.sgid_index = DEFAULT_GID_INDEX as u8;
            a.ah_attr.grh.hop_limit = 255;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_AV.0
                    | ibv_qp_attr_mask::IBV_QP_PATH_MTU.0
                    | ibv_qp_attr_mask::IBV_QP_DEST_QPN.0
                    | ibv_qp_attr_mask::IBV_QP_RQ_PSN.0
                    | ibv_qp_attr_mask::IBV_QP_MAX_DEST_RD_ATOMIC.0
                    | ibv_qp_attr_mask::IBV_QP_MIN_RNR_TIMER.0) as i32,
            )?;
            // RTS
            a.qp_state = IBV_QPS_RTS;
            a.sq_psn = 0;
            a.timeout = 14;
            a.retry_cnt = 7;
            a.rnr_retry = 7;
            a.max_rd_atomic = 16;
            Self::modify(
                qp,
                &mut a,
                (ibv_qp_attr_mask::IBV_QP_STATE.0
                    | ibv_qp_attr_mask::IBV_QP_SQ_PSN.0
                    | ibv_qp_attr_mask::IBV_QP_TIMEOUT.0
                    | ibv_qp_attr_mask::IBV_QP_RETRY_CNT.0
                    | ibv_qp_attr_mask::IBV_QP_RNR_RETRY.0
                    | ibv_qp_attr_mask::IBV_QP_MAX_QP_RD_ATOMIC.0) as i32,
            )?;
            Ok(())
        }
    }

    #[inline]
    fn modify(qp: *mut ibv_qp, attr: &mut ibv_qp_attr, mask: i32) -> anyhow::Result<()> {
        let r = unsafe { ibv_modify_qp(qp, attr, mask) };
        anyhow::ensure!(r == 0, "ibv_modify_qp: {}", std::io::Error::last_os_error());
        Ok(())
    }

    /// Register a host buffer as an MR and return (mr, lkey, base address).
    pub fn register(&self, buf: &mut [u8]) -> anyhow::Result<RegisteredBuffer> {
        let mr = unsafe {
            ibv_reg_mr(
                self.pd,
                buf.as_mut_ptr() as *mut std::os::raw::c_void,
                buf.len(),
                (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_WRITE.0
                    | ibv_access_flags::IBV_ACCESS_REMOTE_READ.0) as i32,
            )
        };
        anyhow::ensure!(!mr.is_null(), "ibv_reg_mr");
        let lkey = unsafe { shim_mr_lkey(mr) };
        let rkey = unsafe { shim_mr_rkey(mr) };
        Ok(RegisteredBuffer {
            mr,
            lkey,
            rkey,
            base: buf.as_ptr() as u64,
            len: buf.len(),
        })
    }

    /// Post a single receive into `mr` with the given `wr_id`.
    pub fn post_receive_wr(&self, mr: &RegisteredBuffer, wr_id: u64) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base;
            sge.length = mr.len as u32;
            sge.lkey = mr.lkey;
            let mut wr: ibv_recv_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            let mut bad = std::ptr::null_mut();
            if shim_post_recv(self.qp, &wr, &mut bad) != 0 {
                return Err(anyhow::anyhow!(
                    "post_recv: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }

    /// Block (poll) until one receive completion is available; return it.
    pub fn poll_recv_one(&self) -> anyhow::Result<ibv_wc> {
        loop {
            let mut wcs: Vec<ibv_wc> =
                vec![unsafe { std::mem::zeroed::<ibv_wc>() }; 1];
            let n = unsafe { shim_poll_cq(self.recv_cq, 1, wcs.as_mut_ptr()) };
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            if n == 1 {
                let wc = wcs[0];
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("recv WC status {:?}", wc.status));
                }
                return Ok(wc);
            }
            std::thread::yield_now();
        }
    }

    /// Post a SEND from a byte range of a registered buffer.
    pub fn post_send_wr(
        &self,
        mr: &RegisteredBuffer,
        offset: usize,
        len: usize,
        wr_id: u64,
    ) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base + offset as u64;
            sge.length = len as u32;
            sge.lkey = mr.lkey;
            let mut wr: ibv_send_wr = std::mem::zeroed();
            wr.wr_id = wr_id;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_SEND;
            wr.send_flags = ibv_send_flags::IBV_SEND_SIGNALED.0;
            let mut bad = std::ptr::null_mut();
            if shim_post_send(self.qp, &wr, &mut bad) != 0 {
                return Err(anyhow::anyhow!(
                    "post_send: {}",
                    std::io::Error::last_os_error()
                ));
            }
        }
        Ok(())
    }

    /// Post `count` receives of `mr.len()` bytes each onto the given MR.
    pub fn post_receives(&self, mr: &RegisteredBuffer, count: u64) -> anyhow::Result<()> {
        unsafe {
            let mut sge: ibv_sge = std::mem::zeroed();
            sge.addr = mr.base;
            sge.length = mr.len as u32;
            sge.lkey = mr.lkey;
            for i in 0..count {
                let mut wr: ibv_recv_wr = std::mem::zeroed();
                wr.wr_id = i;
                wr.sg_list = &mut sge;
                wr.num_sge = 1;
                let mut bad = std::ptr::null_mut();
                if shim_post_recv(self.qp, &wr, &mut bad) != 0 {
                    return Err(anyhow::anyhow!(
                        "post_recv: {}",
                        std::io::Error::last_os_error()
                    ));
                }
            }
        }
        Ok(())
    }

    /// Poll receive completions; returns how many were completed.
    pub fn poll_recv(&self, max: usize) -> anyhow::Result<usize> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.recv_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("recv WC status {:?}", wc.status));
                }
            }
            Ok(n as usize)
        }
    }

    /// Poll send completions; returns the completed wr_ids (all SUCCESS).
    pub fn poll_send_wrs(&self, max: usize) -> anyhow::Result<Vec<u64>> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.send_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let mut ids = Vec::with_capacity(n as usize);
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("send WC status {:?}", wc.status));
                }
                ids.push(wc.wr_id);
            }
            Ok(ids)
        }
    }

    /// Poll send completions; returns how many were completed.
    pub fn poll_send(&self, max: usize) -> anyhow::Result<usize> {
        unsafe {
            let mut wcs: Vec<ibv_wc> = vec![std::mem::zeroed(); max];
            let n = shim_poll_cq(self.send_cq, max as i32, wcs.as_mut_ptr());
            if n < 0 {
                return Err(anyhow::anyhow!(
                    "poll_cq: {}",
                    std::io::Error::last_os_error()
                ));
            }
            for wc in wcs.iter().take(n as usize) {
                if wc.status != IBV_WC_SUCCESS {
                    return Err(anyhow::anyhow!("send WC status {:?}", wc.status));
                }
            }
            Ok(n as usize)
        }
    }
}

impl Drop for RdmaLink {
    fn drop(&mut self) {
        unsafe {
            if !self.qp.is_null() {
                ibv_destroy_qp(self.qp);
            }
            if !self.send_cq.is_null() {
                ibv_destroy_cq(self.send_cq);
            }
            if !self.recv_cq.is_null() {
                ibv_destroy_cq(self.recv_cq);
            }
            if !self.pd.is_null() {
                ibv_dealloc_pd(self.pd);
            }
            if !self.ctx.is_null() {
                ibv_close_device(self.ctx);
            }
        }
    }
}

/// Receives a stream of gradient chunks over an RdmaLink.
///
/// The peer (replica rank) sends chunks of `chunk_len` f32 values in tensor
/// order. This type manages a RING of registered receive buffers. An RC QP
/// delivers completions in send order, and each ring slot re-posts with the
/// same wr_id (= its index), so `receive_into` drains exactly the next chunk.
pub struct NetworkGradientSource {
    link: RdmaLink,
    /// Ring of registered receive buffers; rings[i] uses wr_id i.
    rings: Vec<RegisteredBuffer>,
    /// Owning backing memory for each ring (keeps the MR valid).
    #[allow(dead_code)]
    ring_memory: Vec<Vec<u8>>,
    chunk_len: usize,
}

impl NetworkGradientSource {
    /// Connect (as listener) to a replica gradient sender.
    pub fn connect(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, _sock) = RdmaLink::connect_with_control(device_name, control_addr, true)?;
        Self::finish(link, chunk_len, ring_count)
    }

    /// Listener connect that keeps the TCP control socket open for
    /// per-message metadata.
    pub fn connect_with_control(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let listener = TcpListener::bind(control_addr)?;
        Self::connect_with_bound_listener(device_name, listener, chunk_len, ring_count)
    }

    /// Listener connect on an already-bound control socket, keeping the TCP
    /// control socket open for per-message metadata.
    pub fn connect_with_bound_listener(
        device_name: &str,
        listener: TcpListener,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, sock) = RdmaLink::connect_with_bound_listener(device_name, listener)?;
        let source = Self::finish(link, chunk_len, ring_count)?;
        Ok((source, sock))
    }

    fn finish(
        link: RdmaLink,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        let mut rings = Vec::with_capacity(ring_count);
        let mut ring_memory = Vec::with_capacity(ring_count);
        for _ in 0..ring_count {
            let mut buf = vec![0u8; chunk_len * std::mem::size_of::<f32>()];
            let mr = link.register(&mut buf)?;
            rings.push(mr);
            ring_memory.push(buf);
        }
        // Post the initial ring so the sender has buffer to run into.
        for (i, ring) in rings.iter().enumerate() {
            link.post_receive_wr(ring, i as u64)?;
        }
        Ok(Self {
            link,
            rings,
            ring_memory,
            chunk_len,
        })
    }

    /// Block until one message arrives, copy it into `out`, and re-post that
    /// ring slot. Uses the completion's byte_len so final (shorter) chunks of
    /// a stream are handled correctly. Returns byte count.
    pub fn receive_bytes_into(&mut self, out: &mut [u8]) -> anyhow::Result<usize> {
        let wc = self
            .link
            .poll_recv_one()?; // blocking-ish: returns next completion
        let ring_idx = wc.wr_id as usize;
        anyhow::ensure!(
            ring_idx < self.rings.len(),
            "unexpected wr_id {ring_idx}"
        );
        let received = wc.byte_len as usize;
        anyhow::ensure!(
            received <= out.len(),
            "receive buffer too small: {} < {}",
            out.len(),
            received
        );
        let ring = &self.rings[ring_idx];
        // Copy the freshly-arrived chunk out of the ring MR.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ring.base as *const u8,
                out.as_mut_ptr() as *mut u8,
                received,
            );
        }
        // Re-post this ring slot (same wr_id) for the next chunk.
        self.link.post_receive_wr(ring, ring_idx as u64)?;
        Ok(received)
    }

    /// Block until one chunk arrives, copy it into `out`, and re-post that
    /// ring slot for the next chunk. Returns f32 count.
    pub fn receive_into(&mut self, out: &mut [f32]) -> anyhow::Result<usize> {
        let bytes = self.receive_bytes_into(bytemuck::cast_slice_mut(out))?;
        Ok(bytes / std::mem::size_of::<f32>())
    }
}

/// Sends a stream of chunks over an RdmaLink.
///
/// Mirrors [`NetworkGradientSource`] on the send side: a ring of registered
/// buffers; `send_chunk` picks the next slot, blocks if that slot still has
/// an in-flight send, copies the data in, and posts a signaled SEND. RC QPs
/// deliver messages in send order, so the receiver sees the exact chunk
/// sequence the sender posted.
pub struct NetworkSendChannel {
    link: RdmaLink,
    rings: Vec<RegisteredBuffer>,
    /// Owning backing memory for each ring (keeps the MR valid).
    #[allow(dead_code)]
    ring_memory: Vec<Vec<u8>>,
    chunk_len: usize,
    next_slot: usize,
    /// In-flight send wr_id per slot (None = free).
    outstanding: Vec<Option<u64>>,
}

impl NetworkSendChannel {
    /// Wire chunk width in f32 values.
    pub fn chunk_len(&self) -> usize {
        self.chunk_len
    }

    pub fn connect(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        is_listener: bool,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(chunk_len > 0, "chunk_len must be positive");
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, _sock) = RdmaLink::connect_with_control(device_name, control_addr, is_listener)?;
        Self::finish(link, chunk_len, ring_count)
    }

    /// Connect that keeps the TCP control socket open for per-message
    /// metadata.
    pub fn connect_with_control(
        device_name: &str,
        control_addr: std::net::SocketAddr,
        is_listener: bool,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<(Self, TcpStream)> {
        anyhow::ensure!(chunk_len > 0, "chunk_len must be positive");
        anyhow::ensure!(ring_count > 0, "ring_count must be positive");
        let (link, sock) = RdmaLink::connect_with_control(device_name, control_addr, is_listener)?;
        let channel = Self::finish(link, chunk_len, ring_count)?;
        Ok((channel, sock))
    }

    fn finish(
        link: RdmaLink,
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        let mut rings = Vec::with_capacity(ring_count);
        let mut ring_memory = Vec::with_capacity(ring_count);
        for _ in 0..ring_count {
            let mut buf = vec![0u8; chunk_len * std::mem::size_of::<f32>()];
            let mr = link.register(&mut buf)?;
            rings.push(mr);
            ring_memory.push(buf);
        }
        Ok(Self {
            link,
            rings,
            ring_memory,
            chunk_len,
            next_slot: 0,
            outstanding: vec![None; ring_count],
        })
    }

    fn free_completed(&mut self) -> anyhow::Result<()> {
        for wr in self.link.poll_send_wrs(64)? {
            let i = wr as usize;
            if i < self.outstanding.len() {
                self.outstanding[i] = None;
            }
        }
        Ok(())
    }

    /// Send one chunk (<= chunk_len f32s). Blocks while the destination slot
    /// is still in flight; the receiver observes messages in send order.
    pub fn send_chunk(&mut self, data: &[f32]) -> anyhow::Result<()> {
        anyhow::ensure!(
            !data.is_empty() && data.len() <= self.chunk_len,
            "send chunk size {} out of range 1..={} f32s",
            data.len(),
            self.chunk_len
        );
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.rings.len();
        // Wait until this slot's previous send has completed.
        while self.outstanding[slot].is_some() {
            self.free_completed()?;
            if self.outstanding[slot].is_some() {
                std::thread::yield_now();
            }
        }
        let ring = &self.rings[slot];
        let bytes = data.len() * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ring.base as *mut u8,
                bytes,
            );
        }
        self.link.post_send_wr(ring, 0, bytes, slot as u64)?;
        self.outstanding[slot] = Some(slot as u64);
        Ok(())
    }

    /// Wait until all in-flight sends have completed.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        while self.outstanding.iter().any(|o| o.is_some()) {
            self.free_completed()?;
            std::thread::yield_now();
        }
        Ok(())
    }
}

/// A registered host memory region used for RDMA staging.
pub struct RegisteredBuffer {
    mr: *mut ibv_mr,
    pub lkey: u32,
    pub rkey: u32,
    pub base: u64,
    pub len: usize,
}

unsafe impl Send for RegisteredBuffer {}

impl Drop for RegisteredBuffer {
    fn drop(&mut self) {
        unsafe {
            if !self.mr.is_null() {
                ibv_dereg_mr(self.mr);
            }
        }
    }
}



/// Send one contiguous f32 block over a channel, split into wire chunks.
pub fn send_block(channel: &mut NetworkSendChannel, data: &[f32]) -> anyhow::Result<()> {
    for chunk in data.chunks(channel.chunk_len()) {
        channel.send_chunk(chunk)?;
    }
    Ok(())
}

/// Receive one contiguous f32 block (len f32s) over a channel.
pub fn recv_block(channel: &mut NetworkGradientSource, len: usize) -> anyhow::Result<Vec<f32>> {
    let mut out = vec![0f32; len];
    let mut cursor = 0usize;
    while cursor < len {
        let got = channel.receive_into(&mut out[cursor..])?;
        cursor += got;
    }
    Ok(out)
}

fn rdz_debug() -> bool {
    std::env::var_os("RING_DEBUG").is_some()
}

/// Send `send_data` to the successor and receive `recv_len` f32s from the
/// predecessor, INTERLEAVED chunk-by-chunk. This keeps both directions draining
/// each other's send rings as they fill, which avoids IBV_WC_RNR_RETRY_EXC_ERR
/// (receiver-not-ready) that occurs when one rank bulk-sends an entire block
/// while its peer is also bulk-sending and not yet receiving.
pub fn exchange_blocks(
    send: &mut NetworkSendChannel,
    recv: &mut NetworkGradientSource,
    send_data: &[f32],
    recv_len: usize,
) -> anyhow::Result<Vec<f32>> {
    let chunk = send.chunk_len();
    let mut recv_out = vec![0f32; recv_len];
    let mut sc = 0usize;
    let mut rc = 0usize;
    if rdz_debug() {
        eprintln!("[dbg] exchange_blocks enter send={} recv={}", send_data.len(), recv_len);
    }
    let mut iter = 0usize;
    while sc < send_data.len() || rc < recv_len {
        if sc < send_data.len() {
            let n = chunk.min(send_data.len() - sc);
            if rdz_debug() && iter < 4 {
                eprintln!("[dbg] exchange SEND iter {} (sc={})", iter, sc / chunk);
            }
            send.send_chunk(&send_data[sc..sc + n])?;
            sc += n;
        }
        if rc < recv_len {
            let n = chunk.min(recv_len - rc);
            if rdz_debug() && iter < 4 {
                eprintln!("[dbg] exchange RECV iter {} (rc={})", iter, rc / chunk);
            }
            let got = recv.receive_into(&mut recv_out[rc..rc + n])?;
            rc += got;
        }
        iter += 1;
    }
    if rdz_debug() {
        eprintln!("[dbg] exchange_blocks EXIT after {} iters", iter);
    }
    Ok(recv_out)
}

/// One rank's position in a ring of `world_size` ranks.
///
/// The ring gives every rank one send channel to its successor
/// ((rank+1) % world_size) and one receive channel from its predecessor
/// ((rank+world_size-1) % world_size). Port scheme: rank r listens on
/// `port_base + r`; the predecessor connects there. Bootstrap is deadlock-free
/// because every rank binds its listener before connecting to its successor.
pub struct RingTransport {
    pub send_to_successor: NetworkSendChannel,
    pub recv_from_predecessor: NetworkGradientSource,
    pub send_ctl: TcpStream,
    /// Persistent BufReader over the predecessor control stream. A fresh
    /// BufReader per read_line would drop buffered lines that a fast peer
    /// already sent, silently desynchronizing the ring allgather.
    recv_ctl: std::io::BufReader<TcpStream>,
    #[allow(dead_code)]
    rank: usize,
    #[allow(dead_code)]
    world_size: usize,
}

impl RingTransport {
    pub fn connect(
        device_name: &str,
        rank: usize,
        world_size: usize,
        port_base: u16,
        peer_ips: &[String],
        chunk_len: usize,
        ring_count: usize,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(world_size >= 2, "ring requires at least 2 ranks");
        anyhow::ensure!(rank < world_size, "rank out of range");
        let succ = (rank + 1) % world_size;
        let pred = (rank + world_size - 1) % world_size;

        // Phase 1: bind our listener for the predecessor.
        let listen_addr: SocketAddr = format!("0.0.0.0:{}", port_base + rank as u16).parse()?;
        let listener = TcpListener::bind(listen_addr)?;
        eprintln!(
            "  [ring] rank {rank}/{world_size} listening on :{} (pred {pred})",
            port_base + rank as u16
        );

        // Phase 2: accept from the predecessor in a separate thread. It blocks
        // until the predecessor connects; running it off the main thread is what
        // breaks the circular handshake wait (this rank's main thread can
        // connect to its successor while the accept thread waits for the
        // predecessor, concurrently).
        let device_name_accept = device_name.to_owned();
        let accept_handle = std::thread::spawn(move || -> anyhow::Result<(RdmaLink, TcpStream)> {
            RdmaLink::accept_from_listener(&device_name_accept, listener)
        });

        // Phase 3: connect to the successor on the main thread (with retry).
        // If the successor never appears, connect_retry fails and this function
        // returns Err, exiting the process and killing the accept thread too.
        let conn_addr: SocketAddr = format!("{}:{}", peer_ips[succ], port_base + succ as u16).parse()?;
        let conn_sock = RdmaLink::connect_retry(conn_addr, 90)?;
        let (send_link, send_ctl) = RdmaLink::connect_from_socket(device_name, conn_sock, false)?;
        let send_channel = NetworkSendChannel::finish(send_link, chunk_len, ring_count)?;

        // Phase 4: join the accept thread now that the send side is up.
        let (recv_link, recv_ctl) = accept_handle
            .join()
            .map_err(|_| anyhow::anyhow!("ring accept thread panicked"))??;
        let recv_channel = NetworkGradientSource::finish(recv_link, chunk_len, ring_count)?;

        eprintln!(
            "  [ring] rank {rank} connected: send->rank{succ}, recv<-rank{pred}"
        );
        Ok(Self {
            send_to_successor: send_channel,
            recv_from_predecessor: recv_channel,
            send_ctl,
            recv_ctl: std::io::BufReader::new(recv_ctl),
            rank,
            world_size,
        })
    }

    /// Send a small newline-terminated JSON message to the successor.
    pub fn send_json(&mut self, line: &str) -> anyhow::Result<()> {
        let mut msg = line.to_string();
        msg.push('\n');
        self.send_ctl.write_all(msg.as_bytes())?;
        self.send_ctl.flush()?;
        Ok(())
    }

    /// Receive a small JSON line from the predecessor.
    pub fn recv_json(&mut self) -> anyhow::Result<String> {
        let mut line = String::new();
        use std::io::BufRead;
        self.recv_ctl.read_line(&mut line)?;
        return Ok(line.trim().to_string());
    }
}

/// Ring allreduce (reduce-scatter + allgather) over a flat f32 buffer.
/// On return every rank holds the element-wise sum of all ranks' inputs.
/// Deterministic: block rotation order depends only on (rank, world_size).
/// `data` is mutated in place (local input -> global sum).
pub fn ring_allreduce_sum(
    transport: &mut RingTransport,
    rank: usize,
    world_size: usize,
    data: &mut Vec<f32>,
) -> anyhow::Result<()> {
    let n = world_size;
    anyhow::ensure!(n >= 2, "ring allreduce needs >= 2 ranks");
    let total = data.len();
    let bs = (total + n - 1) / n; // ceil block size; last block may be smaller
    let range_of = |b: usize| (b * bs)..(b * bs + bs).min(total);
    let modn = |x: isize| (((x % n as isize) + n as isize) % n as isize) as usize;

    if rdz_debug() {
        eprintln!("[dbg] rank {} === PHASE: ring_allreduce_sum start (total={} chunks) ===", rank, total / 262144);
    }
    // Phase 1: reduce-scatter (n-1 steps). After it, rank R holds the fully
    // reduced block (R+1) mod n. Each step sends the send-block and receives the
    // recv-block interleaved so the two directions drain each other's rings.
    for s in 0..n - 1 {
        let send_b = modn(rank as isize - s as isize);
        let recv_b = modn(rank as isize - s as isize - 1);
        let sr = range_of(send_b);
        let rr = range_of(recv_b);
        if rdz_debug() {
            eprintln!("[dbg] rank {} reduce-scatter step {} send_block {} recv_block {}", rank, s, send_b, recv_b);
        }
        let received = exchange_blocks(
            &mut transport.send_to_successor,
            &mut transport.recv_from_predecessor,
            &data[sr],
            rr.len(),
        )?;
        for (a, b) in data[rr].iter_mut().zip(received.iter()) {
            *a = *a + b;
        }
    }

    // Phase 2: allgather (n-1 steps). After it, every rank holds all n reduced
    // blocks.
    for t in 0..n - 1 {
        let send_b = modn((rank + 1) as isize - t as isize);
        let recv_b = modn(rank as isize - t as isize);
        let sr = range_of(send_b);
        let rr = range_of(recv_b);
        let received = exchange_blocks(
            &mut transport.send_to_successor,
            &mut transport.recv_from_predecessor,
            &data[sr],
            rr.len(),
        )?;
        data[rr].copy_from_slice(&received);
    }
    Ok(())
}

/// The block index that rank `rank` owns (fully reduced) after a
/// [`ring_reduce_scatter`]. The scatter places reduced block `(rank+1) mod n`
/// on rank `rank` (same convention as phase 1 of `ring_allreduce_sum`).
pub fn reduce_scatter_owned_block(rank: usize, world_size: usize) -> usize {
    let r = (rank as isize + 1) % world_size as isize;
    (((r + world_size as isize) % world_size as isize) as usize)
}

/// Ring reduce-scatter: on entry every rank holds the full f32 buffer; after
/// the call rank `r` holds the fully-reduced (element-wise world sum) block
/// [`reduce_scatter_owned_block(r)`] and all other blocks are stale.
///
/// This is the gradient-scatter primitive for ZeRO-3: each rank does a
/// redundant backward producing the full gradient, and reduce-scatter leaves
/// each rank with only its owned slice of the summed gradient.
pub fn ring_reduce_scatter(
    transport: &mut RingTransport,
    rank: usize,
    world_size: usize,
    data: &mut Vec<f32>,
) -> anyhow::Result<()> {
    let n = world_size;
    anyhow::ensure!(n >= 2, "ring reduce-scatter needs >= 2 ranks");
    let total = data.len();
    let bs = (total + n - 1) / n; // ceil block size; last block may be smaller
    let range_of = |b: usize| (b * bs)..(b * bs + bs).min(total);
    let modn = |x: isize| (((x % n as isize) + n as isize) % n as isize) as usize;

    // Same rotation as phase 1 of `ring_allreduce_sum`: at step s rank r sends
    // block (r-s) and reduces received into block (r-s-1). After n-1 steps rank
    // r holds the fully-reduced block (r+1) mod n.
    for s in 0..n - 1 {
        let send_b = modn(rank as isize - s as isize);
        let recv_b = modn(rank as isize - s as isize - 1);
        let sr = range_of(send_b);
        let rr = range_of(recv_b);
        let received = exchange_blocks(
            &mut transport.send_to_successor,
            &mut transport.recv_from_predecessor,
            &data[sr],
            rr.len(),
        )?;
        for (a, b) in data[rr].iter_mut().zip(received.iter()) {
            *a = *a + b;
        }
    }
    Ok(())
}

/// Segment owner whose payload rank `rank` receives at ring step `step`.
///
/// At step s every rank forwards segment owner `(r - s) mod n`; rank r's
/// predecessor is `(r - 1) mod n`, which at step s forwards owner
/// `((r - 1) - s) mod n = (r - s - 1) mod n`. So rank r receives owner
/// `(r - s - 1) mod n`.
fn ring_recv_owner(rank: usize, n: usize, step: usize) -> usize {
    let r = (rank as isize - step as isize - 1) % n as isize;
    ((r + n as isize) % n as isize) as usize
}

/// Segment owner rank `rank` forwards to its successor at ring step `step`
/// (`(rank - step) mod n`).
fn ring_send_owner(rank: usize, n: usize, step: usize) -> usize {
    let r = (rank as isize - step as isize) % n as isize;
    ((r + n as isize) % n as isize) as usize
}

/// One pure step of the unequal-segment ring allgather rotation.
///
/// `data` is the rank's working buffer; `received` is the segment payload just
/// read from the predecessor. This validates `received` against the expected
/// segment and copies it into `data[seg[recv_owner]]`. The rotation is
/// deterministic: at step s of an n-rank ring, rank r holds segment owner
/// `modn(r - s)`, receives owner `modn(r - s - 1)`, and forwards owner
/// `modn(r - s + 1)`. After n-1 steps every rank owns the full buffer.
fn ring_allgather_sharded_step(
    data: &mut Vec<f32>,
    seg: &[std::ops::Range<usize>],
    rank: usize,
    n: usize,
    step: usize,
    received: &[f32],
) {
    let recv_owner = ring_recv_owner(rank, n, step);
    let recv_range = &seg[recv_owner];
    debug_assert_eq!(
        received.len(),
        recv_range.len(),
        "sharded allgather step {step}: received {} f32s for owner {recv_owner}, expected {}",
        received.len(),
        recv_range.len()
    );
    data[recv_range.clone()].copy_from_slice(received);
}

/// Ring allgather of an f32 buffer split into UNEQUAL per-rank segments.
///
/// `seg[r]` is the byte-range of the full buffer owned by rank r. On entry,
/// rank r has filled `data[seg[r]]` with its own values. After the call,
/// every rank's `data` holds the full buffer with `data[seg[r]]` taken from
/// rank r. Values are copied (never reduced). The n-1 ring steps interleave
/// sends and receives chunk-by-chunk via [`exchange_blocks`], draining each
/// peer's send ring to avoid RNR retries.
///
/// Requires: n >= 2, `seg.len() == n`, segments covering `data` contiguously
/// in owner order (seg[0].start == 0, seg[k].end == seg[k+1].start,
/// seg[n-1].end == data.len()).
pub fn ring_allgather_sharded(
    transport: &mut RingTransport,
    rank: usize,
    world_size: usize,
    seg: &[std::ops::Range<usize>],
    data: &mut Vec<f32>,
) -> anyhow::Result<()> {
    let n = world_size;
    anyhow::ensure!(n >= 2, "ring allgather needs >= 2 ranks");
    anyhow::ensure!(rank < n, "rank out of range");
    anyhow::ensure!(seg.len() == n, "segment count {} != world size {n}", seg.len());
    let mut cursor = 0usize;
    for r in 0..n {
        anyhow::ensure!(
            seg[r].start == cursor,
            "sharded allgather segment {r} starts at {}, expected {}",
            seg[r].start,
            cursor
        );
        anyhow::ensure!(
            seg[r].end <= data.len(),
            "sharded allgather segment {r} exceeds buffer length {}",
            data.len()
        );
        cursor = seg[r].end;
    }
    anyhow::ensure!(
        cursor == data.len(),
        "sharded allgather segments cover {} of {} f32s",
        cursor,
        data.len()
    );

    for s in 0..n - 1 {
        let send_owner = ring_send_owner(rank, n, s);
        let recv_len = seg[ring_recv_owner(rank, n, s)].len();
        let received = exchange_blocks(
            &mut transport.send_to_successor,
            &mut transport.recv_from_predecessor,
            &data[seg[send_owner].clone()],
            recv_len,
        )?;
        ring_allgather_sharded_step(data, seg, rank, n, s, &received);
    }
    Ok(())
}

/// Allgather a list of small (rank, json) items around the ring over TCP.
/// Every rank starts with its own item; after n-1 steps each rank holds all n,
/// sorted by rank for canonical merge order.
pub fn ring_allgather_items(
    transport: &mut RingTransport,
    rank: usize,
    world_size: usize,
    own_json: &str,
) -> anyhow::Result<Vec<(usize, String)>> {
    let n = world_size;
    let mut held: Vec<(usize, String)> = vec![(rank, own_json.to_string())];
    for (t, _t) in (0..n - 1).enumerate() {
        if rdz_debug() {
            eprintln!("[dbg] rank {} meta-gather step {t}", rank);
        }
        let payload = serde_json::to_string(&held)?;
        transport.send_json(&payload)?;
        let recv_str = transport.recv_json()?;
        let incoming: Vec<(usize, String)> = serde_json::from_str(&recv_str)?;
        for (r, m) in incoming {
            if !held.iter().any(|(hr, _)| *hr == r) {
                held.push((r, m));
            }
        }
        if rdz_debug() {
            eprintln!("[dbg] rank {} meta-gather step {} DONE (held={})", rank, t, held.len());
        }
    }
    if rdz_debug() {
        eprintln!("[dbg] rank {} === PHASE: meta-gather complete ===", rank);
    }
    held.sort_by_key(|(r, _)| *r);
    Ok(held)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simulate the full ring with in-memory channels: rank r receives from
    /// rank (r-1) % n and sends to rank (r+1) % n. Drives the exact rotation
    /// helpers used by [`ring_allgather_sharded`] (ring_recv_owner /
    /// ring_send_owner / ring_allgather_sharded_step) and asserts every rank
    /// ends with the same full buffer.
    #[test]
    fn sharded_allgather_rotation_distributes_full_buffer() {
        for &n in &[2usize, 3, 4, 7] {
            // Unequal segments: owner r's segment length = 4 * (r + 1).
            let lens: Vec<usize> = (0..n).map(|r| 4 * (r + 1)).collect();
            let total: usize = lens.iter().sum();
            let mut seg: Vec<std::ops::Range<usize>> = Vec::with_capacity(n);
            let mut cursor = 0usize;
            for len in &lens {
                seg.push(cursor..cursor + *len);
                cursor += *len;
            }
            // Expected full buffer: element i gets value (i + 1) so any
            // transposition is detectable.
            let expected: Vec<f32> = (1..=total).map(|i| i as f32).collect();

            // Initial buffers: only each rank's own segment is filled.
            let mut buffers: Vec<Vec<f32>> = (0..n)
                .map(|r| {
                    let mut buf = vec![f32::NAN; total];
                    buf[seg[r].clone()].copy_from_slice(&expected[seg[r].clone()]);
                    buf
                })
                .collect();

            for s in 0..n - 1 {
                let mut received_from_pred: Vec<Option<Vec<f32>>> = vec![None; n];
                for r in 0..n {
                    let pred = (r + n - 1) % n;
                    let send_owner = ring_send_owner(pred, n, s);
                    let payload = buffers[pred][seg[send_owner].clone()].to_vec();
                    received_from_pred[r] = Some(payload);
                }
                for r in 0..n {
                    let payload = received_from_pred[r]
                        .take()
                        .expect("every rank receives each step");
                    ring_allgather_sharded_step(&mut buffers[r], &seg, r, n, s, &payload);
                }
            }

            for r in 0..n {
                assert_eq!(
                    buffers[r], expected,
                    "n={n} rank {r}: sharded allgather rotation produced wrong buffer"
                );
            }
        }
    }

    /// Simulate the [`ring_reduce_scatter`] rotation in memory: each rank starts
    /// with a buffer where element i carries a distinct value derived from
    /// (rank, block, element). After n-1 ring steps, rank r must hold the full
    /// world sum in its owned block (r+1) mod n.
    #[test]
    fn reduce_scatter_rotation_reduces_owned_block() {
        for &n in &[2usize, 3, 4, 7] {
            let bs = 5usize; // elements per block
            let total = n * bs;
            let modn = |x: isize, n: usize| (((x % n as isize) + n as isize) % n as isize) as usize;
            // rank r, block b, element e -> value r*1000 + b*10 + e.
            let value = |r: usize, b: usize, e: usize| (r * 1000 + b * 10 + e) as f32;
            let mut buffers: Vec<Vec<f32>> = (0..n)
                .map(|r| (0..total).map(|i| value(r, i / bs, i % bs)).collect())
                .collect();
            for s in 0..n - 1 {
                // Each rank sends block send_b to its successor; the successor
                // reduces it into its recv_b (= modn(pred - s), i.e. send_b).
                let mut received: Vec<Vec<f32>> = vec![vec![0.0; total]; n];
                for r in 0..n {
                    let send_b = modn(r as isize - s as isize, n);
                    let succ = (r + 1) % n;
                    let recv_b = modn(succ as isize - s as isize - 1, n);
                    assert_eq!(send_b, recv_b);
                    for e in 0..bs {
                        received[succ][recv_b * bs + e] = buffers[r][send_b * bs + e];
                    }
                }
                for r in 0..n {
                    let recv_b = modn(r as isize - s as isize - 1, n);
                    for e in 0..bs {
                        buffers[r][recv_b * bs + e] += received[r][recv_b * bs + e];
                    }
                }
            }
            for r in 0..n {
                let owned = reduce_scatter_owned_block(r, n);
                for e in 0..bs {
                    let expected: f32 = (0..n).map(|rr| value(rr, owned, e)).sum();
                    assert!(
                        (buffers[r][owned * bs + e] - expected).abs() < 1e-3,
                        "n={n} rank {r} owned block {owned} elem {e}: {} != {}",
                        buffers[r][owned * bs + e],
                        expected
                    );
                }
            }
        }
    }

    /// Segment-owner math must agree between the two helpers at every step:
    /// what rank r forwards, its successor must expect to receive.
    #[test]
    fn sharded_allgather_owner_math_consistent() {
        for n in 2..=6usize {
            for rank in 0..n {
                for s in 0..n - 1 {
                    let succ = (rank + 1) % n;
                    assert_eq!(
                        ring_send_owner(rank, n, s),
                        ring_recv_owner(succ, n, s),
                        "n={n} rank={rank} step={s}: sender/recipient owner mismatch"
                    );
                }
            }
        }
    }
}
