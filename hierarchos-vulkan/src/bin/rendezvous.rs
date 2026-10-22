//! hierarchos-vulkan-rendezvous — a tiny TCP barrier for multi-node training.
//!
//! Each training rank checks in *after* its (slow) Vulkan device init but
//! *before* it attempts to form the RDMA ring. The server accepts exactly
//! `--world-size` connections, then releases every rank with a "GO" line.
//! This eliminates the startup skew between fast and slow nodes: no rank ever
//! tries to connect to a peer that has not finished initializing yet.
//!
//! A structured state file (default ~/hierarchos-rdz/state.json) is updated on
//! every check-in and when GO is sent, so the web dashboard can render a live
//! "Rendezvous" overview.
//!
//! Usage:
//!   hierarchos-vulkan-rendezvous --listen 0.0.0.0:18599 --world-size 2 \
//!     [--state-file ~/hierarchos-rdz/state.json] [--timeout 1800]
//!
//! Then every rank is launched with `--rendezvous <this-host>:18599`.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct CheckIn {
    addr: String,
    ts: u64,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut listen = None;
    let mut world_size: usize = 0;
    let mut timeout_secs: u64 = 1800;
    let mut state_file: Option<PathBuf> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--listen" => listen = Some(args.next().expect("missing --listen value")),
            "--world-size" => world_size = args.next().expect("missing --world-size value").parse()?,
            "--timeout" => timeout_secs = args.next().expect("missing --timeout value").parse()?,
            "--state-file" => state_file = Some(PathBuf::from(args.next().expect("missing --state-file value"))),
            other => {
                eprintln!("unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }
    let listen = listen.expect("--listen is required");
    anyhow::ensure!(world_size >= 1, "--world-size must be >= 1");

    // Default state file: ~/hierarchos-rdz/state.json
    let state_file = state_file.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join("hierarchos-rdz").join("state.json")
    });
    if let Some(parent) = state_file.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let start_ts = now_secs();
    write_state(
        &state_file,
        &listen,
        world_size,
        &[],
        false,
        false,
        "waiting",
        start_ts,
    );

    let listener = TcpListener::bind(listen.clone())?;
    println!("rendezvous: listening on {listen} for {world_size} ranks (timeout {timeout_secs}s)");

    let mut conns: Vec<(TcpStream, String)> = Vec::with_capacity(world_size);
    let mut check_ins: Vec<CheckIn> = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);

    while conns.len() < world_size {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining == Duration::ZERO {
            write_state(&state_file, &listen, world_size, &check_ins, false, true, "timeout", now_secs());
            anyhow::bail!(
                "rendezvous: timed out after {timeout_secs}s with {}/{} ranks checked in",
                conns.len(),
                world_size
            );
        }
        let _ = listener.set_nonblocking(true);
        match listener.accept() {
            Ok((sock, addr)) => {
                let _ = sock.set_nonblocking(false);
                let addr_str = addr.to_string();
                println!("rendezvous: check-in from {addr} ({}/{})", conns.len() + 1, world_size);
                check_ins.push(CheckIn { addr: addr_str.clone(), ts: now_secs() });
                write_state(&state_file, &listen, world_size, &check_ins, false, false, "waiting", now_secs());
                conns.push((sock, addr_str));
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => anyhow::bail!("rendezvous: accept failed: {e}"),
        }
    }

    println!("rendezvous: all {world_size} ranks ready — sending GO");
    for (sock, addr) in &mut conns {
        sock.write_all(b"GO\n")?;
        sock.flush()?;
        println!("rendezvous: GO -> {addr}");
    }
    write_state(&state_file, &listen, world_size, &check_ins, true, false, "go-sent", now_secs());
    println!("rendezvous: complete; ranks may now form the ring");
    Ok(())
}

fn write_state(
    path: &PathBuf,
    listen: &str,
    world_size: usize,
    check_ins: &[CheckIn],
    go_sent: bool,
    timed_out: bool,
    status: &str,
    ts: u64,
) {
    let body = serde_json::json!({
        "listen": listen,
        "world_size": world_size,
        "checked_in": check_ins.len(),
        "check_ins": check_ins.iter().map(|c| serde_json::json!({ "addr": c.addr, "ts": c.ts })).collect::<Vec<_>>(),
        "go_sent": go_sent,
        "timed_out": timed_out,
        "status": status,
        "ts": ts,
    });
    let _ = std::fs::write(path, body.to_string());
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
