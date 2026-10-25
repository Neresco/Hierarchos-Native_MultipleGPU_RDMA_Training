//! CPU offload store for model parameters.
//!
//! Between training windows the model's parameter GPU buffers are copied to
//! host memory and replaced with 1-element stubs, freeing device VRAM. Before
//! the next window's forward/backward the buffers are re-allocated and the
//! values copied back. Because Vulkan descriptor sets are bound at dispatch
//! time (not baked in at kernel creation), swapping the buffer handles inside
//! the leaf ops is transparent to the compiled token tape.
//!
//! The store is keyed by the canonical trainable parameter name (the same name
//! used by the optimizer registry), so offload and restore are symmetric and
//! order-independent.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::vulkan::{GpuBuffer, VulkanDevice};

/// Build a collision-free store key from an op-level prefix and a canonical
/// parameter name. An empty prefix returns the canonical name unchanged.
pub fn prefixed_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// CPU-side parameter store used by [`super::param_offload`]. Holds each
/// offloaded parameter's f32 values in host memory.
#[derive(Default)]
pub struct ParamOffloadStore {
    /// Maps canonical param name -> (element count, host values).
    params: HashMap<String, (usize, Vec<f32>)>,
}

impl ParamOffloadStore {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// True when no parameters are currently offloaded.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// Number of distinct parameters currently held on the CPU.
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Total host-memory bytes currently held (4 bytes per f32).
    pub fn offloaded_bytes(&self) -> usize {
        self.params
            .values()
            .map(|(len, _)| len * std::mem::size_of::<f32>())
            .sum()
    }

    /// Whether a parameter is currently offloaded.
    pub fn contains(&self, name: &str) -> bool {
        self.params.contains_key(name)
    }

    /// Copy a live GPU parameter buffer into host memory and record it. The
    /// caller is responsible for then replacing the GPU buffer with a stub.
    pub fn absorb(&mut self, device: &VulkanDevice, name: &str, buffer: &GpuBuffer) -> Result<usize> {
        let len = buffer.f32_capacity();
        let values = buffer.read_f32(len)?;
        if values.len() != len {
            bail!(
                "offload readback for {name:?} returned {} values, expected {len}",
                values.len()
            );
        }
        let prev = self.params.insert(name.to_string(), (len, values));
        if prev.is_some() {
            bail!("offload store already contains {name:?}; double offload");
        }
        let _ = device; // device retained for API symmetry with restore
        Ok(len)
    }

    /// Reallocate a GPU buffer from a previously absorbed parameter and copy the
    /// values back. Returns the fresh buffer for the caller to install.
    pub fn release(&self, device: &VulkanDevice, name: &str) -> Result<GpuBuffer> {
        let Some((len, values)) = self.params.get(name) else {
            bail!("offload store has no entry for {name:?}; double restore");
        };
        let buffer = GpuBuffer::from_f32(device, values)?;
        if buffer.f32_capacity() != *len {
            bail!(
                "offload restore for {name:?} allocated {} elements, expected {len}",
                buffer.f32_capacity()
            );
        }
        Ok(buffer)
    }

    /// Drop a parameter from the store once it has been restored to the GPU.
    /// Returns the number of host bytes released.
    pub fn forget(&mut self, name: &str) -> usize {
        self.params
            .remove(name)
            .map(|(len, _)| len * std::mem::size_of::<f32>())
            .unwrap_or(0)
    }
}

/// A `ParamOffloadStore` whose CPU backing is a memory-mapped file instead of a
/// heap `Vec`. The OS page cache balances RAM against SSD automatically:
/// tensors written recently stay resident in RAM, and cold tensors are paged
/// out to the backing file on SSD. This is the standard "CPU offload with SSD
/// backing" strategy — it lets the offloaded optimizer/parameter state exceed
/// physical RAM by spilling to fast NVMe, at the cost of page faults when a
/// paged-out tensor is restored.
///
/// The `capacity` (bytes) must cover the sum of all simultaneously-absorbed
/// tensors. The backing file is created sparse, so it only consumes SSD space
/// for regions actually written.
pub struct MmapOffloadStore {
    file: File,
    ptr: *mut u8,
    capacity: usize,
    /// Running cursor (bytes) for the next absorb. Reset to zero whenever the
    /// store becomes empty, so each offload cycle reuses the same layout.
    next_offset: usize,
    /// name -> (offset_bytes, len_bytes)
    tensors: HashMap<String, (usize, usize)>,
}

// The store owns a raw mmap pointer and a `File`; both are cleaned up in `Drop`.
// It is single-owner (not `Sync`), matching the offload cycle usage.
impl MmapOffloadStore {
    /// Create (or truncate) the backing file at `path` and map `capacity` bytes
    /// read/write. The file is kept open for the store's lifetime and unmapped
    /// on drop.
    pub fn new(path: &Path, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            bail!("mmap offload capacity must be non-zero");
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("opening mmap offload backing file {}", path.display()))?;
        file.set_len(capacity as u64)
            .with_context(|| "pre-sizing sparse mmap offload file")?;
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                capacity,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(io::Error::last_os_error()).context("mmap of offload backing file failed");
        }
        Ok(Self {
            file,
            ptr: ptr as *mut u8,
            capacity,
            next_offset: 0,
            tensors: HashMap::new(),
        })
    }

    fn ptr_at(&self, offset: usize) -> *mut f32 {
        unsafe { self.ptr.add(offset) as *mut f32 }
    }

    /// Absorb a live GPU parameter buffer into the mapped region. Returns the
    /// bytes stored. The caller then replaces the GPU buffer with a stub.
    pub fn absorb(
        &mut self,
        device: &VulkanDevice,
        name: &str,
        buffer: &GpuBuffer,
    ) -> Result<usize> {
        if self.tensors.contains_key(name) {
            bail!("mmap offload store already holds tensor {name:?}");
        }
        let len = buffer.f32_capacity();
        let bytes = len * 4;
        // A new offload cycle starts when the store is empty; reuse the layout.
        if self.tensors.is_empty() {
            self.next_offset = 0;
        }
        if self.next_offset + bytes > self.capacity {
            bail!(
                "mmap offload capacity {} bytes too small for {name:?} ({bytes} bytes); cursor {}",
                self.capacity,
                self.next_offset
            );
        }
        let values = buffer.read_f32(len)?;
        let dst = self.ptr_at(self.next_offset);
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), dst, len) };
        self.tensors.insert(name.to_string(), (self.next_offset, bytes));
        self.next_offset += bytes;
        Ok(bytes)
    }

    /// Re-allocate a GPU buffer and copy the tensor's bytes back out.
    pub fn release(&self, device: &VulkanDevice, name: &str) -> Result<GpuBuffer> {
        let (offset, bytes) = *self
            .tensors
            .get(name)
            .with_context(|| format!("mmap offload store has no tensor {name:?}"))?;
        let len = bytes / 4;
        let src = self.ptr_at(offset);
        let values = unsafe { std::slice::from_raw_parts(src, len).to_vec() };
        GpuBuffer::from_f32(device, &values)
    }

    /// Whether a tensor is currently offloaded.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn offloaded_bytes(&self) -> usize {
        self.tensors.values().map(|(_, b)| *b).sum()
    }

    pub fn param_count(&self) -> usize {
        self.tensors.len()
    }

    /// Forget a tensor's entry. Bytes are not reclaimed from the mapping; the
    /// cursor resets on the next cycle once the store is empty.
    pub fn forget(&mut self, name: &str) {
        self.tensors.remove(name);
    }
}

impl Drop for MmapOffloadStore {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.capacity) };
        // The file is closed when `self.file` drops. It is left on disk so the
        // caller can inspect or delete it; the kernel writes back dirty pages
        // during unmap.
    }
}
