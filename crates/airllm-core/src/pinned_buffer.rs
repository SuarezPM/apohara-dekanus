//! Pinned host memory buffer for DMA-eligible H2D transfers.
//!
//! Phase 1 uses `cudarc::CudaContext::alloc_pinned()` for DMA-eligible memory.
//! Avoids the mmap+mlock+cuMemHostRegister anti-pattern (cudarc does it automatically).

pub struct PinnedHostBuffer {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    /// Allocate a pinned host buffer of `size` bytes.
    /// Phase 1: cudarc::CudaContext::alloc_pinned wrapper.
    pub fn alloc(size: usize) -> Self {
        // Skeleton: in Phase 1, this calls cudarc::CudaContext::alloc_pinned(size)
        // and stores the resulting device pointer.
        Self {
            ptr: std::ptr::null_mut(),
            len: size,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        // Phase 1: call cudarc free
        if !self.ptr.is_null() {
            // SAFETY: ptr was allocated by cudarc alloc_pinned, only freed here
            unsafe {
                // cudarc::CudaContext::free_pinned(self.ptr)
            }
        }
    }
}