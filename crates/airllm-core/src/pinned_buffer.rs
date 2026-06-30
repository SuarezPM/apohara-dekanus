//! Pinned host memory buffer for DMA-eligible H2D transfers (Phase 1 placeholder).
//!
//! Phase 1: CPU-only. This struct is a placeholder for the Phase 2 pinned buffer
//! that will use `cudarc::CudaContext::alloc_pinned()` for DMA-eligible memory.
//!
//! Phase 2 design notes:
//! - cudarc does pinned allocation automatically (cuMemAllocHost = page-locked for DMA)
//! - Avoid the mmap + mlock + cuMemHostRegister anti-pattern
//! - Real impl will use raw pointers + unsafe, requiring `#![allow(unsafe_code)]`
//!   or moving this module to a separate crate with appropriate safety contract

/// CPU-only placeholder. Phase 2 will wrap cudarc::CudaContext::alloc_pinned result.
pub struct PinnedHostBuffer {
    bytes: Vec<u8>,
}

impl PinnedHostBuffer {
    /// Allocate a buffer of `size` bytes (CPU heap in Phase 1).
    pub fn alloc(size: usize) -> Self {
        Self {
            bytes: vec![0u8; size],
        }
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes
    }
}