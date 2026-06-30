//! KV cache quantization (FWHT 3-bit + Lloyd-Max + TurboQuant PQO variant).
//!
//! Targets sm_75 (RTX 2060 SUPER Turing) using FP16 mma.sync tensor cores.
//! No BF16 (sm_80+), no FP8 HW acceleration (sm_80+).
//!
//! Phase 1: port turboquant-turing FWHT + scalar Lloyd-Max decoder.
//! Phase 5: differential precision (3-bit deep, FP8 shallow).

#![forbid(unsafe_code)]

pub mod fwht;
pub mod quantize;
pub mod dequantize;
pub mod kv_cache;

pub use kv_cache::CompressedKVCache;