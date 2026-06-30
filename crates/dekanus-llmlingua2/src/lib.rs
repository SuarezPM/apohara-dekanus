//! LLMLingua2 prompt compression (BERT-base classifier head, INT8 on sm_75).
//!
//! Algorithm: chunk (160 words) → forward classifier head → p_preserve
//! per word → sort desc + force sentence boundaries → top-N.
//!
//! Phase 5: candle BERT-base INT8 path.

#![forbid(unsafe_code)]

pub mod chunker;
pub mod classifier;
pub mod compressor;

pub use compressor::Lingua2Compressor;