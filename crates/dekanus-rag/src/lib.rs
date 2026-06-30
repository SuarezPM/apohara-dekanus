//! TurboVec-RAG ANN codec (TurboQuant codec_v8, RAM-optimised, group_size=256).
//!
//! Phase 5: port from Apohara Context Forge turbovec-rag.

#![forbid(unsafe_code)]

pub mod codec;
pub mod store;

pub use codec::TurboVecCodec;
pub use store::TurbovecStore;