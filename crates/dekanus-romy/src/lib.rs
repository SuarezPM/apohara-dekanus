//! ROMY multi-agent safety protocol (cache_salt + Z3-proven INV-15).
//!
//! Inherited from Apohara Context Forge.
//! Phase 5+: full Z3 proofs for INV-10..15.

#![forbid(unsafe_code)]

pub mod cache_salt;
pub mod invariants;

pub use cache_salt::CacheSalt;
pub use invariants::Inv15;