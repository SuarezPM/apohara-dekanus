//! AUDIT.md honest ledger primitives + check_honesty runtime guard.
//!
//! Every speed claim must include:
//! - Commit SHA
//! - Hardware fingerprint
//! - Model SHA
//! - active_params_per_token (sparsity ratio)
//! - Profiler dump (ncu or tokio-console)
//!
//! Phase 0: library + check_honesty.sh shell wrapper.

#![forbid(unsafe_code)]

pub mod claim;
pub mod fingerprint;
pub mod ledger;

pub use claim::{Claim, Evidence};
pub use fingerprint::HardwareFingerprint;
pub use ledger::AuditLedger;