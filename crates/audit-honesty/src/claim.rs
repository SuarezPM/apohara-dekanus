//! Stub module — Claim + Evidence types land in Phase 1.
pub struct Claim {
    pub text: String,
    pub commit_sha: String,
}

pub struct Evidence {
    pub artifact_path: String,
}

impl Claim {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(),
            commit_sha: String::new(),
        }
    }
}