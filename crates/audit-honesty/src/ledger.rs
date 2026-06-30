//! Stub module — AuditLedger lands in Phase 1.
pub struct AuditLedger {
    entries: Vec<String>,
}

impl AuditLedger {
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    pub fn append(&mut self, entry: &str) {
        self.entries.push(entry.to_string());
    }

    pub fn entries(&self) -> &[String] {
        &self.entries
    }
}