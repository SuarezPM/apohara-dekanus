//! Stub module — HardwareFingerprint lands in Phase 1.
pub struct HardwareFingerprint {
    pub gpu_model: String,
    pub gpu_vram_mib: u32,
    pub cpu_model: String,
    pub ram_mib: u64,
}

impl HardwareFingerprint {
    pub fn unknown() -> Self {
        Self {
            gpu_model: "unknown".to_string(),
            gpu_vram_mib: 0,
            cpu_model: "unknown".to_string(),
            ram_mib: 0,
        }
    }
}