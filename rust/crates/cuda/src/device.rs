//! What the library reports about the device it runs on.

use std::fmt;

/// The device the library bound: its streaming multiprocessors, compute capability and memory.
///
/// Read once at load. The values are the device's own; nothing here is a default, so a plan that sizes work by them
/// (a split count of 8 per SM, a graph budget) takes them from this record on whatever machine it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Streaming multiprocessors.
    pub sm_count: u32,
    /// Compute capability, `(major, minor)`.
    pub compute_capability: (u32, u32),
    /// Total device memory in bytes.
    pub total_bytes: u64,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.compute_capability;
        write!(
            formatter,
            "sm_{major}{minor}, {} SMs, {} bytes",
            self.sm_count, self.total_bytes
        )
    }
}
