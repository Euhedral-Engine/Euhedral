use super::allocation::Allocation;

/// An allocation whose KL against the reference was measured: what the fidelity model is fitted to.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    /// The allocation that was scored.
    pub allocation: Allocation,
    /// Its measured KL.
    pub kl: f64,
}

impl Measured {
    /// `allocation` with the KL `kl` measured for it.
    pub fn new(allocation: Allocation, kl: f64) -> Measured {
        Measured { allocation, kl }
    }
}
