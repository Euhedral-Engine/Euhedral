//! The rates a trellis calibration rounds each linear at, checked before the run touches the card or the directory.
//!
//! The allocation of `exl3::allocation` can name a rate the kernels do not have (1, 7 or 8 bits), and the Viterbi
//! scratch must hold one tile of the largest rate the run rounds at (the MTP block's included). Both are refused here,
//! so a run that cannot finish is refused before it starts.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use super::CalibrateError;
use crate::exl3::MTP_BITS;
use crate::exl3::allocation::{Linear, allocate};
use crate::trellis::Rate;

/// The bits per weight of every budgeted linear of a trellis run, and the target they average to.
#[derive(Clone, Debug, PartialEq)]
pub struct Allocation {
    target: f64,
    /// By tensor name (without `.weight`), sorted by name.
    rates: Vec<(String, f64)>,
}

impl Allocation {
    /// The allocation of `linears` over a `layers`-layer stack for the average `target`, when every rate it names is
    /// one the kernels have and `scratch_bytes` of Viterbi scratch holds one tile of each of them and of the MTP rate.
    pub fn refuse(
        linears: &[Linear],
        layers: usize,
        target: f64,
        scratch_bytes: u64,
    ) -> Result<Allocation, CalibrateError> {
        let allocated =
            allocate(linears, layers, target).map_err(|error| CalibrateError::Invalid(error.to_string()))?;
        let mut rates = allocated;
        rates.sort_by(|left, right| left.0.cmp(&right.0));
        for (name, bits) in &rates {
            Rate::new(*bits).map_err(|error| {
                CalibrateError::Invalid(format!(
                    "--bits {target} gives {name} {bits} bits per weight, which the trellis kernels do not have ({error})"
                ))
            })?;
        }
        let largest = rates
            .iter()
            .map(|(_, bits)| *bits)
            .chain([f64::from(MTP_BITS)])
            .map(|bits| Rate::new(bits).map(|rate| (bits, rate.scratch_bytes(1))))
            .try_fold((0.0, 0), |most: (f64, u64), found| {
                found.map(|found| if found.1 > most.1 { found } else { most })
            })
            .map_err(|error| CalibrateError::Invalid(error.to_string()))?;
        if scratch_bytes < largest.1 {
            return Err(CalibrateError::Invalid(format!(
                "--viterbi-scratch-mib: {scratch_bytes} bytes of Viterbi scratch hold no tile at {} bits (one needs {})",
                largest.0, largest.1
            )));
        }
        Ok(Allocation { target, rates })
    }

    /// The average bits per weight asked for.
    pub fn target(&self) -> f64 {
        self.target
    }

    /// The rate of `tensor`, if it is budgeted.
    pub fn rate(&self, tensor: &str) -> Option<f64> {
        self.rates
            .binary_search_by(|(name, _)| name.as_str().cmp(tensor))
            .ok()
            .map(|at| self.rates[at].1)
    }

    /// Every rate by tensor name, as the writer takes them.
    pub fn by_linear(&self) -> HashMap<String, f64> {
        self.rates.iter().cloned().collect()
    }

    /// Names the allocation in the resume key: another allocation is another run.
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(self.target.to_le_bytes());
        for (name, bits) in &self.rates {
            hash.update(name.as_bytes());
            hash.update([0]);
            hash.update(bits.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn linear(name: &str, layer: usize, numel: u64) -> Linear {
        Linear {
            name: name.to_owned(),
            layer,
            numel,
            priority: 0,
            group: name.to_owned(),
        }
    }

    fn stack() -> Vec<Linear> {
        (0..4)
            .flat_map(|layer| {
                ["up", "down"]
                    .into_iter()
                    .map(move |role| linear(&format!("layers.{layer}.{role}"), layer, 1 << 20))
            })
            .collect()
    }

    const ENOUGH: u64 = 1 << 30;

    #[test]
    fn a_target_the_kernels_have_every_rate_of_is_allocated() {
        let allocation = Allocation::refuse(&stack(), 4, 3.0, ENOUGH).unwrap();
        assert_eq!(allocation.target(), 3.0);
        assert_eq!(allocation.rate("layers.0.up"), Some(3.0));
        assert_eq!(allocation.by_linear().len(), 8);
    }

    #[test]
    fn a_target_that_is_not_a_positive_number_is_refused() {
        for target in [f64::NAN, f64::INFINITY, 0.0, -2.0] {
            assert!(Allocation::refuse(&stack(), 4, target, ENOUGH).is_err(), "{target}");
        }
    }

    #[test]
    fn a_linear_past_the_last_layer_is_refused() {
        let mut linears = stack();
        linears.push(linear("layers.9.up", 9, 1 << 20));
        assert!(Allocation::refuse(&linears, 4, 3.0, ENOUGH).is_err());
    }

    #[test]
    fn a_target_whose_rates_the_kernels_lack_is_refused_naming_the_rate() {
        // 8 bits for every linear: the kernels stop at 6.
        let refused = Allocation::refuse(&stack(), 4, 8.0, ENOUGH).unwrap_err().to_string();
        assert!(refused.contains("8 bits"), "{refused}");
        assert!(Allocation::refuse(&stack(), 4, 1.0, ENOUGH).is_err());
    }

    fn tile(bits: f64) -> u64 {
        Rate::new(bits).unwrap().scratch_bytes(1)
    }

    #[test]
    fn a_scratch_below_one_tile_of_the_rate_that_needs_most_is_refused() {
        // A tile needs the more scratch the fewer bits it has: a run of 2 bits is bound by its own rate.
        assert!(Allocation::refuse(&stack(), 4, 2.0, tile(2.0)).is_ok());
        let refused = Allocation::refuse(&stack(), 4, 2.0, tile(2.0) - 1)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("--viterbi-scratch-mib"), "{refused}");
    }

    #[test]
    fn the_mtp_rate_counts_towards_the_scratch_when_it_needs_the_most() {
        assert!(Allocation::refuse(&stack(), 4, 6.0, tile(f64::from(MTP_BITS))).is_ok());
        assert!(Allocation::refuse(&stack(), 4, 6.0, tile(f64::from(MTP_BITS)) - 1).is_err());
    }

    #[test]
    fn the_digest_follows_the_rates() {
        let three = Allocation::refuse(&stack(), 4, 3.0, ENOUGH).unwrap();
        let again = Allocation::refuse(&stack(), 4, 3.0, ENOUGH).unwrap();
        let four = Allocation::refuse(&stack(), 4, 4.0, ENOUGH).unwrap();
        assert_eq!(three.digest(), again.digest());
        assert_ne!(three.digest(), four.digest());
    }
}
