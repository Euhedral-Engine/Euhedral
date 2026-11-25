//! The copy of a host-held weight into its staging slot, ahead of the stage that reads it.

use std::fmt;
use std::sync::Arc;

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{DevicePtr, HostWeightsArena, KernelOps};

use super::{Launch, Quantum, fatal};

/// Copies one weight from the pinned host arena into a staging slot.
///
/// The stage reads nothing of the quantum: it is ordered by the plan's edges alone, behind the stage that last read
/// the slot and ahead of the one that reads the weight. It never joins a captured graph, as a staged plan's views are
/// not captured.
pub struct WeightTransfer {
    name: String,
    source: Option<Arc<HostWeightsArena>>,
    offset: u64,
    bytes: u64,
    destination: DevicePtr,
}

impl WeightTransfer {
    /// The copy of `bytes` bytes at `offset` of `source` to `destination`. A transfer of a plan over described weights
    /// has no `source` and fails when it runs.
    pub fn new(
        name: String,
        source: Option<Arc<HostWeightsArena>>,
        offset: u64,
        bytes: u64,
        destination: DevicePtr,
    ) -> Self {
        Self {
            name,
            source,
            offset,
            bytes,
            destination,
        }
    }

    /// The bytes the stage copies.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The slot the stage fills.
    pub fn destination(&self) -> DevicePtr {
        self.destination
    }
}

impl fmt::Debug for WeightTransfer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightTransfer")
            .field("name", &self.name)
            .field("offset", &self.offset)
            .field("bytes", &self.bytes)
            .finish_non_exhaustive()
    }
}

impl Launch for WeightTransfer {
    fn launch(&self, _quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| fatal(format!("{} has no host arena to be staged from", self.name)))?;
        let offset = usize::try_from(self.offset).map_err(fatal)?;
        let bytes = usize::try_from(self.bytes).map_err(fatal)?;
        // SAFETY: the stage holds the arena, which is only read once the model is loaded, so its bytes in the range
        // stay unmodified; the plan orders the copy behind the last read of the slot and ahead of the next, so no
        // other copy writes the slot meanwhile.
        unsafe { kernels.copy_host_weights_to_device(self.destination, source, offset, bytes) }.map_err(fatal)
    }
}

super::eager!(WeightTransfer);
