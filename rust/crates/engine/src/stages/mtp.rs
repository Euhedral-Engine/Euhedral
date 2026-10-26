//! The MTP stem (`docs/speculation/MTP_CONTRACT.md` section 2, `Stages.MtpStem`): row `r` of the packed output is
//! `[RMSNorm(1+w)(embedding r, embedding_norm); RMSNorm(1+w)(seed r, hidden_norm)]`, the seed rows coming from the
//! quantum (the base hidden rows for a catch-up, the previous MTP hidden for a recursive row).

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{DevicePtr, Submission, ops};

use super::{DraftSeed, Launch, Quantum, eager, fatal, launch_u32};
use crate::workspace::BufferSlot;

/// Where the stem's two halves land in a packed row, in bytes: the embedding half first, then the hidden half.
fn half_offsets(hidden_bytes: u64) -> [u64; 2] {
    [0, hidden_bytes]
}

/// The MTP stem: normalizes the embedding rows and the seed rows, and packs them side by side for the MTP layer's
/// input projection.
#[derive(Clone, Debug)]
pub struct MtpStem {
    embedding: BufferSlot,
    embedding_norm: DevicePtr,
    hidden_norm: DevicePtr,
    normalized: BufferSlot,
    packed: BufferSlot,
}

impl MtpStem {
    /// The stem over the embedding rows of `embedding`, normalized with `embedding_norm`, and the seed rows,
    /// normalized with `hidden_norm`. `normalized` holds one normalized half at a time and `packed` receives
    /// `[embedding; hidden]` rows twice the hidden size wide.
    pub fn new(
        embedding: BufferSlot,
        embedding_norm: DevicePtr,
        hidden_norm: DevicePtr,
        normalized: BufferSlot,
        packed: BufferSlot,
    ) -> Self {
        Self {
            embedding,
            embedding_norm,
            hidden_norm,
            normalized,
            packed,
        }
    }
}

impl Launch for MtpStem {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        let hidden = quantum.config.hidden_size;
        let rows = quantum.rows_u32()?;
        let width = launch_u32(hidden, "the hidden size")?;
        let row_bytes = hidden as u64 * 2;
        let seed = match quantum.draft_seed {
            Some(DraftSeed::Address(address)) => address,
            Some(DraftSeed::SeedRow(first)) => {
                let draft = quantum
                    .seq
                    .device()
                    .scratch
                    .draft()
                    .ok_or_else(|| fatal("the sequence has no draft seed rows"))?;
                if u64::from(first) + quantum.rows as u64 > draft.seed_capacity {
                    return Err(fatal(format!(
                        "seed rows {first}.. of {} exceed the {} the sequence holds",
                        quantum.rows, draft.seed_capacity
                    )));
                }
                draft.seed_rows.ptr().offset(u64::from(first) * row_bytes)
            }
            None => return Err(fatal("an MTP stem needs the rows that seed it")),
        };
        let normalized = quantum.slot(self.normalized);
        let packed = quantum.slot(self.packed);
        let [embedding_half, hidden_half] = half_offsets(row_bytes);
        let epsilon = quantum.epsilon();
        for (input, weight, half) in [
            (quantum.slot(self.embedding), self.embedding_norm, embedding_half),
            (seed, self.hidden_norm, hidden_half),
        ] {
            ops::rms_norm_unit_offset_bf16(submission, input, weight, normalized, rows, width, epsilon)
                .map_err(fatal)?;
            ops::copy_device_to_device_2d(
                submission,
                packed.offset(half),
                2 * row_bytes,
                normalized,
                row_bytes,
                row_bytes,
                u64::from(rows),
            )
            .map_err(fatal)?;
        }
        Ok(())
    }
}

eager!(MtpStem);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_packed_row_is_the_embedding_half_then_the_hidden_half() {
        assert_eq!(half_offsets(10_240), [0, 10_240]);
    }
}
