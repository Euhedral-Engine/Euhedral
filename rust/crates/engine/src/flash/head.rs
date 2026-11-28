//! The head of a Flash-Next quantum (`Stages.Head`): the logits of the quantum's last row and what the host takes
//! from them, either the selection key of the greedy argmax or the whole BF16 row.
//!
//! The final mix of the last row is the head's norm: [`super::block::FinalMix`] writes it into
//! [`FlashRegions::final_mixed`], and the head multiplies it by the output
//! weights. The head runs only for a quantum that scores a row ([`Quantum::logits_rows`] above 0), so it runs
//! only for the last chunk of a prompt.

use euhedral_graph::StageError;
use euhedral_inference_cuda::{DevicePtr, KernelOps};

use super::buffers::FlashRegions;
use super::sequence::{Op, OpList, queue_all};
use super::{FlashError, Weight, ops};
use crate::stages::sampling::KEY_BYTES;
use crate::stages::{Launch, Quantum, fatal};

const OP: &str = "head";
const BF16: u64 = 2;

/// What the host reads of the logits row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Select {
    /// The selection key of the greedy argmax, written at the address.
    Key(DevicePtr),
    /// The whole row.
    Row,
}

/// The logits of the last row and the copy of what the host reads.
#[derive(Debug)]
pub struct HeadStage {
    head: Weight,
    logits: DevicePtr,
    final_mixed: DevicePtr,
    hidden: u32,
    vocabulary: u32,
}

impl HeadStage {
    /// The head over the output weights `head` (`vocabulary` rows of `hidden` BF16 entries), writing one BF16 logits
    /// row at `logits`, which the workspace owner carves (the residency plan counts it) and which holds at least
    /// `vocabulary` entries.
    pub fn new(
        head: Weight,
        logits: DevicePtr,
        regions: &FlashRegions,
        hidden: u32,
        vocabulary: u32,
    ) -> Result<Self, FlashError> {
        if hidden == 0 || vocabulary == 0 || !hidden.is_multiple_of(8) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a head of {vocabulary} entries over {hidden} hidden values: both are positive and the width is a multiple of 8"
                ),
            ));
        }
        if hidden != regions.shape().hidden {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a head over {hidden} hidden values, the regions hold {}",
                    regions.shape().hidden
                ),
            ));
        }
        let expected = u64::from(vocabulary) * u64::from(hidden) * BF16;
        if head.bytes != expected {
            return Err(FlashError::refused(
                OP,
                format!(
                    "the output weights hold {} bytes, {vocabulary} x {hidden} BF16 need {expected}",
                    head.bytes
                ),
            ));
        }
        if !logits.get().is_multiple_of(16) {
            return Err(FlashError::refused(
                OP,
                format!("the logits row at {:#x} is not 16-byte aligned", logits.get()),
            ));
        }
        Ok(Self {
            head,
            logits,
            final_mixed: regions.final_mixed,
            hidden,
            vocabulary,
        })
    }

    /// The bytes of the logits row.
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.vocabulary) * BF16
    }

    /// The operations that write the logits row, and the key when `select` names one, in the order they are queued.
    pub fn ops<'a>(&self, exact: bool, select: Select) -> Result<OpList<'a>, FlashError> {
        let mut list = OpList::new();
        list.push(Op::Launch(ops::linear_bf16(
            self.final_mixed,
            self.head.address,
            self.logits,
            1,
            self.hidden,
            self.vocabulary,
            exact,
        )?));
        if let Select::Key(keys) = select {
            list.push(Op::Argmax {
                logits: self.logits,
                entries: self.vocabulary,
                keys,
            });
        }
        Ok(list)
    }

    /// The device bytes the host copy of `select` reads: where they are and how many.
    pub fn readback(&self, select: Select) -> (DevicePtr, usize) {
        match select {
            Select::Key(keys) => (keys, KEY_BYTES),
            Select::Row => (self.logits, self.row_bytes() as usize),
        }
    }

    /// Where the head's row goes and how the host reads it, from what the quantum scores and offers: `logits_rows`
    /// rows (the head writes one: the last of one sequence's rows), a row readback (`row_bytes` of its rows and where
    /// they are copied to) and a selection (its keys and where they are copied to). The row readback wins when both
    /// are offered. Refuses a count of rows other than one, a readback of another row size and an offer of neither.
    pub fn route<D>(
        &self,
        logits_rows: usize,
        row: Option<(usize, D)>,
        keys: Option<(DevicePtr, D)>,
    ) -> Result<(Select, D), FlashError> {
        if logits_rows != 1 {
            return Err(FlashError::refused(
                OP,
                format!("the head writes one logits row, the quantum scores {logits_rows}"),
            ));
        }
        match (row, keys) {
            (Some((bytes, destination)), _) => {
                if bytes as u64 != self.row_bytes() {
                    return Err(FlashError::refused(
                        OP,
                        format!(
                            "the row readback holds rows of {bytes} bytes, the logits row is {}",
                            self.row_bytes()
                        ),
                    ));
                }
                Ok((Select::Row, destination))
            }
            (None, Some((keys, destination))) => Ok((Select::Key(keys), destination)),
            (None, None) => Err(FlashError::refused(
                OP,
                "a quantum that scores a row has neither a selection nor a row readback".to_owned(),
            )),
        }
    }
}

/// The one segment the head serves: a quantum of any other count is refused, naming it.
fn single_segment<T>(segments: &[T]) -> Result<&T, FlashError> {
    match segments {
        [segment] => Ok(segment),
        other => Err(FlashError::refused(
            OP,
            format!("the head serves one sequence, the quantum has {} segments", other.len()),
        )),
    }
}

impl Launch for HeadStage {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.logits_rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        let segment = single_segment(quantum.segments()).map_err(fatal)?;
        let row = segment
            .row_readback
            .as_ref()
            // The pinned buffer belongs to the quantum, which retires only after the marker that follows this launch.
            .map(|rows| (rows.row(0).map_or(0, <[u8]>::len), rows.pinned()));
        let keys = segment
            .selection
            .as_ref()
            .map(|selection| (selection.device_keys(), selection.readback()));
        let (select, destination) = self.route(quantum.logits_rows, row, keys).map_err(fatal)?;
        let mut list = self.ops(kernels.exact_numerics(), select).map_err(fatal)?;
        let (source, bytes) = self.readback(select);
        list.push(Op::Readback {
            destination,
            offset: 0,
            source,
            bytes,
        });
        queue_all(&list, kernels).map_err(fatal)
    }
}

eager!(HeadStage);

#[cfg(test)]
mod tests;
