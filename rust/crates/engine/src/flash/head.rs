//! The head of a Flash-Next quantum (`Stages.Head`): the logits of the quantum's last row and what the host takes
//! from them, either the selection key of the greedy argmax or the whole BF16 row.
//!
//! The final mix of the last row is the head's norm: [`super::block::FinalMix`] writes it into
//! [`FlashRegions::final_mixed`] (Java has no separate norm either), and the head multiplies it by the output
//! weights. The head runs only for a quantum that scores a row ([`Quantum::logits_rows`] above 0), as Java's head runs
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

    /// What the host reads, from what the quantum offers: the row when it has a row readback (whose rows are
    /// `row_bytes` bytes), else the key of its selection at `keys`.
    pub fn select(&self, row_bytes: Option<usize>, keys: Option<DevicePtr>) -> Result<Select, FlashError> {
        if let Some(bytes) = row_bytes {
            if bytes as u64 != self.row_bytes() {
                return Err(FlashError::refused(
                    OP,
                    format!(
                        "the row readback holds rows of {bytes} bytes, the logits row is {}",
                        self.row_bytes()
                    ),
                ));
            }
            return Ok(Select::Row);
        }
        keys.map(Select::Key).ok_or_else(|| {
            FlashError::refused(
                OP,
                "a quantum that scores a row has neither a selection nor a row readback".to_owned(),
            )
        })
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
        // The head writes one row: the last of one sequence's rows.
        let [segment] = quantum.segments() else {
            return Err(fatal(format!(
                "the head serves one sequence, the quantum has {} segments",
                quantum.segments().len()
            )));
        };
        if quantum.logits_rows != 1 {
            return Err(fatal(format!(
                "the head writes one logits row, the quantum scores {}",
                quantum.logits_rows
            )));
        }
        let select = self
            .select(
                segment
                    .row_readback
                    .as_ref()
                    .map(|rows| rows.row(0).map_or(0, <[u8]>::len)),
                segment.selection.as_ref().map(|selection| selection.device_keys()),
            )
            .map_err(fatal)?;
        // The pinned buffer belongs to the quantum, which retires only after the marker that follows this launch.
        let destination = match (&segment.row_readback, &segment.selection) {
            (Some(rows), _) => rows.pinned(),
            (None, Some(selection)) => selection.readback(),
            (None, None) => return Err(fatal("a quantum that scores a row has nowhere to copy it")),
        };
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
