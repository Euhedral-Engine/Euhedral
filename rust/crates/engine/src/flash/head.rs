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
        if hidden == 0 || vocabulary == 0 {
            return Err(FlashError::refused(
                OP,
                format!("a head of {vocabulary} entries over {hidden} hidden values"),
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

    /// What the quantum's host reads: the row when its segment has a row readback, else the key of its selection.
    fn select_of(&self, quantum: &Quantum) -> Result<Select, FlashError> {
        let segment = quantum.segment();
        if let Some(rows) = &segment.row_readback {
            let bytes = rows.row(0).map_or(0, <[u8]>::len) as u64;
            if bytes != self.row_bytes() {
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
        match &segment.selection {
            Some(selection) => Ok(Select::Key(selection.device_keys())),
            None => Err(FlashError::refused(
                OP,
                "a quantum that scores a row has neither a selection nor a row readback".to_owned(),
            )),
        }
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
        let select = self.select_of(quantum).map_err(fatal)?;
        let mut list = self.ops(kernels.exact_numerics(), select).map_err(fatal)?;
        let (source, bytes) = self.readback(select);
        let segment = quantum.segment();
        // The pinned buffer belongs to the quantum, which retires only after the marker that follows this launch.
        let destination = match (&segment.row_readback, &segment.selection) {
            (Some(rows), _) => rows.pinned(),
            (None, Some(selection)) => selection.readback(),
            (None, None) => return Err(fatal("a quantum that scores a row has nowhere to copy it")),
        };
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
