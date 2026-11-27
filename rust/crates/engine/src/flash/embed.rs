//! The embedding of a quantum's tokens over the residual streams (`Stages.Embed`): the table lookup into one row of
//! `hidden` values per token, then the repeat of each row into every stream.

use euhedral_graph::StageError;
use euhedral_inference_cuda::{DevicePtr, KernelOps};

use super::buffers::{FlashRegions, StreamShape};
use super::sequence::{Op, OpList, queue_all};
use super::{FlashError, Weight, ops};
use crate::stages::{Launch, Quantum, fatal};
use crate::workspace::BufferSlot;

/// The embedding lookup and the stream repeat of one quantum.
#[derive(Clone, Debug)]
pub struct EmbedStreams {
    table: Weight,
    vocabulary: u32,
    shape: StreamShape,
    regions: FlashRegions,
}

impl EmbedStreams {
    /// The lookup in `table` (`[vocabulary, hidden]` BF16) into the streams of `regions`.
    pub fn new(table: Weight, vocabulary: u32, shape: StreamShape, regions: FlashRegions) -> Result<Self, FlashError> {
        let bytes = u64::from(vocabulary) * u64::from(shape.hidden) * 2;
        if vocabulary == 0 || table.bytes != bytes {
            return Err(FlashError::refused(
                "embed",
                format!(
                    "the table holds {} bytes, a {vocabulary} x {} BF16 table holds {bytes}",
                    table.bytes, shape.hidden
                ),
            ));
        }
        Ok(Self {
            table,
            vocabulary,
            shape,
            regions,
        })
    }

    /// The operations for `rows` rows reading their ids from `ids`.
    pub fn ops(&self, ids: DevicePtr, rows: u32) -> Result<OpList<'static>, FlashError> {
        let mut list = OpList::new();
        list.push(Op::Launch(ops::embedding(
            self.table.address,
            ids,
            self.regions.embedded,
            rows,
            self.shape.hidden,
            self.vocabulary,
        )?));
        list.push(Op::Launch(ops::repeat_streams(
            self.regions.embedded,
            self.regions.state,
            rows,
            self.shape.streams,
            self.shape.hidden,
        )?));
        Ok(list)
    }
}

impl Launch for EmbedStreams {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    /// The upload of the quantum's input record, which holds the ids.
    fn waits(&self, quantum: &Quantum) -> Vec<euhedral_graph::Marker> {
        quantum.prepared.into_iter().collect()
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = quantum.rows_u32()?;
        if rows > self.regions.capacity_rows() {
            return Err(fatal(format!(
                "a quantum of {rows} rows exceeds the {} rows the flash regions hold",
                self.regions.capacity_rows()
            )));
        }
        let ops = self.ops(quantum.slot(BufferSlot::TokenIds), rows).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

eager!(EmbedStreams);

#[cfg(test)]
mod tests;
