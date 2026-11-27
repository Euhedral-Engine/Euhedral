//! The layer's nodes around its block (`Stages.Attention` with `whole`, `Stages.EndInject` and the mix of `Stages.Head`).
//!
//! A [`BlockStage`] is Java's BLOCK: the mix of the streams for the block, the block, the injection of its output into
//! the streams, and the mix that feeds the MoE block, as one stage. The block is a [`BlockBody`]: GDN is
//! [`super::gdn::GdnBody`]; sparse attention supplies its own. [`EndInject`] injects what the MoE block wrote, and
//! [`FinalMix`] mixes the last row for the output head.

use std::fmt::Debug;
use std::sync::Arc;

use euhedral_graph::StageError;
use euhedral_inference_cuda::{DevicePtr, KernelOps};

use super::FlashError;
use super::buffers::{FlashRegions, ScratchSizes, StreamShape};
use super::hyper::{HyperScratch, HyperShape, HyperWeights};
use super::linear::Expansion;
use super::sequence::{Op, OpList, queue_all};
use crate::stages::{Launch, Quantum, fatal};

/// What a block is called with.
#[derive(Clone, Copy, Debug)]
pub struct BlockCall {
    /// The mixed rows the block reads and the rows it writes: `[rows, hidden]` each.
    pub input: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    /// Whether the lane runs the exact-numerics kernels.
    pub exact: bool,
    /// The scratch of the block in flight ([`FlashRegions::layer_scratch`]).
    pub scratch: DevicePtr,
    /// The expansion scratch of the NVFP4 linears' native route.
    pub expansion: Expansion,
}

/// The token mixer of a layer between its two hyper-connection sites.
pub trait BlockBody: Send + Sync + Debug {
    /// The operations of the block for `call`, in the order they are queued.
    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError>;

    /// The bytes of layer scratch the block takes for `rows` rows.
    fn scratch_bytes(&self, rows: u32) -> u64;

    /// The bytes of expansion scratch its NVFP4 linears take for `rows` rows.
    fn expansion_bytes(&self, rows: u32) -> u64;
}

/// The stream shape of `hyper`.
fn stream_shape(hyper: &HyperShape) -> Result<StreamShape, FlashError> {
    StreamShape::new(hyper.hidden(), hyper.streams())
}

/// Refuses `regions` that do not fit `hyper`'s streams and mix scratch.
fn require_hyper(stage: &'static str, hyper: &HyperShape, regions: &FlashRegions, rows: u32) -> Result<(), FlashError> {
    regions.require(
        stage,
        stream_shape(hyper)?,
        ScratchSizes {
            hyper: hyper.scratch_bytes(rows),
            ..ScratchSizes::default()
        },
    )
}

/// The rows of `quantum`, within the regions' capacity.
fn rows_of(quantum: &Quantum, regions: &FlashRegions) -> Result<u32, StageError> {
    let rows = quantum.rows_u32()?;
    if rows > regions.capacity_rows() {
        return Err(fatal(format!(
            "a quantum of {rows} rows exceeds the {} rows the flash regions hold",
            regions.capacity_rows()
        )));
    }
    Ok(rows)
}

/// The mix for the block, the block, its injection, and the mix for the MoE block (Java's BLOCK stage).
#[derive(Debug)]
pub struct BlockStage {
    hyper: HyperShape,
    attention_site: HyperWeights,
    moe_site: HyperWeights,
    body: Arc<dyn BlockBody>,
    regions: FlashRegions,
}

impl BlockStage {
    /// The stage of a layer whose sites are `attention_site` (before the block) and `moe_site` (after it).
    pub fn new(
        hyper: HyperShape,
        attention_site: HyperWeights,
        moe_site: HyperWeights,
        body: Arc<dyn BlockBody>,
        regions: FlashRegions,
    ) -> Result<Self, FlashError> {
        if attention_site.inject.is_none() || moe_site.inject.is_none() {
            return Err(FlashError::refused(
                "block stage",
                "both sites of a layer carry an injection projection".to_owned(),
            ));
        }
        let capacity = regions.capacity_rows();
        regions.require(
            "block stage",
            stream_shape(&hyper)?,
            ScratchSizes {
                hyper: hyper.scratch_bytes(capacity),
                layer: body.scratch_bytes(capacity),
                expansion: body.expansion_bytes(capacity),
            },
        )?;
        Ok(Self {
            hyper,
            attention_site,
            moe_site,
            body,
            regions,
        })
    }

    /// The operations for `rows` rows, in order: the mix for the block, the block, the injection, the mix for MoE.
    pub fn ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
        let regions = &self.regions;
        let scratch = self.hyper.scratch(regions.hyper_scratch, rows);
        let mut list = self.hyper.mix(
            &self.attention_site,
            regions.state,
            &scratch,
            regions.mixed,
            rows,
            exact,
        )?;
        list.extend(self.body.ops(&BlockCall {
            input: regions.mixed,
            output: regions.block_output,
            rows,
            exact,
            scratch: regions.layer_scratch,
            expansion: regions.expansion_scratch(),
        })?);
        list.push(
            self.hyper
                .inject(regions.state, regions.block_output, &scratch, regions.state, rows)?,
        );
        list.extend(
            self.hyper
                .mix(&self.moe_site, regions.state, &scratch, regions.mixed, rows, exact)?,
        );
        Ok(list)
    }
}

impl Launch for BlockStage {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let ops = self.ops(rows, kernels.exact_numerics()).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

/// The injection of the MoE block's output into the streams (`Stages.EndInject`), by the raw weights of the MoE site's
/// mix.
#[derive(Debug)]
pub struct EndInject {
    hyper: HyperShape,
    regions: FlashRegions,
}

impl EndInject {
    /// Refuses regions of another stream shape or with too little mix scratch.
    pub fn new(hyper: HyperShape, regions: FlashRegions) -> Result<Self, FlashError> {
        require_hyper("end inject", &hyper, &regions, regions.capacity_rows())?;
        Ok(Self { hyper, regions })
    }

    /// The injection of `rows` rows.
    pub fn op(&self, rows: u32) -> Result<Op<'static>, FlashError> {
        let regions = &self.regions;
        let scratch = self.hyper.scratch(regions.hyper_scratch, rows);
        self.hyper
            .inject(regions.state, regions.block_output, &scratch, regions.state, rows)
    }
}

impl Launch for EndInject {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let op = self.op(rows).map_err(fatal)?;
        op.queue(kernels).map_err(fatal)
    }
}

/// The final mixer over the last row (`Stages.Head`'s mix): the streams of the quantum's last row mixed into
/// [`FlashRegions::final_mixed`], with no injection.
#[derive(Debug)]
pub struct FinalMix {
    hyper: HyperShape,
    site: HyperWeights,
    regions: FlashRegions,
}

impl FinalMix {
    /// The final mixer, whose site has no injection projection.
    pub fn new(hyper: HyperShape, site: HyperWeights, regions: FlashRegions) -> Result<Self, FlashError> {
        if site.inject.is_some() {
            return Err(FlashError::refused(
                "final mix",
                "the final mixer has no injection projection".to_owned(),
            ));
        }
        require_hyper("final mix", &hyper, &regions, 1)?;
        Ok(Self { hyper, site, regions })
    }

    /// The operations over the last of `rows` rows.
    pub fn ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
        if rows == 0 {
            return Err(FlashError::refused(
                "final mix",
                "a final mix takes at least one row, got 0".to_owned(),
            ));
        }
        let width = u64::from(self.hyper.state_width());
        let last = self.regions.state.offset(u64::from(rows - 1) * width * 2);
        let scratch: HyperScratch = self.hyper.scratch(self.regions.hyper_scratch, 1);
        self.hyper
            .mix(&self.site, last, &scratch, self.regions.final_mixed, 1, exact)
    }
}

impl Launch for FinalMix {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let ops = self.ops(rows, kernels.exact_numerics()).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

eager!(BlockStage, EndInject, FinalMix);

#[cfg(test)]
mod tests;
