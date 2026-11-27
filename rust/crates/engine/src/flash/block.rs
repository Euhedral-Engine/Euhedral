//! The layer's nodes around its block (`Stages.Attention` with `whole`, `Stages.EndInject` and the mix of `Stages.Head`).
//!
//! A [`BlockStage`] is Java's BLOCK: the mix of the streams for the block, the block, the injection of its output into
//! the streams, and the mix that feeds the MoE block, as one stage. The block is a [`BlockBody`]: GDN is
//! [`super::gdn::GdnBody`]; sparse attention supplies its own. [`EndInject`] injects what the MoE block wrote, and
//! [`FinalMix`] mixes the last row for the output head.

use std::sync::Arc;

use euhedral_graph::StageError;
use euhedral_inference_cuda::{DevicePtr, KernelOps};

use super::FlashError;
use super::buffers::FlashRegions;
use super::hyper::{HyperScratch, HyperShape, HyperWeights};
use super::sequence::{OpList, queue_all};
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
}

/// The token mixer of a layer between its two hyper-connection sites.
pub trait BlockBody: Send + Sync + std::fmt::Debug {
    /// The operations of the block for `call`, in the order they are queued.
    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError>;
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
    pub fn new(hyper: HyperShape, regions: FlashRegions) -> Self {
        Self { hyper, regions }
    }
}

impl Launch for EndInject {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.regions)?;
        let regions = &self.regions;
        let scratch = self.hyper.scratch(regions.hyper_scratch, rows);
        let op = self
            .hyper
            .inject(regions.state, regions.block_output, &scratch, regions.state, rows)
            .map_err(fatal)?;
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
        Ok(Self { hyper, site, regions })
    }

    /// The operations over the last of `rows` rows.
    pub fn ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
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
