//! One MoE layer's configuration: its shape, weights and device scratch, and the operations of each of its steps.
//!
//! The steps are pure: given the rows and the plan of a block they list the launches and copies a stage queues, so a
//! test reads them as lines and the stages ([`super::stages`], [`super::expert`]) only choose when to queue them.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_inference_cuda::DevicePtr;
use smallvec::smallvec;

use super::ops::{self, ExpertGeometry, ExpertItems, ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K};
use super::routing::DescriptorLayout;
use super::source::ExpertSource;
use crate::flash::buffers::{Carver, FlashRegions, ScratchSizes};
use crate::flash::linear::{Nvfp4Device, Nvfp4Weight};
use crate::flash::ops::linear_bf16;
use crate::flash::sequence::{Op, OpList};
use crate::flash::{FlashError, Weight};
use crate::formats::shapes::Launch;

/// The rows from which the shared expert's down projection reads the activations padded to a multiple of 256 values
/// (`MoeLayer.PREFILL_ROWS`): fewer rows take the unpadded weight.
pub const PREFILL_ROWS: u32 = 8;

const BF16: u64 = 2;

/// The extents of a layer's MoE block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeShape {
    layout: DescriptorLayout,
    geometry: ExpertGeometry,
    shared_inter: u32,
}

impl MoeShape {
    /// Refuses a shared expert whose width is not a positive multiple of 16, and a layout the router cannot serve: more
    /// than [`ROUTER_MAX_EXPERTS`] experts, more than [`ROUTER_MAX_TOP_K`] picks, or more picks than experts.
    pub fn new(layout: DescriptorLayout, geometry: ExpertGeometry, shared_inter: u32) -> Result<Self, FlashError> {
        let (experts, top_k) = (layout.experts(), layout.top_k());
        if experts > ROUTER_MAX_EXPERTS || top_k > ROUTER_MAX_TOP_K || top_k > experts {
            return Err(FlashError::refused(
                "moe shape",
                format!(
                    "{experts} experts with {top_k} picks per row: the router takes at most {ROUTER_MAX_EXPERTS} experts and {ROUTER_MAX_TOP_K} picks, never more than the experts"
                ),
            ));
        }
        if shared_inter == 0 || !shared_inter.is_multiple_of(16) {
            return Err(FlashError::refused(
                "moe shape",
                format!("the shared expert's width must be a positive multiple of 16, got {shared_inter}"),
            ));
        }
        Ok(Self {
            layout,
            geometry,
            shared_inter,
        })
    }

    pub fn layout(&self) -> &DescriptorLayout {
        &self.layout
    }

    pub fn geometry(&self) -> &ExpertGeometry {
        &self.geometry
    }

    pub fn hidden(&self) -> u32 {
        self.geometry.hidden()
    }

    pub fn shared_inter(&self) -> u32 {
        self.shared_inter
    }

    /// The shared expert's width rounded up to what the padded down projection reads.
    pub fn padded_width(&self) -> u32 {
        self.shared_inter.next_multiple_of(256)
    }
}

/// The weights of a layer's MoE block that stay on the device: the router, the shared expert and its gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeWeights {
    /// The router `[experts][hidden]`, BF16.
    pub router: Weight,
    pub shared_gate: Nvfp4Weight,
    pub shared_up: Nvfp4Weight,
    pub shared_down: Nvfp4Weight,
    /// The shared down projection over inputs padded to [`MoeShape::padded_width`], for blocks of many rows.
    pub shared_down_padded: Option<Nvfp4Weight>,
    /// The shared expert's gate `[1][hidden]`, BF16.
    pub shared_expert_gate: Weight,
}

/// Where a layer's device scratch lies: carved, in this order, from the block scratch of the regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeScratch {
    logits: DevicePtr,
    ids: DevicePtr,
    route_weights: DevicePtr,
    shared_gate: DevicePtr,
    shared_up: DevicePtr,
    shared_act: DevicePtr,
    shared: DevicePtr,
    routed: DevicePtr,
    gate_raw: DevicePtr,
    descriptor: DevicePtr,
    experts: DevicePtr,
}

impl MoeScratch {
    /// The bytes the scratch of `shape` occupies.
    pub fn bytes(shape: &MoeShape) -> u64 {
        Self::carve(DevicePtr::NULL, shape).1
    }

    fn carve(base: DevicePtr, shape: &MoeShape) -> (Self, u64) {
        let layout = &shape.layout;
        let (rows, pairs) = (u64::from(layout.max_rows()), u64::from(layout.max_pairs()));
        let (hidden, inter, padded) = (
            u64::from(shape.hidden()),
            u64::from(shape.shared_inter),
            u64::from(shape.padded_width()),
        );
        let mut carver = Carver::new(base);
        let scratch = Self {
            logits: carver.take(rows * u64::from(layout.experts()) * BF16),
            ids: carver.take(pairs * 4),
            route_weights: carver.take(pairs * BF16),
            shared_gate: carver.take(rows * inter * BF16),
            shared_up: carver.take(rows * inter * BF16),
            shared_act: carver.take(rows * padded * BF16),
            shared: carver.take(rows * hidden * BF16),
            routed: carver.take(rows * hidden * BF16),
            gate_raw: carver.take(rows * BF16),
            descriptor: carver.take(layout.bytes() as u64),
            experts: carver.take(shape.geometry.scratch_bytes(layout.max_pairs())),
        };
        (scratch, carver.bytes())
    }
}

/// The step a stage is of, for the words of its refusals.
const LAYER: &str = "moe layer";

/// The shared state of one layer's MoE stages: what they run, where, and from which cache they take experts.
pub struct MoeLayer {
    shape: MoeShape,
    weights: MoeWeights,
    scratch: MoeScratch,
    regions: FlashRegions,
    device: Nvfp4Device,
    bank: u32,
    index: u32,
    source: Arc<dyn ExpertSource>,
}

impl Debug for MoeLayer {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("MoeLayer")
            .field("layer", &self.index)
            .field("bank", &self.bank)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl MoeLayer {
    /// The MoE block of layer `index`, whose routed experts are bank `bank` of `source`. Refuses regions that hold
    /// fewer rows, other streams or less scratch than the block needs.
    pub fn new(
        index: u32,
        bank: u32,
        shape: MoeShape,
        weights: MoeWeights,
        regions: FlashRegions,
        device: Nvfp4Device,
        source: Arc<dyn ExpertSource>,
    ) -> Result<Self, FlashError> {
        let rows = shape.layout.max_rows();
        if regions.capacity_rows() < rows {
            return Err(FlashError::refused(
                LAYER,
                format!(
                    "layer {index} routes up to {rows} rows, the regions hold {}",
                    regions.capacity_rows()
                ),
            ));
        }
        if regions.shape().hidden != shape.hidden() {
            return Err(FlashError::refused(
                LAYER,
                format!(
                    "layer {index} is {} wide, the regions {}",
                    shape.hidden(),
                    regions.shape().hidden
                ),
            ));
        }
        let held = regions.sizes();
        let needed = Self::scratch_sizes(&shape, weights.shared_down_padded.is_some(), &device);
        regions.require(LAYER, regions.shape(), needed).map_err(|error| {
            FlashError::refused(
                LAYER,
                format!(
                    "layer {index}: {error} (the regions hold {} bytes of block scratch, {} of expansion)",
                    held.layer, held.expansion
                ),
            )
        })?;
        let (scratch, _) = MoeScratch::carve(regions.layer_scratch, &shape);
        Ok(Self {
            shape,
            weights,
            scratch,
            regions,
            device,
            bank,
            index,
            source,
        })
    }

    /// The block scratch and expansion scratch a layer of `shape` needs from the regions, for sizing them.
    pub fn scratch_sizes(shape: &MoeShape, padded_down: bool, device: &Nvfp4Device) -> ScratchSizes {
        ScratchSizes {
            hyper: 0,
            layer: MoeScratch::bytes(shape),
            expansion: Self::expansion_bytes(shape, padded_down, device),
        }
    }

    fn expansion_bytes(shape: &MoeShape, padded_down: bool, device: &Nvfp4Device) -> u64 {
        let (rows, hidden, inter) = (shape.layout.max_rows(), shape.hidden(), shape.shared_inter);
        let down = if padded_down { shape.padded_width() } else { inter };
        device
            .expansion_bytes(rows, hidden, inter)
            .max(device.expansion_bytes(rows, down, hidden))
    }

    pub fn shape(&self) -> &MoeShape {
        &self.shape
    }

    /// The layer's number, for the words of a failure.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The bank of the expert cache that holds the layer's routed experts.
    pub fn bank(&self) -> u32 {
        self.bank
    }

    pub fn source(&self) -> &Arc<dyn ExpertSource> {
        &self.source
    }

    /// The device address of the block's description.
    pub fn descriptor(&self) -> DevicePtr {
        self.scratch.descriptor
    }

    /// The device addresses the router's results are copied from.
    pub fn router_outputs(&self) -> (DevicePtr, DevicePtr) {
        (self.scratch.ids, self.scratch.route_weights)
    }

    /// The router's linear and the router itself, for `rows` rows of the block input.
    pub fn route_launches(&self, rows: u32, exact: bool) -> Result<[Launch; 2], FlashError> {
        let (layout, scratch) = (&self.shape.layout, &self.scratch);
        Ok([
            linear_bf16(
                self.regions.mixed,
                self.weights.router.address,
                scratch.logits,
                rows,
                self.shape.hidden(),
                layout.experts(),
                exact,
            )?,
            ops::router(
                scratch.logits,
                scratch.ids,
                scratch.route_weights,
                rows,
                layout.experts(),
                layout.top_k(),
            )?,
        ])
    }

    /// The shared expert and its gate for `rows` rows (`MoeLayer.submitShared`).
    pub fn shared_ops(&self, rows: u32, exact: bool) -> Result<OpList<'static>, FlashError> {
        let (shape, weights, scratch) = (&self.shape, &self.weights, &self.scratch);
        let (hidden, inter) = (shape.hidden(), shape.shared_inter);
        let (input, expansion) = (self.regions.mixed, self.regions.expansion_scratch());
        let mut list: OpList<'static> = smallvec![
            self.device.linear(
                input,
                weights.shared_gate,
                scratch.shared_gate,
                rows,
                hidden,
                inter,
                expansion
            )?,
            self.device.linear(
                input,
                weights.shared_up,
                scratch.shared_up,
                rows,
                hidden,
                inter,
                expansion
            )?,
        ];
        match weights.shared_down_padded.filter(|_| rows > PREFILL_ROWS) {
            Some(padded) => {
                let width = shape.padded_width();
                list.push(Op::Launch(ops::swiglu_padded(
                    scratch.shared_gate,
                    scratch.shared_up,
                    scratch.shared_act,
                    rows,
                    inter,
                    width,
                )?));
                list.push(self.device.linear(
                    scratch.shared_act,
                    padded,
                    scratch.shared,
                    rows,
                    width,
                    hidden,
                    expansion,
                )?);
            }
            None => {
                list.push(Op::Launch(ops::swiglu(
                    scratch.shared_gate,
                    scratch.shared_up,
                    scratch.shared_act,
                    rows * inter,
                )?));
                list.push(self.device.linear(
                    scratch.shared_act,
                    weights.shared_down,
                    scratch.shared,
                    rows,
                    inter,
                    hidden,
                    expansion,
                )?);
            }
        }
        list.push(Op::Launch(linear_bf16(
            input,
            weights.shared_expert_gate.address,
            scratch.gate_raw,
            rows,
            hidden,
            1,
            exact,
        )?));
        Ok(list)
    }

    /// The device address of the `index`-th slot of the description, where the expert's record address is written.
    pub fn slot_address(&self, index: u32) -> DevicePtr {
        self.scratch
            .descriptor
            .offset(self.shape.layout.slot_offset(index) as u64)
    }

    /// The expert launches of an expert that has `item_count` work items from `first_item`, in a block of `pairs`
    /// pairs: the gate and up projections with their activation, then the down projection.
    pub fn expert_launches(&self, first_item: u32, item_count: u32, pairs: u32) -> Result<[Launch; 2], FlashError> {
        let (layout, scratch) = (&self.shape.layout, &self.scratch);
        let description = |offset: usize| scratch.descriptor.offset(offset as u64);
        let items = ExpertItems {
            slots: description(layout.slots_offset()),
            items: description(layout.items_offset() + 16 * first_item as usize),
            item_count,
            pairs: description(layout.pairs_offset()),
        };
        let geometry = &self.shape.geometry;
        let act = scratch.experts;
        let weighted = act.offset(geometry.weighted_offset(pairs));
        Ok([
            ops::expert_gate_up(geometry, &items, self.regions.mixed, act)?,
            ops::expert_down(geometry, &items, act, weighted)?,
        ])
    }

    /// The ordered sum of the block's experts' outputs, combined with the shared expert's gated output into the block's
    /// output (`MoeLayer.submitFinish`). A block that kept no pair has no expert output: the sum is zero.
    pub fn finish_ops(&self, rows: u32, pairs: u32) -> Result<OpList<'static>, FlashError> {
        let (layout, scratch, geometry) = (&self.shape.layout, &self.scratch, &self.shape.geometry);
        let hidden = self.shape.hidden();
        let mut list: OpList<'static> = OpList::new();
        if pairs > 0 {
            let description = |offset: usize| scratch.descriptor.offset(offset as u64);
            list.push(Op::Launch(ops::expert_combine(
                geometry,
                scratch.experts.offset(geometry.weighted_offset(pairs)),
                description(layout.row_offsets_offset()),
                description(layout.row_pairs_offset()),
                scratch.routed,
                rows,
            )?));
        } else {
            list.push(Op::Zero {
                address: scratch.routed,
                bytes: u64::from(rows) * u64::from(hidden) * BF16,
            });
        }
        list.push(Op::Launch(ops::moe_finish(
            scratch.routed,
            scratch.shared,
            scratch.gate_raw,
            self.regions.block_output,
            rows,
            hidden,
        )?));
        Ok(list)
    }

    /// The rows a layer of this shape routes at most.
    pub fn max_rows(&self) -> u32 {
        self.shape.layout.max_rows()
    }
}

#[cfg(test)]
mod tests;
