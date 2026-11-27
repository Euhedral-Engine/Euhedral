//! The hyper-connection of a Flash-Next layer (`HyperConnection.java`): the residual state is `streams` parallel
//! streams of `hidden` values per row. A site mixes them down to one stream that a block reads
//! ([`HyperConnection::mix`]) and injects the block's output back into every stream ([`HyperConnection::inject`]).
//!
//! The mix is a grouped RMS norm of the streams, a low-rank projection down, a SiLU scaled by the stream count, a
//! projection up, and the `hc_mix` kernel; a site with an injection projection (every one but the final mixer's) also
//! projects the normed streams to the raw injection weights that its inject reads.

use euhedral_inference_cuda::DevicePtr;

use super::buffers::{Carver, bf16_bytes};
use super::ops;
use super::sequence::{Op, OpList};
use super::{FlashError, Weight};

/// The size of a hyper-connection: its streams, their width and the rank of the low-rank mix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperShape {
    streams: u32,
    hidden: u32,
    lowrank: u32,
    epsilon: f32,
}

impl HyperShape {
    /// `streams` streams (more than one) of `hidden` values, mixed through rank `lowrank`.
    pub fn new(streams: u32, hidden: u32, lowrank: u32, epsilon: f32) -> Result<Self, FlashError> {
        let state_width = u64::from(streams) * u64::from(hidden);
        if streams <= 1
            || hidden == 0
            || lowrank == 0
            || !lowrank.is_multiple_of(8)
            || !state_width.is_multiple_of(8)
            || state_width > u64::from(u32::MAX)
            || !epsilon.is_finite()
            || epsilon < 0.0
        {
            return Err(FlashError::refused(
                "hyper-connection",
                format!(
                    "{streams} streams of {hidden} values at rank {lowrank} with epsilon {epsilon}: \
                     more than one stream, a rank and a stream width that are positive multiples of 8 (the BF16 linears' K) and a finite epsilon"
                ),
            ));
        }
        Ok(Self {
            streams,
            hidden,
            lowrank,
            epsilon,
        })
    }

    pub fn streams(&self) -> u32 {
        self.streams
    }

    pub fn hidden(&self) -> u32 {
        self.hidden
    }

    /// The values of one row of all streams.
    pub fn state_width(&self) -> u32 {
        self.streams * self.hidden
    }

    /// The bytes of the scratch of `rows` rows.
    pub fn scratch_bytes(&self, rows: u32) -> u64 {
        let mut carver = Carver::new(DevicePtr::NULL);
        self.carve_with(&mut carver, rows);
        carver.bytes()
    }

    /// The scratch buffers of `rows` rows laid out from `base`.
    pub fn scratch(&self, base: DevicePtr, rows: u32) -> HyperScratch {
        self.carve_with(&mut Carver::new(base), rows)
    }

    fn carve_with(&self, carver: &mut Carver, rows: u32) -> HyperScratch {
        HyperScratch {
            normed: carver.take(bf16_bytes(rows, self.state_width())),
            down: carver.take(bf16_bytes(rows, self.lowrank)),
            activated: carver.take(bf16_bytes(rows, self.lowrank)),
            up: carver.take(bf16_bytes(rows, self.state_width())),
            raw_injection: carver.take(bf16_bytes(rows, self.streams)),
        }
    }

    /// The operations of the mix of `state` into `mixed` at one site, in Java's order. `exact` is the numerics mode of
    /// the lane's kernels.
    pub fn mix(
        &self,
        weights: &HyperWeights,
        state: DevicePtr,
        scratch: &HyperScratch,
        mixed: DevicePtr,
        rows: u32,
        exact: bool,
    ) -> Result<OpList<'static>, FlashError> {
        let width = self.state_width();
        let activations = rows.checked_mul(self.lowrank).ok_or_else(|| {
            FlashError::refused("hyper-connection mix", format!("{rows} rows at rank {}", self.lowrank))
        })?;
        let mut list = OpList::new();
        list.push(Op::Launch(ops::grouped_rms_norm(
            state,
            weights.norm.address,
            scratch.normed,
            rows,
            self.streams,
            self.hidden,
            self.epsilon,
        )?));
        list.push(Op::Launch(ops::linear_bf16(
            scratch.normed,
            weights.down.address,
            scratch.down,
            rows,
            width,
            self.lowrank,
            exact,
        )?));
        list.push(Op::Launch(ops::scaled_silu(
            scratch.down,
            scratch.activated,
            activations,
            self.streams as f32,
        )?));
        list.push(Op::Launch(ops::linear_bf16(
            scratch.activated,
            weights.up.address,
            scratch.up,
            rows,
            self.lowrank,
            width,
            exact,
        )?));
        list.push(Op::Launch(ops::hc_mix(
            scratch.normed,
            scratch.up,
            mixed,
            rows,
            self.streams,
            self.hidden,
        )?));
        if let Some(inject) = weights.inject {
            list.push(Op::Launch(ops::linear_bf16(
                scratch.normed,
                inject.address,
                scratch.raw_injection,
                rows,
                width,
                self.streams,
                exact,
            )?));
        }
        Ok(list)
    }

    /// The injection of `block` (one stream's worth of rows) into the streams of `state`, written to `output`.
    pub fn inject(
        &self,
        state: DevicePtr,
        block: DevicePtr,
        scratch: &HyperScratch,
        output: DevicePtr,
        rows: u32,
    ) -> Result<Op<'static>, FlashError> {
        Ok(Op::Launch(ops::hc_inject(
            state,
            block,
            scratch.raw_injection,
            output,
            rows,
            self.streams,
            self.hidden,
        )?))
    }
}

/// The weights of one site: the BF16 norm over the streams, the low-rank projections, and the injection projection
/// (none at the final mixer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HyperWeights {
    pub norm: Weight,
    pub down: Weight,
    pub up: Weight,
    pub inject: Option<Weight>,
}

/// The scratch buffers of the mixes of `rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HyperScratch {
    pub normed: DevicePtr,
    pub down: DevicePtr,
    pub activated: DevicePtr,
    pub up: DevicePtr,
    /// The raw injection weights of the site mixed last: `[rows, streams]`.
    pub raw_injection: DevicePtr,
}

#[cfg(test)]
mod tests;
