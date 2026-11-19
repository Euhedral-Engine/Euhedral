//! The kernel operations a stage queues, as the trait [`KernelOps`], and [`LaneOps`] for the owner's work that also
//! allocates and frees in stream order. [`RusteeOps`](crate::launch::ops::RusteeOps) implements both over a RustEE
//! lane.
//!
//! Device addresses are [`DevicePtr`]s: the host never dereferences them, and a wrong device address is reported by
//! the driver (or faults the kernel), not by the host's memory model, so the operations on device memory are safe.
//! Host memory is different. The two asynchronous host copies read or write host memory after they return and are
//! `unsafe`; so is taking the device address of mapped host memory
//! ([`PinnedBuffer::device_address`](crate::PinnedBuffer::device_address)), which is how a kernel gets to host bytes. The host's side of that contract is stated there.
//!
//! Every operation mirrors the parameter list of its C entry point (`native/include/euhedral_cuda.h`), which documents
//! the shapes, alignments and routes.

// Each operation mirrors one C entry point's parameter list.
#![allow(clippy::too_many_arguments)]

use euhedral_graph::Marker;

use crate::error::CudaError;
use crate::memory::{DeviceBuffer, DevicePtr, HostWeightsArena, PinnedBuffer};

/// How a quantum launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmitMode {
    /// Programmatic dependent launch: a registered kernel starts while the previous kernel on the lane drains.
    pub pdl: bool,
    /// Row-exact execution (speculative verification): every operator whose kernel changes with the row count runs
    /// each row through its single-row path.
    pub row_exact: bool,
}

/// The address of `bytes` at `offset` of the `len` bytes at `base`, refused when they leave them.
pub(crate) fn range_start(
    base: *mut u8,
    len: usize,
    offset: usize,
    bytes: usize,
    what: &str,
) -> Result<*mut u8, CudaError> {
    match offset.checked_add(bytes) {
        Some(end) if end <= len => Ok(base.wrapping_add(offset)),
        _ => Err(CudaError::Argument(format!(
            "{bytes} bytes at offset {offset} lie outside a {len} byte {what}"
        ))),
    }
}

/// The operations the stages queue, as a trait, so a stage does not name the lane it runs on: the stages take a
/// `&mut dyn KernelOps`. [`RusteeOps`](crate::launch::ops::RusteeOps) implements it by planning each call and
/// enqueueing the launches on a RustEE lane.
pub trait KernelOps {
    fn copy_device_to_device(&mut self, destination: DevicePtr, source: DevicePtr, bytes: u64)
    -> Result<(), CudaError>;
    fn copy_device_to_device_2d(
        &mut self,
        destination: DevicePtr,
        destination_pitch: u64,
        source: DevicePtr,
        source_pitch: u64,
        row_bytes: u64,
        rows: u64,
    ) -> Result<(), CudaError>;
    fn zero_device_memory(&mut self, address: DevicePtr, bytes: u64) -> Result<(), CudaError>;
    fn embed_q3(
        &mut self,
        token_ids: DevicePtr,
        embedding: DevicePtr,
        hidden: DevicePtr,
        token_count: u32,
        vocabulary_size: u32,
        hidden_size: u32,
        embedding_bytes: u64,
    ) -> Result<(), CudaError>;
    fn embed_q3_p2e2(
        &mut self,
        token_ids: DevicePtr,
        embedding: DevicePtr,
        hidden: DevicePtr,
        token_count: u32,
        vocabulary_size: u32,
        hidden_size: u32,
        embedding_bytes: u64,
    ) -> Result<(), CudaError>;
    fn linear_launch(
        &mut self,
        kernel: i32,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        words: &[u64],
        sizes: &[u8],
    ) -> Result<(), CudaError>;
    fn qwen4_launch(
        &mut self,
        kernel: i32,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        words: &[u64],
        sizes: &[u8],
    ) -> Result<(), CudaError>;
    fn linear_nvfp4_bf16(
        &mut self,
        input: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        rows: u32,
        in_features: u32,
        out_features: u32,
        weights_bytes: u64,
    ) -> Result<(), CudaError>;
    fn linear_nvfp4_native_bf16(
        &mut self,
        input: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        scratch: DevicePtr,
        rows: u32,
        in_features: u32,
        out_features: u32,
        weights_bytes: u64,
        scratch_bytes: u64,
    ) -> Result<(), CudaError>;
    fn attention_qk_norm_rope_bf16(
        &mut self,
        query_key: DevicePtr,
        query_norm: DevicePtr,
        key_norm: DevicePtr,
        output: DevicePtr,
        rows: u32,
        query_heads: u32,
        key_value_heads: u32,
        head_dim: u32,
        rotary_dim: u32,
        start_position: u64,
        position: DevicePtr,
        epsilon: f32,
        rope_theta: f64,
    ) -> Result<(), CudaError>;
    fn attention_kv_append_nvfp4(
        &mut self,
        query_key: DevicePtr,
        gate: DevicePtr,
        keys: DevicePtr,
        values: DevicePtr,
        rows: u32,
        query_width: u32,
        key_width: u32,
        start: u64,
        position: DevicePtr,
    ) -> Result<(), CudaError>;
    fn attention_kv_append_bf16(
        &mut self,
        query_key: DevicePtr,
        gate: DevicePtr,
        keys: DevicePtr,
        values: DevicePtr,
        rows: u32,
        query_width: u32,
        key_width: u32,
        start: u64,
        position: DevicePtr,
    ) -> Result<(), CudaError>;
    fn attention_causal_nvfp4(
        &mut self,
        query_key: DevicePtr,
        gate: DevicePtr,
        keys: DevicePtr,
        values: DevicePtr,
        output: DevicePtr,
        rows: u32,
        query_heads: u32,
        key_heads: u32,
        head_dim: u32,
        cache_length: u32,
        start: u64,
        position: DevicePtr,
        scratch: DevicePtr,
    ) -> Result<(), CudaError>;
    fn attention_causal_bf16(
        &mut self,
        query_key: DevicePtr,
        gate: DevicePtr,
        keys: DevicePtr,
        values: DevicePtr,
        output: DevicePtr,
        rows: u32,
        query_heads: u32,
        key_heads: u32,
        head_dim: u32,
        cache_length: u32,
        start: u64,
        position: DevicePtr,
        scratch: DevicePtr,
    ) -> Result<(), CudaError>;
    fn gdn_project_control_fp32(
        &mut self,
        input: DevicePtr,
        a_weight: DevicePtr,
        b_weight: DevicePtr,
        a_log: DevicePtr,
        dt_bias: DevicePtr,
        alpha: DevicePtr,
        beta: DevicePtr,
        rows: u32,
        width: u32,
        heads: u32,
    ) -> Result<(), CudaError>;
    fn gdn_convolution_bf16(
        &mut self,
        query_key: DevicePtr,
        value_z: DevicePtr,
        convolution_weights: DevicePtr,
        convolution_state: DevicePtr,
        output: DevicePtr,
        rows: u32,
        query_key_width: u32,
        value_width: u32,
        convolution_width: u32,
        kernel_size: u32,
    ) -> Result<(), CudaError>;
    fn gdn_recurrence_bf16(
        &mut self,
        convolved: DevicePtr,
        alpha: DevicePtr,
        beta: DevicePtr,
        recurrent_state: DevicePtr,
        output: DevicePtr,
        rows: u32,
        key_heads: u32,
        value_heads: u32,
        key_head_dim: u32,
        value_head_dim: u32,
        output_scale: f32,
    ) -> Result<(), CudaError>;
    fn gdn_gated_rms_norm_bf16(
        &mut self,
        recurrent: DevicePtr,
        value_z: DevicePtr,
        norm_weight: DevicePtr,
        output: DevicePtr,
        rows: u32,
        value_heads: u32,
        head_dim: u32,
        epsilon: f32,
    ) -> Result<(), CudaError>;
    fn residual_rms_norm_bf16(
        &mut self,
        residual: DevicePtr,
        delta: DevicePtr,
        weight: DevicePtr,
        hidden: DevicePtr,
        normalized: DevicePtr,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), CudaError>;
    fn rms_norm_unit_offset_bf16(
        &mut self,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), CudaError>;
    fn swiglu_bf16(
        &mut self,
        gate_up: DevicePtr,
        output: DevicePtr,
        rows: u32,
        intermediate_size: u32,
    ) -> Result<(), CudaError>;
    fn residual_add_bf16(
        &mut self,
        residual: DevicePtr,
        delta: DevicePtr,
        output: DevicePtr,
        rows: u32,
        width: u32,
    ) -> Result<(), CudaError>;
    fn argmax_bf16(&mut self, logits: DevicePtr, count: u32, result: DevicePtr) -> Result<(), CudaError>;
    fn argmax_logprob_bf16(&mut self, logits: DevicePtr, count: u32, result: DevicePtr) -> Result<(), CudaError>;
    fn selected_token(&mut self, key: DevicePtr, ids: DevicePtr, token: DevicePtr) -> Result<(), CudaError>;

    /// Selects row-exact execution for the launches queued after this call.
    fn set_row_exact(&mut self, enabled: bool);

    /// Whether the operations were made in row-exact mode ([`SubmitMode::row_exact`]). It does not
    /// follow [`KernelOps::set_row_exact`], which selects around part of a stage and leaves this answer alone: a
    /// format that asks reads the mode of the quantum, not the selection of the moment.
    fn row_exact(&self) -> bool;

    /// Whether the launches run on the exact-numerics kernels (every quantized linear on its scalar reference, every
    /// relaxed-order operator on its exact twin): the decision of a format that picks its route by it.
    fn exact_numerics(&self) -> bool;

    /// Whether kernel `index` of the shared linear table runs on this device: the kernels of an
    /// architecture-specific module are unavailable elsewhere ([`CudaLibrary::linear_kernel_available`](crate::CudaLibrary::linear_kernel_available)).
    fn linear_kernel_available(&self, index: usize) -> Result<bool, CudaError>;

    /// Queues the copy of `bytes` device bytes into a pinned buffer at `offset`, for a buffer the host shares by `&`.
    ///
    /// # Safety
    /// The copy runs after this call returns. `destination` must stay alive until the lane's work up to this copy has
    /// retired, and the host must not access its bytes in that range, other than through [`PinnedBuffer::word`],
    /// before then. No other copy may write the range meanwhile.
    unsafe fn copy_device_to_readback_shared(
        &mut self,
        destination: &PinnedBuffer,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// Queues the copy of `bytes` bytes at `offset` of a pinned buffer to the device.
    ///
    /// # Safety
    /// The copy runs after this call returns. `source` must stay alive, and its bytes in that range unmodified, until
    /// the lane's work up to this copy has retired.
    unsafe fn copy_upload_to_device(
        &mut self,
        destination: DevicePtr,
        source: &PinnedBuffer,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// Queues the copy of `bytes` device bytes into a pinned buffer at `offset`.
    ///
    /// # Safety
    /// The copy runs after this call returns. `destination` must stay alive, and its bytes in that range unread,
    /// until the lane's work up to this copy has retired.
    unsafe fn copy_device_to_readback(
        &mut self,
        destination: &mut PinnedBuffer,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// Queues the copy of `bytes` bytes at `offset` of a host-weights arena to the device.
    ///
    /// # Safety
    /// The copy runs after this call returns. `source` must stay alive, and its bytes in that range unmodified, until
    /// the lane's work up to this copy has retired.
    unsafe fn copy_host_weights_to_device(
        &mut self,
        destination: DevicePtr,
        source: &HostWeightsArena,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// Queues the copy of `bytes` device bytes into a host-weights arena at `offset`.
    ///
    /// # Safety
    /// The copy runs after this call returns. `destination` must stay alive, and its bytes in that range neither read
    /// nor written by the host, until the lane's work up to this copy has retired. No other copy may write the range
    /// meanwhile.
    unsafe fn copy_device_to_host_weights(
        &mut self,
        destination: &HostWeightsArena,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError>;
}

/// [`KernelOps`] on a lane that also allocates and frees device memory in stream order, on the same lane the kernels
/// run on: what the owner's work on a sequence (its cache's pages, its state) needs, where an allocation and the copy
/// or launch that first uses it must be ordered on one stream.
pub trait LaneOps: KernelOps {
    /// The marker of value 0 of this lane: reached before any work, and naming the lane that allocations here are
    /// ordered on.
    fn home(&self) -> Result<Marker, CudaError>;

    /// Allocates `bytes` in stream order on this lane: usable by the work queued after this call on it.
    fn alloc_async(&mut self, bytes: usize) -> Result<DeviceBuffer, CudaError>;

    /// Frees `buffer` in stream order on this lane, after the work queued before this call on it. A buffer that was
    /// not allocated stream-ordered is refused with an error, and dropped (the plain, waiting free).
    fn free_async(&mut self, buffer: DeviceBuffer) -> Result<(), CudaError>;
}

#[cfg(test)]
mod range_tests {
    use super::*;

    #[test]
    fn a_range_starts_at_its_offset_and_may_end_at_the_end() {
        let base = 0x1000 as *mut u8;
        assert_eq!(range_start(base, 64, 0, 64, "buffer").unwrap() as usize, 0x1000);
        assert_eq!(range_start(base, 64, 16, 48, "buffer").unwrap() as usize, 0x1010);
        assert_eq!(range_start(base, 64, 64, 0, "buffer").unwrap() as usize, 0x1040);
    }

    #[test]
    fn a_range_past_the_end_or_overflowing_is_refused() {
        let base = 0x1000 as *mut u8;
        assert!(matches!(
            range_start(base, 64, 16, 49, "buffer"),
            Err(CudaError::Argument(_))
        ));
        assert!(matches!(
            range_start(base, 64, 65, 0, "buffer"),
            Err(CudaError::Argument(_))
        ));
        assert!(matches!(
            range_start(base, 64, usize::MAX, 2, "buffer"),
            Err(CudaError::Argument(_))
        ));
    }
}
