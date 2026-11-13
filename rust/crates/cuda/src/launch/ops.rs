//! [`KernelOps`] over a RustEE lane: each call is planned by its family's planner and the planned launches are enqueued,
//! in order, on the lane's enqueue handle.
//!
//! [`RusteeOps`] is what a stage launches through when the kernels run on RustEE: the handle a lane's `submit` lends, the
//! [`KernelTable`] that [`load`](super::load::load) made, and the [`Modes`] the native library keeps per thread
//! (row-exact, exact numerics). A planner's refusal is the status the C wrapper would have returned
//! ([`CudaError::Call`] with the library's code, so a caller that treats `ROUTE_UNAVAILABLE` as "take another route"
//! still can); a kernel the device lacks, or a launch RustEE refuses, is [`CudaError::Argument`] with the reason. Neither
//! falls back to the native library.
//!
//! Copies and the memset are the lane's own operations. RustEE has no pitched copy at this revision, so the 2D copy is
//! one copy per row (a single copy when both pitches equal the row).

use std::fmt;

use euhedral_gpu::cuda::CudaKernel;
use euhedral_gpu::{GpuError, LaneEnqueue};

use crate::error::CudaError;
use crate::launch::attention::{
    AppendShape, CacheFormat, CausalShape, RopeShape, plan_causal, plan_kv_append, plan_qk_norm_rope,
};
use crate::launch::catalog::{Launch, linear_table, lookup};
use crate::launch::elementwise::{
    ArgmaxShape, ResidualAddShape, ResidualRmsNormShape, RmsNormShape, SelectedTokenShape, SwigluShape,
    check_zero_device_memory, plan_argmax, plan_argmax_logprob, plan_residual_add, plan_residual_rms_norm,
    plan_rms_norm_unit_offset, plan_selected_token, plan_swiglu,
};
use crate::launch::embedding::{EmbedShape, plan_embed_q3, plan_embed_q3_p2e2};
use crate::launch::gdn::{
    ConvolutionShape, GatedRmsNormShape, ProjectControlShape, RecurrenceShape, plan_convolution, plan_gated_rms_norm,
    plan_project_control, plan_recurrence,
};
use crate::launch::load::KernelTable;
use crate::launch::nvfp4::{Nvfp4Operands, plan_linear, plan_native};
use crate::launch::queue::{EnqueueError, enqueue};
use crate::launch::table::{Geometry, plan_linear_launch, plan_qwen4_launch};
use crate::launch::{Modes, PlanError, advance};
use crate::memory::{DevicePtr, HostWeightsArena, PinnedBuffer};
use crate::ops::{KernelOps, arena_range, pinned_range};
use crate::sys::{
    EUHEDRAL_CUDA_FORMAT_MISMATCH as FORMAT_MISMATCH, EUHEDRAL_CUDA_INVALID_ARGUMENT as INVALID_ARGUMENT,
    EUHEDRAL_CUDA_ROUTE_UNAVAILABLE as ROUTE_UNAVAILABLE, EUHEDRAL_CUDA_SIZE_OVERFLOW as SIZE_OVERFLOW,
};

/// The error of a planner's refusal: the call it refused and the library's status for it.
fn refused(call: &'static str, error: PlanError) -> CudaError {
    let status = match error {
        PlanError::InvalidArgument => INVALID_ARGUMENT,
        PlanError::SizeOverflow => SIZE_OVERFLOW,
        PlanError::FormatMismatch => FORMAT_MISMATCH,
        PlanError::RouteUnavailable => ROUTE_UNAVAILABLE,
    };
    CudaError::Call { call, status }
}

fn rejected(call: &'static str) -> CudaError {
    CudaError::Call {
        call,
        status: INVALID_ARGUMENT,
    }
}

/// The error of a launch that was planned and not enqueued.
fn not_enqueued(error: &EnqueueError) -> CudaError {
    CudaError::Argument(error.to_string())
}

fn lane_error(call: &'static str, error: &GpuError) -> CudaError {
    CudaError::Argument(format!("{call}: {error}"))
}

/// What a stage launches through when the kernels run on a RustEE lane.
pub struct RusteeOps<'a, Q> {
    queue: &'a mut Q,
    table: &'a KernelTable,
    modes: Modes,
    began_row_exact: bool,
}

impl<Q> fmt::Debug for RusteeOps<'_, Q> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RusteeOps")
            .field("modes", &self.modes)
            .field("began_row_exact", &self.began_row_exact)
            .finish_non_exhaustive()
    }
}

impl<'a, Q> RusteeOps<'a, Q>
where
    Q: LaneEnqueue<Kernel = CudaKernel>,
{
    /// Launches on `queue` the kernels of `table`, in `modes`. The quantum's row-exactness is `modes.row_exact` as it is
    /// now: [`KernelOps::row_exact`] keeps answering it after [`KernelOps::set_row_exact`] selects around part of a stage.
    pub fn new(queue: &'a mut Q, table: &'a KernelTable, modes: Modes) -> Self {
        Self {
            queue,
            table,
            modes,
            began_row_exact: modes.row_exact,
        }
    }

    /// The modes later launches are planned in.
    pub fn modes(&self) -> Modes {
        self.modes
    }

    /// Enqueues `launches` in order.
    fn enqueue_all(&mut self, launches: &[Launch]) -> Result<(), CudaError> {
        for launch in launches {
            // SAFETY: the contract of `KernelOps`, as for the native wrappers: the addresses a stage names stay valid
            // for the kernels' accesses until the lane's work has retired. The plan's argument sizes are checked
            // against the kernel's parameters by `enqueue`.
            unsafe { enqueue(self.queue, self.table, launch) }.map_err(|error| not_enqueued(&error))?;
        }
        Ok(())
    }

    /// Plans a call with `plan` and enqueues each launch as the planner produces it. A planner that refuses has
    /// produced nothing before it refuses, except in the row loops that refuse an unavailable kernel; the launches
    /// already enqueued stay queued, as with the native wrappers.
    fn planned(
        &mut self,
        call: &'static str,
        plan: impl FnOnce(&mut dyn FnMut(Launch)) -> Result<(), PlanError>,
    ) -> Result<(), CudaError> {
        let (queue, table) = (&mut *self.queue, self.table);
        let mut failure = None;
        let planned = plan(&mut |launch| {
            if failure.is_none() {
                // SAFETY: as in `enqueue_all`.
                if let Err(error) = unsafe { enqueue(queue, table, &launch) } {
                    failure = Some(error);
                }
            }
        });
        if let Some(error) = failure {
            return Err(not_enqueued(&error));
        }
        planned.map_err(|error| refused(call, error))
    }

    fn append(&mut self, call: &'static str, format: CacheFormat, shape: AppendShape) -> Result<(), CudaError> {
        self.planned(call, |sink| plan_kv_append(format, &shape, &mut |launch| sink(launch)))
    }

    fn causal(&mut self, call: &'static str, format: CacheFormat, shape: CausalShape) -> Result<(), CudaError> {
        let modes = self.modes;
        self.planned(call, |sink| {
            plan_causal(format, &shape, modes, &mut |launch| sink(launch))
        })
    }
}

impl<Q> KernelOps for RusteeOps<'_, Q>
where
    Q: LaneEnqueue<Kernel = CudaKernel>,
{
    fn copy_device_to_device(
        &mut self,
        destination: DevicePtr,
        source: DevicePtr,
        bytes: u64,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_copy_device_to_device";
        if bytes == 0 {
            return Ok(());
        }
        if destination.is_null() || source.is_null() {
            return Err(rejected(CALL));
        }
        let bytes = usize::try_from(bytes).map_err(|_| refused(CALL, PlanError::SizeOverflow))?;
        // SAFETY: the contract of `KernelOps`: both ranges stay allocated until the lane's work has retired.
        unsafe { self.queue.copy_on_device(destination.get(), source.get(), bytes) }
            .map_err(|error| lane_error(CALL, &error))
    }

    fn copy_device_to_device_2d(
        &mut self,
        destination: DevicePtr,
        destination_pitch: u64,
        source: DevicePtr,
        source_pitch: u64,
        row_bytes: u64,
        rows: u64,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_copy_device_to_device_2d";
        if row_bytes == 0 || rows == 0 {
            return Ok(());
        }
        if destination.is_null() || source.is_null() || destination_pitch < row_bytes || source_pitch < row_bytes {
            return Err(rejected(CALL));
        }
        let row = usize::try_from(row_bytes).map_err(|_| refused(CALL, PlanError::SizeOverflow))?;
        if destination_pitch == row_bytes && source_pitch == row_bytes {
            let total = row_bytes
                .checked_mul(rows)
                .ok_or_else(|| refused(CALL, PlanError::SizeOverflow))?;
            return self.copy_device_to_device(destination, source, total);
        }
        let end = |pitch: u64| {
            pitch
                .checked_mul(rows - 1)
                .and_then(|last| last.checked_add(row_bytes))
                .ok_or_else(|| refused(CALL, PlanError::SizeOverflow))
        };
        end(destination_pitch)?;
        end(source_pitch)?;
        for index in 0..rows {
            let to = advance(destination, index * destination_pitch);
            let from = advance(source, index * source_pitch);
            // SAFETY: as in `copy_device_to_device`.
            unsafe { self.queue.copy_on_device(to.get(), from.get(), row) }
                .map_err(|error| lane_error(CALL, &error))?;
        }
        Ok(())
    }

    fn zero_device_memory(&mut self, address: DevicePtr, bytes: u64) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_zero_device_memory";
        check_zero_device_memory(address, bytes).map_err(|error| refused(CALL, error))?;
        let bytes = usize::try_from(bytes).map_err(|_| refused(CALL, PlanError::SizeOverflow))?;
        // SAFETY: as in `copy_device_to_device`.
        unsafe { self.queue.memset(address.get(), 0, bytes) }.map_err(|error| lane_error(CALL, &error))
    }

    fn embed_q3(
        &mut self,
        token_ids: DevicePtr,
        embedding: DevicePtr,
        hidden: DevicePtr,
        token_count: u32,
        vocabulary_size: u32,
        hidden_size: u32,
        embedding_bytes: u64,
    ) -> Result<(), CudaError> {
        let shape = EmbedShape {
            token_ids,
            embedding,
            hidden,
            token_count,
            vocabulary_size,
            hidden_size,
            embedding_bytes,
        };
        self.planned("euhedral_cuda_embed_q3", |sink| {
            plan_embed_q3(&shape, &mut |launch| sink(launch))
        })
    }

    fn embed_q3_p2e2(
        &mut self,
        token_ids: DevicePtr,
        embedding: DevicePtr,
        hidden: DevicePtr,
        token_count: u32,
        vocabulary_size: u32,
        hidden_size: u32,
        embedding_bytes: u64,
    ) -> Result<(), CudaError> {
        let shape = EmbedShape {
            token_ids,
            embedding,
            hidden,
            token_count,
            vocabulary_size,
            hidden_size,
            embedding_bytes,
        };
        self.planned("euhedral_cuda_embed_q3_p2e2", |sink| {
            plan_embed_q3_p2e2(&shape, &mut |launch| sink(launch))
        })
    }

    fn linear_launch(
        &mut self,
        kernel: i32,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        words: &[u64],
        sizes: &[u8],
    ) -> Result<(), CudaError> {
        let geometry = Geometry {
            grid,
            block,
            shared_bytes,
        };
        let target = self.table.architecture_target();
        self.planned("euhedral_cuda_linear_launch", |sink| {
            plan_linear_launch(kernel, geometry, words, sizes, target, &mut |launch| sink(launch))
        })
    }

    fn qwen4_launch(
        &mut self,
        kernel: i32,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        words: &[u64],
        sizes: &[u8],
    ) -> Result<(), CudaError> {
        let geometry = Geometry {
            grid,
            block,
            shared_bytes,
        };
        let target = self.table.architecture_target();
        self.planned("euhedral_cuda_qwen4_launch", |sink| {
            plan_qwen4_launch(kernel, geometry, words, sizes, target, &mut |launch| sink(launch))
        })
    }

    fn linear_nvfp4_bf16(
        &mut self,
        input: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        rows: u32,
        in_features: u32,
        out_features: u32,
        weights_bytes: u64,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_linear_nvfp4_bf16";
        let operands = Nvfp4Operands {
            input,
            weights,
            output,
            rows,
            in_features,
            out_features,
            weights_byte_size: weights_bytes,
        };
        let plan = plan_linear(&operands, self.modes.exact_numerics).map_err(|error| refused(CALL, error))?;
        self.enqueue_all(plan.launches())
    }

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
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_linear_nvfp4_native_bf16";
        let operands = Nvfp4Operands {
            input,
            weights,
            output,
            rows,
            in_features,
            out_features,
            weights_byte_size: weights_bytes,
        };
        let plan = plan_native(&operands, scratch, scratch_bytes, self.modes.exact_numerics)
            .map_err(|error| refused(CALL, error))?;
        self.enqueue_all(plan.launches())
    }

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
    ) -> Result<(), CudaError> {
        let shape = RopeShape {
            query_key,
            query_norm,
            key_norm,
            output,
            position,
            rows,
            query_heads,
            key_value_heads,
            head_dim,
            rotary_dim,
            start_position,
            epsilon,
            rope_theta,
        };
        let modes = self.modes;
        self.planned("euhedral_cuda_attention_qk_norm_rope_bf16", |sink| {
            plan_qk_norm_rope(&shape, modes, &mut |launch| sink(launch))
        })
    }

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
    ) -> Result<(), CudaError> {
        let shape = AppendShape {
            query_key,
            gate_value: gate,
            key_cache: keys,
            value_cache: values,
            position,
            rows,
            query_width,
            key_value_width: key_width,
            start_position: start,
        };
        self.append("euhedral_cuda_attention_kv_append_nvfp4", CacheFormat::Nvfp4, shape)
    }

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
    ) -> Result<(), CudaError> {
        let shape = AppendShape {
            query_key,
            gate_value: gate,
            key_cache: keys,
            value_cache: values,
            position,
            rows,
            query_width,
            key_value_width: key_width,
            start_position: start,
        };
        self.append("euhedral_cuda_attention_kv_append_bf16", CacheFormat::Bf16, shape)
    }

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
    ) -> Result<(), CudaError> {
        let shape = CausalShape {
            query_key,
            gate,
            keys,
            values,
            output,
            scratch,
            position,
            rows,
            query_heads,
            key_heads,
            head_dim,
            cache_length,
            start,
        };
        self.causal("euhedral_cuda_attention_causal_nvfp4", CacheFormat::Nvfp4, shape)
    }

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
    ) -> Result<(), CudaError> {
        let shape = CausalShape {
            query_key,
            gate,
            keys,
            values,
            output,
            scratch,
            position,
            rows,
            query_heads,
            key_heads,
            head_dim,
            cache_length,
            start,
        };
        self.causal("euhedral_cuda_attention_causal_bf16", CacheFormat::Bf16, shape)
    }

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
    ) -> Result<(), CudaError> {
        let shape = ProjectControlShape {
            input,
            a_weight,
            b_weight,
            a_log,
            dt_bias,
            alpha,
            beta,
            rows,
            width,
            heads,
        };
        self.planned("euhedral_cuda_gdn_project_control_fp32", |sink| {
            plan_project_control(&shape, &mut |launch| sink(launch))
        })
    }

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
    ) -> Result<(), CudaError> {
        let shape = ConvolutionShape {
            query_key,
            value_z,
            weights: convolution_weights,
            state: convolution_state,
            output,
            rows,
            query_key_width,
            value_width,
            convolution_width,
            kernel_size,
        };
        self.planned("euhedral_cuda_gdn_convolution_bf16", |sink| {
            plan_convolution(&shape, &mut |launch| sink(launch))
        })
    }

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
    ) -> Result<(), CudaError> {
        let shape = RecurrenceShape {
            convolved,
            alpha,
            beta,
            recurrent_state,
            output,
            rows,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            output_scale,
        };
        let modes = self.modes;
        self.planned("euhedral_cuda_gdn_recurrence_bf16", |sink| {
            plan_recurrence(&shape, modes, &mut |launch| sink(launch))
        })
    }

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
    ) -> Result<(), CudaError> {
        let shape = GatedRmsNormShape {
            recurrent,
            value_z,
            norm_weight,
            output,
            rows,
            value_heads,
            head_dim,
            epsilon,
        };
        self.planned("euhedral_cuda_gdn_gated_rms_norm_bf16", |sink| {
            plan_gated_rms_norm(&shape, &mut |launch| sink(launch))
        })
    }

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
    ) -> Result<(), CudaError> {
        let shape = ResidualRmsNormShape {
            residual,
            delta,
            weight,
            hidden,
            normalized,
            rows,
            width,
            epsilon,
        };
        let modes = self.modes;
        self.planned("euhedral_cuda_residual_rms_norm_bf16", |sink| {
            plan_residual_rms_norm(&shape, modes, &mut |launch| sink(launch))
        })
    }

    fn rms_norm_unit_offset_bf16(
        &mut self,
        input: DevicePtr,
        weight: DevicePtr,
        output: DevicePtr,
        rows: u32,
        width: u32,
        epsilon: f32,
    ) -> Result<(), CudaError> {
        let shape = RmsNormShape {
            input,
            weight,
            output,
            rows,
            width,
            epsilon,
        };
        self.planned("euhedral_cuda_rms_norm_unit_offset_bf16", |sink| {
            plan_rms_norm_unit_offset(&shape, &mut |launch| sink(launch))
        })
    }

    fn swiglu_bf16(
        &mut self,
        gate_up: DevicePtr,
        output: DevicePtr,
        rows: u32,
        intermediate_size: u32,
    ) -> Result<(), CudaError> {
        let shape = SwigluShape {
            gate_up,
            output,
            rows,
            intermediate_size,
        };
        self.planned("euhedral_cuda_swiglu_bf16", |sink| {
            plan_swiglu(&shape, &mut |launch| sink(launch))
        })
    }

    fn residual_add_bf16(
        &mut self,
        residual: DevicePtr,
        delta: DevicePtr,
        output: DevicePtr,
        rows: u32,
        width: u32,
    ) -> Result<(), CudaError> {
        let shape = ResidualAddShape {
            residual,
            delta,
            output,
            rows,
            width,
        };
        self.planned("euhedral_cuda_residual_add_bf16", |sink| {
            plan_residual_add(&shape, &mut |launch| sink(launch))
        })
    }

    fn argmax_bf16(&mut self, logits: DevicePtr, count: u32, result: DevicePtr) -> Result<(), CudaError> {
        let shape = ArgmaxShape { logits, count, result };
        self.planned("euhedral_cuda_argmax_bf16", |sink| {
            plan_argmax(&shape, &mut |launch| sink(launch))
        })
    }

    fn argmax_logprob_bf16(&mut self, logits: DevicePtr, count: u32, result: DevicePtr) -> Result<(), CudaError> {
        let shape = ArgmaxShape { logits, count, result };
        self.planned("euhedral_cuda_argmax_logprob_bf16", |sink| {
            plan_argmax_logprob(&shape, &mut |launch| sink(launch))
        })
    }

    fn selected_token(&mut self, key: DevicePtr, ids: DevicePtr, token: DevicePtr) -> Result<(), CudaError> {
        let shape = SelectedTokenShape { key, ids, token };
        self.planned("euhedral_cuda_selected_token", |sink| {
            plan_selected_token(&shape, &mut |launch| sink(launch))
        })
    }

    fn set_row_exact(&mut self, enabled: bool) {
        self.modes.row_exact = enabled;
    }

    fn row_exact(&self) -> bool {
        self.began_row_exact
    }

    fn exact_numerics(&self) -> bool {
        self.modes.exact_numerics
    }

    fn linear_kernel_available(&self, index: usize) -> Result<bool, CudaError> {
        let entry = linear_table()
            .get(index)
            .ok_or_else(|| CudaError::Argument(format!("linear kernel {index}")))?;
        let kernel = lookup(entry.name).ok_or_else(|| CudaError::Argument(format!("linear kernel {index}")))?;
        Ok(self.table.resolved(kernel).is_some())
    }

    unsafe fn copy_device_to_readback_shared(
        &mut self,
        destination: &PinnedBuffer,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_copy_device_to_readback";
        let start = pinned_range(destination, offset, bytes)?;
        if bytes == 0 || source.is_null() {
            return Err(rejected(CALL));
        }
        // SAFETY: the range is inside the pinned buffer; the caller's contract keeps it alive and the host's accesses
        // atomic until the copy has retired.
        unsafe { self.queue.copy_to_host(start.cast_mut(), source.get(), bytes) }
            .map_err(|error| lane_error(CALL, &error))
    }

    unsafe fn copy_host_weights_to_device(
        &mut self,
        destination: DevicePtr,
        source: &HostWeightsArena,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_copy_upload_to_device";
        let start = arena_range(source, offset, bytes)?;
        if bytes == 0 || destination.is_null() {
            return Err(rejected(CALL));
        }
        // SAFETY: the range is inside the arena, which the caller keeps alive and unmodified until the copy has retired.
        unsafe { self.queue.copy_to_device(destination.get(), start.cast_const(), bytes) }
            .map_err(|error| lane_error(CALL, &error))
    }

    unsafe fn copy_device_to_host_weights(
        &mut self,
        destination: &HostWeightsArena,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError> {
        const CALL: &str = "euhedral_cuda_copy_device_to_readback";
        let start = arena_range(destination, offset, bytes)?;
        if bytes == 0 || source.is_null() {
            return Err(rejected(CALL));
        }
        // SAFETY: the range is inside the arena, which the caller keeps alive and unread until the copy has retired.
        unsafe { self.queue.copy_to_host(start, source.get(), bytes) }.map_err(|error| lane_error(CALL, &error))
    }
}

#[cfg(test)]
mod tests;
