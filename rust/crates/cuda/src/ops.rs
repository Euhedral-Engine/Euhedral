//! One function per C entry point the pilot path uses. Each takes the [`Submission`] first and queues its work on the
//! submission's lane; none waits for the device unless it says so.
//!
//! Device addresses are [`DevicePtr`]s: the host never dereferences them, and a wrong device address is reported by
//! the driver (or faults the kernel), not by the host's memory model, so the functions on device memory are safe.
//! Host memory is different. The two asynchronous host copies read or write host memory after they return and are
//! `unsafe`; so is taking the device address of mapped host memory
//! ([`PinnedBuffer::device_address`](crate::PinnedBuffer::device_address)), which is how a kernel or a flag operation
//! (`wait_host_flag`, `write_host_flag`) gets to host bytes. The host's side of that contract is stated there.
//!
//! Every function mirrors the parameter list of its C entry point (`native/include/euhedral_cuda.h`), which documents
//! the shapes, alignments and routes.

// Each function mirrors one C entry point's parameter list.
#![allow(clippy::too_many_arguments)]

use crate::error::{CudaError, check};
use crate::library::CudaLibrary;
use crate::memory::{DevicePtr, HostWeightsArena, PinnedBuffer};
use crate::submission::Submission;
use std::sync::Arc;

/// Calls the C entry point `$name` and turns its status into a `Result`.
macro_rules! entry {
    ($submission:expr, $name:ident($($argument:expr),* $(,)?)) => {{
        // SAFETY: the arguments are plain values and device addresses, which the host never dereferences; the
        // submission has selected the lane's stream; pointers into host memory are built from slices that outlive
        // the call.
        let status = unsafe { $submission.api().$name($($argument),*) };
        check(stringify!($name), status)
    }};
}

/// The error for an operation that synchronizes or allocates, which a capture cannot take.
fn refuse_capture(submission: &Submission<'_>, what: &'static str) -> Result<(), CudaError> {
    if submission.lane().is_capturing() {
        Err(CudaError::Capturing(what))
    } else {
        Ok(())
    }
}

// Copies.

/// Copies `source` to the device and returns when it is done: the selected stream is synchronized. Startup and tests.
/// Refused (`Capturing`) while the lane is captured.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, synchronous copy: documented as startup and tests only"
)]
pub fn copy_host_to_device(
    submission: &Submission<'_>,
    destination: DevicePtr,
    source: &[u8],
) -> Result<(), CudaError> {
    refuse_capture(submission, "a blocking host copy")?;
    entry!(
        submission,
        euhedral_cuda_copy_host_to_device(destination.as_mut(), source.as_ptr().cast(), source.len() as u64)
    )
}

/// Copies device memory into `destination` and returns when it is done: the selected stream is synchronized.
/// Startup and tests. Refused (`Capturing`) while the lane is captured.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, synchronous copy: documented as startup and tests only"
)]
pub fn copy_device_to_host(
    submission: &Submission<'_>,
    destination: &mut [u8],
    source: DevicePtr,
) -> Result<(), CudaError> {
    refuse_capture(submission, "a blocking host copy")?;
    entry!(
        submission,
        euhedral_cuda_copy_device_to_host(
            destination.as_mut_ptr().cast(),
            source.as_const(),
            destination.len() as u64
        )
    )
}

/// Queues the copy of `bytes` bytes at `offset` of a pinned buffer to the device.
///
/// # Safety
/// The copy runs after this call returns. `source` must stay alive, and its bytes in that range unmodified, until the
/// lane's work up to this copy has retired.
pub unsafe fn copy_upload_to_device(
    submission: &Submission<'_>,
    destination: DevicePtr,
    source: &PinnedBuffer,
    offset: usize,
    bytes: usize,
) -> Result<(), CudaError> {
    let start = pinned_range(source, offset, bytes)?;
    // SAFETY: `start` is inside the pinned buffer, which the caller keeps alive and unmodified.
    entry!(
        submission,
        euhedral_cuda_copy_upload_to_device(destination.as_mut(), start.cast(), bytes as u64)
    )
}

/// Queues the copy of `bytes` device bytes into a pinned buffer at `offset`.
///
/// # Safety
/// The copy runs after this call returns. `destination` must stay alive, and its bytes in that range unread, until the
/// lane's work up to this copy has retired.
pub unsafe fn copy_device_to_readback(
    submission: &Submission<'_>,
    destination: &mut PinnedBuffer,
    offset: usize,
    source: DevicePtr,
    bytes: usize,
) -> Result<(), CudaError> {
    pinned_range(destination, offset, bytes)?;
    // SAFETY: the range is inside the pinned buffer, which the caller keeps alive and unread.
    let start = unsafe { destination.as_mut_ptr().add(offset) };
    entry!(
        submission,
        euhedral_cuda_copy_device_to_readback(start.cast(), source.as_const(), bytes as u64)
    )
}

/// Queues the copy of `bytes` device bytes into a pinned buffer at `offset`, for a buffer the host shares by `&`: the
/// host reads it only through [`PinnedBuffer::word`], and only once the lane's work up to this copy has retired
/// ([`copy_device_to_readback`] is the exclusive form).
///
/// # Safety
/// The copy runs after this call returns. `destination` must stay alive until the lane's work up to this copy has
/// retired, and the host must not access its bytes in that range, other than through [`PinnedBuffer::word`], before
/// then. No other copy may write the range meanwhile.
pub unsafe fn copy_device_to_readback_shared(
    submission: &Submission<'_>,
    destination: &PinnedBuffer,
    offset: usize,
    source: DevicePtr,
    bytes: usize,
) -> Result<(), CudaError> {
    let start = pinned_range(destination, offset, bytes)?;
    // SAFETY: the range is inside the pinned buffer, which the caller keeps alive; the caller's contract keeps the
    // host's accesses atomic until the copy has retired.
    entry!(
        submission,
        euhedral_cuda_copy_device_to_readback(start.cast_mut().cast(), source.as_const(), bytes as u64)
    )
}

/// Queues the copy of `bytes` bytes at `offset` of a host-weights arena to the device.
///
/// # Safety
/// The copy runs after this call returns. `source` must stay alive, and its bytes in that range unmodified, until the
/// lane's work up to this copy has retired.
pub unsafe fn copy_host_weights_to_device(
    submission: &Submission<'_>,
    destination: DevicePtr,
    source: &HostWeightsArena,
    offset: usize,
    bytes: usize,
) -> Result<(), CudaError> {
    let start = arena_range(source, offset, bytes)?;
    // SAFETY: `start` is inside the arena, which the caller keeps alive and unmodified.
    entry!(
        submission,
        euhedral_cuda_copy_upload_to_device(destination.as_mut(), start.cast_const().cast(), bytes as u64)
    )
}

/// Queues the copy of `bytes` device bytes into a host-weights arena at `offset`.
///
/// # Safety
/// The copy runs after this call returns. `destination` must stay alive, and its bytes in that range neither read
/// nor written by the host, until the lane's work up to this copy has retired. No other copy may write the range
/// meanwhile.
pub unsafe fn copy_device_to_host_weights(
    submission: &Submission<'_>,
    destination: &HostWeightsArena,
    offset: usize,
    source: DevicePtr,
    bytes: usize,
) -> Result<(), CudaError> {
    let start = arena_range(destination, offset, bytes)?;
    // SAFETY: the range is inside the arena, which the caller keeps alive and unread.
    entry!(
        submission,
        euhedral_cuda_copy_device_to_readback(start.cast(), source.as_const(), bytes as u64)
    )
}

/// The address of `bytes` at `offset` of `arena`, refused when they leave it.
fn arena_range(arena: &HostWeightsArena, offset: usize, bytes: usize) -> Result<*mut u8, CudaError> {
    match offset.checked_add(bytes) {
        // SAFETY: `offset` is within the arena.
        Some(end) if end <= arena.len() => Ok(unsafe { arena.device_writable_ptr().add(offset) }),
        _ => Err(CudaError::Argument(format!(
            "{bytes} bytes at offset {offset} lie outside a {} byte host arena",
            arena.len()
        ))),
    }
}

/// The address of `bytes` at `offset` of `buffer`, refused when they leave it.
fn pinned_range(buffer: &PinnedBuffer, offset: usize, bytes: usize) -> Result<*const u8, CudaError> {
    match offset.checked_add(bytes) {
        // SAFETY: `offset` is within the buffer.
        Some(end) if end <= buffer.len() => Ok(unsafe { buffer.as_ptr().add(offset) }),
        _ => Err(CudaError::Argument(format!(
            "{bytes} bytes at offset {offset} lie outside a {} byte pinned buffer",
            buffer.len()
        ))),
    }
}

pub fn copy_device_to_device(
    submission: &Submission<'_>,
    destination: DevicePtr,
    source: DevicePtr,
    bytes: u64,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_copy_device_to_device(destination.as_mut(), source.as_const(), bytes)
    )
}

/// Copies `rows` rows of `row_bytes` between pitched device regions.
pub fn copy_device_to_device_2d(
    submission: &Submission<'_>,
    destination: DevicePtr,
    destination_pitch: u64,
    source: DevicePtr,
    source_pitch: u64,
    row_bytes: u64,
    rows: u64,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_copy_device_to_device_2d(
            destination.as_mut(),
            destination_pitch,
            source.as_const(),
            source_pitch,
            row_bytes,
            rows,
        )
    )
}

pub fn zero_device_memory(submission: &Submission<'_>, address: DevicePtr, bytes: u64) -> Result<(), CudaError> {
    entry!(submission, euhedral_cuda_zero_device_memory(address.as_mut(), bytes))
}

/// Selects row-exact execution for the launches queued after this call on the calling thread (the `row_exact`
/// of the submission's [`SubmitMode`](crate::SubmitMode) is selected when it begins and cleared when it ends).
/// A stage that must run part of its work row-exact, and the rest as its quantum does, selects around that part
/// and then restores the submission's mode.
pub fn select_row_exact(submission: &Submission<'_>, enabled: bool) {
    // SAFETY: the call sets this thread's launch state, which the submission's drop clears.
    unsafe { submission.api().euhedral_cuda_row_exact_select(i32::from(enabled)) }
}

// Stream flags.

/// Queues a wait until the host-mapped 32-bit word at `flag` equals `value`.
pub fn wait_host_flag(submission: &Submission<'_>, flag: DevicePtr, value: u32) -> Result<(), CudaError> {
    entry!(submission, euhedral_cuda_wait_host_flag(flag.get(), value))
}

/// Queues a write of `value` to the host-mapped 32-bit word at `flag`.
pub fn write_host_flag(submission: &Submission<'_>, flag: DevicePtr, value: u32) -> Result<(), CudaError> {
    entry!(submission, euhedral_cuda_write_host_flag(flag.get(), value))
}

// Embedding.

pub fn embed_q3(
    submission: &Submission<'_>,
    token_ids: DevicePtr,
    embedding: DevicePtr,
    hidden: DevicePtr,
    token_count: u32,
    vocabulary_size: u32,
    hidden_size: u32,
    embedding_bytes: u64,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_embed_q3(
            token_ids.as_const().cast(),
            embedding.as_const(),
            hidden.as_mut(),
            token_count,
            vocabulary_size,
            hidden_size,
            embedding_bytes,
        )
    )
}

pub fn embed_q3_p2e2(
    submission: &Submission<'_>,
    token_ids: DevicePtr,
    embedding: DevicePtr,
    hidden: DevicePtr,
    token_count: u32,
    vocabulary_size: u32,
    hidden_size: u32,
    embedding_bytes: u64,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_embed_q3_p2e2(
            token_ids.as_const().cast(),
            embedding.as_const(),
            hidden.as_mut(),
            token_count,
            vocabulary_size,
            hidden_size,
            embedding_bytes,
        )
    )
}

// Linear.

/// Launches kernel `kernel` of the shared linear table. `words` holds the arguments, each 8 bytes wide in the array
/// and `sizes[i]` (4 or 8) bytes wide in the kernel's signature; the library checks both against the compiled kernel.
pub fn linear_launch(
    submission: &Submission<'_>,
    kernel: i32,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    words: &[u64],
    sizes: &[u8],
) -> Result<(), CudaError> {
    if words.len() != sizes.len() {
        return Err(CudaError::Argument(format!(
            "a linear launch has {} argument words but {} sizes",
            words.len(),
            sizes.len()
        )));
    }
    let count = i32::try_from(words.len())
        .map_err(|_| CudaError::Argument("a linear launch has too many arguments".to_string()))?;
    entry!(
        submission,
        euhedral_cuda_linear_launch(
            kernel,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_bytes,
            words.as_ptr(),
            sizes.as_ptr(),
            count,
        )
    )
}

/// Launches kernel `kernel` of the Flash-Next table, with the arguments laid out as [`linear_launch`] takes them.
pub fn qwen4_launch(
    submission: &Submission<'_>,
    kernel: i32,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    words: &[u64],
    sizes: &[u8],
) -> Result<(), CudaError> {
    if words.len() != sizes.len() {
        return Err(CudaError::Argument(format!(
            "a qwen4 launch has {} argument words but {} sizes",
            words.len(),
            sizes.len()
        )));
    }
    let count = i32::try_from(words.len())
        .map_err(|_| CudaError::Argument("a qwen4 launch has too many arguments".to_string()))?;
    entry!(
        submission,
        euhedral_cuda_qwen4_launch(
            kernel,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_bytes,
            words.as_ptr(),
            sizes.as_ptr(),
            count,
        )
    )
}

/// NVFP4 linear on BF16 rows: one row on the decode kernel, 2 to 8 on its row twins, other shapes on the reference.
pub fn linear_nvfp4_bf16(
    submission: &Submission<'_>,
    input: DevicePtr,
    weights: DevicePtr,
    output: DevicePtr,
    rows: u32,
    in_features: u32,
    out_features: u32,
    weights_bytes: u64,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_linear_nvfp4_bf16(
            input.as_const(),
            weights.as_const(),
            output.as_mut(),
            rows,
            in_features,
            out_features,
            weights_bytes,
        )
    )
}

/// Native Blackwell NVFP4 linear. `ROUTE_UNAVAILABLE` (status -5) under exact numerics or when `in_features` is not a
/// multiple of 128.
pub fn linear_nvfp4_native_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_linear_nvfp4_native_bf16(
            input.as_const(),
            weights.as_const(),
            output.as_mut(),
            scratch.as_mut(),
            rows,
            in_features,
            out_features,
            weights_bytes,
            scratch_bytes,
        )
    )
}

// Attention.

pub fn attention_qk_norm_rope_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_attention_qk_norm_rope_bf16(
            query_key.as_const(),
            query_norm.as_const(),
            key_norm.as_const(),
            output.as_mut(),
            rows,
            query_heads,
            key_value_heads,
            head_dim,
            rotary_dim,
            start_position,
            position.as_const(),
            epsilon,
            rope_theta,
        )
    )
}

/// Appends the rows' keys and values to the NVFP4 pages (`keys` and `values` are device page-address tables).
pub fn attention_kv_append_nvfp4(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_attention_kv_append_nvfp4(
            query_key.as_const(),
            gate.as_const(),
            keys.as_mut(),
            values.as_mut(),
            rows,
            query_width,
            key_width,
            start,
            position.as_const(),
        )
    )
}

/// As [`attention_kv_append_nvfp4`] into BF16 pages: each head row stored as computed, 512 bytes.
pub fn attention_kv_append_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_attention_kv_append_bf16(
            query_key.as_const(),
            gate.as_const(),
            keys.as_mut(),
            values.as_mut(),
            rows,
            query_width,
            key_width,
            start,
            position.as_const(),
        )
    )
}

/// Causal attention over NVFP4 pages. `scratch` holds `query_heads * 64 * 258` FP32 elements for single-token decode
/// and is null for prefill.
pub fn attention_causal_nvfp4(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_attention_causal_nvfp4(
            query_key.as_const(),
            gate.as_const(),
            keys.as_const(),
            values.as_const(),
            output.as_mut(),
            rows,
            query_heads,
            key_heads,
            head_dim,
            cache_length,
            start,
            position.as_const(),
            scratch.as_mut(),
        )
    )
}

/// As [`attention_causal_nvfp4`] over BF16 pages.
pub fn attention_causal_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_attention_causal_bf16(
            query_key.as_const(),
            gate.as_const(),
            keys.as_const(),
            values.as_const(),
            output.as_mut(),
            rows,
            query_heads,
            key_heads,
            head_dim,
            cache_length,
            start,
            position.as_const(),
            scratch.as_mut(),
        )
    )
}

// Gated delta net.

pub fn gdn_project_control_fp32(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_gdn_project_control_fp32(
            input.as_const(),
            a_weight.as_const(),
            b_weight.as_const(),
            a_log.as_const().cast(),
            dt_bias.as_const().cast(),
            alpha.as_mut().cast(),
            beta.as_mut().cast(),
            rows,
            width,
            heads,
        )
    )
}

pub fn gdn_convolution_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_gdn_convolution_bf16(
            query_key.as_const(),
            value_z.as_const(),
            convolution_weights.as_const(),
            convolution_state.as_mut(),
            output.as_mut(),
            rows,
            query_key_width,
            value_width,
            convolution_width,
            kernel_size,
        )
    )
}

pub fn gdn_recurrence_bf16(
    submission: &Submission<'_>,
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
    entry!(
        submission,
        euhedral_cuda_gdn_recurrence_bf16(
            convolved.as_const(),
            alpha.as_const().cast(),
            beta.as_const().cast(),
            recurrent_state.as_mut().cast(),
            output.as_mut(),
            rows,
            key_heads,
            value_heads,
            key_head_dim,
            value_head_dim,
            output_scale,
        )
    )
}

pub fn gdn_gated_rms_norm_bf16(
    submission: &Submission<'_>,
    recurrent: DevicePtr,
    value_z: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    value_heads: u32,
    head_dim: u32,
    epsilon: f32,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_gdn_gated_rms_norm_bf16(
            recurrent.as_const(),
            value_z.as_const(),
            norm_weight.as_const(),
            output.as_mut(),
            rows,
            value_heads,
            head_dim,
            epsilon,
        )
    )
}

// Norms and elementwise.

/// `hidden = residual + delta` and `normalized = rms_norm(hidden) * weight`.
pub fn residual_rms_norm_bf16(
    submission: &Submission<'_>,
    residual: DevicePtr,
    delta: DevicePtr,
    weight: DevicePtr,
    hidden: DevicePtr,
    normalized: DevicePtr,
    rows: u32,
    width: u32,
    epsilon: f32,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_residual_rms_norm_bf16(
            residual.as_const(),
            delta.as_const(),
            weight.as_const(),
            hidden.as_mut(),
            normalized.as_mut(),
            rows,
            width,
            epsilon,
        )
    )
}

pub fn rms_norm_unit_offset_bf16(
    submission: &Submission<'_>,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    rows: u32,
    width: u32,
    epsilon: f32,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_rms_norm_unit_offset_bf16(
            input.as_const(),
            weight.as_const(),
            output.as_mut(),
            rows,
            width,
            epsilon,
        )
    )
}

pub fn swiglu_bf16(
    submission: &Submission<'_>,
    gate_up: DevicePtr,
    output: DevicePtr,
    rows: u32,
    intermediate_size: u32,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_swiglu_bf16(gate_up.as_const(), output.as_mut(), rows, intermediate_size)
    )
}

pub fn residual_add_bf16(
    submission: &Submission<'_>,
    residual: DevicePtr,
    delta: DevicePtr,
    output: DevicePtr,
    rows: u32,
    width: u32,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_residual_add_bf16(residual.as_const(), delta.as_const(), output.as_mut(), rows, width)
    )
}

// Sampling.

/// Greedy selection over `count` BF16 logits: one 64-bit key at `result` whose low word is `0xFFFFFFFF` minus the
/// selected token id (the lowest id among equal maxima), or 0 when no logit is selectable.
pub fn argmax_bf16(
    submission: &Submission<'_>,
    logits: DevicePtr,
    count: u32,
    result: DevicePtr,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_argmax_bf16(logits.as_const(), count, result.as_mut())
    )
}

/// As [`argmax_bf16`], writing 16 bytes: the key, then the selected logit's natural-log softmax probability as a float.
pub fn argmax_logprob_bf16(
    submission: &Submission<'_>,
    logits: DevicePtr,
    count: u32,
    result: DevicePtr,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_argmax_logprob_bf16(logits.as_const(), count, result.as_mut())
    )
}

/// Writes the token a selection key chose as an int32 at `token`: `ids[row]` of the selected shortlist row, or the
/// row itself when `ids` is [`DevicePtr::NULL`].
pub fn selected_token(
    submission: &Submission<'_>,
    key: DevicePtr,
    ids: DevicePtr,
    token: DevicePtr,
) -> Result<(), CudaError> {
    entry!(
        submission,
        euhedral_cuda_selected_token(key.as_const(), ids.as_const(), token.as_mut())
    )
}

/// The operations the stages queue, as a trait, so a stage does not name the submission it runs on: the stages take a
/// `&mut dyn KernelOps`. [`Submission`] (and a shared reference to one) implements it by the functions of this module,
/// which keep their meaning and their documentation; each method here has the parameters of the function of the same
/// name, minus the submission.
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

    /// Selects row-exact execution for the launches queued after this call: [`select_row_exact`].
    fn set_row_exact(&mut self, enabled: bool);

    /// Whether the launches queued now run row-exact: the mode the submission began with, or what
    /// [`KernelOps::set_row_exact`] selected since.
    fn row_exact(&self) -> bool;

    /// The library the launches run in, for the stages' decisions that depend on what it was built with.
    fn library(&self) -> &Arc<CudaLibrary>;

    /// [`copy_device_to_readback_shared`].
    ///
    /// # Safety
    /// As [`copy_device_to_readback_shared`].
    unsafe fn copy_device_to_readback_shared(
        &mut self,
        destination: &PinnedBuffer,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// [`copy_host_weights_to_device`].
    ///
    /// # Safety
    /// As [`copy_host_weights_to_device`].
    unsafe fn copy_host_weights_to_device(
        &mut self,
        destination: DevicePtr,
        source: &HostWeightsArena,
        offset: usize,
        bytes: usize,
    ) -> Result<(), CudaError>;

    /// [`copy_device_to_host_weights`].
    ///
    /// # Safety
    /// As [`copy_device_to_host_weights`].
    unsafe fn copy_device_to_host_weights(
        &mut self,
        destination: &HostWeightsArena,
        offset: usize,
        source: DevicePtr,
        bytes: usize,
    ) -> Result<(), CudaError>;
}

/// Implements [`KernelOps`] for a type that dereferences to a [`Submission`], each method calling the function of the
/// same name.
macro_rules! submission_ops {
    ($target:ty; $($name:ident($($argument:ident: $kind:ty),*);)*) => {
        impl KernelOps for $target {
            $(
                fn $name(&mut self, $($argument: $kind),*) -> Result<(), CudaError> {
                    let submission: &Submission<'_> = self;
                    $name(submission, $($argument),*)
                }
            )*

            fn set_row_exact(&mut self, enabled: bool) {
                let submission: &Submission<'_> = self;
                select_row_exact(submission, enabled);
            }

            fn row_exact(&self) -> bool {
                let submission: &Submission<'_> = self;
                submission.mode().row_exact
            }

            fn library(&self) -> &Arc<CudaLibrary> {
                let submission: &Submission<'_> = self;
                submission.lane().library()
            }

            unsafe fn copy_device_to_readback_shared(
                &mut self,
                destination: &PinnedBuffer,
                offset: usize,
                source: DevicePtr,
                bytes: usize,
            ) -> Result<(), CudaError> {
                let submission: &Submission<'_> = self;
                // SAFETY: the caller's contract is the function's.
                unsafe { copy_device_to_readback_shared(submission, destination, offset, source, bytes) }
            }

            unsafe fn copy_host_weights_to_device(
                &mut self,
                destination: DevicePtr,
                source: &HostWeightsArena,
                offset: usize,
                bytes: usize,
            ) -> Result<(), CudaError> {
                let submission: &Submission<'_> = self;
                // SAFETY: the caller's contract is the function's.
                unsafe { copy_host_weights_to_device(submission, destination, source, offset, bytes) }
            }

            unsafe fn copy_device_to_host_weights(
                &mut self,
                destination: &HostWeightsArena,
                offset: usize,
                source: DevicePtr,
                bytes: usize,
            ) -> Result<(), CudaError> {
                let submission: &Submission<'_> = self;
                // SAFETY: the caller's contract is the function's.
                unsafe { copy_device_to_host_weights(submission, destination, offset, source, bytes) }
            }
        }
    };
}

submission_ops! {
    Submission<'_>;
    copy_device_to_device(destination: DevicePtr, source: DevicePtr, bytes: u64);
    copy_device_to_device_2d(destination: DevicePtr, destination_pitch: u64, source: DevicePtr, source_pitch: u64, row_bytes: u64, rows: u64);
    zero_device_memory(address: DevicePtr, bytes: u64);
    embed_q3(token_ids: DevicePtr, embedding: DevicePtr, hidden: DevicePtr, token_count: u32, vocabulary_size: u32, hidden_size: u32, embedding_bytes: u64);
    embed_q3_p2e2(token_ids: DevicePtr, embedding: DevicePtr, hidden: DevicePtr, token_count: u32, vocabulary_size: u32, hidden_size: u32, embedding_bytes: u64);
    linear_launch(kernel: i32, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, words: &[u64], sizes: &[u8]);
    qwen4_launch(kernel: i32, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, words: &[u64], sizes: &[u8]);
    linear_nvfp4_bf16(input: DevicePtr, weights: DevicePtr, output: DevicePtr, rows: u32, in_features: u32, out_features: u32, weights_bytes: u64);
    linear_nvfp4_native_bf16(input: DevicePtr, weights: DevicePtr, output: DevicePtr, scratch: DevicePtr, rows: u32, in_features: u32, out_features: u32, weights_bytes: u64, scratch_bytes: u64);
    attention_qk_norm_rope_bf16(query_key: DevicePtr, query_norm: DevicePtr, key_norm: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_value_heads: u32, head_dim: u32, rotary_dim: u32, start_position: u64, position: DevicePtr, epsilon: f32, rope_theta: f64);
    attention_kv_append_nvfp4(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, rows: u32, query_width: u32, key_width: u32, start: u64, position: DevicePtr);
    attention_kv_append_bf16(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, rows: u32, query_width: u32, key_width: u32, start: u64, position: DevicePtr);
    attention_causal_nvfp4(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_heads: u32, head_dim: u32, cache_length: u32, start: u64, position: DevicePtr, scratch: DevicePtr);
    attention_causal_bf16(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_heads: u32, head_dim: u32, cache_length: u32, start: u64, position: DevicePtr, scratch: DevicePtr);
    gdn_project_control_fp32(input: DevicePtr, a_weight: DevicePtr, b_weight: DevicePtr, a_log: DevicePtr, dt_bias: DevicePtr, alpha: DevicePtr, beta: DevicePtr, rows: u32, width: u32, heads: u32);
    gdn_convolution_bf16(query_key: DevicePtr, value_z: DevicePtr, convolution_weights: DevicePtr, convolution_state: DevicePtr, output: DevicePtr, rows: u32, query_key_width: u32, value_width: u32, convolution_width: u32, kernel_size: u32);
    gdn_recurrence_bf16(convolved: DevicePtr, alpha: DevicePtr, beta: DevicePtr, recurrent_state: DevicePtr, output: DevicePtr, rows: u32, key_heads: u32, value_heads: u32, key_head_dim: u32, value_head_dim: u32, output_scale: f32);
    gdn_gated_rms_norm_bf16(recurrent: DevicePtr, value_z: DevicePtr, norm_weight: DevicePtr, output: DevicePtr, rows: u32, value_heads: u32, head_dim: u32, epsilon: f32);
    residual_rms_norm_bf16(residual: DevicePtr, delta: DevicePtr, weight: DevicePtr, hidden: DevicePtr, normalized: DevicePtr, rows: u32, width: u32, epsilon: f32);
    rms_norm_unit_offset_bf16(input: DevicePtr, weight: DevicePtr, output: DevicePtr, rows: u32, width: u32, epsilon: f32);
    swiglu_bf16(gate_up: DevicePtr, output: DevicePtr, rows: u32, intermediate_size: u32);
    residual_add_bf16(residual: DevicePtr, delta: DevicePtr, output: DevicePtr, rows: u32, width: u32);
    argmax_bf16(logits: DevicePtr, count: u32, result: DevicePtr);
    argmax_logprob_bf16(logits: DevicePtr, count: u32, result: DevicePtr);
    selected_token(key: DevicePtr, ids: DevicePtr, token: DevicePtr);
}

submission_ops! {
    &Submission<'_>;
    copy_device_to_device(destination: DevicePtr, source: DevicePtr, bytes: u64);
    copy_device_to_device_2d(destination: DevicePtr, destination_pitch: u64, source: DevicePtr, source_pitch: u64, row_bytes: u64, rows: u64);
    zero_device_memory(address: DevicePtr, bytes: u64);
    embed_q3(token_ids: DevicePtr, embedding: DevicePtr, hidden: DevicePtr, token_count: u32, vocabulary_size: u32, hidden_size: u32, embedding_bytes: u64);
    embed_q3_p2e2(token_ids: DevicePtr, embedding: DevicePtr, hidden: DevicePtr, token_count: u32, vocabulary_size: u32, hidden_size: u32, embedding_bytes: u64);
    linear_launch(kernel: i32, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, words: &[u64], sizes: &[u8]);
    qwen4_launch(kernel: i32, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, words: &[u64], sizes: &[u8]);
    linear_nvfp4_bf16(input: DevicePtr, weights: DevicePtr, output: DevicePtr, rows: u32, in_features: u32, out_features: u32, weights_bytes: u64);
    linear_nvfp4_native_bf16(input: DevicePtr, weights: DevicePtr, output: DevicePtr, scratch: DevicePtr, rows: u32, in_features: u32, out_features: u32, weights_bytes: u64, scratch_bytes: u64);
    attention_qk_norm_rope_bf16(query_key: DevicePtr, query_norm: DevicePtr, key_norm: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_value_heads: u32, head_dim: u32, rotary_dim: u32, start_position: u64, position: DevicePtr, epsilon: f32, rope_theta: f64);
    attention_kv_append_nvfp4(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, rows: u32, query_width: u32, key_width: u32, start: u64, position: DevicePtr);
    attention_kv_append_bf16(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, rows: u32, query_width: u32, key_width: u32, start: u64, position: DevicePtr);
    attention_causal_nvfp4(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_heads: u32, head_dim: u32, cache_length: u32, start: u64, position: DevicePtr, scratch: DevicePtr);
    attention_causal_bf16(query_key: DevicePtr, gate: DevicePtr, keys: DevicePtr, values: DevicePtr, output: DevicePtr, rows: u32, query_heads: u32, key_heads: u32, head_dim: u32, cache_length: u32, start: u64, position: DevicePtr, scratch: DevicePtr);
    gdn_project_control_fp32(input: DevicePtr, a_weight: DevicePtr, b_weight: DevicePtr, a_log: DevicePtr, dt_bias: DevicePtr, alpha: DevicePtr, beta: DevicePtr, rows: u32, width: u32, heads: u32);
    gdn_convolution_bf16(query_key: DevicePtr, value_z: DevicePtr, convolution_weights: DevicePtr, convolution_state: DevicePtr, output: DevicePtr, rows: u32, query_key_width: u32, value_width: u32, convolution_width: u32, kernel_size: u32);
    gdn_recurrence_bf16(convolved: DevicePtr, alpha: DevicePtr, beta: DevicePtr, recurrent_state: DevicePtr, output: DevicePtr, rows: u32, key_heads: u32, value_heads: u32, key_head_dim: u32, value_head_dim: u32, output_scale: f32);
    gdn_gated_rms_norm_bf16(recurrent: DevicePtr, value_z: DevicePtr, norm_weight: DevicePtr, output: DevicePtr, rows: u32, value_heads: u32, head_dim: u32, epsilon: f32);
    residual_rms_norm_bf16(residual: DevicePtr, delta: DevicePtr, weight: DevicePtr, hidden: DevicePtr, normalized: DevicePtr, rows: u32, width: u32, epsilon: f32);
    rms_norm_unit_offset_bf16(input: DevicePtr, weight: DevicePtr, output: DevicePtr, rows: u32, width: u32, epsilon: f32);
    swiglu_bf16(gate_up: DevicePtr, output: DevicePtr, rows: u32, intermediate_size: u32);
    residual_add_bf16(residual: DevicePtr, delta: DevicePtr, output: DevicePtr, rows: u32, width: u32);
    argmax_bf16(logits: DevicePtr, count: u32, result: DevicePtr);
    argmax_logprob_bf16(logits: DevicePtr, count: u32, result: DevicePtr);
    selected_token(key: DevicePtr, ids: DevicePtr, token: DevicePtr);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records the names of the operations it is given, and queues nothing.
    #[derive(Default)]
    struct Recorder {
        calls: Vec<&'static str>,
        row_exact: bool,
    }

    impl KernelOps for Recorder {
        fn copy_device_to_device(&mut self, _: DevicePtr, _: DevicePtr, _: u64) -> Result<(), CudaError> {
            self.calls.push("copy");
            Ok(())
        }
        fn copy_device_to_device_2d(
            &mut self,
            _: DevicePtr,
            _: u64,
            _: DevicePtr,
            _: u64,
            _: u64,
            _: u64,
        ) -> Result<(), CudaError> {
            self.calls.push("copy2d");
            Ok(())
        }
        fn zero_device_memory(&mut self, _: DevicePtr, _: u64) -> Result<(), CudaError> {
            self.calls.push("zero");
            Ok(())
        }
        fn embed_q3(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn embed_q3_p2e2(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn linear_launch(
            &mut self,
            _: i32,
            _: [u32; 3],
            _: [u32; 3],
            _: u32,
            _: &[u64],
            _: &[u8],
        ) -> Result<(), CudaError> {
            self.calls.push("linear");
            Ok(())
        }
        fn qwen4_launch(
            &mut self,
            _: i32,
            _: [u32; 3],
            _: [u32; 3],
            _: u32,
            _: &[u64],
            _: &[u8],
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn linear_nvfp4_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn linear_nvfp4_native_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: u64,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn attention_qk_norm_rope_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: DevicePtr,
            _: f32,
            _: f64,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn attention_kv_append_nvfp4(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: DevicePtr,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn attention_kv_append_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: DevicePtr,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn attention_causal_nvfp4(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: DevicePtr,
            _: DevicePtr,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn attention_causal_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u64,
            _: DevicePtr,
            _: DevicePtr,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn gdn_project_control_fp32(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn gdn_convolution_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn gdn_recurrence_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: u32,
            _: f32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn gdn_gated_rms_norm_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: u32,
            _: f32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn residual_rms_norm_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: f32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn rms_norm_unit_offset_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
            _: f32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn swiglu_bf16(&mut self, _: DevicePtr, _: DevicePtr, _: u32, _: u32) -> Result<(), CudaError> {
            Ok(())
        }
        fn residual_add_bf16(
            &mut self,
            _: DevicePtr,
            _: DevicePtr,
            _: DevicePtr,
            _: u32,
            _: u32,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn argmax_bf16(&mut self, _: DevicePtr, _: u32, _: DevicePtr) -> Result<(), CudaError> {
            Ok(())
        }
        fn argmax_logprob_bf16(&mut self, _: DevicePtr, _: u32, _: DevicePtr) -> Result<(), CudaError> {
            Ok(())
        }
        fn selected_token(&mut self, _: DevicePtr, _: DevicePtr, _: DevicePtr) -> Result<(), CudaError> {
            Ok(())
        }
        fn set_row_exact(&mut self, enabled: bool) {
            self.row_exact = enabled;
        }
        unsafe fn copy_device_to_readback_shared(
            &mut self,
            _: &PinnedBuffer,
            _: usize,
            _: DevicePtr,
            _: usize,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        unsafe fn copy_host_weights_to_device(
            &mut self,
            _: DevicePtr,
            _: &HostWeightsArena,
            _: usize,
            _: usize,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        unsafe fn copy_device_to_host_weights(
            &mut self,
            _: &HostWeightsArena,
            _: usize,
            _: DevicePtr,
            _: usize,
        ) -> Result<(), CudaError> {
            Ok(())
        }
        fn row_exact(&self) -> bool {
            self.row_exact
        }
        fn library(&self) -> &Arc<CudaLibrary> {
            unreachable!("the recorder has no library")
        }
    }

    /// Code written the way a stage is: against `&mut dyn KernelOps`.
    fn stage(ops: &mut dyn KernelOps) -> Result<(), CudaError> {
        ops.set_row_exact(true);
        ops.zero_device_memory(DevicePtr::NULL, 8)?;
        ops.copy_device_to_device(DevicePtr::NULL, DevicePtr::NULL, 8)?;
        ops.linear_launch(0, [1; 3], [1; 3], 0, &[], &[])
    }

    #[test]
    fn a_recording_implementation_serves_code_written_against_the_trait_object() {
        let mut recorder = Recorder::default();
        stage(&mut recorder).unwrap();
        assert_eq!(recorder.calls, ["zero", "copy", "linear"]);
        assert!(recorder.row_exact());
    }
}
