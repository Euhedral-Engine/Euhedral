//! One linear quantized in the trellis format (QTIP's tail-biting trellis with the `mul1` codebook), entirely on the
//! device: the damped, rotated Hessian and its block factor, the output and input scales, the global scale, LDLQ error
//! feedback through the Viterbi encode, and two rounds of closed-form refit of the scales in the Hessian metric.
//!
//! The weight is handled as `W [k, n]` in float32 (`k` inputs, `n` outputs). The work is one chain of stages on a
//! lane. A stage submits and ends in a readback; its retirement runs the next stage as a frame. The readbacks that
//! decide what runs next are the diagonal mean, the status of each Cholesky attempt, the tile energies, the coarse
//! and then the fine candidate errors, and the end. The strip loop and the refit are one submission, because nothing
//! on the host depends on what they compute until the end.
//!
//! Device memory beyond the caller's weight and Hessian is allocated when the quantization starts and handed back,
//! with the lane, in [`Finished`]: it is stream-ordered on the lane and charged to the claims the caller took, from
//! [`TrellisLane::quantize_bytes`], so no frame waits for the device to free it.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::take;
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaModule, CudaStream};
use euhedral_gpu::{GpuError, LaneStream, Stream};
use euhedral_graph::Completions;
use euhedral_inference_cuda::math::{Fill, LaneHandles, MathMode, Matrix, Operation, Product};
use euhedral_inference_cuda::{CudaError, DevicePtr};
use thiserror::Error;

use super::launch::{
    Axis, Scaling, Triangle, plan_add_diagonal, plan_axpy_floats, plan_candidate_mse, plan_candidate_tiles,
    plan_column_dots, plan_column_ratio, plan_column_rms, plan_copy_floats, plan_diagonal_io, plan_float_to_half,
    plan_gather_tiles, plan_hessian_product, plan_input_scales, plan_invert_diagonal_blocks, plan_output_scales,
    plan_right_multiply_blocks, plan_row_dots, plan_row_rms, plan_scale_floats, plan_scale_scalar, plan_strided_mean,
    plan_strip_error, plan_strip_gather_sum, plan_subtract_floats, plan_triangle, plan_weight_to_float,
};
use super::search::{
    CHOLESKY_RETRIES, COARSE, COARSE_STRIDE, FINE, SIGMA_REG, STRIP_SPAN, coarse_scales, drift, fine_scales,
    refined_scale, sample_count, sample_tiles, signs, smallest,
};
use super::{
    Rate, Reconstruction, Step, TILE_WEIGHTS, TrellisError, TrellisKernels, plan_clean_row_scales, plan_fill_floats,
    plan_from_tiles, plan_hadamard_columns, plan_hadamard_rows, plan_pack, plan_reconstruct, plan_sum_products,
    plan_tile_mean_square, plan_viterbi,
};
use crate::chain::{self, ChainError, Finished, Next, Parts, Stage, Unsettled, start};
use crate::java_random::SplittableRandom;
use crate::kernels::{Module, ModuleError, Modules};
use crate::memory::{Claims, Device, LaneMemory, Memory, MemoryError, Staged, StepBytes, pinned_bytes, reclaim};

/// The sizes the quantizer is tuned by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisSettings {
    /// The most device memory for the Viterbi history of one chain; tiles run in chunks that fit. A chunk size changes
    /// no result.
    pub viterbi_scratch_bytes: u64,
}

impl Default for TrellisSettings {
    fn default() -> Self {
        Self {
            viterbi_scratch_bytes: 256 << 20,
        }
    }
}

/// Why a quantization could not start or did not finish.
#[derive(Debug, Error)]
pub enum QuantizeError {
    /// The device or the driver failed.
    #[error("device failure: {0}")]
    Gpu(#[from] GpuError),
    /// A CUDA library call failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// A launch could not be planned or run.
    #[error(transparent)]
    Trellis(#[from] TrellisError),
    /// The trellis module did not compile.
    #[error(transparent)]
    Module(#[from] ModuleError),
    /// The request does not fit what the quantizer works at.
    #[error("{0}")]
    Request(String),
    /// The Hessian stayed indefinite through every retry of the factorization.
    #[error("Cholesky factorization failed after {retries} retries")]
    NotPositiveDefinite { retries: u32 },
    /// The stream lends no driver handle, so the math libraries cannot be called on it.
    #[error("the stream lends no driver handle")]
    NoStream,
    /// The chain broke its own rules.
    #[error("{0}")]
    Broken(String),
}

impl From<MemoryError> for QuantizeError {
    fn from(error: MemoryError) -> Self {
        match error {
            MemoryError::Cuda(error) => QuantizeError::Cuda(error),
            MemoryError::Refused(text) => QuantizeError::Request(text),
        }
    }
}

impl ChainError for QuantizeError {
    fn broken(message: &str) -> Self {
        QuantizeError::Broken(message.to_owned())
    }
}

/// What to quantize.
#[derive(Clone, Copy, Debug)]
pub struct Request {
    /// The BF16 `nn.Linear` weight `[n, k]` on the device.
    pub weight: u64,
    /// Output features, a positive multiple of 128.
    pub n: u32,
    /// Input features, a positive multiple of 128.
    pub k: u32,
    /// The FP32 Hessian `[k, k]` on the device, symmetric, normalised by `count` and undamped; it is left as it is. The
    /// weight is quantized without one when this is `None`, when `count` is zero, or when the mean diagonal is not
    /// finite or is below 1e-20.
    pub hessian: Option<u64>,
    /// The number of calibration rows the Hessian was summed over.
    pub count: u64,
    /// The rate: 2 to 6, or 2.5 or 3.5.
    pub bits: f64,
    /// Seeds the sign vectors.
    pub seed: u64,
    /// Whether the per-output-channel scales are applied.
    pub apply_out_scales: bool,
}

/// The result of one quantization.
#[derive(Debug)]
pub struct Quantized {
    /// The packed trellis `[k / 16, n / 16, 16 * bits]` words.
    pub trellis: Vec<u16>,
    /// The fp16 bits of the input-side scales (the signs and the global scale folded in), `[k]`.
    pub suh: Vec<u16>,
    /// The fp16 bits of the output-side scales, `[n]`.
    pub svh: Vec<u16>,
    /// The BF16 `[n, k]` weight the tensors reconstruct to, on the device, owned by the caller. Dropping it gives it
    /// to the lane's queue, which frees it in stream order.
    pub decoded: Device,
    /// `tr(E H Eᵀ) / tr(W H Wᵀ)`, the error in the Hessian metric (the mean squared error of the rotated weight when
    /// there was no Hessian).
    pub proxy_error: f64,
    /// The global scale that was folded into `suh`.
    pub global_scale: f32,
    /// How many refit rounds failed to factor the input-side system and so left the input scales alone.
    pub refit_failures: u32,
    /// How many more attempts the Cholesky factorization of the damped Hessian took, each with more damping.
    pub cholesky_retries: u32,
}

/// A lane's parts apart from its stream: what a stage uses besides the stream.
pub struct LaneParts {
    handles: LaneHandles,
    kernels: Arc<TrellisKernels<CudaModule>>,
    memory: LaneMemory,
    completions: Arc<Completions>,
    settings: TrellisSettings,
    /// The state of chains whose retirement could not be armed, with work possibly still queued on the stream: kept
    /// until a later retirement is confirmed.
    unsettled: Unsettled,
    #[cfg(feature = "test-hooks")]
    shutdown_probe: Option<Box<dyn Fn() + Send>>,
}

impl Debug for LaneParts {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("LaneParts").finish_non_exhaustive()
    }
}

impl Parts for LaneParts {
    fn completions(&self) -> &Arc<Completions> {
        &self.completions
    }

    fn unsettled(&mut self) -> &mut Unsettled {
        &mut self.unsettled
    }
}

/// A lane's stream with the math handles and the kernels the quantizer needs. A quantization takes the lane by value
/// and gives it back in [`Finished`] (or in [`Refused`], when it could not start).
#[derive(Debug)]
pub struct TrellisLane {
    stream: CudaStream,
    parts: LaneParts,
}

/// A quantization that did not start: the lane comes back unused.
#[derive(Debug)]
pub struct Refused {
    pub lane: TrellisLane,
    pub error: QuantizeError,
}

type Enq<'a> = <CudaStream as Stream>::Enqueue<'a>;
type Move = Next<LaneParts, State, Quantized, QuantizeError>;
type Chain = Stage<LaneParts, State, Quantized, QuantizeError>;

impl TrellisLane {
    /// The trellis kernels, compiled once (or found in the module cache) and shared by every lane.
    pub fn kernels(modules: &mut Modules<CudaModule>) -> Result<Arc<TrellisKernels<CudaModule>>, QuantizeError> {
        let compiled = modules.load(Module::Trellis)?;
        Ok(Arc::new(TrellisKernels::new(&compiled)?))
    }

    /// A lane from its stream, its handles (cuBLAS products run in full FP32), the memory it works with and the
    /// completions that deliver its retirements.
    pub fn new(
        stream: CudaStream,
        mut handles: LaneHandles,
        kernels: Arc<TrellisKernels<CudaModule>>,
        memory: LaneMemory,
        completions: Arc<Completions>,
        settings: TrellisSettings,
    ) -> Result<TrellisLane, QuantizeError> {
        handles.set_math_mode(MathMode::Default)?;
        Ok(TrellisLane {
            stream,
            parts: LaneParts {
                handles,
                kernels,
                memory,
                completions,
                settings,
                unsettled: Unsettled::default(),
                #[cfg(feature = "test-hooks")]
                shutdown_probe: None,
            },
        })
    }

    /// Takes the lane apart for shutdown: the stream and the handles are destroyed by their owner. The device buffers
    /// given up are freed in stream order first, and the error of that, if any, comes back with the parts. A
    /// [`Quantized::decoded`] must not outlive its lane: when it is dropped after this, its memory is freed with the
    /// plain `cudaFree`.
    ///
    /// State parked by a failed chain is released only after the stream has run its queued work. This is a shutdown
    /// call, so it waits for the stream here; no frame calls it.
    pub fn into_parts(mut self) -> (CudaStream, LaneHandles, Result<u64, QuantizeError>) {
        #[cfg(feature = "test-hooks")]
        let probe = self.parts.shutdown_probe.take();
        self.parts.unsettled.settle(&mut self.stream, || {
            #[cfg(feature = "test-hooks")]
            if let Some(probe) = &probe {
                probe();
            }
        });
        let reclaimed = self.reclaim();
        (self.stream, self.parts.handles, reclaimed)
    }

    /// Arms a retirement on the stream that nobody collects, so the next chain's own cannot be armed and fails after it
    /// enqueued its work. For tests of that path.
    #[cfg(feature = "test-hooks")]
    pub fn block_arming(&mut self) -> Result<(), GpuError> {
        chain::block_arming(&mut self.stream)
    }

    /// Collects the retirement [`block_arming`](Self::block_arming) armed, once the stream has run up to it. Returns
    /// false when it has not arrived yet.
    #[cfg(feature = "test-hooks")]
    pub fn unblock_arming(&mut self) -> bool {
        chain::unblock_arming(&mut self.stream)
    }

    /// Sets a call made by [`into_parts`](Self::into_parts) just before it waits for the stream, when a failed chain's
    /// state is parked.
    #[cfg(feature = "test-hooks")]
    pub fn set_shutdown_probe(&mut self, probe: Box<dyn Fn() + Send>) {
        self.parts.shutdown_probe = Some(probe);
    }

    /// How many failed chains' states the lane still holds because work of theirs may be queued on the stream.
    pub fn unsettled(&self) -> usize {
        self.parts.unsettled.len()
    }

    /// Frees, in stream order on this lane, the device buffers given up since the last call, and returns how many. A
    /// step does it before it allocates; call it after the last step to free what that left.
    pub fn reclaim(&mut self) -> Result<u64, QuantizeError> {
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            Ok(reclaim(&memory.released, &memory.library, queue)?)
        })
    }

    /// The memory [`quantize`](Self::quantize) takes for a weight of `n` outputs by `k` inputs at `bits`, to be claimed
    /// before it, with a Hessian when `calibrated`. It asks cuSOLVER for the workspace the factorizations need, which
    /// enqueues nothing.
    pub fn quantize_bytes(&mut self, n: u32, k: u32, bits: f64, calibrated: bool) -> Result<StepBytes, QuantizeError> {
        let rate = Rate::new(bits)?;
        let shape = Shape::new(n, k, rate, self.parts.settings)?;
        let floats = if calibrated { self.workspace_floats(k)? } else { 1 };
        Ok(Sizes::of(&shape, calibrated).step_bytes(floats))
    }

    /// The floats of workspace the factorizations of a `k` by `k` matrix ask for.
    fn workspace_floats(&mut self, k: u32) -> Result<i32, QuantizeError> {
        let n = i32::try_from(k).map_err(|_| QuantizeError::Request(format!("{k} does not fit a 32-bit count")))?;
        let handles = &mut self.parts.handles;
        let floats = submit(&mut self.stream, |queue| {
            let raw = queue.raw_stream().ok_or(QuantizeError::NoStream)?;
            // SAFETY: the stream is the lane's own; the size query reads no memory, so no matrix is given.
            Ok(unsafe { handles.solver_workspace(&raw, false, Fill::Upper, n, DevicePtr::NULL) }?)
        })?;
        Ok(floats.max(1))
    }

    /// Quantizes the weight of `request`. `done` is called, from the frame that ends the chain, with the lane back, the
    /// memory the chain used (to drop off the lattice) and the result. The request's weight and Hessian stay live and
    /// unchanged until then.
    pub fn quantize(
        mut self,
        request: &Request,
        claims: Claims,
        done: impl FnOnce(Finished<TrellisLane, Quantized, QuantizeError>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        match self.prepare(request, claims) {
            Ok(state) => {
                let TrellisLane { stream, parts } = self;
                start(
                    stream,
                    parts,
                    state,
                    |stream, parts| TrellisLane { stream, parts },
                    Box::new(done),
                    Box::new(begin),
                );
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare(&mut self, request: &Request, claims: Claims) -> Result<State, QuantizeError> {
        let Request {
            weight,
            n,
            k,
            hessian,
            count,
            bits,
            seed,
            apply_out_scales,
        } = *request;
        if weight == 0 || hessian == Some(0) {
            return Err(QuantizeError::Request(
                "the weight and the Hessian are device addresses".into(),
            ));
        }
        let rate = Rate::new(bits)?;
        let shape = Shape::new(n, k, rate, self.parts.settings)?;
        let hessian = hessian.filter(|_| count > 0);
        let sizes = Sizes::of(&shape, hessian.is_some());
        let workspace_floats = if hessian.is_some() { self.workspace_floats(k)? } else { 1 };

        let mut random = SplittableRandom::new(seed);
        let su_values = signs(&mut random, k as usize);
        let sv_values = signs(&mut random, n as usize);
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            reclaim(&memory.released, &memory.library, queue)?;
            let mut memory = Memory::new(queue, &memory.library, &memory.released, &memory.pinned, claims);
            let memory = &mut memory;
            Ok(State {
                rate,
                shape,
                weight,
                hessian,
                apply_out_scales,
                bits,
                diagonal_mean: 0.0,
                attempt: 0,
                fine: [0.0; FINE],
                global_scale: 1.0,
                workspace_floats,
                su_signs: Staged::of_f32(memory, &su_values)?,
                sv_signs: Staged::of_f32(memory, &sv_values)?,
                small: staged(memory, 64)?,
                original: device(memory, sizes.matrix)?,
                w: device(memory, sizes.matrix)?,
                decoded: device(memory, sizes.matrix)?,
                cache: device(memory, sizes.matrix)?,
                encoded: device(memory, sizes.encoded)?,
                words: staged(memory, sizes.words)?,
                su: device(memory, sizes.su)?,
                sv: device(memory, sizes.sv)?,
                suh: staged(memory, sizes.suh)?,
                svh: staged(memory, sizes.svh)?,
                rms: device(memory, sizes.rms)?,
                divisor: device(memory, sizes.divisor)?,
                energy: staged(memory, sizes.energy)?,
                tile_k: staged(memory, sizes.samples_index)?,
                tile_n: staged(memory, sizes.samples_index)?,
                sampled: device(memory, sizes.sampled)?,
                candidate_scales: staged(memory, sizes.candidates)?,
                candidate_errors: staged(memory, sizes.candidates)?,
                scaled: device(memory, sizes.candidate_values)?,
                quantized: device(memory, sizes.candidate_values)?,
                indices: device(memory, sizes.candidate_indices)?,
                viterbi: device(memory, sizes.viterbi)?,
                strip: device(memory, sizes.strip)?,
                rounded: device(memory, sizes.strip)?,
                error: device(memory, sizes.error)?,
                diagonal: device(memory, sizes.diagonal)?,
                inverse: device(memory, sizes.inverse)?,
                workspace: device(memory, 4 * u64::from(workspace_floats.unsigned_abs()))?,
                rebuilt: Some(device(memory, sizes.rebuilt)?),
                factor: device(memory, sizes.factor)?,
            })
        })
    }
}

/// What a shape fixes: the sampled tiles, the candidate rounds and how the Viterbi scratch is cut.
#[derive(Clone, Copy, Debug)]
struct Shape {
    n: u32,
    k: u32,
    rate: Rate,
    /// Tiles of the sampled scale search.
    samples: usize,
    /// The most tiles one candidate round rounds (the larger of the coarse and the fine round).
    candidate_tiles: usize,
    /// Tiles per Viterbi chunk.
    scratch_tiles: u32,
}

impl Shape {
    fn new(n: u32, k: u32, rate: Rate, settings: TrellisSettings) -> Result<Shape, QuantizeError> {
        if n == 0 || k == 0 || !n.is_multiple_of(128) || !k.is_multiple_of(128) {
            return Err(QuantizeError::Request(format!(
                "shape [{n}, {k}] is not a positive multiple of 128"
            )));
        }
        let samples = sample_count((k / 16) as usize, (n / 16) as usize);
        let candidate_tiles = (COARSE * samples.div_ceil(COARSE_STRIDE as usize)).max(FINE * samples);
        let fits = rate.scratch_tiles(settings.viterbi_scratch_bytes)?;
        let wanted = u32::try_from(((n / 16) as usize).max(candidate_tiles)).unwrap_or(u32::MAX);
        Ok(Shape {
            n,
            k,
            rate,
            samples,
            candidate_tiles,
            scratch_tiles: wanted.min(fits),
        })
    }

    fn kn(&self) -> u64 {
        u64::from(self.k) * u64::from(self.n)
    }

    fn tiles(&self) -> usize {
        (self.k / 16) as usize * (self.n / 16) as usize
    }
}

/// The byte sizes of the buffers a quantization holds.
#[derive(Clone, Copy, Debug)]
struct Sizes {
    /// A float matrix `[k, n]`.
    matrix: u64,
    factor: u64,
    encoded: u64,
    words: u64,
    su: u64,
    sv: u64,
    suh: u64,
    svh: u64,
    rms: u64,
    divisor: u64,
    energy: u64,
    samples_index: u64,
    sampled: u64,
    candidates: u64,
    candidate_values: u64,
    candidate_indices: u64,
    viterbi: u64,
    strip: u64,
    error: u64,
    diagonal: u64,
    inverse: u64,
    rebuilt: u64,
    su_signs: u64,
    sv_signs: u64,
}

impl Sizes {
    fn of(shape: &Shape, calibrated: bool) -> Sizes {
        let (n, k) = (u64::from(shape.n), u64::from(shape.k));
        let when = |bytes: u64| if calibrated { bytes } else { 0 };
        let tile = TILE_WEIGHTS as u64;
        Sizes {
            matrix: 4 * shape.kn(),
            factor: when(4 * k * k),
            encoded: 2 * shape.kn(),
            words: 2 * (shape.tiles() * shape.rate.words_per_tile()) as u64,
            su: 4 * k,
            sv: 4 * n,
            suh: 2 * k,
            svh: 2 * n,
            rms: 4 * n.max(k),
            divisor: 4 * n,
            energy: 4 * shape.tiles() as u64,
            samples_index: 4 * shape.samples as u64,
            sampled: 4 * tile * shape.samples as u64,
            candidates: 4 * COARSE as u64,
            candidate_values: 4 * tile * shape.candidate_tiles as u64,
            candidate_indices: 2 * tile * shape.candidate_tiles as u64,
            viterbi: shape.rate.scratch_bytes(u64::from(shape.scratch_tiles)),
            strip: 4 * 16 * n,
            error: when(4 * u64::from(STRIP_SPAN) * n),
            diagonal: when(4 * k),
            inverse: when(4 * 256 * (k / 16)),
            rebuilt: 2 * shape.kn(),
            su_signs: 4 * k,
            sv_signs: 4 * n,
        }
    }

    /// The device buffers, each a plain allocation, and the staged ones (a pinned buffer with a device twin), as the
    /// quantizer allocates them.
    fn buffers(&self) -> (Vec<u64>, Vec<u64>) {
        let plain = vec![
            self.matrix,
            self.matrix,
            self.matrix,
            self.matrix,
            self.encoded,
            self.su,
            self.sv,
            self.rms,
            self.divisor,
            self.sampled,
            self.candidate_values,
            self.candidate_values,
            self.candidate_indices,
            self.viterbi,
            self.strip,
            self.strip,
            self.error,
            self.diagonal,
            self.inverse,
            self.rebuilt,
            self.factor,
        ];
        let staged = vec![
            64,
            self.words,
            self.suh,
            self.svh,
            self.energy,
            self.samples_index,
            self.samples_index,
            self.candidates,
            self.candidates,
            self.su_signs,
            self.sv_signs,
        ];
        (plain, staged)
    }

    /// What a quantization charges: every device allocation (a zero size takes one byte), the cuSOLVER workspace of
    /// `workspace_floats`, and the pinned twin of each staged buffer at its pool size class.
    fn step_bytes(&self, workspace_floats: i32) -> StepBytes {
        let (plain, staged) = self.buffers();
        let workspace = 4 * u64::from(workspace_floats.unsigned_abs());
        let device = plain
            .iter()
            .chain(&staged)
            .copied()
            .chain([workspace])
            .map(|bytes| bytes.max(1))
            .fold(0, u64::saturating_add);
        StepBytes {
            device,
            pinned: pinned_bytes(staged),
        }
    }
}

// ---- memory ---------------------------------------------------------------------------------------------------

fn bytes_of(bytes: u64) -> Result<usize, QuantizeError> {
    usize::try_from(bytes.max(1)).map_err(|_| QuantizeError::Request(format!("{bytes} bytes do not fit the host")))
}

/// The allocations of a quantization's preparation, made on the lane's stream.
type Mem<'a, 'q> = Memory<'a, Enq<'q>>;

fn device(memory: &mut Mem<'_, '_>, bytes: u64) -> Result<Device, QuantizeError> {
    Ok(memory.device(bytes_of(bytes)?)?)
}

fn staged(memory: &mut Mem<'_, '_>, bytes: u64) -> Result<Staged, QuantizeError> {
    Ok(Staged::new(memory, bytes_of(bytes)?)?)
}

/// Runs `body` as one submission, keeping this module's error type.
fn submit<R>(
    stream: &mut CudaStream,
    body: impl for<'a> FnOnce(&mut Enq<'a>) -> Result<R, QuantizeError>,
) -> Result<R, QuantizeError> {
    chain::submit(stream, body)
}

/// Planned launches waiting to go onto the queue in order, so the library calls between them keep their place.
struct Work<'k> {
    kernels: &'k TrellisKernels<CudaModule>,
    steps: Vec<Step>,
}

impl<'k> Work<'k> {
    fn new(kernels: &'k TrellisKernels<CudaModule>) -> Work<'k> {
        Work {
            kernels,
            steps: Vec::new(),
        }
    }

    fn add(&mut self, step: Result<Step, TrellisError>) -> Result<(), QuantizeError> {
        self.steps.push(step?);
        Ok(())
    }

    fn all(&mut self, steps: Result<Vec<Step>, TrellisError>) -> Result<(), QuantizeError> {
        self.steps.extend(steps?);
        Ok(())
    }

    /// Puts what is planned onto the queue.
    fn flush(&mut self, queue: &mut Enq<'_>) -> Result<(), QuantizeError> {
        let steps = take(&mut self.steps);
        // SAFETY: every address in a step names a buffer the chain's state owns (or the caller's weight and Hessian,
        // live until the result is delivered), and nothing else uses them until the retirement is confirmed.
        unsafe { self.kernels.enqueue(queue, &steps) }?;
        Ok(())
    }
}

/// The state the stages share: the request, what the host decided so far and every buffer.
struct State {
    rate: Rate,
    shape: Shape,
    weight: u64,
    hessian: Option<u64>,
    apply_out_scales: bool,
    bits: f64,
    diagonal_mean: f64,
    attempt: u32,
    fine: [f64; FINE],
    global_scale: f32,
    workspace_floats: i32,
    su_signs: Staged,
    sv_signs: Staged,
    /// Floats 0 (mean), ints at bytes 8 and 12 (statuses), doubles at 16, 24, 32, an int at 48 (refit failures).
    small: Staged,
    original: Device,
    /// The rotated and scaled weight.
    w: Device,
    decoded: Device,
    /// The feedback of the rounding errors; the refit's product, and the error of the uncalibrated path.
    cache: Device,
    encoded: Device,
    words: Staged,
    su: Device,
    sv: Device,
    suh: Staged,
    svh: Staged,
    rms: Device,
    divisor: Device,
    energy: Staged,
    tile_k: Staged,
    tile_n: Staged,
    sampled: Device,
    candidate_scales: Staged,
    candidate_errors: Staged,
    scaled: Device,
    quantized: Device,
    indices: Device,
    viterbi: Device,
    strip: Device,
    rounded: Device,
    error: Device,
    diagonal: Device,
    inverse: Device,
    workspace: Device,
    rebuilt: Option<Device>,
    /// The block factor of the Hessian; the refit's system afterwards.
    factor: Device,
}

impl State {
    fn dims(&self) -> (u32, u32) {
        (self.shape.n, self.shape.k)
    }

    fn kn(&self) -> u64 {
        self.shape.kn()
    }
}

/// A product in the column-major call the library takes: `m` by `n` by `k`, scaled by one.
fn product(a: Operation, b: Operation, m: u32, n: u32, k: u32, beta: f32) -> Product {
    Product {
        a_operation: a,
        b_operation: b,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        alpha: 1.0,
        beta,
    }
}

fn matrix(address: u64, leading: u32) -> Matrix {
    // SAFETY: the address names a live device allocation of the chain's state or the caller's; the library call that
    // takes the matrix checks nothing, so the products keep to the sizes the allocations have.
    let pointer = unsafe { DevicePtr::from_raw(address) };
    Matrix {
        pointer,
        leading: leading as i32,
    }
}

/// Runs one product on the queue after what is planned.
fn gemm(
    work: &mut Work<'_>,
    handles: &mut LaneHandles,
    queue: &mut Enq<'_>,
    shape: &Product,
    operands: [Matrix; 3],
) -> Result<(), QuantizeError> {
    work.flush(queue)?;
    let raw = queue.raw_stream().ok_or(QuantizeError::NoStream)?;
    let [a, b, c] = operands;
    // SAFETY: the stream is the lane's own; the operands are live allocations of the chain's state (or the caller's
    // Hessian) of the sizes the product and leading dimensions imply, kept until the retirement is confirmed.
    unsafe { handles.gemm_fp32(&raw, shape, a, b, c) }?;
    Ok(())
}

/// `out [k, n] = H [k, k] x [k, n]`, row-major.
fn gemm_h(
    work: &mut Work<'_>,
    handles: &mut LaneHandles,
    queue: &mut Enq<'_>,
    hessian: u64,
    (n, k): (u32, u32),
    x: u64,
    out: u64,
) -> Result<(), QuantizeError> {
    let shape = product(Operation::None, Operation::None, n, k, k, 0.0);
    gemm(
        work,
        handles,
        queue,
        &shape,
        [matrix(x, n), matrix(hessian, k), matrix(out, n)],
    )
}

/// One Cholesky attempt on the `k` by `k` matrix at `target`, its status into the small buffer at byte 8.
fn factor_attempt(
    work: &mut Work<'_>,
    handles: &mut LaneHandles,
    queue: &mut Enq<'_>,
    state: &State,
    target: u64,
) -> Result<(), QuantizeError> {
    work.flush(queue)?;
    let raw = queue.raw_stream().ok_or(QuantizeError::NoStream)?;
    let info = state.small.device.ptr().offset(8);
    // SAFETY: the stream is the lane's own; the matrix is a live k by k allocation, the workspace holds the floats the
    // library named and the status is one int of the small buffer; all are kept until the retirement is confirmed.
    // The matrix is row-major, so the library's upper fill is the row-major lower triangle.
    unsafe {
        handles.cholesky(
            &raw,
            false,
            Fill::Upper,
            state.shape.k as i32,
            DevicePtr::from_raw(target),
            state.workspace.ptr(),
            state.workspace_floats,
            info,
        )
    }?;
    Ok(())
}

// ---- the stages -----------------------------------------------------------------------------------------------

fn wait(
    next: impl FnOnce(&mut CudaStream, &mut LaneParts, &mut State) -> Result<Move, QuantizeError> + Send + 'static,
) -> Move {
    let stage: Chain = Box::new(next);
    Next::Wait(stage)
}

/// Uploads the signs, rotates the weight to float and, with a Hessian, reads back the mean of its diagonal.
fn begin(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let (n, k) = state.dims();
    let kernels = Arc::clone(&parts.kernels);
    let hessian = state.hessian;
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        state.su_signs.upload(queue)?;
        state.sv_signs.upload(queue)?;
        work.add(plan_weight_to_float(state.weight, n, k, state.original.addr()))?;
        if let Some(hessian) = hessian {
            work.add(plan_strided_mean(hessian, k, u64::from(k) + 1, state.small.addr()))?;
            work.flush(queue)?;
            Ok(state.small.download(queue)?)
        } else {
            work.flush(queue)
        }
    })?;
    match hessian {
        Some(_) => Ok(wait(after_mean)),
        None => regularize(stream, parts, state),
    }
}

/// Reads the mean; a Hessian with none worth damping by is dropped. Otherwise the rotated, damped Hessian and its
/// first factorization attempt.
fn after_mean(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let mean = state.small.float_at(0);
    state.diagonal_mean = f64::from(mean);
    if !mean.is_finite() || state.diagonal_mean < 1e-20 {
        state.hessian = None;
        return regularize(stream, parts, state);
    }
    let Some(hessian) = state.hessian else {
        return Err(QuantizeError::Broken("a mean was read without a Hessian".into()));
    };
    let (_, k) = state.dims();
    let kernels = Arc::clone(&parts.kernels);
    let handles = &mut parts.handles;
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        let h = state.factor.addr();
        let (signs, diagonal) = (state.su_signs.addr(), state.diagonal.addr());
        work.add(plan_copy_floats(hessian, h, u64::from(k) * u64::from(k)))?;
        work.add(plan_add_diagonal(h, k, state.small.addr(), SIGMA_REG as f32))?;
        work.add(plan_scale_floats(h, k, k, signs, Axis::Columns, Scaling::Multiply))?;
        work.all(plan_hadamard_columns(h, k, k))?;
        work.add(plan_scale_floats(h, k, k, signs, Axis::Rows, Scaling::Multiply))?;
        work.all(plan_hadamard_rows(h, k, k))?;
        // the upper triangle (what the factorization leaves alone) mirrors the lower, to restore it on a retry
        work.add(plan_triangle(h, k, Triangle::LowerToUpper))?;
        work.add(plan_diagonal_io(h, k, diagonal, true))?;
        factor_attempt(&mut work, handles, queue, state, h)?;
        Ok(state.small.download(queue)?)
    })?;
    Ok(wait(after_factor))
}

/// Reads the status: a clean factor goes on to its block form; otherwise more damping and another attempt.
fn after_factor(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let (_, k) = state.dims();
    let kernels = Arc::clone(&parts.kernels);
    let handles = &mut parts.handles;
    let h = state.factor.addr();
    let status = state.small.int_at(8);
    if status != 0 && state.attempt == CHOLESKY_RETRIES {
        return Err(QuantizeError::NotPositiveDefinite {
            retries: CHOLESKY_RETRIES,
        });
    }
    if status != 0 {
        state.attempt += 1;
        submit(stream, |queue| {
            let mut work = Work::new(&kernels);
            let diagonal = state.diagonal.addr();
            work.add(plan_triangle(h, k, Triangle::UpperToLower))?;
            work.add(plan_diagonal_io(h, k, diagonal, false))?;
            work.add(plan_strided_mean(h, k, u64::from(k) + 1, state.small.addr()))?;
            work.add(plan_add_diagonal(h, k, state.small.addr(), (2.0 * SIGMA_REG) as f32))?;
            work.add(plan_diagonal_io(h, k, diagonal, true))?;
            factor_attempt(&mut work, handles, queue, state, h)?;
            Ok(state.small.download(queue)?)
        })?;
        return Ok(wait(after_factor));
    }
    // the block factor: each 16-column block times the inverse of its diagonal block, the diagonal blocks zeroed
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        work.add(plan_triangle(h, k, Triangle::ClearUpper))?;
        work.add(plan_invert_diagonal_blocks(h, k, state.inverse.addr()))?;
        work.add(plan_right_multiply_blocks(h, k, state.inverse.addr()))?;
        work.flush(queue)
    })?;
    regularize(stream, parts, state)
}

/// Rotates the weight into the distribution the trellis expects and fills `su` and `sv` (signs folded in): output
/// scales (column RMS, mean one), `W / sv`, Hadamard along n, input scales (row RMS), `W / su`, Hadamard along k.
/// Then the tile energies come back, to choose the tiles the global scale is searched on.
fn regularize(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let (n, k) = state.dims();
    let kernels = Arc::clone(&parts.kernels);
    let kn = state.kn();
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        let (w, rms, small) = (state.w.addr(), state.rms.addr(), state.small.addr());
        work.add(plan_copy_floats(state.original.addr(), w, kn))?;
        work.add(plan_column_rms(w, k, n, rms))?;
        work.add(plan_strided_mean(rms, n, 1, small))?;
        work.add(plan_output_scales(
            rms,
            small,
            n,
            state.sv_signs.addr(),
            state.apply_out_scales,
            state.divisor.addr(),
            state.sv.addr(),
        ))?;
        work.add(plan_scale_floats(
            w,
            k,
            n,
            state.divisor.addr(),
            Axis::Columns,
            Scaling::Divide,
        ))?;
        work.all(plan_hadamard_columns(w, k, n))?;
        work.add(plan_row_rms(w, k, n, rms))?;
        work.add(plan_input_scales(rms, state.su_signs.addr(), k, state.su.addr()))?;
        work.add(plan_scale_floats(w, k, n, state.su.addr(), Axis::Rows, Scaling::Divide))?;
        work.all(plan_hadamard_rows(w, k, n))?;
        work.all(plan_tile_mean_square(w, k, n, state.energy.addr()))?;
        work.flush(queue)?;
        Ok(state.energy.download(queue)?)
    })?;
    Ok(wait(after_energy))
}

/// Chooses the sampled tiles, gathers them and rounds the coarse candidates.
fn after_energy(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let (n, k) = state.dims();
    let (tiles_k, tiles_n) = ((k / 16) as usize, (n / 16) as usize);
    let energy = state.energy.floats(tiles_k * tiles_n);
    let samples = sample_tiles(tiles_k, tiles_n, &energy);
    if samples.count() != state.shape.samples {
        return Err(QuantizeError::Broken(
            "the sampled tiles are not the planned count".into(),
        ));
    }
    state.tile_k.fill_i32(&samples.tile_k);
    state.tile_n.fill_i32(&samples.tile_n);
    let kernels = Arc::clone(&parts.kernels);
    let count = state.shape.samples as u32;
    let scales = coarse_scales().map(|scale| scale as f32);
    state.candidate_scales.fill_f32(&scales);
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        state.tile_k.upload(queue)?;
        state.tile_n.upload(queue)?;
        work.add(plan_gather_tiles(
            state.w.addr(),
            n,
            state.tile_k.addr(),
            state.tile_n.addr(),
            drift(state.bits),
            count,
            state.sampled.addr(),
        ))?;
        round(&mut work, queue, state, COARSE_STRIDE, COARSE as u32)
    })?;
    Ok(wait(after_coarse))
}

/// One round of candidate scales: every `stride`th sampled tile at each scale, rounded, and the mean squared error of
/// each scale read back.
fn round(
    work: &mut Work<'_>,
    queue: &mut Enq<'_>,
    state: &mut State,
    stride: u32,
    candidates: u32,
) -> Result<(), QuantizeError> {
    let samples = state.shape.samples as u32;
    let per = samples.div_ceil(stride);
    let count = candidates * per;
    let scales = state.candidate_scales.addr();
    state.candidate_scales.upload(queue)?;
    work.add(plan_candidate_tiles(
        state.sampled.addr(),
        stride,
        per,
        scales,
        TILE_WEIGHTS as u64 * u64::from(count),
        state.scaled.addr(),
    ))?;
    work.all(plan_viterbi(
        state.rate,
        state.scaled.addr(),
        count,
        state.indices.addr(),
        state.quantized.addr(),
        state.viterbi.addr(),
        state.shape.scratch_tiles,
    ))?;
    work.add(plan_candidate_mse(
        state.quantized.addr(),
        state.sampled.addr(),
        stride,
        per,
        scales,
        candidates,
        state.candidate_errors.addr(),
    ))?;
    work.flush(queue)?;
    Ok(state.candidate_errors.download(queue)?)
}

fn after_coarse(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let errors = state.candidate_errors.floats(COARSE);
    let coarse = coarse_scales();
    state.fine = fine_scales(coarse[smallest(&errors)]);
    let scales = state.fine.map(|scale| scale as f32);
    state.candidate_scales.fill_f32(&scales);
    let kernels = Arc::clone(&parts.kernels);
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        round(&mut work, queue, state, 1, FINE as u32)
    })?;
    Ok(wait(after_fine))
}

/// The global scale is set; LDLQ rounds the strips, then the scales are refitted and the tensors reconstructed.
fn after_fine(stream: &mut CudaStream, parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let errors = state.candidate_errors.floats(FINE);
    state.global_scale = refined_scale(&state.fine, &errors);
    let (n, k) = state.dims();
    let kn = state.kn();
    let tiles = (k / 16) * (n / 16);
    let tiles_n = n / 16;
    let global = state.global_scale;
    let LaneParts { handles, kernels, .. } = parts;
    let kernels = Arc::clone(kernels);
    submit(stream, |queue| {
        let mut work = Work::new(&kernels);
        let (w, decoded, cache) = (state.w.addr(), state.decoded.addr(), state.cache.addr());
        work.add(plan_scale_scalar(w, kn, global, Scaling::Multiply))?;
        work.add(plan_scale_scalar(
            state.su.addr(),
            u64::from(k),
            global,
            Scaling::Divide,
        ))?;
        let calibrated = state.hessian.is_some();
        if calibrated {
            work.all(plan_fill_floats(cache, kn, 0.0))?;
        }
        let feedback = if calibrated { cache } else { 0 };
        // LDLQ over strips of 16 rows from the last up, in spans of 128 rows: a strip's tiles are its weights plus the
        // feedback of the error of every later row (earlier spans through the cache, its own span directly).
        let mut j = k;
        while j > 0 {
            let i = j - STRIP_SPAN;
            let mut bj = STRIP_SPAN;
            while bj > 0 {
                let bi = bj - 16;
                let row = i + bi;
                let later = STRIP_SPAN - bj;
                if calibrated && later > 0 {
                    let shape = product(Operation::None, Operation::Transpose, n, 16, later, 1.0);
                    let at = |base: DevicePtr, elements: u64| base.offset(4 * elements);
                    let a = at(state.error.ptr(), u64::from(bj) * u64::from(n));
                    let b = at(
                        state.factor.ptr(),
                        (u64::from(i) + u64::from(bj)) * u64::from(k) + u64::from(row),
                    );
                    let c = at(state.cache.ptr(), u64::from(row) * u64::from(n));
                    let operands = [
                        Matrix {
                            pointer: a,
                            leading: n as i32,
                        },
                        Matrix {
                            pointer: b,
                            leading: k as i32,
                        },
                        Matrix {
                            pointer: c,
                            leading: n as i32,
                        },
                    ];
                    gemm(&mut work, handles, queue, &shape, operands)?;
                }
                work.add(plan_strip_gather_sum(w, feedback, n, row, state.strip.addr()))?;
                work.all(plan_viterbi(
                    state.rate,
                    state.strip.addr(),
                    tiles_n,
                    state.encoded.addr() + 2 * u64::from(row / 16) * u64::from(tiles_n) * 256,
                    state.rounded.addr(),
                    state.viterbi.addr(),
                    state.shape.scratch_tiles,
                ))?;
                work.all(plan_from_tiles(state.rounded.addr(), k, n, row, decoded))?;
                if calibrated {
                    work.add(plan_strip_error(w, decoded, n, row, state.error.addr(), bi))?;
                }
                bj -= 16;
            }
            if calibrated && i > 0 {
                let shape = product(Operation::None, Operation::Transpose, n, i, STRIP_SPAN, 1.0);
                let operands = [
                    Matrix {
                        pointer: state.error.ptr(),
                        leading: n as i32,
                    },
                    Matrix {
                        pointer: state.factor.ptr().offset(4 * u64::from(i) * u64::from(k)),
                        leading: k as i32,
                    },
                    Matrix {
                        pointer: state.cache.ptr(),
                        leading: n as i32,
                    },
                ];
                gemm(&mut work, handles, queue, &shape, operands)?;
            }
            j -= STRIP_SPAN;
        }
        work.all(plan_pack(state.rate, state.encoded.addr(), tiles, state.words.addr()))?;
        if !calibrated {
            // the error of the rotated weight, in double
            let small = state.small.addr() + 16;
            work.all(plan_fill_floats(small, 8, 0.0))?;
            work.add(plan_subtract_floats(w, decoded, cache, kn))?;
            work.all(plan_sum_products(cache, cache, kn, small))?;
        }
        // back to the original basis: Q = had_r(had_l(decoded) * su) * sv
        work.all(plan_hadamard_rows(decoded, k, n))?;
        work.add(plan_scale_floats(
            decoded,
            k,
            n,
            state.su.addr(),
            Axis::Rows,
            Scaling::Multiply,
        ))?;
        work.all(plan_hadamard_columns(decoded, k, n))?;
        work.add(plan_scale_floats(
            decoded,
            k,
            n,
            state.sv.addr(),
            Axis::Columns,
            Scaling::Multiply,
        ))?;
        if let Some(hessian) = state.hessian {
            refit(&mut work, handles, queue, state, hessian)?;
        }
        work.add(plan_float_to_half(state.su.addr(), u64::from(k), state.suh.addr()))?;
        work.add(plan_float_to_half(state.sv.addr(), u64::from(n), state.svh.addr()))?;
        let out = state
            .rebuilt
            .as_ref()
            .ok_or_else(|| QuantizeError::Broken("the rebuilt weight is gone".into()))?
            .addr();
        work.all(plan_reconstruct(
            state.rate,
            &Reconstruction {
                words: state.words.addr(),
                suh: state.suh.addr(),
                svh: state.svh.addr(),
                k,
                n,
                out,
                scratch: decoded,
            },
        ))?;
        work.flush(queue)?;
        state.words.download(queue)?;
        state.suh.download(queue)?;
        state.svh.download(queue)?;
        Ok(state.small.download(queue)?)
    })?;
    Ok(wait(finish))
}

/// Two rounds of closed-form refit of the scales to the rounded weight `q` (the original basis, `[k, n]`, scaled in
/// place) in the Hessian metric with the damping `d`: each output channel `c = qᵀ H w / qᵀ H q` (into `q` and `sv`),
/// then the input scales solve `((Q Qᵀ) ∘ H) r = rowsum(Q ∘ H W)` (into `q` and `su`). A failed factorization leaves a
/// round's input scales alone and is counted. The sums for the proxy error land in the small buffer.
fn refit(
    work: &mut Work<'_>,
    handles: &mut LaneHandles,
    queue: &mut Enq<'_>,
    state: &mut State,
    hessian: u64,
) -> Result<(), QuantizeError> {
    let (n, k) = state.dims();
    let kn = state.kn();
    let damping = (SIGMA_REG * state.diagonal_mean) as f32;
    let dims = (n, k);
    let small = state.small.addr();
    let (original, q, hw, feedback, system) = (
        state.original.addr(),
        state.decoded.addr(),
        // the rotated weight is dead once the strips are rounded: the refit's product reuses its memory
        state.w.addr(),
        state.cache.addr(),
        state.factor.addr(),
    );
    // the row and the column numerators share one buffer, the denominators the divisor's
    let (numerator, denominator) = (state.rms.addr(), state.divisor.addr());
    work.all(plan_fill_floats(small + 16, 9, 0.0))?;
    gemm_h(work, handles, queue, hessian, dims, original, hw)?;
    work.all(plan_sum_products(original, hw, kn, small + 16))?;
    work.add(plan_axpy_floats(hw, original, kn, damping))?;
    for _ in 0..2 {
        gemm_h(work, handles, queue, hessian, dims, q, feedback)?;
        work.add(plan_column_dots(q, hw, 0.0, k, n, numerator))?;
        work.add(plan_column_dots(q, feedback, damping, k, n, denominator))?;
        work.add(plan_column_ratio(numerator, denominator, n))?;
        work.add(plan_scale_floats(q, k, n, numerator, Axis::Columns, Scaling::Multiply))?;
        work.add(plan_scale_floats(
            state.sv.addr(),
            1,
            n,
            numerator,
            Axis::Columns,
            Scaling::Multiply,
        ))?;

        let gram = product(Operation::Transpose, Operation::None, k, k, n, 0.0);
        gemm(
            work,
            handles,
            queue,
            &gram,
            [matrix(q, n), matrix(q, n), matrix(system, k)],
        )?;
        work.add(plan_hessian_product(system, hessian, k, damping))?;
        work.add(plan_strided_mean(system, k, u64::from(k) + 1, small))?;
        work.add(plan_add_diagonal(system, k, small, 1e-6))?;
        work.add(plan_row_dots(q, hw, k, n, numerator))?;
        factor_attempt(work, handles, queue, state, system)?;
        work.flush(queue)?;
        let raw = queue.raw_stream().ok_or(QuantizeError::NoStream)?;
        // SAFETY: the stream is the lane's own; the factor is the live k by k system, the right-hand side `k` floats
        // of the numerator buffer and the status one int of the small buffer; all are kept until the retirement is
        // confirmed.
        unsafe {
            handles.solve_cholesky(
                &raw,
                Fill::Upper,
                k as i32,
                1,
                state.factor.ptr(),
                state.rms.ptr(),
                state.small.device.ptr().offset(12),
            )
        }?;
        work.all(plan_clean_row_scales(numerator, small + 8, k, small + 48))?;
        work.add(plan_scale_floats(q, k, n, numerator, Axis::Rows, Scaling::Multiply))?;
        work.add(plan_scale_floats(
            state.su.addr(),
            k,
            1,
            numerator,
            Axis::Rows,
            Scaling::Multiply,
        ))?;
    }
    work.add(plan_subtract_floats(original, q, feedback, kn))?;
    gemm_h(work, handles, queue, hessian, dims, feedback, hw)?;
    work.all(plan_sum_products(feedback, hw, kn, small + 24))
}

/// Reads everything back and hands the result over.
fn finish(_stream: &mut CudaStream, _parts: &mut LaneParts, state: &mut State) -> Result<Move, QuantizeError> {
    let kn = state.kn() as f64;
    let (first, second) = (state.small.double_at(16), state.small.double_at(24));
    let (proxy_error, refit_failures) = if state.hessian.is_some() {
        (
            second / first.max(1e-30),
            u32::try_from(state.small.int_at(48)).unwrap_or(0),
        )
    } else {
        (first / kn, 0)
    };
    let decoded = state
        .rebuilt
        .take()
        .ok_or_else(|| QuantizeError::Broken("the rebuilt weight is gone".into()))?;
    Ok(Next::Done(Quantized {
        trellis: state.words.halves(),
        suh: state.suh.halves(),
        svh: state.svh.halves(),
        decoded,
        proxy_error,
        global_scale: state.global_scale,
        refit_failures,
        cholesky_retries: state.attempt,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_off_the_tile_multiple_is_refused() {
        let rate = Rate::new(3.0).unwrap();
        for (n, k) in [(0, 128), (128, 0), (130, 128), (128, 136)] {
            assert!(matches!(
                Shape::new(n, k, rate, TrellisSettings::default()),
                Err(QuantizeError::Request(_))
            ));
        }
    }

    #[test]
    fn the_candidate_rounds_fit_the_buffers_sized_for_them() {
        let shape = Shape::new(256, 384, Rate::new(2.5).unwrap(), TrellisSettings::default()).unwrap();
        let per = |stride: usize| shape.samples.div_ceil(stride);
        assert!(COARSE * per(COARSE_STRIDE as usize) <= shape.candidate_tiles);
        assert!(FINE * per(1) <= shape.candidate_tiles);
    }

    #[test]
    fn a_smaller_scratch_changes_the_chunking_not_the_buffer_a_chunk_needs() {
        let rate = Rate::new(2.0).unwrap();
        let small = TrellisSettings {
            viterbi_scratch_bytes: rate.scratch_bytes(3),
        };
        let shape = Shape::new(256, 256, rate, small).unwrap();
        assert_eq!(shape.scratch_tiles, 3);
        assert_eq!(Sizes::of(&shape, true).viterbi, rate.scratch_bytes(3));
    }
}
