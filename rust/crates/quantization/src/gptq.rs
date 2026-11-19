//! GPTQ rounding of one linear: the Hessian's inverse factor, the tensor-wide NVFP4 scales and the block-wise rounding
//! with error feedback, for Q3, Q4 and Q5 over groups of 64, NVFP4 and NVFP4 with an SD4 table.
//!
//! Each of the three steps is a chain of frames on one lane. A stage submits work and ends in a readback; the
//! retirement of that work, delivered by the lane's [`Completions`], runs the next stage, which reads what came back.
//! Nothing waits on the device inside a frame. The lane's stream and cuBLAS and cuSOLVER handles travel with the
//! chain from one frame to the next and come back with the result in [`Finished`], together with the scratch memory,
//! which the caller drops off the lattice (freeing device and pinned memory waits for the device).
//!
//! - [`format`]: the formats, the NVFP4 search's reach, the column order and the damped diagonal.
//! - [`sd4`]: the SD4 table of an NVFP4 object, chosen from the costs the device summed.
//! - [`state`]: one object's rounding and the `.npz` it is stored in.

pub mod format;
pub mod sd4;
pub mod state;

use std::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_gpu::cuda::{CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{
    Args, Dim3, GpuError, LaneEnqueue, LaneStream, Launch, OnceFrame, RetireError, RetireTicket, Stream,
    retire_into_with,
};
use euhedral_graph::Completions;
use euhedral_inference_cuda::math::{Fill, LaneHandles, MathMode, Matrix, Operation, Product, check_info};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, PinnedBuffer};
use thiserror::Error;

use crate::gptq::format::{
    BLOCK, E2M1_MAX, Format, FormatError, Nvfp4Search, SEARCH_BELOW, column_order, damped_diagonal, dead_columns,
    group_of_places, positions,
};
use crate::gptq::sd4::{CODES, ENTRIES};
use crate::gptq::state::{QuantState, StateError};
use crate::kernels::{Compiled, Module, ModuleError, Modules};

/// Why a step could not be started or did not finish.
#[derive(Debug, Error)]
pub enum GptqError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Module(#[from] ModuleError),
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error(transparent)]
    State(#[from] StateError),
    /// The request does not fit the formats or the sizes GPTQ works at.
    #[error("{0}")]
    Request(String),
    #[error("the stream lends no driver handle")]
    NoStream,
    #[error("the kernel {0} is not in the module")]
    Kernel(&'static str),
}

/// The sizes and reach GPTQ is tuned by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GptqSettings {
    /// The NVFP4 block-scale search.
    pub search: Nvfp4Search,
    /// The FP32 bytes of one pass over the weight when the SD4 table's costs are summed.
    pub scale_chunk_bytes: u64,
}

impl Default for GptqSettings {
    fn default() -> Self {
        Self {
            search: Nvfp4Search::DEFAULT,
            scale_chunk_bytes: 256 << 20,
        }
    }
}

/// The kernels of `gptq.cu`, resolved once.
#[derive(Debug)]
pub struct GptqKernels {
    grouped_choose: CudaKernel,
    nvfp4_choose: CudaKernel,
    sd4_costs: CudaKernel,
    segment: CudaKernel,
    gather_group: CudaKernel,
    bf16_to_float_permuted: CudaKernel,
    float_to_bf16: CudaKernel,
    permute_symmetric: CudaKernel,
    read_diagonal: CudaKernel,
    write_diagonal: CudaKernel,
    mirror_lower: CudaKernel,
    amax_bf16: CudaKernel,
    slice_columns: CudaKernel,
    zero_columns: CudaKernel,
}

impl GptqKernels {
    /// Compiles (or finds) the module and resolves its kernels.
    pub fn load(modules: &mut Modules<CudaModule>) -> Result<Arc<GptqKernels>, GptqError> {
        let compiled = modules.load(Module::Gptq)?;
        Ok(Arc::new(Self::resolve(&compiled)?))
    }

    fn resolve(compiled: &Compiled<CudaModule>) -> Result<GptqKernels, GptqError> {
        let get = |name: &'static str| compiled.kernel(name).cloned().ok_or(GptqError::Kernel(name));
        Ok(GptqKernels {
            grouped_choose: get("grouped_choose")?,
            nvfp4_choose: get("nvfp4_choose")?,
            sd4_costs: get("sd4_costs")?,
            segment: get("segment")?,
            gather_group: get("gather_group")?,
            bf16_to_float_permuted: get("bf16_to_float_permuted")?,
            float_to_bf16: get("float_to_bf16")?,
            permute_symmetric: get("permute_symmetric")?,
            read_diagonal: get("read_diagonal")?,
            write_diagonal: get("write_diagonal")?,
            mirror_lower: get("mirror_lower")?,
            amax_bf16: get("amax_bf16")?,
            slice_columns: get("slice_columns")?,
            zero_columns: get("zero_columns")?,
        })
    }
}

/// Everything of a lane except its stream: what a stage uses besides the stream.
pub struct LaneParts {
    handles: LaneHandles,
    kernels: Arc<GptqKernels>,
    library: Arc<CudaLibrary>,
    completions: Arc<Completions>,
    settings: GptqSettings,
}

impl Debug for LaneParts {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("LaneParts").finish_non_exhaustive()
    }
}

/// A lane's stream with the math handles and the kernels GPTQ needs. A step takes the lane by value and gives it back
/// in [`Finished`] (or in [`Refused`], when it could not start).
#[derive(Debug)]
pub struct GptqLane {
    stream: CudaStream,
    parts: LaneParts,
}

/// A step that did not start: the lane comes back untouched by the chain.
#[derive(Debug)]
pub struct Refused {
    pub lane: GptqLane,
    pub error: GptqError,
}

/// Memory a chain used, handed back to the caller to drop off the lattice: freeing it waits for the device.
pub struct Scratch(Box<dyn Any + Send>);

impl Scratch {
    /// Frees the memory, waiting for the device: call it off the lattice.
    pub fn release(self) {
        drop(self.0);
    }
}

impl Debug for Scratch {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Scratch").finish_non_exhaustive()
    }
}

/// What a chain hands to its `done` call, from the frame that ended it.
#[derive(Debug)]
pub struct Finished<T> {
    pub lane: GptqLane,
    pub scratch: Scratch,
    pub result: Result<T, GptqError>,
}

impl GptqLane {
    /// A lane from its stream, its handles (cuBLAS products run in full FP32, as the rounding needs) and the
    /// completions that deliver its retirements.
    pub fn new(
        stream: CudaStream,
        mut handles: LaneHandles,
        kernels: Arc<GptqKernels>,
        library: Arc<CudaLibrary>,
        completions: Arc<Completions>,
        settings: GptqSettings,
    ) -> Result<GptqLane, GptqError> {
        handles.set_math_mode(MathMode::Default)?;
        Ok(GptqLane {
            stream,
            parts: LaneParts {
                handles,
                kernels,
                library,
                completions,
                settings,
            },
        })
    }

    /// Takes the lane apart for shutdown: the stream and the handles are destroyed by their owner.
    pub fn into_parts(self) -> (CudaStream, LaneHandles) {
        (self.stream, self.parts.handles)
    }
}

/// An object's prepared Hessian: the column order, each column's position in it, the importance (diag H, in weight
/// order), and the upper factor of the permuted, damped inverse (column-major K x K on the device).
#[derive(Debug, Clone)]
pub struct Factor {
    pub k: usize,
    pub order: Vec<u32>,
    pub position: Vec<u32>,
    pub importance: Vec<f32>,
    pub upper: Arc<DeviceBuffer>,
}

/// An NVFP4 object's tensor-wide scales: the FP32 global scale and, for SD4, the 16-entry table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorScales {
    pub global_scale: f32,
    pub table: Option<[u8; ENTRIES]>,
}

// ---- the chain -----------------------------------------------------------------------------------------------

type Enq<'a> = <CudaStream as Stream>::Enqueue<'a>;
type Done<T> = Box<dyn FnOnce(Finished<T>) + Send>;
type Stage<S, T> = Box<dyn FnOnce(&mut CudaStream, &mut LaneParts, &mut S) -> Result<Step<S, T>, GptqError> + Send>;

/// What a stage ends with.
enum Step<S, T> {
    /// It submitted work ending in a readback: run the next stage when that retires.
    Wait(Stage<S, T>),
    /// The result is complete.
    Done(T),
}

struct Link<S, T> {
    parts: LaneParts,
    state: S,
    done: Done<T>,
    /// A stage failed with work possibly in flight: the chain waits for the retirement, then ends with this.
    failure: Option<GptqError>,
}

/// Runs the first stage here, then each next one in the frame the lane's completions emit.
fn start<S: Send + 'static, T: Send + 'static>(lane: GptqLane, state: S, done: Done<T>, first: Stage<S, T>) {
    let GptqLane { stream, parts } = lane;
    advance(
        stream,
        Link {
            parts,
            state,
            done,
            failure: None,
        },
        first,
    );
}

fn advance<S: Send + 'static, T: Send + 'static>(mut stream: CudaStream, mut link: Link<S, T>, stage: Stage<S, T>) {
    match stage(&mut stream, &mut link.parts, &mut link.state) {
        Ok(Step::Wait(next)) => arm(stream, link, Some(next)),
        Ok(Step::Done(value)) => finish(stream, link, Ok(value)),
        Err(error) => {
            link.failure = Some(error);
            arm(stream, link, None);
        }
    }
}

fn arm<S: Send + 'static, T: Send + 'static>(stream: CudaStream, link: Link<S, T>, next: Option<Stage<S, T>>) {
    let completions = Arc::clone(&link.parts.completions);
    let id = u64::from(stream.lane().0);
    let armed = retire_into_with(
        stream,
        (link, next),
        &completions,
        move |stream, (link, next), ticket| {
            let frame: Box<dyn Frame> = Box::new(OnceFrame::new(id, move || resume(stream, link, next, ticket)));
            frame
        },
    );
    if let Err(RetireError {
        mut stream,
        state: (mut link, _),
        error,
    }) = armed
    {
        // Work may still be in flight against the scratch: let it end before the scratch goes back.
        stream.quiesce();
        let cause = link.failure.take().unwrap_or(GptqError::Gpu(error));
        finish(stream, link, Err(cause));
    }
}

fn resume<S: Send + 'static, T: Send + 'static>(
    mut stream: CudaStream,
    mut link: Link<S, T>,
    next: Option<Stage<S, T>>,
    ticket: RetireTicket,
) {
    let confirmed = stream.confirm_retired(ticket);
    if let Some(failure) = link.failure.take() {
        return finish(stream, link, Err(failure));
    }
    if let Some(error) = confirmed {
        return finish(stream, link, Err(GptqError::Gpu(error)));
    }
    match next {
        Some(stage) => advance(stream, link, stage),
        None => finish(
            stream,
            link,
            Err(GptqError::Request("a chain resumed with no stage".into())),
        ),
    }
}

fn finish<S: Send + 'static, T: Send + 'static>(stream: CudaStream, link: Link<S, T>, result: Result<T, GptqError>) {
    let Link { parts, state, done, .. } = link;
    done(Finished {
        lane: GptqLane { stream, parts },
        scratch: Scratch(Box::new(state)),
        result,
    });
}

/// Runs `body` as one submission, keeping its own error type.
fn submit<R>(
    stream: &mut CudaStream,
    body: impl for<'a> FnOnce(&mut Enq<'a>) -> Result<R, GptqError>,
) -> Result<R, GptqError> {
    let mut failure = None;
    let outcome = stream.submit(|queue| match body(queue) {
        Ok(value) => Ok(Some(value)),
        Err(error) => {
            failure = Some(error);
            Ok(None)
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    outcome?.ok_or_else(|| GptqError::Request("a submission produced nothing".into()))
}

// ---- memory and launches -------------------------------------------------------------------------------------

fn device(library: &Arc<CudaLibrary>, bytes: usize) -> Result<DeviceBuffer, GptqError> {
    Ok(DeviceBuffer::alloc(library, bytes.max(1))?)
}

fn pinned(library: &Arc<CudaLibrary>, bytes: usize) -> Result<PinnedBuffer, GptqError> {
    Ok(PinnedBuffer::alloc(library, bytes.max(1))?)
}

/// A device buffer with a page-locked twin of the same size: what goes up, or what comes back.
struct Staged {
    host: PinnedBuffer,
    device: DeviceBuffer,
}

impl Staged {
    fn new(library: &Arc<CudaLibrary>, bytes: usize) -> Result<Staged, GptqError> {
        Ok(Staged {
            host: pinned(library, bytes)?,
            device: device(library, bytes)?,
        })
    }

    fn of_u32(library: &Arc<CudaLibrary>, values: &[u32]) -> Result<Staged, GptqError> {
        let mut staged = Staged::new(library, values.len() * 4)?;
        staged.fill_u32(values);
        Ok(staged)
    }

    fn of_f32(library: &Arc<CudaLibrary>, values: &[f32]) -> Result<Staged, GptqError> {
        let mut staged = Staged::new(library, values.len() * 4)?;
        staged.fill_f32(values);
        Ok(staged)
    }

    fn addr(&self) -> u64 {
        self.device.addr()
    }

    fn fill_u32(&mut self, values: &[u32]) {
        for (slot, value) in self.host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
    }

    fn fill_f32(&mut self, values: &[f32]) {
        for (slot, value) in self.host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
    }

    fn read_f32(&self) -> Vec<f32> {
        let (words, _) = self.host.as_slice().as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }

    /// Enqueues the copy of the host bytes to the device. The host bytes must stay as they are until it retires.
    fn upload<Q: LaneEnqueue>(&self, queue: &mut Q) -> Result<(), GptqError> {
        // SAFETY: both ranges are live buffers of `len` bytes owned by this value, which the chain keeps until the
        // retirement is confirmed; nothing writes the host side meanwhile.
        unsafe { queue.copy_to_device(self.device.addr(), self.host.as_ptr(), self.host.len()) }?;
        Ok(())
    }

    /// Enqueues the copy of the device bytes to the host buffer, which is read after the retirement.
    fn download<Q: LaneEnqueue>(&mut self, queue: &mut Q) -> Result<(), GptqError> {
        let bytes = self.host.len();
        // SAFETY: as `upload`; nothing reads the host side before the retirement.
        unsafe { queue.copy_to_host(self.host.as_mut_ptr(), self.device.addr(), bytes) }?;
        Ok(())
    }
}

/// The grid of `count` items at `threads` per block, at least one block.
fn grid(count: u64, threads: u64) -> Result<u32, GptqError> {
    u32::try_from(count.div_ceil(threads).max(1))
        .map_err(|_| GptqError::Request(format!("{count} items need more blocks than a launch has")))
}

fn launch<Q: LaneEnqueue<Kernel = CudaKernel>>(
    queue: &mut Q,
    kernel: &CudaKernel,
    blocks: u32,
    threads: u32,
    shared_bytes: u32,
    args: &Args,
) -> Result<(), GptqError> {
    let geometry = Launch {
        grid: Dim3::x(blocks),
        block: Dim3::x(threads),
        shared_bytes,
        programmatic: false,
    };
    // SAFETY: the arguments are checked against the kernel's parameter sizes by the launch, and name buffers the chain's
    // state owns until the retirement is confirmed.
    unsafe { queue.launch(kernel, geometry, args) }?;
    Ok(())
}

/// A linear launch over `count` items, 256 to a block.
fn launch_linear<Q: LaneEnqueue<Kernel = CudaKernel>>(
    queue: &mut Q,
    kernel: &CudaKernel,
    count: u64,
    args: &Args,
) -> Result<(), GptqError> {
    launch(queue, kernel, grid(count, 256)?, 256, 0, args)
}

fn zero<Q: LaneEnqueue>(queue: &mut Q, address: u64, bytes: usize) -> Result<(), GptqError> {
    // SAFETY: the range is a live buffer of the chain's state, kept until the retirement is confirmed.
    unsafe { queue.memset(address, 0, bytes) }?;
    Ok(())
}

fn int(value: usize) -> Result<i32, GptqError> {
    i32::try_from(value).map_err(|_| GptqError::Request(format!("{value} does not fit a 32-bit count")))
}

// ---- the Hessian's factor ------------------------------------------------------------------------------------

const POTRF_H: &str = "potrf (H)";
const POTRI: &str = "potri";
const POTRF_INVERSE: &str = "potrf (H^-1)";

struct FactorState {
    k: usize,
    damp: f32,
    act_order: bool,
    hessian: u64,
    diagonal: Staged,
    order: Staged,
    upper: Arc<DeviceBuffer>,
    workspace: DeviceBuffer,
    workspace_floats: i32,
    info: Staged,
    prepared: Option<(Vec<f32>, Vec<u32>, Vec<u32>)>,
}

impl GptqLane {
    /// Prepares the factor of `hessian` (FP32 `[k, k]` on the device, left unchanged, and kept until `done` is called):
    /// act-order (columns by decreasing diag H, stable) or the natural order; dead columns (diag 0) get a unit diagonal,
    /// the diagonal gains `damp` times its mean; then `potrf` (lower), `potri` (lower), mirror and `potrf` (upper).
    ///
    /// A matrix that is not positive definite ends with the cuSOLVER step that refused it. The three `info` words are
    /// read back together, so the later steps have run on what the failed one left; the first failure is the one
    /// reported.
    pub fn factor(
        mut self,
        hessian: u64,
        k: usize,
        damp: f32,
        act_order: bool,
        done: impl FnOnce(Finished<Factor>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        let prepared = self.prepare_factor(hessian, k, damp, act_order);
        match prepared {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(factor_read_diagonal));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_factor(&mut self, hessian: u64, k: usize, damp: f32, act_order: bool) -> Result<FactorState, GptqError> {
        if k == 0 || hessian == 0 {
            return Err(GptqError::Request("a Hessian needs a device matrix and k > 0".into()));
        }
        let n = int(k)?;
        let library = Arc::clone(&self.parts.library);
        let upper = Arc::new(device(&library, 4 * k * k)?);
        let handles = &mut self.parts.handles;
        let queries = [(false, Fill::Lower), (true, Fill::Lower), (false, Fill::Upper)];
        let floats = submit(&mut self.stream, |queue| {
            let raw = queue.raw_stream().ok_or(GptqError::NoStream)?;
            let mut most = 1;
            for (inverse, fill) in queries {
                // SAFETY: the stream is the lane's own; the matrix is a live n by n allocation, not read.
                let floats = unsafe { handles.solver_workspace(&raw, inverse, fill, n, upper.ptr()) }?;
                most = most.max(floats);
            }
            Ok(most)
        })?;
        Ok(FactorState {
            k,
            damp,
            act_order,
            hessian,
            diagonal: Staged::new(&library, 4 * k)?,
            order: Staged::new(&library, 4 * k)?,
            workspace: device(&library, 4 * usize::try_from(floats).unwrap_or(1))?,
            workspace_floats: floats,
            info: Staged::new(&library, 12)?,
            upper,
            prepared: None,
        })
    }
}

fn factor_read_diagonal(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    state: &mut FactorState,
) -> Result<Step<FactorState, Factor>, GptqError> {
    let kernels = Arc::clone(&parts.kernels);
    let args = Args::new()
        .arg(state.hessian)
        .arg(int(state.k)?)
        .arg(state.diagonal.addr());
    submit(stream, |queue| {
        launch_linear(queue, &kernels.read_diagonal, state.k as u64, &args)?;
        state.diagonal.download(queue)
    })?;
    Ok(Step::Wait(Box::new(factor_decompose)))
}

fn factor_decompose(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    state: &mut FactorState,
) -> Result<Step<FactorState, Factor>, GptqError> {
    let k = state.k;
    let n = int(k)?;
    let importance = state.diagonal.read_f32();
    let order = column_order(&importance, state.act_order);
    let position = positions(&order);
    state
        .diagonal
        .fill_f32(&damped_diagonal(&importance, &order, state.damp));
    state.order.fill_u32(&order);
    state.prepared = Some((importance, order, position));

    let kernels = Arc::clone(&parts.kernels);
    let handles = &mut parts.handles;
    let upper = state.upper.ptr();
    let workspace = state.workspace.ptr();
    let floats = state.workspace_floats;
    let info = state.info.device.ptr();
    let permute = Args::new()
        .arg(state.hessian)
        .arg(state.order.addr())
        .arg(n)
        .arg(upper.get());
    let write = Args::new().arg(upper.get()).arg(n).arg(state.diagonal.addr());
    let mirror = Args::new().arg(upper.get()).arg(n);
    submit(stream, |queue| {
        state.order.upload(queue)?;
        state.diagonal.upload(queue)?;
        launch_linear(queue, &kernels.permute_symmetric, (k * k) as u64, &permute)?;
        launch_linear(queue, &kernels.write_diagonal, k as u64, &write)?;
        let steps = [
            (false, Fill::Lower, info),
            (true, Fill::Lower, info.offset(4)),
            (false, Fill::Upper, info.offset(8)),
        ];
        for (index, (inverse, fill, word)) in steps.into_iter().enumerate() {
            if index == 2 {
                launch_linear(queue, &kernels.mirror_lower, (k * k) as u64, &mirror)?;
            }
            let raw = queue.raw_stream().ok_or(GptqError::NoStream)?;
            // SAFETY: the stream is the lane's own; the matrix, workspace and info word are live allocations of the
            // sizes the calls take, kept in the chain's state until the retirement is confirmed.
            unsafe { handles.cholesky(&raw, inverse, fill, n, upper, workspace, floats, word) }?;
        }
        state.info.download(queue)
    })?;
    Ok(Step::Wait(Box::new(factor_check)))
}

fn factor_check(
    _stream: &mut CudaStream,
    _parts: &mut LaneParts,
    state: &mut FactorState,
) -> Result<Step<FactorState, Factor>, GptqError> {
    let (words, _) = state.info.host.as_slice().as_chunks::<4>();
    for (call, word) in [POTRF_H, POTRI, POTRF_INVERSE].into_iter().zip(words) {
        check_info(call, i32::from_le_bytes(*word))?;
    }
    let (importance, order, position) = state
        .prepared
        .take()
        .ok_or_else(|| GptqError::Request("the factor was never decomposed".into()))?;
    Ok(Step::Done(Factor {
        k: state.k,
        order,
        position,
        importance,
        upper: Arc::clone(&state.upper),
    }))
}

// ---- the tensor-wide scales ----------------------------------------------------------------------------------

struct Sd4Scratch {
    costs: Staged,
    energy: Staged,
    counts: Staged,
    weights: Staged,
    identity: Staged,
    values: DeviceBuffer,
    chunk_rows: u64,
}

struct ScalesState {
    weight: u64,
    rows: u64,
    k: usize,
    sd4: bool,
    scale_max: f32,
    amax: Staged,
    global: f32,
    table: Option<Sd4Scratch>,
}

impl GptqLane {
    /// The tensor-wide scales of an NVFP4 object (BF16 `[rows, k]` on the device at `weight`, kept until `done` is
    /// called): the global scale from the largest magnitude and, for SD4, the table chosen under diag H.
    pub fn tensor_scales(
        mut self,
        weight: u64,
        rows: usize,
        factor: &Factor,
        format: &Format,
        done: impl FnOnce(Finished<TensorScales>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        match self.prepare_scales(weight, rows, factor, format) {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(scales_amax));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_scales(
        &mut self,
        weight: u64,
        rows: usize,
        factor: &Factor,
        format: &Format,
    ) -> Result<ScalesState, GptqError> {
        let k = factor.k;
        if !format.nvfp4 {
            return Err(GptqError::Request("tensor scales belong to NVFP4".into()));
        }
        if rows == 0 || weight == 0 || k == 0 || factor.importance.len() != k {
            return Err(GptqError::Request("the weight and its factor do not agree".into()));
        }
        let library = Arc::clone(&self.parts.library);
        let settings = self.parts.settings;
        let table = if format.sd4 {
            let chunk_rows = (settings.scale_chunk_bytes / (4 * k as u64)).max(1);
            let held = chunk_rows.min(rows as u64) as usize;
            let identity: Vec<u32> = (0..k as u32).collect();
            Some(Sd4Scratch {
                costs: Staged::new(&library, 8 * CODES * CODES)?,
                energy: Staged::new(&library, 8 * CODES)?,
                counts: Staged::new(&library, 8 * CODES)?,
                weights: Staged::of_f32(&library, &factor.importance)?,
                identity: Staged::of_u32(&library, &identity)?,
                values: device(&library, 4 * held * k)?,
                chunk_rows,
            })
        } else {
            None
        };
        Ok(ScalesState {
            weight,
            rows: rows as u64,
            k,
            sd4: format.sd4,
            scale_max: settings.search.scale_max(),
            amax: Staged::new(&library, 4)?,
            global: 0.0,
            table,
        })
    }
}

fn scales_amax(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    state: &mut ScalesState,
) -> Result<Step<ScalesState, TensorScales>, GptqError> {
    let kernels = Arc::clone(&parts.kernels);
    let args = Args::new()
        .arg(state.weight)
        .arg((state.rows * state.k as u64) as i64)
        .arg(state.amax.addr());
    submit(stream, |queue| {
        zero(queue, state.amax.addr(), 4)?;
        launch(queue, &kernels.amax_bf16, 1024, 256, 0, &args)?;
        state.amax.download(queue)
    })?;
    Ok(Step::Wait(Box::new(scales_global)))
}

fn scales_global(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    state: &mut ScalesState,
) -> Result<Step<ScalesState, TensorScales>, GptqError> {
    let largest = state.amax.read_f32().first().copied().unwrap_or(0.0);
    let global = largest / (E2M1_MAX * state.scale_max);
    state.global = global;
    if !state.sd4 {
        return Ok(Step::Done(TensorScales {
            global_scale: global,
            table: None,
        }));
    }
    if global <= 0.0 {
        let mut identity = [0u8; ENTRIES];
        for (entry, slot) in identity.iter_mut().enumerate() {
            *slot = entry as u8;
        }
        return Ok(Step::Done(TensorScales {
            global_scale: global,
            table: Some(identity),
        }));
    }
    let kernels = Arc::clone(&parts.kernels);
    let (weight, rows, k) = (state.weight, state.rows, state.k);
    let scratch = state
        .table
        .as_mut()
        .ok_or_else(|| GptqError::Request("an SD4 object has no table scratch".into()))?;
    submit(stream, |queue| {
        scratch.weights.upload(queue)?;
        scratch.identity.upload(queue)?;
        zero(queue, scratch.costs.addr(), scratch.costs.host.len())?;
        zero(queue, scratch.energy.addr(), scratch.energy.host.len())?;
        zero(queue, scratch.counts.addr(), scratch.counts.host.len())?;
        let mut begin = 0u64;
        while begin < rows {
            let count = scratch.chunk_rows.min(rows - begin);
            let convert = Args::new()
                .arg(weight + 2 * begin * k as u64)
                .arg(count as i64)
                .arg(k as i64)
                .arg(scratch.identity.addr())
                .arg(scratch.values.addr());
            launch_linear(queue, &kernels.bf16_to_float_permuted, count * k as u64, &convert)?;
            let blocks = count * k as u64 / 16;
            let costs = Args::new()
                .arg(scratch.values.addr())
                .arg(blocks as i64)
                .arg(int(k)?)
                .arg(global)
                .arg(scratch.weights.addr())
                .arg(SEARCH_BELOW)
                .arg(scratch.costs.addr())
                .arg(scratch.energy.addr())
                .arg(scratch.counts.addr());
            launch(queue, &kernels.sd4_costs, grid(blocks, 128)?, 128, 0, &costs)?;
            begin += count;
        }
        scratch.costs.download(queue)?;
        scratch.energy.download(queue)?;
        scratch.counts.download(queue)
    })?;
    Ok(Step::Wait(Box::new(scales_table)))
}

fn scales_table(
    _stream: &mut CudaStream,
    _parts: &mut LaneParts,
    state: &mut ScalesState,
) -> Result<Step<ScalesState, TensorScales>, GptqError> {
    let scratch = state
        .table
        .as_ref()
        .ok_or_else(|| GptqError::Request("an SD4 object has no table scratch".into()))?;
    let doubles = |bytes: &[u8]| -> Vec<f64> {
        let (words, _) = bytes.as_chunks::<8>();
        words.iter().map(|word| f64::from_le_bytes(*word)).collect()
    };
    let flat = doubles(scratch.costs.host.as_slice());
    let costs: Vec<Vec<f64>> = flat.chunks(CODES).map(<[f64]>::to_vec).collect();
    let energy = doubles(scratch.energy.host.as_slice());
    let (words, _) = scratch.counts.host.as_slice().as_chunks::<8>();
    let counts: Vec<u64> = words.iter().map(|word| u64::from_le_bytes(*word)).collect();
    let table = sd4::choose(&costs, &energy, &counts, SEARCH_BELOW as usize);
    Ok(Step::Done(TensorScales {
        global_scale: state.global,
        table: Some(table),
    }))
}

// ---- the rounding --------------------------------------------------------------------------------------------

struct RoundState {
    n: usize,
    k: usize,
    group: usize,
    groups: usize,
    nvfp4: bool,
    qmin: i32,
    qmax: i32,
    above: i32,
    global_scale: f32,
    weight: u64,
    decoded_out: Option<u64>,
    order: Vec<u32>,
    shared: u32,
    state_format: Format,
    table: Option<[u8; ENTRIES]>,
    upper: Arc<DeviceBuffer>,
    w: DeviceBuffer,
    decoded: DeviceBuffer,
    decodes: DeviceBuffer,
    block: DeviceBuffer,
    errors: DeviceBuffer,
    values: DeviceBuffer,
    codes: Staged,
    scales: Staged,
    order_up: Staged,
    position_up: Staged,
    group_up: Staged,
    weights_up: Staged,
    dead_up: Option<(Staged, usize)>,
    table_up: Option<Staged>,
}

/// What [`GptqLane::round`] rounds.
#[derive(Debug, Clone, Copy)]
pub struct RoundRequest<'a> {
    /// BF16 `[n, k]` on the device, kept until the step is done.
    pub weight: u64,
    pub n: usize,
    pub factor: &'a Factor,
    pub format: &'a Format,
    pub tensor: &'a TensorScales,
    /// Where the decoded BF16 `[n, k]` weights go, if wanted.
    pub decoded_out: Option<u64>,
}

impl GptqLane {
    /// Rounds rows `[0, n)` of an object under the factor; `tensor` comes from
    /// [`tensor_scales`](Self::tensor_scales) for NVFP4 and is ignored by the grouped formats.
    pub fn round(
        mut self,
        request: &RoundRequest<'_>,
        done: impl FnOnce(Finished<QuantState>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        match self.prepare_round(request) {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(round_all));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_round(&mut self, request: &RoundRequest<'_>) -> Result<RoundState, GptqError> {
        let RoundRequest {
            weight,
            n,
            factor,
            format,
            tensor,
            decoded_out,
        } = *request;
        let k = factor.k;
        let group = format.group;
        if n == 0 || k == 0 || weight == 0 || !k.is_multiple_of(group) || factor.order.len() != k {
            return Err(GptqError::Request(format!(
                "cannot round {n} rows of {k} columns in groups of {group}"
            )));
        }
        if format.sd4 && tensor.table.is_none() {
            return Err(GptqError::Request("an SD4 object needs its table".into()));
        }
        int(n)?;
        int(k)?;
        let groups = k / group;
        let library = Arc::clone(&self.parts.library);
        let dead = dead_columns(&factor.importance, &factor.order);
        let dead_up = if dead.is_empty() {
            None
        } else {
            Some((Staged::of_u32(&library, &dead)?, dead.len()))
        };
        let table_up = match (format.sd4, tensor.table) {
            (true, Some(table)) => {
                let mut staged = Staged::new(&library, ENTRIES)?;
                staged.host.as_mut_slice().copy_from_slice(&table);
                Some(staged)
            }
            _ => None,
        };
        let scale_bytes = if format.nvfp4 { 1 } else { 2 };
        Ok(RoundState {
            n,
            k,
            group,
            groups,
            nvfp4: format.nvfp4,
            qmin: format.qmin,
            qmax: format.qmax,
            above: self.parts.settings.search.above(),
            global_scale: tensor.global_scale,
            weight,
            decoded_out,
            order: factor.order.clone(),
            shared: 32 * (BLOCK as u32 + 1) * 4,
            state_format: format.clone(),
            table: if format.sd4 { tensor.table } else { None },
            upper: Arc::clone(&factor.upper),
            w: device(&library, 4 * n * k)?,
            decoded: device(&library, 4 * n * k)?,
            decodes: device(&library, 4 * n * groups)?,
            block: device(&library, 4 * n * BLOCK)?,
            errors: device(&library, 4 * n * BLOCK)?,
            values: device(&library, 4 * n * group)?,
            codes: Staged::new(&library, 2 * n * k)?,
            scales: Staged::new(&library, scale_bytes * n * groups)?,
            order_up: Staged::of_u32(&library, &factor.order)?,
            position_up: Staged::of_u32(&library, &factor.position)?,
            group_up: Staged::of_u32(&library, &group_of_places(&factor.order, group))?,
            weights_up: Staged::of_f32(&library, &factor.importance)?,
            dead_up,
            table_up,
        })
    }
}

/// Launches the rounding kernel over places `first..last` of the block that starts at `begin`.
fn round_segment<Q: LaneEnqueue<Kernel = CudaKernel>>(
    queue: &mut Q,
    kernels: &GptqKernels,
    state: &RoundState,
    begin: usize,
    first: usize,
    last: usize,
) -> Result<(), GptqError> {
    let width = (state.k - begin).min(BLOCK);
    let args = Args::new()
        .arg(state.block.addr())
        .arg(state.errors.addr())
        .arg(state.upper.addr())
        .arg(state.decodes.addr())
        .arg(state.group_up.addr() + 4 * begin as u64)
        .arg(state.order_up.addr() + 4 * begin as u64)
        .arg(state.codes.addr())
        .arg(state.decoded.addr())
        .arg(int(state.n)?)
        .arg(int(width)?)
        .arg(state.k as i64)
        .arg(1i64)
        .arg(state.k as i64)
        .arg(int(begin)?)
        .arg(int(first)?)
        .arg(int(last)?)
        .arg(int(state.groups)?)
        .arg(i32::from(state.nvfp4))
        .arg(state.qmin as f32)
        .arg(state.qmax as f32);
    launch(
        queue,
        &kernels.segment,
        grid(state.n as u64, 32)?,
        128,
        state.shared,
        &args,
    )
}

/// Chooses group `g`'s scale from the error-updated weights of its columns.
fn round_choose<Q: LaneEnqueue<Kernel = CudaKernel>>(
    queue: &mut Q,
    kernels: &GptqKernels,
    state: &RoundState,
    g: usize,
    begin: usize,
    end: usize,
) -> Result<(), GptqError> {
    let width = end - begin;
    let (n, group) = (int(state.n)?, int(state.group)?);
    let gather = Args::new()
        .arg(state.w.addr())
        .arg(state.k as i64)
        .arg(state.block.addr())
        .arg(int(width)?)
        .arg(int(begin)?)
        .arg(int(end)?)
        .arg(state.position_up.addr() + 4 * (g * state.group) as u64)
        .arg(group)
        .arg(n)
        .arg(state.values.addr());
    launch(
        queue,
        &kernels.gather_group,
        grid(state.n as u64, 128)?,
        128,
        0,
        &gather,
    )?;
    let weights = state.weights_up.addr() + 4 * (g * state.group) as u64;
    let groups = int(state.groups)?;
    if state.nvfp4 {
        let table = state.table_up.as_ref().map_or(0, Staged::addr);
        let args = Args::new()
            .arg(state.values.addr())
            .arg(group)
            .arg(weights)
            .arg(n)
            .arg(state.global_scale)
            .arg(table)
            .arg(SEARCH_BELOW)
            .arg(state.above)
            .arg(state.scales.addr() + g as u64)
            .arg(state.decodes.addr() + 4 * g as u64)
            .arg(groups);
        launch(queue, &kernels.nvfp4_choose, grid(state.n as u64, 128)?, 128, 0, &args)
    } else {
        let args = Args::new()
            .arg(state.values.addr())
            .arg(group)
            .arg(weights)
            .arg(n)
            .arg(group)
            .arg(state.qmin)
            .arg(state.qmax)
            .arg(state.scales.addr() + 2 * g as u64)
            .arg(state.decodes.addr() + 4 * g as u64)
            .arg(groups);
        launch(
            queue,
            &kernels.grouped_choose,
            grid(state.n as u64, 128)?,
            128,
            0,
            &args,
        )
    }
}

fn round_all(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    state: &mut RoundState,
) -> Result<Step<RoundState, QuantState>, GptqError> {
    let kernels = Arc::clone(&parts.kernels);
    let handles = &mut parts.handles;
    submit(stream, |queue| {
        let (n, k) = (state.n, state.k);
        state.order_up.upload(queue)?;
        state.position_up.upload(queue)?;
        state.group_up.upload(queue)?;
        state.weights_up.upload(queue)?;
        if let Some(table) = &state.table_up {
            table.upload(queue)?;
        }
        let convert = Args::new()
            .arg(state.weight)
            .arg(n as i64)
            .arg(k as i64)
            .arg(state.order_up.addr())
            .arg(state.w.addr());
        launch_linear(queue, &kernels.bf16_to_float_permuted, (n * k) as u64, &convert)?;
        if let Some((dead, count)) = &state.dead_up {
            dead.upload(queue)?;
            let args = Args::new()
                .arg(state.w.addr())
                .arg(k as i64)
                .arg(n as i64)
                .arg(dead.addr())
                .arg(int(*count)?);
            launch_linear(queue, &kernels.zero_columns, (n * count) as u64, &args)?;
        }
        zero(queue, state.decodes.addr(), 4 * n * state.groups)?;
        let mut chosen = vec![false; state.groups];
        let mut begin = 0;
        while begin < k {
            let end = k.min(begin + BLOCK);
            let width = end - begin;
            let slice = Args::new()
                .arg(state.w.addr())
                .arg(k as i64)
                .arg(int(begin)?)
                .arg(int(width)?)
                .arg(n as i64)
                .arg(state.block.addr());
            launch_linear(queue, &kernels.slice_columns, (n * width) as u64, &slice)?;
            zero(queue, state.errors.addr(), 4 * n * width)?;
            let mut first = 0;
            for place in 0..width {
                let g = state.order[begin + place] as usize / state.group;
                if chosen[g] {
                    continue;
                }
                if place > first {
                    round_segment(queue, &kernels, state, begin, first, place)?;
                }
                first = place;
                round_choose(queue, &kernels, state, g, begin, end)?;
                chosen[g] = true;
            }
            round_segment(queue, &kernels, state, begin, first, width)?;
            if end < k {
                // w[:, end:] -= errors @ U[begin:end, end:]; in column-major terms the transposed update below.
                let product = Product {
                    a_operation: Operation::Transpose,
                    b_operation: Operation::None,
                    m: int(k - end)?,
                    n: int(n)?,
                    k: int(width)?,
                    alpha: -1.0,
                    beta: 1.0,
                };
                let a = Matrix {
                    pointer: state.upper.ptr().offset(4 * (end * k + begin) as u64),
                    leading: int(k)?,
                };
                let b = Matrix {
                    pointer: state.errors.ptr(),
                    leading: int(width)?,
                };
                let c = Matrix {
                    pointer: state.w.ptr().offset(4 * end as u64),
                    leading: int(k)?,
                };
                let raw = queue.raw_stream().ok_or(GptqError::NoStream)?;
                // SAFETY: the stream is the lane's own; the three operands are live allocations of the sizes the
                // product and leading dimensions imply, owned by the chain's state until the retirement is confirmed.
                unsafe { handles.gemm_fp32(&raw, &product, a, b, c) }?;
            }
            begin = end;
        }
        if let Some(out) = state.decoded_out {
            let args = Args::new().arg(state.decoded.addr()).arg((n * k) as i64).arg(out);
            launch_linear(queue, &kernels.float_to_bf16, (n * k) as u64, &args)?;
        }
        state.codes.download(queue)?;
        state.scales.download(queue)
    })?;
    Ok(Step::Wait(Box::new(round_collect)))
}

fn round_collect(
    _stream: &mut CudaStream,
    _parts: &mut LaneParts,
    state: &mut RoundState,
) -> Result<Step<RoundState, QuantState>, GptqError> {
    let (wide, _) = state.codes.host.as_slice().as_chunks::<2>();
    let codes: Vec<u8> = wide.iter().map(|code| i16::from_le_bytes(*code) as u8).collect();
    let format = &state.state_format;
    let quant = QuantState {
        format_name: format.name.clone(),
        layout: format.state_layout().to_string(),
        rows: state.n as u64,
        columns: state.k as u64,
        codes,
        scales: state.scales.host.as_slice().to_vec(),
        global_scale: if state.nvfp4 { state.global_scale } else { 0.0 },
        table: state.table,
    };
    Ok(Step::Done(quant))
}
