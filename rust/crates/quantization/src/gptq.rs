//! GPTQ rounding of one linear: the Hessian's inverse factor, the tensor-wide NVFP4 scales and the block-wise rounding
//! with error feedback, for Q3, Q4 and Q5 over groups of 64, NVFP4 and NVFP4 with an SD4 table.
//!
//! Each of the three steps is a chain of frames on one lane. A stage submits work and ends in a readback; the
//! retirement of that work, delivered by the lane's [`Completions`], runs the next stage, which reads what came back.
//! Nothing waits on the device inside a frame. The lane's stream and cuBLAS and cuSOLVER handles travel with the
//! chain from one frame to the next and come back with the result in [`Finished`], together with the scratch memory.
//! Nothing in a frame frees device memory with a call that waits: see [`memory`].
//!
//! - [`memory`]: stream-ordered device buffers, pooled pinned staging and the claims that account for them.
//! - [`format`]: the formats, the NVFP4 search's reach, the column order and the damped diagonal.
//! - [`sd4`]: the SD4 table of an NVFP4 object, chosen from the costs the device summed.
//! - [`state`]: one object's rounding and the `.npz` it is stored in.

pub mod format;
pub mod memory;
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
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, PinnedPool};
use thiserror::Error;

use crate::gptq::format::{
    BLOCK, E2M1_MAX, Format, FormatError, Nvfp4Search, SEARCH_BELOW, column_order, damped_diagonal, dead_columns,
    group_of_places, positions,
};
use crate::gptq::memory::{Claims, Device, Host, Memory, Released, StepBytes, pinned_bytes, reclaim};
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
    #[error("the factor was made on another lane: its memory is freed on that lane's stream")]
    ForeignFactor,
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

/// The memory a lane works with: the C library its stream-ordered allocator belongs to, the queue where dropped device
/// buffers wait to be freed in stream order, and the pool the pinned staging comes from (its limit is the pinned
/// buffer pool setting). The queue belongs to the one lane it is made for: a buffer is freed on the stream it was
/// allocated on, so two lanes never share one.
#[derive(Debug)]
pub struct LaneMemory {
    library: Arc<CudaLibrary>,
    released: Arc<Released>,
    pinned: Arc<PinnedPool>,
}

impl LaneMemory {
    /// The memory of a new lane, with a queue of its own.
    pub fn new(library: Arc<CudaLibrary>, pinned: Arc<PinnedPool>) -> LaneMemory {
        LaneMemory {
            library,
            released: Arc::new(Released::new()),
            pinned,
        }
    }

    /// The device buffers given up and not yet freed.
    pub fn waiting(&self) -> u64 {
        self.released.waiting()
    }
}

/// Everything of a lane except its stream: what a stage uses besides the stream.
pub struct LaneParts {
    handles: LaneHandles,
    kernels: Arc<GptqKernels>,
    memory: LaneMemory,
    completions: Arc<Completions>,
    settings: GptqSettings,
    /// The state of chains whose retirement could not be armed, with work possibly still queued on the stream: kept
    /// until a later retirement is confirmed.
    unsettled: Unsettled,
}

/// State parked until the lane's stream has run the work queued before it.
///
/// A chain that fails to arm its retirement ends at once, but the copies and launches it enqueued are still on the
/// stream and write into its pinned staging. That staging must not go back to the shared pool where another lane could
/// take it: it waits here, and the lane's next confirmed retirement (which the stream's order puts behind that work)
/// releases it.
#[derive(Default)]
struct Unsettled {
    held: Vec<Box<dyn Any + Send>>,
}

impl Unsettled {
    fn park(&mut self, state: Box<dyn Any + Send>) {
        self.held.push(state);
    }

    /// Releases what is parked: the stream has retired everything enqueued before now.
    fn release(&mut self) {
        self.held.clear();
    }

    fn len(&self) -> usize {
        self.held.len()
    }
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

/// Memory a chain used. Dropping it gives its device buffers to the lane's [`Released`] queue (freed in stream order by
/// the next step on a lane, or by [`GptqLane::reclaim`]) and its pinned staging to the pool: nothing waits for the
/// device. A chain that failed to arm its retirement hands back an empty scratch: its state stays with the lane until a
/// later retirement confirms the stream has run the work still queued for it.
pub struct Scratch {
    _state: Box<dyn Any + Send>,
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

/// A range of device memory a step reads or writes, with its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRange {
    addr: u64,
    bytes: u64,
}

impl DeviceRange {
    /// `bytes` of device memory at `addr`.
    ///
    /// # Safety
    /// The range must be live device memory of at least `bytes` bytes, which nothing writes (when the step reads it)
    /// or reads (when the step writes it) and which is not freed until the step's `done` call.
    pub unsafe fn new(addr: u64, bytes: u64) -> DeviceRange {
        DeviceRange { addr, bytes }
    }

    pub fn addr(&self) -> u64 {
        self.addr
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The address, when the range holds at least `needed` bytes.
    fn holding(&self, what: &str, needed: u64) -> Result<u64, GptqError> {
        if self.addr == 0 || self.bytes < needed {
            return Err(GptqError::Request(format!(
                "the {what} needs {needed} device bytes and the range holds {}",
                self.bytes
            )));
        }
        Ok(self.addr)
    }
}

impl GptqLane {
    /// A lane from its stream, its handles (cuBLAS products run in full FP32, as the rounding needs), the memory it
    /// works with and the completions that deliver its retirements.
    pub fn new(
        stream: CudaStream,
        mut handles: LaneHandles,
        kernels: Arc<GptqKernels>,
        memory: LaneMemory,
        completions: Arc<Completions>,
        settings: GptqSettings,
    ) -> Result<GptqLane, GptqError> {
        handles.set_math_mode(MathMode::Default)?;
        Ok(GptqLane {
            stream,
            parts: LaneParts {
                handles,
                kernels,
                memory,
                completions,
                settings,
                unsettled: Unsettled::default(),
            },
        })
    }

    /// Takes the lane apart for shutdown: the stream and the handles are destroyed by their owner. The device buffers
    /// given up are freed in stream order first, and the error of that, if any, comes back with the parts. A [`Factor`]
    /// must not outlive its lane: when it is dropped after this, its memory is freed with the plain `cudaFree`.
    pub fn into_parts(mut self) -> (CudaStream, LaneHandles, Result<u64, GptqError>) {
        let reclaimed = self.reclaim();
        (self.stream, self.parts.handles, reclaimed)
    }

    /// The lane's stream, for work the owner enqueues on it besides the chains. A chain that ends with work still queued
    /// keeps its state until the stream's next confirmed retirement: see [`unsettled`](Self::unsettled).
    pub fn stream_mut(&mut self) -> &mut CudaStream {
        &mut self.stream
    }

    /// How many failed chains' states the lane still holds because work of theirs may be queued on the stream.
    pub fn unsettled(&self) -> usize {
        self.parts.unsettled.len()
    }

    /// Frees, in stream order on this lane, the device buffers given up since the last call, and returns how many. A
    /// step does it before it allocates; call it after the last step to free what that left.
    pub fn reclaim(&mut self) -> Result<u64, GptqError> {
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            Ok(reclaim(&memory.released, &memory.library, queue)?)
        })
    }
}

/// An object's prepared Hessian: the column order, each column's position in it, the importance (diag H, in weight
/// order), and the upper factor of the permuted, damped inverse (column-major K x K on the device). Only a factor step
/// makes one, so its parts always agree.
#[derive(Debug, Clone)]
pub struct Factor {
    k: usize,
    order: Vec<u32>,
    position: Vec<u32>,
    importance: Vec<f32>,
    upper: Arc<Device>,
}

impl Factor {
    /// The number of columns.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The columns in the order they are rounded in.
    pub fn order(&self) -> &[u32] {
        &self.order
    }

    /// Each column's place in that order: the inverse permutation.
    pub fn position(&self) -> &[u32] {
        &self.position
    }

    /// The diagonal of the Hessian, in weight order.
    pub fn importance(&self) -> &[f32] {
        &self.importance
    }
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
        stream,
        state: (mut link, _),
        error,
    }) = armed
    {
        // Work may still be queued behind this failure: the state stays with the lane until a retirement is confirmed.
        let cause = link.failure.take().unwrap_or(GptqError::Gpu(error));
        let Link {
            mut parts, state, done, ..
        } = link;
        parts.unsettled.park(Box::new(state));
        done(Finished {
            lane: GptqLane { stream, parts },
            scratch: Scratch { _state: Box::new(()) },
            result: Err(cause),
        });
    }
}

fn resume<S: Send + 'static, T: Send + 'static>(
    mut stream: CudaStream,
    mut link: Link<S, T>,
    next: Option<Stage<S, T>>,
    ticket: RetireTicket,
) {
    let confirmed = stream.confirm_retired(ticket);
    // The stream has run everything enqueued before this retirement, what an earlier failed arm left included.
    link.parts.unsettled.release();
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
        scratch: Scratch {
            _state: Box::new(state),
        },
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

/// The allocations of a step's preparation, made on the lane's stream.
type Mem<'a, 'q> = Memory<'a, Enq<'q>>;

/// A device buffer with a page-locked twin of the same size: what goes up, or what comes back.
struct Staged {
    host: Host,
    device: Device,
}

impl Staged {
    fn new(memory: &mut Mem<'_, '_>, bytes: usize) -> Result<Staged, GptqError> {
        Ok(Staged {
            host: memory.pinned(bytes)?,
            device: memory.device(bytes)?,
        })
    }

    fn of_u32(memory: &mut Mem<'_, '_>, values: &[u32]) -> Result<Staged, GptqError> {
        let mut staged = Staged::new(memory, values.len() * 4)?;
        staged.fill_u32(values);
        Ok(staged)
    }

    fn of_f32(memory: &mut Mem<'_, '_>, values: &[f32]) -> Result<Staged, GptqError> {
        let mut staged = Staged::new(memory, values.len() * 4)?;
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
    upper: Arc<Device>,
    workspace: Device,
    workspace_floats: i32,
    info: Staged,
    prepared: Option<(Vec<f32>, Vec<u32>, Vec<u32>)>,
}

impl GptqLane {
    /// The memory [`factor`](Self::factor) of a `k` by `k` Hessian takes, to be claimed before it. It asks cuSOLVER
    /// for the workspace it needs, which enqueues nothing.
    pub fn factor_bytes(&mut self, k: usize) -> Result<StepBytes, GptqError> {
        let floats = self.workspace_floats(k)?;
        let k = k as u64;
        let words = 4 * k;
        Ok(StepBytes {
            device: 4 * k * k + 2 * words + 12 + 4 * u64::try_from(floats.max(1)).unwrap_or(1),
            pinned: pinned_bytes([words, words, 12]),
        })
    }

    /// The most floats of workspace any of the three cuSOLVER calls on a `k` by `k` matrix asks for.
    fn workspace_floats(&mut self, k: usize) -> Result<i32, GptqError> {
        let n = int(k)?;
        let handles = &mut self.parts.handles;
        let queries = [(false, Fill::Lower), (true, Fill::Lower), (false, Fill::Upper)];
        submit(&mut self.stream, |queue| {
            let raw = queue.raw_stream().ok_or(GptqError::NoStream)?;
            let mut most = 1;
            for (inverse, fill) in queries {
                // SAFETY: the stream is the lane's own; the size query reads no memory, so no matrix is given.
                let floats = unsafe { handles.solver_workspace(&raw, inverse, fill, n, DevicePtr::NULL) }?;
                most = most.max(floats);
            }
            Ok(most)
        })
    }

    /// Prepares the factor of `hessian` (FP32 `[k, k]` on the device, left unchanged, and kept until `done` is called):
    /// act-order (columns by decreasing diag H, stable) or the natural order; dead columns (diag 0) get a unit diagonal,
    /// the diagonal gains `damp` times its mean; then `potrf` (lower), `potri` (lower), mirror and `potrf` (upper).
    /// `claims` account for [`factor_bytes`](Self::factor_bytes).
    ///
    /// A matrix that is not positive definite ends with the cuSOLVER step that refused it. The three `info` words are
    /// read back together, so the later steps have run on what the failed one left; the first failure is the one
    /// reported.
    pub fn factor(
        mut self,
        hessian: DeviceRange,
        k: usize,
        damp: f32,
        act_order: bool,
        claims: Claims,
        done: impl FnOnce(Finished<Factor>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        let prepared = self.prepare_factor(hessian, k, damp, act_order, claims);
        match prepared {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(factor_read_diagonal));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_factor(
        &mut self,
        hessian: DeviceRange,
        k: usize,
        damp: f32,
        act_order: bool,
        claims: Claims,
    ) -> Result<FactorState, GptqError> {
        if k == 0 {
            return Err(GptqError::Request("a Hessian needs k > 0".into()));
        }
        let hessian = hessian.holding("Hessian", 4 * (k as u64) * (k as u64))?;
        let floats = self.workspace_floats(k)?;
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            reclaim(&memory.released, &memory.library, queue)?;
            let mut memory = Memory::new(queue, &memory.library, &memory.released, &memory.pinned, claims);
            Ok(FactorState {
                k,
                damp,
                act_order,
                hessian,
                diagonal: Staged::new(&mut memory, 4 * k)?,
                order: Staged::new(&mut memory, 4 * k)?,
                upper: Arc::new(memory.device(4 * k * k)?),
                workspace: memory.device(4 * usize::try_from(floats).unwrap_or(1))?,
                workspace_floats: floats,
                info: Staged::new(&mut memory, 12)?,
                prepared: None,
            })
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
    values: Device,
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
    /// The memory [`tensor_scales`](Self::tensor_scales) of `rows` rows of `k` columns takes, to be claimed before it.
    pub fn scales_bytes(&self, rows: usize, k: usize, format: &Format) -> Result<StepBytes, GptqError> {
        if k == 0 {
            return Err(GptqError::Request("a weight needs k > 0".into()));
        }
        let (rows, k) = (rows as u64, k as u64);
        let mut bytes = StepBytes {
            device: 4,
            pinned: pinned_bytes([4]),
        };
        if format.sd4 {
            let (costs, small) = (8 * (CODES * CODES) as u64, 8 * CODES as u64);
            bytes.device += costs + 2 * small + 8 * k + 4 * self.chunk_rows(k).min(rows) * k;
            bytes.pinned += pinned_bytes([costs, small, small, 4 * k, 4 * k]);
        }
        Ok(bytes)
    }

    /// The factor, when it was made on this lane: its device memory is freed on the stream it was allocated on.
    fn own(&self, factor: &Factor) -> Result<(), GptqError> {
        if factor.upper.belongs_to(&self.parts.memory.released) {
            Ok(())
        } else {
            Err(GptqError::ForeignFactor)
        }
    }

    /// Rows of the weight one pass of the SD4 costs converts.
    fn chunk_rows(&self, k: u64) -> u64 {
        (self.parts.settings.scale_chunk_bytes / (4 * k)).max(1)
    }

    /// The tensor-wide scales of an NVFP4 object (BF16 `[rows, k]` on the device at `weight`, kept until `done` is
    /// called): the global scale from the largest magnitude and, for SD4, the table chosen under diag H. `claims`
    /// account for [`scales_bytes`](Self::scales_bytes).
    pub fn tensor_scales(
        mut self,
        weight: DeviceRange,
        rows: usize,
        factor: &Factor,
        format: &Format,
        claims: Claims,
        done: impl FnOnce(Finished<TensorScales>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        match self.prepare_scales(weight, rows, factor, format, claims) {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(scales_amax));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_scales(
        &mut self,
        weight: DeviceRange,
        rows: usize,
        factor: &Factor,
        format: &Format,
        claims: Claims,
    ) -> Result<ScalesState, GptqError> {
        let k = factor.k;
        self.own(factor)?;
        if !format.nvfp4 {
            return Err(GptqError::Request("tensor scales belong to NVFP4".into()));
        }
        if rows == 0 || k == 0 {
            return Err(GptqError::Request("the weight and its factor do not agree".into()));
        }
        if !k.is_multiple_of(16) {
            return Err(GptqError::Request(format!(
                "{k} columns do not make whole NVFP4 blocks of 16"
            )));
        }
        let weight = weight.holding("weight", 2 * rows as u64 * k as u64)?;
        let settings = self.parts.settings;
        let chunk_rows = self.chunk_rows(k as u64);
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            reclaim(&memory.released, &memory.library, queue)?;
            let mut memory = Memory::new(queue, &memory.library, &memory.released, &memory.pinned, claims);
            let table = if format.sd4 {
                let held = chunk_rows.min(rows as u64) as usize;
                let identity: Vec<u32> = (0..k as u32).collect();
                Some(Sd4Scratch {
                    costs: Staged::new(&mut memory, 8 * CODES * CODES)?,
                    energy: Staged::new(&mut memory, 8 * CODES)?,
                    counts: Staged::new(&mut memory, 8 * CODES)?,
                    weights: Staged::of_f32(&mut memory, &factor.importance)?,
                    identity: Staged::of_u32(&mut memory, &identity)?,
                    values: memory.device(4 * held * k)?,
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
                amax: Staged::new(&mut memory, 4)?,
                global: 0.0,
                table,
            })
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
    upper: Arc<Device>,
    w: Device,
    decoded: Device,
    decodes: Device,
    block: Device,
    errors: Device,
    values: Device,
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
    pub weight: DeviceRange,
    pub n: usize,
    pub factor: &'a Factor,
    pub format: &'a Format,
    pub tensor: &'a TensorScales,
    /// Where the decoded BF16 `[n, k]` weights go, if wanted.
    pub decoded_out: Option<DeviceRange>,
}

impl GptqLane {
    /// The memory [`round`](Self::round) of `n` rows takes, to be claimed before it.
    pub fn round_bytes(&self, n: usize, factor: &Factor, format: &Format) -> StepBytes {
        let (n, k, group) = (n as u64, factor.k as u64, format.group as u64);
        let groups = k / group;
        let dead = dead_columns(&factor.importance, &factor.order).len() as u64;
        let scale_bytes = if format.nvfp4 { 1 } else { 2 };
        let mut staged = vec![2 * n * k, scale_bytes * n * groups, 4 * k, 4 * k, 4 * k, 4 * k];
        if dead > 0 {
            staged.push(4 * dead);
        }
        if format.sd4 {
            staged.push(ENTRIES as u64);
        }
        let device = 4 * (2 * n * k + n * groups + 2 * n * BLOCK as u64 + n * group) + staged.iter().sum::<u64>();
        StepBytes {
            device,
            pinned: pinned_bytes(staged),
        }
    }

    /// Rounds rows `[0, n)` of an object under the factor; `tensor` comes from
    /// [`tensor_scales`](Self::tensor_scales) for NVFP4 and is ignored by the grouped formats. `claims` account for
    /// [`round_bytes`](Self::round_bytes).
    pub fn round(
        mut self,
        request: &RoundRequest<'_>,
        claims: Claims,
        done: impl FnOnce(Finished<QuantState>) + Send + 'static,
    ) -> Result<(), Box<Refused>> {
        match self.prepare_round(request, claims) {
            Ok(state) => {
                start(self, state, Box::new(done), Box::new(round_all));
                Ok(())
            }
            Err(error) => Err(Box::new(Refused { lane: self, error })),
        }
    }

    fn prepare_round(&mut self, request: &RoundRequest<'_>, claims: Claims) -> Result<RoundState, GptqError> {
        let RoundRequest {
            weight,
            n,
            factor,
            format,
            tensor,
            decoded_out,
        } = *request;
        let k = factor.k;
        self.own(factor)?;
        let group = format.group;
        if n == 0 || k == 0 || !k.is_multiple_of(group) {
            return Err(GptqError::Request(format!(
                "cannot round {n} rows of {k} columns in groups of {group}"
            )));
        }
        if format.sd4 && tensor.table.is_none() {
            return Err(GptqError::Request("an SD4 object needs its table".into()));
        }
        int(n)?;
        int(k)?;
        let matrix = 2 * n as u64 * k as u64;
        let weight = weight.holding("weight", matrix)?;
        let decoded_out = decoded_out
            .map(|out| out.holding("decoded output", matrix))
            .transpose()?;
        let groups = k / group;
        let dead = dead_columns(&factor.importance, &factor.order);
        let scale_bytes = if format.nvfp4 { 1 } else { 2 };
        let above = self.parts.settings.search.above();
        let memory = &self.parts.memory;
        submit(&mut self.stream, |queue| {
            reclaim(&memory.released, &memory.library, queue)?;
            let mut memory = Memory::new(queue, &memory.library, &memory.released, &memory.pinned, claims);
            let dead_up = if dead.is_empty() {
                None
            } else {
                Some((Staged::of_u32(&mut memory, &dead)?, dead.len()))
            };
            let table_up = match (format.sd4, tensor.table) {
                (true, Some(table)) => {
                    let mut staged = Staged::new(&mut memory, ENTRIES)?;
                    staged.host.as_mut_slice().copy_from_slice(&table);
                    Some(staged)
                }
                _ => None,
            };
            Ok(RoundState {
                n,
                k,
                group,
                groups,
                nvfp4: format.nvfp4,
                qmin: format.qmin,
                qmax: format.qmax,
                above,
                global_scale: tensor.global_scale,
                weight,
                decoded_out,
                order: factor.order.clone(),
                shared: 32 * (BLOCK as u32 + 1) * 4,
                state_format: format.clone(),
                table: if format.sd4 { tensor.table } else { None },
                upper: Arc::clone(&factor.upper),
                w: memory.device(4 * n * k)?,
                decoded: memory.device(4 * n * k)?,
                decodes: memory.device(4 * n * groups)?,
                block: memory.device(4 * n * BLOCK)?,
                errors: memory.device(4 * n * BLOCK)?,
                values: memory.device(4 * n * group)?,
                codes: Staged::new(&mut memory, 2 * n * k)?,
                scales: Staged::new(&mut memory, scale_bytes * n * groups)?,
                order_up: Staged::of_u32(&mut memory, &factor.order)?,
                position_up: Staged::of_u32(&mut memory, &factor.position)?,
                group_up: Staged::of_u32(&mut memory, &group_of_places(&factor.order, group))?,
                weights_up: Staged::of_f32(&mut memory, &factor.importance)?,
                dead_up,
                table_up,
            })
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
