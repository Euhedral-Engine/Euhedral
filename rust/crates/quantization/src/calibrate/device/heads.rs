//! The heads on the card: the calls [`Heads`] makes over the hidden states the layers left in host memory.
//!
//! Every call is one or a few submissions on the main lane, each walking the batches of sequences as the layer passes
//! do: a batch's states go up to the batch buffer, a step runs over them, and the pass may write them back. The
//! Hessians of the final norm's output and of the MTP block's are accumulated in the order Java does, so the rounding
//! reads the same bits. The rounding itself is the layers': [`round_all`] over one linear.
//!
//! The MTP block reads each position's hidden state with the next position's token: its input is the stem, the
//! embedding of the next token and the normed hidden state side by side, `[rows, 2 * hidden]` with `rows = length - 1`.
//! Its input projection is rounded under the stem's Hessian, the projection makes the states the MTP layer runs over
//! (compacted in the host states, a sequence `rows` long), and the layer is the layers' own chain with that length.

use std::mem::{forget, take};
use std::sync::Arc;

use euhedral_gpu::cuda::CudaLaneEnqueue;
use euhedral_gpu::{Lane, LaneEnqueue};
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::math::LaneHandles;
use euhedral_inference_cuda::{DeviceBuffer, KernelOps, LaneOps, PinnedBuffer, PinnedRange};
use euhedral_inference_engine::Handoff;

use super::staging::{HeadMemory, PartReader, read_parts};
use super::trellis::mtp_tensors;
use super::{BF16_BYTES, EMBEDDING, FP32_BYTES, Inner, after, gpu_error, run_target, save_manifest};
use crate::calibrate::CalibrateError;
use crate::calibrate::driver::{Then, end, single};
use crate::calibrate::head_state::{HeadState, MtpBuffers, at_pass_length, drop_resume_file, release_failed};
use crate::calibrate::heads::{DraftHessian, Heads};
use crate::calibrate::manifest::Part;
use crate::calibrate::rounding::{Ended, Linear, round_all};
use crate::checkpoint::CheckpointError;
use crate::checkpoint::objects::{MTP_OBJECTS, Qwen38Objects};
use crate::checkpoint::safetensors::Safetensors;
use crate::forward::{ForwardError, LaneContext, Target};
use crate::gptq::state::QuantState;
use crate::step::idle;

/// The final norm, applied to the hidden states before the output head.
const FINAL_NORM: &str = "model.language_model.norm.weight";
/// The MTP block's norms: of the next token's embedding, of the hidden state, and of its output.
const EMBEDDING_NORM: &str = "mtp.pre_fc_norm_embedding.weight";
const HIDDEN_NORM: &str = "mtp.pre_fc_norm_hidden.weight";
const MTP_NORM: &str = "mtp.norm.weight";
const INPUT_PROJECTION: &str = "mtp/input_projection";
pub(super) const OUTPUT_HEAD: &str = "text/output_head";
const DRAFT_HEAD: &str = "text/draft_head";
const I32_BYTES: usize = 4;

/// What the heads keep on the card between their calls.
type State = HeadState<DeviceBuffer>;

/// The device calls of the heads.
pub struct DeviceHeads {
    inner: Arc<Inner>,
}

impl DeviceHeads {
    pub fn new(inner: Arc<Inner>) -> Self {
        Self { inner }
    }

    /// `then`, which on a failure is first preceded by the freeing of everything the heads hold on the card, in
    /// stream order: a failed call leaves buffers in the state, and a plain drop of them would wait for the device.
    fn cleaned<T: Send + 'static>(&self, then: Then<T>) -> Then<T> {
        let inner = Arc::clone(&self.inner);
        Box::new(move |result| {
            if result.is_err() {
                release_failed(&inner.heads, |buffers| free_in_stream(&inner, buffers));
            }
            then(result);
        })
    }
}

/// The sequences `first..first + count` of a pass.
#[derive(Clone, Copy)]
struct Batch {
    first: usize,
    count: usize,
}

/// How a pass walks the states: `rows` of each sequence are read, and `write` rows of each are written back, the
/// sequences following each other.
#[derive(Clone, Copy)]
struct Layout {
    rows: usize,
    write: Option<usize>,
}

pub(super) type Context<'a, 'q> = LaneContext<'a, CudaLaneEnqueue<'q>>;

pub(super) fn lane<'a, 'q>(
    inner: &'a Inner,
    queue: &'a mut CudaLaneEnqueue<'q>,
    handles: &'a mut LaneHandles,
) -> Context<'a, 'q> {
    let ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
    LaneContext::new(ops, handles, &inner.kernels)
}

/// Runs `work` on the main lane with its math handles, and gives `then` its value, with `keep`, in the frame that
/// follows the marker. `keep` is what the work reads from the host: it stays until the lane has retired the work, and
/// after a failed submission, which may have queued copies from it, it is never freed.
pub(super) fn on_main<K, T>(
    inner: &Arc<Inner>,
    what: &'static str,
    keep: K,
    work: impl FnOnce(&mut CudaLaneEnqueue<'_>, &mut LaneHandles, &K) -> Result<T, ForwardError>,
    then: impl FnOnce(Result<(T, K), CalibrateError>) + Send + 'static,
) where
    K: Send + 'static,
    T: Send + 'static,
{
    let submitted = inner.main.submit_marked(|queue| {
        let mut handles = inner
            .handles
            .take()
            .ok_or_else(|| gpu_error("the main lane's handles are in use"))?;
        let made = work(queue, &mut handles, &keep);
        inner.handles.send(handles);
        made.map_err(gpu_error)
    });
    match submitted {
        Err(error) => {
            forget(keep);
            then(Err(CalibrateError::failed(what, error)));
        }
        Ok((value, marker)) => after(inner, marker, move || then(Ok((value, keep)))),
    }
}

/// Runs `work` over the heads' state.
fn with_state<R>(inner: &Inner, work: impl FnOnce(&mut State) -> Result<R, ForwardError>) -> Result<R, ForwardError> {
    let mut state = inner
        .heads
        .take()
        .ok_or_else(|| ForwardError::Invalid("the heads' state is in use".to_owned()))?;
    let made = work(&mut state);
    inner.heads.send(state);
    made
}

fn held<'a>(buffer: &'a Option<DeviceBuffer>, what: &str) -> Result<&'a DeviceBuffer, ForwardError> {
    buffer
        .as_ref()
        .ok_or_else(|| ForwardError::Invalid(format!("the heads have no {what}")))
}

/// Walks the batches of the states: each goes up to the batch buffer, `step` runs over it, and the pass's output, if
/// it writes, goes back.
fn walk(
    inner: &Inner,
    queue: &mut CudaLaneEnqueue<'_>,
    handles: &mut LaneHandles,
    layout: Layout,
    mut step: impl FnMut(&mut Context<'_, '_>, Batch) -> Result<(), ForwardError>,
) -> Result<(), ForwardError> {
    let row_bytes = inner.row_bytes();
    let sequences = inner.settings.sequences();
    let mut first = 0;
    while first < sequences {
        let count = inner.settings.batch.min(sequences - first);
        let (start, bytes) = (first * layout.rows * row_bytes, count * layout.rows * row_bytes);
        if start + bytes > inner.states.bytes {
            return Err(ForwardError::Invalid(format!(
                "the batch ends at byte {}, past the {} host bytes",
                start + bytes,
                inner.states.bytes
            )));
        }
        // SAFETY: the states are the run's own pinned memory, which nothing else touches while the heads run, and
        // they outlive the lane's work: they are never freed before the run ends. The batch buffer holds a batch.
        unsafe {
            queue.copy_to_device(inner.batch_states.addr(), inner.states.pointer().add(start), bytes)?;
        }
        step(&mut lane(inner, queue, handles), Batch { first, count })?;
        if let Some(rows) = layout.write {
            let (start, bytes) = (first * rows * row_bytes, count * rows * row_bytes);
            if start + bytes > inner.states.bytes {
                return Err(ForwardError::Invalid(format!(
                    "the batch writes to byte {}, past the {} host bytes",
                    start + bytes,
                    inner.states.bytes
                )));
            }
            // SAFETY: as for the copy in; the lane orders the copy after the step. The bytes written end before the
            // bytes the next batch reads begin, since `rows` is at most the rows read.
            unsafe {
                queue.copy_to_host(inner.states.pointer().add(start), inner.batch_states.addr(), bytes)?;
            }
        }
        first += count;
    }
    Ok(())
}

/// A new buffer of `bytes`, zeroed, on the lane.
fn zeroed(context: &mut Context<'_, '_>, bytes: usize) -> Result<DeviceBuffer, ForwardError> {
    let buffer = context.ops().alloc_async(bytes)?;
    if let Err(error) = context.ops().zero_device_memory(buffer.ptr(), bytes as u64) {
        // Freed in stream order: a plain drop waits for the device.
        let _ = context.ops().free_async(buffer);
        return Err(error.into());
    }
    Ok(buffer)
}

/// Uploads part `index` of `memory` to a new buffer on the lane, in copies of at most the run's chunk.
///
/// # Safety
/// `memory` stays live, and unchanged, until the lane has retired the work queued up to this call.
pub(super) unsafe fn upload_part(
    inner: &Inner,
    context: &mut Context<'_, '_>,
    memory: &HeadMemory,
    index: usize,
) -> Result<DeviceBuffer, ForwardError> {
    let Some((start, bytes)) = memory.part(index) else {
        return Err(ForwardError::Invalid(format!("the head tensors have no part {index}")));
    };
    let buffer = context.ops().alloc_async(bytes)?;
    let mut offset = 0;
    while offset < bytes {
        let chunk = inner.settings.chunk_bytes.min(bytes - offset);
        // SAFETY: the caller keeps `memory` as this function's contract says; the destination range lies in `buffer`,
        // and the part lies in the staging buffer.
        let queued = unsafe {
            context.ops().copy_upload_to_device(
                buffer.ptr().offset(offset as u64),
                memory.buffer(),
                start + offset,
                chunk,
            )
        };
        if let Err(error) = queued {
            let _ = context.ops().free_async(buffer);
            return Err(error.into());
        }
        offset += chunk;
    }
    Ok(buffer)
}

/// Frees `buffers` in stream order on the lane.
pub(super) fn free(
    context: &mut Context<'_, '_>,
    buffers: impl IntoIterator<Item = DeviceBuffer>,
) -> Result<(), ForwardError> {
    for buffer in buffers {
        context.ops().free_async(buffer)?;
    }
    Ok(())
}

/// After a failure: frees `buffers` in stream order on the main lane, without waiting for it. When the lane takes no
/// more work they are left allocated (the run is ending), never dropped, since a plain drop waits for the device.
fn free_in_stream(inner: &Arc<Inner>, buffers: Vec<DeviceBuffer>) {
    let held = Handoff::empty();
    held.send(buffers);
    let submitted = inner.main.submit_marked(|queue| {
        let mut ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
        for buffer in held.take().unwrap_or_default() {
            ops.free_async(buffer).map_err(gpu_error)?;
        }
        Ok(())
    });
    if submitted.is_err() {
        forget(held.take());
    }
}

/// The calls that read tensors into the heads' staging buffer. Each call takes the names of its tensors from
/// [`Call::names`] and the staging buffer is sized from the same names ([`head_groups`]), so a call cannot ask for more
/// than the buffer holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Call {
    /// The embedding table, for the states of layer 0 and for nothing else.
    Embedding,
    /// The final norm, for the output head's Hessian.
    FinalNorm,
    /// The MTP block's stem, in a GPTQ run: its parts are [`StemPart`]s.
    Stem,
    /// The output head's weight.
    OutputHead,
    /// The eight linears of the MTP block, in a trellis run.
    TrellisMtp,
}

impl Call {
    const ALL: [Call; 5] = [
        Call::Embedding,
        Call::FinalNorm,
        Call::Stem,
        Call::OutputHead,
        Call::TrellisMtp,
    ];

    /// Whether a run of this kind makes the call.
    fn made_by(self, trellis: bool) -> bool {
        match self {
            Call::Embedding | Call::FinalNorm | Call::OutputHead => true,
            Call::Stem => !trellis,
            Call::TrellisMtp => trellis,
        }
    }

    /// The names of the tensors the call reads, in the order of its parts.
    pub(super) fn names(self, objects: &Qwen38Objects) -> Result<Vec<String>, CalibrateError> {
        match self {
            Call::Embedding => Ok(vec![EMBEDDING.to_owned()]),
            Call::FinalNorm => Ok(vec![FINAL_NORM.to_owned()]),
            Call::Stem => Ok(stem_names(&tensor_of(objects, INPUT_PROJECTION)?.0)),
            Call::OutputHead => Ok(vec![tensor_of(objects, OUTPUT_HEAD)?.0]),
            Call::TrellisMtp => Ok(mtp_tensors()),
        }
    }
}

/// The parts of the MTP block's stem, in the order [`Call::Stem`] reads them: a part's number is its index.
#[derive(Clone, Copy)]
enum StemPart {
    Table,
    EmbeddingNorm,
    HiddenNorm,
    MtpNorm,
    Projection,
}

/// The stem's tensors, `projection` being the input projection's, in the order of [`StemPart`].
fn stem_names(projection: &str) -> Vec<String> {
    [EMBEDDING, EMBEDDING_NORM, HIDDEN_NORM, MTP_NORM, projection]
        .map(str::to_owned)
        .into()
}

/// The sizes of the tensors each call of a run reads together, for the staging buffer. A tensor the checkpoint lacks
/// (`size` knows no such name) is left out, and its call fails when it reads.
pub(super) fn head_groups(
    trellis: bool,
    names: impl Fn(Call) -> Result<Vec<String>, CalibrateError>,
    size: impl Fn(&str) -> Option<usize>,
) -> Vec<Vec<usize>> {
    Call::ALL
        .into_iter()
        .filter(|call| call.made_by(trellis))
        .map(|call| {
            names(call)
                .unwrap_or_default()
                .iter()
                .filter_map(|name| size(name))
                .collect()
        })
        .collect()
}

/// The sizes of the tensors `names`, in order.
fn sizes_of(inner: &Inner, names: &[String]) -> Result<Vec<usize>, CalibrateError> {
    names
        .iter()
        .map(|name| {
            let tensor = inner
                .objects
                .store()
                .tensor(name)
                .map_err(|error| CalibrateError::failed(name, error))?;
            Ok(tensor.bytes as usize)
        })
        .collect()
}

/// Reads tensors from the checkpoint into ranges of the staging buffer.
struct StoreReader {
    store: Arc<Safetensors>,
}

impl PartReader<PinnedBuffer> for StoreReader {
    fn read(
        &self,
        name: &str,
        index: usize,
        mut lease: HeadMemory,
        then: Box<dyn FnOnce(Result<HeadMemory, CalibrateError>) + Send>,
    ) {
        let Some(range) = lease.range(index) else {
            return then(Err(CalibrateError::Failed(format!(
                "{name} has no part to be read into"
            ))));
        };
        let rest = Arc::new(Handoff::empty());
        rest.send((lease, then));
        let (ended, label) = (Arc::clone(&rest), name.to_owned());
        let started = self.store.read_tensor(
            name,
            range,
            Arc::new(move |result: Result<PinnedRange, CheckpointError>| {
                let Some((lease, then)) = ended.take() else {
                    return;
                };
                match result {
                    // The range is let go of before the lease goes on: the host reads the part only when none is out.
                    Ok(range) => {
                        drop(range);
                        then(Ok(lease));
                    }
                    Err(error) => then(Err(CalibrateError::failed(&label, error))),
                }
            }),
        );
        if let Err(error) = started
            && let Some((_, then)) = rest.take()
        {
            then(Err(CalibrateError::failed(name, error)));
        }
    }
}

/// Reads the tensors `names`, each whole, one after the other, into the parts of the heads' staging buffer.
pub(super) fn read_tensors(inner: &Arc<Inner>, names: Vec<String>, then: Then<HeadMemory>) {
    let cell = single(then);
    let leased = sizes_of(inner, &names).and_then(|sizes| inner.head_staging.lease(&sizes));
    match leased {
        Ok(memory) => {
            let reader = Arc::new(StoreReader {
                store: Arc::clone(inner.objects.store()),
            });
            read_parts(reader, names, 0, memory, Box::new(move |read| end(&cell, read)));
        }
        Err(error) => end(&cell, Err(error)),
    }
}

/// The tensor of the single-piece object `name`.
fn tensor_of(objects: &Qwen38Objects, name: &str) -> Result<(String, u64, usize), CalibrateError> {
    let source = objects
        .object(name)
        .map_err(|error| CalibrateError::failed(name, error))?;
    match source.pieces.as_slice() {
        [piece] if piece.first_row == 0 && piece.rows == source.rows => {
            Ok((piece.tensor.clone(), source.rows, source.columns as usize))
        }
        _ => Err(CalibrateError::Failed(format!("{name} is not one whole tensor"))),
    }
}

/// Rounds one linear on the GPTQ lanes; `then` gets the decoded weight back, in the frame the lanes end in.
///
/// # Safety
/// The weight and the Hessian at the addresses stay live, and unwritten, until `then` is called; the Hessian is
/// complete.
unsafe fn round_one(
    inner: &Arc<Inner>,
    name: &str,
    shape: (u64, usize),
    addresses: (u64, u64),
    decoded: DeviceBuffer,
    then: Then<DeviceBuffer>,
) {
    let Some(lanes) = inner.lanes.take() else {
        return then(Err(CalibrateError::Failed("the GPTQ lanes are in use".to_owned())));
    };
    let cell = Handoff::empty();
    cell.send(decoded);
    let linear = Linear {
        name: name.to_owned(),
        rows: shape.0,
        columns: shape.1,
        weight: addresses.0,
        hessian: addresses.1,
        decoded: cell,
    };
    let back = Arc::clone(inner);
    // SAFETY: the caller's contract is `round_all`'s.
    unsafe {
        round_all(
            Arc::clone(&inner.services),
            lanes,
            vec![linear],
            vec![0],
            Box::new(move |ended| match ended {
                Err(error) => then(Err(error)),
                Ok(Ended { lanes, mut linears }) => {
                    back.lanes.send(lanes);
                    match linears.pop().and_then(|linear| linear.decoded.take()) {
                        Some(decoded) => then(Ok(decoded)),
                        None => then(Err(CalibrateError::Failed("a linear was not rounded".to_owned()))),
                    }
                }
            }),
        );
    }
}

/// What the MTP block's stem is built from, on the card.
struct Stem<'a> {
    inner: &'a Inner,
    /// The final norm's weight and a batch of states after it.
    norm: &'a DeviceBuffer,
    normed: &'a DeviceBuffer,
    /// The embedding table and the next tokens, `[sequences, rows]`.
    table: &'a DeviceBuffer,
    ids: &'a DeviceBuffer,
    embedding_norm: &'a DeviceBuffer,
    hidden_norm: &'a DeviceBuffer,
    /// A batch of rows being normed, and their result.
    scratch: &'a DeviceBuffer,
    scratch_normed: &'a DeviceBuffer,
    /// A batch of stem rows: the next token's embedding, then the hidden state, each normed.
    stem: &'a DeviceBuffer,
}

impl<'a> Stem<'a> {
    /// The stem over what the state and the MTP block hold; each must be set.
    fn of(
        inner: &'a Inner,
        (norm, normed): (&'a Option<DeviceBuffer>, &'a Option<DeviceBuffer>),
        mtp: &'a MtpBuffers<DeviceBuffer>,
    ) -> Result<Stem<'a>, ForwardError> {
        Ok(Stem {
            inner,
            norm: held(norm, "final norm")?,
            normed: held(normed, "normed states")?,
            table: held(&mtp.table, "embedding table")?,
            ids: held(&mtp.ids, "next tokens")?,
            embedding_norm: held(&mtp.embedding_norm, "embedding norm")?,
            hidden_norm: held(&mtp.hidden_norm, "hidden norm")?,
            scratch: held(&mtp.scratch, "scratch rows")?,
            scratch_normed: held(&mtp.scratch_normed, "normed scratch rows")?,
            stem: held(&mtp.stem, "stem rows")?,
        })
    }
}

impl Stem<'_> {
    fn length(&self) -> usize {
        self.inner.settings.length
    }

    /// The rows of a sequence the MTP block runs over: each position with the token after it.
    fn rows(&self) -> usize {
        self.length() - 1
    }

    /// Builds the stem of the batch in the batch buffer.
    fn build(&self, context: &mut Context<'_, '_>, batch: Batch) -> Result<(), ForwardError> {
        let inner = self.inner;
        let (hidden, epsilon) = (inner.hidden as u32, inner.objects.config.eps as f32);
        let (row_bytes, rows) = (inner.row_bytes() as u64, self.rows());
        let (count, stem_rows) = (batch.count, (batch.count * rows) as u32);
        let ops = context.ops();
        ops.rms_norm_unit_offset_bf16(
            inner.batch_states.ptr(),
            self.norm.ptr(),
            self.normed.ptr(),
            (count * self.length()) as u32,
            hidden,
            epsilon,
        )?;
        ops.copy_device_to_device_2d(
            self.scratch.ptr(),
            rows as u64 * row_bytes,
            self.normed.ptr(),
            self.length() as u64 * row_bytes,
            rows as u64 * row_bytes,
            count as u64,
        )?;
        ops.rms_norm_unit_offset_bf16(
            self.scratch.ptr(),
            self.hidden_norm.ptr(),
            self.scratch_normed.ptr(),
            stem_rows,
            hidden,
            epsilon,
        )?;
        ops.copy_device_to_device_2d(
            self.stem.ptr().offset(row_bytes),
            2 * row_bytes,
            self.scratch_normed.ptr(),
            row_bytes,
            row_bytes,
            u64::from(stem_rows),
        )?;
        context.gather_rows(
            self.table.ptr(),
            self.ids.ptr().offset((batch.first * rows * I32_BYTES) as u64),
            stem_rows,
            hidden,
            self.scratch.ptr(),
        )?;
        let ops = context.ops();
        ops.rms_norm_unit_offset_bf16(
            self.scratch.ptr(),
            self.embedding_norm.ptr(),
            self.scratch_normed.ptr(),
            stem_rows,
            hidden,
            epsilon,
        )?;
        ops.copy_device_to_device_2d(
            self.stem.ptr(),
            2 * row_bytes,
            self.scratch_normed.ptr(),
            row_bytes,
            row_bytes,
            u64::from(stem_rows),
        )?;
        Ok(())
    }
}

/// What the MTP call reads from the host: the tensors and the next tokens.
struct MtpHost {
    memory: HeadMemory,
    ids: Vec<i32>,
}

impl Heads for DeviceHeads {
    fn observe(&self, then: Then<()>) {
        let then = self.cleaned(then);
        let names = match Call::FinalNorm.names(&self.inner.objects) {
            Ok(names) => names,
            Err(error) => return then(Err(error)),
        };
        let next = Arc::clone(&self.inner);
        read_tensors(
            &self.inner,
            names,
            Box::new(move |read| match read {
                Ok(read) => observe_states(next, read, then),
                Err(error) => then(Err(error)),
            }),
        );
    }

    fn mtp(&self, then: Then<Vec<String>>) {
        let then = self.cleaned(then);
        let inner = &self.inner;
        let names = match Call::Stem.names(&inner.objects) {
            Ok(names) => names,
            Err(error) => return then(Err(error)),
        };
        let next = Arc::clone(inner);
        read_tensors(
            inner,
            names,
            Box::new(move |read| match read {
                Ok(memory) => stem_states(next, memory, then),
                Err(error) => then(Err(error)),
            }),
        );
    }

    fn output_head(&self, then: Then<Vec<String>>) {
        let then = self.cleaned(then);
        let inner = &self.inner;
        let shape = tensor_of(&inner.objects, OUTPUT_HEAD).map(|(_, rows, columns)| (rows, columns));
        let (names, (rows, columns)) = match Call::OutputHead
            .names(&inner.objects)
            .and_then(|names| Ok((names, shape?)))
        {
            Ok(found) => found,
            Err(error) => return then(Err(error)),
        };
        let next = Arc::clone(inner);
        read_tensors(
            inner,
            names,
            Box::new(move |read| match read {
                Ok(memory) => round_output_head(next, memory, (rows, columns), then),
                Err(error) => then(Err(error)),
            }),
        );
    }

    fn draft_head(&self, hessian: DraftHessian, then: Then<Vec<String>>) {
        round_draft_head(Arc::clone(&self.inner), hessian, self.cleaned(then));
    }

    fn seal(&self, part: Part, files: Vec<String>, then: Then<()>) {
        Arc::clone(&self.inner.ledger).sealed(part, files, save_manifest(&self.inner), self.cleaned(then));
    }

    fn drop_checkpoint(&self, then: Then<()>) {
        let inner = Arc::clone(&self.inner);
        drop_resume_file(
            &self.inner.ledger,
            save_manifest(&self.inner),
            move |unlisted| remove_resume_file(&inner, unlisted),
            then,
        );
    }
}

/// Removes the resume file, which the manifest no longer lists.
fn remove_resume_file(inner: &Arc<Inner>, then: Then<()>) {
    let Some(file) = inner.resume.take() else {
        return then(Err(CalibrateError::Failed("the resume file is not open".to_owned())));
    };
    let ended = Arc::new(Handoff::empty());
    ended.send(then);
    let reported = Arc::clone(&ended);
    let started = file.remove(Box::new(move |removed| {
        if let Some(then) = reported.take() {
            then(removed.map_err(|error| CalibrateError::failed("removing the resume file", error)));
        }
        idle()
    }));
    if let Err(error) = started
        && let Some(then) = ended.take()
    {
        then(Err(CalibrateError::failed("removing the resume file", error)));
    }
}

/// The Hessian of the final norm's output over every sequence, for the output head.
fn observe_states(inner: Arc<Inner>, read: HeadMemory, then: Then<()>) {
    let (hidden, epsilon) = (inner.hidden, inner.objects.config.eps as f32);
    let length = inner.settings.length;
    let rows_of = |count: usize| (count * length) as u32;
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "the output head's Hessian",
        read,
        move |queue, handles, read| {
            let inner = &*working;
            with_state(inner, |state| {
                let normed_bytes = inner.settings.batch * length * inner.row_bytes();
                {
                    let mut context = lane(inner, queue, handles);
                    // Each buffer is the state's the moment it exists: a failure after it frees it in stream order.
                    // SAFETY: `read` is kept until the lane has retired this submission.
                    state.norm = Some(unsafe { upload_part(inner, &mut context, read, 0) }?);
                    state.normed = Some(context.ops().alloc_async(normed_bytes)?);
                    state.hessian = Some(zeroed(&mut context, FP32_BYTES * hidden * hidden)?);
                }
                let norm = held(&state.norm, "final norm")?.ptr();
                let normed = held(&state.normed, "normed states")?.ptr();
                let hessian = held(&state.hessian, "output head Hessian")?.ptr();
                walk(
                    inner,
                    queue,
                    handles,
                    Layout {
                        rows: length,
                        write: None,
                    },
                    |context, batch| {
                        let rows = rows_of(batch.count);
                        context.ops().rms_norm_unit_offset_bf16(
                            inner.batch_states.ptr(),
                            norm,
                            normed,
                            rows,
                            hidden as u32,
                            epsilon,
                        )?;
                        context.accumulate_hessian(hessian, normed, rows, hidden as u32)
                    },
                )?;
                lane(inner, queue, handles).symmetrize(hessian, hidden as u32)
            })
        },
        move |made| then(made.map(|((), read)| drop(read))),
    );
}

/// The MTP block, first step: the stem's Hessian over every sequence, the projection's decoded weight allocated.
fn stem_states(inner: Arc<Inner>, memory: HeadMemory, then: Then<Vec<String>>) {
    let (hidden, length) = (inner.hidden, inner.settings.length);
    let Some(rows) = length.checked_sub(1).filter(|rows| *rows > 0) else {
        return then(Err(CalibrateError::Invalid(
            "the MTP block needs sequences of at least two tokens".to_owned(),
        )));
    };
    let sequences = inner.settings.sequences();
    let ids: Vec<i32> = (0..sequences)
        .flat_map(|sequence| {
            let first = sequence * length + 1;
            inner.settings.tokens[first..first + rows].iter().copied()
        })
        .collect();
    let host = MtpHost { memory, ids };
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "the MTP block's stem",
        host,
        move |queue, handles, host| {
            let inner = &*working;
            let batch_rows = inner.settings.batch * rows;
            let row_bytes = inner.row_bytes();
            with_state(inner, |state| {
                {
                    let mut context = lane(inner, queue, handles);
                    let mtp = &mut state.mtp;
                    // Each buffer is the state's the moment it exists: a failure after it frees it in stream order.
                    // SAFETY: the tensors are kept until the lane has retired this submission.
                    unsafe {
                        mtp.table = Some(upload_part(
                            inner,
                            &mut context,
                            &host.memory,
                            StemPart::Table as usize,
                        )?);
                        mtp.embedding_norm = Some(upload_part(
                            inner,
                            &mut context,
                            &host.memory,
                            StemPart::EmbeddingNorm as usize,
                        )?);
                        mtp.hidden_norm = Some(upload_part(
                            inner,
                            &mut context,
                            &host.memory,
                            StemPart::HiddenNorm as usize,
                        )?);
                        state.mtp_norm = Some(upload_part(
                            inner,
                            &mut context,
                            &host.memory,
                            StemPart::MtpNorm as usize,
                        )?);
                        mtp.weight = Some(upload_part(
                            inner,
                            &mut context,
                            &host.memory,
                            StemPart::Projection as usize,
                        )?);
                    }
                    mtp.decoded = Some(context.ops().alloc_async(BF16_BYTES * hidden * 2 * hidden)?);
                    mtp.ids = Some(context.ops().alloc_async(I32_BYTES * host.ids.len())?);
                    mtp.scratch = Some(context.ops().alloc_async(batch_rows * row_bytes)?);
                    mtp.scratch_normed = Some(context.ops().alloc_async(batch_rows * row_bytes)?);
                    mtp.stem = Some(context.ops().alloc_async(2 * batch_rows * row_bytes)?);
                    mtp.hessian = Some(zeroed(&mut context, FP32_BYTES * 4 * hidden * hidden)?);
                }
                let ids = held(&state.mtp.ids, "next tokens")?.addr();
                // SAFETY: the ids are kept until the lane has retired this submission.
                unsafe {
                    queue.copy_to_device(ids, host.ids.as_ptr().cast(), I32_BYTES * host.ids.len())?;
                }
                let stem = Stem::of(inner, (&state.norm, &state.normed), &state.mtp)?;
                let (stem_rows, hessian) = (stem.stem.ptr(), held(&state.mtp.hessian, "stem Hessian")?.ptr());
                walk(
                    inner,
                    queue,
                    handles,
                    Layout {
                        rows: length,
                        write: None,
                    },
                    |context, batch| {
                        stem.build(context, batch)?;
                        context.accumulate_hessian(hessian, stem_rows, (batch.count * rows) as u32, 2 * hidden as u32)
                    },
                )?;
                lane(inner, queue, handles).symmetrize(hessian, 2 * hidden as u32)
            })
        },
        move |made| match made {
            Err(error) => then(Err(error)),
            Ok(((), host)) => {
                drop(host);
                round_projection(inner, then);
            }
        },
    );
}

/// The MTP block, second step: the input projection rounded under the stem's Hessian.
fn round_projection(inner: Arc<Inner>, then: Then<Vec<String>>) {
    let hidden = inner.hidden;
    let taken = with_state(&inner, |state| {
        let weight = held(&state.mtp.weight, "projection weight")?.addr();
        let hessian = held(&state.mtp.hessian, "stem Hessian")?.addr();
        let decoded = state
            .mtp
            .decoded
            .take()
            .ok_or_else(|| ForwardError::Invalid("the projection has no decoded weight".to_owned()))?;
        Ok(((weight, hessian), decoded))
    });
    let (addresses, decoded) = match taken {
        Ok(taken) => taken,
        Err(error) => return then(Err(CalibrateError::failed("the MTP block's projection", error))),
    };
    let next = Arc::clone(&inner);
    // SAFETY: the weight and the Hessian are in the heads' state, which holds them until the rounding has ended.
    unsafe {
        round_one(
            &inner,
            INPUT_PROJECTION,
            (hidden as u64, 2 * hidden),
            addresses,
            decoded,
            Box::new(move |rounded| match rounded {
                Ok(decoded) => {
                    let kept = with_state(&next, |state| {
                        state.mtp.decoded = Some(decoded);
                        Ok(())
                    });
                    match kept {
                        Ok(()) => project_states(next, then),
                        Err(error) => then(Err(CalibrateError::failed("the MTP block's projection", error))),
                    }
                }
                Err(error) => then(Err(error)),
            }),
        );
    }
}

/// The MTP block, third step: the rounded projection of every stem replaces the states, `rows` per sequence.
fn project_states(inner: Arc<Inner>, then: Then<Vec<String>>) {
    let (hidden, length) = (inner.hidden, inner.settings.length);
    let rows = length - 1;
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "the MTP block's projection",
        (),
        move |queue, handles, ()| {
            let inner = &*working;
            with_state(inner, |state| {
                let decoded = held(&state.mtp.decoded, "decoded projection")?.ptr();
                let stem = Stem::of(inner, (&state.norm, &state.normed), &state.mtp)?;
                let stem_rows = stem.stem.ptr();
                let layout = Layout {
                    rows: length,
                    write: Some(rows),
                };
                walk(inner, queue, handles, layout, |context, batch| {
                    stem.build(context, batch)?;
                    context.linear(
                        stem_rows,
                        decoded,
                        inner.batch_states.ptr(),
                        (batch.count * rows) as u32,
                        2 * hidden as u32,
                        hidden as u32,
                    )
                })
            })
        },
        move |made| match made {
            Err(error) => then(Err(error)),
            Ok(((), ())) => release_projection(inner, then),
        },
    );
}

/// Frees what only the projection needed, then runs the MTP layer over the states it made.
fn release_projection(inner: Arc<Inner>, then: Then<Vec<String>>) {
    let next = Arc::clone(&inner);
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "freeing the MTP block's projection",
        (),
        move |queue, handles, ()| {
            with_state(&working, |state| {
                let buffers = take(&mut state.mtp).every();
                free(&mut lane(&working, queue, handles), buffers)
            })
        },
        move |freed| match freed {
            Ok(((), ())) => run_mtp_layer(next, then),
            Err(error) => then(Err(error)),
        },
    );
}

/// The MTP block, fourth step: the layer rounded and run over the projection's states, with `rows` per sequence.
fn run_mtp_layer(inner: Arc<Inner>, then: Then<Vec<String>>) {
    let length = inner.settings.length;
    let (running, next) = (Arc::clone(&inner), Arc::clone(&inner));
    at_pass_length(
        &inner.pass_length,
        length - 1,
        move |ran| run_target(&running, Target::Mtp, ran),
        Box::new(move |ran| match ran {
            Ok(()) => draft_states(next, then),
            Err(error) => then(Err(error)),
        }),
    );
}

/// The MTP block, last step: the Hessian of the layer's output under the MTP norm, for the draft head.
fn draft_states(inner: Arc<Inner>, then: Then<Vec<String>>) {
    let (hidden, epsilon) = (inner.hidden, inner.objects.config.eps as f32);
    let rows = inner.settings.length - 1;
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "the draft head's Hessian",
        (),
        move |queue, handles, ()| {
            let inner = &*working;
            with_state(inner, |state| {
                let normed = held(&state.normed, "normed states")?.ptr();
                let mtp_norm = held(&state.mtp_norm, "MTP norm")?.ptr();
                state.draft = Some(zeroed(&mut lane(inner, queue, handles), FP32_BYTES * hidden * hidden)?);
                let hessian = held(&state.draft, "draft head Hessian")?.ptr();
                walk(inner, queue, handles, Layout { rows, write: None }, |context, batch| {
                    let count = (batch.count * rows) as u32;
                    context.ops().rms_norm_unit_offset_bf16(
                        inner.batch_states.ptr(),
                        mtp_norm,
                        normed,
                        count,
                        hidden as u32,
                        epsilon,
                    )?;
                    context.accumulate_hessian(hessian, normed, count, hidden as u32)
                })?;
                let mut context = lane(inner, queue, handles);
                context.symmetrize(hessian, hidden as u32)?;
                free(&mut context, state.mtp_norm.take())
            })
        },
        move |made| then(made.map(|((), ())| mtp_files())),
    );
}

/// The file names of the MTP block's objects.
fn mtp_files() -> Vec<String> {
    MTP_OBJECTS.iter().map(|name| QuantState::file_name(name)).collect()
}

/// The output head: its weight uploaded and kept for the draft head, rounded under the head Hessian.
fn round_output_head(inner: Arc<Inner>, read: HeadMemory, shape: (u64, usize), then: Then<Vec<String>>) {
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "uploading the output head",
        read,
        move |queue, handles, read| {
            let inner = &*working;
            with_state(inner, |state| {
                let hessian = held(&state.hessian, "output head Hessian")?.addr();
                let mut context = lane(inner, queue, handles);
                // SAFETY: `read` is kept until the lane has retired this submission.
                let weight = unsafe { upload_part(inner, &mut context, read, 0) }?;
                let weight = state.weight.insert(weight).addr();
                let decoded = context.ops().alloc_async(BF16_BYTES * shape.0 as usize * shape.1)?;
                Ok(((weight, hessian), decoded))
            })
        },
        move |made| match made {
            Err(error) => then(Err(error)),
            Ok(((addresses, decoded), read)) => {
                drop(read);
                let next = Arc::clone(&inner);
                // SAFETY: the weight and the Hessian are in the heads' state, which holds them until the draft head
                // frees them.
                unsafe {
                    round_one(
                        &inner,
                        OUTPUT_HEAD,
                        shape,
                        addresses,
                        decoded,
                        Box::new(move |rounded| match rounded {
                            Ok(decoded) => free_decoded(next, decoded, vec![QuantState::file_name(OUTPUT_HEAD)], then),
                            Err(error) => then(Err(error)),
                        }),
                    );
                }
            }
        },
    );
}

/// Frees a decoded weight, and gives `files`.
fn free_decoded(inner: Arc<Inner>, decoded: DeviceBuffer, files: Vec<String>, then: Then<Vec<String>>) {
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "freeing a decoded weight",
        (),
        move |queue, handles, ()| free(&mut lane(&working, queue, handles), [decoded]),
        move |freed| then(freed.map(|((), ())| files)),
    );
}

/// The draft head: the rows of the output head's weight the draft ids name, rounded under `hessian`; then every buffer
/// of the heads is freed.
fn round_draft_head(inner: Arc<Inner>, hessian: DraftHessian, then: Then<Vec<String>>) {
    let (hidden, ids) = (inner.hidden, inner.settings.draft_ids.clone());
    let rows = ids.len();
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "gathering the draft head",
        ids,
        move |queue, handles, ids| {
            let inner = &*working;
            with_state(inner, |state| {
                let source = held(&state.weight, "output head weight")?.ptr();
                let under = match hessian {
                    DraftHessian::Mtp => held(&state.draft, "draft head Hessian")?,
                    DraftHessian::OutputHead => held(&state.hessian, "output head Hessian")?,
                }
                .addr();
                {
                    let mut context = lane(inner, queue, handles);
                    state.gathered.push(context.ops().alloc_async(I32_BYTES * ids.len())?);
                    state
                        .gathered
                        .push(context.ops().alloc_async(BF16_BYTES * rows * hidden)?);
                }
                let [ids_buffer, weight] = &state.gathered[..] else {
                    return Err(ForwardError::Invalid(
                        "the draft head's buffers are not both made".to_owned(),
                    ));
                };
                // SAFETY: the ids are kept until the lane has retired this submission.
                unsafe {
                    queue.copy_to_device(ids_buffer.addr(), ids.as_ptr().cast(), I32_BYTES * ids.len())?;
                }
                let mut context = lane(inner, queue, handles);
                context.gather_rows(source, ids_buffer.ptr(), rows as u32, hidden as u32, weight.ptr())?;
                let decoded = context.ops().alloc_async(BF16_BYTES * rows * hidden)?;
                Ok(((weight.addr(), under), decoded))
            })
        },
        move |made| match made {
            Err(error) => then(Err(error)),
            Ok((((weight_address, hessian_address), decoded), ids)) => {
                drop(ids);
                let next = Arc::clone(&inner);
                // SAFETY: the gathered weight and the Hessian are in the heads' state, which holds them until the
                // rounding has ended.
                unsafe {
                    round_one(
                        &inner,
                        DRAFT_HEAD,
                        (rows as u64, hidden),
                        (weight_address, hessian_address),
                        decoded,
                        Box::new(move |rounded| match rounded {
                            Ok(decoded) => release_heads(next, decoded, then),
                            Err(error) => then(Err(error)),
                        }),
                    );
                }
            }
        },
    );
}

/// Frees the draft head's decoded weight and every buffer the heads kept, then gives the draft head's file.
fn release_heads(inner: Arc<Inner>, decoded: DeviceBuffer, then: Then<Vec<String>>) {
    let working = Arc::clone(&inner);
    on_main(
        &Arc::clone(&inner),
        "freeing the heads",
        (),
        move |queue, handles, ()| {
            with_state(&working, |state| {
                let mut buffers = state.every();
                buffers.push(decoded);
                free(&mut lane(&working, queue, handles), buffers)
            })
        },
        move |freed| then(freed.map(|((), ())| vec![QuantState::file_name(DRAFT_HEAD)])),
    );
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::super::staging::{HeadStaging, capacity_of};
    use super::*;

    /// The names each call reads in a checkpoint with these tensors; the stem and the output head use their object's
    /// tensor, `projection` and `head`.
    fn names_of(call: Call) -> Result<Vec<String>, CalibrateError> {
        match call {
            Call::Embedding => Ok(vec![EMBEDDING.to_owned()]),
            Call::FinalNorm => Ok(vec![FINAL_NORM.to_owned()]),
            Call::Stem => Ok(stem_names("projection")),
            Call::OutputHead => Ok(vec!["head".to_owned()]),
            Call::TrellisMtp => Ok(mtp_tensors()),
        }
    }

    fn sizes() -> HashMap<String, usize> {
        let mut sizes = HashMap::from([
            (EMBEDDING.to_owned(), 9000),
            (FINAL_NORM.to_owned(), 10),
            (EMBEDDING_NORM.to_owned(), 10),
            (HIDDEN_NORM.to_owned(), 10),
            (MTP_NORM.to_owned(), 10),
            ("projection".to_owned(), 5000),
            ("head".to_owned(), 9000),
        ]);
        sizes.extend(mtp_tensors().into_iter().map(|name| (name, 3000)));
        sizes
    }

    #[test]
    fn every_call_a_run_makes_fits_the_staging_buffer_sized_for_the_run() {
        let sizes = sizes();
        for trellis in [false, true] {
            let groups = head_groups(trellis, names_of, |name| sizes.get(name).copied());
            let capacity = capacity_of(&groups).unwrap();
            let staging = HeadStaging::<Vec<u8>>::open_with(vec![0; capacity], capacity);
            for call in Call::ALL.into_iter().filter(|call| call.made_by(trellis)) {
                let wanted: Vec<usize> = names_of(call).unwrap().iter().map(|name| sizes[name]).collect();
                assert!(
                    staging.lease(&wanted).is_ok(),
                    "{call:?} is refused by the buffer of a {} run",
                    if trellis { "trellis" } else { "GPTQ" }
                );
            }
        }
    }

    #[test]
    fn a_run_is_sized_only_for_the_mtp_calls_it_makes() {
        let sizes = sizes();
        let gptq = head_groups(false, names_of, |name| sizes.get(name).copied());
        let trellis = head_groups(true, names_of, |name| sizes.get(name).copied());
        assert!(gptq.contains(&vec![9000, 10, 10, 10, 5000]), "the stem");
        assert!(
            !gptq.iter().any(|group| group.len() == 8),
            "no trellis MTP in a GPTQ run"
        );
        assert!(trellis.iter().any(|group| group.len() == 8), "the trellis MTP");
        assert!(
            !trellis.contains(&vec![9000, 10, 10, 10, 5000]),
            "no stem in a trellis run"
        );
    }

    #[test]
    fn the_stem_parts_are_numbered_in_the_order_its_names_come() {
        let names = stem_names("projection");
        for (part, name) in [
            (StemPart::Table, EMBEDDING),
            (StemPart::EmbeddingNorm, EMBEDDING_NORM),
            (StemPart::HiddenNorm, HIDDEN_NORM),
            (StemPart::MtpNorm, MTP_NORM),
            (StemPart::Projection, "projection"),
        ] {
            assert_eq!(names[part as usize], name);
        }
    }

    #[test]
    fn a_tensor_the_checkpoint_lacks_is_left_out_of_the_sizing() {
        let groups = head_groups(false, names_of, |name| (name != "head").then_some(7));
        assert!(
            groups.contains(&vec![]),
            "the output head's group is empty, not an error"
        );
    }
}
