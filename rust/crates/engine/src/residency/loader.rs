//! The weight upload: every placement of the plan read from the artifact into pinned staging and copied to its place
//! in the arena, without a thread waiting for either.
//!
//! The plan's copies are cut into pieces (at most `piece_bytes` each) in file order. The pieces are dealt to chains:
//! chain `c` of `K` handles pieces `c`, `c + K`, `c + 2K`, ... A chain has no queue and no lock; it is a handful of
//! frames that hand one another what they own:
//!
//! 1. the read: `AsyncIo::read_into` fills a pinned loan with a piece. A chain owns exactly two loans, acquired before
//!    the upload starts ([`acquire_loans`]): it reads piece `j` into one while piece `j - 1` is checked and copied
//!    from the other, and the pool allocates nothing while the chains run;
//! 2. the check, when the piece holds something the format validates (NVFP4 block scales and the global scale), in a
//!    frame of the `host` lake;
//! 3. the copy, in a frame of the `stages` lake: `copy_upload_to_device` on the chain's own transfer lane, then a
//!    marker that `Completions` finds reached;
//! 4. the completion, which hands the loan to the read of the chain's piece two ahead.
//!
//! A piece of an object the plan holds in pinned host memory has no copy: its read lands straight in the object's range
//! of the host arena ([`Target::Host`]), where kernels read it in place, and the same check runs on those bytes. The
//! chain's loan goes along unused, to the read two pieces ahead.
//!
//! Nothing here owns the end of the load. The chains share one [`Shared`]; the last frame to let go of it drops it,
//! and its `Drop` reports the outcome, wherever that frame ran. A failure sets a flag, so the chains stop starting
//! reads, and the first error is the one reported.

use std::io;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_gpu::Lane;
use euhedral_gpu::cuda::CudaQueue;
use euhedral_graph::{Completions, DeviceQueue, LaneId, Marker};
use euhedral_inference_cuda::{ArenaRange, DeviceBuffer, DevicePtr, HostWeightsArena, PooledPinned};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};
use euhedral_io::{IoDoneOwned, OwnedFile};

use crate::error::EngineError;
use crate::frames::{Kind, step};
use crate::residency::planner::{Placement, PlacementSource, ResidencyPlan, Tier, nvfp4};
use crate::runtime::{Core, raw_file};
use crate::stages::SubmitError;

/// What a piece's bytes must satisfy before they are copied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Check {
    /// Nothing: the bytes are payload the format cannot validate.
    None,
    /// The E4M3 block scales of an NVFP4 plane (`from` bytes into the plane): none is NaN (code 0x7F or 0xFF).
    Nvfp4Scales { from: u64 },
    /// An NVFP4 trailer: for an SD4 layout the 16-entry scale table (no NaN), then the global scale, a finite
    /// non-negative little-endian FP32.
    Nvfp4Trailer { sd4: bool },
}

/// Where a piece's bytes go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// To `arena_offset` of the device weight arena, through a staging loan and a copy.
    Device { arena_offset: u64 },
    /// To `arena_offset` of the pinned host arena, read in place: no copy follows.
    Host { arena_offset: u64 },
}

/// One read: `len` bytes at `file_offset` of the artifact go to `target`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub file_offset: u64,
    pub len: usize,
    pub target: Target,
    pub check: Check,
    /// The placement the bytes belong to (an index into the plan's `device`).
    pub placement: usize,
}

impl Piece {
    /// Where the piece starts in the arena it goes to, the device's or the host's.
    pub fn arena_offset(&self) -> u64 {
        match self.target {
            Target::Device { arena_offset } | Target::Host { arena_offset } => arena_offset,
        }
    }
}

/// A copy before it is cut into pieces.
struct Copy {
    file_offset: u64,
    arena_offset: u64,
    len: u64,
    check: Check,
}

/// The pieces of a plan, at most `piece_bytes` long, in file order.
pub fn pieces(plan: &ResidencyPlan, piece_bytes: u64) -> Vec<Piece> {
    let mut out = Vec::new();
    for (index, placement) in plan.device.iter().enumerate() {
        // An object left in the artifact file is read when it is used, not loaded.
        let base = match placement.tier {
            Tier::Device => placement.offset,
            Tier::Host { offset } => offset,
            Tier::Disk => continue,
        };
        for copy in copies(placement) {
            let mut done = 0;
            while done < copy.len {
                let len = (copy.len - done).min(piece_bytes);
                let check = match copy.check {
                    Check::Nvfp4Scales { from } => Check::Nvfp4Scales { from: from + done },
                    other => other,
                };
                out.push(Piece {
                    file_offset: copy.file_offset + done,
                    len: len as usize,
                    target: {
                        let arena_offset = base + copy.arena_offset + done;
                        match placement.tier {
                            Tier::Host { .. } => Target::Host { arena_offset },
                            _ => Target::Device { arena_offset },
                        }
                    },
                    check,
                    placement: index,
                });
                done += len;
            }
        }
    }
    out.sort_by_key(|piece| piece.file_offset);
    out
}

/// The copies that fill one placement: an object whole (an NVFP4 object in the regions its format validates), or the
/// rows of a packed projection region by region.
fn copies(placement: &Placement) -> Vec<Copy> {
    let nvfp4_shape = match (placement.format, placement.shape.as_slice()) {
        (WeightFormat::Nvfp4, [rows, k]) => Some((*rows, *k)),
        _ => None,
    };
    let sd4 = placement.layout == WeightLayout::RowSplitK128Sd4V1;
    match (&placement.source, nvfp4_shape) {
        (PlacementSource::Object, Some((rows, k))) => {
            let (Some(scales), Some(trailer), Some(row_scale)) = (
                Some(nvfp4::scale_offset(rows, k)),
                nvfp4::trailer_offset(rows, k, placement.layout),
                nvfp4::row_scale_bytes(k, placement.layout),
            ) else {
                return whole(placement);
            };
            let scales_end = scales + rows * row_scale;
            let region = |from: u64, to: u64, check: Check| Copy {
                file_offset: placement.file_offset + from,
                arena_offset: from,
                len: to - from,
                check,
            };
            let plane_check = if sd4 {
                Check::None
            } else {
                Check::Nvfp4Scales { from: 0 }
            };
            let mut copies = vec![region(0, scales, Check::None), region(scales, scales_end, plane_check)];
            if scales_end < trailer {
                copies.push(region(scales_end, trailer, Check::None));
            }
            copies.push(region(trailer, placement.bytes, Check::Nvfp4Trailer { sd4 }));
            copies
        }
        (
            PlacementSource::Rows {
                first_row, object_rows, ..
            },
            Some((rows, k)),
        ) => {
            // The halves of a packed projection are cut where the row-split layouts cut rows: a row's codes, a row's
            // scales, and the one trailer they share (`Nvfp4Format.rows`).
            let layout = placement.layout;
            let (Some(row_scale), Some(trailer_source), Some(trailer_here), Some(trailer_bytes)) = (
                nvfp4::row_scale_bytes(k, layout),
                nvfp4::trailer_offset(*object_rows, k, layout),
                nvfp4::trailer_offset(rows, k, layout),
                nvfp4::trailer_bytes(layout),
            ) else {
                return Vec::new();
            };
            let code = nvfp4::row_code_bytes(k);
            let scales_source = nvfp4::scale_offset(*object_rows, k);
            let scales_here = nvfp4::scale_offset(rows, k);
            let plane_check = if sd4 {
                Check::None
            } else {
                Check::Nvfp4Scales { from: 0 }
            };
            let base = placement.file_offset;
            vec![
                Copy {
                    file_offset: base + first_row * code,
                    arena_offset: 0,
                    len: rows * code,
                    check: Check::None,
                },
                Copy {
                    file_offset: base + scales_source + first_row * row_scale,
                    arena_offset: scales_here,
                    len: rows * row_scale,
                    check: plane_check,
                },
                Copy {
                    file_offset: base + trailer_source,
                    arena_offset: trailer_here,
                    len: trailer_bytes,
                    check: Check::Nvfp4Trailer { sd4 },
                },
            ]
        }
        _ => whole(placement),
    }
}

fn whole(placement: &Placement) -> Vec<Copy> {
    vec![Copy {
        file_offset: placement.file_offset,
        arena_offset: 0,
        len: placement.bytes,
        check: Check::None,
    }]
}

/// Applies `check` to a piece's bytes.
pub fn check_piece(check: Check, bytes: &[u8]) -> Result<(), String> {
    match check {
        Check::None => Ok(()),
        Check::Nvfp4Scales { from } => match bytes.iter().position(|byte| byte & 0x7f == 0x7f) {
            Some(at) => Err(format!("NVFP4 block scale at byte {} is NaN", from + at as u64)),
            None => Ok(()),
        },
        Check::Nvfp4Trailer { sd4 } => {
            let table = if sd4 { nvfp4::TABLE as usize } else { 0 };
            if bytes.len() != table + 4 {
                return Err(format!(
                    "NVFP4 trailer of {} bytes, expected {}",
                    bytes.len(),
                    table + 4
                ));
            }
            if let Some(at) = bytes[..table].iter().position(|byte| byte & 0x7f == 0x7f) {
                return Err(format!("NVFP4 scale table entry {at} is NaN"));
            }
            let global = f32::from_le_bytes([bytes[table], bytes[table + 1], bytes[table + 2], bytes[table + 3]]);
            if !global.is_finite() || global < 0.0 {
                return Err(format!(
                    "NVFP4 global scale is not a finite non-negative number: {global}"
                ));
            }
            Ok(())
        }
    }
}

/// How the staging ring is sized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    /// Chains: each has its own transfer lane and exactly two loans, one being read into while the other is copied.
    pub depth: usize,
    /// The longest piece, a power of two: the size of a loan.
    pub piece_bytes: u64,
}

/// The granularity of the copies: no piece is shorter than one read piece of `AsyncIo` (256 KiB), and none longer
/// than 32 MiB, where a longer copy gains nothing.
const PIECE_RANGE: (u64, u64) = (256 << 10, 32 << 20);

impl Ring {
    /// Sizes the ring.
    /// - `depth` is `task_parallelism` (`host.task-parallelism`, the workers that take host work at once), at least 2
    ///   and at most `max_depth`, the transfer lanes the library can give.
    /// - The ring is two loans per chain, so a piece is `budget / (2 x depth)`, rounded down to a power of two within
    ///   [`PIECE_RANGE`]. `budget` is the pinned pool the settings resolved (`memory.pinned-buffer-pool`), at most
    ///   `pinned_budget`, the host memory the load may pin.
    pub fn sized(task_parallelism: u32, pool_bytes: u64, pinned_budget: u64, max_depth: usize) -> Ring {
        let depth = (task_parallelism as usize).clamp(2, max_depth.max(2));
        let budget = pool_bytes.min(pinned_budget);
        let piece = (budget / (2 * depth as u64)).clamp(PIECE_RANGE.0, PIECE_RANGE.1);
        Ring {
            depth,
            piece_bytes: 1 << piece.ilog2(),
        }
    }

    /// The pinned bytes of the ring: two loans per chain.
    pub fn pinned_bytes(&self) -> u64 {
        2 * self.depth as u64 * self.piece_bytes
    }
}

/// What a finished load reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadStats {
    pub pieces: usize,
    pub bytes: u64,
    /// Bytes read straight into the host arena.
    pub host_bytes: u64,
    pub elapsed: Duration,
    /// Pinned bytes the pool allocated while the chains ran; the ring is acquired before, so this is 0.
    pub pinned_growth: u64,
}

/// The end of a load: the arena (every byte of it written, on success) and how it went.
#[derive(Debug)]
pub struct LoadOutcome {
    pub arena: DeviceBuffer,
    /// The host arena, when the plan holds weights there; every byte of it is written on success.
    pub host: Option<Arc<HostWeightsArena>>,
    pub result: Result<LoadStats, EngineError>,
}

type Done = Box<dyn FnOnce(LoadOutcome) + Send + Sync>;

/// What the chains share. Its last owner reports the outcome when it drops.
struct Shared {
    core: Arc<Core>,
    file: Arc<OwnedFile>,
    pieces: Arc<[Piece]>,
    names: Arc<[String]>,
    chains: usize,
    base: DevicePtr,
    host: Option<Arc<HostWeightsArena>>,
    failed: AtomicBool,
    error: OnceLock<EngineError>,
    uploaded: AtomicU64,
    read_in_place: AtomicU64,
    started: Instant,
    pinned_at_start: u64,
    arena: Option<DeviceBuffer>,
    done: Option<Done>,
}

impl Shared {
    fn fail(&self, error: EngineError) {
        self.failed.store(true, Ordering::Release);
        let _ = self.error.set(error);
    }

    fn has_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        let (Some(arena), Some(done)) = (self.arena.take(), self.done.take()) else {
            return;
        };
        let result = match self.error.take() {
            Some(error) => Err(error),
            None => Ok(LoadStats {
                pieces: self.pieces.len(),
                bytes: self.uploaded.load(Ordering::Acquire),
                host_bytes: self.read_in_place.load(Ordering::Acquire),
                elapsed: self.started.elapsed(),
                pinned_growth: self.core.pool.allocated_bytes().saturating_sub(self.pinned_at_start),
            }),
        };
        done(LoadOutcome {
            arena,
            host: self.host.take(),
            result,
        });
    }
}

/// Acquires the ring before the upload starts: two loans of the ring's piece size for each chain. This is the only
/// pinned allocation of the upload; once it runs, no chain asks the pool for memory. The waiting allocation is the
/// startup's.
pub(crate) fn acquire_loans(core: &Core, ring: &Ring) -> Result<Vec<PooledPinned>, EngineError> {
    (0..2 * ring.depth)
        .map(|_| core.pool.acquire(ring.piece_bytes as usize).map_err(EngineError::from))
        .collect()
}

/// Starts the upload of `pieces` into `arena` (and the pieces of the host objects into `host`) along `chains` chains
/// with their `loans` (two each, from [`acquire_loans`]). `done` runs when the last frame of the load lets go, with the arena and the outcome; it must
/// hand its work to a lake, not do it.
#[allow(
    clippy::too_many_arguments,
    reason = "the startup's state, moved from one frame to the next"
)]
pub(crate) fn start(
    core: &Arc<Core>,
    file: Arc<OwnedFile>,
    names: Arc<[String]>,
    pieces: Vec<Piece>,
    chains: usize,
    arena: DeviceBuffer,
    host: Option<Arc<HostWeightsArena>>,
    loans: Vec<PooledPinned>,
    done: impl FnOnce(LoadOutcome) + Send + Sync + 'static,
) {
    let shared = Arc::new(Shared {
        core: Arc::clone(core),
        file,
        names,
        base: arena.ptr(),
        host,
        chains,
        failed: AtomicBool::new(false),
        error: OnceLock::new(),
        uploaded: AtomicU64::new(0),
        read_in_place: AtomicU64::new(0),
        started: Instant::now(),
        pinned_at_start: core.pool.allocated_bytes(),
        arena: Some(arena),
        done: Some(Box::new(done)),
        pieces: pieces.into(),
    });
    let mut loans = loans.into_iter();
    for chain in 0..chains {
        let (Some(first), Some(second)) = (loans.next(), loans.next()) else {
            shared.fail(EngineError::Runtime(
                "the upload was given fewer loans than it has chains".to_string(),
            ));
            break;
        };
        let handle = Arc::clone(&shared);
        // The first reads start in frames of their own, one per chain, so the chains spread over the workers.
        let frame = step(Kind::UploadRead, chain, move || {
            issue(Arc::clone(&handle), chain, chain, first);
            issue(handle, chain, chain + chains, second);
        });
        if core.host.offer(frame).is_err() {
            shared.fail(EngineError::Runtime("the host lake is closed".to_string()));
            break;
        }
    }
}

/// Starts the read of piece `index` for `chain` into `loan`, or lets the loan go back to the pool when the chain has
/// no piece left (or the load has failed).
fn issue(shared: Arc<Shared>, chain: usize, index: usize, mut loan: PooledPinned) {
    if index >= shared.pieces.len() || shared.has_failed() {
        return;
    }
    let piece = shared.pieces[index];
    if let Target::Host { arena_offset } = piece.target {
        return issue_in_place(shared, chain, index, arena_offset, loan);
    }
    if let Err(error) = loan.set_len(piece.len) {
        return shared.fail(EngineError::Cuda(error));
    }
    let raw = raw_file(&shared.file);
    let next = Arc::clone(&shared);
    let then = Box::new(move |done: IoDoneOwned<PooledPinned>| -> Box<dyn Frame> {
        step(Kind::UploadRead, chain, move || read_done(next, chain, index, done))
    });
    if let Err(error) = shared.core.io.read_into(raw, piece.file_offset, loan, then) {
        shared.fail(io_error(&shared, piece, error));
    }
}

/// Starts the read of a host piece straight into its range of the host arena. The chain's `loan` is not used by it: it
/// waits in the continuation, and goes to the read two pieces ahead when this one has ended.
fn issue_in_place(shared: Arc<Shared>, chain: usize, index: usize, arena_offset: u64, loan: PooledPinned) {
    let piece = shared.pieces[index];
    let range = shared.host.as_ref().and_then(|host| {
        let offset = usize::try_from(arena_offset).ok()?;
        // SAFETY: the plan gives each host object its own 4096-aligned place, and `pieces` cuts an object's copies
        // into pieces that do not overlap, so no other range of the arena is alive over these bytes; nothing else
        // touches the arena while the load runs, and the device reads it only after the load has ended.
        unsafe { host.range(offset, piece.len) }
    });
    let Some(range) = range else {
        return shared.fail(EngineError::Runtime(format!(
            "'{}' has no place in the host arena at offset {arena_offset}",
            shared.names[piece.placement]
        )));
    };
    let raw = raw_file(&shared.file);
    let next = Arc::clone(&shared);
    let then = Box::new(move |done: IoDoneOwned<ArenaRange>| -> Box<dyn Frame> {
        step(Kind::UploadRead, chain, move || {
            in_place_done(next, chain, index, loan, done)
        })
    });
    if let Err(error) = shared.core.io.read_into(raw, piece.file_offset, range, then) {
        shared.fail(io_error(&shared, piece, error));
    }
}

/// A host piece's read ended: check the bytes where they landed, when its format asks for it, and hand the loan on.
fn in_place_done(shared: Arc<Shared>, chain: usize, index: usize, loan: PooledPinned, done: IoDoneOwned<ArenaRange>) {
    let piece = shared.pieces[index];
    let range = match (done.buf, done.result) {
        (Some(range), Ok(read)) if read == piece.len => range,
        (Some(_), Ok(read)) => {
            return shared.fail(short_read(&shared, piece, read));
        }
        (_, Err(error)) => return shared.fail(io_error(&shared, piece, error)),
        (None, Ok(_)) => {
            return shared.fail(EngineError::Runtime(
                "the I/O ring failed while a read may still have been writing: the host arena was abandoned"
                    .to_string(),
            ));
        }
    };
    let finish = move |shared: Arc<Shared>, loan: PooledPinned| {
        shared.read_in_place.fetch_add(piece.len as u64, Ordering::AcqRel);
        let next = index + 2 * shared.chains;
        issue(shared, chain, next, loan);
    };
    if piece.check == Check::None {
        return finish(shared, loan);
    }
    let host = Arc::clone(&shared.core);
    let frame = step(Kind::UploadCheck, chain, move || {
        match check_piece(piece.check, range.as_slice()) {
            Ok(()) => finish(shared, loan),
            Err(reason) => {
                let name = shared.names[piece.placement].clone();
                shared.fail(EngineError::Payload { name, reason });
            }
        }
    });
    // A refused frame is dropped with the chain's handle on the load; the load then ends as failed or never started.
    let _ = host.host.offer(frame);
}

fn short_read(shared: &Shared, piece: Piece, read: usize) -> EngineError {
    EngineError::Io {
        what: format!(
            "reading '{}' at offset {} of the artifact",
            shared.names[piece.placement], piece.file_offset
        ),
        source: io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!("{read} of {} bytes arrived", piece.len),
        ),
    }
}

fn io_error(shared: &Shared, piece: Piece, source: io::Error) -> EngineError {
    EngineError::Io {
        what: format!(
            "reading {} bytes of '{}' at offset {} of the artifact",
            piece.len, shared.names[piece.placement], piece.file_offset
        ),
        source,
    }
}

/// A read ended: check the piece when its format asks for it, then copy it.
fn read_done(shared: Arc<Shared>, chain: usize, index: usize, done: IoDoneOwned<PooledPinned>) {
    let piece = shared.pieces[index];
    let loan = match (done.buf, done.result) {
        (Some(loan), Ok(read)) if read == piece.len => loan,
        (Some(_), Ok(read)) => return shared.fail(short_read(&shared, piece, read)),
        (_, Err(error)) => return shared.fail(io_error(&shared, piece, error)),
        (None, Ok(_)) => {
            return shared.fail(EngineError::Runtime(
                "the I/O ring failed while a read may still have been writing: the staging was abandoned".to_string(),
            ));
        }
    };
    if piece.check == Check::None {
        copy(shared, chain, index, loan);
        return;
    }
    let host = Arc::clone(&shared.core);
    let frame = step(Kind::UploadCheck, chain, move || {
        let bytes = &loan.buffer().as_slice()[..piece.len];
        match check_piece(piece.check, bytes) {
            Ok(()) => copy(shared, chain, index, loan),
            Err(reason) => {
                let name = shared.names[piece.placement].clone();
                shared.fail(EngineError::Payload { name, reason });
            }
        }
    });
    // A refused frame is dropped with the chain's handle on the load; the load then ends as failed or never started.
    let _ = host.host.offer(frame);
}

/// Queues the copy of a piece on the chain's lane, and a continuation for when it has run. That continuation hands the
/// loan to the chain's next read: the one piece of the chain that follows the piece being copied by two.
fn copy(shared: Arc<Shared>, chain: usize, index: usize, loan: PooledPinned) {
    let stages = Arc::clone(&shared.core.stages);
    let frame = step(Kind::UploadCopy, chain, move || {
        let piece = shared.pieces[index];
        let destination = shared.base.offset(piece.arena_offset());
        let queued = shared
            .core
            .lane_table
            .submit_ops(shared.core.transfer_lane(chain), false, |kernels| {
                // SAFETY: the loan stays alive, and its bytes unmodified, in the frame the completion registers
                // below, which runs only once the lane has reached the marker, after the copy has run.
                unsafe { kernels.copy_upload_to_device(destination, loan.buffer(), 0, piece.len) }
            });
        let marker = match queued {
            Ok(((), marker)) => marker,
            Err(SubmitError::Busy) => {
                // Nothing was queued, so the loan is still the copy's. A full ring does not arise in a normal load
                // (each chain has its own lane and at most two loans against its markers); Busy is an awaiter's
                // one-call slot pin, or a ring that drained meanwhile. The copy is tried again when the lane's oldest
                // marker completes, not at once: re-offering it now could only spin a worker.
                let lane = Arc::clone(shared.core.transfer_lane(chain));
                let completions = Arc::clone(&shared.core.completions);
                let stages = Arc::clone(&shared.core.stages);
                let retry = step(Kind::UploadCopy, chain, move || copy(shared, chain, index, loan));
                retry_or_offer(&*lane, &completions, retry, |frame| {
                    let _ = stages.offer(frame);
                });
                return;
            }
            Err(error) => {
                // The copy may be queued and no marker will say when it ran: the loan must not return to the pool
                // while the device may still read it. It is leaked; the load has failed and ends the startup.
                mem::forget(loan);
                return shared.fail(EngineError::from(error));
            }
        };
        let completions = Arc::clone(&shared.core.completions);
        let chains = shared.chains;
        let done = step(Kind::UploadDone, chain, move || {
            shared.uploaded.fetch_add(piece.len as u64, Ordering::AcqRel);
            issue(shared, chain, index + 2 * chains, loan);
        });
        completions.after_marker(marker, done);
    });
    // A refused frame is dropped with this chain's handle on the load, which then ends as failed.
    let _ = stages.offer(frame);
}

/// A lane the loader may wait on: its queue, and its id in the pool.
trait RetryLane: DeviceQueue {
    fn lane_id(&self) -> LaneId;
}

impl RetryLane for CudaQueue {
    fn lane_id(&self) -> LaneId {
        Lane::lane(self)
    }
}

/// What a copy refused with `Busy` does: `retry` runs once the lane has completed its oldest marker still in flight,
/// which frees a slot of its ring. Nothing waits and nothing is offered meanwhile. When the lane has no marker in
/// flight there is no completion to wait for (something else holds the slot) and `retry` goes to `offer` at once, one
/// offer per call.
fn retry_or_offer(
    lane: &dyn RetryLane,
    completions: &Completions,
    retry: Box<dyn Frame>,
    offer: impl FnOnce(Box<dyn Frame>),
) {
    let oldest = lane.completed() + 1;
    if oldest > lane.issued() {
        return offer(retry);
    }
    completions.after_marker(Marker::new(lane.id(), lane.lane_id(), oldest), retry);
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use euhedral_core::generics::LatticeSource;
    use euhedral_graph::{LanePool, QueueId};

    use super::*;

    /// A lane whose ring holds `slots` markers: it refuses a submission while that many are in flight.
    #[derive(Debug)]
    struct Ring {
        id: QueueId,
        slots: u64,
        issued: AtomicU64,
        completed: AtomicU64,
    }

    impl Ring {
        fn submit(&self) -> bool {
            if self.issued.load(Ordering::Acquire) - self.completed.load(Ordering::Acquire) >= self.slots {
                return false;
            }
            self.issued.fetch_add(1, Ordering::AcqRel);
            true
        }
    }

    impl RetryLane for Ring {
        fn lane_id(&self) -> LaneId {
            LaneId(0)
        }
    }

    impl DeviceQueue for Ring {
        fn id(&self) -> QueueId {
            self.id
        }
        fn completed(&self) -> u64 {
            self.completed.load(Ordering::Acquire)
        }
        fn issued(&self) -> u64 {
            self.issued.load(Ordering::Acquire)
        }
        fn name(&self) -> &str {
            "ring"
        }
    }

    /// The loader's copy protocol over a fake lane: an attempt that the ring refuses waits for a completion.
    struct Copies {
        ring: Arc<Ring>,
        completions: Arc<Completions>,
        attempts: AtomicUsize,
        done: AtomicUsize,
        offered: AtomicUsize,
    }

    fn attempt(copies: Arc<Copies>) {
        copies.attempts.fetch_add(1, Ordering::AcqRel);
        if copies.ring.submit() {
            copies.done.fetch_add(1, Ordering::AcqRel);
            return;
        }
        let again = Arc::clone(&copies);
        let retry = step(Kind::UploadCopy, 0, move || attempt(again));
        retry_or_offer(&*copies.ring, &copies.completions, retry, |_| {
            copies.offered.fetch_add(1, Ordering::AcqRel);
        });
    }

    /// Runs every frame the completions hold ready.
    fn run_ready(completions: &Completions) -> usize {
        let mut ran = 0;
        completions.reap();
        completions.pull(
            &mut |mut frame| {
                frame.execute().expect("the retry runs");
                ran += 1;
            },
            &mut |_| false,
            u64::MAX,
        );
        ran
    }

    #[test]
    fn retry_waits_for_a_completion_and_the_load_finishes() {
        let ring = Arc::new(Ring {
            id: QueueId::next(),
            slots: 2,
            issued: 0.into(),
            completed: 0.into(),
        });
        let queues = Arc::new(LanePool::new(vec![ring.clone()]));
        let completions = Completions::new(&queues);
        let copies = Arc::new(Copies {
            ring: Arc::clone(&ring),
            completions: Arc::clone(&completions),
            attempts: 0.into(),
            done: 0.into(),
            offered: 0.into(),
        });

        // Six copies against a ring of two: two are queued, four are refused and wait.
        for _ in 0..6 {
            attempt(Arc::clone(&copies));
        }
        assert_eq!(copies.done.load(Ordering::Acquire), 2);
        assert_eq!(copies.attempts.load(Ordering::Acquire), 6);
        assert_eq!(completions.pending(), 4, "each refused copy is one registration");

        // No completion: the refused copies are not offered again, however long the workers look.
        for _ in 0..1000 {
            assert_eq!(run_ready(&completions), 0);
        }
        assert_eq!(
            copies.attempts.load(Ordering::Acquire),
            6,
            "no attempt without a completion"
        );

        // The device completes markers one by one: each frees one slot, and the retries behind it may run.
        let mut finished = 0;
        while copies.done.load(Ordering::Acquire) < 6 {
            finished += 1;
            assert!(finished <= 6, "the load is finishing");
            ring.completed.fetch_add(1, Ordering::AcqRel);
            if finished == 1 {
                assert!(
                    run_ready(&completions) > 0,
                    "the retry runs at the first completion, not at the newest marker's"
                );
            }
            while run_ready(&completions) > 0 {}
        }
        assert_eq!(copies.done.load(Ordering::Acquire), 6);
        assert_eq!(
            copies.offered.load(Ordering::Acquire),
            0,
            "markers were in flight: nothing was offered"
        );
        // At most the waiters behind the completed marker retry each time: bounded by waiters per completion.
        assert!(copies.attempts.load(Ordering::Acquire) <= 6 + 4 * 4);
    }

    #[test]
    fn a_lane_with_nothing_in_flight_hands_the_frame_back() {
        let ring = Arc::new(Ring {
            id: QueueId::next(),
            slots: 2,
            issued: 0.into(),
            completed: 0.into(),
        });
        let queues = Arc::new(LanePool::new(vec![ring.clone()]));
        let completions = Completions::new(&queues);
        let retry = step(Kind::UploadCopy, 0, || {});
        let offered = AtomicUsize::new(0);
        retry_or_offer(&*ring, &completions, retry, |_| {
            offered.fetch_add(1, Ordering::AcqRel);
        });
        assert_eq!(offered.load(Ordering::Acquire), 1, "offered once");
        assert_eq!(completions.pending(), 0);
    }
}
