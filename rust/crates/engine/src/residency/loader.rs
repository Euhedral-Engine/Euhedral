//! The weight upload: every placement of the plan read from the artifact into pinned staging and copied to its place
//! in the arena, without a thread waiting for either.
//!
//! The plan's copies are cut into pieces (at most `piece_bytes` each) in file order. The pieces are dealt to chains:
//! chain `c` of `K` handles pieces `c`, `c + K`, `c + 2K`, ... A chain has no queue and no lock; it is a handful of
//! frames that hand one another what they own:
//!
//! 1. the read: `AsyncIo::read_into` fills a pinned loan with a piece. The chain starts its next read the moment
//!    this one has ended, so a read of one piece overlaps the check and the copy of the one before;
//! 2. the check, when the piece holds something the format validates (NVFP4 block scales and the global scale), in a
//!    frame of the `host` lake;
//! 3. the copy, in a frame of the `stages` lake: `copy_upload_to_device` on the chain's own transfer lane, then a
//!    marker that `Completions` finds reached;
//! 4. the completion, which returns the loan to the pool.
//!
//! Nothing here owns the end of the load. The chains share one [`Shared`]; the last frame to let go of it drops it,
//! and its `Drop` reports the outcome, wherever that frame ran. A failure sets a flag, so the chains stop starting
//! reads, and the first error is the one reported.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_core::frames::Frame;
use euhedral_graph::{IoDoneOwned, OwnedFile};
use euhedral_inference_cuda::{DeviceBuffer, DevicePtr, PooledPinned, Submission, SubmitMode, ops};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};
use euhedral_inference_settings::Settings;

use crate::error::EngineError;
use crate::frames::{Kind, step};
use crate::residency::planner::{Placement, PlacementSource, ResidencyPlan, nvfp4};
use crate::runtime::{Core, raw_file};

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

/// One read-and-copy: `len` bytes at `file_offset` of the artifact go to `arena_offset` of the weight arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Piece {
    pub file_offset: u64,
    pub len: usize,
    pub arena_offset: u64,
    pub check: Check,
    /// The placement the bytes belong to (an index into the plan's `device`).
    pub placement: usize,
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
                    arena_offset: placement.offset + copy.arena_offset + done,
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
    /// Chains, which is pieces in flight: each has its own transfer lane.
    pub depth: usize,
    /// The longest piece, a power of two.
    pub piece_bytes: u64,
}

/// The smallest piece worth a read and a copy of its own, and the largest one a pinned loan is allocated for.
const PIECE_RANGE: (u64, u64) = (1 << 20, 32 << 20);

impl Ring {
    /// Sizes the ring from the settings and the artifact.
    /// - `depth` is `host.task-parallelism`, the workers that take host work at once, at least 2 and at most the
    ///   lanes a library holds, less one for the lane that zeroes the state.
    /// - `memory.pinned-buffer-pool`, when stated, is the ring's pinned bytes: the pieces are the largest power of
    ///   two that `depth` of them fill. When it is `auto`, a piece is the largest object cut 64 ways (so the
    ///   biggest objects take about 64 reads), rounded down to a power of two within [`PIECE_RANGE`].
    /// - No ring takes more than `pinned_budget` bytes, the host memory the load may pin.
    pub fn sized(settings: &Settings, largest_object: u64, pinned_budget: u64, max_depth: usize) -> Ring {
        let depth = (settings.host.task_parallelism as usize).clamp(2, max_depth.max(2));
        let piece = match settings.memory.pinned_buffer_pool.value() {
            Some(pool) => pool.bytes() / depth as u64,
            None => (largest_object / 64).clamp(PIECE_RANGE.0, PIECE_RANGE.1),
        };
        let capped = piece.min(pinned_budget / depth as u64);
        let piece_bytes = if capped == 0 { 4096 } else { 1 << capped.ilog2() };
        Ring {
            depth,
            piece_bytes: piece_bytes.max(4096),
        }
    }

    /// The pinned bytes the ring holds when every chain has a read and a copy in flight.
    pub fn pinned_bytes(&self) -> u64 {
        2 * self.depth as u64 * self.piece_bytes
    }
}

/// What a finished load reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadStats {
    pub pieces: usize,
    pub bytes: u64,
    pub elapsed: Duration,
}

/// The end of a load: the arena (every byte of it written, on success) and how it went.
#[derive(Debug)]
pub struct LoadOutcome {
    pub arena: DeviceBuffer,
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
    failed: AtomicBool,
    error: OnceLock<EngineError>,
    uploaded: AtomicU64,
    started: Instant,
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
                elapsed: self.started.elapsed(),
            }),
        };
        done(LoadOutcome { arena, result });
    }
}

/// Starts the upload of `pieces` into `arena` along `chains` chains. `done` runs when the last frame of the load lets
/// go, with the arena and the outcome; it must hand its work to a lake, not do it.
pub(crate) fn start(
    core: &Arc<Core>,
    file: Arc<OwnedFile>,
    names: Arc<[String]>,
    pieces: Vec<Piece>,
    chains: usize,
    arena: DeviceBuffer,
    done: impl FnOnce(LoadOutcome) + Send + Sync + 'static,
) {
    let shared = Arc::new(Shared {
        core: Arc::clone(core),
        file,
        names,
        base: arena.ptr(),
        chains,
        failed: AtomicBool::new(false),
        error: OnceLock::new(),
        uploaded: AtomicU64::new(0),
        started: Instant::now(),
        arena: Some(arena),
        done: Some(Box::new(done)),
        pieces: pieces.into(),
    });
    for chain in 0..chains {
        let handle = Arc::clone(&shared);
        // The first reads start in frames of their own, one per chain, so the chains spread over the workers.
        let first = step(Kind::UploadRead, chain, move || issue(handle, chain, chain));
        if core.host.offer(first).is_err() {
            shared.fail(EngineError::Runtime("the host lake is closed".to_string()));
            break;
        }
    }
}

/// Starts the read of piece `index` for `chain`.
fn issue(shared: Arc<Shared>, chain: usize, index: usize) {
    if index >= shared.pieces.len() || shared.has_failed() {
        return;
    }
    let piece = shared.pieces[index];
    let loan = match shared.core.pool.acquire(piece.len) {
        Ok(loan) => loan,
        Err(error) => return shared.fail(EngineError::Cuda(error)),
    };
    let raw = raw_file(&shared.file);
    let next = Arc::clone(&shared);
    let then = Box::new(move |done: IoDoneOwned<PooledPinned>| -> Box<dyn Frame> {
        step(Kind::UploadRead, chain, move || read_done(next, chain, index, done))
    });
    if let Err(error) = shared.core.io.read_into(raw, piece.file_offset, loan, then) {
        shared.fail(io_error(&shared, piece, error));
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

/// A read ended: start the chain's next one, then check and copy this piece.
fn read_done(shared: Arc<Shared>, chain: usize, index: usize, done: IoDoneOwned<PooledPinned>) {
    let piece = shared.pieces[index];
    let loan = match (done.buf, done.result) {
        (Some(loan), Ok(read)) if read == piece.len => loan,
        (Some(_), Ok(read)) => {
            return shared.fail(EngineError::Io {
                what: format!(
                    "reading '{}' at offset {} of the artifact",
                    shared.names[piece.placement], piece.file_offset
                ),
                source: io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("{read} of {} bytes arrived", piece.len),
                ),
            });
        }
        (_, Err(error)) => return shared.fail(io_error(&shared, piece, error)),
        (None, Ok(_)) => {
            return shared.fail(EngineError::Runtime(
                "the I/O ring failed while a read may still have been writing: the staging was abandoned".to_string(),
            ));
        }
    };
    // The chain's next read starts now, beside this piece's check and copy.
    issue(Arc::clone(&shared), chain, index + shared.chains);
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

/// Queues the copy of a piece on the chain's lane, and a continuation for when it has run.
fn copy(shared: Arc<Shared>, chain: usize, index: usize, loan: PooledPinned) {
    let stages = Arc::clone(&shared.core.stages);
    let frame = step(Kind::UploadCopy, chain, move || {
        let piece = shared.pieces[index];
        let lane = &shared.core.transfer[chain];
        let queued = Submission::begin(lane, SubmitMode::default()).and_then(|submission| {
            let destination = shared.base.offset(piece.arena_offset);
            // SAFETY: the loan stays alive, and its bytes unmodified, in the frame the completion registers below,
            // which runs only once the lane has reached the marker, after the copy has run.
            unsafe { ops::copy_upload_to_device(&submission, destination, loan.buffer(), 0, piece.len) }?;
            submission.finish_signalled()
        });
        let marker = match queued {
            Ok(marker) => marker,
            Err(error) => {
                shared.fail(EngineError::Cuda(error));
                return;
            }
        };
        let completions = Arc::clone(&shared.core.completions);
        let done = step(Kind::UploadDone, chain, move || {
            shared.uploaded.fetch_add(piece.len as u64, Ordering::AcqRel);
            drop(loan);
        });
        completions.after_marker(marker, done);
    });
    // A refused frame is dropped with this chain's handle on the load, which then ends as failed.
    let _ = stages.offer(frame);
}
