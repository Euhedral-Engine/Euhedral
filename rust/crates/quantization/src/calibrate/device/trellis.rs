//! The trellis arm of the calibration on the card: a layer rounded through the trellis library and written by the exl3
//! writer, and the MTP block and the checkpoint's closing files.
//!
//! A trellis layer is the GPTQ layer's chain with another rounding step:
//! 1. on the main lane, the Hessians are completed (symmetrized, then divided by the rows they were summed over) and
//!    each quantized linear's rows are gathered from the layer's fused objects into a buffer of its own;
//! 2. the linears are rounded on the trellis lanes, largest first, each from its gathered weight and its Hessian;
//! 3. on the main lane, each linear's decoded rows are copied back over the rows they came from, so the write pass runs
//!    the rounded layer, and the gathered weights and the Hessians are freed;
//! 4. the exl3 writer writes the layer's file, which the manifest then lists.
//!
//! The lanes' width is fixed when the device opens, from the memory the largest step of the model takes.

use std::collections::{HashMap, HashSet};
use std::mem::forget;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::{Completions, Lake};
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, DevicePtr, KernelOps, LaneOps, PinnedPool, PooledPinned};
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;

use super::heads::{Context, DeviceHeads, free, lane, on_main, read_tensors, upload_pinned};
use super::{BF16_BYTES, FP32_BYTES, Inner, Settings, lane_handles, write_pass};
use crate::calibrate::CalibrateError;
use crate::calibrate::allocation::Allocation;
use crate::calibrate::driver::{Once, Then, end, single};
use crate::calibrate::heads::{DraftHessian, Heads, TrellisHeads};
use crate::calibrate::hessians::HessianSet;
use crate::calibrate::manifest::{Part, in_heads};
use crate::calibrate::plan::round_order;
use crate::calibrate::rounding::{Services, claims_for};
use crate::calibrate::schedule::{Schedule, Settled};
use crate::calibrate::trellis_plan::{RowCopy, held_bytes, lanes_that_fit, row_copies, written};
use crate::checkpoint::objects::Qwen38Objects;
use crate::exl3::linears::{MTP_LINEARS, layer, placements};
use crate::exl3::{
    Done, Exl3Settings, Exl3Writer, Listing, MTP_BITS, MTP_FILE, Quantized as Written, closing_files, layer_file,
};
use crate::forward::{ForwardError, LayerWeights};
use crate::kernels::Modules;
use crate::memory::{DeviceRange, LaneMemory, StepBytes};
use crate::trellis::{
    Finished, QuantizeError, Quantized, Refused, Request, TrellisKernels, TrellisLane, TrellisSettings,
    plan_normalise_hessian,
};

/// Java's seed of the MTP block's linears.
const MTP_SEED: u64 = 64;
/// The most linears a layer has: full attention's four and the MLP's three.
const MOST_LINEARS: usize = 7;

/// What a trellis run is, besides the settings every calibration has.
#[derive(Clone, Debug)]
pub struct TrellisRun {
    /// The rate of each linear.
    pub allocation: Allocation,
    /// Device bytes of Viterbi history one lane holds.
    pub viterbi_scratch_bytes: u64,
    /// Bytes of the buffer each written file is copied through.
    pub write_buffer_bytes: usize,
    /// Linears rounded at once; none fits as many as the card holds.
    pub rounding_lanes: Option<usize>,
    /// The source checkpoint's directory.
    pub source: PathBuf,
    /// The regular files of the source directory.
    pub source_files: Vec<String>,
}

/// What a trellis run keeps on the card and the host between its calls.
pub(super) struct Trellis {
    kernels: Arc<TrellisKernels<CudaModule>>,
    lanes: Handoff<Vec<TrellisLane>>,
    writer: Handoff<Exl3Writer>,
    bits: HashMap<String, f64>,
    /// The checkpoint's closing files, which the writer writes once the heads are rounded.
    closing: Vec<String>,
    source_files: Vec<String>,
    /// The rows a Hessian is summed over.
    count: u64,
}

/// What opening the trellis lanes needs of the device that is open.
pub(super) struct Opening<'a> {
    pub gpu: &'a Arc<CudaGpu>,
    pub math: &'a Arc<MathLibraries>,
    pub modules: &'a mut Modules<CudaModule>,
    pub library: &'a Arc<CudaLibrary>,
    pub objects: &'a Arc<Qwen38Objects>,
    pub lane_pinned: &'a Arc<PinnedPool>,
    /// Device bytes the planner accounts.
    pub capacity: u64,
    pub settings: &'a Settings,
    pub io: &'a Arc<AsyncIo>,
    pub host: &'a Arc<Lake>,
}

/// The distinct steps the model rounds, as the device sizes them: `(n, k, bits)` and whether a Hessian is used.
type Shapes = HashSet<(u32, u32, u64, bool)>;

fn dimension(value: u64) -> Result<u32, CalibrateError> {
    u32::try_from(value).map_err(|_| CalibrateError::Invalid(format!("{value} does not fit a 32-bit dimension")))
}

/// What the model's linears take of the card: the distinct steps, the layer that holds most weights, and the most
/// linears a layer rounds.
fn steps_of(objects: &Qwen38Objects, run: &TrellisRun) -> Result<(Shapes, u64, usize), CalibrateError> {
    let mut shapes = Shapes::new();
    let (mut held, mut most) = (0, 1);
    for index in 0..objects.config.layers {
        let mut numel = Vec::new();
        for linear in layer(objects, index).map_err(|error| CalibrateError::failed("the layer's linears", error))? {
            if !linear.quantized {
                continue;
            }
            let bits = run
                .allocation
                .rate(&linear.tensor)
                .ok_or_else(|| CalibrateError::Failed(format!("{} has no rate in the allocation", linear.tensor)))?;
            shapes.insert((
                dimension(linear.rows)?,
                dimension(linear.columns)?,
                bits.to_bits(),
                true,
            ));
            numel.push(linear.rows * linear.columns);
        }
        most = most.max(numel.len());
        held = held.max(held_bytes(numel));
    }
    for name in MTP_LINEARS {
        let tensor = objects
            .store()
            .tensor(&format!("{name}.weight"))
            .map_err(|error| CalibrateError::failed(name, error))?;
        let [n, k] = tensor.shape[..] else {
            return Err(CalibrateError::Failed(format!("{name} is not a matrix")));
        };
        shapes.insert((dimension(n)?, dimension(k)?, f64::from(MTP_BITS).to_bits(), false));
    }
    Ok((shapes, held, most.min(MOST_LINEARS)))
}

/// Opens the trellis lanes and the writer. The lane pool stays open for the lanes' life: the queues hold it weakly.
/// Returns the run's state and the completions that deliver its lanes' retirements.
pub(super) fn open(opening: Opening<'_>, run: &TrellisRun) -> Result<(Trellis, Arc<Completions>), CalibrateError> {
    let Opening {
        gpu,
        math,
        modules,
        library,
        objects,
        lane_pinned,
        capacity,
        settings,
        io,
        host,
    } = opening;
    let kernels =
        TrellisLane::kernels(modules).map_err(|error| CalibrateError::failed("the trellis kernels", error))?;
    let (shapes, held, most) = steps_of(objects, run)?;
    let count = run.rounding_lanes.unwrap_or(most).max(1);
    let names: Vec<String> = (0..count).map(|lane| format!("trellis{lane}")).collect();
    let named: Vec<&str> = names.iter().map(String::as_str).collect();
    let (pool, streams) = gpu
        .open_lanes(&named, StreamOptions::default())
        .map_err(|error| CalibrateError::failed("the trellis lanes", error))?;
    let completions = Completions::new(&pool);
    let trellis_settings = TrellisSettings {
        viterbi_scratch_bytes: run.viterbi_scratch_bytes,
    };
    let mut lanes = Vec::new();
    for mut stream in streams {
        let handles = lane_handles(math, &mut stream, "trellis")?;
        let memory = LaneMemory::new(Arc::clone(library), Arc::clone(lane_pinned));
        let lane = TrellisLane::new(
            stream,
            handles,
            Arc::clone(&kernels),
            memory,
            Arc::clone(&completions),
            trellis_settings,
        )
        .map_err(|error| CalibrateError::failed("a trellis lane", error))?;
        lanes.push(lane);
    }
    if run.rounding_lanes.is_none() {
        let first = lanes
            .first_mut()
            .ok_or_else(|| CalibrateError::Failed("no trellis lane opened".to_owned()))?;
        let mut peak = StepBytes { device: 0, pinned: 0 };
        for (n, k, bits, calibrated) in shapes {
            let bytes = first
                .quantize_bytes(n, k, f64::from_bits(bits), calibrated)
                .map_err(|error| CalibrateError::failed("sizing a trellis step", error))?;
            peak = StepBytes {
                device: peak.device.max(bytes.device),
                pinned: peak.pinned.max(bytes.pinned),
            };
        }
        let width = lanes_that_fit(
            (peak.device, peak.pinned),
            (capacity.saturating_sub(held), settings.lane_pinned_bytes),
            lanes.len(),
        )
        .ok_or_else(|| {
            CalibrateError::Failed(format!(
                "the device holds no trellis step: one takes {} device and {} pinned bytes beside the {held} the layer's \
                 weights hold, of {capacity} and {}",
                peak.device, peak.pinned, settings.lane_pinned_bytes
            ))
        })?;
        lanes.truncate(width);
    }
    let writer = Exl3Writer::new(
        Arc::clone(io),
        Arc::clone(host),
        Qwen38Objects::new(Arc::clone(objects.store()), objects.config.clone()),
        &run.source,
        &settings.out,
        Exl3Settings {
            bits: run.allocation.target(),
            rows: settings.sequences() as u64,
            columns: settings.length as u64,
            buffer_bytes: run.write_buffer_bytes,
        },
    );
    let vision = objects.store().names().any(|name| name.starts_with("model.visual."));
    let closing = closing_files(vision, run.source_files.iter().map(String::as_str));
    forget(pool);
    let trellis = Trellis {
        kernels,
        lanes: Handoff::empty(),
        writer: Handoff::empty(),
        bits: run.allocation.by_linear(),
        closing,
        source_files: run.source_files.clone(),
        count: (settings.sequences() * settings.length) as u64,
    };
    trellis.lanes.send(lanes);
    trellis.writer.send(writer);
    Ok((trellis, completions))
}

fn trellis_of(inner: &Inner) -> Result<&Trellis, CalibrateError> {
    inner
        .trellis
        .as_ref()
        .ok_or_else(|| CalibrateError::Failed("the run is not a trellis run".to_owned()))
}

// ---- rounding a set of linears on the lanes ----

/// One linear to round: its tensor, shape and rate, and where its Hessian is (none rounds it without one).
struct Job {
    name: String,
    n: u32,
    k: u32,
    bits: f64,
    hessian: Option<u64>,
}

/// A round in progress, shared by its lanes.
struct Round {
    services: Arc<Services>,
    jobs: Vec<Job>,
    seed: u64,
    count: u64,
    schedule: Schedule<TrellisLane, DeviceBuffer>,
    results: Arc<Vec<Handoff<Quantized>>>,
}

/// What a round gives back: the lanes, the gathered weights, and each job's result, in the jobs' own order.
struct Rounded {
    lanes: Vec<TrellisLane>,
    weights: Vec<DeviceBuffer>,
    results: Vec<Quantized>,
}

/// Rounds `jobs`, each from its BF16 `[n, k]` buffer in `weights`, on `lanes`, largest first.
///
/// # Safety
/// The Hessians the jobs name stay live, and unwritten, until `then` is called; they are complete on the device.
unsafe fn round_jobs(
    services: Arc<Services>,
    lanes: Vec<TrellisLane>,
    (jobs, weights): (Vec<Job>, Vec<DeviceBuffer>),
    (seed, count): (u64, u64),
    then: Then<Rounded>,
) {
    let order = round_order(
        &jobs
            .iter()
            .map(|job| (job.name.clone(), (u64::from(job.n), u64::from(job.k))))
            .collect::<Vec<_>>(),
    );
    let names = jobs.iter().map(|job| job.name.clone()).collect();
    let held = weights
        .into_iter()
        .map(|weight| {
            let held = Handoff::empty();
            held.send(weight);
            held
        })
        .collect();
    let results: Arc<Vec<Handoff<Quantized>>> = Arc::new(jobs.iter().map(|_| Handoff::empty()).collect());
    let taken = Arc::clone(&results);
    let (schedule, started) = Schedule::new(
        names,
        order,
        held,
        lanes.len(),
        Box::new(move |settled| then(settled.and_then(|settled| rounded(&taken, settled)))),
    );
    let round = Arc::new(Round {
        services,
        jobs,
        seed,
        count,
        schedule,
        results,
    });
    for (number, lane) in lanes.into_iter().enumerate() {
        if number < started {
            pull(&round, number, lane);
        } else {
            round.schedule.park(number, lane);
        }
    }
}

fn rounded(
    results: &[Handoff<Quantized>],
    settled: Settled<TrellisLane, DeviceBuffer>,
) -> Result<Rounded, CalibrateError> {
    let weights = settled
        .decoded
        .into_iter()
        .map(|weight| weight.ok_or_else(|| CalibrateError::Failed("a gathered weight was not given back".to_owned())))
        .collect::<Result<Vec<_>, _>>()?;
    let results = results
        .iter()
        .map(|result| {
            result
                .take()
                .ok_or_else(|| CalibrateError::Failed("a linear was not rounded".to_owned()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Rounded {
        lanes: settled.lanes,
        weights,
        results,
    })
}

/// Takes the next linear, or ends the lane.
fn pull(round: &Arc<Round>, number: usize, lane: TrellisLane) {
    round.schedule.pull(number, lane, |index, weight, lane| {
        start(round, (number, index), weight, lane)
    });
}

fn start(round: &Arc<Round>, (number, index): (usize, usize), weight: DeviceBuffer, mut lane: TrellisLane) {
    let job = &round.jobs[index];
    let prepared = lane
        .quantize_bytes(job.n, job.k, job.bits, job.hessian.is_some())
        .map_err(|error| CalibrateError::failed(&job.name, error))
        .and_then(|bytes| claims_for(&round.services, bytes, "trellis"));
    let addr = weight.addr();
    // The weight stays in the round while its lane reads it, and is given back with the others.
    round.schedule.keep(index, weight);
    let claims = match prepared {
        Ok(claims) => claims,
        Err(error) => return round.schedule.abort(number, lane, error),
    };
    // SAFETY: the weight is kept by the round and the Hessian by the caller (the contract of `round_jobs`), both until
    // the round ends, which is after this lane gives itself back.
    let request = unsafe {
        Request {
            weight: DeviceRange::new(addr, 2 * u64::from(job.n) * u64::from(job.k)),
            n: job.n,
            k: job.k,
            hessian: job
                .hessian
                .map(|hessian| DeviceRange::new(hessian, 4 * u64::from(job.k) * u64::from(job.k))),
            count: round.count,
            bits: job.bits,
            seed: round.seed,
            apply_out_scales: true,
        }
    };
    let held = Arc::clone(round);
    let outcome = lane.quantize(&request, claims, move |finished| {
        ended(&held, (number, index), finished)
    });
    if let Err(refused) = outcome {
        let Refused { lane, error } = *refused;
        round
            .schedule
            .abort(number, lane, CalibrateError::failed(&round.jobs[index].name, error));
    }
}

fn ended(
    round: &Arc<Round>,
    (number, index): (usize, usize),
    finished: Finished<TrellisLane, Quantized, QuantizeError>,
) {
    let Finished { lane, scratch, result } = finished;
    drop(scratch);
    match result {
        Ok(quantized) => {
            round.results[index].send(quantized);
            pull(round, number, lane);
        }
        Err(error) => round
            .schedule
            .abort(number, lane, CalibrateError::failed(&round.jobs[index].name, error)),
    }
}

/// The results in the writer's shape, once the work that read their decoded weights has retired: the decoded weights
/// are given back to their lanes, which free them in stream order, and the lanes go back to the run.
fn settle(inner: &Inner, lanes: Vec<TrellisLane>, results: Vec<Quantized>) -> Result<Vec<Written>, CalibrateError> {
    let written_results = results
        .into_iter()
        .map(|result| {
            let Quantized {
                trellis,
                suh,
                svh,
                decoded,
                ..
            } = result;
            drop(decoded);
            written(trellis, suh, svh)
        })
        .collect();
    let mut reclaimed = Ok(());
    let mut returned = Vec::with_capacity(lanes.len());
    for mut lane in lanes {
        if let Err(error) = lane.reclaim() {
            reclaimed = Err(CalibrateError::failed("freeing a trellis lane's buffers", error));
        }
        returned.push(lane);
    }
    trellis_of(inner)?.lanes.send(returned);
    reclaimed.map(|()| written_results)
}

// ---- the writer ----

/// Runs `call` on the writer, which comes back when it ends; `then` gets the writer's outcome.
fn write(inner: &Arc<Inner>, call: impl FnOnce(Exl3Writer, Done), then: Then<()>) {
    let cell = single(then);
    let writer = match trellis_of(inner).and_then(|trellis| {
        trellis
            .writer
            .take()
            .ok_or_else(|| CalibrateError::Failed("the checkpoint writer is in use".to_owned()))
    }) {
        Ok(writer) => writer,
        Err(error) => return end(&cell, Err(error)),
    };
    let back = Arc::clone(inner);
    call(
        writer,
        Arc::new(move |writer, result| {
            if let Some(trellis) = back.trellis.as_ref() {
                trellis.writer.send(writer);
            }
            end(
                &cell,
                result.map_err(|error| CalibrateError::failed("writing the checkpoint", error)),
            );
        }),
    );
}

// ---- a layer ----

/// A piece of a linear's rows, with the object's address.
struct Piece {
    object: DevicePtr,
    copy: RowCopy,
}

/// The jobs of a layer and where each one's rows sit in the layer's objects.
struct Planned {
    jobs: Vec<Job>,
    pieces: Vec<Vec<Piece>>,
}

fn plan_layer(
    inner: &Inner,
    index: usize,
    weights: &LayerWeights,
    set: &HessianSet<DeviceBuffer>,
) -> Result<Planned, CalibrateError> {
    let run = trellis_of(inner)?;
    let (mut jobs, mut pieces) = (Vec::new(), Vec::new());
    let linears = layer(&inner.objects, index).map_err(|error| CalibrateError::failed("the layer's linears", error))?;
    for linear in linears.into_iter().filter(|linear| linear.quantized) {
        let name = &linear.tensor;
        let bits = run
            .bits
            .get(name)
            .copied()
            .ok_or_else(|| CalibrateError::Failed(format!("{name} has no rate in the allocation")))?;
        let hessian = set
            .matrix(&linear.hessian_object)
            .ok_or_else(|| CalibrateError::Failed(format!("{name} was not observed")))?;
        let (n, k) = (dimension(linear.rows)?, dimension(linear.columns)?);
        if hessian.bytes() != FP32_BYTES * k as usize * k as usize {
            return Err(CalibrateError::Failed(format!(
                "{name}'s Hessian has {} bytes, not those of a {k} by {k} matrix",
                hessian.bytes()
            )));
        }
        let placed = placements(&inner.objects, index, name)
            .map_err(|error| CalibrateError::failed(name, error))
            .and_then(|placed| row_copies(&placed, linear.rows, linear.columns))?;
        let located = placed
            .into_iter()
            .map(|copy| {
                let object = weights
                    .object(&copy.object)
                    .ok_or_else(|| CalibrateError::Failed(format!("the layer has no object {}", copy.object)))?;
                Ok(Piece { object, copy })
            })
            .collect::<Result<Vec<_>, CalibrateError>>()?;
        jobs.push(Job {
            name: name.clone(),
            n,
            k,
            bits,
            hessian: Some(hessian.addr()),
        });
        pieces.push(located);
    }
    Ok(Planned { jobs, pieces })
}

/// Rounds the layer `index` whose Hessians are made, and writes its file; then goes on to the write pass.
pub(super) fn round_layer(
    inner: Arc<Inner>,
    index: usize,
    weights: LayerWeights,
    set: HessianSet<DeviceBuffer>,
    then: Once<()>,
) {
    let planned = match plan_layer(&inner, index, &weights, &set) {
        Ok(planned) => planned,
        Err(error) => return end(&then, Err(error)),
    };
    let Planned { jobs, pieces } = planned;
    let sizes: Vec<usize> = jobs
        .iter()
        .map(|job| BF16_BYTES * job.n as usize * job.k as usize)
        .collect();
    let pieces = Arc::new(pieces);
    let (gathering, next) = (Arc::clone(&inner), Arc::clone(&inner));
    let copied = Arc::clone(&pieces);
    on_main(
        &inner,
        "gathering the layer's weights",
        (weights, set),
        move |queue, handles, (_, set)| {
            let kernels = &trellis_of(&gathering)
                .map_err(|error| ForwardError::Invalid(error.to_string()))?
                .kernels;
            let count = trellis_of(&gathering)
                .map_err(|error| ForwardError::Invalid(error.to_string()))?
                .count;
            let mut steps = Vec::new();
            {
                let mut context = lane(&gathering, &mut *queue, &mut *handles);
                for matrix in set.matrices() {
                    let k = (matrix.bytes() / FP32_BYTES).isqrt() as u32;
                    context.symmetrize(matrix.ptr(), k)?;
                }
            }
            for matrix in set.matrices() {
                let k = (matrix.bytes() / FP32_BYTES).isqrt() as u32;
                steps.extend(
                    plan_normalise_hessian(matrix.addr(), k, count)
                        .map_err(|error| ForwardError::Invalid(error.to_string()))?,
                );
            }
            // SAFETY: the Hessians are the layer's own, complete, and nothing else uses them until the round ends.
            unsafe { kernels.enqueue(&mut *queue, &steps) }
                .map_err(|error| ForwardError::Invalid(error.to_string()))?;
            let mut context = lane(&gathering, &mut *queue, &mut *handles);
            gather(&mut context, &sizes, &copied)
        },
        move |gathered| match gathered {
            Ok((gathered, (weights, set))) => {
                let Ok(trellis) = trellis_of(&next) else {
                    return end(
                        &then,
                        Err(CalibrateError::Failed("the run is not a trellis run".to_owned())),
                    );
                };
                let Some(lanes) = trellis.lanes.take() else {
                    return end(
                        &then,
                        Err(CalibrateError::Failed("the trellis lanes are in use".to_owned())),
                    );
                };
                let (services, count) = (Arc::clone(&next.services), trellis.count);
                let after = Arc::clone(&next);
                // SAFETY: the Hessians are in `set`, which the continuation holds until the round has ended.
                unsafe {
                    round_jobs(
                        services,
                        lanes,
                        (jobs, gathered),
                        (index as u64, count),
                        Box::new(move |ended| match ended {
                            Ok(ended) => scatter(after, index, (weights, set), (ended, pieces), then),
                            Err(error) => end(&then, Err(error)),
                        }),
                    );
                }
            }
            Err(error) => end(&then, Err(error)),
        },
    );
}

/// Allocates each linear's gathered weight, of `sizes[i]` bytes, and copies its rows from the objects.
fn gather(
    context: &mut Context<'_, '_>,
    sizes: &[usize],
    pieces: &[Vec<Piece>],
) -> Result<Vec<DeviceBuffer>, ForwardError> {
    let mut gathered = Vec::with_capacity(sizes.len());
    for (bytes, parts) in sizes.iter().zip(pieces) {
        let buffer = context.ops().alloc_async(*bytes)?;
        for piece in parts {
            context.ops().copy_device_to_device(
                buffer.ptr().offset(piece.copy.linear_offset),
                piece.object.offset(piece.copy.object_offset),
                piece.copy.bytes,
            )?;
        }
        gathered.push(buffer);
    }
    Ok(gathered)
}

/// Copies each linear's decoded rows over its rows in the objects, frees the gathered weights and the Hessians, then
/// has the writer write the layer.
fn scatter(
    inner: Arc<Inner>,
    index: usize,
    (weights, set): (LayerWeights, HessianSet<DeviceBuffer>),
    (ended, pieces): (Rounded, Arc<Vec<Vec<Piece>>>),
    then: Once<()>,
) {
    let Rounded {
        lanes,
        weights: gathered,
        results,
    } = ended;
    let working = Arc::clone(&inner);
    let next = Arc::clone(&inner);
    on_main(
        &inner,
        "copying the rounded weights back",
        (weights, results, lanes),
        move |queue, handles, (_, results, _)| {
            let mut context = lane(&working, queue, handles);
            for (result, parts) in results.iter().zip(pieces.iter()) {
                for piece in parts {
                    context.ops().copy_device_to_device(
                        piece.object.offset(piece.copy.object_offset),
                        result.decoded.ptr().offset(piece.copy.linear_offset),
                        piece.copy.bytes,
                    )?;
                }
            }
            free(&mut context, gathered)?;
            free(&mut context, set.into_matrices())
        },
        move |copied| match copied {
            Ok(((), (weights, results, lanes))) => {
                let specs = layer_names(&next, index);
                let settled = specs.and_then(|names| {
                    settle(&next, lanes, results).map(|results| names.into_iter().zip(results).collect())
                });
                match settled {
                    Ok(results) => write_layer(next, index, weights, results, then),
                    Err(error) => end(&then, Err(error)),
                }
            }
            Err(error) => end(&then, Err(error)),
        },
    );
}

/// The quantized linears of layer `index`, in the order the layer's jobs have.
fn layer_names(inner: &Inner, index: usize) -> Result<Vec<String>, CalibrateError> {
    Ok(layer(&inner.objects, index)
        .map_err(|error| CalibrateError::failed("the layer's linears", error))?
        .into_iter()
        .filter(|linear| linear.quantized)
        .map(|linear| linear.tensor)
        .collect())
}

/// Has the writer write layer `index`, lists its file, and runs the write pass.
fn write_layer(
    inner: Arc<Inner>,
    index: usize,
    weights: LayerWeights,
    results: HashMap<String, Written>,
    then: Once<()>,
) {
    let bits = match trellis_of(&inner) {
        Ok(trellis) => trellis.bits.clone(),
        Err(error) => return end(&then, Err(error)),
    };
    let held = Arc::new(Handoff::empty());
    held.send(weights);
    let next = Arc::clone(&inner);
    let failed = Arc::clone(&then);
    write(
        &inner,
        |writer, done| writer.layer(index, results, &bits, done),
        Box::new(move |written| match written {
            Err(error) => end(&failed, Err(error)),
            Ok(()) => {
                if !next.ledger.list([layer_file(index)]) {
                    return end(
                        &failed,
                        Err(CalibrateError::Failed(
                            "the manifest was in use when the layer's file was listed".to_owned(),
                        )),
                    );
                }
                match held.take() {
                    Some(weights) => write_pass(next, weights, then),
                    None => end(&failed, Err(CalibrateError::Failed("the layer was lost".to_owned()))),
                }
            }
        }),
    );
}

// ---- the heads ----

/// The device calls of a trellis run's heads: the output head and the draft head are the GPTQ ones, rounded with the
/// run's recipe into the heads directory; the MTP block and the closing files are the trellis writer's.
pub struct TrellisDeviceHeads {
    heads: DeviceHeads,
    inner: Arc<Inner>,
}

impl TrellisDeviceHeads {
    pub(super) fn new(inner: Arc<Inner>) -> Self {
        Self {
            heads: DeviceHeads::new(Arc::clone(&inner)),
            inner,
        }
    }
}

/// `then` over the names of files in the heads directory, as the manifest lists them.
fn listed_in_heads(then: Then<Vec<String>>) -> Then<Vec<String>> {
    Box::new(move |result| then(result.map(|files| files.iter().map(|file| in_heads(file)).collect())))
}

impl Heads for TrellisDeviceHeads {
    fn observe(&self, then: Then<()>) {
        self.heads.observe(then);
    }

    /// Rounds the MTP block's eight linears without a Hessian, as Java does, and has the writer write them.
    fn mtp(&self, then: Then<Vec<String>>) {
        let inner = Arc::clone(&self.inner);
        let names = MTP_LINEARS.iter().map(|name| format!("{name}.weight")).collect();
        read_tensors(
            &self.inner,
            names,
            Box::new(move |read| match read {
                Ok(memory) => upload_mtp(inner, memory, then),
                Err(error) => then(Err(error)),
            }),
        );
    }

    fn output_head(&self, then: Then<Vec<String>>) {
        self.heads.output_head(listed_in_heads(then));
    }

    fn draft_head(&self, hessian: DraftHessian, then: Then<Vec<String>>) {
        self.heads.draft_head(hessian, listed_in_heads(then));
    }

    fn seal(&self, part: Part, files: Vec<String>, then: Then<()>) {
        self.heads.seal(part, files, then);
    }

    fn drop_checkpoint(&self, then: Then<()>) {
        self.heads.drop_checkpoint(then);
    }
}

impl TrellisHeads for TrellisDeviceHeads {
    fn close_checkpoint(&self, then: Then<Vec<String>>) {
        let inner = Arc::clone(&self.inner);
        let prepared = trellis_of(&inner).and_then(|trellis| {
            let manifest = inner
                .ledger
                .manifest()
                .ok_or_else(|| CalibrateError::Failed("the manifest is in use".to_owned()))?;
            let out = manifest.files.into_iter().filter(|file| !file.contains('/')).collect();
            let listing = Listing {
                source: trellis.source_files.clone(),
                out,
            };
            Ok((trellis.bits.clone(), listing, trellis.closing.clone()))
        });
        let (bits, listing, closing) = match prepared {
            Ok(prepared) => prepared,
            Err(error) => return then(Err(error)),
        };
        write(
            &inner,
            |writer, done| writer.finish(&bits, listing, done),
            Box::new(move |closed| then(closed.map(|()| closing))),
        );
    }
}

/// The MTP weights, read into pinned memory, up on the main lane; then rounds them.
fn upload_mtp(inner: Arc<Inner>, memory: Vec<PooledPinned>, then: Then<Vec<String>>) {
    let working = Arc::clone(&inner);
    let next = Arc::clone(&inner);
    on_main(
        &inner,
        "uploading the MTP weights",
        memory,
        move |queue, handles, memory| {
            let mut context = lane(&working, queue, handles);
            let mut uploaded = Vec::with_capacity(memory.len());
            for weight in memory {
                // SAFETY: the memory is kept until the lane has retired the copies, and is left alone.
                uploaded.push(unsafe { upload_pinned(&working, &mut context, weight) }?);
            }
            Ok(uploaded)
        },
        move |uploaded| match uploaded {
            Ok((weights, memory)) => {
                drop(memory);
                round_mtp(next, weights, then);
            }
            Err(error) => then(Err(error)),
        },
    );
}

fn round_mtp(inner: Arc<Inner>, weights: Vec<DeviceBuffer>, then: Then<Vec<String>>) {
    let jobs = MTP_LINEARS
        .iter()
        .map(|name| {
            let tensor = inner
                .objects
                .store()
                .tensor(&format!("{name}.weight"))
                .map_err(|error| CalibrateError::failed(name, error))?;
            let [n, k] = tensor.shape[..] else {
                return Err(CalibrateError::Failed(format!("{name} is not a matrix")));
            };
            Ok(Job {
                name: (*name).to_owned(),
                n: dimension(n)?,
                k: dimension(k)?,
                bits: f64::from(MTP_BITS),
                hessian: None,
            })
        })
        .collect::<Result<Vec<_>, CalibrateError>>();
    let Ok(jobs) = jobs else {
        return then(jobs.map(|_| Vec::new()));
    };
    let Some(lanes) = trellis_of(&inner).ok().and_then(|trellis| trellis.lanes.take()) else {
        return then(Err(CalibrateError::Failed("the trellis lanes are in use".to_owned())));
    };
    let next = Arc::clone(&inner);
    // SAFETY: the jobs name no Hessian.
    unsafe {
        round_jobs(
            Arc::clone(&inner.services),
            lanes,
            (jobs, weights),
            (MTP_SEED, 0),
            Box::new(move |rounded| match rounded {
                Ok(rounded) => free_mtp(next, rounded, then),
                Err(error) => then(Err(error)),
            }),
        );
    }
}

/// Frees the MTP weights on the main lane, then has the writer write the block.
fn free_mtp(inner: Arc<Inner>, rounded: Rounded, then: Then<Vec<String>>) {
    let Rounded {
        lanes,
        weights,
        results,
    } = rounded;
    let working = Arc::clone(&inner);
    let next = Arc::clone(&inner);
    on_main(
        &inner,
        "freeing the MTP weights",
        (results, lanes),
        move |queue, handles, _| {
            let mut context = lane(&working, queue, handles);
            free(&mut context, weights)
        },
        move |freed| match freed {
            Ok(((), (results, lanes))) => match settle(&next, lanes, results) {
                Ok(results) => {
                    let by_name: HashMap<String, Written> =
                        MTP_LINEARS.iter().map(|name| (*name).to_owned()).zip(results).collect();
                    write(
                        &next,
                        |writer, done| writer.mtp(&by_name, done),
                        Box::new(move |written| then(written.map(|()| vec![MTP_FILE.to_owned()]))),
                    );
                }
                Err(error) => then(Err(error)),
            },
            Err(error) => then(Err(error)),
        },
    );
}
