//! The GPTQ rounding of a layer's linears: each lane rounds one linear after another, each from its Hessian, in
//! slices of rows that share the factor, and writes its state; the lanes end when none is left.
//!
//! Every hand-off is a continuation: a step takes the lane by value and the frame the lane's completions emit when it
//! retires goes on with the next step.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_core::frames::Frame;
use euhedral_graph::Lake;
use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;
use euhedral_memory::{LocationId, OwnerId, Planner};

use super::CalibrateError;
use super::driver::Then;
use crate::gptq::format::Format;
use crate::gptq::state::QuantState;
use crate::gptq::{DeviceRange, Factor, Finished, GptqLane, Refused, RoundRequest, TensorScales};
use crate::memory::{Claims, StepBytes};
use crate::recipe::{Format as StoredFormat, Recipe};
use crate::resume::ResumeFile;
use crate::step::idle;

/// Rows of a linear one rounding step takes, as Java's: the factor and the tensor scales are shared by the slices.
pub const HEAD_ROWS: usize = 16384;

/// How the linears are rounded.
#[derive(Clone, Debug)]
pub struct RoundSettings {
    pub recipe: Recipe,
    pub damp: f32,
    pub act_order: bool,
    /// Rows of a linear one step takes.
    pub slice_rows: usize,
}

/// What rounding uses besides the lanes: the planner that accounts the lanes' memory, and the files.
pub struct Services {
    pub planner: Arc<Planner>,
    pub owner: OwnerId,
    pub device: LocationId,
    pub pinned: LocationId,
    pub settings: RoundSettings,
    pub out: PathBuf,
    pub io: Arc<AsyncIo>,
    pub host: Arc<Lake>,
}

/// One linear of a layer: where its weight and its Hessian are, and the decoded weight rounding makes.
pub struct Linear {
    pub name: String,
    pub rows: u64,
    pub columns: usize,
    /// Device address of the BF16 weight `[rows, columns]`.
    pub weight: u64,
    /// Device address of the complete FP32 Hessian `[columns, columns]`.
    pub hessian: u64,
    /// The decoded BF16 weight: given as a buffer of the weight's size, allocated in stream order on the lane that
    /// owns the layer (the layer replaces the weight with it), and given back once rounded.
    pub decoded: Handoff<DeviceBuffer>,
}

/// A layer's rounding in progress, shared by its lanes.
pub struct Round {
    services: Arc<Services>,
    linears: Vec<Linear>,
    /// The linears' indices, largest first.
    order: Vec<usize>,
    next: AtomicUsize,
    running: AtomicUsize,
    failed: AtomicBool,
    error: Handoff<CalibrateError>,
    lanes: Vec<Handoff<GptqLane>>,
    ended: Handoff<Then<Ended>>,
}

/// What a layer's rounding gives back: the lanes and the linears with their decoded weights.
pub struct Ended {
    pub lanes: Vec<GptqLane>,
    pub linears: Vec<Linear>,
}

impl Debug for Round {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Round").finish_non_exhaustive()
    }
}

impl Debug for Services {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Services").finish_non_exhaustive()
    }
}

impl Debug for Linear {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Linear").field("name", &self.name).finish_non_exhaustive()
    }
}

impl Debug for Ended {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Ended").finish_non_exhaustive()
    }
}

/// Rounds `linears` on `lanes`, largest first; `then` gets the lanes and the linears, or the first failure.
///
/// # Safety
/// The weights and the Hessians the linears name stay live, and unwritten, until `then` is called; the Hessians are
/// complete on the device.
pub unsafe fn round_all(
    services: Arc<Services>,
    lanes: Vec<GptqLane>,
    linears: Vec<Linear>,
    order: Vec<usize>,
    then: Then<Ended>,
) {
    let count = lanes.len().min(linears.len()).max(1);
    let round = Arc::new(Round {
        services,
        order,
        next: AtomicUsize::new(0),
        running: AtomicUsize::new(count),
        failed: AtomicBool::new(false),
        error: Handoff::empty(),
        lanes: lanes.iter().map(|_| Handoff::empty()).collect(),
        linears,
        ended: Handoff::empty(),
    });
    round.ended.send(then);
    for (number, lane) in lanes.into_iter().enumerate() {
        if number < count {
            pull(&round, number, lane);
        } else {
            round.lanes[number].send(lane);
        }
    }
}

fn claims(services: &Services, bytes: StepBytes) -> Result<Claims, CalibrateError> {
    let take = |location, bytes| {
        services
            .planner
            .claim(services.owner, location, bytes, "gptq")
            .map_err(|error| CalibrateError::failed("the memory planner refused a claim", format!("{error:?}")))
    };
    Ok(Claims {
        device: Some(take(services.device, bytes.device)?),
        pinned: Some(take(services.pinned, bytes.pinned)?),
    })
}

/// The lane is done: it goes back to the round, and the last lane to end ends the round.
fn give_back(round: &Arc<Round>, number: usize, lane: GptqLane) {
    round.lanes[number].send(lane);
    if round.running.fetch_sub(1, Ordering::AcqRel) != 1 {
        return;
    }
    let lanes = round.lanes.iter().filter_map(Handoff::take).collect();
    let Some(then) = round.ended.take() else { return };
    if round.failed.load(Ordering::Acquire) {
        let error = round
            .error
            .take()
            .unwrap_or_else(|| CalibrateError::Failed("rounding failed".to_owned()));
        then(Err(error));
        return;
    }
    // Every lane has sent its end before the count reached 0, so no frame touches the round's linears any more.
    let linears = round.linears.iter().map(|linear| Linear {
        name: linear.name.clone(),
        rows: linear.rows,
        columns: linear.columns,
        weight: linear.weight,
        hessian: linear.hessian,
        decoded: Handoff::empty(),
    });
    let linears: Vec<Linear> = linears
        .zip(&round.linears)
        .map(|(copy, own)| {
            if let Some(buffer) = own.decoded.take() {
                copy.decoded.send(buffer);
            }
            copy
        })
        .collect();
    then(Ok(Ended { lanes, linears }));
}

fn abort(round: &Arc<Round>, number: usize, lane: GptqLane, error: CalibrateError) {
    if !round.failed.swap(true, Ordering::AcqRel) {
        round.error.send(error);
    }
    give_back(round, number, lane);
}

/// Takes the next linear, or ends the lane.
fn pull(round: &Arc<Round>, number: usize, lane: GptqLane) {
    if round.failed.load(Ordering::Acquire) {
        give_back(round, number, lane);
        return;
    }
    let at = round.next.fetch_add(1, Ordering::AcqRel);
    let Some(&index) = round.order.get(at) else {
        give_back(round, number, lane);
        return;
    };
    let linear = &round.linears[index];
    let decoded = linear.decoded.take();
    let format = round.services.settings.recipe.format(&linear.name);
    match (decoded, format) {
        (Some(decoded), Ok(format)) => {
            factor(Box::new(Object::new(round, number, index, decoded, rounded(format))), lane);
        }
        (None, _) => abort(
            round,
            number,
            lane,
            CalibrateError::failed(&linear.name, "no buffer was given for the decoded weight"),
        ),
        (_, Err(error)) => abort(round, number, lane, CalibrateError::failed(&linear.name, error)),
    }
}

/// One linear on its lane: its factor, its tensor scales and the slices rounded so far.
struct Object {
    round: Arc<Round>,
    number: usize,
    index: usize,
    decoded: DeviceBuffer,
    format: Format,
    factor: Option<Factor>,
    tensor: Option<TensorScales>,
    slices: Vec<QuantState>,
    rows_done: usize,
}

impl Object {
    fn new(round: &Arc<Round>, number: usize, index: usize, decoded: DeviceBuffer, format: Format) -> Object {
        Object {
            round: Arc::clone(round),
            number,
            index,
            decoded,
            format,
            factor: None,
            tensor: None,
            slices: Vec::new(),
            rows_done: 0,
        }
    }

    fn linear(&self) -> &Linear {
        &self.round.linears[self.index]
    }

    /// Ends the lane with `error`; the decoded weight goes with the object.
    fn fail(self: Box<Self>, lane: GptqLane, error: CalibrateError) {
        let round = Arc::clone(&self.round);
        let number = self.number;
        drop(self);
        abort(&round, number, lane, error);
    }
}

/// The format the rounding works in, from the one the recipe stores the object in.
fn rounded(stored: StoredFormat) -> Format {
    Format {
        name: stored.name,
        layout: stored.layout,
        group: stored.group as usize,
        qmin: stored.qmin,
        qmax: stored.qmax,
        nvfp4: stored.nvfp4,
        sd4: stored.sd4,
    }
}

/// A step that did not start gives its lane back and ends the round's work on it.
fn started(round: &Arc<Round>, number: usize, outcome: Result<(), Box<Refused>>) {
    if let Err(refused) = outcome {
        let Refused { lane, error } = *refused;
        abort(round, number, lane, CalibrateError::failed("gptq", error));
    }
}

fn factor(object: Box<Object>, mut lane: GptqLane) {
    let (round, number) = (Arc::clone(&object.round), object.number);
    let (k, services) = (object.linear().columns, Arc::clone(&round.services));
    let prepared = lane
        .factor_bytes(k)
        .map_err(|error| CalibrateError::failed("factor", error))
        .and_then(|bytes| claims(&services, bytes));
    let claims = match prepared {
        Ok(claims) => claims,
        Err(error) => return object.fail(lane, error),
    };
    // SAFETY: the Hessian is live and complete until the round ends (the contract of `round_all`).
    let hessian = unsafe { DeviceRange::new(object.linear().hessian, 4 * (k * k) as u64) };
    let (damp, act_order) = (services.settings.damp, services.settings.act_order);
    let outcome = lane.factor(hessian, k, damp, act_order, claims, move |finished| after_factor(object, finished));
    started(&round, number, outcome);
}

fn after_factor(mut object: Box<Object>, finished: Finished<Factor>) {
    let Finished { lane, scratch, result } = finished;
    drop(scratch);
    match result {
        Err(error) => object.fail(lane, CalibrateError::failed("factor", error)),
        Ok(factor) => {
            object.factor = Some(factor);
            if object.format.nvfp4 {
                scales(object, lane);
            } else {
                object.tensor = Some(TensorScales {
                    global_scale: 0.0,
                    table: None,
                });
                slice(object, lane);
            }
        }
    }
}

fn weight_range(object: &Object, first: usize, rows: usize) -> DeviceRange {
    let linear = object.linear();
    let row_bytes = 2 * linear.columns as u64;
    // SAFETY: the weight is live and unwritten until the round ends, and `first + rows` is within the linear.
    unsafe { DeviceRange::new(linear.weight + first as u64 * row_bytes, rows as u64 * row_bytes) }
}

fn scales(object: Box<Object>, lane: GptqLane) {
    let (round, number) = (Arc::clone(&object.round), object.number);
    let (rows, k) = (object.linear().rows as usize, object.linear().columns);
    let Some(factor) = object.factor.clone() else {
        return object.fail(lane, CalibrateError::Failed("no factor".to_owned()));
    };
    let format = object.format.clone();
    let range = weight_range(&object, 0, rows);
    let prepared = lane
        .scales_bytes(rows, k, &format)
        .map_err(|error| CalibrateError::failed("tensor scales", error))
        .and_then(|bytes| claims(&round.services, bytes));
    let claims = match prepared {
        Ok(claims) => claims,
        Err(error) => return object.fail(lane, error),
    };
    let outcome = lane.tensor_scales(range, rows, &factor, &format, claims, move |finished| {
        after_scales(object, finished);
    });
    started(&round, number, outcome);
}

fn after_scales(mut object: Box<Object>, finished: Finished<TensorScales>) {
    let Finished { lane, scratch, result } = finished;
    drop(scratch);
    match result {
        Err(error) => object.fail(lane, CalibrateError::failed("tensor scales", error)),
        Ok(tensor) => {
            object.tensor = Some(tensor);
            slice(object, lane);
        }
    }
}

/// Rounds the next slice of rows, or writes the state when every row is done.
fn slice(object: Box<Object>, lane: GptqLane) {
    let (round, number) = (Arc::clone(&object.round), object.number);
    let total = object.linear().rows as usize;
    if object.rows_done >= total {
        return write(object, lane);
    }
    let rows = (total - object.rows_done).min(round.services.settings.slice_rows);
    let (Some(factor), Some(tensor)) = (object.factor.clone(), object.tensor) else {
        return object.fail(lane, CalibrateError::Failed("no factor".to_owned()));
    };
    let format = object.format.clone();
    let weight = weight_range(&object, object.rows_done, rows);
    let row_bytes = 2 * object.linear().columns as u64;
    // SAFETY: the decoded buffer is the object's own and lives until its state is written.
    let decoded = unsafe {
        DeviceRange::new(
            object.decoded.addr() + object.rows_done as u64 * row_bytes,
            rows as u64 * row_bytes,
        )
    };
    let claims = match claims(&round.services, lane.round_bytes(rows, &factor, &format)) {
        Ok(claims) => claims,
        Err(error) => return object.fail(lane, error),
    };
    let request = RoundRequest {
        weight,
        n: rows,
        factor: &factor,
        format: &format,
        tensor: &tensor,
        decoded_out: Some(decoded),
    };
    let outcome = lane.round(&request, claims, move |finished| after_slice(object, rows, finished));
    started(&round, number, outcome);
}

fn after_slice(mut object: Box<Object>, rows: usize, finished: Finished<QuantState>) {
    let Finished { lane, scratch, result } = finished;
    drop(scratch);
    match result {
        Err(error) => object.fail(lane, CalibrateError::failed("rounding", error)),
        Ok(state) => {
            object.slices.push(state);
            object.rows_done += rows;
            slice(object, lane);
        }
    }
}

/// The slices as one state: rows follow each other in the codes and the scales, and the tensor scales are shared.
fn joined(slices: Vec<QuantState>) -> Option<QuantState> {
    let mut slices = slices.into_iter();
    let mut state = slices.next()?;
    for next in slices {
        state.rows += next.rows;
        state.codes.extend_from_slice(&next.codes);
        state.scales.extend_from_slice(&next.scales);
    }
    Some(state)
}

/// Saves the state next to its siblings (written whole under a temporary name, then renamed, so a stop leaves a file
/// of a finished object or none), keeps the decoded weight and takes the next linear.
fn write(mut object: Box<Object>, lane: GptqLane) {
    let (round, number) = (Arc::clone(&object.round), object.number);
    let name = object.linear().name.clone();
    let Some(state) = joined(std::mem::take(&mut object.slices)) else {
        return object.fail(lane, CalibrateError::Failed(format!("{name}: no rows were rounded")));
    };
    let bytes = match state.to_npz() {
        Ok(bytes) => bytes,
        Err(error) => return object.fail(lane, CalibrateError::failed(&name, error)),
    };
    let path = round.services.out.join(QuantState::file_name(&name));
    let index = object.index;
    let Object { decoded, .. } = *object;
    round.linears[index].decoded.send(decoded);
    let parked = Arc::new(Handoff::<GptqLane>::empty());
    parked.send(lane);
    let (sender, next) = (parked, Arc::clone(&round));
    save(
        &round.services,
        path,
        bytes,
        Box::new(move |saved| {
            let Some(lane) = sender.take() else { return idle() };
            match saved {
                Ok(()) => pull(&next, number, lane),
                Err(error) => abort(&next, number, lane, CalibrateError::failed(&name, error)),
            }
            idle()
        }),
    );
}

/// What a save tells its caller; the frame it returns is what the file's call hands the lake.
pub type Saved = Box<dyn FnOnce(Result<(), CalibrateError>) -> Box<dyn Frame> + Send>;

fn end(finish: &Handoff<Saved>, result: Result<(), CalibrateError>) -> Box<dyn Frame> {
    match finish.take() {
        Some(finish) => finish(result),
        None => idle(),
    }
}

/// Writes `bytes` whole to `path`: to a temporary file next to it, made durable and renamed over `path`.
pub fn save(services: &Services, path: PathBuf, bytes: Vec<u8>, then: Saved) {
    let finish = Arc::new(Handoff::<Saved>::empty());
    finish.send(then);
    let mut temporary = path.clone().into_os_string();
    temporary.push(".tmp");
    let len = bytes.len() as u64;
    let open_finish = Arc::clone(&finish);
    let opened = ResumeFile::open(
        &services.io,
        &services.host,
        &PathBuf::from(temporary),
        Box::new(move |opened| {
            let file = match opened {
                Ok(file) => file,
                Err(error) => return end(&open_finish, Err(CalibrateError::failed("open", error))),
            };
            let write_finish = Arc::clone(&open_finish);
            let sealing = file.clone();
            let wrote = file.write(
                0,
                bytes,
                Box::new(move |wrote| {
                    if let Err(error) = wrote {
                        return end(&write_finish, Err(CalibrateError::failed("write", error)));
                    }
                    let seal_finish = Arc::clone(&write_finish);
                    let renamed = sealing.seal(
                        len,
                        &path,
                        Box::new(move |sealed| {
                            end(&seal_finish, sealed.map_err(|error| CalibrateError::failed("seal", error)))
                        }),
                    );
                    match renamed {
                        Ok(()) => idle(),
                        Err(error) => end(&write_finish, Err(CalibrateError::failed("seal", error))),
                    }
                }),
            );
            match wrote {
                Ok(()) => idle(),
                Err(error) => end(&open_finish, Err(CalibrateError::failed("write", error))),
            }
        }),
    );
    if let Err(error) = opened {
        drop(end(&finish, Err(CalibrateError::failed("open", error))));
    }
}
