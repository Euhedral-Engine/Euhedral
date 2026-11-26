//! The GPTQ rounding of a layer's linears: each lane rounds one linear after another, each from its Hessian, in
//! slices of rows that share the factor, and writes its state; the lanes end when none is left.
//!
//! Every hand-off is a continuation: a step takes the lane by value and the frame the lane's completions emit when it
//! retires goes on with the next step.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::take;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;
use euhedral_memory::{LocationId, OwnerId, Planner};

use super::CalibrateError;
use super::driver::Then;
use super::files::save;
use super::schedule::{Schedule, Settled};
use crate::gptq::format::Format;
use crate::gptq::state::QuantState;
use crate::gptq::{DeviceRange, Factor, Finished, GptqLane, Refused, RoundRequest, TensorScales};
use crate::memory::{Claims, StepBytes};
use crate::recipe::{Format as StoredFormat, Recipe};
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

impl Linear {
    /// The same linear with no decoded weight held.
    fn shell(&self) -> Linear {
        Linear {
            name: self.name.clone(),
            rows: self.rows,
            columns: self.columns,
            weight: self.weight,
            hessian: self.hessian,
            decoded: Handoff::empty(),
        }
    }
}

/// A layer's rounding in progress, shared by its lanes.
pub struct Round {
    services: Arc<Services>,
    linears: Vec<Linear>,
    schedule: Schedule<GptqLane, DeviceBuffer>,
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
        formatter
            .debug_struct("Linear")
            .field("name", &self.name)
            .finish_non_exhaustive()
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
    let names = linears.iter().map(|linear| linear.name.clone()).collect();
    let decoded = linears
        .iter()
        .map(|linear| {
            let held = Handoff::empty();
            if let Some(buffer) = linear.decoded.take() {
                held.send(buffer);
            }
            held
        })
        .collect();
    let shells: Vec<Linear> = linears.iter().map(Linear::shell).collect();
    let (schedule, count) = Schedule::new(
        names,
        order,
        decoded,
        lanes.len(),
        Box::new(move |settled| then(settled.map(|settled| ended(shells, settled)))),
    );
    let round = Arc::new(Round {
        services,
        linears,
        schedule,
    });
    for (number, lane) in lanes.into_iter().enumerate() {
        if number < count {
            pull(&round, number, lane);
        } else {
            round.schedule.park(number, lane);
        }
    }
}

/// The linears with the decoded weights the round gave back.
fn ended(shells: Vec<Linear>, settled: Settled<GptqLane, DeviceBuffer>) -> Ended {
    for (shell, buffer) in shells.iter().zip(settled.decoded) {
        if let Some(buffer) = buffer {
            shell.decoded.send(buffer);
        }
    }
    Ended {
        lanes: settled.lanes,
        linears: shells,
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

fn abort(round: &Arc<Round>, number: usize, lane: GptqLane, error: CalibrateError) {
    round.schedule.abort(number, lane, error);
}

/// Takes the next linear, or ends the lane.
fn pull(round: &Arc<Round>, number: usize, lane: GptqLane) {
    round.schedule.pull(number, lane, |index, decoded, lane| {
        let linear = &round.linears[index];
        match round.services.settings.recipe.format(&linear.name) {
            Ok(format) => factor(
                Box::new(Object::new(round, number, index, decoded, rounded(format))),
                lane,
            ),
            Err(error) => abort(round, number, lane, CalibrateError::failed(&linear.name, error)),
        }
    });
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
    let outcome = lane.factor(hessian, k, damp, act_order, claims, move |finished| {
        after_factor(object, finished)
    });
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
    let Some(state) = joined(take(&mut object.slices)) else {
        return object.fail(lane, CalibrateError::Failed(format!("{name}: no rows were rounded")));
    };
    let bytes = match state.to_npz() {
        Ok(bytes) => bytes,
        Err(error) => return object.fail(lane, CalibrateError::failed(&name, error)),
    };
    let path = round.services.out.join(QuantState::file_name(&name));
    let index = object.index;
    let Object { decoded, .. } = *object;
    round.schedule.keep(index, decoded);
    let parked = Arc::new(Handoff::<GptqLane>::empty());
    parked.send(lane);
    let (sender, next) = (parked, Arc::clone(&round));
    save(
        &round.services.io,
        &round.services.host,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn state(rows: u64, fill: u8) -> QuantState {
        QuantState {
            format_name: "nvfp4".to_owned(),
            layout: "layout".to_owned(),
            rows,
            columns: 2,
            codes: vec![fill; (rows * 2) as usize],
            scales: vec![fill; rows as usize],
            global_scale: 0.5,
            table: None,
        }
    }

    #[test]
    fn no_slices_join_to_nothing() {
        assert!(joined(Vec::new()).is_none());
    }

    #[test]
    fn one_slice_joins_to_itself() {
        let one = joined(vec![state(3, 7)]).unwrap();
        assert_eq!(one, state(3, 7));
    }

    #[test]
    fn slices_join_in_order_with_their_rows_added() {
        let all = joined(vec![state(2, 1), state(3, 2)]).unwrap();
        assert_eq!((all.rows, all.global_scale), (5, 0.5), "the tensor scale is shared");
        assert_eq!(all.codes, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
        assert_eq!(all.scales, [1, 1, 2, 2, 2]);
    }
}
