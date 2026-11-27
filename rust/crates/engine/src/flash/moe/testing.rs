//! Fakes of the cache's side of the stages: an expert source whose arrivals a test orders, and leases that say how they
//! were closed.

#![allow(
    clippy::disallowed_types,
    reason = "test fixture: the fake source's request queue and log are Mutexes, read by one test thread"
)]

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use euhedral_graph::{DeviceQueue, Marker, StageCx, StageError, Submitted};

use super::source::{Arrival, Arrived, ExpertSource, HeldExpert};
use crate::residency::experts::ExpertId;

/// What happened to leases and requests, in order.
#[derive(Clone, Debug, Default)]
pub(crate) struct Log(Arc<Mutex<Vec<String>>>);

impl Log {
    pub(crate) fn push(&self, line: String) {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).push(line);
    }

    pub(crate) fn lines(&self) -> Vec<String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

/// A held expert at `address` that logs its fence and its drop.
pub(crate) struct Lease {
    address: u64,
    ready: Option<Marker>,
    log: Log,
}

impl Lease {
    pub(crate) fn boxed(address: u64, ready: Option<Marker>, log: &Log) -> Box<dyn HeldExpert> {
        Box::new(Self {
            address,
            ready,
            log: log.clone(),
        })
    }
}

impl HeldExpert for Lease {
    fn device_address(&self) -> u64 {
        self.address
    }

    fn ready(&self) -> Option<Marker> {
        self.ready
    }

    fn fence_after(&mut self, marker: Marker) {
        self.log
            .push(format!("fence {:#x} behind {}", self.address, marker.value));
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.log.push(format!("close {:#x}", self.address));
    }
}

/// A request made of the source and not yet answered.
pub(crate) struct Pending {
    pub(crate) expert: ExpertId,
    pub(crate) scan: bool,
    pub(crate) cancel: Arc<AtomicBool>,
    arrival: Arrival,
}

impl Pending {
    pub(crate) fn answer(self, arrived: Arrived) {
        (self.arrival)(arrived);
    }
}

/// A source that keeps each request until the test answers it, or refuses every request.
#[derive(Default)]
pub(crate) struct Source {
    pending: Mutex<Vec<Pending>>,
    refusal: Option<String>,
}

impl Source {
    pub(crate) fn refusing(why: &str) -> Self {
        Self {
            pending: Mutex::default(),
            refusal: Some(why.to_owned()),
        }
    }

    /// The request for `expert`, taken out of the pending ones.
    pub(crate) fn take(&self, expert: u32) -> Option<Pending> {
        let mut pending = self.pending.lock().unwrap_or_else(PoisonError::into_inner);
        let at = pending.iter().position(|request| request.expert.expert == expert)?;
        Some(pending.remove(at))
    }

    pub(crate) fn pending(&self) -> usize {
        self.pending.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

impl ExpertSource for Source {
    fn fetch(&self, expert: ExpertId, scan: bool, cancel: Arc<AtomicBool>, arrival: Arrival) -> Result<(), String> {
        if let Some(why) = &self.refusal {
            return Err(why.clone());
        }
        self.pending
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Pending {
                expert,
                scan,
                cancel,
                arrival,
            });
        Ok(())
    }
}

use euhedral_inference_cuda::DevicePtr;

use super::layer::{MoeLayer, MoeShape, MoeWeights};
use super::ops::ExpertGeometry;
use super::routing::DescriptorLayout;
use crate::flash::Weight;
use crate::flash::buffers::{FlashRegions, StreamShape};
use crate::flash::linear::{Nvfp4Device, Nvfp4Weight};

pub(crate) fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

fn weight(address: u64) -> Weight {
    Weight {
        address: at(address),
        bytes: 4096,
    }
}

fn nvfp4(address: u64) -> Nvfp4Weight {
    Nvfp4Weight {
        weight: weight(address),
        sd4: false,
    }
}

/// The base of the regions of the fixture layer.
pub(crate) const REGIONS: u64 = 0x1000_0000;

pub(crate) fn shape() -> MoeShape {
    let layout = DescriptorLayout::new(8, 2, 4).unwrap();
    let geometry = ExpertGeometry::new(128, 128, 0, 4096).unwrap();
    MoeShape::new(layout, geometry, 192).unwrap()
}

pub(crate) fn weights(padded: bool) -> MoeWeights {
    MoeWeights {
        router: weight(0x100000),
        shared_gate: nvfp4(0x110000),
        shared_up: nvfp4(0x120000),
        shared_down: nvfp4(0x130000),
        shared_down_padded: padded.then(|| nvfp4(0x140000)),
        shared_expert_gate: weight(0x150000),
    }
}

/// Regions with as much block scratch as the fixture layer needs, over [`REGIONS`].
pub(crate) fn regions(extra_scratch: u64) -> FlashRegions {
    let device = Nvfp4Device::with_native_route(false, 8);
    let mut sizes = MoeLayer::scratch_sizes(&shape(), true, &device);
    sizes.layer += extra_scratch;
    sizes.hyper = 256;
    FlashRegions::over(at(REGIONS), 4, StreamShape::new(128, 2).unwrap(), sizes).unwrap()
}

/// Layer 3, over bank 5 of `source`.
pub(crate) fn layer(source: Arc<dyn ExpertSource>, padded: bool) -> MoeLayer {
    MoeLayer::new(
        3,
        5,
        shape(),
        weights(padded),
        regions(0),
        Nvfp4Device::with_native_route(false, 8),
        source,
    )
    .unwrap()
}

use super::block::BlockPlan;
use super::routing::Routing;

/// A plan of the fixture shape for `rows` rows whose router chose `ids` (two per row), published.
pub(crate) fn planned(rows: u32, ids: &[i32]) -> Arc<BlockPlan> {
    let layout = *shape().layout();
    let mut routing = Routing::new(layout);
    routing.plan(rows, |i| ids[i], |_| 0x3f80).unwrap();
    let plan = Arc::new(BlockPlan::new(layout.max_active()));
    plan.publish(&routing).unwrap();
    plan
}

use super::host::{MoeHost, MoeWork};
use crate::handoff::Handoff;

/// A quantum as the MoE stages see it: the plan, the router's picks, the words of the description and the work the
/// stages queue, which is logged and answered from a script.
pub(crate) struct Host {
    pub(crate) plan: Option<Arc<BlockPlan>>,
    pub(crate) stopped: AtomicBool,
    pub(crate) cancel: Option<Arc<AtomicBool>>,
    pub(crate) fits: bool,
    pub(crate) rows: usize,
    pub(crate) log: Log,
    routing: Handoff<Box<Routing>>,
    picks: Vec<(i32, u16)>,
    words: Vec<AtomicU32>,
    outcomes: Mutex<Vec<Result<Submitted, StageError>>>,
}

impl Host {
    pub(crate) fn new(plan: Arc<BlockPlan>) -> Self {
        let layout = *shape().layout();
        let routing = Handoff::empty();
        routing.send(Box::new(Routing::new(layout)));
        Self {
            plan: Some(plan),
            stopped: AtomicBool::new(false),
            cancel: None,
            fits: true,
            rows: 3,
            log: Log::default(),
            routing,
            picks: Vec::new(),
            words: (0..layout.bytes() / 4).map(|_| AtomicU32::new(0)).collect(),
            outcomes: Mutex::default(),
        }
    }

    /// The router's choice for the next plan, `(expert id, BF16 weight bits)` per pair.
    pub(crate) fn picking(mut self, picks: &[(i32, u16)], rows: usize) -> Self {
        self.picks = picks.to_vec();
        self.rows = rows;
        self
    }

    /// The answer of the next queued work (in order); work past the script succeeds with no marker.
    pub(crate) fn answering(self, outcome: Result<Submitted, StageError>) -> Self {
        self.outcomes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(0, outcome);
        self
    }

    pub(crate) fn word(&self, index: usize) -> u32 {
        self.words[index].load(Ordering::Relaxed)
    }

    pub(crate) fn routing_is_home(&self) -> bool {
        self.routing.is_set()
    }
}

impl MoeHost for Host {
    fn block_plan(&self) -> Result<&Arc<BlockPlan>, StageError> {
        self.plan
            .as_ref()
            .ok_or_else(|| StageError::Fatal("no block".to_owned()))
    }

    fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    fn cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.cancel.clone()
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn check_staging(&self, layout: &DescriptorLayout) -> Result<(), StageError> {
        if self.fits && layout == shape().layout() {
            Ok(())
        } else {
            Err(StageError::Fatal("staging does not fit".to_owned()))
        }
    }

    fn take_routing(&self) -> Result<Box<Routing>, StageError> {
        self.routing
            .take()
            .ok_or_else(|| StageError::Fatal("the routing scratch is in use".to_owned()))
    }

    fn give_back(&self, routing: Box<Routing>) {
        self.routing.send(routing);
    }

    fn router_pick(&self, i: usize) -> (i32, u16) {
        self.picks[i]
    }

    fn put_word(&self, word: usize, value: u32) {
        self.words[word].store(value, Ordering::Relaxed);
    }

    fn await_unmarked(&self, _home: &dyn DeviceQueue) {
        self.log.push("await unmarked".to_owned());
    }

    fn queue(
        &self,
        _cx: &StageCx<'_>,
        _layer: &MoeLayer,
        waits: &[Marker],
        work: MoeWork,
    ) -> Result<Submitted, StageError> {
        let waits: Vec<u64> = waits.iter().map(|marker| marker.value).collect();
        self.log.push(format!("queue {work:?} after {waits:?}"));
        self.outcomes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_else(|| Ok(Submitted::done()))
    }
}
