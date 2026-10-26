//! The graphs a sequence's quanta have been captured into, by capture key.
//!
//! A key is seen once and its quantum runs stage by stage. On the second sight the quantum runs stage by stage again
//! while its launches are recorded into a graph on a lane of its own (the quantum's record stage), so the device never
//! waits for a capture or an instantiation. From the next sight after the recording has finished, every quantum with
//! the key is one graph launch on the home lane. The cache belongs to the sequence's owner, which admits one quantum
//! at a time ([`GraphCache::admit`]); the slots it hands out are shared with the quanta in flight, and a slot's graph
//! is written once, by the record stage of the quantum that claimed it, so no lock is needed.
//!
//! A graph repeats the allocations it captured, so a key names them ([`CaptureKey::buffer_generations`]) and a cache
//! serves one sequence: another sequence's allocations key differently. A cache keeps at most `captures-per-view`
//! graphs for each view (`kernels.graphs.captures-per-view`); a new key first releases the keys of the same quantum
//! whose attention geometry the sequence has passed, and then, if the view is still full, the key admitted longest
//! ago. A released graph is freed once the quanta that hold it have retired.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_inference_cuda::{CudaError, CudaLane, GraphExec, GraphRecorder, Submission, SubmitMode};

use super::capture_key::{CaptureKey, geometry_rank};
use crate::plan::View;

/// One key's graph, once it is recorded.
#[derive(Debug)]
pub struct GraphSlot {
    graph: OnceLock<GraphExec>,
    recording: AtomicBool,
    broken: AtomicBool,
    replays: AtomicU64,
    /// The hash of the submissions the graph was recorded from, for the test oracle (0: none).
    hash: AtomicU64,
}

impl GraphSlot {
    fn new() -> GraphSlot {
        GraphSlot {
            graph: OnceLock::new(),
            recording: AtomicBool::new(false),
            broken: AtomicBool::new(false),
            replays: AtomicU64::new(0),
            hash: AtomicU64::new(0),
        }
    }

    /// The recorded graph, if there is one.
    pub fn graph(&self) -> Option<&GraphExec> {
        self.graph.get()
    }

    /// Whether the key's quanta can no longer be captured (the graph could not be recorded or instantiated); they run
    /// stage by stage.
    pub fn is_broken(&self) -> bool {
        self.broken.load(Ordering::Acquire)
    }

    /// Launches of the recorded graph so far, the launch that recorded it included.
    pub fn launches(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }

    pub(crate) fn launched(&self) {
        self.replays.fetch_add(1, Ordering::Relaxed);
    }

    /// Claims the right to record; false when another quantum is recording this key.
    pub(crate) fn claim(&self) -> bool {
        self.recording
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn release(&self) {
        self.recording.store(false, Ordering::Release);
    }

    /// Keeps the recorded graph. The claim is released.
    pub(crate) fn publish(&self, graph: GraphExec) -> &GraphExec {
        let kept = self.graph.get_or_init(|| graph);
        self.release();
        kept
    }

    /// The key cannot be captured: its quanta run stage by stage from now on. The claim is released.
    pub(crate) fn abandon(&self) {
        self.broken.store(true, Ordering::Release);
        self.release();
    }

    /// The hash the graph was recorded from (the test oracle), or 0.
    pub fn recorded_hash(&self) -> u64 {
        self.hash.load(Ordering::Acquire)
    }

    #[cfg(feature = "test-oracles")]
    pub(crate) fn set_recorded_hash(&self, hash: u64) {
        self.hash.store(hash, Ordering::Release);
    }
}

/// What the cache decided for a quantum.
#[derive(Clone, Debug)]
pub enum Admission {
    /// Not captured: its key is new (the first sight), it has none, another quantum is recording its key, or its key
    /// cannot be captured. It runs stage by stage.
    Eager,
    /// The key's second sight, or the first after a recording that was given up: the quantum runs stage by stage and
    /// its record stage records the slot's graph on the capture lane, which this admission has claimed.
    Record(Arc<GraphSlot>),
    /// The slot has its graph: the quantum is one launch of it.
    Replay(Arc<GraphSlot>),
}

#[derive(Debug)]
struct Entry {
    slot: Arc<GraphSlot>,
    sightings: u32,
    admitted: u64,
}

/// The captured graphs of one sequence.
#[derive(Debug)]
pub struct GraphCache {
    per_view: usize,
    entries: HashMap<CaptureKey, Entry>,
    clock: u64,
    evictions: u64,
}

impl GraphCache {
    /// A cache that captures nothing: every quantum runs stage by stage. The reference that a capturing cache is
    /// compared with.
    pub fn disabled() -> GraphCache {
        GraphCache {
            per_view: 0,
            entries: HashMap::new(),
            clock: 0,
            evictions: 0,
        }
    }

    /// A cache keeping up to `per_view` graphs for each view (`kernels.graphs.captures-per-view`).
    pub fn with_capacity(per_view: usize) -> GraphCache {
        GraphCache {
            per_view: per_view.max(1),
            entries: HashMap::new(),
            clock: 0,
            evictions: 0,
        }
    }

    /// Graphs kept for each view at most.
    pub fn capacity(&self) -> usize {
        self.per_view
    }

    /// Keys the cache holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keys of `view` the cache holds.
    pub fn len_of(&self, view: View) -> usize {
        self.entries
            .keys()
            .filter(|key| key.view.variant() == view.variant())
            .count()
    }

    /// Keys with a recorded graph.
    pub fn recorded(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.slot.graph().is_some())
            .count()
    }

    /// The keys the cache holds, each with whether its graph is recorded.
    pub fn keys(&self) -> impl Iterator<Item = (&CaptureKey, bool)> {
        self.entries
            .iter()
            .map(|(key, entry)| (key, entry.slot.graph().is_some()))
    }

    /// Keys released to make room or because the sequence passed their geometry.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Admits a quantum with `key` (`None`: it is not capturable): the first sight runs stage by stage, the second
    /// records, and once a graph exists the quantum replays it.
    pub fn admit(&mut self, key: Option<CaptureKey>) -> Admission {
        let Some(key) = key else { return Admission::Eager };
        if self.per_view == 0 {
            return Admission::Eager;
        }
        self.clock += 1;
        if !self.entries.contains_key(&key) {
            self.make_room(&key);
        }
        let clock = self.clock;
        let entry = self.entries.entry(key).or_insert_with(|| Entry {
            slot: Arc::new(GraphSlot::new()),
            sightings: 0,
            admitted: clock,
        });
        entry.admitted = clock;
        entry.sightings = entry.sightings.saturating_add(1);
        if entry.slot.is_broken() {
            Admission::Eager
        } else if entry.slot.graph().is_some() {
            Admission::Replay(Arc::clone(&entry.slot))
        } else if entry.sightings >= 2 && entry.slot.claim() {
            Admission::Record(Arc::clone(&entry.slot))
        } else {
            Admission::Eager
        }
    }

    /// Releases the keys `key` makes unnecessary: those of the same quantum at a lower geometry (the sequence has
    /// passed them), then the oldest of its view while the view is full.
    fn make_room(&mut self, key: &CaptureKey) {
        let rank = geometry_rank(key.split_geometry);
        let before = self.entries.len();
        self.entries
            .retain(|held, _| !(held.same_but_geometry(key) && geometry_rank(held.split_geometry) < rank));
        self.evictions += (before - self.entries.len()) as u64;
        let variant = key.view.variant();
        while self
            .entries
            .keys()
            .filter(|held| held.view.variant() == variant)
            .count()
            >= self.per_view
        {
            let Some(oldest) = self
                .entries
                .iter()
                .filter(|(held, _)| held.view.variant() == variant)
                .min_by_key(|(_, entry)| entry.admitted)
                .map(|(held, _)| held.clone())
            else {
                break;
            };
            self.entries.remove(&oldest);
            self.evictions += 1;
        }
    }

    /// Admits `key` like [`GraphCache::admit`] and, when that says to record, records the launches `record` queues
    /// on a capture of `lane` and keeps the graph. The graph is not launched. A running engine records in the
    /// quantum's record stage, while the quantum runs; this is the same step for a caller that holds a lane.
    pub fn get_or_record(
        &mut self,
        key: CaptureKey,
        lane: &CudaLane,
        mode: SubmitMode,
        record: impl FnOnce(&Submission<'_>) -> Result<(), CudaError>,
    ) -> Result<Sighting, CudaError> {
        match self.admit(Some(key)) {
            Admission::Eager => Ok(Sighting::First),
            Admission::Replay(slot) => Ok(Sighting::Graph(slot)),
            Admission::Record(slot) => match record_on(lane, mode, record) {
                Ok(graph) => {
                    slot.publish(graph);
                    Ok(Sighting::Graph(slot))
                }
                Err(error) => {
                    slot.abandon();
                    Err(error)
                }
            },
        }
    }
}

/// What [`GraphCache::get_or_record`] found.
#[derive(Debug)]
pub enum Sighting {
    /// The key is new: the quantum runs stage by stage.
    First,
    /// The key's slot, with its graph recorded.
    Graph(Arc<GraphSlot>),
}

/// Records the launches `record` queues on a capture of `lane`, and instantiates the graph.
pub(crate) fn record_on<E>(
    lane: &CudaLane,
    mode: SubmitMode,
    record: impl FnOnce(&Submission<'_>) -> Result<(), E>,
) -> Result<GraphExec, E>
where
    E: From<CudaError>,
{
    let recorder = GraphRecorder::begin(lane)?;
    {
        let submission = recorder.submission(mode)?;
        record(&submission)?;
    }
    Ok(recorder.end()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::QuantumKind;

    fn key(geometry: u64) -> CaptureKey {
        CaptureKey {
            view: View::Decode,
            kind: QuantumKind::Decode,
            rows: 1,
            logits: 1,
            selects: true,
            seeds_draft: false,
            draft_seed: None,
            draft_read: None,
            draft_write: None,
            writes_current_token: false,
            row_exact: false,
            split_geometry: geometry,
            reserved_host_pages: [0, 0],
            replay_rows: Vec::new(),
            buffer_generations: Vec::new(),
        }
    }

    fn verify(rows: usize, geometry: u64) -> CaptureKey {
        CaptureKey {
            view: View::Verify { rows },
            kind: QuantumKind::Verify,
            rows,
            logits: rows,
            row_exact: true,
            ..key(geometry)
        }
    }

    #[test]
    fn a_key_runs_stage_by_stage_once_records_on_its_second_sight_and_replays_once_it_has_a_graph() {
        let mut cache = GraphCache::with_capacity(8);
        assert!(
            matches!(cache.admit(Some(key(1 << 8))), Admission::Eager),
            "the first sight"
        );
        let Admission::Record(slot) = cache.admit(Some(key(1 << 8))) else {
            panic!("the second sight records")
        };
        assert!(slot.graph().is_none());
        // The next quantum is admitted while the recording runs: it neither records nor replays.
        assert!(matches!(cache.admit(Some(key(1 << 8))), Admission::Eager));
        assert_eq!(cache.len(), 1);
        assert!(
            matches!(cache.admit(Some(key(2 << 8))), Admission::Eager),
            "another key starts over"
        );
        assert!(
            matches!(cache.admit(None), Admission::Eager),
            "a quantum with no key is never captured"
        );
    }

    #[test]
    fn a_recording_that_was_given_up_is_tried_again_and_one_that_cannot_be_made_never_is() {
        let mut cache = GraphCache::with_capacity(8);
        let _ = cache.admit(Some(key(256)));
        let Admission::Record(slot) = cache.admit(Some(key(256))) else {
            panic!("record")
        };
        assert!(!slot.claim(), "one quantum records at a time");
        slot.release();
        assert!(
            matches!(cache.admit(Some(key(256))), Admission::Record(_)),
            "the lane was busy: again"
        );
        assert!(
            matches!(cache.admit(Some(key(256))), Admission::Eager),
            "while that one records, none does"
        );
    }

    #[test]
    fn a_key_that_cannot_be_captured_runs_stage_by_stage_for_good() {
        let mut cache = GraphCache::with_capacity(8);
        let _ = cache.admit(Some(key(256)));
        let Admission::Record(slot) = cache.admit(Some(key(256))) else {
            panic!("record")
        };
        slot.abandon();
        assert!(slot.is_broken());
        assert!(matches!(cache.admit(Some(key(256))), Admission::Eager));
    }

    #[test]
    fn a_new_geometry_releases_the_geometries_the_sequence_has_passed() {
        let mut cache = GraphCache::with_capacity(8);
        for span in 1..=5u64 {
            let _ = cache.admit(Some(key(span << 8)));
            assert_eq!(cache.len(), 1, "span {span}: the spans before it are passed");
        }
        assert_eq!(cache.evictions(), 4);
        // A quantum of another kind at a lower geometry is not passed: it is another key of the same view.
        let _ = cache.admit(Some(verify(3, 1 << 8)));
        let _ = cache.admit(Some(verify(5, 1 << 8)));
        assert_eq!(cache.len(), 3);
        let _ = cache.admit(Some(verify(3, 2 << 8)));
        assert_eq!(
            cache.len(),
            3,
            "verify of 3 rows passed its first span; verify of 5 rows did not"
        );
    }

    #[test]
    fn a_view_holds_at_most_its_capacity_and_releases_the_oldest() {
        let mut cache = GraphCache::with_capacity(2);
        // Three decode keys that differ in a buffer (the selection words), none of which is passed.
        let with = |selection: u64| CaptureKey {
            buffer_generations: vec![euhedral_inference_cuda::AllocationId(selection)],
            ..key(1 << 8)
        };
        let _ = cache.admit(Some(with(1)));
        let _ = cache.admit(Some(with(2)));
        let _ = cache.admit(Some(with(1)));
        assert_eq!(cache.len_of(View::Decode), 2);
        let _ = cache.admit(Some(with(3)));
        assert_eq!(cache.len_of(View::Decode), 2, "the capacity of a view");
        assert_eq!(cache.evictions(), 1);
        // The oldest by admission was key 2.
        assert!(matches!(
            cache.admit(Some(with(1))),
            Admission::Record(_) | Admission::Eager
        ));
        let _ = cache.admit(Some(with(2)));
        assert_eq!(cache.len_of(View::Decode), 2);
        // Another view has its own room.
        let _ = cache.admit(Some(verify(3, 1 << 8)));
        assert_eq!(cache.len_of(View::Verify { rows: 3 }), 1);
        assert_eq!(cache.len(), 3);
        assert_eq!(GraphCache::with_capacity(5).capacity(), 5);
        assert!(matches!(GraphCache::disabled().admit(Some(key(1))), Admission::Eager));
    }

    #[test]
    fn the_row_exact_mode_and_the_fused_choice_key_differently() {
        let exact = CaptureKey {
            row_exact: true,
            ..key(1 << 8)
        };
        assert_ne!(exact, key(1 << 8), "a row-exact quantum takes other routes");
        assert!(!exact.same_but_geometry(&key(1 << 8)));
        let fused = key(1 << 16 | 64);
        assert_ne!(fused, key(64), "the fused verifier is another launch");
        assert!(fused.same_but_geometry(&key(64)));
    }
}
