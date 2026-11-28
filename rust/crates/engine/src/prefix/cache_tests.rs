//! The cache's owner against an in-memory device: a sequence's buffers and the arena are byte vectors, and a test plays the
//! sequence's owner, running (or failing, or dropping) the jobs the cache hands it. Nothing waits: the mailbox's
//! hand-offs run on the calling thread, so every answer has arrived when a request returns.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_graph::Marker;
use euhedral_inference_cuda::PinnedBuffer;
use euhedral_inference_settings::KvFormat;

use super::fixtures::{arena, compact, drafter, tokens};
use super::{
    Buffer, CacheParts, CheckpointLayout, CopyJob, Direction, Ended, Hit, HostArena, NodeId, OpenError,
    PrefixCacheHandle, PrefixCacheStats, Resolve, RestoreError, Speculation, Submit,
};
use crate::generation::Spawn;
use crate::handoff::Handoff;
use crate::state::SequenceId;

pub(super) const FORMAT: KvFormat = KvFormat::Nvfp4;
pub(super) const CHUNK: u32 = 512;
const INTERVAL: u32 = 1024;
const SHIFT: u32 = 40;

/// A sequence's buffers, each a byte vector the address of a byte of leads to.
pub(super) struct Device {
    format: KvFormat,
    rows: u64,
    /// Every buffer a job asked this device the address of, in order.
    asked: Mutex<Vec<Buffer>>,
    memory: Mutex<HashMap<u64, Vec<u8>>>,
}

impl Device {
    pub(super) fn new(format: KvFormat, rows: u64) -> Arc<Device> {
        Arc::new(Device {
            format,
            rows,
            asked: Mutex::new(Vec::new()),
            memory: Mutex::new(HashMap::new()),
        })
    }

    fn id(buffer: Buffer) -> u64 {
        match buffer {
            Buffer::Convolution { layer } => 1 + 2 * layer as u64,
            Buffer::Recurrent { layer } => 2 + 2 * layer as u64,
            Buffer::KvPage { layer, page } => 1000 + 64 * layer as u64 + page as u64,
            Buffer::MtpPage { page } => 5000 + page as u64,
            Buffer::SeedRow => 6000,
            Buffer::DrafterKeys { layer } => 7000 + 2 * layer as u64,
            Buffer::DrafterValues { layer } => 7001 + 2 * layer as u64,
        }
    }

    /// Fills every buffer of `layout`'s checkpoint of `position` tokens with bytes only `salt` could have made.
    pub(super) fn fill(&self, layout: &CheckpointLayout, position: u32, salt: u8) {
        for copy in everything(layout, position) {
            let at = (Device::id(copy.buffer) << SHIFT) + copy.offset;
            // A buffer holds the same bytes at every position (the pages a checkpoint shares with its parent agree),
            // except the seed row, which is the hidden row of the checkpoint's own position.
            let own = if copy.buffer == Buffer::SeedRow {
                position as u8
            } else {
                0
            };
            let bytes: Vec<u8> = (0..copy.bytes)
                .map(|byte| {
                    (byte as u8)
                        .wrapping_mul(31)
                        .wrapping_add(salt)
                        .wrapping_add(Device::id(copy.buffer) as u8)
                        .wrapping_add(own)
                })
                .collect();
            self.write(at, &bytes);
        }
    }

    fn write(&self, at: u64, bytes: &[u8]) {
        let mut memory = self.memory.lock().unwrap();
        let buffer = memory.entry(at >> SHIFT).or_default();
        let from = (at & ((1 << SHIFT) - 1)) as usize;
        if buffer.len() < from + bytes.len() {
            buffer.resize(from + bytes.len(), 0);
        }
        buffer[from..from + bytes.len()].copy_from_slice(bytes);
    }

    pub(super) fn read(&self, at: u64, bytes: u64) -> Vec<u8> {
        let memory = self.memory.lock().unwrap();
        let from = (at & ((1 << SHIFT) - 1)) as usize;
        let mut out = vec![0; bytes as usize];
        if let Some(buffer) = memory.get(&(at >> SHIFT)) {
            for (slot, byte) in out.iter_mut().zip(buffer.iter().skip(from)) {
                *slot = *byte;
            }
        }
        out
    }
}

impl Resolve for Device {
    fn address(&self, buffer: Buffer, offset: u64, _bytes: u64) -> Option<u64> {
        self.asked.lock().unwrap().push(buffer);
        Some((Device::id(buffer) << SHIFT) + offset)
    }

    fn kv_format(&self) -> KvFormat {
        self.format
    }

    fn rows(&self) -> u64 {
        self.rows
    }

    fn ready(&self) -> Option<Marker> {
        None
    }
}

/// Every copy of a checkpoint of `position` tokens, as a chain of nodes of 256-token spans would make them, read as
/// the whole state: the GDN state, every page, the seed row and the MTP pages.
fn everything(layout: &CheckpointLayout, position: u32) -> Vec<super::SlotCopy> {
    let mut tree = super::PrefixTree::new(arena(256, layout.slot_bytes()));
    let root = tree.root();
    let prompt = tokens(0, position + 1);
    let node = tree
        .reserve(
            root,
            &prompt,
            position,
            Some(Speculation::Mtp),
            layout.slots(0, position, Some(Speculation::Mtp)),
        )
        .expect("room");
    layout.copies(tree.get(node).expect("reserved"))
}

/// The arena: bytes the jobs copy to and from.
struct Host {
    memory: Mutex<Vec<u8>>,
}

impl HostArena for Host {
    fn pinned(&self) -> Option<&PinnedBuffer> {
        None
    }
}

/// A cache over a device and a host arena, whose jobs wait for the test to run them.
pub(super) struct Rig {
    pub(super) cache: PrefixCacheHandle,
    host: Arc<Host>,
    pub(super) layout: CheckpointLayout,
    jobs: Arc<Mutex<VecDeque<CopyJob>>>,
}

pub(super) fn inline() -> Spawn {
    Box::new(|work| {
        work();
        true
    })
}

impl Rig {
    pub(super) fn new(slots: u32) -> Rig {
        Rig::with(slots, 1 << 20)
    }

    pub(super) fn with(slots: u32, piece_bytes: u64) -> Rig {
        let layout = CheckpointLayout::new(&compact(), FORMAT).expect("a layout");
        let arena = arena(slots, layout.slot_bytes());
        let host = Arc::new(Host {
            memory: Mutex::new(vec![0; (u64::from(slots) * layout.slot_bytes()) as usize]),
        });
        let cache = PrefixCacheHandle::open(
            CacheParts {
                layout: layout.clone(),
                arena,
                host: Arc::clone(&host) as Arc<dyn HostArena>,
                format: FORMAT,
                interval: INTERVAL,
                chunk: CHUNK,
                drafting: None,
                piece_bytes,
                first_serial: 1 << 40,
            },
            inline(),
        )
        .expect("opens");
        Rig {
            cache,
            host,
            layout,
            jobs: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    pub(super) fn submit(&self) -> Submit {
        let jobs = Arc::clone(&self.jobs);
        Box::new(move |job| jobs.lock().unwrap().push_back(job))
    }

    pub(super) fn lookup(&self, prompt: &[u32], kind: Option<Speculation>) -> Option<Hit> {
        let answer = Arc::new(Mutex::new(None));
        let told = Arc::clone(&answer);
        self.cache.lookup(prompt.to_vec().into(), kind, move |hit| {
            *told.lock().unwrap() = Some(hit)
        });
        answer.lock().unwrap().take().expect("a lookup is answered")
    }

    pub(super) fn capture(
        &self,
        device: &Arc<Device>,
        prompt: &[u32],
        parent: NodeId,
        end: u32,
        kind: Option<Speculation>,
    ) -> Arc<Mutex<Option<NodeId>>> {
        let answer = Arc::new(Mutex::new(None));
        let told = Arc::clone(&answer);
        self.cache.capture(
            SequenceId(0),
            Arc::clone(device) as Arc<dyn Resolve>,
            prompt.to_vec().into(),
            parent,
            end,
            kind,
            self.submit(),
            move |node| *told.lock().unwrap() = Some(node),
        );
        answer
    }

    pub(super) fn restore(
        &self,
        hit: Hit,
        device: &Arc<Device>,
        kind: Option<Speculation>,
        cancel: &Arc<AtomicBool>,
    ) -> Arc<Mutex<Option<Result<u32, RestoreError>>>> {
        let answer = Arc::new(Mutex::new(None));
        let told = Arc::clone(&answer);
        self.cache.restore(
            hit,
            SequenceId(1),
            Arc::clone(device) as Arc<dyn Resolve>,
            kind,
            Arc::clone(cancel),
            self.submit(),
            move |result| *told.lock().unwrap() = Some(result),
        );
        answer
    }

    pub(super) fn next_job(&self) -> CopyJob {
        self.jobs.lock().unwrap().pop_front().expect("a job was submitted")
    }

    pub(super) fn queued(&self) -> usize {
        self.jobs.lock().unwrap().len()
    }

    /// Runs `job` against `device` and the arena, and reports it done.
    pub(super) fn run(&self, mut job: CopyJob, device: &Device) {
        for piece in job.pieces().to_vec() {
            let mut host = self.host.memory.lock().unwrap();
            let range = piece.host as usize..(piece.host + piece.bytes) as usize;
            match job.direction() {
                Direction::ToHost => host[range].copy_from_slice(&device.read(piece.device, piece.bytes)),
                Direction::ToDevice => device.write(piece.device, &host[range]),
            }
        }
        job.finish(Ended::Done);
    }

    pub(super) fn run_all(&self, device: &Device) {
        while self.queued() > 0 {
            let job = self.next_job();
            self.run(job, device);
        }
    }

    pub(super) fn stats(&self) -> PrefixCacheStats {
        self.cache.stats()
    }
}

pub(super) fn answer<T>(slot: &Arc<Mutex<Option<T>>>) -> T {
    slot.lock().unwrap().take().expect("the request was answered")
}

/// Whether the two devices hold the same base state of `position` tokens.
pub(super) fn same_state(layout: &CheckpointLayout, position: u32, left: &Device, right: &Device) -> bool {
    base_state(layout, position, left, right, false)
}

/// Whether they hold the same state, the speculative part (the seed row and the MTP pages) included.
pub(super) fn same_speculative_state(layout: &CheckpointLayout, position: u32, left: &Device, right: &Device) -> bool {
    base_state(layout, position, left, right, true)
}

fn base_state(layout: &CheckpointLayout, position: u32, left: &Device, right: &Device, speculative: bool) -> bool {
    let speculation = |buffer| matches!(buffer, Buffer::SeedRow | Buffer::MtpPage { .. });
    everything(layout, position)
        .iter()
        .filter(|copy| speculative || !speculation(copy.buffer))
        .all(|copy| {
            let at = (Device::id(copy.buffer) << SHIFT) + copy.offset;
            left.read(at, copy.bytes) == right.read(at, copy.bytes)
        })
}

#[test]
fn a_capture_is_restored_into_another_sequence_byte_for_byte() {
    let rig = Rig::new(64);
    let prompt = tokens(0, 700);
    let source = Device::new(FORMAT, 1024);
    source.fill(&rig.layout, 512, 7);
    let captured = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    assert_eq!(rig.queued(), 1);
    assert!(
        captured.lock().unwrap().is_none(),
        "answered when the job ends, not before"
    );
    rig.run_all(&source);
    let node = answer(&captured);
    assert_ne!(node, rig.cache.root());
    assert_eq!(rig.stats().captures, 1);
    assert_eq!(rig.stats().nodes, 1);

    let hit = rig.lookup(&prompt, None).expect("the prefix is cached");
    assert_eq!(hit.position(), 512);
    assert_eq!(hit.leaf(), node);
    let target = Device::new(FORMAT, 1024);
    assert!(!same_state(&rig.layout, 512, &source, &target));
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(512));
    assert!(same_state(&rig.layout, 512, &source, &target));
    let stats = rig.stats();
    assert_eq!(
        (stats.lookups, stats.hits, stats.restored_tokens, stats.restores),
        (1, 1, 512, 1)
    );
}

#[test]
fn a_lookup_that_matches_nothing_is_a_miss_and_is_counted() {
    let rig = Rig::new(64);
    assert!(rig.lookup(&tokens(0, 700), None).is_none());
    let stats = rig.stats();
    assert_eq!((stats.lookups, stats.hits, stats.restored_tokens), (1, 0, 0));
}

#[test]
fn a_match_always_leaves_a_token_to_prefill() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    source.fill(&rig.layout, 512, 1);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&captured);
    assert!(
        rig.lookup(&tokens(0, 512), None).is_none(),
        "the prompt ends where the checkpoint does"
    );
    assert_eq!(
        rig.lookup(&tokens(0, 513), None).expect("one token is left").position(),
        512
    );
}

#[test]
fn a_span_already_stored_is_not_captured_again() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    source.fill(&rig.layout, 512, 1);
    let prompt = tokens(0, 600);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    let node = answer(&first);
    let again = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    assert_eq!(rig.queued(), 0, "nothing is copied for a stored span");
    assert_eq!(answer(&again), node);
    assert_eq!(rig.stats().captures, 1);
}

#[test]
fn a_second_checkpoint_extends_the_first() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 2048);
    let prompt = tokens(0, 1300);
    source.fill(&rig.layout, 512, 3);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    let parent = answer(&first);
    source.fill(&rig.layout, 1024, 3);
    let second = rig.capture(&source, &prompt, parent, 1024, None);
    rig.run_all(&source);
    let child = answer(&second);
    assert_ne!(child, parent);
    let hit = rig.lookup(&prompt, None).expect("hit");
    assert_eq!((hit.position(), hit.leaf()), (1024, child));
    let target = Device::new(FORMAT, 2048);
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(1024));
    assert!(same_state(&rig.layout, 1024, &source, &target));
}

#[test]
fn a_checkpoint_larger_than_the_arena_is_skipped_and_counted() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let captured = rig.capture(&source, &tokens(0, 1100), rig.cache.root(), 1024, None);
    assert_eq!(answer(&captured), rig.cache.root(), "the cursor stays where it was");
    assert_eq!(rig.queued(), 0);
    let stats = rig.stats();
    assert_eq!((stats.skipped, stats.failed, stats.captures), (1, 0, 0));
}

#[test]
fn a_full_arena_evicts_the_least_recently_used_checkpoint() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let one = layout.slots(0, 256, None) as u32;
    let rig = Rig::new(2 * one);
    let source = Device::new(FORMAT, 1024);
    let (a, b, c): (Vec<u32>, Vec<u32>, Vec<u32>) = (tokens(0, 300), tokens(1000, 300), tokens(2000, 300));
    for prompt in [&a, &b] {
        let captured = rig.capture(&source, prompt, rig.cache.root(), 256, None);
        rig.run_all(&source);
        assert_ne!(answer(&captured), rig.cache.root());
    }
    drop(rig.lookup(&a, None).expect("a is cached"));
    let third = rig.capture(&source, &c, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&third), rig.cache.root());
    assert!(rig.lookup(&b, None).is_none(), "b was the least recently used");
    assert!(rig.lookup(&a, None).is_some() && rig.lookup(&c, None).is_some());
    assert_eq!(rig.stats().evictions, 1);
}

#[test]
fn a_pinned_checkpoint_is_not_evicted_and_a_released_one_is() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let one = layout.slots(0, CHUNK, None) as u32;
    let rig = Rig::new(one);
    let source = Device::new(FORMAT, 1024);
    let (a, b) = (tokens(0, 300), tokens(1000, 300));
    let first = rig.capture(&source, &a, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let hit = rig.lookup(&a, None).expect("hit");
    let refused = rig.capture(&source, &b, rig.cache.root(), 256, None);
    assert_eq!(answer(&refused), rig.cache.root());
    assert_eq!(rig.stats().skipped, 1, "the only checkpoint is pinned");
    drop(hit);
    let stored = rig.capture(&source, &b, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root(), "the dropped hit let it go");
    assert!(rig.lookup(&a, None).is_none());
}

#[test]
fn a_closed_cache_stores_and_restores_nothing() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&first);
    let hit = rig.lookup(&prompt, None).expect("hit");
    assert!(rig.cache.close(Duration::from_secs(5)));
    assert!(rig.lookup(&prompt, None).is_none());
    let captured = rig.capture(&source, &prompt, rig.cache.root(), 256, None);
    assert_eq!(answer(&captured), rig.cache.root());
    assert_eq!(rig.queued(), 0);
    let restored = rig.restore(hit, &source, None, &Arc::new(AtomicBool::new(false)));
    assert_eq!(answer(&restored), Err(RestoreError::Closed));
    assert_eq!(rig.queued(), 0);
}

#[test]
fn a_cache_neither_stores_nor_restores_another_formats_state() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&first);
    let other = Device::new(KvFormat::Bf16, 1024);
    let stored = rig.capture(&other, &prompt, rig.cache.root(), 256, None);
    assert_eq!(answer(&stored), rig.cache.root());
    assert_eq!(rig.queued(), 0);
    let hit = rig.lookup(&prompt, None).expect("hit");
    let restored = rig.restore(hit, &other, None, &Arc::new(AtomicBool::new(false)));
    assert!(matches!(answer(&restored), Err(RestoreError::Format { .. })));
    assert_eq!(rig.queued(), 0);
}

#[test]
fn a_capture_past_the_rows_the_sequence_holds_is_not_taken() {
    let rig = Rig::new(64);
    let short = Device::new(FORMAT, 300);
    let captured = rig.capture(&short, &tokens(0, 600), rig.cache.root(), 512, None);
    assert_eq!(answer(&captured), rig.cache.root());
    assert_eq!(rig.queued(), 0);
}

#[test]
fn a_cancel_before_the_copies_unpins_without_an_error_or_a_job() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let (a, b) = (tokens(0, 300), tokens(1000, 300));
    let first = rig.capture(&source, &a, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let hit = rig.lookup(&a, None).expect("hit");
    let restored = rig.restore(hit, &source, None, &Arc::new(AtomicBool::new(true)));
    assert_eq!(answer(&restored), Err(RestoreError::Cancelled));
    assert_eq!(rig.queued(), 0);
    let stored = rig.capture(&source, &b, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root(), "the pin is gone");
}

#[test]
fn a_cancel_while_the_copies_run_unpins_and_is_not_counted_as_a_restore() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let (a, b) = (tokens(0, 300), tokens(1000, 300));
    let first = rig.capture(&source, &a, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let cancel = Arc::new(AtomicBool::new(false));
    let target = Device::new(FORMAT, 1024);
    let restored = rig.restore(rig.lookup(&a, None).expect("hit"), &target, None, &cancel);
    assert_eq!(rig.queued(), 1);
    cancel.store(true, Ordering::Release);
    rig.run_all(&target);
    assert_eq!(answer(&restored), Err(RestoreError::Cancelled));
    assert_eq!(rig.stats().restores, 0);
    let stored = rig.capture(&source, &b, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root(), "the pin is gone");
}

#[test]
fn a_restore_whose_copy_fails_unpins_and_answers() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let (a, b) = (tokens(0, 300), tokens(1000, 300));
    let first = rig.capture(&source, &a, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let restored = rig.restore(
        rig.lookup(&a, None).expect("hit"),
        &source,
        None,
        &Arc::new(AtomicBool::new(false)),
    );
    rig.next_job().finish(Ended::Failed("the lane failed".to_string()));
    assert_eq!(
        answer(&restored),
        Err(RestoreError::Copy("the lane failed".to_string()))
    );
    assert_eq!(rig.stats().restores, 0);
    let stored = rig.capture(&source, &b, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root());
}

#[test]
fn a_capture_whose_copy_fails_frees_its_slots_and_publishes_nothing() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let captured = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    assert!(rig.stats().used_bytes > 0, "the slots are leased while the job runs");
    rig.next_job().finish(Ended::Failed("the lane failed".to_string()));
    assert_eq!(answer(&captured), rig.cache.root());
    let stats = rig.stats();
    assert_eq!(
        (stats.failed, stats.captures, stats.nodes, stats.used_bytes),
        (1, 0, 0, 0)
    );
    assert!(rig.lookup(&prompt, None).is_none());
}

#[test]
fn a_job_dropped_without_running_ends_the_request_as_a_failure() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    drop(rig.next_job());
    assert_eq!(answer(&captured), rig.cache.root());
    assert_eq!(rig.stats().failed, 1);
    assert_eq!(rig.stats().used_bytes, 0);

    let first = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&first);
    let restored = rig.restore(
        rig.lookup(&tokens(0, 600), None).expect("hit"),
        &source,
        None,
        &Arc::new(AtomicBool::new(false)),
    );
    drop(rig.next_job());
    assert!(matches!(answer(&restored), Err(RestoreError::Copy(_))));
}

#[test]
fn a_hit_that_is_dropped_unpins() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let first = rig.capture(&source, &tokens(0, 300), rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    drop(rig.lookup(&tokens(0, 300), None).expect("hit"));
    let stored = rig.capture(&source, &tokens(1000, 300), rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root());
}

#[test]
fn a_node_with_speculative_state_matches_only_a_request_for_it() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let plain = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&plain);
    assert!(
        rig.lookup(&prompt, Some(Speculation::Mtp)).is_none(),
        "the node holds no speculative state"
    );
    let other = tokens(5000, 600);
    source.fill(&rig.layout, 512, 9);
    let speculative = rig.capture(&source, &other, rig.cache.root(), 512, Some(Speculation::Mtp));
    rig.run_all(&source);
    answer(&speculative);
    let hit = rig
        .lookup(&other, Some(Speculation::Mtp))
        .expect("the speculative node matches");
    let target = Device::new(FORMAT, 1024);
    let restored = rig.restore(hit, &target, Some(Speculation::Mtp), &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(512));
    assert!(
        same_speculative_state(&rig.layout, 512, &source, &target),
        "the seed row and the MTP pages came back too"
    );
}

#[test]
fn a_hit_without_the_speculative_state_asked_for_is_refused() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let plain = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&plain);
    let hit = rig.lookup(&prompt, None).expect("a plain lookup matches");
    let restored = rig.restore(hit, &source, Some(Speculation::Mtp), &Arc::new(AtomicBool::new(false)));
    assert_eq!(answer(&restored), Err(RestoreError::Speculation));
    assert_eq!(rig.queued(), 0);
}

#[test]
fn jobs_are_cut_into_pieces_no_larger_than_the_setting() {
    let rig = Rig::with(64, 1000);
    let source = Device::new(FORMAT, 1024);
    source.fill(&rig.layout, 512, 5);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    let job = rig.next_job();
    assert!(job.pieces().len() > 4);
    assert!(job.pieces().iter().all(|piece| piece.bytes <= 1000 && piece.bytes > 0));
    assert_eq!(job.direction(), Direction::ToHost);
    rig.run(job, &source);
    assert_ne!(answer(&captured), rig.cache.root());
    let hit = rig.lookup(&tokens(0, 600), None).expect("hit");
    let target = Device::new(FORMAT, 1024);
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(512));
    assert!(same_state(&rig.layout, 512, &source, &target));
}

#[test]
fn checkpoints_fall_on_the_interval_and_at_the_last_chunk_before_the_prompt_ends() {
    let rig = Rig::new(8);
    let wants = |end, len| rig.cache.wants_checkpoint(end, len);
    assert!(wants(INTERVAL, 5000), "on the interval");
    assert!(!wants(CHUNK, 5000), "off the interval and far from the end");
    assert!(wants(CHUNK, 900), "the last chunk before the prompt ends");
    assert!(!wants(1100, 1100), "the prompt ends there: nothing is left to prefill");
    assert!(wants(2560, 2600) && !wants(2048 + CHUNK, 5000));
    rig.cache.limit_checkpoints(1024);
    assert!(
        wants(1024, 5000) && !wants(2048, 5000),
        "past the limit nothing is wanted"
    );
    assert!(!wants(1280, 1300));
}

#[test]
fn a_cache_over_the_wrong_slot_size_or_an_off_grid_interval_does_not_open() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let host: Arc<dyn HostArena> = Arc::new(Host {
        memory: Mutex::new(Vec::new()),
    });
    let parts = |slot_bytes: u64, interval: u32| CacheParts {
        layout: layout.clone(),
        arena: arena(4, slot_bytes),
        host: Arc::clone(&host),
        format: FORMAT,
        interval,
        chunk: CHUNK,
        drafting: None,
        piece_bytes: 1 << 20,
        first_serial: 0,
    };
    let wrong = PrefixCacheHandle::open(parts(layout.slot_bytes() + 1, INTERVAL), inline());
    assert_eq!(
        wrong.err(),
        Some(OpenError::SlotBytes {
            arena: layout.slot_bytes() + 1,
            layout: layout.slot_bytes()
        })
    );
    let off = PrefixCacheHandle::open(parts(layout.slot_bytes(), 700), inline());
    assert_eq!(off.err(), Some(OpenError::Interval));
    assert!(PrefixCacheHandle::open(parts(layout.slot_bytes(), 0), inline()).is_err());
}

/// Parts over `layout` with an arena of `slots` slots and `drafting` state in the checkpoints.
fn parts_with(layout: &CheckpointLayout, slots: u32, drafting: Option<Speculation>) -> CacheParts {
    CacheParts {
        layout: layout.clone(),
        arena: arena(slots, layout.slot_bytes()),
        host: Arc::new(Host {
            memory: Mutex::new(Vec::new()),
        }),
        format: FORMAT,
        interval: INTERVAL,
        chunk: CHUNK,
        drafting,
        piece_bytes: 1 << 20,
        first_serial: 0,
    }
}

#[test]
fn a_cache_whose_arena_cannot_hold_one_base_checkpoint_does_not_open() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let needed = layout.slots(0, CHUNK, None);
    let short = needed as u32 - 1;
    let refused = PrefixCacheHandle::open(parts_with(&layout, short, None), inline()).err();
    assert_eq!(refused, Some(OpenError::BaseCapacity { needed, slots: short }));
    let message = refused.unwrap().to_string();
    assert!(
        message.contains(&needed.to_string()) && message.contains(&short.to_string()),
        "{message}"
    );
    assert!(PrefixCacheHandle::open(parts_with(&layout, needed as u32, None), inline()).is_ok());
}

#[test]
fn a_cache_whose_arena_cannot_hold_one_mtp_checkpoint_does_not_open_when_requests_draft_with_it() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let (base, needed) = (
        layout.slots(0, CHUNK, None),
        layout.slots(0, CHUNK, Some(Speculation::Mtp)),
    );
    assert!(needed > base, "the MTP state takes slots of its own");
    let short = needed as u32 - 1;
    let refused = PrefixCacheHandle::open(parts_with(&layout, short, Some(Speculation::Mtp)), inline()).err();
    assert_eq!(
        refused,
        Some(OpenError::Capacity {
            kind: Speculation::Mtp,
            needed,
            slots: short
        })
    );
    let message = refused.unwrap().to_string();
    assert!(
        message.contains(&needed.to_string()) && message.contains(&short.to_string()),
        "{message}"
    );
    assert!(PrefixCacheHandle::open(parts_with(&layout, needed as u32, Some(Speculation::Mtp)), inline()).is_ok());
    assert!(PrefixCacheHandle::open(parts_with(&layout, base as u32, None), inline()).is_ok());
}

#[test]
fn a_cache_for_requests_that_draft_with_state_the_layout_lacks_does_not_open() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let refused = PrefixCacheHandle::open(parts_with(&layout, 64, Some(Speculation::Dflash2)), inline()).err();
    assert_eq!(
        refused,
        Some(OpenError::NoState {
            kind: Speculation::Dflash2
        })
    );
}

#[test]
fn a_cache_whose_arena_cannot_hold_one_drafting_checkpoint_does_not_open() {
    let layout = CheckpointLayout::new(&compact(), FORMAT)
        .expect("layout")
        .with_drafter(&drafter(3, 2000))
        .expect("a drafter");
    let needed = layout.slots(0, CHUNK, Some(Speculation::Dflash2));
    let short = needed as u32 - 1;
    let refused = PrefixCacheHandle::open(parts_with(&layout, short, Some(Speculation::Dflash2)), inline()).err();
    assert_eq!(
        refused,
        Some(OpenError::Capacity {
            kind: Speculation::Dflash2,
            needed,
            slots: short
        })
    );
    let message = refused.unwrap().to_string();
    assert!(
        message.contains(&needed.to_string()) && message.contains(&short.to_string()),
        "{message}"
    );
    assert!(PrefixCacheHandle::open(parts_with(&layout, needed as u32, Some(Speculation::Dflash2)), inline()).is_ok());
}

#[test]
fn a_capture_copies_the_buffers_of_the_state_and_no_others() {
    // Written out from the model, not from the layout: `compact()` has the GDN state on layers 0, 2, 4 and 6 and the
    // attention caches on layers 1, 3, 5 and 7, and 512 tokens are two pages of 256.
    let expected: HashSet<Buffer> = [0, 2, 4, 6]
        .into_iter()
        .flat_map(|layer| [Buffer::Convolution { layer }, Buffer::Recurrent { layer }])
        .chain(
            [1, 3, 5, 7]
                .into_iter()
                .flat_map(|layer| [0, 1].map(|page| Buffer::KvPage { layer, page })),
        )
        .collect();
    assert_eq!(expected.len(), 8 + 8);
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let captured = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&captured);
    assert_eq!(
        source.asked.lock().unwrap().iter().copied().collect::<HashSet<_>>(),
        expected
    );
    let target = Device::new(FORMAT, 1024);
    let hit = rig.lookup(&prompt, None).expect("hit");
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(512));
    assert_eq!(
        target.asked.lock().unwrap().iter().copied().collect::<HashSet<_>>(),
        expected
    );
}

#[test]
fn a_request_without_speculation_restores_a_speculative_node_without_its_draft_state() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    source.fill(&rig.layout, 512, 9);
    let stored = rig.capture(&source, &prompt, rig.cache.root(), 512, Some(Speculation::Mtp));
    rig.run_all(&source);
    answer(&stored);
    let hit = rig
        .lookup(&prompt, None)
        .expect("a plain lookup matches a speculative node");
    let target = Device::new(FORMAT, 1024);
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    rig.run_all(&target);
    assert_eq!(answer(&restored), Ok(512));
    assert!(
        same_state(&rig.layout, 512, &source, &target),
        "the base state came back"
    );
    assert!(
        !target
            .asked
            .lock()
            .unwrap()
            .iter()
            .any(|buffer| matches!(buffer, Buffer::MtpPage { .. } | Buffer::SeedRow)),
        "the draft's state was left alone"
    );
}

#[test]
fn a_restore_into_pages_that_are_not_reserved_is_refused_without_a_job() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 300);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let short = Device::new(FORMAT, 100);
    let hit = rig.lookup(&prompt, None).expect("hit");
    let restored = rig.restore(hit, &short, None, &Arc::new(AtomicBool::new(false)));
    assert_eq!(
        answer(&restored),
        Err(RestoreError::Unreserved {
            rows: 100,
            position: 256
        })
    );
    assert_eq!(rig.queued(), 0);
    let other = rig.capture(&source, &tokens(1000, 300), rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&other), rig.cache.root(), "the refusal unpinned the match");
}

#[test]
fn a_restore_that_fails_sets_the_requests_kill_flag_before_it_answers() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let first = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    rig.run_all(&source);
    answer(&first);
    for how in 0..3 {
        let kill = Arc::new(AtomicBool::new(false));
        let hit = rig.lookup(&prompt, None).expect("hit");
        let restored = rig.restore(hit, &source, None, &kill);
        let mut job = rig.next_job();
        match how {
            0 => job.finish(Ended::Failed("the lane failed".to_string())),
            1 => drop(job),
            _ => job.finish(Ended::Cancelled),
        }
        let flag = kill.load(Ordering::Acquire);
        match (how, answer(&restored)) {
            (0, Err(RestoreError::Copy(why))) => assert_eq!(why, "the lane failed"),
            (1, Err(RestoreError::Copy(_))) => {}
            (2, Err(RestoreError::Cancelled)) => {}
            (_, other) => panic!("case {how}: {other:?}"),
        }
        assert_eq!(
            flag,
            how != 2,
            "case {how}: a failed copy kills the request, a cancel needs no kill"
        );
    }
}

#[test]
fn a_restores_match_stays_pinned_until_its_job_ended() {
    let layout = CheckpointLayout::new(&compact(), FORMAT).expect("layout");
    let rig = Rig::new(layout.slots(0, CHUNK, None) as u32);
    let source = Device::new(FORMAT, 1024);
    let (a, b) = (tokens(0, 300), tokens(1000, 300));
    let first = rig.capture(&source, &a, rig.cache.root(), 256, None);
    rig.run_all(&source);
    answer(&first);
    let target = Device::new(FORMAT, 1024);
    let hit = rig.lookup(&a, None).expect("hit");
    let restored = rig.restore(hit, &target, None, &Arc::new(AtomicBool::new(false)));
    // The job is built and submitted but has not run: its slots are still being read.
    let job = rig.next_job();
    let refused = rig.capture(&source, &b, rig.cache.root(), 256, None);
    assert_eq!(
        answer(&refused),
        rig.cache.root(),
        "the only checkpoint is pinned by the restore"
    );
    assert_eq!(rig.stats().skipped, 1);
    assert_eq!(rig.stats().evictions, 0);
    rig.run(job, &target);
    assert_eq!(answer(&restored), Ok(256));
    let stored = rig.capture(&source, &b, rig.cache.root(), 256, None);
    rig.run_all(&source);
    assert_ne!(answer(&stored), rig.cache.root(), "the ended job let it go");
}

#[test]
fn a_job_runs_in_groups_and_only_the_last_reports() {
    let rig = Rig::with(64, 1000);
    let source = Device::new(FORMAT, 1024);
    let prompt = tokens(0, 600);
    let captured = rig.capture(&source, &prompt, rig.cache.root(), 512, None);
    let job = rig.next_job();
    let total: u64 = job.pieces().iter().map(|piece| piece.bytes).sum();
    let mut groups = job.split();
    assert!(groups.len() > 2);
    let bytes = |group: &CopyJob| group.pieces().iter().map(|piece| piece.bytes).sum::<u64>();
    assert_eq!(groups.iter().map(bytes).sum::<u64>(), total);
    assert!(groups.iter().all(|group| bytes(group) <= 1000 && bytes(group) > 0));
    let last = groups.len() - 1;
    assert!(
        groups
            .iter()
            .enumerate()
            .all(|(index, group)| group.is_last() == (index == last))
    );
    for group in &mut groups[..last] {
        group.finish(Ended::Done);
    }
    assert!(
        captured.lock().unwrap().is_none(),
        "the groups before the last say nothing"
    );
    groups[last].finish(Ended::Done);
    assert_ne!(answer(&captured), rig.cache.root());
    assert_eq!(rig.stats().captures, 1);
}

#[test]
fn a_group_that_fails_fails_the_job_at_the_last_group() {
    let rig = Rig::with(64, 1000);
    let source = Device::new(FORMAT, 1024);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    let mut groups = rig.next_job().split();
    let last = groups.len() - 1;
    groups[0].finish(Ended::Failed("the first group failed".to_string()));
    assert!(
        groups[1].skipped() && groups[last].skipped(),
        "the groups after a failure copy nothing"
    );
    for group in &mut groups[1..last] {
        group.finish(Ended::Done);
    }
    assert!(captured.lock().unwrap().is_none());
    groups[last].finish(Ended::Done);
    assert_eq!(answer(&captured), rig.cache.root());
    let stats = rig.stats();
    assert_eq!((stats.failed, stats.captures, stats.used_bytes), (1, 0, 0));
}

/// How long a test waits for something that must happen before it calls it a hang.
const HANG_GUARD: Duration = Duration::from_secs(30);

#[test]
fn closing_waits_for_a_copy_in_flight_and_then_frees_the_cache() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    let mut job = rig.next_job();
    let ended = AtomicBool::new(false);
    let closed = thread::scope(|scope| {
        scope.spawn(|| {
            // A cache that has taken the shutdown no longer counts a lookup: the copy is still in flight then.
            let deadline = Instant::now() + HANG_GUARD;
            loop {
                let before = rig.stats().lookups;
                let told = Arc::new(Handoff::new());
                let answer = Arc::clone(&told);
                rig.cache.lookup(tokens(0, 8).into(), None, move |_| answer.send(()));
                told.wait(HANG_GUARD).expect("the lookup was never answered");
                if rig.stats().lookups == before {
                    break;
                }
                assert!(Instant::now() < deadline, "the cache never took the shutdown");
                thread::yield_now();
            }
            ended.store(true, Ordering::SeqCst);
            job.finish(Ended::Done);
        });
        let closed = rig.cache.close(HANG_GUARD);
        assert!(
            ended.load(Ordering::SeqCst),
            "the shutdown returned while the copy was in flight"
        );
        closed
    });
    assert!(closed, "the copy ended inside the bound");
    assert_ne!(
        answer(&captured),
        rig.cache.root(),
        "the copy in flight was published, not abandoned"
    );
}

#[test]
fn closing_abandons_a_copy_that_does_not_end_within_the_bound() {
    let rig = Rig::new(64);
    let source = Device::new(FORMAT, 1024);
    let captured = rig.capture(&source, &tokens(0, 600), rig.cache.root(), 512, None);
    let job = rig.next_job();
    assert!(!rig.cache.close(Duration::from_millis(50)));
    assert_eq!(answer(&captured), rig.cache.root(), "the pending capture was failed");
    assert_eq!(rig.stats().failed, 1);
    drop(job);
}
