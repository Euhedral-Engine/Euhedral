//! The store's contracts on synthetic expert banks in a temporary artifact: a record's parts and their join, the
//! refusals, cancellation, the tier's rules, and the settle records a cache owner applies. The reads are made by a fake
//! port that holds each one until a test releases it (so an order, a failure or a leak is forced, never waited for),
//! and by the real ring, driven by hand, for the file and the positional fallback.

#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the artifact is written and read with blocking calls, and the fake port's queue is a Mutex"
)]

use std::collections::BTreeSet;
use std::collections::VecDeque;
use std::fs::write;
use std::io;
use std::mem::forget;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_inference_edrl::{
    ComponentGroup, ExpertBank, ExpertProjection, ExpertRecord, TensorDataType, WeightFormat, WeightLayout,
};
use euhedral_io::{AsyncIo, AsyncIoOptions, Continuation, IoDoneOwned};

use super::*;
use crate::residency::disk::{ArtifactFile, DiskReadPiece};
use euhedral_inference_settings::Auto;

const PAGE: u64 = 4096;
/// A hang deadline: the bound of a hand-driven loop whose failure names what never ended.
const HANG: Duration = Duration::from_secs(60);

fn content(position: usize) -> u8 {
    (position * 7 + position / 251) as u8
}

/// The synthetic artifact: bank 0 has four records of 10000 bytes (each in a 12288-byte cell), bank 1 three of 4096,
/// bank 2 one of 5000 at the very end of the file (a direct read of its tail would pass the end).
struct Rig {
    _dir: tempfile::TempDir,
    path: PathBuf,
    bytes: Vec<u8>,
    banks: Vec<ExpertBank>,
}

fn bank(name: &str, records: Vec<ExpertRecord>) -> ExpertBank {
    let projection = ExpertProjection {
        name: "gate".to_owned(),
        shape: vec![1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        record_offset: 0,
        bytes: 1,
    };
    ExpertBank::new(name, ComponentGroup::RoutedExpert, 0, vec![projection], records).expect("a bank")
}

impl Rig {
    fn new() -> Rig {
        let mut cells = Vec::new();
        let mut offset = PAGE;
        let mut plan = |count: usize, bytes: u64, cell: u64| {
            let first = offset;
            offset += count as u64 * cell;
            (0..count)
                .map(move |i| (first + i as u64 * cell, bytes))
                .collect::<Vec<_>>()
        };
        cells.push(plan(4, 10000, 12288));
        cells.push(plan(3, 4096, 4096));
        cells.push(plan(1, 5000, 5000));
        let file_bytes = offset as usize;
        let bytes: Vec<u8> = (0..file_bytes).map(content).collect();
        let banks = cells
            .iter()
            .enumerate()
            .map(|(index, records)| {
                bank(
                    &format!("bank{index}"),
                    records
                        .iter()
                        .map(|&(offset, len)| ExpertRecord {
                            offset,
                            bytes: len,
                            crc32: crc32fast::hash(&bytes[offset as usize..(offset + len) as usize]),
                        })
                        .collect(),
                )
            })
            .collect();
        let dir = tempfile::tempdir().expect("a directory");
        let path = dir.path().join("artifact.bin");
        write(&path, &bytes).expect("the artifact is written");
        Rig {
            _dir: dir,
            path,
            bytes,
            banks,
        }
    }

    fn layout(&self) -> Arc<ExpertLayout> {
        Arc::new(ExpertLayout::new(&self.banks, self.bytes.len() as u64).expect("a layout"))
    }

    fn record(&self, id: ExpertId) -> &[u8] {
        let record = self.layout().record(id).expect("a record");
        &self.bytes[record.offset as usize..(record.offset + record.bytes) as usize]
    }
}

fn id(bank: u32, expert: u32) -> ExpertId {
    ExpertId { bank, expert }
}

// ---------------------------------------------------------------- the fake port

/// What a released read does.
enum Fault {
    None,
    Short(usize),
    Os(i32),
    Leak,
}

struct Held {
    offset: u64,
    memory: WritePart,
    then: Continuation<IoDoneOwned<WritePart>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Started1 {
    offset: u64,
    len: usize,
    direct: bool,
}

struct FakePort {
    /// The artifact's bytes, which the reads copy from.
    bytes: Vec<u8>,
    direct: bool,
    held: Mutex<VecDeque<Held>>,
    starts: Mutex<Vec<Started1>>,
    /// Starts from this one on are refused.
    refuse_from: AtomicUsize,
    /// Set on the first start (a cancel that arrives while the parts are being submitted).
    cancel_on_start: Mutex<Option<Arc<AtomicBool>>>,
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl FakePort {
    fn new(rig: &Rig, direct: bool) -> Arc<FakePort> {
        Arc::new(FakePort {
            bytes: rig.bytes.clone(),
            direct,
            held: Mutex::new(VecDeque::new()),
            starts: Mutex::new(Vec::new()),
            refuse_from: AtomicUsize::new(usize::MAX),
            cancel_on_start: Mutex::new(None),
        })
    }

    fn starts(&self) -> Vec<Started1> {
        locked(&self.starts).clone()
    }

    fn held(&self) -> usize {
        locked(&self.held).len()
    }

    /// Ends the `which`-th held read as `fault` says and runs the frame that ends it.
    fn release(&self, which: usize, fault: Fault) {
        let Held {
            offset,
            mut memory,
            then,
        } = locked(&self.held).remove(which).expect("a read is held");
        let result = match fault {
            Fault::Os(code) => Err(io::Error::from_raw_os_error(code)),
            Fault::Leak => {
                let mut frame = then(IoDoneOwned {
                    buf: None,
                    result: Err(io::Error::other("the ring broke")),
                    filled: 0,
                });
                forget(memory);
                return frame.execute().expect("the frame runs");
            }
            Fault::Short(moved) => {
                memory.bytes_mut()[..moved].copy_from_slice(&self.bytes[offset as usize..][..moved]);
                Ok(moved)
            }
            Fault::None => {
                let len = memory.len();
                memory
                    .bytes_mut()
                    .copy_from_slice(&self.bytes[offset as usize..][..len]);
                Ok(len)
            }
        };
        let filled = result.as_ref().map_or(0, |&moved| moved);
        let mut frame = then(IoDoneOwned {
            buf: Some(memory),
            result,
            filled,
        });
        frame.execute().expect("the frame runs");
    }

    fn release_all(&self) {
        while self.held() > 0 {
            self.release(0, Fault::None);
        }
    }
}

impl PartPort for FakePort {
    fn start(
        &self,
        offset: u64,
        direct: bool,
        memory: WritePart,
        then: Continuation<IoDoneOwned<WritePart>>,
    ) -> io::Result<()> {
        let count = {
            let mut starts = locked(&self.starts);
            starts.push(Started1 {
                offset,
                len: memory.len(),
                direct,
            });
            starts.len()
        };
        if let Some(cancel) = locked(&self.cancel_on_start).as_ref() {
            cancel.store(true, Ordering::Release);
        }
        if count > self.refuse_from.load(Ordering::Acquire) {
            return Err(io::Error::other("the ring refused the read"));
        }
        locked(&self.held).push_back(Held { offset, memory, then });
        Ok(())
    }

    fn has_direct(&self) -> bool {
        self.direct
    }
}

/// What a load told, and how often.
#[derive(Default)]
struct Probe {
    calls: AtomicUsize,
    outcome: OnceLock<ReadOutcome>,
    end: OnceLock<LoadEnd>,
}

impl Probe {
    fn done(self: &Arc<Self>) -> Done {
        let probe = Arc::clone(self);
        Box::new(move |outcome| {
            probe.calls.fetch_add(1, Ordering::SeqCst);
            let _ = probe.outcome.set(outcome);
        })
    }

    fn load_done(self: &Arc<Self>) -> Box<dyn FnOnce(LoadEnd) + Send> {
        let probe = Arc::clone(self);
        Box::new(move |end| {
            probe.calls.fetch_add(1, Ordering::SeqCst);
            let _ = probe.end.set(end);
        })
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

use super::source::Done;
use crate::residency::offload::HOST_ALIGNMENT;

fn source(rig: &Rig, port: &Arc<FakePort>, split: usize, verify: bool) -> FileRecordSource {
    FileRecordSource::new(rig.layout(), Arc::clone(port) as Arc<dyn PartPort>, split, verify).expect("a source")
}

fn arena(rig: &Rig, slots: usize) -> Arc<HostArena> {
    HostArena::new(slots, rig.layout().slot_bytes()).expect("an arena")
}

/// Starts a bypass-style read of `target` into slot `slot` of `memory`, returning the frame when one is handed back.
fn read(
    source: &FileRecordSource,
    memory: &Arc<HostArena>,
    slot: usize,
    target: ExpertId,
    copy_to: Option<(&Arc<HostArena>, usize)>,
    cancel: &Arc<AtomicBool>,
    probe: &Arc<Probe>,
) -> Result<Started, ExpertError> {
    source.begin(ReadLoad {
        id: target,
        destination: memory.write(slot).expect("the slot is free"),
        copy_to: copy_to.map(|(arena, slot)| arena.write(slot).expect("the slot is free")),
        cancel: Arc::clone(cancel),
        done: probe.done(),
    })
}

fn no_cancel() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

fn slot_prefix(memory: &Arc<HostArena>, slot: usize, len: usize) -> Vec<u8> {
    memory.read(slot).expect("the slot is readable").bytes()[..len].to_vec()
}

// ---------------------------------------------------------------- layout and arena

#[test]
fn a_record_outside_the_file_is_refused_naming_its_bank_and_expert() {
    let rig = Rig::new();
    let error = ExpertLayout::new(&rig.banks, rig.bytes.len() as u64 - 1).expect_err("the last record passes the end");
    assert_eq!(
        error,
        ExpertError::OutsideFile {
            bank: "bank2".to_owned(),
            expert: 0,
            offset: rig.bytes.len() as u64 - 5000,
            bytes: 5000,
            file_bytes: rig.bytes.len() as u64 - 1,
        }
    );
    assert!(error.to_string().contains("bank2"), "{error}");
    assert_eq!(ExpertLayout::new(&[], 10).expect_err("no banks"), ExpertError::NoBanks);
}

#[test]
fn keys_and_ids_map_to_each_other_across_banks_and_unknown_ids_are_refused() {
    let layout = Rig::new().layout();
    assert_eq!(layout.keys(), 8);
    for key in 0..layout.keys() {
        assert_eq!(layout.key(layout.id_of(key)).expect("a key"), key);
    }
    assert_eq!(layout.key(id(1, 0)).expect("a key"), 4);
    assert!(matches!(
        layout.key(id(3, 0)),
        Err(ExpertError::NoSuchBank { bank: 3, banks: 3 })
    ));
    assert!(matches!(
        layout.key(id(1, 3)),
        Err(ExpertError::NoSuchExpert {
            bank: 1,
            expert: 3,
            experts: 3
        })
    ));
    assert_eq!(layout.slot_bytes(), 12288);
}

#[test]
fn shards_are_chosen_by_the_hash_the_cache_shard_uses() {
    // Fixed values of the hash, so a cache shard's owner and the tier cannot drift apart.
    let owners: Vec<usize> = (0..8).map(|key| shard_of(key, 3)).collect();
    assert_eq!(owners, [0, 1, 2, 1, 2, 1, 2, 1]);
    assert!((0..1000).all(|key| shard_of(key, 5) < 5));
}

#[test]
fn a_slot_leased_for_writing_refuses_every_other_lease_until_its_last_part_is_gone() {
    let memory = HostArena::new(2, 4096).expect("an arena");
    let mut write = memory.write(0).expect("a write lease");
    assert!(matches!(memory.write(0), Err(ExpertError::SlotBusy { slot: 0, .. })));
    assert!(matches!(memory.read(0), Err(ExpertError::SlotBusy { slot: 0, .. })));
    let part = write.cut(100).expect("a part");
    assert!(write.bytes_mut().is_none(), "a part is out");
    drop(write);
    assert!(memory.is_leased(0), "the part holds the lease");
    drop(part);
    assert!(!memory.is_leased(0));
    let first = memory.read(0).expect("a read lease");
    let second = memory.read(0).expect("readers share");
    assert!(matches!(memory.write(0), Err(ExpertError::SlotBusy { .. })));
    drop((first, second));
    assert!(memory.write(0).is_ok());
    assert!(matches!(
        memory.write(2),
        Err(ExpertError::SlotOutOfRange { slot: 2, slots: 2 })
    ));
}

#[test]
fn parts_are_cut_in_order_without_overlap_and_never_past_the_slot() {
    let memory = HostArena::new(1, 4096).expect("an arena");
    let mut write = memory.write(0).expect("a write lease");
    let mut first = write.cut(4000).expect("a part");
    let mut second = write.cut(96).expect("a part");
    assert!(write.cut(1).is_none(), "the slot is cut up");
    first.bytes_mut().fill(1);
    second.bytes_mut().fill(2);
    drop((first, second));
    let bytes = write.bytes_mut().expect("no part is out");
    assert_eq!((bytes[3999], bytes[4000], bytes[4095]), (1, 2, 2));
}

// ---------------------------------------------------------------- the parts and their join

#[test]
fn a_record_is_read_in_page_aligned_parts_that_join_into_one_end() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 3, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    let started = read(&source, &memory, 0, id(0, 1), None, &no_cancel(), &probe).expect("started");
    assert!(matches!(started, Started::Reading));
    // 10000 bytes in 3 parts: chunks of 4096 (ceil(10000/3) = 3334, up to a page); the last is the 1808-byte tail,
    // read as a whole page because the cell has the room.
    let cell = PAGE + 12288;
    assert_eq!(
        port.starts(),
        [
            Started1 {
                offset: cell,
                len: 4096,
                direct: true
            },
            Started1 {
                offset: cell + 4096,
                len: 4096,
                direct: true
            },
            Started1 {
                offset: cell + 8192,
                len: 4096,
                direct: true
            },
        ]
    );
    // Out of order: the join waits for all of them, and fires once.
    port.release(2, Fault::None);
    port.release(0, Fault::None);
    assert_eq!(probe.calls(), 0, "a part is still out");
    assert!(memory.is_leased(0));
    port.release(0, Fault::None);
    assert_eq!(probe.calls(), 1);
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
    assert!(!memory.is_leased(0), "the lease is let go before the end is told");
    assert_eq!(slot_prefix(&memory, 0, 10000), rig.record(id(0, 1)));
}

#[test]
fn a_record_whose_bytes_do_not_end_on_a_page_is_read_buffered_where_a_whole_page_would_pass_the_file() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 1, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(2, 0), None, &no_cancel(), &probe).expect("started");
    assert_eq!(port.starts().len(), 1);
    assert!(!port.starts()[0].direct);
    assert_eq!(port.starts()[0].len, 5000);
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
    assert_eq!(slot_prefix(&memory, 0, 5000), rig.record(id(2, 0)));
}

#[test]
fn without_a_direct_handle_every_part_is_buffered() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, false);
    let source = source(&rig, &port, 2, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 0), None, &no_cancel(), &probe).expect("started");
    assert!(port.starts().iter().all(|start| !start.direct));
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
}

#[test]
fn a_fill_copies_each_part_to_staging_as_it_arrives() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 3, true);
    let (tier, staging) = (arena(&rig, 1), arena(&rig, 1));
    let probe = Arc::new(Probe::default());
    read(&source, &tier, 0, id(0, 2), Some((&staging, 0)), &no_cancel(), &probe).expect("started");
    port.release(1, Fault::None);
    {
        let record = rig.record(id(0, 2));
        let copied = staging.is_leased(0);
        assert!(copied, "the staging slot is still leased by the load");
        assert_eq!(probe.calls(), 0);
        port.release(0, Fault::None);
        port.release(0, Fault::None);
        assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
        assert_eq!(slot_prefix(&staging, 0, 10000), record);
        assert_eq!(slot_prefix(&tier, 0, 10000), record);
    }
}

#[test]
fn a_part_refused_by_the_ring_fails_the_load_once_after_the_parts_out_have_ended() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    port.refuse_from.store(1, Ordering::Release);
    let source = source(&rig, &port, 3, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    let started = read(&source, &memory, 0, id(0, 0), None, &no_cancel(), &probe).expect("started");
    assert!(matches!(started, Started::Reading));
    // The second start was refused and the third went on; the first two held reads are the only ones out.
    assert_eq!(probe.calls(), 0, "a read is still out");
    assert!(memory.is_leased(0), "its memory is still the ring's");
    port.release_all();
    assert_eq!(probe.calls(), 1);
    assert!(matches!(
        probe.outcome.get(),
        Some(ReadOutcome::Failed(ExpertError::Read { bank, expert: 0, .. })) if bank == "bank0"
    ));
    assert!(!memory.is_leased(0));
}

#[test]
fn when_every_part_ends_before_the_start_returns_the_start_hands_back_the_frame_that_ends_the_load() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    port.refuse_from.store(0, Ordering::Release);
    let source = source(&rig, &port, 2, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    let started = read(&source, &memory, 0, id(0, 0), None, &no_cancel(), &probe).expect("started");
    assert_eq!(probe.calls(), 0, "the end is never told inside the start");
    let Started::Frame(mut frame) = started else {
        panic!("every start was refused: the load ended inside the start, and its frame was not handed back");
    };
    frame.execute().expect("the frame runs");
    assert_eq!(probe.calls(), 1);
    assert!(matches!(
        probe.outcome.get(),
        Some(ReadOutcome::Failed(ExpertError::Read { .. }))
    ));
}

#[test]
fn a_leaked_read_makes_the_load_lost_and_keeps_the_slot_leased_for_good() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 2, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 0), None, &no_cancel(), &probe).expect("started");
    port.release(0, Fault::Leak);
    port.release(0, Fault::None);
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Lost));
    assert!(memory.is_leased(0), "the kernel may still write the slot");
    assert!(matches!(memory.write(0), Err(ExpertError::SlotBusy { .. })));
}

#[test]
fn a_short_read_fails_the_load_naming_the_record_and_the_bytes() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 1, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(1, 2), None, &no_cancel(), &probe).expect("started");
    port.release(0, Fault::Short(100));
    let Some(ReadOutcome::Failed(error)) = probe.outcome.get() else {
        panic!("a short read did not fail the load: {:?}", probe.outcome.get());
    };
    assert!(matches!(
        error,
        ExpertError::Short { bank, expert: 2, wanted: 4096, got: 100, .. } if bank == "bank1"
    ));
}

/// What the platform answers a direct read it cannot align (what `refuses_alignment` checks): EINVAL, or
/// ERROR_INVALID_PARAMETER.
#[cfg(windows)]
const ALIGNMENT_ERRNO: i32 = 87;
#[cfg(not(windows))]
const ALIGNMENT_ERRNO: i32 = 22;

#[test]
fn a_direct_read_the_file_system_cannot_align_is_made_again_buffered_once() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 1, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 3), None, &no_cancel(), &probe).expect("started");
    assert!(port.starts()[0].direct);
    port.release(0, Fault::Os(ALIGNMENT_ERRNO));
    assert_eq!(probe.calls(), 0, "the retry is out");
    let starts = port.starts();
    assert_eq!(starts.len(), 2);
    assert!(!starts[1].direct);
    assert_eq!(starts[1].len, 10000, "the retry reads the record's own bytes");
    port.release(0, Fault::None);
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
    assert_eq!(slot_prefix(&memory, 0, 10000), rig.record(id(0, 3)));
}

#[test]
fn any_other_error_fails_the_load_without_a_retry() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 1, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 0), None, &no_cancel(), &probe).expect("started");
    port.release(0, Fault::Os(5));
    assert_eq!(port.starts().len(), 1);
    assert!(matches!(
        probe.outcome.get(),
        Some(ReadOutcome::Failed(ExpertError::Read { .. }))
    ));
}

// ---------------------------------------------------------------- a damaged record

#[test]
fn a_damaged_record_is_refused_naming_its_bank_expert_and_both_checksums() {
    let mut rig = Rig::new();
    let record = rig.layout().record(id(0, 2)).expect("a record");
    let expected = record.crc32;
    rig.bytes[record.offset as usize + 7000] ^= 0x40;
    write(&rig.path, &rig.bytes).expect("the artifact is damaged");
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 2, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 2), None, &no_cancel(), &probe).expect("started");
    port.release_all();
    let Some(ReadOutcome::Failed(ExpertError::Damaged {
        bank,
        expert,
        expected: wanted,
        found,
    })) = probe.outcome.get()
    else {
        panic!("a damaged record was not refused: {:?}", probe.outcome.get());
    };
    assert_eq!((bank.as_str(), *expert, *wanted), ("bank0", 2, expected));
    assert_ne!(*found, expected);
    // Its neighbour reads clean, and without verification the damage is not looked for.
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 1), None, &no_cancel(), &probe).expect("started");
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
    let lax = FileRecordSource::new(rig.layout(), Arc::clone(&port) as Arc<dyn PartPort>, 2, false).expect("a source");
    let probe = Arc::new(Probe::default());
    read(&lax, &memory, 0, id(0, 2), None, &no_cancel(), &probe).expect("started");
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete));
}

// ---------------------------------------------------------------- refusals and cancellation

#[test]
fn an_unknown_expert_a_slot_too_small_and_a_zero_split_are_refused_before_anything_starts() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 2, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    let refused = read(&source, &memory, 0, id(0, 9), None, &no_cancel(), &probe);
    assert!(matches!(refused, Err(ExpertError::NoSuchExpert { expert: 9, .. })));
    let small = HostArena::new(1, 4096).expect("an arena");
    let refused = read(&source, &small, 0, id(0, 0), None, &no_cancel(), &probe);
    assert!(matches!(
        refused,
        Err(ExpertError::SlotTooSmall {
            slot_bytes: 4096,
            needed: 12288,
            ..
        })
    ));
    assert!(port.starts().is_empty());
    assert_eq!(probe.calls(), 0);
    assert!(
        !memory.is_leased(0) && !small.is_leased(0),
        "a refused load keeps no lease"
    );
    let error = FileRecordSource::new(rig.layout(), port as Arc<dyn PartPort>, 0, true).expect_err("a zero split");
    assert_eq!(error, ExpertError::BadSplit { split: 0 });
}

#[test]
fn a_bypass_cancelled_before_it_starts_submits_nothing_and_ends_cancelled() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 3, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    let cancel = no_cancel();
    cancel.store(true, Ordering::Release);
    let started = read(&source, &memory, 0, id(0, 0), None, &cancel, &probe).expect("started");
    let Started::Frame(mut frame) = started else {
        panic!("no read was submitted, yet the load did not end");
    };
    assert!(port.starts().is_empty());
    frame.execute().expect("the frame runs");
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Cancelled { filled: false }));
    assert!(!memory.is_leased(0));
}

#[test]
fn a_cancel_that_arrives_between_two_submissions_stops_the_rest_and_the_reads_out_keep_their_memory() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let cancel = no_cancel();
    *locked(&port.cancel_on_start) = Some(Arc::clone(&cancel));
    let source = source(&rig, &port, 3, true);
    let memory = arena(&rig, 1);
    let probe = Arc::new(Probe::default());
    read(&source, &memory, 0, id(0, 0), None, &cancel, &probe).expect("started");
    assert_eq!(port.starts().len(), 1, "the parts after the cancel were not submitted");
    assert_eq!(probe.calls(), 0);
    assert!(memory.is_leased(0), "the read that is out may still write the slot");
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Cancelled { filled: false }));
    assert!(!memory.is_leased(0), "the slot is free once the read returned");
}

#[test]
fn a_fill_that_is_cancelled_before_any_copy_reads_on_into_the_tier_and_ends_cancelled_not_complete() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 2, true);
    let (tier, staging) = (arena(&rig, 1), arena(&rig, 1));
    let probe = Arc::new(Probe::default());
    let cancel = no_cancel();
    read(&source, &tier, 0, id(0, 0), Some((&staging, 0)), &cancel, &probe).expect("started");
    cancel.store(true, Ordering::Release);
    port.release_all();
    assert_eq!(
        probe.outcome.get(),
        Some(&ReadOutcome::Cancelled { filled: true }),
        "the record reached the tier, not staging"
    );
    assert_eq!(slot_prefix(&tier, 0, 10000), rig.record(id(0, 0)));
    assert!(
        slot_prefix(&staging, 0, 10000).iter().all(|&byte| byte == 0),
        "staging was not written"
    );
}

#[test]
fn a_fill_that_is_cancelled_between_two_copies_ends_cancelled_though_staging_holds_a_part() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = source(&rig, &port, 2, true);
    let (tier, staging) = (arena(&rig, 1), arena(&rig, 1));
    let probe = Arc::new(Probe::default());
    let cancel = no_cancel();
    read(&source, &tier, 0, id(0, 0), Some((&staging, 0)), &cancel, &probe).expect("started");
    port.release(0, Fault::None);
    cancel.store(true, Ordering::Release);
    port.release_all();
    assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Cancelled { filled: true }));
    assert_eq!(slot_prefix(&tier, 0, 10000), rig.record(id(0, 0)));
    let staged = slot_prefix(&staging, 0, 10000);
    assert_eq!(
        staged[..PAGE as usize],
        rig.record(id(0, 0))[..PAGE as usize],
        "the first part was copied"
    );
    assert!(
        staged[10000 - 100..].iter().all(|&byte| byte == 0),
        "the last part was not"
    );
}

// ---------------------------------------------------------------- the tier

fn tier_over(rig: &Rig, slots: usize, shards: usize) -> (Arc<HostArena>, Vec<RamTierShard>) {
    RamTier::new(rig.layout(), slots, shards).expect("a tier").into_parts()
}

/// A sampler that draws the slots of a script, over and over.
struct Script(Vec<usize>, usize);

impl VictimSampler for Script {
    fn below(&mut self, bound: usize) -> usize {
        let draw = self.0[self.1 % self.0.len()] % bound;
        self.1 += 1;
        draw
    }
}

fn fill(shard: &mut RamTierShard, target: ExpertId, scan: bool) -> usize {
    let TierDirective::Fill { slot } = shard.plan(target, scan).expect("a plan") else {
        panic!("{target:?} did not get a slot");
    };
    shard.apply(TierSettle::Filled(slot)).expect("filled");
    slot
}

#[test]
fn a_record_is_a_miss_then_a_fill_then_a_hit_that_pins_its_slot_until_used() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 4, 1);
    let shard = &mut shards[0];
    let first = shard.plan_decode(id(0, 0)).expect("a plan");
    let TierDirective::Fill { slot } = first else {
        panic!("an empty tier did not fill: {first:?}");
    };
    assert!(shard.is_filling(id(0, 0)));
    assert_eq!(shard.pinned_slots(), 1);
    // Another load of the record while it fills reads the file into staging alone.
    assert_eq!(shard.plan_decode(id(0, 0)).expect("a plan"), TierDirective::Bypass);
    shard.apply(TierSettle::Filled(slot)).expect("filled");
    assert!(shard.is_ready(id(0, 0)));
    assert_eq!(
        shard.plan_decode(id(0, 0)).expect("a plan"),
        TierDirective::Hit { slot }
    );
    assert_eq!(shard.pinned_slots(), 1, "the hit holds the slot");
    shard.apply(TierSettle::Used(slot)).expect("used");
    assert_eq!(shard.pinned_slots(), 0);
    shard.check_invariants();
    let stats = shard.stats();
    assert_eq!((stats.hits[0], stats.misses[0], stats.bypasses[0]), (1, 1, 1));
}

#[test]
fn a_fill_that_failed_gives_its_slot_back_and_one_that_was_lost_never_does() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 2, 1);
    let shard = &mut shards[0];
    let TierDirective::Fill { slot: first } = shard.plan_decode(id(0, 0)).expect("a plan") else {
        panic!("no fill");
    };
    shard.apply(TierSettle::Abandoned(first)).expect("abandoned");
    assert!(!shard.is_filling(id(0, 0)) && !shard.is_ready(id(0, 0)));
    let TierDirective::Fill { slot: again } = shard.plan_decode(id(0, 1)).expect("a plan") else {
        panic!("no fill");
    };
    assert_eq!(again, first, "the abandoned slot is the next one taken");
    shard.apply(TierSettle::Lost(again)).expect("lost");
    shard.check_invariants();
    // Only one slot is left to the tier.
    let kept = fill(shard, id(0, 2), false);
    assert_ne!(kept, again, "the lost slot is never handed out");
    assert_eq!(
        shard.plan_decode(id(0, 3)).expect("a plan"),
        TierDirective::Fill { slot: kept }
    );
    shard.check_invariants();
}

#[test]
fn a_settle_that_does_not_match_its_slot_is_refused() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 2, 1);
    let shard = &mut shards[0];
    assert!(matches!(
        shard.apply(TierSettle::Used(0)),
        Err(ExpertError::BadSettle { slot: 0, .. })
    ));
    assert!(matches!(
        shard.apply(TierSettle::Filled(0)),
        Err(ExpertError::BadSettle { .. })
    ));
    assert!(matches!(
        shard.apply(TierSettle::Used(7)),
        Err(ExpertError::BadSettle { slot: 7, .. })
    ));
    let slot = fill(shard, id(0, 0), false);
    assert!(
        matches!(
            shard.apply(TierSettle::Filled(slot)),
            Err(ExpertError::BadSettle { .. })
        ),
        "twice"
    );
}

#[test]
fn a_full_tier_replaces_the_least_requested_of_the_sample_and_never_a_pinned_slot() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 3, 1);
    let shard = &mut shards[0];
    shard.set_sampler(Box::new(Script(vec![0, 1, 2], 0)));
    // Records 0, 1, 2 fill the slots; 1 and 2 are asked for again, so 0 is the least requested.
    let slots: Vec<usize> = (0..3).map(|expert| fill(shard, id(0, expert), false)).collect();
    for expert in 1..3 {
        let TierDirective::Hit { slot } = shard.plan_decode(id(0, expert)).expect("a plan") else {
            panic!("expert {expert} is not a hit");
        };
        shard.apply(TierSettle::Used(slot)).expect("used");
    }
    assert_eq!(
        shard.plan_decode(id(0, 3)).expect("a plan"),
        TierDirective::Fill { slot: slots[0] },
        "the least requested record's slot goes to the new one"
    );
    assert!(!shard.is_ready(id(0, 0)));
    assert_eq!(shard.stats().evictions[0], 1);
    // Pin the others with hits: nothing is left to take, so the next record bypasses.
    shard.apply(TierSettle::Filled(slots[0])).expect("filled");
    for expert in 1..3 {
        assert!(matches!(
            shard.plan_decode(id(0, expert)).expect("a plan"),
            TierDirective::Hit { .. }
        ));
    }
    assert!(matches!(
        shard.plan_decode(id(0, 3)).expect("a plan"),
        TierDirective::Hit { .. }
    ));
    assert_eq!(
        shard.plan_decode(id(1, 0)).expect("a plan"),
        TierDirective::Bypass,
        "every slot is pinned"
    );
    shard.check_invariants();
}

#[test]
fn a_prefill_scan_replaces_a_victim_only_if_its_record_was_asked_for_more_often() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 1, 1);
    let shard = &mut shards[0];
    shard.set_sampler(Box::new(Script(vec![0], 0)));
    fill(shard, id(0, 0), true);
    // Asked once before (when it filled): a first scan of another record does not replace it; a second ties (1 >= 1).
    assert_eq!(shard.plan_scan(id(0, 1)).expect("a plan"), TierDirective::Bypass);
    assert_eq!(shard.plan_scan(id(0, 1)).expect("a plan"), TierDirective::Bypass);
    // The third scan was asked for twice before: more than the victim's one.
    assert!(matches!(
        shard.plan_scan(id(0, 1)).expect("a plan"),
        TierDirective::Fill { .. }
    ));
    assert!(!shard.is_ready(id(0, 0)));
    shard.check_invariants();
}

#[test]
fn a_decode_load_always_enters_even_against_a_hot_victim() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 1, 1);
    let shard = &mut shards[0];
    shard.set_sampler(Box::new(Script(vec![0], 0)));
    let slot = fill(shard, id(0, 0), false);
    for _ in 0..5 {
        assert_eq!(
            shard.plan_decode(id(0, 0)).expect("a plan"),
            TierDirective::Hit { slot }
        );
        shard.apply(TierSettle::Used(slot)).expect("used");
    }
    assert!(matches!(
        shard.plan_decode(id(0, 1)).expect("a plan"),
        TierDirective::Fill { .. }
    ));
}

#[test]
fn request_counts_are_halved_so_an_old_record_does_not_hold_its_slot() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 1, 1);
    let shard = &mut shards[0];
    shard.set_sampler(Box::new(Script(vec![0], 0)));
    let slot = fill(shard, id(0, 0), false);
    for _ in 0..4 {
        shard.plan_decode(id(0, 0)).expect("a plan");
        shard.apply(TierSettle::Used(slot)).expect("used");
    }
    // 5 requests so far; AGE_PER_SLOT * 1 slot = 10 requests per halving. Other records are asked for, never admitted.
    for _ in 0..4 {
        assert_eq!(
            shard.plan_scan(id(0, 1)).expect("a plan"),
            TierDirective::Bypass,
            "the old record holds"
        );
    }
    // The tenth request halves every count (the old record's 5 becomes 2) before the victim is compared with the
    // newcomer's 4 earlier requests: the newcomer is now asked for more often.
    assert!(matches!(
        shard.plan_scan(id(0, 1)).expect("a plan"),
        TierDirective::Fill { .. }
    ));
}

#[test]
fn a_prefetch_takes_a_slot_without_a_request_or_admission_and_counts_when_it_is_used() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 2, 1);
    let shard = &mut shards[0];
    let TierDirective::Fill { slot } = shard.plan_prefetch(id(0, 0)).expect("a plan").expect("a free slot") else {
        panic!("a prefetch is a fill");
    };
    assert_eq!(
        shard.plan_prefetch(id(0, 0)).expect("a plan"),
        None,
        "it is filling already"
    );
    shard.apply(TierSettle::Filled(slot)).expect("filled");
    assert_eq!(shard.plan_prefetch(id(0, 0)).expect("a plan"), None, "it is held");
    assert_eq!(shard.stats().prefetches_used, 0);
    assert_eq!(shard.plan_scan(id(0, 0)).expect("a plan"), TierDirective::Hit { slot });
    assert_eq!(shard.stats().prefetches_used, 1);
    shard.apply(TierSettle::Used(slot)).expect("used");
    let other = shard.plan_prefetch(id(0, 1)).expect("a plan").expect("the second slot");
    assert!(matches!(other, TierDirective::Fill { .. }));
    // No free slot: a prefetch replaces a ready, unpinned record without admission, and finds nothing once every slot
    // is filling or pinned.
    assert!(matches!(
        shard.plan_prefetch(id(0, 2)).expect("a plan"),
        Some(TierDirective::Fill { slot: 0 })
    ));
    assert_eq!(shard.plan_prefetch(id(0, 3)).expect("a plan"), None);
}

#[test]
fn a_tier_with_a_slot_for_every_record_is_resident_and_never_fills_or_evicts() {
    let rig = Rig::new();
    let (memory, mut shards) = tier_over(&rig, 8, 2);
    assert_eq!(memory.slots(), 8);
    let mut slots = Vec::new();
    for shard in &mut shards {
        assert!(shard.holds_all());
        slots.extend(shard.assigned());
    }
    assert_eq!(slots.len(), 8);
    for target in (0..8).map(|key| rig.layout().id_of(key)) {
        let owner = shard_of(rig.layout().key(target).expect("a key"), 2);
        let TierDirective::Hit { slot } = shards[owner].plan_decode(target).expect("a plan") else {
            panic!("{target:?} is not resident");
        };
        shards[owner].apply(TierSettle::Used(slot)).expect("used");
    }
    for shard in &shards {
        shard.check_invariants();
        assert_eq!(shard.stats().evictions.iter().sum::<u64>(), 0);
    }
}

#[test]
fn a_shared_assignment_holds_slots_by_bank_until_a_request_replaces_them() {
    let rig = Rig::new();
    let (_memory, mut shards) = tier_over(&rig, 4, 1);
    let shard = &mut shards[0];
    shard.assign_share();
    let assigned = shard.assigned();
    // 4 slots over 4 + 3 + 1 records: 2 of bank 0, 1 of bank 1, 0 of bank 2 (each bank's share rounded down).
    let per_bank: Vec<usize> = (0..3)
        .map(|bank| assigned.iter().filter(|(id, _)| id.bank == bank).count())
        .collect();
    assert_eq!(per_bank, [2, 1, 0]);
    assert!(shard.is_ready(id(0, 0)) && shard.is_ready(id(0, 1)) && shard.is_ready(id(1, 0)));
    shard.check_invariants();
    // A scan of an unassigned record replaces an assigned one at once (-1 is below any request count), the free
    // slot first.
    let first = shard.plan_scan(id(2, 0)).expect("a plan");
    assert!(matches!(first, TierDirective::Fill { .. }));
    shard
        .apply(TierSettle::Filled(match first {
            TierDirective::Fill { slot } => slot,
            _ => unreachable!(),
        }))
        .expect("filled");
    let second = shard.plan_scan(id(0, 3)).expect("a plan");
    assert!(
        matches!(second, TierDirective::Fill { .. }),
        "the assigned records give way: {second:?}"
    );
    shard.check_invariants();
}

#[test]
fn a_tier_with_fewer_slots_than_shards_is_refused_naming_both() {
    let rig = Rig::new();
    let error = RamTier::new(rig.layout(), 1, 2).expect_err("refused");
    assert_eq!(error, ExpertError::TierTooSmall { slots: 1, shards: 2 });
}

// ---------------------------------------------------------------- the store, with a fake owner

/// What the cache owner does with the loads: plans on its shard, starts the store, applies the settle it is told.
struct Owner {
    shard: RamTierShard,
    store: ExpertStore,
    staging: Arc<HostArena>,
    port: Arc<FakePort>,
}

impl Owner {
    fn new(rig: &Rig, tier_slots: usize, split: usize) -> Owner {
        let port = FakePort::new(rig, true);
        let source =
            FileRecordSource::new(rig.layout(), Arc::clone(&port) as Arc<dyn PartPort>, split, true).expect("a source");
        let (memory, mut shards) = tier_over(rig, tier_slots, 1);
        let staging = arena(rig, 4);
        Owner {
            shard: shards.remove(0),
            store: ExpertStore::new(source, Some(memory), Arc::clone(&staging)),
            staging,
            port,
        }
    }

    /// Loads `target` into staging slot `staging_slot` as a decode step would; returns what it told.
    fn load(&mut self, target: ExpertId, staging_slot: usize, cancel: &Arc<AtomicBool>) -> (TierDirective, Arc<Probe>) {
        let directive = self.shard.plan_decode(target).expect("a plan");
        let probe = Arc::new(Probe::default());
        let started = self
            .store
            .begin(StoreLoad {
                id: target,
                directive,
                staging_slot,
                cancel: Arc::clone(cancel),
                done: probe.load_done(),
            })
            .expect("started");
        if let Started::Frame(mut frame) = started {
            frame.execute().expect("the frame runs");
        }
        self.port.release_all();
        (directive, probe)
    }

    /// The owner's record of an end: the settle is applied to the shard.
    fn settle(&mut self, probe: &Probe) -> LoadEnd {
        let end = probe.end.get().expect("the load ended").clone();
        if let Some(settle) = end.settle {
            self.shard.apply(settle).expect("a settle that matches");
        }
        end
    }
}

#[test]
fn the_store_fills_then_hits_then_bypasses_and_the_owner_applies_each_settle_in_order() {
    let rig = Rig::new();
    let mut owner = Owner::new(&rig, 1, 2);
    let cancel = no_cancel();
    let target = id(0, 1);
    let record = rig.record(target);

    // A miss fills the tier slot and delivers to staging.
    let (directive, probe) = owner.load(target, 0, &cancel);
    assert!(matches!(directive, TierDirective::Fill { .. }));
    let end = owner.settle(&probe);
    assert_eq!(end.outcome, ReadOutcome::Complete);
    assert!(matches!(end.settle, Some(TierSettle::Filled(_))));
    assert_eq!(slot_prefix(&owner.staging, 0, 10000), record);
    let reads = owner.port.starts().len();
    assert_eq!(reads, 2, "two parts");

    // The next load of it is a hit: one frame, no read of the file.
    let (directive, probe) = owner.load(target, 1, &cancel);
    assert!(matches!(directive, TierDirective::Hit { .. }));
    assert_eq!(owner.port.starts().len(), reads);
    assert!(
        owner.shard.pinned_slots() == 1,
        "the hit pins the slot until its settle"
    );
    let end = owner.settle(&probe);
    assert!(matches!(end.settle, Some(TierSettle::Used(_))));
    assert_eq!(owner.shard.pinned_slots(), 0);
    assert_eq!(slot_prefix(&owner.staging, 1, 10000), record);

    // While the record fills, a second load of it bypasses: read from the file into staging, no settle.
    let other = id(0, 2);
    let first = owner.shard.plan_decode(other).expect("a plan");
    assert!(
        matches!(first, TierDirective::Fill { .. }),
        "the single slot goes to the newcomer"
    );
    owner
        .shard
        .apply(
            first
                .settle(&ReadOutcome::Cancelled { filled: false })
                .expect("a settle"),
        )
        .expect("abandoned");
    let (_, probe) = owner.load(other, 2, &cancel);
    assert!(matches!(owner.settle(&probe).settle, Some(TierSettle::Filled(_))));
    owner.shard.check_invariants();
    assert!(!owner.staging.is_leased(0) && !owner.staging.is_leased(1) && !owner.staging.is_leased(2));
}

#[test]
fn a_failed_fill_settles_abandoned_and_a_bypass_settles_nothing() {
    let rig = Rig::new();
    let mut owner = Owner::new(&rig, 1, 1);
    let cancel = no_cancel();
    let directive = owner.shard.plan_decode(id(0, 0)).expect("a plan");
    let probe = Arc::new(Probe::default());
    owner
        .store
        .begin(StoreLoad {
            id: id(0, 0),
            directive,
            staging_slot: 0,
            cancel: Arc::clone(&cancel),
            done: probe.load_done(),
        })
        .expect("started");
    owner.port.release(0, Fault::Os(5));
    let end = owner.settle(&probe);
    let TierDirective::Fill { slot } = directive else {
        panic!("not a fill")
    };
    assert_eq!(end.settle, Some(TierSettle::Abandoned(slot)));
    assert!(matches!(end.outcome, ReadOutcome::Failed(_)));
    assert_eq!(owner.shard.pinned_slots(), 0);
    owner.shard.check_invariants();

    // A bypass: pin the only slot with a hit, then ask for another record.
    let (_, probe) = owner.load(id(0, 0), 0, &cancel);
    owner.settle(&probe);
    let held = owner.shard.plan_decode(id(0, 0)).expect("a plan");
    assert!(matches!(held, TierDirective::Hit { .. }));
    let (directive, probe) = owner.load(id(0, 3), 1, &cancel);
    assert_eq!(directive, TierDirective::Bypass);
    assert_eq!(owner.settle(&probe).settle, None);
    assert_eq!(slot_prefix(&owner.staging, 1, 10000), rig.record(id(0, 3)));
}

#[test]
fn a_fill_whose_memory_the_ring_leaked_settles_lost_and_the_slot_leaves_the_tier() {
    let rig = Rig::new();
    let mut owner = Owner::new(&rig, 2, 1);
    let directive = owner.shard.plan_decode(id(0, 0)).expect("a plan");
    let probe = Arc::new(Probe::default());
    owner
        .store
        .begin(StoreLoad {
            id: id(0, 0),
            directive,
            staging_slot: 0,
            cancel: no_cancel(),
            done: probe.load_done(),
        })
        .expect("started");
    owner.port.release(0, Fault::Leak);
    owner.port.release_all();
    let end = owner.settle(&probe);
    assert_eq!(end.outcome, ReadOutcome::Lost);
    assert!(matches!(end.settle, Some(TierSettle::Lost(_))));
    owner.shard.check_invariants();
}

#[test]
fn a_hit_that_was_cancelled_skips_the_copy_and_still_settles_used() {
    let rig = Rig::new();
    let mut owner = Owner::new(&rig, 1, 1);
    let cancel = no_cancel();
    let (_, probe) = owner.load(id(0, 0), 0, &cancel);
    owner.settle(&probe);
    cancel.store(true, Ordering::Release);
    let (directive, probe) = owner.load(id(0, 0), 1, &cancel);
    assert!(matches!(directive, TierDirective::Hit { .. }));
    let end = owner.settle(&probe);
    assert_eq!(end.outcome, ReadOutcome::Cancelled { filled: false });
    assert!(matches!(end.settle, Some(TierSettle::Used(_))));
    assert!(slot_prefix(&owner.staging, 1, 10000).iter().all(|&byte| byte == 0));
    assert_eq!(owner.shard.pinned_slots(), 0);
}

#[test]
fn a_load_refused_at_the_start_leaves_its_directive_to_be_settled_unstarted() {
    let rig = Rig::new();
    let mut owner = Owner::new(&rig, 1, 1);
    let directive = owner.shard.plan_decode(id(0, 0)).expect("a plan");
    let probe = Arc::new(Probe::default());
    let _held = owner.staging.read(3).expect("a staging slot in use");
    let refused = owner.store.begin(StoreLoad {
        id: id(0, 0),
        directive,
        staging_slot: 3,
        cancel: no_cancel(),
        done: probe.load_done(),
    });
    assert!(matches!(refused, Err(ExpertError::SlotBusy { slot: 3, .. })));
    assert_eq!(probe.calls(), 0);
    let settle = directive.settle_unstarted().expect("a fill is settled");
    owner.shard.apply(settle).expect("abandoned");
    assert_eq!(owner.shard.pinned_slots(), 0, "the refused fill's slot is free again");
    owner.shard.check_invariants();
}

// ---------------------------------------------------------------- the preload

#[test]
fn the_preload_reads_every_assigned_record_into_its_slot_with_several_chains() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = Arc::new(source(&rig, &port, 2, true));
    let (memory, shards) = tier_over(&rig, 8, 1);
    let items = shards[0].assigned();
    assert_eq!(items.len(), 8);
    let told = Arc::new(OnceLock::new());
    let sink = Arc::clone(&told);
    preload::preload(
        Arc::clone(&source),
        Arc::clone(&memory),
        items.clone(),
        3,
        Box::new(move |result| {
            let _ = sink.set(result);
        }),
    )
    .expect("started");
    // Three chains are out at once, each with its record's parts.
    let first_wave: BTreeSet<u64> = port.starts().iter().map(|start| start.offset).collect();
    assert!(first_wave.len() > 3, "three records' parts are out together");
    while told.get().is_none() {
        assert!(
            port.held() > 0,
            "the preload ended without telling, or stalled with no read out"
        );
        port.release(0, Fault::None);
    }
    let total: u64 = (0..8)
        .map(|key| rig.layout().record(rig.layout().id_of(key)).expect("a record").bytes)
        .sum();
    assert_eq!(told.get(), Some(&Ok(total)));
    for (target, slot) in items {
        assert_eq!(slot_prefix(&memory, slot, rig.record(target).len()), rig.record(target));
    }
    shards[0].check_invariants();
}

#[test]
fn a_preload_that_meets_a_damaged_record_ends_with_its_name_and_reads_no_more() {
    let mut rig = Rig::new();
    let record = rig.layout().record(id(0, 0)).expect("a record");
    rig.bytes[record.offset as usize + 1] ^= 1;
    write(&rig.path, &rig.bytes).expect("damaged");
    let port = FakePort::new(&rig, true);
    let source = Arc::new(source(&rig, &port, 1, true));
    let (memory, shards) = tier_over(&rig, 8, 1);
    let items = shards[0].assigned();
    let told = Arc::new(OnceLock::new());
    let sink = Arc::clone(&told);
    preload::preload(
        source,
        memory,
        items,
        1,
        Box::new(move |result| {
            let _ = sink.set(result);
        }),
    )
    .expect("started");
    while told.get().is_none() {
        assert!(port.held() > 0, "the preload stalled with no read out");
        port.release(0, Fault::None);
    }
    assert!(matches!(told.get(), Some(Err(ExpertError::Damaged { bank, expert: 0, .. })) if bank == "bank0"));
    assert!(port.starts().len() < 8, "the chain went on after the failure");
}

// ---------------------------------------------------------------- the real ring, driven by hand

/// Runs the ring's frames until `ended` is set: the lattice's work (reap, run what is ready) done by the test.
fn drive(io: &Arc<AsyncIo>, ended: &dyn Fn() -> bool, what: &str) {
    let limit = Instant::now() + HANG;
    while !ended() {
        assert!(
            Instant::now() < limit,
            "{what} never ended: the ring held {} operations",
            io.pending()
        );
        io.reap();
        io.pull(
            &mut |mut frame| frame.execute().expect("the frame runs"),
            &mut |_| false,
            u64::MAX,
        );
    }
}

fn read_through_the_ring(options: AsyncIoOptions) -> bool {
    let rig = Rig::new();
    let io = AsyncIo::open_with_options(options).expect("the ring opens");
    let file = Arc::new(ArtifactFile::open(&rig.path).expect("the artifact opens"));
    let direct = file.has_direct();
    let port: Arc<dyn PartPort> = Arc::new(FilePartPort::new(
        Arc::clone(&io),
        file,
        DiskReadPiece::from_setting(Auto::Auto).expect("a piece"),
    ));
    println!("has_direct = {direct}");
    let source = FileRecordSource::new(rig.layout(), port, 3, true).expect("a source");
    let memory = arena(&rig, 1);
    for target in (0..rig.layout().keys()).map(|key| rig.layout().id_of(key)) {
        let probe = Arc::new(Probe::default());
        if let Started::Frame(mut frame) =
            read(&source, &memory, 0, target, None, &no_cancel(), &probe).expect("started")
        {
            frame.execute().expect("the frame runs");
        }
        drive(&io, &|| probe.calls() > 0, "the read of a record");
        assert_eq!(probe.outcome.get(), Some(&ReadOutcome::Complete), "{target:?}");
        assert_eq!(
            slot_prefix(&memory, 0, rig.record(target).len()),
            rig.record(target),
            "{target:?}"
        );
    }
    direct
}

#[test]
fn the_ring_reads_every_record_of_the_artifact_in_parts() {
    read_through_the_ring(AsyncIoOptions::default());
}

#[test]
fn where_the_ring_is_positional_the_same_reads_end_the_same_way() {
    read_through_the_ring(AsyncIoOptions {
        positional_io: true,
        ..AsyncIoOptions::default()
    });
}

// ---------------------------------------------------------------- refusals

#[test]
fn an_arena_refuses_a_slot_that_is_not_whole_pages_and_one_that_cannot_exist_naming_the_values() {
    assert_eq!(
        HostArena::new(2, 100).expect_err("not paged"),
        ExpertError::SlotNotPaged {
            bytes: 100,
            block: HOST_ALIGNMENT
        }
    );
    assert!(matches!(
        HostArena::new(2, 0).expect_err("empty slots"),
        ExpertError::SlotNotPaged { bytes: 0, .. }
    ));
    assert_eq!(
        HostArena::new(0, 4096).expect_err("no slot"),
        ExpertError::ArenaSize { slots: 0, bytes: 4096 }
    );
    assert_eq!(
        HostArena::new(usize::MAX, 4096).expect_err("overflow"),
        ExpertError::ArenaSize {
            slots: usize::MAX,
            bytes: 4096
        }
    );
}

#[test]
fn a_preload_with_no_readers_is_refused_before_anything_is_read() {
    let rig = Rig::new();
    let port = FakePort::new(&rig, true);
    let source = Arc::new(source(&rig, &port, 1, true));
    let (memory, shards) = tier_over(&rig, 8, 1);
    let error = preload::preload(
        source,
        memory,
        shards[0].assigned(),
        0,
        Box::new(|_| panic!("a refused preload tells nothing")),
    )
    .expect_err("no readers");
    assert_eq!(error, ExpertError::NoReaders { readers: 0 });
    assert!(port.starts().is_empty());
}

#[test]
fn a_shard_whose_slots_hold_all_its_records_is_assigned_them_whatever_the_whole_tier_is() {
    let rig = Rig::new();
    let layout = rig.layout();
    assert_eq!(layout.keys(), 8);
    // 7 slots over 2 shards: the tier cannot hold every record, yet a shard with a slot for each of its own can.
    let (_, mut shards) = tier_over(&rig, 7, 2);
    let mut held = 0;
    for (index, shard) in shards.iter_mut().enumerate() {
        let owned: Vec<usize> = (0..layout.keys()).filter(|&key| shard_of(key, 2) == index).collect();
        if !shard.holds_all() {
            continue;
        }
        held += 1;
        assert_eq!(
            shard.assigned().len(),
            owned.len(),
            "every record of the shard has its slot"
        );
        for key in owned {
            let directive = shard.plan(layout.id_of(key), false).expect("a plan");
            assert!(
                matches!(directive, TierDirective::Hit { .. }),
                "key {key}: {directive:?}"
            );
        }
        shard.check_invariants();
    }
    assert!(held > 0, "the case has a shard that holds all its records");
}
