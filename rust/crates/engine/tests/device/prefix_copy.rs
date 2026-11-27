//! The prefix cache's copy jobs on a GPU: a checkpoint captured from a sequence's buffers into the pinned arena and
//! restored after the buffers were overwritten must bring back every byte. `#[ignore]`d; run locally with the CUDA
//! library named by `euhedral.cuda.library` (`EUHEDRAL_TEST_CUDA_LIBRARY`), skipping without it. Every wait is bounded.
//!
//! The model is the first four layers of the RCO model (three GDN layers and one full-attention layer); no weights are
//! loaded, and the sequences do not speculate. Two sequences capture and restore at the same time, on different lanes, each its own bytes.

#![allow(clippy::print_stderr, reason = "a test reports what it skipped")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the test thread waits for the cache's answers against a deadline and runs the cache's owner on a thread"
)]

use crate::common;
use crate::typed;

use std::fmt::Display;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_gpu::cuda::CudaQueue;
use euhedral_graph::{Completions, DEFAULT_RETRY_BACKSTOP, GraphRuntime, Lake, LaneId, RetryCounts, UnitGuard};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, LaneOps, PinnedBuffer, PinnedPool};
use euhedral_inference_engine::generation::Spawn;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{LayerKind, ModelConfig};
use euhedral_inference_engine::prefix::{
    CacheSpec, CheckpointLayout, Copier, PrefixCacheHandle, SequenceAddresses, Submit, open_prefix_cache,
};
use euhedral_inference_engine::stages::SequenceRefs;
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::released::ReleasedBuffers;
use euhedral_inference_engine::state::{SequenceDevice, SequenceId, SequenceState};
use euhedral_inference_settings::{ByteSize, KvFormat, ModelSettings, PrefixCacheSettings, Settings};
use typed::Typed;

/// How long the cache may take to answer: a few copies of a few megabytes.
const DEADLINE: Duration = Duration::from_secs(30);

/// Layers of the reduced model: the artifact's layers 0 to 3.
const LAYERS: usize = 4;

/// Tokens of the checkpoint: two cache pages.
const TOKENS: u32 = 512;

/// Sequences that copy at once, each on its own lane and chain.
const SEQUENCES: usize = 2;

fn pattern(seed: u8, bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

struct World {
    library: Arc<CudaLibrary>,
    typed: Typed,
    runtime: GraphRuntime,
    config: Arc<ModelConfig>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
}

impl World {
    /// One lane per sequence for its copies (lanes 0 and 1), and the owner's after them.
    fn open() -> Option<World> {
        let path = test_support::require_property("euhedral.cuda.library")?;
        let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
        let mut config = common::config();
        config.layer_kinds.truncate(LAYERS);
        config.num_hidden_layers = LAYERS;
        assert_eq!(config.layer_kinds[3], LayerKind::FullAttention);
        let typed = Typed::plain(&library, 1 + SEQUENCES, 1024);
        let pool = Arc::clone(&typed.queues);
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
        let lake = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let completions = Completions::new(&pool);
        lattice
            .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
            .unwrap();
        let runtime = GraphRuntime {
            stages: Arc::clone(&lake) as Arc<dyn FrameSink>,
            completions,
            lanes: pool,
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let unit = lake.admit().unwrap();
        Some(World {
            library,
            typed,
            runtime,
            config: Arc::new(config),
            lattice,
            lake,
            unit: Some(unit),
        })
    }

    fn owner(&self) -> &Arc<CudaQueue> {
        self.typed.lane(SEQUENCES as u16)
    }

    fn on_owner<E: Display>(&self, work: impl FnOnce(&mut dyn LaneOps) -> Result<(), E>) {
        self.typed.table.submit_ops(self.owner(), false, work).unwrap();
        self.typed.sync();
    }

    fn close(mut self) {
        drop(self.unit.take());
        self.lake.complete_gracefully();
        self.lattice.close();
        self.typed.sync();
    }

    /// Overwrites `bytes` at `ptr` with `data`.
    fn write(&self, ptr: DevicePtr, data: &[u8]) {
        let mut pinned = PinnedBuffer::alloc(&self.library, data.len()).unwrap();
        pinned.as_mut_slice()[..data.len()].copy_from_slice(data);
        self.on_owner(|lane| {
            // SAFETY: `pinned` outlives the copy: the device is waited for before it drops.
            unsafe { lane.copy_upload_to_device(ptr, &pinned, 0, data.len()) }
        });
    }
}

/// Where the state of a sequence the checkpoint holds is, as `(device address, bytes)`.
fn regions(refs: &SequenceRefs, config: &ModelConfig) -> Vec<(DevicePtr, usize)> {
    let device = refs.device();
    let mut regions = Vec::new();
    for (layer, kind) in config.layer_kinds.iter().enumerate() {
        if *kind == LayerKind::GatedDeltaNet {
            let state = device.gdn.layer(layer).unwrap();
            regions.push((state.convolution().ptr(), state.convolution().bytes()));
            regions.push((state.recurrent().ptr(), state.recurrent().bytes()));
        }
    }
    let view = refs.kv(KvSource::Base).unwrap();
    for page in 0..(TOKENS / 256) as usize {
        regions.push((view.page(0, page).unwrap(), 2 * view.plane_page_bytes() as usize));
    }
    regions
}

fn expect<T>(receiver: &Receiver<T>, what: &str) -> T {
    receiver
        .recv_timeout(DEADLINE)
        .unwrap_or_else(|_| panic!("the cache did not answer the {what} within {DEADLINE:?}"))
}

/// One sequence of the test: its state, the addresses a copy resolves, and where its checkpointed bytes are.
struct Sequence {
    addresses: Arc<SequenceAddresses>,
    regions: Vec<(DevicePtr, usize)>,
    original: Vec<Vec<u8>>,
    copier: Arc<Copier>,
    tokens: Arc<[u32]>,
    // Held so that the sequence's buffers outlive the copies.
    _state: (SequenceState, KvRetained, Arc<PinnedPool>),
}

impl Sequence {
    /// The sequence `index`: its state, filled with bytes only it could have written, copied on lane `index`.
    fn new(world: &World, index: usize, format: KvFormat) -> Sequence {
        let mut gdn = None;
        world.on_owner(|lane| {
            gdn = Some(GdnStates::allocate(&world.library, &world.config, 8, lane)?);
            Ok::<(), CudaError>(())
        });
        let released = Arc::new(ReleasedBuffers::new());
        let pinned = PinnedPool::new(&world.library, 16 << 20);
        let mut sequence = SequenceState::from_parts(
            KvCache::new(
                &world.library,
                &released,
                1,
                world.config.kv_width() as u64,
                format,
                u64::from(TOKENS),
                0,
            )
            .unwrap(),
            None,
            Arc::new(SequenceDevice {
                gdn: gdn.unwrap(),
                scratch: AttentionScratch::allocate(&world.library, &world.config, 8, false, 0).unwrap(),
            }),
            SelectionPool::new(&world.library, 4, 8).unwrap(),
        );
        let mut retained = KvRetained::default();
        world.on_owner(|lane| sequence.kv.reserve(lane, &pinned, 0, u64::from(TOKENS), &mut retained));
        sequence.kv.appended(u64::from(TOKENS)).unwrap();
        let addresses =
            Arc::new(SequenceAddresses::new(sequence.refs(KvSource::Base).unwrap(), &world.config).unwrap());
        let regions = regions(addresses.refs(), &world.config);
        let original: Vec<Vec<u8>> = regions
            .iter()
            .enumerate()
            .map(|(region, (_, bytes))| pattern((region + 50 * index) as u8 + 1, *bytes))
            .collect();
        for ((ptr, _), data) in regions.iter().zip(&original) {
            world.write(*ptr, data);
        }
        let copier = Arc::new(
            Copier::new(
                &world.runtime,
                world.typed.table.clone(),
                LaneId(index as u16),
                SequenceId(index),
                Box::new(|_| ()),
            )
            .expect("the copier builds"),
        );
        // A prompt of its own: the two checkpoints are different nodes.
        let tokens: Arc<[u32]> = (0..=TOKENS).map(|token| token + 1000 * index as u32).collect();
        Sequence {
            addresses,
            regions,
            original,
            copier,
            tokens,
            _state: (sequence, retained, pinned),
        }
    }

    fn submit(&self) -> Submit {
        let copier = Arc::clone(&self.copier);
        Box::new(move |job| {
            copier
                .start(job, Arc::new(AtomicBool::new(false)), &[])
                .expect("the copy job starts");
        })
    }

    /// Overwrites the state with bytes that are not the checkpoint's.
    fn clobber(&self, world: &World, index: usize) {
        for (region, (ptr, bytes)) in self.regions.iter().enumerate() {
            world.write(*ptr, &pattern(0xA5 ^ (region + index) as u8, *bytes));
            assert_ne!(
                world.typed.download(*ptr, *bytes),
                self.original[region],
                "region {region} of sequence {index} was not overwritten"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU"]
fn two_sequences_capture_and_restore_their_own_bytes_at_once() {
    let _light = common::light();
    let Some(world) = World::open() else {
        eprintln!("skipped: no CUDA library");
        return;
    };
    let format = KvFormat::Nvfp4;
    let layout = CheckpointLayout::new(&world.config, format).unwrap();
    let slot = layout.slot_bytes();
    let sequences: Vec<Sequence> = (0..SEQUENCES)
        .map(|index| Sequence::new(&world, index, format))
        .collect();

    // The cache, with copy pieces smaller than a slot so that a copy is cut into groups.
    let granted = (SEQUENCES as u64 * (layout.gdn_slots() as u64 + 2) + 2) * slot;
    let spawn: Spawn = Box::new(|work| {
        thread::spawn(work);
        true
    });
    let cache: PrefixCacheHandle = open_prefix_cache(
        CacheSpec {
            library: &world.library,
            config: &world.config,
            drafter: None,
            format,
            settings: PrefixCacheSettings {
                size: ByteSize(granted),
                checkpoint_tokens: TOKENS,
                copy_piece_bytes: ByteSize((slot / 3).max(1)),
            },
            granted,
            chunk: TOKENS,
            device_tokens: TOKENS,
        },
        spawn,
    )
    .expect("the cache opens");

    // Both captures are asked before either answer is waited for: the two copies are in flight together.
    let answers: Vec<_> = sequences
        .iter()
        .enumerate()
        .map(|(index, sequence)| {
            let (told, answer) = channel();
            cache.capture(
                SequenceId(index),
                sequence.addresses.clone(),
                Arc::clone(&sequence.tokens),
                cache.root(),
                TOKENS,
                None,
                sequence.submit(),
                move |node| told.send(node).unwrap(),
            );
            answer
        })
        .collect();
    for (index, answer) in answers.iter().enumerate() {
        let node = expect(answer, "capture");
        assert_ne!(node, cache.root(), "the capture of sequence {index} stored nothing");
    }

    for (index, sequence) in sequences.iter().enumerate() {
        sequence.clobber(&world, index);
    }
    let answers: Vec<_> = sequences
        .iter()
        .enumerate()
        .map(|(index, sequence)| {
            let (told, hit) = channel();
            cache.lookup(Arc::clone(&sequence.tokens), None, move |hit| told.send(hit).unwrap());
            let hit = expect(&hit, "lookup").expect("the lookup finds the checkpoint");
            assert_eq!(hit.position(), TOKENS);
            let (told, answer) = channel();
            cache.restore(
                hit,
                SequenceId(index),
                sequence.addresses.clone(),
                None,
                Arc::new(AtomicBool::new(false)),
                sequence.submit(),
                move |restored| told.send(restored).unwrap(),
            );
            answer
        })
        .collect();
    for answer in &answers {
        assert_eq!(expect(answer, "restore"), Ok(TOKENS));
    }
    world.typed.sync();
    for (index, sequence) in sequences.iter().enumerate() {
        for (region, (ptr, bytes)) in sequence.regions.iter().enumerate() {
            assert!(
                world.typed.download(*ptr, *bytes) == sequence.original[region],
                "region {region} of {} of sequence {index} differs after the restore",
                sequence.regions.len()
            );
        }
    }
    let stats = cache.stats();
    assert_eq!((stats.captures, stats.hits, stats.restores), (2, 2, 2));

    assert!(
        cache.close(Duration::from_secs(30)),
        "the copies ended and the cache closed"
    );
    drop(sequences);
    world.close();
}
