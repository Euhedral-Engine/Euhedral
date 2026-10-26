//! Exact numerics is a switch of the whole process, so the test that sets it has a binary of its own (as
//! `formats_oracle_gpu` has): no quantum is captured while it is on.
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test capture_exact_gpu -- --ignored`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes a lane to read results"
)]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::LaneId;
use euhedral_inference_cuda::PinnedPool;
use euhedral_inference_cuda::RetireList;
use euhedral_inference_cuda::{CudaLane, CudaLibrary, Submission, SubmitMode};
use euhedral_inference_engine::model::{LayerKind, ModelConfig};
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::stages::{InputRecordRef, LaneTable, Quantum, QuantumKind};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::{SequenceDevice, SequenceState};
use euhedral_inference_engine::workspace::{Workspace, WorkspaceRows};
use euhedral_inference_settings::KvFormat;

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_is_never_captured_under_exact_numerics() {
    let Some(path) = test_support::require_property("euhedral.cuda.library") else {
        return;
    };
    let library = CudaLibrary::load(&PathBuf::from(path)).unwrap();
    let mut config = common::config();
    config.layer_kinds = vec![LayerKind::GatedDeltaNet, LayerKind::FullAttention];
    config.num_hidden_layers = 2;
    let config: Arc<ModelConfig> = Arc::new(config);
    let lane = CudaLane::new(&library, LaneId(0), 16).unwrap();
    let table = LaneTable::new(vec![Arc::clone(&lane)]).unwrap();
    let retire = Arc::new(RetireList::new());
    let pinned = PinnedPool::new(&library, 16 << 20);
    let workspace = Arc::new(
        Workspace::allocate(
            &library,
            &config,
            WorkspaceRows {
                rows: 8,
                logits_rows: 8,
            },
            &[],
        )
        .unwrap(),
    );
    let submission = Submission::begin(&lane, SubmitMode::default()).unwrap();
    let gdn = GdnStates::allocate(&library, &config, 8, &submission).unwrap();
    submission.finish_signalled().unwrap();
    lane.synchronize().unwrap();
    let kv = KvCache::new(&library, &retire, 1, config.kv_width() as u64, KvFormat::Nvfp4, 1024, 0).unwrap();
    let scratch = AttentionScratch::allocate(&library, &config, 8, true, 16).unwrap();
    let mut sequence = SequenceState::from_parts(
        kv,
        None,
        Arc::new(SequenceDevice { gdn, scratch }),
        SelectionPool::new(&library, 2, 8).unwrap(),
    );
    // The decode quantum at `start`, after the one reservation of its row.
    let mut retained = KvRetained::default();
    let mut quantum = |sequence: &mut SequenceState, start: u64| {
        let submission = Submission::begin(&lane, SubmitMode::default()).unwrap();
        sequence
            .kv
            .reserve(&submission, &pinned, start, 1, &mut retained)
            .unwrap();
        sequence.kv.appended(1).unwrap();
        drop(submission);
        lane.synchronize().unwrap();
        sequence.kv.commit(1).unwrap();
        Quantum::new(
            QuantumKind::Decode,
            1,
            InputRecordRef::new(start),
            Arc::clone(&workspace),
            sequence.refs(KvSource::Base).unwrap(),
            SubmitMode::default(),
            Arc::clone(&config),
            table.clone(),
        )
        .unwrap()
        .with_selection_lease(sequence.selections.acquire().unwrap())
    };
    assert!(
        quantum(&mut sequence, 0).capture_key().is_none(),
        "the first reservation allocates"
    );
    assert!(
        quantum(&mut sequence, 1).capture_key().is_some(),
        "decode inside a reserved page is keyed"
    );
    library.exact_numerics_for_tests(true);
    let exact = quantum(&mut sequence, 2).capture_key();
    library.exact_numerics_for_tests(false);
    assert!(
        exact.is_none(),
        "exact numerics runs every launch on its scalar reference: nothing is captured"
    );
    assert!(
        quantum(&mut sequence, 3).capture_key().is_some(),
        "and the key is back with the switch off"
    );
}
