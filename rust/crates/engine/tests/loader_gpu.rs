//! The startup on a GPU, with the RCO artifact. The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test loader_gpu -- --ignored`.
//!
//! The artifact comes from the `euhedral.qwen.rco-artifact` property and the library from `euhedral.cuda.library`
//! (`EUHEDRAL_TEST_QWEN_RCO_ARTIFACT` and `EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or `gradle.properties`); a
//! test skips with a message when either is absent. The corrupt copies of the artifact are sparse files under cargo's
//! `CARGO_TARGET_TMPDIR` (inside the target directory, on the work drive), deleted afterwards.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the ready time and the throughput it measured"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files, copies device memory and waits on the latch"
)]

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use euhedral_graph::{LaneId, Latch};
use euhedral_inference_cuda::{CudaLane, CudaLibrary, DevicePtr, Submission, SubmitMode, ops};
use euhedral_inference_edrl::WeightFormat;
use euhedral_inference_engine::error::{EngineError, PlanError};
use euhedral_inference_engine::model::{LayerIndex, NormKind, Role};
use euhedral_inference_engine::residency::loader::pieces;
use euhedral_inference_engine::residency::planner::{DEFAULT_RUNTIME_RESERVE_BYTES, nvfp4};
use euhedral_inference_engine::workspace::BufferSlot;
use euhedral_inference_engine::{EngineRuntime, RuntimeOwner};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};
use sha2::{Digest, Sha256};

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.rco-artifact names no file");
        return None;
    }
    Some(Machine { artifact, library })
}

fn settings(artifact: &Path, max_tokens: u32) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = max_tokens;
    settings
}

/// Starts the engine and waits for the latch, as `main` does.
fn start(settings: &Settings, library: &Path) -> (Result<Arc<EngineRuntime>, EngineError>, RuntimeOwner) {
    let (ready, waiting) = Latch::new();
    let guard = EngineRuntime::start(settings, library, ready);
    (waiting.wait(), guard)
}

fn read_file(path: &Path, offset: u64, length: usize) -> Vec<u8> {
    let mut file = File::open(path).unwrap();
    file.seek(SeekFrom::Start(offset)).unwrap();
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

/// `bytes` bytes at `ptr`, copied to the host on a lane of the test's own.
fn download(library: &Arc<CudaLibrary>, lane: &CudaLane, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
    let submission = Submission::begin(lane, SubmitMode::default()).unwrap();
    let mut host = vec![0; bytes];
    ops::copy_device_to_host(&submission, &mut host, ptr).unwrap();
    drop(submission);
    let _ = library;
    host
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn startup_loads_rco_and_reports_ready() {
    let Some(machine) = machine() else { return };
    let settings = settings(&machine.artifact, 8192);
    let (runtime, owner) = start(&settings, &machine.library);
    let runtime = runtime.expect("the engine starts");
    let (plan, report) = (runtime.plan(), runtime.report());
    println!(
        "ready in {:.2} s: {:.2} GiB of {} objects uploaded in {:.2} s ({:.2} GB/s) as {} pieces on {} chains of up to {} MiB; \
         {:.2} GiB allocated by the engine, {:.2} GiB in use on the device",
        report.ready.as_secs_f64(),
        report.uploaded_bytes as f64 / f64::from(1 << 30),
        report.objects,
        report.load.as_secs_f64(),
        report.uploaded_bytes as f64 / report.load.as_secs_f64() / 1e9,
        report.pieces,
        report.chains,
        report.piece_bytes >> 20,
        report.allocated_bytes as f64 / f64::from(1 << 30),
        report.device_bytes_in_use as f64 / f64::from(1 << 30),
    );

    // The modules that need only the engine's own buffers were launched once before ready.
    println!(
        "warmed {:?}, not warmed {:?}",
        report.warmed.modules, report.warmed.failed
    );
    assert!(report.warmed.failed.is_empty(), "{:?}", report.warmed.failed);
    for module in ["norm", "elementwise", "sampling", "embedding", "nvfp4"] {
        assert!(report.warmed.modules.contains(&module), "{module} was not warmed");
    }

    // Every object of the plan was uploaded: the pieces account for every byte the plan copies.
    let expected: u64 = pieces(plan, report.piece_bytes)
        .iter()
        .map(|piece| piece.len as u64)
        .sum();
    assert_eq!(report.uploaded_bytes, expected);
    assert!(
        report.uploaded_bytes + (1 << 20) > plan.weights_bytes,
        "{} < {}",
        report.uploaded_bytes,
        plan.weights_bytes
    );
    assert_eq!(
        report.objects, 1746,
        "1745 objects, the packed projection as its two halves"
    );
    // 785 objects and the split make 786 tensors by layer and role; the trellis companions ride with their objects.
    assert_eq!(runtime.weights().len(), 786);
    assert_eq!(runtime.weights().device_bytes(), plan.arena_bytes);

    // The device bytes in use are reported, and the working memory stays within the reserve the plan kept for it.
    assert!(report.device_bytes_in_use >= report.allocated_bytes);
    assert!(report.allocated_bytes >= plan.arena_bytes);
    let working = report.allocated_bytes - plan.arena_bytes;
    assert!(
        working < DEFAULT_RUNTIME_RESERVE_BYTES,
        "the workspace and the state take {working}, the reserve is {DEFAULT_RUNTIME_RESERVE_BYTES}"
    );

    // The staging ring is acquired before the upload and never grows: two loans per chain, and the pool allocated nothing
    // while the chains ran. Beside the ring the pool holds only the cache's table staging.
    let ring = 2 * report.chains as u64 * report.piece_bytes;
    assert_eq!(report.pinned_growth_during_load, 0);
    assert!(
        report.pinned_staging_bytes >= ring,
        "{} < {ring}",
        report.pinned_staging_bytes
    );
    assert!(
        report.pinned_staging_bytes <= ring + (1 << 20),
        "{} > {ring} + 1 MiB",
        report.pinned_staging_bytes
    );
    assert!(report.chains <= runtime.library().max_lanes() / 2);

    // The model is addressable by layer and role.
    let weights = runtime.weights();
    let down = weights.get(LayerIndex(10), Role::Down).unwrap();
    assert_eq!(down.format, WeightFormat::Trellis);
    let parts = down.trellis.as_ref().expect("a trellis object carries its companions");
    assert_eq!(parts.suh_rows, parts.segment_ends.len() as u64);
    assert_eq!(parts.segment_ends.last().copied(), Some(5120));
    assert!(
        weights.get(LayerIndex(3), Role::QueryKey).is_some(),
        "layer 3 is full attention"
    );
    assert!(weights.get(LayerIndex(3), Role::Qk).is_none());
    assert!(
        weights.get(LayerIndex(0), Role::Qk).is_some(),
        "layer 0 is Gated DeltaNet"
    );
    assert!(
        weights.get(LayerIndex(64), Role::GateValue).is_some(),
        "the MTP layer follows layer 63"
    );
    assert!(weights.get(LayerIndex::GLOBAL, Role::Norm(NormKind::Final)).is_some());
    assert_eq!(
        weights.get(LayerIndex::GLOBAL, Role::DraftHead).unwrap().shape,
        [131_072, 5120]
    );

    // The workspace and the state are there, sized for the plan.
    let workspace = runtime.workspace();
    assert_eq!(workspace.rows().rows, 512);
    assert_eq!(workspace.view(512).unwrap().rows(), 512);
    assert!(workspace.view(513).is_err());
    assert!(workspace.buffer(BufferSlot::Logits).bytes() > 0);
    let sequence = runtime.sequence();
    assert_eq!(sequence.kv.layers(), 16);
    assert_eq!(sequence.kv.planned_rows(), 8192);
    assert_eq!(sequence.mtp_kv.as_ref().map(|kv| kv.layers()), Some(1));
    assert!(sequence.device.gdn.layer(0).unwrap().speculative().is_some());
    assert!(sequence.device.scratch.draft().is_some());
    owner.shutdown(runtime).unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn uploaded_bytes_match_file_sha() {
    let Some(machine) = machine() else { return };
    let settings = settings(&machine.artifact, 4096);
    let (runtime, owner) = start(&settings, &machine.library);
    let runtime = runtime.expect("the engine starts");
    let library = Arc::clone(runtime.library());
    let lane = CudaLane::new(&library, LaneId(48), 4).unwrap();
    let plan = runtime.plan();
    let base = runtime.weights().get(LayerIndex(10), Role::Down).unwrap().ptr.get()
        - plan.placement("text/layers/10/mlp/down").unwrap().offset;

    // Three whole objects of three formats: a trellis projection (and its input scale vectors), an NVFP4 head and a
    // BF16 convolution kernel. The device bytes are the file's.
    for name in [
        "text/layers/10/mlp/down",
        "text/layers/10/mlp/down.suh",
        "text/draft_head",
        "text/layers/0/gdn/convolution",
    ] {
        let placement = plan.placement(name).unwrap();
        let on_device = download(
            &library,
            &lane,
            DevicePtr::NULL.offset(base + placement.offset),
            placement.bytes as usize,
        );
        let in_file = read_file(&machine.artifact, placement.file_offset, placement.bytes as usize);
        assert_eq!(sha(&on_device), sha(&in_file), "{name}");
    }

    // The packed MTP projection became two halves: codes and scales cut by rows, the global scale shared.
    let source = settings.model.artifact.clone();
    let pack = "mtp/layer/attention/query_key_gate_value";
    let (rows, k) = (14_336u64, 5120u64);
    let half = rows / 2;
    let layout = euhedral_inference_edrl::WeightLayout::RowSplitK128V1;
    let pack_offset = {
        let first = plan.placement("mtp/layer/attention/query_key").unwrap();
        first.file_offset
    };
    let code = nvfp4::row_code_bytes(k);
    let row_scale = nvfp4::row_scale_bytes(k, layout).unwrap();
    for (name, first_row) in [
        ("mtp/layer/attention/query_key", 0),
        ("mtp/layer/attention/gate_value", half),
    ] {
        let placement = plan.placement(name).unwrap();
        assert_eq!(placement.bytes, nvfp4::byte_size(half, k, layout).unwrap());
        let device = |offset: u64, bytes: u64| {
            download(
                &library,
                &lane,
                DevicePtr::NULL.offset(base + placement.offset + offset),
                bytes as usize,
            )
        };
        let file = |offset: u64, bytes: u64| read_file(&source, pack_offset + offset, bytes as usize);
        assert_eq!(
            sha(&device(0, half * code)),
            sha(&file(first_row * code, half * code)),
            "{name}: codes"
        );
        assert_eq!(
            sha(&device(nvfp4::scale_offset(half, k), half * row_scale)),
            sha(&file(
                nvfp4::scale_offset(rows, k) + first_row * row_scale,
                half * row_scale
            )),
            "{name}: scales"
        );
        assert_eq!(
            device(nvfp4::trailer_offset(half, k, layout).unwrap(), 4),
            file(nvfp4::trailer_offset(rows, k, layout).unwrap(), 4),
            "{name}: global scale"
        );
    }
    let _ = pack;
    lane.synchronize().unwrap();
    drop(lane);
    owner.shutdown(runtime).unwrap();
}

/// A truncated copy of the artifact, under the scratch directory, deleted when the guard drops.
struct Truncated(PathBuf);

impl Drop for Truncated {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// A copy of the artifact's header, metadata and table that ends at `bytes`: the file is sparse, so it costs a few
/// hundred KiB on disk, and the table names payloads past its end.
fn truncated_copy(artifact: &Path, bytes: u64) -> Truncated {
    let copy =
        Truncated(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("corrupt-artifact-{}.edrl", process::id())));
    let header_bytes = read_file(artifact, 0, 48);
    let header = euhedral_inference_edrl::Header::parse(header_bytes.as_slice().try_into().unwrap()).unwrap();
    let mut out = File::create(&copy.0).unwrap();
    out.set_len(bytes).unwrap();
    out.write_all(&read_file(artifact, 0, header.data_offset() as usize))
        .unwrap();
    out.flush().unwrap();
    copy
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn startup_fails_before_device_alloc_on_corrupt_artifact() {
    let Some(machine) = machine() else { return };
    let truncated = truncated_copy(&machine.artifact, 64 << 20);
    let library = CudaLibrary::load(&machine.library).unwrap();
    let before = library.free_bytes().unwrap();

    // The artifact ends inside its payloads: the table names tensors the file does not hold.
    let (outcome, _guard) = start(&settings(&truncated.0, 4096), &machine.library);
    let error = outcome.expect_err("a truncated artifact does not start");
    println!("{error}");
    assert!(matches!(error, EngineError::Artifact { .. }), "{error:?}");
    let message = error.to_string();
    assert!(message.contains("corrupt-artifact"), "{message}");
    assert!(
        message.contains("extends beyond the file") || message.contains("truncated"),
        "{message}"
    );
    // Nothing was allocated on the device: the 12 GiB arena would show.
    let after = library.free_bytes().unwrap();
    assert!(
        before.abs_diff(after) < (256 << 20),
        "free memory moved from {before} to {after}"
    );

    // A file that is not an artifact at all.
    fs::write(&truncated.0, vec![0x41u8; 4096]).unwrap();
    let (outcome, _guard) = start(&settings(&truncated.0, 4096), &machine.library);
    let error = outcome.expect_err("garbage does not start");
    assert!(matches!(error, EngineError::Artifact { .. }), "{error:?}");
    assert!(error.to_string().contains("magic"), "{error}");

    // A file that is not there.
    let (outcome, _guard) = start(&settings(Path::new("no-such-artifact.edrl"), 4096), &machine.library);
    let error = outcome.expect_err("a missing artifact does not start");
    assert!(matches!(error, EngineError::Io { .. }), "{error:?}");
    assert!(error.to_string().contains("no-such-artifact.edrl"), "{error}");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_plan_that_does_not_fit_fails_before_device_alloc() {
    let Some(machine) = machine() else { return };
    let library = CudaLibrary::load(&machine.library).unwrap();
    let before = library.free_bytes().unwrap();
    let mut settings = settings(&machine.artifact, 4096);
    settings.memory.gpu_limit = euhedral_inference_settings::Auto::Value(ByteSize::gib(2));
    let (outcome, _guard) = start(&settings, &machine.library);
    let error = outcome.expect_err("12 GiB of weights do not fit in 2 GiB");
    let EngineError::Plan(PlanError::WeightsDoNotFit {
        weights_needed,
        available,
        ..
    }) = &error
    else {
        panic!("expected a plan refusal, got {error:?}");
    };
    assert!(*weights_needed > 11 << 30 && *available == 2 << 30);
    let message = error.to_string();
    assert!(
        message.contains(&weights_needed.to_string()) && message.contains(&available.to_string()),
        "{message}"
    );
    let after = library.free_bytes().unwrap();
    assert!(
        before.abs_diff(after) < (256 << 20),
        "free memory moved from {before} to {after}"
    );
}

/// A sparse copy of the artifact of the same length: the header, the metadata, the table, the footer and every payload
/// under 1 MiB (norms, the trellis companions) are real; the big payloads read as zeros, which every format accepts.
fn sparse_copy(artifact: &Path) -> Truncated {
    use euhedral_inference_edrl::{Header, ReadPlan, Region, TensorTable, TokenizerFooter, content_end};

    let copy =
        Truncated(Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("sparse-artifact-{}.edrl", process::id())));
    let length = fs::metadata(artifact).unwrap().len();
    let probe = ReadPlan::probe(length).unwrap();
    let header_bytes = read_file(artifact, 0, 48);
    let header = Header::parse(header_bytes.as_slice().try_into().unwrap()).unwrap();
    let footer_range = probe.range(Region::TokenizerFooter).unwrap();
    let footer_bytes = read_file(artifact, footer_range.start, 24);
    let footer = TokenizerFooter::parse(footer_bytes.as_slice().try_into().unwrap()).unwrap();
    let plan = ReadPlan::new(&header, length, footer.as_ref()).unwrap();
    let table_range = plan.range(Region::Table).unwrap();
    let table = TensorTable::parse(
        &read_file(
            artifact,
            table_range.start,
            (table_range.end - table_range.start) as usize,
        ),
        &header,
        content_end(length, footer.as_ref()).unwrap(),
    )
    .unwrap();
    let mut out = File::create(&copy.0).unwrap();
    out.set_len(length).unwrap();
    let mut put = |offset: u64, bytes: &[u8]| {
        out.seek(SeekFrom::Start(offset)).unwrap();
        out.write_all(bytes).unwrap();
    };
    put(0, &read_file(artifact, 0, header.data_offset() as usize));
    for entry in table.entries().iter().filter(|entry| entry.bytes() < (1 << 20)) {
        put(
            entry.offset(),
            &read_file(artifact, entry.offset(), entry.bytes() as usize),
        );
    }
    put(footer_range.start, &footer_bytes);
    copy
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_nan_nvfp4_scale_stops_the_load_and_names_the_tensor() {
    use euhedral_inference_edrl::{Header, ReadPlan, Region, TensorTable, TokenizerFooter, content_end};

    let Some(machine) = machine() else { return };
    let sparse = sparse_copy(&machine.artifact);
    let length = fs::metadata(&sparse.0).unwrap().len();
    let header_bytes = read_file(&sparse.0, 0, 48);
    let header = Header::parse(header_bytes.as_slice().try_into().unwrap()).unwrap();
    let probe = ReadPlan::probe(length).unwrap();
    let footer_bytes = read_file(&sparse.0, probe.range(Region::TokenizerFooter).unwrap().start, 24);
    let footer = TokenizerFooter::parse(footer_bytes.as_slice().try_into().unwrap()).unwrap();
    let plan = ReadPlan::new(&header, length, footer.as_ref()).unwrap();
    let table_range = plan.range(Region::Table).unwrap();
    let table = TensorTable::parse(
        &read_file(
            &sparse.0,
            table_range.start,
            (table_range.end - table_range.start) as usize,
        ),
        &header,
        content_end(length, footer.as_ref()).unwrap(),
    )
    .unwrap();
    let head = table.get("text/draft_head").unwrap();
    let scale_plane = head.offset() + nvfp4::scale_offset(131_072, 5120);

    // The zeros load: every scale is 0.0 and the global scale 0.0 is finite and not negative.
    let settings = settings(&sparse.0, 4096);
    let (runtime, guard) = start(&settings, &machine.library);
    guard.shutdown(runtime.expect("an artifact of zeros loads")).unwrap();

    // One E4M3 NaN among the draft head's scales stops the load, and the error says which tensor and which byte.
    let mut file = fs::OpenOptions::new().write(true).open(&sparse.0).unwrap();
    file.seek(SeekFrom::Start(scale_plane + 1234)).unwrap();
    file.write_all(&[0x7F]).unwrap();
    drop(file);
    let (outcome, _guard) = start(&settings, &machine.library);
    let error = outcome.expect_err("a NaN scale does not load");
    println!("{error}");
    let EngineError::Payload { name, reason } = &error else {
        panic!("expected a payload error, got {error:?}");
    };
    assert_eq!(name, "text/draft_head");
    assert_eq!(reason, "NVFP4 block scale at byte 1234 is NaN");
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn startup_works_with_workers_that_poll_one_frame_at_a_time() {
    let Some(machine) = machine() else { return };
    let mut settings = settings(&machine.artifact, 4096);
    settings.host.max_batch_size = euhedral_inference_settings::Auto::Value(2);
    let (runtime, owner) = start(&settings, &machine.library);
    let runtime = runtime.expect("the engine starts in poll mode");
    println!(
        "poll mode: ready in {:.2} s, upload {:.2} s",
        runtime.report().ready.as_secs_f64(),
        runtime.report().load.as_secs_f64()
    );
    assert_eq!(runtime.weights().len(), 786);
    owner.shutdown(runtime).unwrap();
}
