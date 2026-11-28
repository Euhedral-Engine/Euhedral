//! The activations a trellis artifact's prepass meets: the BF16 forward of the checkpoint over calibration sequences,
//! every trellis linear's input rows multiplied by that linear's `suh` and rotated by the 128-point Hadamard, as the
//! engine's prepass does before it rounds to FP16. Fails, naming the linear, when an output would not fit FP16, and
//! writes the extremes per linear to `euhedral.trellis.extremes.report` when that key is set (the fixture
//! `native/tests/test_trellis.py` reads). `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test trellis_range_gpu -- --ignored --nocapture`.
//!
//! Keys: `euhedral.cuda.library`, `euhedral.quantization.cuda-libraries`, `euhedral.quantization.checkpoint` (BF16)
//! and `euhedral.quantization.calibration-tokens`, then the source of `suh`: `euhedral.trellis.checkpoint` (a finished
//! trellis checkpoint holding `<linear>.suh`) or, when that is not set,
//! `euhedral.qwen.rco-artifact` (an artifact holding `<object>.suh` and `<object>.segments`, one `suh` row per
//! segment). `euhedral.trellis.extremes.sequences` (default 2) sequences run. The test skips when a key is missing.
//! The builtins library of NVRTC must be on `LD_LIBRARY_PATH`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    clippy::print_stdout,
    reason = "a test: reads files and drives the device with synchronous calls, and prints what it measured"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod checkpoint_support;
mod forward_support;
mod range_support;

use std::collections::HashMap;
use std::fs::{File, create_dir_all, read, write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use checkpoint_support::drive;
use euhedral_core::generics::LatticeSource;
use euhedral_inference_cuda::{KernelOps, PinnedPool};
use euhedral_inference_edrl::{TensorTable, TrellisCompanions};
use euhedral_inference_quantization::assemble::ArtifactFile;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::{ReadRange, Safetensors};
use euhedral_inference_quantization::exl3::linears;
use euhedral_inference_quantization::forward::{LaneContext, LayerForward, LayerWeights, Observation, Target, stage};
use euhedral_inference_quantization::npy::Npy;
use euhedral_io::AsyncIo;
use forward_support::{Rig, rig};
use range_support::{Extremes, half_to_f32, prepass_extremes, report_json, segment_of};

const UPLOAD_CHUNK: usize = 64 << 20;
const EMBEDDINGS: &str = "model.language_model.embed_tokens.weight";
const DEFAULT_SEQUENCES: usize = 2;

/// A linear's `suh` under the name its extremes are reported by.
struct LinearSuh {
    linear: String,
    suh: Vec<f32>,
}

/// Where the `suh` of every linear comes from.
enum SuhSource {
    /// A trellis checkpoint: `<tensor>.suh` of each quantized linear.
    Checkpoint(Arc<Safetensors>, Arc<AsyncIo>),
    /// An artifact: `<object>.suh` rows, one per segment of `<object>.segments`.
    Artifact(File, TensorTable),
}

impl SuhSource {
    /// The `suh` each observed object's input is multiplied by, by object name, for layer `index`.
    fn layer(&self, objects: &Qwen38Objects, index: usize) -> HashMap<String, Vec<LinearSuh>> {
        let mut by_object: HashMap<String, Vec<LinearSuh>> = HashMap::new();
        for linear in linears::layer(objects, index).unwrap() {
            if !linear.quantized {
                continue;
            }
            match self {
                SuhSource::Checkpoint(store, io) => {
                    let suh = read_half(io, store, &format!("{}.suh", linear.tensor));
                    by_object
                        .entry(linear.hessian_object.clone())
                        .or_default()
                        .push(LinearSuh {
                            linear: linear.tensor,
                            suh,
                        });
                }
                SuhSource::Artifact(file, table) => {
                    let placements = linears::placements(objects, index, &linear.tensor).unwrap();
                    let several = placements.iter().any(|place| place.object != placements[0].object);
                    let mut seen: Vec<(String, usize)> = Vec::new();
                    for place in placements {
                        let Some(rows) = table.get(&format!("{}.suh", place.object)) else {
                            continue;
                        };
                        let [segments, k] = rows.shape()[..] else {
                            panic!("{}.suh is not a matrix", place.object);
                        };
                        let ends = read_ends(file, table, &place.object);
                        assert_eq!(
                            ends.len() as u64,
                            segments,
                            "{}: segments differ from suh rows",
                            place.object
                        );
                        let segment = segment_of(&ends, place.object_row, place.rows)
                            .unwrap_or_else(|why| panic!("{} in {}: {why}", linear.tensor, place.object));
                        if seen.contains(&(place.object.clone(), segment)) {
                            continue;
                        }
                        seen.push((place.object.clone(), segment));
                        let mut bytes = vec![0u8; 2 * k as usize];
                        file.read_exact_at(&mut bytes, rows.offset() + 2 * k * segment as u64)
                            .unwrap();
                        let suh = bytes
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|pair| half_to_f32(u16::from_le_bytes(*pair)))
                            .collect();
                        let name = if several {
                            format!("{}@{}", linear.tensor, place.object)
                        } else {
                            linear.tensor.clone()
                        };
                        by_object
                            .entry(place.object)
                            .or_default()
                            .push(LinearSuh { linear: name, suh });
                    }
                }
            }
        }
        by_object
    }
}

fn read_ends(file: &File, table: &TensorTable, object: &str) -> Vec<u32> {
    let entry = table
        .get(&format!("{object}.segments"))
        .expect("a .suh companion has .segments");
    let mut bytes = vec![0u8; usize::try_from(entry.bytes()).unwrap()];
    file.read_exact_at(&mut bytes, entry.offset()).unwrap();
    TrellisCompanions::parse(&bytes).unwrap()
}

/// Tensor `name` of `store`, as half-precision values widened.
fn read_half(io: &Arc<AsyncIo>, store: &Arc<Safetensors>, name: &str) -> Vec<f32> {
    let bytes = usize::try_from(store.tensor(name).unwrap().bytes).unwrap();
    let held = drive(io, |done| store.read(name, 0, vec![0u8; bytes], done)).unwrap();
    held.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| half_to_f32(u16::from_le_bytes(*pair)))
        .collect()
}

/// Opens artifact `path` and its table, driving `io` by hand.
fn open_artifact(io: &Arc<AsyncIo>, path: &Path) -> ArtifactFile {
    let slot: Arc<Mutex<Option<ArtifactFile>>> = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&slot);
    ArtifactFile::open(
        io,
        path,
        false,
        Arc::new(move |opened| *sink.lock().unwrap() = Some(opened.expect("the artifact opens"))),
    )
    .unwrap();
    loop {
        if let Some(artifact) = slot.lock().unwrap().take() {
            return artifact;
        }
        io.reap();
        let mut frames = Vec::new();
        io.pull(&mut |frame| frames.push(frame), &mut |_| false, u64::MAX);
        for mut frame in frames {
            let _ = frame.execute();
            frame.do_finally();
        }
    }
}

/// The calibration sequences: the first `limit` rows of an `[n, length]` `i32` array.
fn read_tokens(path: &Path, limit: usize) -> Vec<Vec<i32>> {
    let bytes = read(path).unwrap();
    let npy = Npy::parse(&bytes).unwrap();
    let [count, length] = npy.shape[..] else {
        panic!("{}: no 2-D shape", path.display());
    };
    let all = npy.to_vec::<i32>().unwrap();
    all.chunks(usize::try_from(length).unwrap())
        .take(limit.min(usize::try_from(count).unwrap()))
        .map(<[i32]>::to_vec)
        .collect()
}

/// The embedding rows (BF16) of every token of every sequence, in order.
fn embed(io: &Arc<AsyncIo>, store: &Arc<Safetensors>, tokens: &[Vec<i32>], hidden: usize) -> Vec<u8> {
    let vocabulary = store.tensor(EMBEDDINGS).unwrap().shape[0];
    let ranges: Vec<ReadRange<Vec<u8>>> = tokens
        .iter()
        .flatten()
        .map(|token| {
            let token = u64::try_from(*token)
                .ok()
                .filter(|id| *id < vocabulary)
                .expect("a token outside the vocabulary");
            ReadRange {
                tensor: EMBEDDINGS.to_owned(),
                skip: token * 2 * hidden as u64,
                memory: vec![0u8; 2 * hidden],
            }
        })
        .collect();
    drive(io, |done| store.read_ranges(ranges, done)).unwrap().concat()
}

fn property(key: &str) -> Option<PathBuf> {
    test_support::require_property(key).map(PathBuf::from)
}

/// Runs the layers over `hidden` (BF16 rows of every sequence, in place on the device), and the extremes of every
/// target's prepass.
fn range_of_every_layer(
    rig: &mut Rig,
    io: &Arc<AsyncIo>,
    objects: &Qwen38Objects,
    source: &SuhSource,
    embedded: &[u8],
    sequences: u32,
    length: u32,
) -> Vec<Extremes> {
    let rows = sequences * length;
    let (library, pool) = (Arc::clone(&rig.library), Arc::clone(&rig.pool));
    // The readbacks are separate from the staged weights: a layer's observed inputs are held until its work retires.
    let readbacks = PinnedPool::new(&library, 1 << 30);
    let config = &objects.config;
    let hidden = rig.upload(embedded);
    let hidden_ptr = hidden.ptr();
    let mut forward = rig.run(|context| LayerForward::new(context.ops(), &library, &pool, config, rows));
    let mut found = Vec::new();
    for index in 0..config.layers {
        let targets = source.layer(objects, index);
        let staged = drive(io, |done| {
            stage(objects, Target::Layer(index), |bytes| pool.acquire(bytes), done)
        })
        .unwrap();
        // SAFETY: `staged` outlives the call, and `rig.run` waits for the lane to retire the copies.
        let weights = rig.run(|context| unsafe { LayerWeights::upload(context.ops(), &staged, UPLOAD_CHUNK) });
        let mut seen = Vec::new();
        rig.run(|context| {
            let mut observer = |input: &Observation<'_>, lane: &mut LaneContext<'_, _>| {
                if !targets.contains_key(input.object) {
                    return Ok(());
                }
                let bytes = input.rows as usize * input.columns as usize * 2;
                let mut loan = readbacks.acquire(bytes)?;
                // SAFETY: `loan` is held in `seen`, which outlives the run: it returns once the lane has retired the copy.
                unsafe {
                    lane.ops()
                        .copy_device_to_readback(loan.buffer_mut(), 0, input.input, bytes)
                }?;
                seen.push((input.object.to_owned(), input.columns as usize, loan));
                Ok(())
            };
            forward.forward(context, &weights, hidden_ptr, sequences, length, Some(&mut observer))
        });
        forward.retire();
        rig.run(|context| weights.release(context.ops()));
        for (object, columns, loan) in &seen {
            let x = &loan.buffer().as_slice()[..loan.len()];
            for target in &targets[object] {
                assert_eq!(
                    target.suh.len(),
                    *columns,
                    "{}: suh does not span the input of {object}",
                    target.linear
                );
                found.push(prepass_extremes(
                    &target.linear,
                    x,
                    rows as usize,
                    *columns,
                    &target.suh,
                ));
            }
        }
        println!("layer {index}: {} linears", found.len());
    }
    rig.run(|context| forward.release(context.ops()));
    found
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC, the math libraries, the checkpoint, the calibration tokens and a source of suh"]
fn prepass_outputs_fit_fp16() {
    let (Some(_), Some(_), Some(checkpoint), Some(tokens_file)) = (
        property("euhedral.cuda.library"),
        property("euhedral.quantization.cuda-libraries"),
        property("euhedral.quantization.checkpoint"),
        property("euhedral.quantization.calibration-tokens"),
    ) else {
        return;
    };
    let Some(trellis) = property("euhedral.trellis.checkpoint").or_else(|| property("euhedral.qwen.rco-artifact"))
    else {
        return;
    };
    let from_checkpoint =
        test_support::test_setting("euhedral.trellis.checkpoint").is_some_and(|set| Path::new(&set) == trellis);
    let sequences = test_support::test_setting("euhedral.trellis.extremes.sequences")
        .map_or(DEFAULT_SEQUENCES, |text| {
            text.parse().expect("euhedral.trellis.extremes.sequences is a count")
        });
    let tokens = read_tokens(&tokens_file, sequences);
    let length = tokens[0].len();

    let io = AsyncIo::open().unwrap();
    let store = Arc::new(drive(&io, |done| Safetensors::open(&io, &checkpoint, done)).unwrap());
    let objects = drive(&io, |done| Qwen38Objects::open(Arc::clone(&store), &checkpoint, done)).unwrap();
    let source = if from_checkpoint {
        let quantized = Arc::new(drive(&io, |done| Safetensors::open(&io, &trellis, done)).unwrap());
        SuhSource::Checkpoint(quantized, Arc::clone(&io))
    } else {
        let artifact = open_artifact(&io, &trellis);
        SuhSource::Artifact(File::open(&trellis).unwrap(), artifact.table().clone())
    };
    let embedded = embed(&io, &store, &tokens, usize::try_from(objects.config.hidden).unwrap());

    let mut rig = rig();
    let found = range_of_every_layer(
        &mut rig,
        &io,
        &objects,
        &source,
        &embedded,
        u32::try_from(tokens.len()).unwrap(),
        u32::try_from(length).unwrap(),
    );
    rig.finish();

    if let Some(report) = test_support::test_setting("euhedral.trellis.extremes.report") {
        let path = PathBuf::from(report);
        create_dir_all(path.parent().unwrap()).unwrap();
        write(&path, report_json(&found)).unwrap();
    }
    assert!(!found.is_empty(), "no trellis linear had a suh: nothing was checked");
    let widest = found
        .iter()
        .max_by(|a, b| a.max_prepass.total_cmp(&b.max_prepass))
        .unwrap();
    println!(
        "{} linears over {} sequences of {length}; largest prepass output {} in {}",
        found.len(),
        tokens.len(),
        widest.max_prepass,
        widest.linear
    );
    for e in &found {
        assert!(e.fits_fp16(), "{}: prepass reaches {}", e.linear, e.max_prepass);
    }
}
