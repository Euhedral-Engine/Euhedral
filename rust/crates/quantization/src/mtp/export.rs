//! The `export-drafter` run: an artifact whose MTP layer is a trained drafter's. Every object of the base artifact is
//! kept, except that the MTP layer's objects are replaced: the projections are NVFP4 (round to nearest,
//! [`nvfp4`](super::nvfp4), as the drafter trained them) and the norms BF16, from a `train-drafter` parameter file (FP32
//! values in the drafter's parameter order after the file's header) or, without one, from the checkpoint's MTP weights.
//! The base's own MTP objects and their companions are left out; the base's tokenizer is carried.
//!
//! The new objects are written to `<out>.mtp-payloads` first, in the order the converter writes them, and the artifact
//! is then written by [`assemble::write`](crate::assemble::write), which copies each payload from its source and
//! renames the finished file, so a run that stops leaves no output. The payload file is removed whatever the outcome.
//!
//! [`export`] runs on the calling thread, which waits for each call as the startup of `train-drafter` does: it is for
//! a binary's main thread, never a frame.

use std::io::Error as IoError;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use euhedral_graph::Lake;
use euhedral_inference_edrl::{ModelMetadata, TensorDataType, TensorEntry, WeightFormat, WeightLayout};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, OwnedFile};
use thiserror::Error;

use super::layout::{
    DOWN, EMBEDDING_NORM, FINAL_NORM, GATE_UP, HIDDEN_NORM, INPUT_NORM, INPUT_PROJECTION, KEY_NORM, NORMS, OUTPUT,
    POST_ATTENTION_NORM, PROJECTIONS, QUERY_KEY_GATE_VALUE, QUERY_NORM,
};
use super::nvfp4::{self, Nvfp4Error};
use crate::assemble::{ArtifactFile, AssembleError, Piece, View, write};
use crate::checkpoint::objects::{Config, Qwen38Objects};
use crate::checkpoint::safetensors::Safetensors;
use crate::checkpoint::writer::{DEFAULT_BUFFER_BYTES, bytes_source, write_file};
use crate::checkpoint::{CheckpointError, io::open_read};
use crate::durable;
use crate::mixed::QuantOption;
use crate::resume::{HEADER, ResumeError, ResumeFile, Then};
use crate::step::idle;

/// The MTP objects in the order the converter writes them.
const ORDER: [&str; 12] = [
    INPUT_PROJECTION,
    EMBEDDING_NORM,
    HIDDEN_NORM,
    INPUT_NORM,
    QUERY_KEY_GATE_VALUE,
    QUERY_NORM,
    KEY_NORM,
    OUTPUT,
    POST_ATTENTION_NORM,
    GATE_UP,
    DOWN,
    FINAL_NORM,
];

/// The checkpoint tensor of each norm, in the order of [`NORMS`].
const NORM_TENSORS: [&str; 7] = [
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.norm.weight",
];

/// The longest one call is waited for: a bound on the wait, not a rate.
const LONGEST_CALL: Duration = Duration::from_secs(24 * 60 * 60);

/// What an export is asked to do.
#[derive(Clone, Debug)]
pub struct Settings {
    /// A `train-drafter` parameter file; without one the checkpoint's own MTP weights are exported.
    pub parameters: Option<PathBuf>,
    /// The base model's checkpoint directory: the shapes of the layer, and its weights without `parameters`.
    pub model: PathBuf,
    /// The base artifact.
    pub artifact: PathBuf,
    /// The artifact to write.
    pub out: PathBuf,
}

/// Why an export did not happen.
#[derive(Debug, Error)]
pub enum ExportError {
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    #[error(transparent)]
    Artifact(#[from] AssembleError),
    #[error(transparent)]
    Resume(#[from] ResumeError),
    #[error(transparent)]
    Rounding(#[from] Nvfp4Error),
    #[error("{0}")]
    Io(#[from] IoError),
    /// An input the export cannot use: it is not what the drafter's layer needs.
    #[error("{0}")]
    Invalid(String),
    /// A call that never answered.
    #[error("{0}")]
    Broken(String),
}

fn invalid(message: String) -> ExportError {
    ExportError::Invalid(message)
}

/// Starts a call that reports to the `Done` it is given, and waits for its answer.
fn call<T, E, S>(start: impl FnOnce(Arc<dyn Fn(Result<T, E>) + Send + Sync>) -> Result<(), S>) -> Result<T, ExportError>
where
    T: Send + 'static,
    E: Into<ExportError> + 'static,
    S: Into<ExportError>,
{
    let cell = Handoff::new();
    let sender = Arc::clone(&cell);
    start(Arc::new(move |result| sender.send(result.map_err(Into::into)))).map_err(Into::into)?;
    wait(&cell)
}

/// Starts a call of a resume file, whose continuation runs a frame, and waits for its answer.
fn then_call<T: Send + 'static>(start: impl FnOnce(Then<T>) -> Result<(), ResumeError>) -> Result<T, ExportError> {
    let cell = Handoff::new();
    let sender = Arc::clone(&cell);
    start(Box::new(move |result| {
        sender.send(result.map_err(ExportError::from));
        idle()
    }))?;
    wait(&cell)
}

#[allow(
    clippy::disallowed_methods,
    reason = "startup and shutdown of a command: the calling thread is the binary's own, never a worker"
)]
fn wait<T>(cell: &Arc<Handoff<Result<T, ExportError>>>) -> Result<T, ExportError> {
    cell.wait(LONGEST_CALL)
        .map_err(|_| ExportError::Broken("a call did not answer in time".to_owned()))?
}

/// The shape of one MTP object: `[rows, columns]` for a projection, `[width]` for a norm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Projection { rows: u64, columns: u32 },
    Norm { width: u64 },
}

impl Shape {
    fn count(self) -> u64 {
        match self {
            Shape::Projection { rows, columns } => rows * u64::from(columns),
            Shape::Norm { width } => width,
        }
    }

    fn dimensions(self) -> Vec<u64> {
        match self {
            Shape::Projection { rows, columns } => vec![rows, u64::from(columns)],
            Shape::Norm { width } => vec![width],
        }
    }
}

/// Each MTP object's shape, in the order of the parameter file (the projections, then the norms).
fn shapes(objects: &Qwen38Objects) -> Result<Vec<(&'static str, Shape)>, ExportError> {
    let mut shapes = Vec::new();
    for name in PROJECTIONS {
        let source = objects.object(name)?;
        let columns = u32::try_from(source.columns)
            .map_err(|_| invalid(format!("{name} has {} columns, too many", source.columns)))?;
        if columns % 128 != 0 {
            return Err(Nvfp4Error::RowLength(columns).into());
        }
        shapes.push((
            name,
            Shape::Projection {
                rows: source.rows,
                columns,
            },
        ));
    }
    let Config { hidden, head_dim, .. } = objects.config;
    for name in NORMS {
        let width = if name == QUERY_NORM || name == KEY_NORM {
            head_dim
        } else {
            hidden
        };
        shapes.push((name, Shape::Norm { width }));
    }
    Ok(shapes)
}

/// Where each object's FP32 values start in a parameter file: after its header, in the parameter order.
fn parameter_offsets(shapes: &[(&'static str, Shape)]) -> Vec<(&'static str, u64)> {
    let mut offset = HEADER;
    shapes
        .iter()
        .map(|(name, shape)| {
            let at = offset;
            offset += 4 * shape.count();
            (*name, at)
        })
        .collect()
}

/// The little-endian FP32 values of `bytes`.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The FP32 values of little-endian BF16 `bytes`: a BF16 is the top half of a float.
fn widen(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|word| f32::from_bits(u32::from(u16::from_le_bytes(*word)) << 16))
        .collect()
}

/// The BF16 bits of `value`, rounded to nearest even; every NaN becomes the quiet NaN `0x7fc0`.
fn bf16(value: f32) -> u16 {
    if value.is_nan() {
        return 0x7fc0;
    }
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Where an object's values come from.
enum Values<'a> {
    /// The parameter file, and where each object starts in it.
    Parameters(&'a ResumeFile, Vec<(&'static str, u64)>),
    /// The checkpoint.
    Checkpoint(&'a Qwen38Objects),
}

impl Values<'_> {
    /// The FP32 values of object `name` of `shape`.
    fn read(&self, name: &'static str, shape: Shape) -> Result<Vec<f32>, ExportError> {
        match self {
            Values::Parameters(file, offsets) => {
                let at = offsets
                    .iter()
                    .find(|(known, _)| *known == name)
                    .map_or(HEADER, |(_, at)| *at);
                let len = usize::try_from(4 * shape.count())
                    .map_err(|_| invalid(format!("{name} does not fit in memory")))?;
                Ok(floats(&then_call(|then| file.read(at, len, then))?))
            }
            Values::Checkpoint(objects) => {
                let bf16 = match shape {
                    Shape::Projection { .. } => {
                        let source = objects.object(name)?;
                        call(|done| objects.read_bytes(&source, done))?
                    }
                    Shape::Norm { .. } => {
                        let tensor = NORMS
                            .iter()
                            .zip(NORM_TENSORS)
                            .find(|(norm, _)| **norm == name)
                            .map_or("", |(_, tensor)| tensor);
                        let size = usize::try_from(2 * shape.count())
                            .map_err(|_| invalid(format!("{name} does not fit in memory")))?;
                        call(|done| objects.store().read_tensor(tensor, vec![0u8; size], done))?
                    }
                };
                Ok(widen(&bf16))
            }
        }
    }
}

/// The payload of an object from its FP32 `values`: NVFP4 for a projection, BF16 for a norm.
fn payload(values: &[f32], shape: Shape) -> Result<Vec<u8>, ExportError> {
    Ok(match shape {
        Shape::Projection { rows, columns } => nvfp4::round(values, rows, columns)?,
        Shape::Norm { .. } => values.iter().flat_map(|value| bf16(*value).to_le_bytes()).collect(),
    })
}

/// The new objects, packed one after the other, and their descriptors by offset into that packing.
struct Payloads {
    bytes: Vec<u8>,
    entries: Vec<TensorEntry>,
}

/// Builds every MTP object of `shapes` in [`ORDER`] from `values`.
fn build(shapes: &[(&'static str, Shape)], values: &Values<'_>) -> Result<Payloads, ExportError> {
    let mut built = Payloads {
        bytes: Vec::new(),
        entries: Vec::new(),
    };
    for name in ORDER {
        let Some((_, shape)) = shapes.iter().find(|(known, _)| *known == name) else {
            return Err(invalid(format!("{name} is not an object of the drafter")));
        };
        let payload = payload(&values.read(name, *shape)?, *shape)?;
        let (format, layout) = match shape {
            Shape::Projection { .. } => (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1),
            Shape::Norm { .. } => (WeightFormat::Bf16, WeightLayout::ContiguousLeV1),
        };
        built.entries.push(TensorEntry::new(
            name,
            shape.dimensions(),
            TensorDataType::Bf16,
            format,
            layout,
            built.bytes.len() as u64,
            payload.len() as u64,
        ));
        built.bytes.extend_from_slice(&payload);
    }
    Ok(built)
}

/// The base's objects, with the new MTP objects where the base's first MTP object was and the base's own left out.
fn view(
    base: &ArtifactFile,
    metadata: ModelMetadata,
    entries: Vec<TensorEntry>,
    payloads: (&Arc<OwnedFile>, &Path),
) -> Result<View, ExportError> {
    let mut pieces = Vec::new();
    let mut placed = false;
    // The writer reads the pieces' source, not their option: every piece is labelled with the one that names no
    // calibration.
    let label = QuantOption::Nvfp4;
    for entry in base.table().entries() {
        if entry.name().starts_with("mtp/") {
            if !placed {
                placed = true;
                for new in &entries {
                    pieces.push(Piece {
                        entry: new.clone(),
                        option: label,
                        file: Arc::clone(payloads.0),
                        path: payloads.1.to_path_buf(),
                    });
                }
            }
            continue;
        }
        pieces.push(Piece {
            entry: entry.clone(),
            option: label,
            file: Arc::clone(base.file()),
            path: base.path().to_path_buf(),
        });
    }
    if !placed {
        return Err(invalid(format!("{} has no MTP layer", base.path().display())));
    }
    Ok(View {
        metadata,
        pieces,
        tokenizer: base.tokenizer().cloned(),
        sources: vec![Arc::clone(base.file()), Arc::clone(payloads.0)],
    })
}

/// The name of the file the new objects are written to before the artifact: beside `out`.
fn payloads_path(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap_or_default().to_owned();
    name.push(".mtp-payloads");
    out.with_file_name(name)
}

/// Writes the artifact `settings` describe, copying in buffers of at most `copy_buffer_bytes`. Waits for each call, and
/// returns once `settings.out` is complete under its name, or has failed and left neither it nor the payload file.
pub fn export(
    settings: &Settings,
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    copy_buffer_bytes: usize,
) -> Result<(), ExportError> {
    let base = call(|done| ArtifactFile::open(io, &settings.artifact, true, done))?;
    // Refused before any work: the artifact the writer would refuse after it.
    if base.tokenizer().is_none() {
        return Err(invalid(format!("{} carries no tokenizer", settings.artifact.display())));
    }
    if !base
        .table()
        .entries()
        .iter()
        .any(|entry| entry.name().starts_with("mtp/"))
    {
        return Err(invalid(format!("{} has no MTP layer", settings.artifact.display())));
    }
    let store = Arc::new(call(|done| Safetensors::open(io, &settings.model, done))?);
    let objects = call(|done| Qwen38Objects::open(store, &settings.model, done))?;
    let shapes = shapes(&objects)?;
    let _unit = host.admit().map_err(|_| ExportError::Resume(ResumeError::LakeClosed))?;
    let built = match &settings.parameters {
        Some(path) => {
            let file = then_call(|then| ResumeFile::open_read_only(io, host, path, then))?;
            let expected = HEADER + 4 * shapes.iter().map(|(_, shape)| shape.count()).sum::<u64>();
            let found = then_call(|then| file.len(then))?;
            if found != expected {
                return Err(invalid(format!(
                    "{} holds {found} bytes, a drafter of this checkpoint's shape needs {expected}",
                    path.display()
                )));
            }
            build(&shapes, &Values::Parameters(&file, parameter_offsets(&shapes)))?
        }
        None => build(&shapes, &Values::Checkpoint(&objects))?,
    };
    let payloads = payloads_path(&settings.out);
    let size = built.bytes.len() as u64;
    call(|done| {
        write_file(
            &payloads,
            size,
            bytes_source(built.bytes),
            DEFAULT_BUFFER_BYTES,
            io,
            host,
            done,
        )
    })?;
    let written = assemble(settings, io, host, copy_buffer_bytes, &base, built.entries, &payloads);
    let removed = then_call_durable(host, &payloads);
    written.and(removed)
}

/// Writes the artifact from the payload file, which is then closed.
fn assemble(
    settings: &Settings,
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    copy_buffer_bytes: usize,
    base: &ArtifactFile,
    entries: Vec<TensorEntry>,
    payloads: &Path,
) -> Result<(), ExportError> {
    let file = Arc::new(call(|done| {
        open_read(io, payloads, Box::new(move |opened| done(opened)))
    })?);
    let composite = view(base, base.metadata().clone(), entries, (&file, payloads))?;
    call(|done| write(io, host, &composite, &settings.out, copy_buffer_bytes, done))
}

/// Removes `path` (a file that is not there is removed already), waiting for it.
fn then_call_durable(host: &Arc<Lake>, path: &Path) -> Result<(), ExportError> {
    let cell = Handoff::new();
    let sender = Arc::clone(&cell);
    durable::remove(
        host,
        path,
        Box::new(move |result| {
            sender.send(result.map_err(ExportError::from));
            idle()
        }),
    )
    .map_err(|_| ExportError::Resume(ResumeError::LakeClosed))?;
    wait(&cell)
}
