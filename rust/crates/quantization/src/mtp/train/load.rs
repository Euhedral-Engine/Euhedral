//! A run's startup: the device and its lane, the drafter's parameters (from the checkpoint, or from a saved state), the
//! draft head and the token embedding (from the artifact), the corpus, and the progress the run starts from.
//!
//! This runs on the calling thread before any frame of the run: it waits for each read and each copy, which a frame
//! never does. The run itself ([`run`](super::run)) is frames.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::{Completions, LanePool};
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, HostWeightsArena};
use euhedral_inference_engine::runtime::open_device;
use euhedral_io::OwnedFile;

use super::data::{Data, Files};
use super::error::RunError;
use super::lane::{CHUNK_BYTES, Resources, TrainLane};
use super::options::{Host, Log, Options};
use super::progress::{Progress, Resume};
use super::settings::{MAX_POSITIONS, Settings, expected_steps};
use super::state::{File, pieces};
use super::wait::{call, done, then};
use super::window::Addresses;
use crate::assemble::ArtifactFile;
use crate::checkpoint::io::{open_read, read_exact, read_file};
use crate::checkpoint::objects::{Config as ModelConfig, Qwen38Objects};
use crate::checkpoint::safetensors::Safetensors;
use crate::corpus;
use crate::kernels::{Module, ModuleOptions, Modules};
use crate::mtp::layout::{NORMS, PROJECTIONS};
use crate::mtp::{Config, Drafter, DrafterKernels, Shape, Work, plan_nvfp4_decode_bf16};
use crate::resume::ResumeFile;

/// The checkpoint tensors of the drafter's norms, in the order of [`NORMS`].
const NORM_TENSORS: [&str; 7] = [
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.norm.weight",
];

/// How the run begins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Start {
    /// From the checkpoint: the held-out sequences are evaluated before any training.
    Initial,
    /// From a saved state, which already had that evaluation.
    Continue,
    /// The saved state is of a run that ended: there is nothing to do.
    Complete,
}

/// What stays open for the lane.
#[derive(Debug)]
pub struct Keep {
    _gpu: Arc<CudaGpu>,
    _pool: Arc<LanePool>,
    _math: Arc<MathLibraries>,
    _library: Arc<CudaLibrary>,
}

/// A run ready to go.
pub struct Loaded {
    pub lane: TrainLane,
    pub data: Data,
    /// The width of the hidden state.
    pub width: u32,
    pub hidden: Arc<OwnedFile>,
    /// A token's row in the draft head, -1 for a token it does not hold.
    pub row_of_token: Vec<i32>,
    pub identity: String,
    pub progress: Progress,
    pub state: ResumeFile,
    pub start: Start,
    pub keep: Keep,
}

impl std::fmt::Debug for Loaded {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("Loaded").finish_non_exhaustive()
    }
}

fn align(value: u64) -> u64 {
    (value + 255) & !255
}

fn input(message: String) -> RunError {
    RunError::Input(message)
}

fn size(value: u64, what: &str) -> Result<usize, RunError> {
    usize::try_from(value).map_err(|_| input(format!("{what} of {value} bytes does not fit in memory")))
}

fn narrow(value: u64, what: &str) -> Result<u32, RunError> {
    u32::try_from(value).map_err(|_| input(format!("{what} {value} is too large")))
}

/// The little-endian words of `bytes`.
fn words(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

/// BF16 values as the FP32 values they are: a BF16 is the top half of a float.
pub fn widen(bf16: &[u8]) -> Vec<u8> {
    bf16.as_chunks::<2>()
        .0
        .iter()
        .flat_map(|word| (u32::from(u16::from_le_bytes(*word)) << 16).to_le_bytes())
        .collect()
}

/// What the run is: its corpus, model, artifact and the settings that shape its state. A saved state of another run is
/// not resumed.
pub fn identity(options: &Options, sequences: usize) -> Result<String, RunError> {
    let absolute = |path: &Path| std::path::absolute(path).map(|path| path.display().to_string());
    let settings = &options.settings;
    Ok(format!(
        "corpus={} sequences={sequences} model={} artifact={} depths={} rows={} keys={} tokens={} epochs={} rates={}/{} \
         warmup={} holdout={}",
        absolute(&options.corpus)?,
        absolute(&options.model)?,
        absolute(&options.artifact)?,
        settings.depths,
        settings.rows,
        settings.keys,
        settings.tokens,
        settings.epochs,
        settings.projection_rate,
        settings.norm_rate,
        settings.warmup,
        settings.holdout,
    ))
}

/// Reads `len` bytes at `offset` of `file`.
fn read(host: &Host<'_>, file: &Arc<OwnedFile>, offset: u64, len: u64, context: &str) -> Result<Vec<u8>, RunError> {
    let len = size(len, context)?;
    call(|reply| {
        let reply_to = Arc::clone(&reply);
        read_exact(
            host.io,
            Arc::clone(file),
            offset,
            len,
            context.to_owned(),
            Box::new(move |bytes| reply_to.send(bytes.map_err(RunError::from))),
        )
    })
}

/// The corpus's files in memory, checked.
fn open_corpus(
    options: &Options,
    host: &Host<'_>,
    width: u32,
    vocabulary: u64,
) -> Result<(Data, Arc<OwnedFile>), RunError> {
    let mut entries = corpus::entries(&options.corpus)?;
    if options.settings.limit > 0 {
        entries.truncate(options.settings.limit);
    }
    let file = |name: &str| call(|reply| read_file(host.io, &options.corpus.join(name), done(&reply)));
    let (tokens, labels) = (words(&file("tokens.bin")?), words(&file("labels.bin")?));
    let path = options.corpus.join("hidden.bin");
    let hidden_rows = fs::metadata(&path)?.len() / (2 * u64::from(width));
    let hidden = call(|reply| {
        open_read(
            host.io,
            &path,
            Box::new(move |opened| reply.send(opened.map(Arc::new).map_err(RunError::from))),
        )
    })?;
    let data = Data::new(
        Files {
            entries,
            tokens,
            labels,
            hidden_rows,
            vocabulary,
        },
        &options.settings,
    )?;
    Ok((data, hidden))
}

/// Everything the artifact gives the run.
struct Artifact {
    head: u64,
    draft_rows: u32,
    row_of_token: Vec<i32>,
    table: HostWeightsArena,
    scale_offset: u64,
}

/// The device buffers a run owns and the addresses they give.
struct Buffers {
    library: Arc<CudaLibrary>,
    owned: Vec<DeviceBuffer>,
}

impl Buffers {
    fn take(&mut self, bytes: u64) -> Result<u64, RunError> {
        let buffer = DeviceBuffer::alloc(&self.library, size(bytes.max(4), "a buffer")?)?;
        let address = buffer.addr();
        self.owned.push(buffer);
        Ok(address)
    }
}

/// Reads the draft head and the token embedding from the artifact; the head is decoded on the device after the lane
/// exists (see [`decode_head`]).
fn read_artifact(
    options: &Options,
    host: &Host<'_>,
    library: &Arc<CudaLibrary>,
    width: u32,
    vocabulary: u64,
) -> Result<(ArtifactBytes, Artifact), RunError> {
    let artifact = call(|reply| ArtifactFile::open(host.io, &options.artifact, false, done(&reply)))?;
    let entry = |name: &str| {
        artifact
            .table()
            .get(name)
            .ok_or_else(|| input(format!("{} has no {name}", options.artifact.display())))
    };
    let (head, ids, embedding) = (
        entry("text/draft_head")?,
        entry("text/draft_head_token_ids")?,
        entry("text/token_embedding")?,
    );
    let (draft_rows, k) = match head.shape() {
        [rows, k] => (
            narrow(*rows, "the draft head's rows")?,
            narrow(*k, "the draft head's width")?,
        ),
        other => return Err(input(format!("the draft head has shape {other:?}, not [rows, width]"))),
    };
    if k != width {
        return Err(input(format!(
            "the draft head is {k} wide, the checkpoint's hidden state {width}"
        )));
    }
    if ids.bytes() != 4 * u64::from(draft_rows) {
        return Err(input(format!(
            "the draft head has {draft_rows} rows and {} token ids",
            ids.bytes() / 4
        )));
    }
    let file = artifact.file();
    let ids = words(&read(
        host,
        file,
        ids.offset(),
        ids.bytes(),
        "the draft head's token ids",
    )?);
    let mut row_of_token = vec![-1i32; size(vocabulary, "the vocabulary")?];
    for (row, token) in ids.iter().enumerate() {
        let slot = usize::try_from(*token)
            .ok()
            .and_then(|token| row_of_token.get_mut(token))
            .ok_or_else(|| {
                input(format!(
                    "the draft head's row {row} is token {token}, outside the vocabulary"
                ))
            })?;
        *slot = row as i32;
    }
    let k_pad = k.div_ceil(128) * 128;
    let codes = u64::from(draft_rows) * u64::from(k_pad) / 2;
    let scale_offset = align(codes);
    let global_offset = align(scale_offset + u64::from(draft_rows) * u64::from(k_pad) / 16);
    if head.bytes() < global_offset + 4 {
        return Err(input(format!(
            "the draft head holds {} bytes, fewer than its layout needs",
            head.bytes()
        )));
    }
    let packed = read(host, file, head.offset(), head.bytes(), "the draft head")?;
    let (table_rows, table_width) = match embedding.shape() {
        [rows, columns] => (*rows, *columns),
        other => return Err(input(format!("the token embedding has shape {other:?}"))),
    };
    if table_width != u64::from(width) {
        return Err(input(format!(
            "the token embedding is {table_width} wide, the hidden state {width}"
        )));
    }
    let groups = table_width.div_ceil(128) * 2;
    let table_scales = align(table_rows * groups * 24);
    if embedding.bytes() < table_scales {
        return Err(input("the token embedding is shorter than its layout needs".to_owned()));
    }
    let bytes = read(host, file, embedding.offset(), embedding.bytes(), "the token embedding")?;
    let mut table = HostWeightsArena::alloc(library, bytes.len())?;
    table.as_mut_slice().copy_from_slice(&bytes);
    let head_address = 0;
    Ok((
        ArtifactBytes {
            packed,
            scale_offset,
            global_offset,
            k,
            k_pad,
        },
        Artifact {
            head: head_address,
            draft_rows,
            row_of_token,
            table,
            scale_offset: table_scales,
        },
    ))
}

/// The draft head as the artifact stores it.
struct ArtifactBytes {
    packed: Vec<u8>,
    scale_offset: u64,
    global_offset: u64,
    k: u32,
    k_pad: u32,
}

/// Decodes the artifact's NVFP4 head into BF16 at `head`, once, on the lane. For startup.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: a synchronous copy before any frame runs on the lane"
)]
fn decode_head(
    lane: &mut TrainLane,
    library: &Arc<CudaLibrary>,
    stored: &ArtifactBytes,
    rows: u32,
    head: u64,
) -> Result<(), RunError> {
    let staging = DeviceBuffer::alloc(library, stored.packed.len())?;
    staging.copy_from_host(0, &stored.packed)?;
    let base = staging.addr();
    let steps = plan_nvfp4_decode_bf16(
        base,
        base + stored.scale_offset,
        base + stored.global_offset,
        rows,
        stored.k_pad,
        stored.k,
        head,
    )?;
    lane.run_blocking(&[Work::Steps(steps)])
}

/// The parameters' values from the checkpoint, widened to FP32 and written to the lane.
fn load_checkpoint(objects: &Qwen38Objects, lane: &TrainLane) -> Result<(), RunError> {
    let store = objects.store();
    for parameter in lane.drafter().parameters() {
        let count = parameter.count();
        let bf16 = if let Some(at) = PROJECTIONS.iter().position(|name| *name == parameter.name) {
            let source = objects.object(PROJECTIONS[at])?;
            call(|reply| objects.read_bytes(&source, done(&reply)))?
        } else if let Some(at) = NORMS.iter().position(|name| *name == parameter.name) {
            let name = NORM_TENSORS[at];
            let bytes = store.tensor(name)?.bytes;
            let memory = vec![0u8; size(bytes, name)?];
            call(|reply| store.read_tensor(name, memory, done(&reply)))?
        } else {
            return Err(RunError::Broken(format!(
                "{} is not a parameter of the drafter",
                parameter.name
            )));
        };
        if bf16.len() as u64 != 2 * count {
            return Err(input(format!(
                "{} holds {} bytes in the checkpoint, the drafter needs {}",
                parameter.name,
                bf16.len(),
                2 * count
            )));
        }
        lane.put(parameter.values, &widen(&bf16))?;
    }
    Ok(())
}

/// Writes the saved state's pieces to the lane.
fn restore(host: &Host<'_>, lane: &TrainLane, state: &ResumeFile) -> Result<(), RunError> {
    let parameters = lane.drafter().parameters().to_vec();
    let counts: Vec<u64> = parameters.iter().map(|parameter| parameter.count()).collect();
    for piece in pieces(File::State, &counts, CHUNK_BYTES) {
        let bytes = call(|reply| state.read(piece.offset, piece.bytes, then(&reply)))?;
        let parameter = &parameters[piece.parameter];
        let base = match piece.part {
            super::state::Part::Values => parameter.values,
            super::state::Part::First => parameter.first,
            super::state::Part::Second => parameter.second,
            super::state::Part::Gradient => parameter.gradient,
        };
        lane.put(base + piece.inside, &bytes)?;
    }
    let _ = host;
    Ok(())
}

fn config_of(settings: &Settings, objects: &ModelConfig) -> Result<Config, RunError> {
    Ok(Config {
        shape: Shape {
            hidden: narrow(objects.hidden, "the hidden size")?,
            intermediate: narrow(objects.intermediate, "the intermediate size")?,
            heads: narrow(objects.heads, "the heads")?,
            kv_heads: narrow(objects.kv_heads, "the key and value heads")?,
            head_dim: narrow(objects.head_dim, "the head size")?,
            rotary_dim: narrow(objects.rotary_dim, "the rotary size")?,
            theta: objects.theta,
            eps: objects.eps as f32,
        },
        depths: settings.depths,
        rows: settings.rows,
        max_keys: settings.keys,
        max_positions: MAX_POSITIONS,
    })
}

/// Opens everything the run needs and brings it to where it starts.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the calling thread is the binary's own, it waits for each read and copy before any frame runs"
)]
pub fn load(options: &Options, host: &Host<'_>, log: &Log) -> Result<Loaded, RunError> {
    let settings = options.settings;
    settings.check()?;
    fs::create_dir_all(&options.out)?;
    let store = Arc::new(call(|reply| Safetensors::open(host.io, &options.model, done(&reply)))?);
    let objects = call(|reply| Qwen38Objects::open(Arc::clone(&store), &options.model, done(&reply)))?;
    let config = config_of(&settings, &objects.config)?;
    config.validate()?;
    let (width, vocabulary) = (config.shape.hidden, objects.config.vocabulary);

    let library = CudaLibrary::load(&options.library)?;
    let (data, hidden) = open_corpus(options, host, width, vocabulary)?;
    let identity = identity(options, data.entries().len())?;
    let (stored, mut artifact) = read_artifact(options, host, &library, width, vocabulary)?;

    let gpu = open_device(&library).map_err(|error| RunError::Input(error.to_string()))?;
    let mut modules = Modules::<CudaModule>::open(
        &gpu,
        ModuleOptions::beside(&library).map_err(|error| input(error.to_string()))?,
    )?;
    let (train, drafter_module) = (modules.load(Module::Train)?, modules.load(Module::Drafter)?);
    let kernels = DrafterKernels::new(&train, &drafter_module);
    let (pool, mut streams) = gpu.open_lanes(&["train-drafter"], StreamOptions::default())?;
    let stream = streams
        .pop()
        .ok_or_else(|| RunError::Broken("the device opened no lane".to_owned()))?;
    let completions = Completions::new(&pool);
    (host.attach)(Arc::clone(&completions) as Arc<dyn LatticeSource>).map_err(RunError::Input)?;
    let math = MathLibraries::load(&options.math)?;
    let handles = math.lane_handles().map_err(|failed| {
        if let Some(leftover) = failed.leftover {
            let _ = leftover.close();
        }
        RunError::from(failed.error)
    })?;

    let arena = DeviceBuffer::alloc(&library, size(Drafter::arena_bytes(&config)?, "the drafter's arena")?)?;
    let drafter = Drafter::new(&config, arena.addr())?;
    let mut buffers = Buffers {
        library: Arc::clone(&library),
        owned: Vec::new(),
    };
    let (rows, depths) = (u64::from(settings.rows), u64::from(settings.depths));
    let wide = u64::from(width);
    artifact.head = buffers.take(2 * u64::from(artifact.draft_rows) * wide)?;
    let head_rows = u64::from(super::settings::HEAD_ROWS);
    let draft_rows = u64::from(artifact.draft_rows);
    let hidden_bf16 = buffers.take(2 * rows * wide)?;
    let hidden_input = buffers.take(4 * rows * wide)?;
    let mut embedded = Vec::new();
    for _ in 0..depths {
        embedded.push(buffers.take(4 * rows * wide)?);
    }
    let addresses = Addresses {
        hidden: width,
        draft_rows: artifact.draft_rows,
        depths: settings.depths,
        scale: 1.0 / settings.tokens as f32,
        head: artifact.head,
        // SAFETY: the table is kept by the lane for as long as the addresses are used, and nothing writes it after this.
        table: unsafe { artifact.table.device_address()? }.get(),
        scale_offset: artifact.scale_offset,
        hidden_bf16,
        hidden_input,
        ids: buffers.take(4 * rows * depths)?,
        labels: buffers.take(4 * rows * depths)?,
        predicted: buffers.take(4 * rows * depths)?,
        label_log_probabilities: buffers.take(4 * rows * depths)?,
        loss: buffers.take(8 * depths)?,
        embedded,
        head_input: buffers.take(2 * rows * wide)?,
        logits: buffers.take(4 * head_rows * draft_rows)?,
        logits_bf16: buffers.take(2 * head_rows * draft_rows)?,
    };
    let head = artifact.head;
    let resources = Resources {
        kernels,
        handles,
        drafter,
        addresses,
        rows: settings.rows,
        arena,
        buffers: buffers.owned,
        table: artifact.table,
        completions,
    };
    let mut lane = TrainLane::new(stream, &library, resources)?;
    decode_head(&mut lane, &library, &stored, artifact.draft_rows, head)?;

    let expected = expected_steps(data.lengths(), settings.tokens);
    let state = call(|reply| ResumeFile::open(host.io, host.lake, &options.out.join("resume.bin"), then(&reply)))?;
    let saved = call(|reply| state.header(then(&reply)))?;
    let resume = Progress::resume(settings, data.training().to_vec(), expected, &identity, &saved)?;
    let (progress, start) = match resume {
        Resume::Fresh => {
            load_checkpoint(&objects, &lane)?;
            lane.prepare(true)?;
            (
                Progress::new(settings, data.training().to_vec(), expected),
                Start::Initial,
            )
        }
        Resume::Continue(progress) => {
            restore(host, &lane, &state)?;
            lane.prepare(false)?;
            (progress, Start::Continue)
        }
        Resume::Complete(progress) => (progress, Start::Complete),
    };
    log(&format!(
        "{} training and {} held-out sequences, {} draft-head rows{}",
        data.training().len(),
        data.held_out().len(),
        artifact.draft_rows,
        match start {
            Start::Initial => String::new(),
            Start::Continue | Start::Complete =>
                format!(", resumed at epoch {} step {}", progress.epoch(), progress.step()),
        }
    ));
    Ok(Loaded {
        lane,
        data,
        width,
        hidden,
        row_of_token: artifact.row_of_token,
        identity,
        progress,
        state,
        start,
        keep: Keep {
            _gpu: gpu,
            _pool: pool,
            _math: math,
            _library: library,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bf16_widens_to_the_float_with_the_same_top_half() {
        // 1.0f32 is 0x3f80_0000, -2.5f32 is 0xc020_0000.
        let bf16 = [0x80, 0x3f, 0x20, 0xc0, 0x00, 0x00];
        let floats: Vec<f32> = widen(&bf16)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect();
        assert_eq!(floats, [1.0, -2.5, 0.0]);
        assert!(widen(&[0x80]).is_empty());
    }

    #[test]
    fn the_artifact_layout_aligns_to_256_bytes() {
        assert_eq!([align(0), align(1), align(256), align(257)], [0, 256, 256, 512]);
    }
}
