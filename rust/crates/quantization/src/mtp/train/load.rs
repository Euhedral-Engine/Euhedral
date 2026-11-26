//! A run's startup: the device and its lane, the drafter's parameters (from the checkpoint, or from a saved state), the
//! draft head and the token embedding (from the artifact), the corpus, and the progress the run starts from.
//!
//! This runs on the calling thread before any frame of the run: it waits for each read and each copy, which a frame
//! never does. The run itself ([`run`](super::run)) is frames.

use std::fmt::Debug;
use std::fs;
use std::io::ErrorKind;
use std::mem::take;
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
use sha2::{Digest as _, Sha256};

use super::data::{Data, Files};
use super::error::RunError;
use super::lane::{CHUNK_BYTES, Lane, Resources, TrainLane};
use super::options::{Host, Log, Options};
use super::progress::{Found, Identity, Progress, Resume};
use super::run::{Ready, Start};
use super::settings::{HEAD_ROWS, MAX_POSITIONS, Settings, expected_steps};
use super::state::{File, place};
use super::wait::{call, done, then};
use super::window::Addresses;
use crate::assemble::ArtifactFile;
use crate::checkpoint::io::{open_read, read_exact, read_file};
use crate::checkpoint::objects::{Config as ModelConfig, Qwen38Objects};
use crate::checkpoint::safetensors::Safetensors;
use crate::corpus::{self, Entry};
use crate::kernels::{Module, ModuleOptions, Modules};
use crate::mtp::layout::{NORMS, PROJECTIONS};
use crate::mtp::{Config, Drafter, DrafterKernels, Shape, Work, plan_nvfp4_decode_bf16};
use crate::resume::{ResumeError, ResumeFile};

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

/// What stays open for the lane.
#[derive(Debug)]
pub struct Keep {
    _gpu: Arc<CudaGpu>,
    _pool: Arc<LanePool>,
    _math: Arc<MathLibraries>,
    _library: Arc<CudaLibrary>,
}

impl Keep {
    /// Keeps `gpu`, its lane pool, the math libraries and the engine library open for as long as it lives.
    pub fn new(gpu: Arc<CudaGpu>, pool: Arc<LanePool>, math: Arc<MathLibraries>, library: Arc<CudaLibrary>) -> Keep {
        Keep {
            _gpu: gpu,
            _pool: pool,
            _math: math,
            _library: library,
        }
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

/// A SHA-256 over parts, each prefixed with its length, so that where one part ends and the next begins matters.
struct Digest(Sha256);

impl Digest {
    fn new() -> Digest {
        Digest(Sha256::new())
    }

    fn part(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }

    fn hex(self) -> String {
        self.0.finalize().iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

/// The digest of a corpus's content: its index, its tokens and labels, and the size of its hidden rows, which the base
/// model computes from the tokens. Where the corpus lies is not in it.
fn corpus_digest(entries: &[Entry], tokens: &[u8], labels: &[u8], hidden_bytes: u64) -> String {
    let mut digest = Digest::new();
    for entry in entries {
        digest.part(entry.id.as_bytes());
        for number in [
            entry.token,
            entry.row,
            entry.length,
            entry.prompt,
            u64::from(entry.sampled),
        ] {
            digest.part(&number.to_le_bytes());
        }
    }
    digest.part(tokens);
    digest.part(labels);
    digest.part(&hidden_bytes.to_le_bytes());
    digest.hex()
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

/// The corpus's files in memory, checked, and the digest of its content.
fn open_corpus(
    options: &Options,
    host: &Host<'_>,
    width: u32,
    vocabulary: u64,
) -> Result<(Data, Arc<OwnedFile>, String), RunError> {
    let mut entries = corpus::entries(&options.corpus)?;
    if options.settings.limit > 0 {
        entries.truncate(options.settings.limit);
    }
    let file = |name: &str| call(|reply| read_file(host.io, &options.corpus.join(name), done(&reply)));
    let (token_bytes, label_bytes) = (file("tokens.bin")?, file("labels.bin")?);
    let path = options.corpus.join("hidden.bin");
    #[allow(
        clippy::disallowed_methods,
        reason = "startup: the size of the corpus file, read before any frame runs"
    )]
    let hidden_bytes = fs::metadata(&path)?.len();
    let hidden_rows = hidden_bytes / (2 * u64::from(width));
    let hidden = call(|reply| {
        open_read(
            host.io,
            &path,
            Box::new(move |opened| reply.send(opened.map(Arc::new).map_err(RunError::from))),
        )
    })?;
    let digest = corpus_digest(&entries, &token_bytes, &label_bytes, hidden_bytes);
    let data = Data::new(
        Files {
            entries,
            tokens: words(&token_bytes),
            labels: words(&label_bytes),
            hidden_rows,
            vocabulary,
        },
        &options.settings,
    )?;
    Ok((data, hidden, digest))
}

/// Everything the artifact gives the run.
struct Artifact {
    head: u64,
    draft_rows: u32,
    row_of_token: Vec<i32>,
    table: HostWeightsArena,
    scale_offset: u64,
    /// The digest of the head's token ids, the head and the embedding.
    digest: String,
}

/// Bytes filled per copy.
const FILL_CHUNK: usize = 8 << 20;

/// Fills a fresh device buffer with `byte`: the poison of a run that looks for a read of memory before its write. For
/// startup: it waits.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: a synchronous copy before any frame runs on the lane"
)]
fn fill(buffer: &DeviceBuffer, byte: u8) -> Result<(), RunError> {
    let filler = vec![byte; buffer.bytes().min(FILL_CHUNK)];
    let mut at = 0;
    while at < buffer.bytes() {
        let end = (at + filler.len()).min(buffer.bytes());
        buffer.copy_from_host(at, &filler[..end - at])?;
        at = end;
    }
    Ok(())
}

/// The device buffers a run owns and the addresses they give. A buffer holds whatever the allocator left unless
/// `poison` fills it: the kernels write what they read, and the poison is how a test finds one that does not.
struct Buffers {
    library: Arc<CudaLibrary>,
    owned: Vec<DeviceBuffer>,
    poison: Option<u8>,
}

impl Buffers {
    fn take(&mut self, bytes: u64) -> Result<u64, RunError> {
        let buffer = DeviceBuffer::alloc(&self.library, size(bytes.max(4), "a buffer")?)?;
        if let Some(byte) = self.poison {
            fill(&buffer, byte)?;
        }
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
    let id_bytes = read(host, file, ids.offset(), ids.bytes(), "the draft head's token ids")?;
    let ids = words(&id_bytes);
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
    let mut digest = Digest::new();
    for part in [&id_bytes, &packed, &bytes] {
        digest.part(part);
    }
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
            digest: digest.hex(),
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
    // The copy from pageable memory may return before the transfer lands, and the lane's stream does not wait for it:
    // without this the decode kernel can read the staging buffer before its bytes are there.
    library.synchronize()?;
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

/// The checkpoint's drafter tensors as BF16 bytes, by the name of the parameter each starts.
fn read_checkpoint(objects: &Qwen38Objects) -> Result<Vec<(&'static str, Vec<u8>)>, RunError> {
    let store = objects.store();
    let mut tensors = Vec::new();
    for name in PROJECTIONS {
        let source = objects.object(name)?;
        tensors.push((name, call(|reply| objects.read_bytes(&source, done(&reply)))?));
    }
    for (name, tensor) in NORMS.into_iter().zip(NORM_TENSORS) {
        let memory = vec![0u8; size(store.tensor(tensor)?.bytes, tensor)?];
        tensors.push((name, call(|reply| store.read_tensor(tensor, memory, done(&reply)))?));
    }
    Ok(tensors)
}

/// The digest of the drafter's starting parameters.
fn checkpoint_digest(tensors: &[(&'static str, Vec<u8>)]) -> String {
    let mut digest = Digest::new();
    for (name, bytes) in tensors {
        digest.part(name.as_bytes());
        digest.part(bytes);
    }
    digest.hex()
}

/// Writes the checkpoint's values, widened to FP32, to the lane's parameters.
fn load_checkpoint(tensors: &[(&'static str, Vec<u8>)], lane: &TrainLane) -> Result<(), RunError> {
    for parameter in lane.drafter().parameters() {
        let count = parameter.count();
        let (_, bf16) = tensors
            .iter()
            .find(|(name, _)| *name == parameter.name)
            .ok_or_else(|| RunError::Broken(format!("{} is not a parameter of the drafter", parameter.name)))?;
        if bf16.len() as u64 != 2 * count {
            return Err(input(format!(
                "{} holds {} bytes in the checkpoint, the drafter needs {}",
                parameter.name,
                bf16.len(),
                2 * count
            )));
        }
        lane.put(parameter.values, &widen(bf16))?;
    }
    Ok(())
}

/// Writes the saved state's pieces to the lane.
fn restore(lane: &TrainLane, state: &ResumeFile) -> Result<(), RunError> {
    for placed in place(lane.drafter().parameters(), File::State, CHUNK_BYTES) {
        let bytes = call(|reply| state.read(placed.piece.offset, placed.piece.bytes, then(&reply)))?;
        lane.put(placed.address, &bytes)?;
    }
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

/// Opens the saved state to read it; none when there is no file.
fn open_state(host: &Host<'_>, path: &Path) -> Result<Option<ResumeFile>, RunError> {
    match call(|reply| ResumeFile::open_read_only(host.io, host.lake, path, then(&reply))) {
        Ok(file) => Ok(Some(file)),
        Err(RunError::Resume(ResumeError::Io(error))) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

/// What the start brings the lane from, and where from.
struct Source<'a> {
    library: &'a Arc<CudaLibrary>,
    stored: &'a ArtifactBytes,
    draft_rows: u32,
    head: u64,
    checkpoint: &'a [(&'static str, Vec<u8>)],
    state: Option<&'a ResumeFile>,
}

/// Brings the lane to where `resume` says the run starts: the head decoded, the parameters from the checkpoint or from
/// the saved state, the scales measured. `fresh` is the progress of a run from the checkpoint.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the device is waited for before any frame runs on the lane"
)]
fn bring_up(
    lane: &mut TrainLane,
    resume: Resume,
    fresh: Progress,
    from: &Source<'_>,
) -> Result<(Progress, Start), RunError> {
    decode_head(lane, from.library, from.stored, from.draft_rows, from.head)?;
    match resume {
        Resume::Fresh => {
            load_checkpoint(from.checkpoint, lane)?;
            from.library.synchronize()?;
            lane.prepare(true)?;
            Ok((fresh, Start::Initial))
        }
        Resume::Continue(progress) => {
            let state = from
                .state
                .ok_or_else(|| RunError::Broken("a state to continue from is not open".to_owned()))?;
            // The drafter refuses steps until its zeroing was planned; the saved state then overwrites what it zeroed.
            lane.prepare(true)?;
            restore(lane, state)?;
            from.library.synchronize()?;
            lane.prepare(false)?;
            Ok((progress, Start::Continue))
        }
        Resume::Complete(progress) => Ok((progress, Start::Complete)),
    }
}

/// Opens everything the run needs and brings it to where it starts.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the calling thread is the binary's own, it waits for each read and copy before any frame runs"
)]
pub fn load(options: &Options, host: &Host<'_>, log: &Log) -> Result<Ready<TrainLane>, RunError> {
    let settings = options.settings;
    settings.check()?;
    // The libraries first: a wrong path is refused before the corpus and the artifact are read.
    let library = CudaLibrary::load(&options.library)?;
    let math = MathLibraries::load(&options.math)?;
    let store = Arc::new(call(|reply| Safetensors::open(host.io, &options.model, done(&reply)))?);
    let objects = call(|reply| Qwen38Objects::open(Arc::clone(&store), &options.model, done(&reply)))?;
    let config = config_of(&settings, &objects.config)?;
    config.validate()?;
    let (width, vocabulary) = (config.shape.hidden, objects.config.vocabulary);

    let (data, hidden, corpus) = open_corpus(options, host, width, vocabulary)?;
    let (stored, mut artifact) = read_artifact(options, host, &library, width, vocabulary)?;
    let checkpoint = read_checkpoint(&objects)?;
    let artifact_digest = take(&mut artifact.digest);
    let identity = Identity::new(
        corpus,
        checkpoint_digest(&checkpoint),
        artifact_digest,
        &settings,
        data.entries().len(),
    );
    let expected = expected_steps(data.lengths(), settings.tokens);

    // What was saved decides the start before the device is opened: a refusal costs nothing.
    fs::create_dir_all(&options.out)?;
    let (state_path, best_path) = (options.out.join("resume.bin"), options.out.join("best.bin"));
    if options.fresh {
        // A state of the run being replaced must not outlive the first best parameters of the new one.
        match fs::remove_file(&state_path) {
            Err(error) if error.kind() != ErrorKind::NotFound => return Err(error.into()),
            _ => {}
        }
    }
    let state = open_state(host, &state_path)?;
    let saved = match &state {
        Some(file) => Some(call(|reply| file.header(then(&reply)))?),
        None => None,
    };
    let resume = Progress::resume(
        settings,
        data.training().to_vec(),
        expected,
        &identity,
        saved.as_ref().map_or(Found::Missing, Found::Header),
        best_path.try_exists()?,
        options.fresh,
    )?;

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
    let handles = math.lane_handles().map_err(|failed| {
        if let Some(leftover) = failed.leftover {
            let _ = leftover.close();
        }
        RunError::from(failed.error)
    })?;

    // The start writes every parameter value and every scale; only the gradients and the Adam moments rely on the
    // zero that `Drafter::initialise` writes (a gradient adds into its array, a moment starts empty). Everything else is
    // as the allocator left it, or poison when asked for.
    let arena = DeviceBuffer::alloc(&library, size(Drafter::arena_bytes(&config)?, "the drafter's arena")?)?;
    if let Some(byte) = options.poison {
        fill(&arena, byte)?;
    }
    let drafter = Drafter::new(&config, arena.addr())?;
    let mut buffers = Buffers {
        library: Arc::clone(&library),
        owned: Vec::new(),
        poison: options.poison,
    };
    let (rows, depths) = (u64::from(settings.rows), u64::from(settings.depths));
    let wide = u64::from(width);
    artifact.head = buffers.take(2 * u64::from(artifact.draft_rows) * wide)?;
    let head_rows = u64::from(HEAD_ROWS);
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
        keep: Keep::new(gpu, pool, math, Arc::clone(&library)),
        poison: options.poison,
    };
    let mut lane = TrainLane::new(stream, &library, resources)?;
    // The fills above are copies from pageable memory, which may still be under way: the lane's stream does not wait.
    let source = Source {
        library: &library,
        stored: &stored,
        draft_rows: artifact.draft_rows,
        head,
        checkpoint: &checkpoint,
        state: state.as_ref(),
    };
    let fresh = Progress::new(settings, data.training().to_vec(), expected);
    let begun = library
        .synchronize()
        .map_err(RunError::from)
        .and_then(|()| bring_up(&mut lane, resume, fresh, &source));
    let (progress, start) = match begun {
        Ok(begun) => begun,
        Err(error) => {
            // Close the lane here, on the calling thread: its buffers are device memory.
            let _ = lane.close();
            return Err(error);
        }
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
    let parameters = lane.drafter().parameters();
    let (state_pieces, best_pieces) = (
        place(parameters, File::State, CHUNK_BYTES),
        place(parameters, File::Best, CHUNK_BYTES),
    );
    Ok(Ready {
        lane,
        data,
        width,
        hidden,
        row_of_token: artifact.row_of_token,
        identity,
        progress,
        start,
        state_pieces,
        best_pieces,
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

    fn entry(id: &str, token: u64) -> Entry {
        Entry {
            id: id.to_owned(),
            token,
            row: 0,
            length: 5,
            prompt: 2,
            sampled: false,
        }
    }

    #[test]
    fn a_digest_tells_where_one_part_ends_and_the_next_begins() {
        let of = |parts: &[&[u8]]| {
            let mut digest = Digest::new();
            parts.iter().for_each(|part| digest.part(part));
            digest.hex()
        };
        assert_eq!(of(&[b"ab", b"c"]), of(&[b"ab", b"c"]));
        assert_ne!(of(&[b"ab", b"c"]), of(&[b"a", b"bc"]));
        assert_eq!(of(&[]).len(), 64);
    }

    #[test]
    fn a_corpus_digest_changes_with_any_of_its_content_and_with_nothing_else() {
        let entries = [entry("a", 0), entry("b", 5)];
        let base = corpus_digest(&entries, &[1, 0, 0, 0], &[2, 0, 0, 0], 80);
        assert_eq!(base, corpus_digest(&entries, &[1, 0, 0, 0], &[2, 0, 0, 0], 80));
        let mut moved = entries.clone();
        moved[1].token = 6;
        let mut renamed = entries.clone();
        renamed[0].id = "c".to_owned();
        let mut sampled = entries.clone();
        sampled[0].sampled = true;
        for (name, other) in [
            (
                "an entry's start",
                corpus_digest(&moved, &[1, 0, 0, 0], &[2, 0, 0, 0], 80),
            ),
            (
                "an entry's id",
                corpus_digest(&renamed, &[1, 0, 0, 0], &[2, 0, 0, 0], 80),
            ),
            (
                "an entry's sampling",
                corpus_digest(&sampled, &[1, 0, 0, 0], &[2, 0, 0, 0], 80),
            ),
            ("a token", corpus_digest(&entries, &[3, 0, 0, 0], &[2, 0, 0, 0], 80)),
            ("a label", corpus_digest(&entries, &[1, 0, 0, 0], &[3, 0, 0, 0], 80)),
            (
                "the hidden rows",
                corpus_digest(&entries, &[1, 0, 0, 0], &[2, 0, 0, 0], 96),
            ),
            (
                "an entry fewer",
                corpus_digest(&entries[..1], &[1, 0, 0, 0], &[2, 0, 0, 0], 80),
            ),
        ] {
            assert_ne!(base, other, "{name}");
        }
    }

    #[test]
    fn the_artifact_layout_aligns_to_256_bytes() {
        assert_eq!([align(0), align(1), align(256), align(257)], [0, 256, 256, 512]);
    }
}
