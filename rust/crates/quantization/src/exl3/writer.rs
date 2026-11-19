//! The writer of an EXL3 checkpoint: it takes the quantized linears of one decoder layer, or of the MTP layer, as they
//! are made, and writes the files of the layout around them. Every file is written as `<name>.part`, made durable and
//! renamed, so a stop leaves whole files and no partial one.
//!
//! The writer is a value that moves with the work: each call takes it and gives it back, with the result, to the
//! callback. A callback runs once, from a frame of the lattice that drives the `AsyncIo` or, if that frame is dropped
//! unrun, from wherever it is dropped, and must not wait. A call that cannot be done (a refusal, or a file that could
//! not be written) gives the writer back with the error and with its record of the files written unchanged, so the
//! call can be made again; the one thing it does not give back is the results a [`Exl3Writer::layer`] call took. A call
//! that is dropped while it is in flight (a step that was accepted but never ran, because a lake closed under it) ends
//! with [`CheckpointError::LakeClosed`] and the writer, so the callback runs once in every case.
//!
//! The directory of the output must exist. `AsyncIo` cannot make or list directories, so the caller makes the output
//! directory and lists the source and output directories (as a [`Listing`]) when it calls [`Exl3Writer::finish`].
//!
//! Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;
use serde_json::Value;

use super::layout::{
    Ledger, MTP_BITS, Planned, Quantized, Stored, TEXT, auxiliary_files, bf16_to_half, pretty, quantization_config,
    quantized_names, quantized_tensors,
};
use super::linears::{self, MTP_LINEARS};
use crate::checkpoint::objects::Qwen38Objects;
use crate::checkpoint::safetensors::Safetensors;
use crate::checkpoint::writer::{DEFAULT_BUFFER_BYTES, SafetensorsWriter, Source, bytes_source, copy_file, write_file};
use crate::checkpoint::{CheckpointError, Done as ReadDone, read_file};

/// What a call of the writer ends with: the writer, and whether the call worked.
pub type Done = EndOf<Exl3Writer>;

/// The names of the files in the two directories, which the caller lists.
#[derive(Clone, Debug, Default)]
pub struct Listing {
    /// The regular files of the source checkpoint (not directories): the auxiliary ones are copied, and `config.json`
    /// must be among them.
    pub source: Vec<String>,
    /// The files of the output directory, listed after the last layer or MTP write: an earlier run's layer files are
    /// adopted, a safetensors file that nothing here accounts for is refused, and every file this run wrote must be
    /// in it.
    pub out: Vec<String>,
}

/// What the written checkpoint declares and how it is written.
#[derive(Clone, Copy, Debug)]
pub struct Exl3Settings {
    /// The average bits per weight of the text linears.
    pub bits: f64,
    /// The calibration shape: rows and columns.
    pub rows: u64,
    pub columns: u64,
    /// The buffer each file is copied through: a multiple of 8.
    pub buffer_bytes: usize,
}

impl Exl3Settings {
    /// Settings with the default copy buffer.
    pub fn new(bits: f64, rows: u64, columns: u64) -> Self {
        Self {
            bits,
            rows,
            columns,
            buffer_bytes: DEFAULT_BUFFER_BYTES,
        }
    }
}

/// How a tensor of a file to write gets its bytes.
enum Content {
    Data(Vec<u8>),
    Source(Source),
}

struct Pending {
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    content: Content,
}

/// The tensors of one output file, which is written sorted by name.
struct Shard {
    file: String,
    /// Whether this is the MTP file.
    mtp: bool,
    tensors: BTreeMap<String, Pending>,
}

impl Shard {
    fn new(file: String, mtp: bool) -> Self {
        Self {
            file,
            mtp,
            tensors: BTreeMap::new(),
        }
    }

    fn insert(&mut self, name: String, pending: Pending) -> Result<(), CheckpointError> {
        if self.tensors.contains_key(&name) {
            return Err(CheckpointError::Invalid(format!("tensor {name} written twice")));
        }
        self.tensors.insert(name, pending);
        Ok(())
    }

    fn data(&mut self, planned: Planned) -> Result<(), CheckpointError> {
        let pending = Pending {
            dtype: planned.dtype.to_owned(),
            shape: planned.shape,
            bytes: planned.data.len() as u64,
            content: Content::Data(planned.data),
        };
        self.insert(planned.name, pending)
    }

    /// Copies source tensor `name` unchanged.
    fn copy(&mut self, store: &Arc<Safetensors>, name: &str) -> Result<(), CheckpointError> {
        let tensor = store.tensor(name)?;
        let pending = Pending {
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
            bytes: tensor.bytes,
            content: Content::Source(store.source(name)?),
        };
        self.insert(name.to_owned(), pending)
    }

    /// Copies source tensor `name` as F16 (round to nearest even); a value that overflows F16 fails the write.
    fn copy_as_half(&mut self, store: &Arc<Safetensors>, name: &str) -> Result<(), CheckpointError> {
        let tensor = store.tensor(name)?;
        match tensor.dtype.as_str() {
            "F16" => return self.copy(store, name),
            "BF16" => {}
            other => {
                return Err(CheckpointError::Invalid(format!("{name} is {other}, not BF16 or F16")));
            }
        }
        let inner = store.source(name)?;
        let named = name.to_owned();
        let source: Source = Arc::new(move |offset, chunk, done| {
            let named = named.clone();
            inner(
                offset,
                chunk,
                Arc::new(move |read| {
                    done(read.and_then(|mut bytes| {
                        bf16_to_half(&named, &mut bytes)?;
                        Ok(bytes)
                    }));
                }),
            )
        });
        let pending = Pending {
            dtype: "F16".to_owned(),
            shape: tensor.shape.clone(),
            bytes: tensor.bytes,
            content: Content::Source(source),
        };
        self.insert(name.to_owned(), pending)
    }
}

/// One step of the end of a run, taken in order.
enum Job {
    Shard(Shard),
    /// Copies a file of the source directory.
    Copy(String),
    /// Writes a file of the given bytes.
    Bytes(String, Vec<u8>),
    /// The index and the two configs, from the source's `config.json`, once every other file is written.
    Documents(Value),
}

/// A callback that runs `step` with the writer and the result, once.
fn once(step: impl FnOnce(Exl3Writer, Result<(), CheckpointError>) + Send + 'static) -> Done {
    let cell = Handoff::new();
    cell.send(step);
    Arc::new(move |writer, result| {
        if let Some(step) = cell.take() {
            step(writer, result);
        }
    })
}

/// A callback that goes on with `next` when the call worked and passes the failure to `then` when it did not.
fn after(then: Done, next: impl FnOnce(Exl3Writer, Done) + Send + 'static) -> Done {
    once(move |writer, result| match result {
        Ok(()) => next(writer, then),
        Err(error) => then(writer, Err(error)),
    })
}

/// What a call ends with, for a writer of type `W`.
type EndOf<W> = Arc<dyn Fn(W, Result<(), CheckpointError>) + Send + Sync>;

/// The writer and the callback of a call, parked while an asynchronous step holds the call. The step takes them; if the
/// step is dropped without running, the last holder's drop ends the call with [`CheckpointError::LakeClosed`], so the
/// callback always runs. The caller of a step keeps one holder until the step is accepted, so a step that is refused
/// reports its own error first.
struct Parked<W, E> {
    cell: Arc<Handoff<(W, EndOf<W>, E)>>,
}

impl<W: Send + 'static, E: Send + 'static> Parked<W, E> {
    fn new(writer: W, then: EndOf<W>, extra: E) -> Arc<Self> {
        let cell = Handoff::new();
        cell.send((writer, then, extra));
        Arc::new(Self { cell })
    }

    /// The call, for the step that runs. It is still the guard's until the step ends it ([`Taken::end`]) or passes it
    /// on ([`Taken::go`]): a step that panics in between ends the call with an error as it unwinds.
    fn take(&self) -> Option<Taken<W, E>> {
        self.cell.take().map(|(writer, then, extra)| Taken {
            parts: Some((writer, then, extra)),
        })
    }

    /// Ends the call with the error of a step that was refused, unless the step took the call.
    fn refused(&self, error: CheckpointError) {
        if let Some((writer, then, _)) = self.cell.take() {
            then(writer, Err(error));
        }
    }
}

/// The call while a step runs it: dropped before it is ended or passed on, it ends the call with an error.
struct Taken<W, E> {
    parts: Option<(W, EndOf<W>, E)>,
}

impl<W, E> Taken<W, E> {
    /// The writer and what the step parked with it, to work on while the call is still held.
    fn parts(&mut self) -> (&mut W, &E) {
        let (writer, _, extra) = self.parts.as_mut().expect("a call that is held has its parts");
        (writer, &*extra)
    }

    /// Ends the call with `result`.
    fn end(mut self, result: Result<(), CheckpointError>) {
        if let Some((writer, then, _)) = self.parts.take() {
            then(writer, result);
        }
    }

    /// Hands the call over to the next step, which must end it or park it.
    fn go(mut self) -> (W, EndOf<W>, E) {
        self.parts.take().expect("a call that is held has its parts")
    }
}

impl<W, E> Drop for Taken<W, E> {
    fn drop(&mut self) {
        if let Some((writer, then, _)) = self.parts.take() {
            then(
                writer,
                Err(CheckpointError::Invalid(
                    "a step of the call failed before it ended".to_owned(),
                )),
            );
        }
    }
}

impl<W, E> Drop for Parked<W, E> {
    fn drop(&mut self) {
        if let Some((writer, then, _)) = self.cell.take() {
            then(writer, Err(CheckpointError::LakeClosed));
        }
    }
}

/// Writes a Qwen 3.8 checkpoint in exllamav3's EXL3 layout from quantized results and the BF16 source.
///
/// - One safetensors file per decoder layer (`model.language_model.layers.<i>.safetensors`), `mtp.safetensors`,
///   `model.language_model.embed_tokens.safetensors`, `lm_head.safetensors` (with the final norm) and
///   `model.visual.safetensors`, plus a standard `model.safetensors.index.json` over all of them. Tensors inside a
///   file are sorted by name, as exllamav3 writes them.
/// - A quantized linear is `.trellis` (I16 `[k/16, n/16, 16 * bits]`), `.suh` (F16 `[k]`), `.svh` (F16 `[n]`) and
///   `.mul1` (I32 scalar, the codebook multiplier). Linears that are not quantized are stored as F16 (the GDN
///   `in_proj_a`/`in_proj_b`, `lm_head`); everything else, vision tower included, keeps its source dtype.
/// - `config.json` is the source config with the `quantization_config` block; `quantization_config.json` adds
///   `tensor_storage` for the text modules, as exllamav3's `create_quantization_config_json` does.
///
/// Only one layer's results are held at a time and source tensors are streamed through a fixed chunk buffer.
pub struct Exl3Writer {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    objects: Qwen38Objects,
    source: PathBuf,
    out: PathBuf,
    settings: Exl3Settings,
    ledger: Ledger,
    mtp_written: bool,
}

impl fmt::Debug for Exl3Writer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Exl3Writer")
            .field("out", &self.out)
            .field("settings", &self.settings)
            .field("mtp_written", &self.mtp_written)
            .finish_non_exhaustive()
    }
}

impl Exl3Writer {
    /// A writer of the checkpoint of `objects` (read from `source`) into the existing directory `out`. The host lake
    /// runs the durability calls of each file.
    pub fn new(
        io: Arc<AsyncIo>,
        host: Arc<Lake>,
        objects: Qwen38Objects,
        source: &Path,
        out: &Path,
        settings: Exl3Settings,
    ) -> Self {
        Self {
            io,
            host,
            objects,
            source: source.to_path_buf(),
            out: out.to_path_buf(),
            settings,
            ledger: Ledger::default(),
            mtp_written: false,
        }
    }

    fn store(&self) -> &Arc<Safetensors> {
        self.objects.store()
    }

    /// The names of the source tensors, sorted.
    fn sorted_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.store().names().map(str::to_owned).collect();
        names.sort();
        names
    }

    /// The `quantization_config` block as the files written now declare it.
    pub fn quantization_config(&self) -> Value {
        let settings = &self.settings;
        quantization_config(settings.bits, settings.rows, settings.columns, self.mtp_written)
    }

    /// `tensor_storage` for the text modules written so far.
    pub fn tensor_storage(&self) -> Value {
        self.ledger.tensor_storage(self.objects.config.layers)
    }

    // ---- layers ----

    /// Writes decoder layer `index`: the quantized linears (keyed by tensor name without `.weight`, as
    /// [`linears::layer`] names them, with their bits in `bits_by_linear`), the unquantized linears as F16 and every
    /// other tensor of the layer as in the source. The results are taken, and each is dropped once its tensors are
    /// made, so a layer's results are not in memory twice.
    /// They are spent on a call that fails, whether it was refused or the file could not be written: the caller
    /// quantizes the layer again to repeat the call.
    pub fn layer(
        self,
        index: usize,
        quantized: HashMap<String, Quantized>,
        bits_by_linear: &HashMap<String, f64>,
        then: Done,
    ) {
        match self.layer_shard(index, quantized, bits_by_linear) {
            Ok(shard) => self.write_shard(shard, then),
            Err(error) => then(self, Err(error)),
        }
    }

    fn layer_shard(
        &self,
        index: usize,
        mut quantized: HashMap<String, Quantized>,
        bits_by_linear: &HashMap<String, f64>,
    ) -> Result<Shard, CheckpointError> {
        let prefix = format!("{TEXT}layers.{index}.");
        let mut shard = Shard::new(format!("{TEXT}layers.{index}.safetensors"), false);
        let mut weights = BTreeSet::new();
        for linear in linears::layer(&self.objects, index)? {
            let weight = format!("{}.weight", linear.tensor);
            if linear.quantized {
                let (Some(result), Some(bits)) = (quantized.remove(&linear.tensor), bits_by_linear.get(&linear.tensor))
                else {
                    return Err(CheckpointError::Invalid(format!(
                        "no result or bits for {}",
                        linear.tensor
                    )));
                };
                for planned in quantized_tensors(&linear.tensor, linear.rows, linear.columns, *bits, &result)? {
                    shard.data(planned)?;
                }
            } else {
                shard.copy_as_half(self.store(), &weight)?;
            }
            weights.insert(weight);
        }
        let mut strays: Vec<&String> = quantized.keys().collect();
        strays.sort();
        if let Some(stray) = strays.first() {
            return Err(CheckpointError::Invalid(format!(
                "{stray} is not a quantized linear of layer {index}"
            )));
        }
        for name in self.sorted_names() {
            if name.starts_with(&prefix) && !weights.contains(&name) {
                shard.copy(self.store(), &name)?;
            }
        }
        Ok(shard)
    }

    /// Writes `mtp.safetensors`: the eight MTP linears (keyed by tensor name without `.weight`, 4 bits each) and the
    /// remaining `mtp.*` tensors as in the source.
    pub fn mtp(self, quantized: &HashMap<String, Quantized>, then: Done) {
        match self.mtp_shard(quantized) {
            Ok(shard) => self.write_shard(shard, then),
            Err(error) => then(self, Err(error)),
        }
    }

    fn mtp_shard(&self, quantized: &HashMap<String, Quantized>) -> Result<Shard, CheckpointError> {
        let expected: BTreeSet<&str> = MTP_LINEARS.into_iter().collect();
        let given: BTreeSet<&str> = quantized.keys().map(String::as_str).collect();
        if given != expected {
            return Err(CheckpointError::Invalid(format!(
                "mtp linears are {MTP_LINEARS:?}, got {given:?}"
            )));
        }
        let mut shard = Shard::new("mtp.safetensors".to_owned(), true);
        for linear in MTP_LINEARS {
            let weight = self.store().tensor(&format!("{linear}.weight"))?;
            let [n, k] = weight.shape[..] else {
                return Err(CheckpointError::Invalid(format!("{linear} is not a matrix")));
            };
            for planned in quantized_tensors(linear, n, k, f64::from(MTP_BITS), &quantized[linear])? {
                shard.data(planned)?;
            }
        }
        for name in self.sorted_names() {
            let is_linear = name
                .strip_suffix(".weight")
                .is_some_and(|stem| MTP_LINEARS.contains(&stem));
            if name.starts_with("mtp.") && !is_linear {
                shard.copy(self.store(), &name)?;
            }
        }
        Ok(shard)
    }

    /// Writes one file; on success the ledger takes its tensors.
    fn write_shard(self, shard: Shard, then: Done) {
        let Shard { file, mtp, tensors } = shard;
        let path = self.out.join(&file);
        let mut writer = match SafetensorsWriter::with_buffer(&path, self.settings.buffer_bytes) {
            Ok(writer) => writer,
            Err(error) => return then(self, Err(error)),
        };
        let mut listed = Vec::with_capacity(tensors.len());
        for (name, pending) in tensors {
            let Pending {
                dtype,
                shape,
                bytes,
                content,
            } = pending;
            let added = match content {
                Content::Data(data) => writer.add(&name, &dtype, &shape, data),
                Content::Source(source) => writer.add_source(&name, &dtype, &shape, bytes, source),
            };
            if let Err(error) = added {
                return then(self, Err(error));
            }
            listed.push((name, Stored { dtype, shape, bytes }));
        }
        let (io, host) = (Arc::clone(&self.io), Arc::clone(&self.host));
        let parked = Parked::new(self, then, (file, listed, mtp));
        let waiting = Arc::clone(&parked);
        let started = writer.finish(
            &io,
            &host,
            Arc::new(move |result| {
                let Some(mut taken) = waiting.take() else {
                    return;
                };
                if result.is_ok() {
                    let (writer, (file, listed, mtp)) = taken.parts();
                    writer.ledger.record_file(file, listed);
                    writer.mtp_written |= *mtp;
                }
                taken.end(result);
            }),
        );
        if let Err(error) = started {
            parked.refused(error);
        }
    }

    // ---- the end ----

    /// Writes the embedding, the final norm, `lm_head` (F16), the vision tower, the auxiliary files, the index and
    /// both configs, once every source tensor is accounted for. Everything is checked before the large files are
    /// written: the layer and MTP files an earlier run left are adopted (each must hold exactly the tensors this
    /// writer would write), every source tensor must be written or about to be, the rates must match
    /// `bits_by_linear`, no safetensors file may be in the output that the index would not list, `listing.source`
    /// must hold `config.json` and `listing.out` every file this run wrote.
    ///
    /// A refusal writes nothing. A failure while writing leaves the files written so far in the output and gives the
    /// writer back; `finish` can be called again with a new listing of the output, and rewrites those files.
    pub fn finish(self, bits_by_linear: &HashMap<String, f64>, listing: Listing, then: Done) {
        // The listing is checked before an earlier file is adopted, so a refusal leaves the record as it was.
        if let Err(error) = self.check_listing(&listing) {
            return then(self, Err(error));
        }
        let earlier = match self.earlier_files(&listing.out) {
            Ok(earlier) => earlier,
            Err(error) => return then(self, Err(error)),
        };
        if earlier.is_empty() {
            return self.checked(bits_by_linear.clone(), listing, then);
        }
        let names: Vec<String> = earlier.iter().map(|(file, _)| file.clone()).collect();
        let (io, out) = (Arc::clone(&self.io), self.out.clone());
        let parked = Parked::new(self, then, (earlier, bits_by_linear.clone(), listing));
        let waiting = Arc::clone(&parked);
        let opened: ReadDone<Safetensors> = Arc::new(move |opened| {
            let Some(mut taken) = waiting.take() else {
                return;
            };
            let (writer, (earlier, ..)) = taken.parts();
            match opened.and_then(|found| writer.adopt(&found, earlier)) {
                Ok(()) => {
                    let (writer, then, (_, bits, listing)) = taken.go();
                    writer.checked(bits, listing, then);
                }
                Err(error) => taken.end(Err(error)),
            }
        });
        if let Err(error) = Safetensors::open_files(&io, &out, names, opened) {
            parked.refused(error);
        }
    }

    fn check_listing(&self, listing: &Listing) -> Result<(), CheckpointError> {
        if !listing.source.iter().any(|file| file == "config.json") {
            return Err(CheckpointError::Invalid(
                "the listing of the source has no config.json".to_owned(),
            ));
        }
        let absent: Vec<&str> = self
            .ledger
            .files()
            .filter(|file| !listing.out.iter().any(|listed| listed == file))
            .collect();
        if !absent.is_empty() {
            return Err(CheckpointError::Invalid(format!(
                "the listing of the output is stale: it lacks {absent:?}, which this run wrote"
            )));
        }
        Ok(())
    }

    /// The files an earlier run left in the output that this writer has not written, with the tensors each must hold.
    fn earlier_files(&self, present: &[String]) -> Result<Vec<(String, BTreeSet<String>)>, CheckpointError> {
        let mut earlier = Vec::new();
        for index in 0..self.objects.config.layers {
            let file = format!("{TEXT}layers.{index}.safetensors");
            if !self.ledger.has_file(&file) && present.contains(&file) {
                earlier.push((file, self.expected_layer_tensors(index)?));
            }
        }
        let file = "mtp.safetensors".to_owned();
        if !self.ledger.has_file(&file) && present.contains(&file) {
            earlier.push((file, self.expected_mtp_tensors()));
        }
        Ok(earlier)
    }

    fn expected_layer_tensors(&self, index: usize) -> Result<BTreeSet<String>, CheckpointError> {
        let prefix = format!("{TEXT}layers.{index}.");
        let mut expected = BTreeSet::new();
        let mut weights = BTreeSet::new();
        for linear in linears::layer(&self.objects, index)? {
            weights.insert(format!("{}.weight", linear.tensor));
            if linear.quantized {
                expected.extend(quantized_names(&linear.tensor));
            } else {
                expected.insert(format!("{}.weight", linear.tensor));
            }
        }
        for name in self.store().names() {
            if name.starts_with(&prefix) && !weights.contains(name) {
                expected.insert(name.to_owned());
            }
        }
        Ok(expected)
    }

    fn expected_mtp_tensors(&self) -> BTreeSet<String> {
        let mut expected = BTreeSet::new();
        for linear in MTP_LINEARS {
            expected.extend(quantized_names(linear));
        }
        for name in self.store().names() {
            let is_linear = name
                .strip_suffix(".weight")
                .is_some_and(|stem| MTP_LINEARS.contains(&stem));
            if name.starts_with("mtp.") && !is_linear {
                expected.insert(name.to_owned());
            }
        }
        expected
    }

    /// Takes the earlier files' tensors (read from their headers) into the ledger.
    fn adopt(&mut self, found: &Safetensors, earlier: &[(String, BTreeSet<String>)]) -> Result<(), CheckpointError> {
        let mut held: HashMap<&str, BTreeMap<String, Stored>> = HashMap::new();
        for name in found.names() {
            let tensor = found.tensor(name)?;
            held.entry(found.shard_of(tensor)).or_default().insert(
                name.to_owned(),
                Stored {
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    bytes: tensor.bytes,
                },
            );
        }
        for (file, expected) in earlier {
            let tensors = held.remove(file.as_str()).unwrap_or_default();
            let names: BTreeSet<String> = tensors.keys().cloned().collect();
            if &names != expected {
                let absent: Vec<&String> = expected.difference(&names).collect();
                let extra: Vec<&String> = names.difference(expected).collect();
                return Err(CheckpointError::Invalid(format!(
                    "{file} from an earlier run does not hold the expected tensors; missing {absent:?}, unexpected {extra:?}"
                )));
            }
            let listed: Vec<(String, Stored)> = tensors.into_iter().collect();
            self.ledger.record_file(file, &listed);
            self.mtp_written |= file == "mtp.safetensors";
        }
        Ok(())
    }

    /// The checks that come before any large file, then the writes.
    fn checked(self, bits_by_linear: HashMap<String, f64>, listing: Listing, then: Done) {
        let jobs = match self.plan_end(&bits_by_linear, &listing) {
            Ok(jobs) => jobs,
            Err(error) => return then(self, Err(error)),
        };
        // The source config is read before anything is written, so a source that is already quantized is refused
        // early.
        let (io, path) = (Arc::clone(&self.io), self.source.join("config.json"));
        let parked = Parked::new(self, then, jobs);
        let waiting = Arc::clone(&parked);
        let read: ReadDone<Vec<u8>> = Arc::new(move |bytes| {
            let Some(taken) = waiting.take() else {
                return;
            };
            match bytes.and_then(|bytes| source_config(&bytes)) {
                Ok(config) => {
                    let (writer, then, mut jobs) = taken.go();
                    jobs.push_back(Job::Documents(config));
                    writer.run(jobs, then);
                }
                Err(error) => taken.end(Err(error)),
            }
        });
        if let Err(error) = read_file(&io, &path, read) {
            parked.refused(error);
        }
    }

    /// Checks everything that can be checked and lists the files to write before the documents.
    fn plan_end(
        &self,
        bits_by_linear: &HashMap<String, f64>,
        listing: &Listing,
    ) -> Result<VecDeque<Job>, CheckpointError> {
        let names = self.sorted_names();
        let head: BTreeSet<String> = [
            format!("{TEXT}embed_tokens.weight"),
            "lm_head.weight".to_owned(),
            format!("{TEXT}norm.weight"),
        ]
        .into();
        let vision: Vec<&String> = names.iter().filter(|name| name.starts_with("model.visual.")).collect();
        let missing: Vec<&String> = names
            .iter()
            .filter(|name| {
                let linear = name.strip_suffix(".weight").unwrap_or(name);
                !self.ledger.has_tensor(name)
                    && !self.ledger.has_tensor(&format!("{linear}.trellis"))
                    && !head.contains(*name)
                    && !vision.contains(name)
            })
            .collect();
        if !missing.is_empty() {
            return Err(CheckpointError::Invalid(format!(
                "{} source tensors were not written, e.g. {:?}",
                missing.len(),
                &missing[..missing.len().min(5)]
            )));
        }
        let mut linears: Vec<&String> = bits_by_linear.keys().collect();
        linears.sort();
        for linear in linears {
            let bits = bits_by_linear[linear];
            if let Some(words) = self.ledger.trellis_words(linear)
                && words as i64 != (bits * 16.0).round() as i64
            {
                return Err(CheckpointError::Invalid(format!(
                    "{linear} is stored at a different rate than {bits}"
                )));
            }
        }
        let embedding = format!("{TEXT}embed_tokens.safetensors");
        let mut expected: BTreeSet<&str> = self.ledger.files().collect();
        expected.insert(&embedding);
        expected.insert("lm_head.safetensors");
        if !vision.is_empty() {
            expected.insert("model.visual.safetensors");
        }
        let mut stray: Vec<&String> = listing
            .out
            .iter()
            .filter(|file| file.ends_with(".safetensors") && !expected.contains(file.as_str()))
            .collect();
        stray.sort();
        if !stray.is_empty() {
            return Err(CheckpointError::Invalid(format!(
                "unexpected safetensors files in the output (a loader reads them all): {stray:?}"
            )));
        }

        let mut jobs = VecDeque::new();
        let mut shard = Shard::new(embedding, false);
        shard.copy(self.store(), &format!("{TEXT}embed_tokens.weight"))?;
        jobs.push_back(Job::Shard(shard));
        let mut shard = Shard::new("lm_head.safetensors".to_owned(), false);
        shard.copy_as_half(self.store(), "lm_head.weight")?;
        shard.copy(self.store(), &format!("{TEXT}norm.weight"))?;
        jobs.push_back(Job::Shard(shard));
        if !vision.is_empty() {
            let mut shard = Shard::new("model.visual.safetensors".to_owned(), false);
            for name in vision {
                shard.copy(self.store(), name)?;
            }
            jobs.push_back(Job::Shard(shard));
        }
        for file in auxiliary_files(listing.source.iter().map(String::as_str)) {
            jobs.push_back(Job::Copy(file));
        }
        Ok(jobs)
    }

    /// Does the jobs in order, then calls `then`.
    fn run(self, mut jobs: VecDeque<Job>, then: Done) {
        let Some(job) = jobs.pop_front() else {
            return then(self, Ok(()));
        };
        match job {
            Job::Shard(shard) => self.write_shard(shard, after(then, move |writer, then| writer.run(jobs, then))),
            Job::Bytes(name, bytes) => {
                self.write_bytes(name, bytes, after(then, move |writer, then| writer.run(jobs, then)))
            }
            Job::Copy(name) => {
                let (io, host) = (Arc::clone(&self.io), Arc::clone(&self.host));
                let (from, to) = (self.source.join(&name), self.out.join(&name));
                let buffer = self.settings.buffer_bytes;
                let parked = Parked::new(self, then, jobs);
                let waiting = Arc::clone(&parked);
                let copied = copy_file(
                    &from,
                    &to,
                    buffer,
                    &io,
                    &host,
                    Arc::new(move |result| {
                        let Some(taken) = waiting.take() else {
                            return;
                        };
                        match result {
                            Ok(()) => {
                                let (writer, then, jobs) = taken.go();
                                writer.run(jobs, then);
                            }
                            Err(error) => taken.end(Err(error)),
                        }
                    }),
                );
                if let Err(error) = copied {
                    parked.refused(error);
                }
            }
            Job::Documents(config) => match self.documents(config) {
                Ok(files) => {
                    for (name, bytes) in files.into_iter().rev() {
                        jobs.push_front(Job::Bytes(name, bytes));
                    }
                    self.run(jobs, then);
                }
                Err(error) => then(self, Err(error)),
            },
        }
    }

    /// Writes `bytes` as the file `name` of the output.
    fn write_bytes(self, name: String, bytes: Vec<u8>, then: Done) {
        let (io, host, path) = (Arc::clone(&self.io), Arc::clone(&self.host), self.out.join(&name));
        let (size, buffer) = (bytes.len() as u64, self.settings.buffer_bytes);
        let parked = Parked::new(self, then, ());
        let waiting = Arc::clone(&parked);
        let started = write_file(
            &path,
            size,
            bytes_source(bytes),
            buffer,
            &io,
            &host,
            Arc::new(move |result| {
                if let Some(taken) = waiting.take() {
                    taken.end(result);
                }
            }),
        );
        if let Err(error) = started {
            parked.refused(error);
        }
    }

    /// The index, `config.json` and `quantization_config.json`, as bytes.
    fn documents(&self, config: Value) -> Result<Vec<(String, Vec<u8>)>, CheckpointError> {
        let Value::Object(mut config) = config else {
            return Err(CheckpointError::Format("config.json is not an object".to_owned()));
        };
        if config.contains_key("quantization_config") {
            return Err(CheckpointError::Invalid(
                "the source checkpoint is already quantized".to_owned(),
            ));
        }
        let quantization = self.quantization_config();
        config.insert("tie_word_embeddings".to_owned(), Value::Bool(false));
        config.insert("quantization_config".to_owned(), quantization.clone());
        let mut full = quantization;
        if let Value::Object(members) = &mut full {
            members.insert("tensor_storage".to_owned(), self.tensor_storage());
        }
        Ok(vec![
            (
                "model.safetensors.index.json".to_owned(),
                pretty(&self.ledger.index()).into_bytes(),
            ),
            ("config.json".to_owned(), pretty(&Value::Object(config)).into_bytes()),
            ("quantization_config.json".to_owned(), pretty(&full).into_bytes()),
        ])
    }
}

/// The source's `config.json`: an object that is not quantized yet.
fn source_config(bytes: &[u8]) -> Result<Value, CheckpointError> {
    let config: Value = serde_json::from_slice(bytes)
        .map_err(|error| CheckpointError::Format(format!("config.json is not JSON: {error}")))?;
    match &config {
        Value::Object(members) if members.contains_key("quantization_config") => Err(CheckpointError::Invalid(
            "the source checkpoint is already quantized".to_owned(),
        )),
        Value::Object(_) => Ok(config),
        _ => Err(CheckpointError::Format("config.json is not an object".to_owned())),
    }
}

#[cfg(test)]
mod tests;
