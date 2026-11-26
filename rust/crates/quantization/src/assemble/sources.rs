//! The calibrated artifacts a mixed artifact is assembled from, one per option, and the base whose objects an
//! allocation does not name.
//!
//! The choosable objects are the quantized linear objects of `text/layers/*` ([`SourceSet::projections`]); the
//! quantized linears of the MTP layer are listed apart ([`SourceSet::drafter_projections`]). Every other object comes
//! from the base. The objects derived directly from the checkpoint (norms, GDN state tensors and convolution, the
//! small BF16 GDN projections, the draft head's token ids) must be bit for bit the same in every source, which
//! [`SourceSet::open`] verifies by the hash of their payload. The quantized objects outside the layers (the token
//! embedding, the output and draft heads, the MTP layer) differ between sources by construction, each source having
//! been calibrated on its own, and are not compared.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::mem::take;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_inference_edrl::{TensorEntry, TensorTable, WeightFormat};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, OwnedFile};
use sha2::{Digest, Sha256};

use super::chunks::{Chunk, read_chunk};
use super::{ArtifactFile, AssembleError, Done};
use crate::mixed::{MixedError, QuantOption, Shape, Sources, shape_of};

/// The suffixes of an object's companions, which travel with it: trellis sign vectors and segment ends.
pub const COMPANION_SUFFIXES: [&str; 3] = [".suh", ".svh", ".segments"];

const LAYERS: &str = "text/layers/";
const DRAFTER_LAYER: &str = "mtp/layer/";

/// The source artifacts, opened and checked against each other.
#[derive(Debug)]
pub struct SourceSet {
    files: BTreeMap<QuantOption, ArtifactFile>,
    base: QuantOption,
    projections: Vec<String>,
    drafter_projections: Vec<String>,
}

impl SourceSet {
    /// Opens `artifacts` (an artifact file for each option that can be chosen) and checks them against each other,
    /// hashing the shared objects in reads of `hash_chunk_bytes`. The base's tokenizer section is read too.
    ///
    /// `done` gets an error naming the object when a source lacks one the base has among the projections or the
    /// objects compared, when their shapes or formats differ where they must agree, or when a compared object's bytes
    /// differ. `Err` started nothing.
    pub fn open(
        io: &Arc<AsyncIo>,
        artifacts: &BTreeMap<QuantOption, PathBuf>,
        base: QuantOption,
        hash_chunk_bytes: usize,
        done: Done<SourceSet>,
    ) -> Result<(), AssembleError> {
        if !artifacts.contains_key(&base) {
            return Err(AssembleError::invalid(format!("no source for the base {}", base.key())));
        }
        if hash_chunk_bytes == 0 {
            return Err(AssembleError::invalid("the hash read size must be at least one byte"));
        }
        Opening {
            io: Arc::clone(io),
            pending: artifacts.iter().map(|(option, path)| (*option, path.clone())).collect(),
            opened: BTreeMap::new(),
            base,
            hash_chunk_bytes,
            done,
        }
        .next()
    }

    /// The option the objects an allocation does not name come from.
    pub fn base(&self) -> QuantOption {
        self.base
    }

    /// The options that have a source, in order.
    pub fn options(&self) -> Vec<QuantOption> {
        self.files.keys().copied().collect()
    }

    /// The source of `option`.
    pub fn artifact(&self, option: QuantOption) -> Result<&ArtifactFile, AssembleError> {
        self.files
            .get(&option)
            .ok_or_else(|| AssembleError::invalid(format!("no source for {}", option.key())))
    }

    /// The file of `option`'s source.
    pub fn path(&self, option: QuantOption) -> Result<&Path, AssembleError> {
        self.artifact(option).map(ArtifactFile::path)
    }

    /// The choosable objects, in the base's order: its quantized linear objects of `text/layers/*`.
    pub fn projections(&self) -> &[String] {
        &self.projections
    }

    /// The quantized linear objects of the MTP layer, in the base's order. An allocation does not choose them.
    pub fn drafter_projections(&self) -> &[String] {
        &self.drafter_projections
    }

    /// The descriptor of `name` in `option`'s source.
    pub fn descriptor(&self, option: QuantOption, name: &str) -> Result<&TensorEntry, AssembleError> {
        self.artifact(option)?
            .table()
            .get(name)
            .ok_or_else(|| AssembleError::invalid(format!("{} has no object {name}", option.key())))
    }

    /// The descriptors of `name` and of its companions in `option`'s source, in the order of
    /// [`COMPANION_SUFFIXES`]: the companions travel with their object.
    pub fn object_with_companions(&self, option: QuantOption, name: &str) -> Result<Vec<&TensorEntry>, AssembleError> {
        let mut all = vec![self.descriptor(option, name)?];
        let table = self.artifact(option)?.table();
        for suffix in COMPANION_SUFFIXES {
            if let Some(companion) = table.get(&format!("{name}{suffix}")) {
                all.push(companion);
            }
        }
        Ok(all)
    }

    /// Whether two stored projections are the same linear: equal input and output features, whatever each format's
    /// storage shape (a trellis object is `[k/16, n/16, w]`, a row-split one `[n, k]`).
    pub fn same_features(a: &TensorEntry, b: &TensorEntry) -> bool {
        matches!((shape_of(a), shape_of(b)), (Ok(a), Ok(b)) if a == b)
    }

    /// The checks of one option against the base that need no payload.
    fn check_option(&self, option: QuantOption) -> Result<(), AssembleError> {
        let base = self.base.key();
        let table = self.artifact(option)?.table();
        let mine: BTreeSet<&str> = quantized_linears(table, LAYERS).into_iter().collect();
        let theirs: BTreeSet<&str> = self.projections.iter().map(String::as_str).collect();
        if mine != theirs {
            return Err(AssembleError::invalid(format!(
                "{} and {base} do not have the same projections, first difference: {}",
                option.key(),
                first_difference(&mine, &theirs)
            )));
        }
        let compared_mine = compared(table);
        let compared_base = compared(self.artifact(self.base)?.table());
        if compared_mine != compared_base {
            return Err(AssembleError::invalid(format!(
                "{} and {base} do not have the same shared objects, first difference: {}",
                option.key(),
                first_difference(&compared_mine, &compared_base)
            )));
        }
        for name in &self.projections {
            let (here, there) = (self.descriptor(option, name)?, self.descriptor(self.base, name)?);
            if !Self::same_features(here, there) {
                return Err(AssembleError::invalid(format!(
                    "{name} has the shape {:?} in {} and {:?} in {base}",
                    here.shape(),
                    option.key(),
                    there.shape()
                )));
            }
        }
        Ok(())
    }
}

impl Sources for SourceSet {
    fn base(&self) -> QuantOption {
        self.base
    }

    fn projections(&self) -> &[String] {
        &self.projections
    }

    fn shape(&self, option: QuantOption, object: &str) -> Result<Shape, MixedError> {
        let entry = self
            .descriptor(option, object)
            .map_err(|error| MixedError::Invalid(error.to_string()))?;
        shape_of(entry)
    }

    fn bytes(&self, option: QuantOption, object: &str) -> Result<i64, MixedError> {
        let parts = self
            .object_with_companions(option, object)
            .map_err(|error| MixedError::Invalid(error.to_string()))?;
        let total = parts
            .iter()
            .try_fold(0u64, |total, part| total.checked_add(part.bytes()))
            .and_then(|total| i64::try_from(total).ok());
        total.ok_or_else(|| MixedError::Invalid(format!("{object}: the bytes do not fit a 64-bit count")))
    }
}

fn quantized(format: WeightFormat) -> bool {
    matches!(
        format,
        WeightFormat::Nvfp4
            | WeightFormat::Trellis
            | WeightFormat::Q3G64Fp16
            | WeightFormat::Q4G64Fp16
            | WeightFormat::Q5G64Fp16
    )
}

/// Whether `name` is the companion of an object of `table`.
fn is_companion(table: &TensorTable, name: &str) -> bool {
    COMPANION_SUFFIXES
        .iter()
        .any(|suffix| name.strip_suffix(suffix).is_some_and(|stem| table.get(stem).is_some()))
}

/// The quantized objects of `table` under `prefix` that are not companions, in the table's order.
fn quantized_linears<'a>(table: &'a TensorTable, prefix: &str) -> Vec<&'a str> {
    table
        .entries()
        .iter()
        .filter(|entry| {
            entry.name().starts_with(prefix) && quantized(entry.format()) && !is_companion(table, entry.name())
        })
        .map(TensorEntry::name)
        .collect()
}

/// The objects of a source that come from the checkpoint as they are: raw (not quantized) objects that are not the
/// MTP layer's and not companions.
fn compared(table: &TensorTable) -> BTreeSet<&str> {
    table
        .entries()
        .iter()
        .filter(|entry| {
            !entry.name().starts_with("mtp/") && !quantized(entry.format()) && !is_companion(table, entry.name())
        })
        .map(TensorEntry::name)
        .collect()
}

fn owned(names: Vec<&str>) -> Vec<String> {
    names.into_iter().map(str::to_owned).collect()
}

/// The first name in only one of `a` and `b`.
fn first_difference<'a>(a: &BTreeSet<&'a str>, b: &BTreeSet<&'a str>) -> &'a str {
    a.symmetric_difference(b).next().copied().unwrap_or_default()
}

/// The steps that open the sources one after the other.
struct Opening {
    io: Arc<AsyncIo>,
    pending: VecDeque<(QuantOption, PathBuf)>,
    opened: BTreeMap<QuantOption, ArtifactFile>,
    base: QuantOption,
    hash_chunk_bytes: usize,
    done: Done<SourceSet>,
}

impl Opening {
    /// Opens the next source, or goes on to the checks when all are open. `Err` started nothing.
    fn next(mut self) -> Result<(), AssembleError> {
        let Some((option, path)) = self.pending.pop_front() else {
            return self.verify();
        };
        let io = Arc::clone(&self.io);
        let wants_tokenizer = option == self.base;
        let cell = Handoff::new();
        cell.send((self, option));
        let waiting = Arc::clone(&cell);
        let started = ArtifactFile::open(
            &io,
            &path,
            wants_tokenizer,
            Arc::new(move |opened| {
                let Some((mut state, option)) = waiting.take() else {
                    return;
                };
                match opened {
                    Err(error) => (state.done)(Err(error)),
                    Ok(file) => {
                        state.opened.insert(option, file);
                        let report = Arc::clone(&state.done);
                        if let Err(error) = state.next() {
                            report(Err(error));
                        }
                    }
                }
            }),
        );
        if started.is_err() {
            cell.take();
        }
        started
    }

    /// All sources are open: lists the projections and compares the shared objects. `Err` started nothing.
    fn verify(self) -> Result<(), AssembleError> {
        let Opening {
            io,
            opened,
            base,
            hash_chunk_bytes,
            done,
            ..
        } = self;
        let base_table = opened
            .get(&base)
            .ok_or_else(|| AssembleError::invalid(format!("no source for the base {}", base.key())))?
            .table();
        let projections = owned(quantized_linears(base_table, LAYERS));
        let drafter_projections = owned(quantized_linears(base_table, DRAFTER_LAYER));
        let names: Vec<String> = compared(base_table).into_iter().map(str::to_owned).collect();
        let others: Vec<QuantOption> = opened.keys().copied().filter(|option| *option != base).collect();
        let set = SourceSet {
            files: opened,
            base,
            projections,
            drafter_projections,
        };
        Verifying {
            io,
            set,
            others,
            option: 0,
            names,
            name: 0,
            base_digests: HashMap::new(),
            hash_chunk_bytes,
            done,
        }
        .run();
        Ok(())
    }
}

/// The comparison of each shared object of each source with the base's, in order.
struct Verifying {
    io: Arc<AsyncIo>,
    set: SourceSet,
    others: Vec<QuantOption>,
    option: usize,
    names: Vec<String>,
    name: usize,
    /// The digest of each of the base's objects once it was hashed, so it is read once.
    base_digests: HashMap<String, [u8; 32]>,
    hash_chunk_bytes: usize,
    done: Done<SourceSet>,
}

impl Verifying {
    fn run(mut self) {
        loop {
            let Some(option) = self.others.get(self.option).copied() else {
                let Verifying { set, done, .. } = self;
                done(Ok(set));
                return;
            };
            if self.name == 0
                && let Err(error) = self.set.check_option(option)
            {
                (self.done)(Err(error));
                return;
            }
            let Some(name) = self.names.get(self.name).cloned() else {
                self.option += 1;
                self.name = 0;
                continue;
            };
            match self.compare_descriptors(option, &name) {
                Err(error) => (self.done)(Err(error)),
                Ok(()) => self.hash_both(option, name),
            }
            return;
        }
    }

    fn compare_descriptors(&self, option: QuantOption, name: &str) -> Result<(), AssembleError> {
        let (mine, theirs) = (
            self.set.descriptor(option, name)?,
            self.set.descriptor(self.set.base, name)?,
        );
        if mine.shape() != theirs.shape()
            || mine.dtype() != theirs.dtype()
            || mine.format() != theirs.format()
            || mine.layout() != theirs.layout()
            || mine.bytes() != theirs.bytes()
        {
            return Err(self.differs(option, name));
        }
        Ok(())
    }

    fn differs(&self, option: QuantOption, name: &str) -> AssembleError {
        AssembleError::invalid(format!(
            "{name} differs between {} and {}",
            option.key(),
            self.set.base.key()
        ))
    }

    /// Hashes the object in `option`'s source and in the base's (unless that was done), and compares.
    fn hash_both(self, option: QuantOption, name: String) {
        let (file, entry) = match self.object(option, &name) {
            Ok(found) => found,
            Err(error) => return (self.done)(Err(error)),
        };
        let cell = Handoff::new();
        let (io, chunk) = (Arc::clone(&self.io), self.hash_chunk_bytes);
        let report = Arc::clone(&self.done);
        cell.send((self, name.clone()));
        let waiting = Arc::clone(&cell);
        let label = format!("{name} of {}", option.key());
        Hashing::start(
            io,
            file,
            entry,
            chunk,
            label,
            Arc::new(move |digest| {
                let Some((state, name)) = waiting.take() else {
                    return;
                };
                match digest {
                    Err(error) => (state.done)(Err(error)),
                    Ok(digest) => state.then_base(option, name, digest),
                }
            }),
        )
        .unwrap_or_else(|error| report(Err(error)));
    }

    fn object(&self, option: QuantOption, name: &str) -> Result<(Arc<OwnedFile>, (u64, u64)), AssembleError> {
        let entry = self.set.descriptor(option, name)?;
        let file = Arc::clone(self.set.artifact(option)?.file());
        Ok((file, (entry.offset(), entry.bytes())))
    }

    /// `mine` is the digest of the object in `option`'s source: gets the base's, then compares.
    fn then_base(self, option: QuantOption, name: String, mine: [u8; 32]) {
        if let Some(theirs) = self.base_digests.get(&name).copied() {
            return self.compared(option, &name, mine, theirs);
        }
        let base = self.set.base;
        let (file, entry) = match self.object(base, &name) {
            Ok(found) => found,
            Err(error) => return (self.done)(Err(error)),
        };
        let cell = Handoff::new();
        let (io, chunk) = (Arc::clone(&self.io), self.hash_chunk_bytes);
        let report = Arc::clone(&self.done);
        let label = format!("{name} of {}", base.key());
        cell.send((self, name));
        let waiting = Arc::clone(&cell);
        Hashing::start(
            io,
            file,
            entry,
            chunk,
            label,
            Arc::new(move |digest| {
                let Some((mut state, name)) = waiting.take() else {
                    return;
                };
                match digest {
                    Err(error) => (state.done)(Err(error)),
                    Ok(theirs) => {
                        state.base_digests.insert(name.clone(), theirs);
                        state.compared(option, &name, mine, theirs);
                    }
                }
            }),
        )
        .unwrap_or_else(|error| report(Err(error)));
    }

    fn compared(mut self, option: QuantOption, name: &str, mine: [u8; 32], theirs: [u8; 32]) {
        if mine != theirs {
            let error = self.differs(option, name);
            return (self.done)(Err(error));
        }
        self.name += 1;
        self.run();
    }
}

/// The SHA-256 of a range of a file, read in chunks.
struct Hashing {
    io: Arc<AsyncIo>,
    file: Arc<OwnedFile>,
    at: u64,
    end: u64,
    hasher: Sha256,
    buffer: Vec<u8>,
    chunk: usize,
    label: String,
    then: Done<[u8; 32]>,
}

impl Hashing {
    /// Hashes `range` (offset, length) of `file`. `Err` started nothing; an empty range ends at once.
    fn start(
        io: Arc<AsyncIo>,
        file: Arc<OwnedFile>,
        range: (u64, u64),
        chunk: usize,
        label: String,
        then: Done<[u8; 32]>,
    ) -> Result<(), AssembleError> {
        Hashing {
            io,
            file,
            at: range.0,
            end: range.0.saturating_add(range.1),
            hasher: Sha256::new(),
            buffer: Vec::new(),
            chunk,
            label,
            then,
        }
        .step()
    }

    fn step(mut self) -> Result<(), AssembleError> {
        if self.at >= self.end {
            let digest: [u8; 32] = self.hasher.finalize_reset().into();
            (self.then)(Ok(digest));
            return Ok(());
        }
        let length = usize::try_from(self.end - self.at).map_or(self.chunk, |left| left.min(self.chunk));
        let (io, file, offset) = (Arc::clone(&self.io), Arc::clone(&self.file), self.at);
        let context = format!("{} (hashing)", self.label);
        let buffer = take(&mut self.buffer);
        let cell = Handoff::new();
        cell.send(self);
        let waiting = Arc::clone(&cell);
        let next: Chunk = Box::new(move |read| {
            let Some(mut state) = waiting.take() else {
                return;
            };
            match read {
                Err(error) => (state.then)(Err(error)),
                Ok(bytes) => {
                    state.hasher.update(&bytes);
                    state.at += bytes.len() as u64;
                    state.buffer = bytes;
                    let report = Arc::clone(&state.then);
                    if let Err(error) = state.step() {
                        report(Err(error));
                    }
                }
            }
        });
        let started = read_chunk(&io, file, offset, length, buffer, context, next);
        if started.is_err() {
            cell.take();
        }
        started
    }
}
