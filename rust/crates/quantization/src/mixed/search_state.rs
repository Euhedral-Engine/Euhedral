use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_data_structures::sync::ClaimCell;
use euhedral_graph::Lake;
use euhedral_io::{AsyncIo, Continuation};
use serde_json::{Map, Number, Value, from_slice};
use thiserror::Error;

use super::allocation::Allocation;
use super::error::MixedError;
use super::option::QuantOption;
use crate::resume::{HEADER, Header, ResumeError, ResumeFile};
use crate::step::idle;

/// The `kind` property of a search's resume file.
const KIND: &str = "mixed-search";
/// Bytes of the length that starts a record.
const LENGTH_BYTES: u64 = 4;

/// What can go wrong with a search state.
#[derive(Debug, Error)]
pub enum SearchError {
    #[error("{0}")]
    Resume(#[from] ResumeError),
    /// The file holds a search with another fingerprint, or is some other resume file.
    #[error(
        "{path} holds a different search (its kind or fingerprint differs from this run's): give --state a new file"
    )]
    DifferentSearch { path: PathBuf },
    /// A file that is not a search state was asked to be read as one.
    #[error("{path} is not a mixed-allocation search state; if the file is yours to lose, delete it and run again")]
    NotSearchState { path: PathBuf },
    /// The header or a record does not say what a state file does.
    #[error("the search state is damaged: {0}")]
    Damaged(String),
}

impl SearchError {
    fn damaged(why: impl Into<String>) -> SearchError {
        SearchError::Damaged(why.into())
    }
}

impl From<MixedError> for SearchError {
    fn from(error: MixedError) -> SearchError {
        SearchError::Damaged(error.to_string())
    }
}

/// What a call's continuation is told.
pub type Then<T> = Continuation<Result<T, SearchError>>;

/// A candidate that was scored: its allocation, KL against the reference and the KL's standard error, the bytes of the
/// composite artifact, the cost table's prefill microseconds per token and verification microseconds, whether the plan
/// at the target context keeps every object on the device, and the round and origin that proposed it.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub allocation: Allocation,
    pub kl: f64,
    pub kl_se: f64,
    pub bytes: i64,
    pub prefill: f64,
    pub decode: f64,
    pub fits: bool,
    pub round: u32,
    pub origin: String,
}

/// How far a search has got: the round in progress (0 while the seeds are measured), the share of each batch proposed
/// for what the ensemble is unsure of, the rounds in a row without a new front point, and the held-out error of the
/// last round that finished. The two errors are NaN until they are set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub round: u32,
    pub explore: f64,
    pub stale: u32,
    pub held_out: f64,
}

impl Default for Progress {
    fn default() -> Progress {
        Progress {
            round: 0,
            explore: f64::NAN,
            stale: 0,
            held_out: f64::NAN,
        }
    }
}

/// What a search has measured and where it stands, in one [`ResumeFile`] that is overwritten in place.
///
/// The header holds the search's fingerprint, the seed, the device's free bytes at the start, how many measured
/// candidates the file holds and where they end, and the [`Progress`]. Each candidate is appended after the previous
/// ones as a little-endian 4-byte length and its JSON, and the header is rewritten to cover it only once the record is
/// on disk, so a stop part way through a record leaves the file as it was before that candidate. A candidate already
/// recorded is refused, so a restarted run never scores one twice nor records it twice.
///
/// Every call takes the state and hands it back to its continuation: the state belongs to the frame that is working
/// with it. A call that fails leaves nothing behind but the file, which holds the last state that was written whole.
/// The file closes when the state is dropped.
#[derive(Debug)]
pub struct SearchState {
    file: ResumeFile,
    header: Header,
    measured: Vec<Measured>,
    known: HashSet<Allocation>,
    end: u64,
    seed: i64,
    free_bytes: u64,
    progress: Progress,
}

impl SearchState {
    /// The state in `path`, or a new one with `seed` and `free_bytes` recorded when the file is missing or empty. A file
    /// that holds a search with another `fingerprint`, or anything else, is refused, and the file is left alone.
    pub fn open(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        fingerprint: &str,
        seed: i64,
        free_bytes: u64,
        then: Then<SearchState>,
    ) -> Result<(), SearchError> {
        let settle = Settle::new(then);
        let path_held = path.to_owned();
        let fingerprint = fingerprint.to_owned();
        let opened = move |opened: Result<ResumeFile, ResumeError>| match opened {
            Err(error) => settle.finish(Err(error.into())),
            Ok(file) => {
                let held = file.clone();
                let next = settle.clone();
                let sized = move |len: Result<u64, ResumeError>| match len {
                    Err(error) => next.finish(Err(error.into())),
                    Ok(len) => {
                        let (file, again) = (held.clone(), next.clone());
                        let header = move |header: Result<Header, ResumeError>| match header {
                            Err(error) => again.finish(Err(error.into())),
                            Ok(header) => {
                                let identity = Identity {
                                    path: path_held,
                                    fingerprint,
                                    seed,
                                    free_bytes,
                                };
                                Self::begin(held, header, len, identity, again)
                            }
                        };
                        next.started(file.header(Box::new(header)))
                    }
                };
                settle.started(file.len(Box::new(sized)))
            }
        };
        Ok(ResumeFile::open(io, host, path, Box::new(opened))?)
    }

    /// The measured candidates of the search in `path`, whatever search it is: for a new search to start from what
    /// another measured (KL does not depend on the context or the budget a search fits to). A file that is not a
    /// search state is refused, and a missing one is an I/O error: the file is opened for reading only, so nothing is created or changed.
    pub fn read(
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        path: &Path,
        then: Then<Vec<Measured>>,
    ) -> Result<(), SearchError> {
        let settle = Settle::new(then);
        let path_held = path.to_owned();
        let opened = move |opened: Result<ResumeFile, ResumeError>| match opened {
            Err(error) => settle.finish(Err(error.into())),
            Ok(file) => {
                let held = file.clone();
                let next = settle.clone();
                let header = move |header: Result<Header, ResumeError>| match header {
                    Err(error) => next.finish(Err(error.into())),
                    Ok(header) => {
                        if header.get("kind") != Some(KIND) {
                            return next.finish(Err(SearchError::NotSearchState { path: path_held }));
                        }
                        match Layout::of(&header) {
                            Ok(layout) => Loader::new(held, layout, next).step(),
                            Err(error) => next.finish(Err(error)),
                        }
                    }
                };
                settle.started(file.header(Box::new(header)))
            }
        };
        Ok(ResumeFile::open_read_only(io, host, path, Box::new(opened))?)
    }

    fn begin(
        file: ResumeFile,
        header: Header,
        len: u64,
        identity: Identity,
        settle: Settle<SearchState>,
    ) -> Box<dyn Frame> {
        if header.is_empty() && len != 0 {
            // A file with something in it that is not a search is not ours to overwrite. The one exception is what a
            // stop during the very first header write leaves: no more than a header's bytes, all zero.
            if len > HEADER {
                return settle.finish(Err(SearchError::NotSearchState { path: identity.path }));
            }
            let again = settle.clone();
            let held = file.clone();
            let read = move |bytes: Result<Vec<u8>, ResumeError>| match bytes {
                Err(error) => again.finish(Err(error.into())),
                Ok(bytes) if bytes.iter().all(|&byte| byte == 0) => Self::fresh(held, identity, again),
                Ok(_) => again.finish(Err(SearchError::NotSearchState { path: identity.path })),
            };
            return settle.started(file.read(0, len as usize, Box::new(read)));
        }
        if header.is_empty() {
            return Self::fresh(file, identity, settle);
        }
        Self::resume(file, header, identity, settle)
    }

    /// A new state: its header is the first thing written.
    fn fresh(file: ResumeFile, identity: Identity, settle: Settle<SearchState>) -> Box<dyn Frame> {
        {
            let mut header = Header::new();
            header.set("kind", KIND);
            header.set("fingerprint", identity.fingerprint);
            header.set("seed", identity.seed.to_string());
            header.set("freeBytes", identity.free_bytes.to_string());
            header.set("count", "0");
            header.set("end", HEADER.to_string());
            let state = SearchState {
                file: file.clone(),
                header: header.clone(),
                measured: Vec::new(),
                known: HashSet::new(),
                end: HEADER,
                seed: identity.seed,
                free_bytes: identity.free_bytes,
                progress: Progress::default(),
            };
            let written = settle.clone();
            let done = move |done: Result<(), ResumeError>| match done {
                Ok(()) => written.finish(Ok(state)),
                Err(error) => written.finish(Err(error.into())),
            };
            settle.started(file.set_header(&header, Box::new(done)))
        }
    }

    /// The state a file holds, if it is this search's.
    fn resume(file: ResumeFile, header: Header, identity: Identity, settle: Settle<SearchState>) -> Box<dyn Frame> {
        if header.get("kind") != Some(KIND) || header.get("fingerprint") != Some(identity.fingerprint.as_str()) {
            return settle.finish(Err(SearchError::DifferentSearch { path: identity.path }));
        }
        let parsed = (|| {
            let seed = number::<i64>(&header, "seed")?;
            let free_bytes = number::<u64>(&header, "freeBytes")?;
            let progress = Progress {
                round: optional::<u32>(&header, "round", 0)?,
                explore: optional_double(&header, "explore")?,
                stale: optional::<u32>(&header, "stale", 0)?,
                held_out: optional_double(&header, "heldOut")?,
            };
            Ok::<_, SearchError>((seed, free_bytes, progress))
        })();
        let (seed, free_bytes, progress) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => return settle.finish(Err(error)),
        };
        let layout = match Layout::of(&header) {
            Ok(layout) => layout,
            Err(error) => return settle.finish(Err(error)),
        };
        let end = layout.end;
        let loaded = settle.clone();
        let held = file.clone();
        let built = Settle::new(Box::new(
            move |measured: Result<Vec<Measured>, SearchError>| match measured {
                Err(error) => loaded.finish(Err(error)),
                Ok(measured) => {
                    let known = measured.iter().map(|each| each.allocation.clone()).collect();
                    loaded.finish(Ok(SearchState {
                        file: held,
                        header,
                        measured,
                        known,
                        end,
                        seed,
                        free_bytes,
                        progress,
                    }))
                }
            },
        ));
        Loader::new(file, layout, built).step()
    }

    /// Every measured candidate, in the order they were measured.
    pub fn measured(&self) -> &[Measured] {
        &self.measured
    }

    /// Whether `allocation` was measured.
    pub fn contains(&self, allocation: &Allocation) -> bool {
        self.known.contains(allocation)
    }

    /// The seed of the run that started the search.
    pub fn seed(&self) -> i64 {
        self.seed
    }

    /// The device's free bytes when the search started.
    pub fn free_bytes(&self) -> u64 {
        self.free_bytes
    }

    /// How far the search has got.
    pub fn progress(&self) -> Progress {
        self.progress
    }

    /// Records `candidate` and hands the state to `then` with `true`; with `false`, and nothing written, when its
    /// allocation is already recorded.
    pub fn add(self, candidate: Measured, then: Then<(SearchState, bool)>) -> Result<(), SearchError> {
        if self.known.contains(&candidate.allocation) {
            let file = self.file.clone();
            let known = move |done: Result<(), ResumeError>| match done {
                Ok(_) => then(Ok((self, false))),
                Err(error) => then(Err(error.into())),
            };
            return Ok(file.settled(Box::new(known))?);
        }
        let body = encode(&candidate).into_bytes();
        let length = u32::try_from(body.len()).map_err(|_| SearchError::damaged("a record is over 4 GiB"))?;
        let mut record = Vec::with_capacity(LENGTH_BYTES as usize + body.len());
        record.extend_from_slice(&length.to_le_bytes());
        record.extend_from_slice(&body);
        let (file, at) = (self.file.clone(), self.end);
        let settle = Settle::new(then);
        let written = move |written: Result<u64, ResumeError>| {
            let mut state = self;
            match written {
                Err(error) => settle.finish(Err(error.into())),
                Ok(end) => {
                    state.end = end;
                    state.known.insert(candidate.allocation.clone());
                    state.measured.push(candidate);
                    state.header.set("count", state.measured.len().to_string());
                    state.header.set("end", end.to_string());
                    let header = state.header.clone();
                    let file = state.file.clone();
                    let covered = settle.clone();
                    let done = move |done: Result<(), ResumeError>| match done {
                        Ok(()) => covered.finish(Ok((state, true))),
                        Err(error) => covered.finish(Err(error.into())),
                    };
                    settle.started(file.set_header(&header, Box::new(done)))
                }
            }
        };
        Ok(file.write(at, record, Box::new(written))?)
    }

    /// Records `progress` and hands the state to `then`.
    pub fn set_progress(mut self, progress: Progress, then: Then<SearchState>) -> Result<(), SearchError> {
        self.header.set("round", progress.round.to_string());
        self.header.set("explore", double_text(progress.explore));
        self.header.set("stale", progress.stale.to_string());
        self.header.set("heldOut", double_text(progress.held_out));
        self.progress = progress;
        let (file, header) = (self.file.clone(), self.header.clone());
        let done = move |done: Result<(), ResumeError>| match done {
            Ok(()) => then(Ok(self)),
            Err(error) => then(Err(error.into())),
        };
        Ok(file.set_header(&header, Box::new(done))?)
    }
}

struct Identity {
    path: PathBuf,
    fingerprint: String,
    seed: i64,
    free_bytes: u64,
}

/// Where the records are: how many, and where they end.
#[derive(Clone, Copy)]
struct Layout {
    count: usize,
    end: u64,
}

impl Layout {
    fn of(header: &Header) -> Result<Layout, SearchError> {
        let layout = Layout {
            count: number(header, "count")?,
            end: number(header, "end")?,
        };
        if layout.end < HEADER {
            return Err(SearchError::damaged(format!(
                "the records end at {}, inside the header",
                layout.end
            )));
        }
        Ok(layout)
    }
}

fn number<T: FromStr>(header: &Header, key: &str) -> Result<T, SearchError> {
    let text = header
        .get(key)
        .ok_or_else(|| SearchError::damaged(format!("the header has no {key}")))?;
    text.trim()
        .parse()
        .map_err(|_| SearchError::damaged(format!("the header's {key} is '{text}', not a number")))
}

fn optional<T: FromStr>(header: &Header, key: &str, absent: T) -> Result<T, SearchError> {
    match header.get(key) {
        None => Ok(absent),
        Some(_) => number(header, key),
    }
}

fn optional_double(header: &Header, key: &str) -> Result<f64, SearchError> {
    match header.get(key) {
        None => Ok(f64::NAN),
        Some(text) => parse_double(text).ok_or_else(|| SearchError::damaged(format!("the header's {key} is '{text}'"))),
    }
}

/// A double as text that [`parse_double`] reads back to the same value: NaN and the infinities by name.
fn double_text(value: f64) -> String {
    if value.is_nan() {
        "NaN".to_string()
    } else if value.is_infinite() {
        if value > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else {
        format!("{value:?}")
    }
}

fn parse_double(text: &str) -> Option<f64> {
    match text.trim() {
        "NaN" => Some(f64::NAN),
        "Infinity" | "+Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        other
            if other
                .chars()
                .any(|character| character.is_ascii_alphabetic() && !matches!(character, 'e' | 'E')) =>
        {
            None
        }
        other => other.parse().ok(),
    }
}

/// A continuation that waits to be taken by the step that ends the call, so that it runs once, and so that a step
/// that cannot start can end the call with its error.
struct Settle<T>(Arc<ClaimCell<Option<Then<T>>>>);

impl<T> Clone for Settle<T> {
    fn clone(&self) -> Settle<T> {
        Settle(Arc::clone(&self.0))
    }
}

impl<T> Settle<T> {
    fn new(then: Then<T>) -> Settle<T> {
        Settle(Arc::new(ClaimCell::new(Some(then))))
    }

    /// Ends the call with `result`; returns the frame its continuation gives, or nothing to do if the call has ended.
    fn finish(&self, result: Result<T, SearchError>) -> Box<dyn Frame> {
        match self.0.try_claim().and_then(|mut claim| claim.take()) {
            Some(then) => then(result),
            None => idle(),
        }
    }

    /// Ends the call with the error of a step that could not start; a step that started is nothing to do.
    fn started<E: Into<SearchError>>(&self, started: Result<(), E>) -> Box<dyn Frame> {
        match started {
            Ok(()) => idle(),
            Err(error) => self.finish(Err(error.into())),
        }
    }
}

/// Reads the records one after the other, each as its length and then its body.
struct Loader {
    file: ResumeFile,
    layout: Layout,
    at: u64,
    measured: Vec<Measured>,
    done: Settle<Vec<Measured>>,
}

impl Loader {
    fn new(file: ResumeFile, layout: Layout, done: Settle<Vec<Measured>>) -> Loader {
        Loader {
            file,
            layout,
            at: HEADER,
            measured: Vec::new(),
            done,
        }
    }

    fn step(self) -> Box<dyn Frame> {
        if self.measured.len() == self.layout.count {
            return if self.at == self.layout.end {
                let Loader {
                    file, measured, done, ..
                } = self;
                drop(file);
                done.finish(Ok(measured))
            } else {
                let at = self.at;
                let recorded = self.layout.end;
                self.done.finish(Err(SearchError::damaged(format!(
                    "the records end at {at}, not {recorded}"
                ))))
            };
        }
        if self.at + LENGTH_BYTES > self.layout.end {
            return self.done.finish(Err(SearchError::damaged(
                "the records stop short of the count the header gives",
            )));
        }
        let (file, at, done) = (self.file.clone(), self.at, self.done.clone());
        let length = move |bytes: Result<Vec<u8>, ResumeError>| match bytes {
            Err(error) => self.done.finish(Err(error.into())),
            Ok(bytes) => self.body(&bytes),
        };
        done.started(file.read(at, LENGTH_BYTES as usize, Box::new(length)))
    }

    fn body(self, length: &[u8]) -> Box<dyn Frame> {
        let length = u64::from(u32::from_le_bytes([length[0], length[1], length[2], length[3]]));
        let start = self.at + LENGTH_BYTES;
        if start + length > self.layout.end {
            let at = self.at;
            return self.done.finish(Err(SearchError::damaged(format!(
                "the record at {at} runs past the end of the records"
            ))));
        }
        let (file, done) = (self.file.clone(), self.done.clone());
        let read = move |bytes: Result<Vec<u8>, ResumeError>| match bytes {
            Err(error) => self.done.finish(Err(error.into())),
            Ok(bytes) => self.record(&bytes, start + length),
        };
        done.started(file.read(start, length as usize, Box::new(read)))
    }

    fn record(mut self, body: &[u8], next: u64) -> Box<dyn Frame> {
        match decode(body) {
            Ok(measured) => {
                self.measured.push(measured);
                self.at = next;
                self.step()
            }
            Err(error) => self.done.finish(Err(error)),
        }
    }
}

fn encode(measured: &Measured) -> String {
    let mut objects = Map::new();
    for (name, option) in measured.allocation.choices() {
        objects.insert(name.clone(), Value::String(option.key().to_string()));
    }
    let mut node = Map::new();
    node.insert("objects".into(), Value::Object(objects));
    node.insert("kl".into(), double(measured.kl));
    node.insert("klSe".into(), double(measured.kl_se));
    node.insert("bytes".into(), Value::from(measured.bytes));
    node.insert("prefill".into(), double(measured.prefill));
    node.insert("decode".into(), double(measured.decode));
    node.insert("fits".into(), Value::Bool(measured.fits));
    node.insert("round".into(), Value::from(measured.round));
    node.insert("origin".into(), Value::String(measured.origin.clone()));
    Value::Object(node).to_string()
}

/// A double as JSON: a number, or for NaN and the infinities (which JSON has no number for) their names as text.
fn double(value: f64) -> Value {
    match Number::from_f64(value) {
        Some(number) => Value::Number(number),
        None => Value::String(double_text(value)),
    }
}

fn decode(body: &[u8]) -> Result<Measured, SearchError> {
    let root: Value = from_slice(body).map_err(|error| SearchError::damaged(format!("a record: {error}")))?;
    let field = |name: &str| {
        root.get(name)
            .ok_or_else(|| SearchError::damaged(format!("a record has no {name}")))
    };
    let real = |name: &str| -> Result<f64, SearchError> {
        match field(name)? {
            Value::Number(number) => number
                .as_f64()
                .ok_or_else(|| SearchError::damaged(format!("a record's {name} is not a number"))),
            Value::String(text) => {
                parse_double(text).ok_or_else(|| SearchError::damaged(format!("a record's {name} is '{text}'")))
            }
            _ => Err(SearchError::damaged(format!("a record's {name} is not a number"))),
        }
    };
    let objects = field("objects")?
        .as_object()
        .ok_or_else(|| SearchError::damaged("a record's objects are not a map"))?;
    let mut choices = BTreeMap::new();
    for (name, option) in objects {
        let key = option
            .as_str()
            .ok_or_else(|| SearchError::damaged(format!("object {name} has no option name")))?;
        choices.insert(
            name.clone(),
            QuantOption::of(key).map_err(|error| SearchError::damaged(format!("object {name}: {error}")))?,
        );
    }
    Ok(Measured {
        allocation: Allocation::new(choices),
        kl: real("kl")?,
        kl_se: real("klSe")?,
        bytes: field("bytes")?
            .as_i64()
            .ok_or_else(|| SearchError::damaged("a record's bytes are not a whole number"))?,
        prefill: real("prefill")?,
        decode: real("decode")?,
        fits: field("fits")?
            .as_bool()
            .ok_or_else(|| SearchError::damaged("a record's fits is not true or false"))?,
        round: field("round")?
            .as_u64()
            .and_then(|round| u32::try_from(round).ok())
            .ok_or_else(|| SearchError::damaged("a record's round is not a whole number"))?,
        origin: field("origin")?
            .as_str()
            .ok_or_else(|| SearchError::damaged("a record's origin is not text"))?
            .to_string(),
    })
}
