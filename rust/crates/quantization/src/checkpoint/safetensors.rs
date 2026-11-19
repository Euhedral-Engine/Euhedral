//! The tensors of a Hugging Face checkpoint directory: `model.safetensors.index.json` and its shards, or a single
//! `model.safetensors`. Tensors are read with positioned reads, never memory-mapped: the checkpoints live on a shared
//! exFAT drive, where page faults on large mappings hung the kernel.

use std::collections::{BTreeSet, HashMap, VecDeque};
use std::io::{Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use euhedral_io::{AsyncIo, IoDoneOwned, IoMemory, OwnedFile};
use serde_json::Value;

use super::io::{self, Done, Next, idle, open_read, raw_file, read_exact, read_file, work};
use super::{CheckpointError, report};

/// The largest header a shard may declare.
pub const MAX_HEADER_BYTES: u64 = 1 << 30;

const INDEX: &str = "model.safetensors.index.json";
const SINGLE: &str = "model.safetensors";
const METADATA: &str = "__metadata__";

/// One tensor: its data type (`BF16`, `F32`, ...), shape, and byte range in its shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    shard: usize,
    /// Where the tensor's bytes start in its shard.
    pub offset: u64,
    pub bytes: u64,
}

impl Tensor {
    /// The number of elements: the product of the shape (1 for a scalar).
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

#[derive(Debug)]
struct Shard {
    name: String,
    file: Arc<OwnedFile>,
}

/// An open checkpoint: its tensors' metadata and its shard files.
#[derive(Debug)]
pub struct Safetensors {
    tensors: HashMap<String, Tensor>,
    shards: Vec<Shard>,
}

/// Part of a tensor to read: `memory.len()` bytes starting `skip` bytes into it.
#[derive(Debug)]
pub struct ReadRange<B> {
    pub tensor: String,
    pub skip: u64,
    pub memory: B,
}

/// The shard file names of an index, sorted and without repeats.
pub fn shard_names(index: &[u8]) -> Result<Vec<String>, CheckpointError> {
    let root: Value = serde_json::from_slice(index)
        .map_err(|error| CheckpointError::Format(format!("{INDEX} is not JSON: {error}")))?;
    let map = root
        .get("weight_map")
        .and_then(Value::as_object)
        .ok_or_else(|| CheckpointError::Format(format!("{INDEX} has no weight_map")))?;
    let mut names = BTreeSet::new();
    for shard in map.values() {
        let name = shard
            .as_str()
            .ok_or_else(|| CheckpointError::Format(format!("{INDEX}: a weight_map value is not a file name")))?;
        names.insert(name.to_owned());
    }
    Ok(names.into_iter().collect())
}

/// The length of a shard's JSON header, from the 8 bytes that start the file.
pub fn header_length(shard: &str, prefix: &[u8]) -> Result<u64, CheckpointError> {
    let bytes: [u8; 8] = prefix
        .try_into()
        .map_err(|_| CheckpointError::Format(format!("{shard}: the file ends inside its header size")))?;
    let length = u64::from_le_bytes(bytes);
    if length == 0 || length > MAX_HEADER_BYTES {
        return Err(CheckpointError::Format(format!(
            "{shard}: bad header size {}",
            length as i64
        )));
    }
    Ok(length)
}

/// The tensors a shard's JSON header describes; `shard` is the index of the shard and `name` its file name.
pub fn parse_header(name: &str, shard: usize, header: &[u8]) -> Result<Vec<Tensor>, CheckpointError> {
    let bad = |what: String| CheckpointError::Format(format!("{name}: {what}"));
    let root: Value =
        serde_json::from_slice(header).map_err(|error| bad(format!("the header is not JSON: {error}")))?;
    let entries = root
        .as_object()
        .ok_or_else(|| bad("the header is not an object".to_owned()))?;
    let data = 8 + header.len() as u64;
    let mut tensors = Vec::with_capacity(entries.len());
    for (tensor, node) in entries {
        if tensor == METADATA {
            continue;
        }
        let dtype = node
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(format!("{tensor} has no dtype")))?;
        let shape = node
            .get("shape")
            .and_then(Value::as_array)
            .and_then(|dimensions| dimensions.iter().map(Value::as_u64).collect::<Option<Vec<u64>>>())
            .ok_or_else(|| bad(format!("{tensor} has no shape of non-negative sizes")))?;
        let (begin, end) = match node.get("data_offsets").and_then(Value::as_array).map(Vec::as_slice) {
            Some([begin, end]) => (begin.as_u64(), end.as_u64()),
            _ => (None, None),
        };
        let (Some(begin), Some(end)) = (begin, end) else {
            return Err(bad(format!("{tensor} has no data_offsets")));
        };
        if end < begin {
            return Err(bad(format!("{tensor} ends at {end}, before its start {begin}")));
        }
        tensors.push(Tensor {
            name: tensor.clone(),
            dtype: dtype.to_owned(),
            shape,
            shard,
            offset: data + begin,
            bytes: end - begin,
        });
    }
    Ok(tensors)
}

impl Safetensors {
    /// Opens the checkpoint in `directory`: its index and the header of each shard are read, and `then` gets the
    /// checkpoint. `Err` started nothing.
    pub fn open(io: &Arc<AsyncIo>, directory: &Path, then: Done<Safetensors>) -> Result<(), CheckpointError> {
        let index = directory.join(INDEX);
        let directory = directory.to_path_buf();
        let next_io = Arc::clone(io);
        read_file(
            io,
            &index,
            Arc::new(move |bytes| {
                let names = match bytes {
                    Ok(bytes) => shard_names(&bytes),
                    Err(CheckpointError::Io { source, .. }) if source.kind() == ErrorKind::NotFound => {
                        Ok(vec![SINGLE.to_owned()])
                    }
                    Err(error) => Err(error),
                };
                match names {
                    Ok(names) => Opening {
                        io: Arc::clone(&next_io),
                        directory: directory.clone(),
                        queue: names.into(),
                        shards: Vec::new(),
                        tensors: HashMap::new(),
                        then: Arc::clone(&then),
                    }
                    .advance(),
                    Err(error) => then(Err(error)),
                }
            }),
        )
    }

    /// The tensor called `name`.
    pub fn tensor(&self, name: &str) -> Result<&Tensor, CheckpointError> {
        self.tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_owned()))
    }

    /// The names of all tensors, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The file name of the shard that holds `tensor`.
    pub fn shard_of(&self, tensor: &Tensor) -> &str {
        &self.shards[tensor.shard].name
    }

    /// Reads `memory.len()` bytes of tensor `name` starting `skip` bytes into it, into `memory`.
    pub fn read<B: IoMemory>(
        self: &Arc<Self>,
        io: &Arc<AsyncIo>,
        name: &str,
        skip: u64,
        memory: B,
        then: Done<B>,
    ) -> Result<(), CheckpointError> {
        let ranges = vec![ReadRange {
            tensor: name.to_owned(),
            skip,
            memory,
        }];
        self.read_ranges(
            io,
            ranges,
            Arc::new(move |result| then(result.map(|mut all| all.remove(0)))),
        )
    }

    /// Reads all of tensor `name` into `memory`, which must be exactly its size.
    pub fn read_tensor<B: IoMemory>(
        self: &Arc<Self>,
        io: &Arc<AsyncIo>,
        name: &str,
        memory: B,
        then: Done<B>,
    ) -> Result<(), CheckpointError> {
        let tensor = self.tensor(name)?;
        if memory.len() as u64 != tensor.bytes {
            return Err(CheckpointError::Invalid(format!(
                "{name} holds {} bytes, not {}",
                tensor.bytes,
                memory.len()
            )));
        }
        self.read(io, name, 0, memory, then)
    }

    /// Reads several ranges at once; `then` gets their memories in the order of `ranges`. Every range is checked first.
    /// `Err` started nothing. A failure after the first read began reaches `then`, once every read has ended.
    pub fn read_ranges<B: IoMemory>(
        self: &Arc<Self>,
        io: &Arc<AsyncIo>,
        ranges: Vec<ReadRange<B>>,
        then: Done<Vec<B>>,
    ) -> Result<(), CheckpointError> {
        if ranges.is_empty() {
            return Err(CheckpointError::Invalid("a read of no ranges".to_owned()));
        }
        let mut places = Vec::with_capacity(ranges.len());
        for range in &ranges {
            let tensor = self.tensor(&range.tensor)?;
            let len = range.memory.len() as u64;
            if len == 0 || range.skip.checked_add(len).is_none_or(|end| end > tensor.bytes) {
                return Err(CheckpointError::Invalid(format!(
                    "{}: range {}+{} outside {} bytes",
                    range.tensor, range.skip, len, tensor.bytes
                )));
            }
            places.push((tensor.shard, tensor.offset + range.skip));
        }
        let count = ranges.len();
        let gather = Arc::new(Gather {
            slots: (0..count).map(|_| Slot(OnceLock::new())).collect(),
            lengths: ranges.iter().map(|range| range.memory.len()).collect(),
            names: ranges.iter().map(|range| range.tensor.clone()).collect(),
            then,
        });
        for (index, (range, (shard, offset))) in ranges.into_iter().zip(places).enumerate() {
            let token = Arc::clone(&gather);
            let store = Arc::clone(self);
            let submitted = io.read_into(
                raw_file(&self.shards[shard].file),
                offset,
                range.memory,
                Box::new(move |done| {
                    let _open = store;
                    match complete(token, index, Outcome::Done(done)) {
                        Some(finished) => work(move || finished.deliver()),
                        None => idle(),
                    }
                }),
            );
            match submitted {
                Ok(()) => {}
                Err(error) if index == 0 => return Err(io::submit_error(&range.tensor, error)),
                Err(error) => {
                    // The reads already submitted still end. The ones that were not are failed here, and the last
                    // reference to the gather delivers.
                    for rest in index..count {
                        let reason =
                            CheckpointError::io(&gather.names[rest], IoError::new(error.kind(), error.to_string()));
                        let _ = gather.slots[rest].0.set(Outcome::Failed(reason));
                    }
                    break;
                }
            }
        }
        // The reads that already ended may have left this reference as the last.
        if let Some(finished) = Arc::into_inner(gather) {
            finished.deliver();
        }
        Ok(())
    }
}

/// How one range ended.
enum Outcome<B> {
    Done(IoDoneOwned<B>),
    Failed(CheckpointError),
}

/// A cell written once, by the read that owns the index, and emptied by whoever holds the last reference.
struct Slot<T>(OnceLock<T>);

// SAFETY: a slot is only ever set (through `OnceLock`'s own synchronisation) and later taken by value once the last
// reference to the gather is the only one left (`Arc::into_inner`); no reference to its content is ever shared, so
// `T: Send` is all it takes to move the content between threads.
unsafe impl<T: Send> Sync for Slot<T> {}

/// The reads of one `read_ranges`: their outcomes, and where the result goes. Each read holds one `Arc` of it; the
/// one that drops the last delivers.
struct Gather<B> {
    slots: Vec<Slot<Outcome<B>>>,
    lengths: Vec<usize>,
    names: Vec<String>,
    then: Done<Vec<B>>,
}

/// Records one outcome and gives up the reference; the gather comes back to whoever held the last.
fn complete<B>(token: Arc<Gather<B>>, index: usize, outcome: Outcome<B>) -> Option<Gather<B>> {
    let _ = token.slots[index].0.set(outcome);
    Arc::into_inner(token)
}

impl<B> Gather<B> {
    fn deliver(self) {
        let Gather {
            slots,
            lengths,
            names,
            then,
        } = self;
        let mut memories = Vec::with_capacity(slots.len());
        let mut failure = None;
        for (index, slot) in slots.into_iter().enumerate() {
            let result = match slot.0.into_inner() {
                Some(Outcome::Done(done)) => io::finish(done, lengths[index], &names[index]),
                Some(Outcome::Failed(error)) => Err(error),
                None => Err(CheckpointError::Invalid(format!(
                    "{}: the read never ended",
                    names[index]
                ))),
            };
            match result {
                Ok(memory) => memories.push(memory),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        then(match failure {
            Some(error) => Err(error),
            None => Ok(memories),
        });
    }
}

/// The chain that reads each shard's header, one after the other.
struct Opening {
    io: Arc<AsyncIo>,
    directory: PathBuf,
    queue: VecDeque<String>,
    shards: Vec<Shard>,
    tensors: HashMap<String, Tensor>,
    then: Done<Safetensors>,
}

impl Opening {
    /// Opens the next shard, or hands over the checkpoint when none is left.
    fn advance(mut self) {
        let Some(name) = self.queue.pop_front() else {
            let store = Safetensors {
                tensors: self.tensors,
                shards: self.shards,
            };
            (self.then)(Ok(store));
            return;
        };
        let path = self.directory.join(&name);
        let io = Arc::clone(&self.io);
        let failed = Arc::clone(&self.then);
        let next: Next<OwnedFile> = Box::new(move |opened| match opened {
            Ok(file) => self.opened(name, Arc::new(file)),
            Err(error) => (self.then)(Err(error)),
        });
        if let Err(error) = open_read(&io, &path, next) {
            report(&failed, error);
        }
    }

    /// Reads the 8 bytes that give the header's length.
    fn opened(self, name: String, file: Arc<OwnedFile>) {
        let io = Arc::clone(&self.io);
        let failed = Arc::clone(&self.then);
        let again = Arc::clone(&file);
        let next: Next<Vec<u8>> = Box::new(move |prefix| match prefix {
            Ok(prefix) => self.sized(name, file, &prefix),
            Err(error) => (self.then)(Err(error)),
        });
        if let Err(error) = read_exact(&io, again, 0, 8, "a shard's header size".to_owned(), next) {
            report(&failed, error);
        }
    }

    /// Reads the header.
    fn sized(self, name: String, file: Arc<OwnedFile>, prefix: &[u8]) {
        let length = match header_length(&name, prefix).and_then(|length| {
            usize::try_from(length).map_err(|_| CheckpointError::Format(format!("{name}: header too large")))
        }) {
            Ok(length) => length,
            Err(error) => {
                (self.then)(Err(error));
                return;
            }
        };
        let io = Arc::clone(&self.io);
        let failed = Arc::clone(&self.then);
        let again = Arc::clone(&file);
        let context = format!("the header of {name}");
        let next: Next<Vec<u8>> = Box::new(move |header| match header {
            Ok(header) => self.parsed(name, file, &header),
            Err(error) => (self.then)(Err(error)),
        });
        if let Err(error) = read_exact(&io, again, 8, length, context, next) {
            report(&failed, error);
        }
    }

    /// Takes the shard's tensors and goes on to the next shard.
    fn parsed(mut self, name: String, file: Arc<OwnedFile>, header: &[u8]) {
        match parse_header(&name, self.shards.len(), header) {
            Ok(tensors) => {
                for tensor in tensors {
                    self.tensors.insert(tensor.name.clone(), tensor);
                }
                self.shards.push(Shard { name, file });
                self.advance();
            }
            Err(error) => (self.then)(Err(error)),
        }
    }
}
