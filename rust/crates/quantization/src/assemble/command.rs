//! The `assemble` command: its flags, and the chain that reads the allocation, opens the sources, selects the objects
//! and writes the artifact.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_edrl::MAX_DRAFT_DEPTH;
use euhedral_io::AsyncIo;

use super::sources::SourceSet;
use super::view::view;
use super::write::write;
use super::{AssembleError, Done};
use crate::checkpoint::io::read_file;
use crate::mixed::{Allocation, QuantOption};

/// What the command was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub allocation: PathBuf,
    pub sources: BTreeMap<QuantOption, PathBuf>,
    pub base: QuantOption,
    pub out: PathBuf,
    /// The draft depth to store; the base's stays when absent.
    pub draft_depth: Option<u8>,
    /// The size of the copy buffer, when the command line gives one.
    pub copy_buffer_mib: Option<u64>,
}

/// What a finished run reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Assembled {
    /// The payload bytes the written artifact holds.
    pub resident_bytes: u64,
}

impl Request {
    /// The request in `args`, the flags after the command's name: `--allocation FILE`, `--source OPTION=ARTIFACT`
    /// (repeated), `--base OPTION`, `--out ARTIFACT`, and optionally `--draft-depth N` and `--copy-buffer-mib N`.
    pub fn parse(args: &[String]) -> Result<Request, AssembleError> {
        let mut allocation = None;
        let mut sources = BTreeMap::new();
        let mut base = None;
        let mut out = None;
        let mut draft_depth = None;
        let mut copy_buffer_mib = None;
        let mut words = args.iter();
        while let Some(flag) = words.next() {
            let mut value = || {
                words
                    .next()
                    .ok_or_else(|| AssembleError::invalid(format!("{flag} needs a value")))
            };
            match flag.as_str() {
                "--allocation" => allocation = Some(PathBuf::from(value()?)),
                "--out" => out = Some(PathBuf::from(value()?)),
                "--base" => base = Some(QuantOption::of(value()?)?),
                "--source" => {
                    let text = value()?;
                    let (key, path) = text
                        .split_once('=')
                        .ok_or_else(|| AssembleError::invalid(format!("--source {text} is not OPTION=ARTIFACT")))?;
                    if sources.insert(QuantOption::of(key)?, PathBuf::from(path)).is_some() {
                        return Err(AssembleError::invalid(format!("--source {key} is given twice")));
                    }
                }
                "--draft-depth" => {
                    let text = value()?;
                    let depth = text
                        .parse::<u8>()
                        .ok()
                        .filter(|depth| (1..=MAX_DRAFT_DEPTH).contains(depth))
                        .ok_or_else(|| {
                            AssembleError::invalid(format!("--draft-depth {text} is not 1 to {MAX_DRAFT_DEPTH}"))
                        })?;
                    draft_depth = Some(depth);
                }
                "--copy-buffer-mib" => {
                    let text = value()?;
                    let mib = text.parse::<u64>().ok().filter(|mib| *mib >= 1).ok_or_else(|| {
                        AssembleError::invalid(format!("--copy-buffer-mib {text} is not a count of at least 1"))
                    })?;
                    copy_buffer_mib = Some(mib);
                }
                other => return Err(AssembleError::invalid(format!("unknown flag {other}"))),
            }
        }
        let missing = |flag: &str| AssembleError::invalid(format!("{flag} is required"));
        let base = base.ok_or_else(|| missing("--base"))?;
        if !sources.contains_key(&base) {
            return Err(AssembleError::invalid(format!(
                "no --source for the base {}",
                base.key()
            )));
        }
        Ok(Request {
            allocation: allocation.ok_or_else(|| missing("--allocation"))?,
            sources,
            base,
            out: out.ok_or_else(|| missing("--out"))?,
            draft_depth,
            copy_buffer_mib,
        })
    }
}

/// The sizes of the reads and writes a run makes.
#[derive(Clone, Copy)]
struct Sizes {
    copy: usize,
    hash: usize,
}

/// Runs `request`: `done` gets the result once the artifact is complete under its final name. `chunk_bytes` (the
/// setting `host.copy-chunk-bytes`) is the size of the reads that hash the sources and, unless the request gives
/// `--copy-buffer-mib`, of the copy buffer. `Err` started nothing.
pub fn run(
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    request: Request,
    chunk_bytes: usize,
    done: Done<Assembled>,
) -> Result<(), AssembleError> {
    if chunk_bytes == 0 {
        return Err(AssembleError::invalid("the read size must be at least one byte"));
    }
    let copy = match request.copy_buffer_mib {
        Some(mib) => {
            usize::try_from(mib << 20).map_err(|_| AssembleError::invalid("the copy buffer does not fit in memory"))?
        }
        None => chunk_bytes,
    };
    let sizes = Sizes {
        copy,
        hash: chunk_bytes,
    };
    let request = Arc::new(request);
    let (stage_io, stage_host, stage_request, report) = (
        Arc::clone(io),
        Arc::clone(host),
        Arc::clone(&request),
        Arc::clone(&done),
    );
    read_file(
        io,
        &request.allocation,
        Arc::new(move |text| {
            let allocation = text
                .map_err(AssembleError::from)
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|_| AssembleError::invalid("the allocation is not UTF-8 text"))
                })
                .and_then(|text| Allocation::parse(&text).map_err(AssembleError::from));
            match allocation {
                Err(error) => report(Err(error)),
                Ok(allocation) => {
                    let started = open_sources(&stage_io, &stage_host, &stage_request, allocation, sizes, &report);
                    if let Err(error) = started {
                        report(Err(error));
                    }
                }
            }
        }),
    )
    .map_err(AssembleError::from)
}

fn open_sources(
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    request: &Arc<Request>,
    allocation: Allocation,
    sizes: Sizes,
    report: &Done<Assembled>,
) -> Result<(), AssembleError> {
    let (write_io, write_host, write_request, finish) = (
        Arc::clone(io),
        Arc::clone(host),
        Arc::clone(request),
        Arc::clone(report),
    );
    SourceSet::open(
        io,
        &request.sources,
        request.base,
        sizes.hash,
        Arc::new(move |sources| {
            let composite = sources.and_then(|sources| view(&sources, &allocation, write_request.draft_depth));
            match composite {
                Err(error) => finish(Err(error)),
                Ok(composite) => {
                    let assembled = Assembled {
                        resident_bytes: composite.resident_bytes(),
                    };
                    let after = Arc::clone(&finish);
                    let started = write(
                        &write_io,
                        &write_host,
                        &composite,
                        &write_request.out,
                        sizes.copy,
                        Arc::new(move |written| after(written.map(|()| assembled))),
                    );
                    if let Err(error) = started {
                        finish(Err(error));
                    }
                }
            }
        }),
    )
}
