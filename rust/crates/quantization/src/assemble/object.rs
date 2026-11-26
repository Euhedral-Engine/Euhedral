//! One object of a source read whole into host memory, with the companions of a trellis object parsed and checked.

use std::collections::VecDeque;
use std::sync::Arc;

use euhedral_inference_edrl::{TensorEntry, TrellisCompanions, WeightFormat};
use euhedral_inference_engine::model::trellis_logical;
use euhedral_io::{AsyncIo, OwnedFile};

use super::chunks::read_chunk;
use super::{ArtifactFile, AssembleError, COMPANION_SUFFIXES, Done};

/// The companions of a trellis object as the device is given them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrellisBytes {
    /// The input scale vectors, one row per segment.
    pub suh: Vec<u8>,
    /// The output scale vector.
    pub svh: Vec<u8>,
    /// The segment ends as stored: little-endian `int32`, as the device reads them.
    pub segments: Vec<u8>,
    /// The segment ends parsed: strictly increasing multiples of 128, one per `suh` row, the last the object's rows.
    pub segment_ends: Arc<[u32]>,
    /// Rows of `suh`.
    pub suh_rows: u64,
}

/// An object's payload, and the companions of a trellis object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectBytes {
    pub bytes: Vec<u8>,
    /// Present for a trellis object, absent for every other format (whose companions, if any, are not read).
    pub trellis: Option<TrellisBytes>,
}

/// The payloads still to read, in the order of the object then [`COMPANION_SUFFIXES`], and those read.
struct Reading {
    io: Arc<AsyncIo>,
    file: Arc<OwnedFile>,
    name: String,
    pending: VecDeque<TensorEntry>,
    read: Vec<Vec<u8>>,
    /// Rows of the object and of its `suh`, for a trellis object.
    rows: Option<(u64, u64)>,
    done: Done<ObjectBytes>,
}

/// Reads the object `name` of `artifact` into memory through `io`, then calls `done` once from a frame of the lattice
/// that drives `io`: with the payload and, for a trellis object, its three companions in the order of
/// [`COMPANION_SUFFIXES`], their segment ends parsed and checked against the object's rows and the rows of `suh`.
///
/// `Err` started nothing and never calls `done`: the object is missing, or a trellis object lacks a companion or is
/// not stored as trellis tiles.
pub fn read_object(
    io: &Arc<AsyncIo>,
    artifact: &ArtifactFile,
    name: &str,
    done: Done<ObjectBytes>,
) -> Result<(), AssembleError> {
    let table = artifact.table();
    let main = table
        .get(name)
        .ok_or_else(|| AssembleError::invalid(format!("{} has no object {name}", artifact.path().display())))?;
    let mut pending = VecDeque::from([main.clone()]);
    let mut rows = None;
    if main.format() == WeightFormat::Trellis {
        let (n, _) = trellis_logical(main.shape())
            .ok_or_else(|| AssembleError::invalid(format!("{name} is not stored as trellis tiles")))?;
        for suffix in COMPANION_SUFFIXES {
            let companion = table
                .get(&format!("{name}{suffix}"))
                .ok_or_else(|| AssembleError::invalid(format!("the trellis object {name} lacks {suffix}")))?;
            pending.push_back(companion.clone());
        }
        let suh_rows = pending[1]
            .shape()
            .first()
            .copied()
            .ok_or_else(|| AssembleError::invalid(format!("{name}.suh has no rows")))?;
        rows = Some((n, suh_rows));
    }
    Reading {
        io: Arc::clone(io),
        file: Arc::clone(artifact.file()),
        name: name.to_owned(),
        pending,
        read: Vec::new(),
        rows,
        done,
    }
    .step();
    Ok(())
}

impl Reading {
    /// Reads the next payload, or finishes when none is left.
    fn step(mut self) {
        let Some(entry) = self.pending.pop_front() else {
            let done = Arc::clone(&self.done);
            return done(self.finish());
        };
        let done = Arc::clone(&self.done);
        let length = match usize::try_from(entry.bytes()) {
            Ok(length) if length > 0 => length,
            _ => {
                return done(Err(AssembleError::invalid(format!(
                    "{}: {} bytes",
                    entry.name(),
                    entry.bytes()
                ))));
            }
        };
        let (io, file, context) = (Arc::clone(&self.io), Arc::clone(&self.file), entry.name().to_owned());
        let started = read_chunk(
            &io,
            file,
            entry.offset(),
            length,
            Vec::new(),
            context,
            Box::new(move |chunk| match chunk {
                Ok(bytes) => {
                    self.read.push(bytes);
                    self.step();
                }
                Err(error) => (self.done)(Err(error)),
            }),
        );
        if let Err(error) = started {
            done(Err(error));
        }
    }

    fn finish(self) -> Result<ObjectBytes, AssembleError> {
        let mut parts = self.read.into_iter();
        let bytes = parts.next().unwrap_or_default();
        let Some((n, suh_rows)) = self.rows else {
            return Ok(ObjectBytes { bytes, trellis: None });
        };
        let (Some(suh), Some(svh), Some(segments)) = (parts.next(), parts.next(), parts.next()) else {
            return Err(AssembleError::invalid(format!(
                "the trellis object {} lacks a companion",
                self.name
            )));
        };
        let invalid = |reason: String| {
            AssembleError::invalid(format!("{}.segments: invalid trellis segments: {reason}", self.name))
        };
        let ends = TrellisCompanions::parse(&segments).map_err(|error| invalid(error.to_string()))?;
        TrellisCompanions::check_rows(&ends, n, suh_rows).map_err(|error| invalid(error.to_string()))?;
        Ok(ObjectBytes {
            bytes,
            trellis: Some(TrellisBytes {
                suh,
                svh,
                segments,
                segment_ends: Arc::from(ends),
                suh_rows,
            }),
        })
    }
}
