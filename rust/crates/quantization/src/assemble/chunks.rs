//! The read of one chunk of a file into a buffer that is used again, so that copying and hashing a large payload
//! allocate once.

use std::sync::Arc;

use euhedral_io::{AsyncIo, IoBuf, IoDone, OwnedFile};

use super::AssembleError;
use crate::checkpoint::CheckpointError;
use crate::checkpoint::io::{raw_file, submit_error};
use crate::step::{Kind, step};

/// What follows a chunk read: the buffer, holding exactly the chunk, or why it could not be read.
pub(crate) type Chunk = Box<dyn FnOnce(Result<Vec<u8>, AssembleError>) + Send>;

/// Reads `length` bytes (at least one) at `offset` of `file` into `buffer`, which is resized to `length` first: its
/// capacity is kept, so a buffer that has held a longer chunk allocates nothing. A file that ends first is an error.
/// `Err` started nothing and drops `then`. `file` stays open until `then` runs.
pub(crate) fn read_chunk(
    io: &AsyncIo,
    file: Arc<OwnedFile>,
    offset: u64,
    length: usize,
    mut buffer: Vec<u8>,
    context: String,
    then: Chunk,
) -> Result<(), AssembleError> {
    buffer.resize(length, 0);
    let raw = raw_file(&file);
    let failed = context.clone();
    io.read(
        raw,
        offset,
        IoBuf {
            bytes: buffer,
            filled: 0,
        },
        Box::new(move |done: IoDone| {
            step(Kind::Resolve, move || {
                let _open = file;
                then(match done.result {
                    Ok(moved) if moved == length => Ok(done.buf.bytes),
                    Ok(moved) => Err(CheckpointError::Format(format!(
                        "{context}: the file ends after {moved} of {length} bytes"
                    ))
                    .into()),
                    Err(error) => Err(CheckpointError::io(format!("cannot read {context}"), error).into()),
                });
            })
        }),
    )
    .map_err(|error| submit_error(&failed, error).into())
}
