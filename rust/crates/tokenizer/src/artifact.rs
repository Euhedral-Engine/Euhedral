//! The tokenizer an artifact file packages, read from its tokenizer section.

use std::fmt::Display;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use euhedral_inference_edrl::{FOOTER_BYTES, ReadPlan, Region, TokenizerFooter, TokenizerSection};

use crate::Tokenizer;

/// Why the artifact's tokenizer could not be read: which step failed, and why.
#[derive(Debug, thiserror::Error)]
#[error("{what}: {reason}")]
pub struct ArtifactTokenizerError {
    /// The step that failed.
    pub what: &'static str,
    /// Why, in the words of the failing layer.
    pub reason: String,
}

fn fail(what: &'static str, reason: &dyn Display) -> ArtifactTokenizerError {
    ArtifactTokenizerError {
        what,
        reason: reason.to_string(),
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "blocking I/O at startup: the tokenizer section is read once, before the engine starts"
)]
fn read_range(file: &mut File, range: Range<u64>) -> io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(range.start))?;
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Reads the tokenizer of the artifact at `path`: the one the engine's prompts are encoded with. Reads the file
/// synchronously, so call it on a thread that may wait, before the engine starts.
pub fn read_artifact_tokenizer(path: &Path) -> Result<Tokenizer, ArtifactTokenizerError> {
    let section = read_artifact_section(path)?;
    Tokenizer::from_section(&section).map_err(|error| fail("cannot build the tokenizer", &error))
}

/// Reads the tokenizer section of the artifact at `path`. Reads the file synchronously, so call it on a thread that
/// may wait, before the engine starts.
#[allow(
    clippy::disallowed_methods,
    reason = "blocking I/O at startup: the tokenizer section is read once, before the engine starts"
)]
pub fn read_artifact_section(path: &Path) -> Result<TokenizerSection, ArtifactTokenizerError> {
    let mut file = File::open(path).map_err(|error| fail("cannot open the artifact", &error))?;
    let length = file
        .metadata()
        .map_err(|error| fail("cannot read the artifact's size", &error))?
        .len();
    let probe = ReadPlan::probe(length).map_err(|error| fail("not an artifact", &error))?;
    let footer_range = probe
        .range(Region::TokenizerFooter)
        .ok_or_else(|| fail("no tokenizer footer", &"the read plan names none"))?;
    let footer_bytes =
        read_range(&mut file, footer_range).map_err(|error| fail("cannot read the tokenizer footer", &error))?;
    let footer_bytes: [u8; FOOTER_BYTES] = footer_bytes
        .as_slice()
        .try_into()
        .map_err(|_| fail("cannot read the tokenizer footer", &"wrong number of bytes"))?;
    let footer = TokenizerFooter::parse(&footer_bytes)
        .map_err(|error| fail("bad tokenizer footer", &error))?
        .ok_or_else(|| fail("no tokenizer", &"the artifact carries no tokenizer section"))?;
    let range = footer
        .body_range(length)
        .map_err(|error| fail("bad tokenizer section", &error))?;
    let body = read_range(&mut file, range).map_err(|error| fail("cannot read the tokenizer section", &error))?;
    TokenizerSection::parse(body, &footer).map_err(|error| fail("bad tokenizer section", &error))
}
