//! The tokenizer the artifact packages, read from its tokenizer section: the same tokenizer the engine's prompts are
//! encoded with, so the benchmark needs no checkpoint directory.

use std::fmt::Display;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use euhedral_inference_edrl::{FOOTER_BYTES, ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::Tokenizer;

use crate::config::ConfigError;

fn read_range(file: &mut File, range: Range<u64>) -> io::Result<Vec<u8>> {
    file.seek(SeekFrom::Start(range.start))?;
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

/// Reads the tokenizer of the artifact at `path`.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: the tokenizer section is read once, on the main thread, before the engine starts"
)]
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer, ConfigError> {
    let fail = |what: &str, error: &dyn Display| ConfigError::Invalid(format!("{}: {what}: {error}", path.display()));
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
    let section = TokenizerSection::parse(body, &footer).map_err(|error| fail("bad tokenizer section", &error))?;
    Tokenizer::from_section(&section).map_err(|error| fail("cannot build the tokenizer", &error))
}
