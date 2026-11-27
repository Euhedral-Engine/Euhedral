//! Opens the real Flash-Next artifact and validates its inventory against its own configuration, reading only the
//! header, the footer, the metadata, the tables and the tokenizer section (never a tensor payload). Needs
//! `euhedral.flash-next.artifact` (in `~/.config/euhedral/test.properties` or `~/.gradle/gradle.properties`, or
//! `EUHEDRAL_TEST_FLASH_NEXT_ARTIFACT`), and skips naming the key without it.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
#![allow(
    clippy::print_stdout,
    reason = "a test prints its skip line and the counts it found when the machine has the artifact"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test reads a file's regions with blocking I/O: the reader itself performs none"
)]

use std::fs::File;
use std::io::{Error, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use euhedral_inference_edrl::{EdrlError, FlashArtifact, ModelFamily};

const KEY: &str = "euhedral.flash-next.artifact";

#[test]
fn the_real_artifact_opens_and_its_inventory_validates() {
    // `require_property` prints the skip line, naming the key, when it is unset or names no path.
    let Some(path) = test_support::require_property(KEY) else {
        return;
    };
    let path = PathBuf::from(path);
    let mut file = File::open(&path).expect("the artifact opens");
    let file_len = file.metadata().expect("the artifact's length").len();
    let mut head = [0; euhedral_inference_edrl::DETECT_BYTES];
    file.read_exact(&mut head).expect("the artifact's first bytes");
    assert_eq!(ModelFamily::detect(&head), Ok(ModelFamily::FlashNextV3));
    let mut read = |range: Range<u64>| {
        let io = |error: Error| EdrlError::Invalid {
            what: format!("reading the artifact failed: {error}"),
        };
        let length = usize::try_from(range.end - range.start).expect("a region fits in memory");
        let mut bytes = vec![0; length];
        file.seek(SeekFrom::Start(range.start)).map_err(io)?;
        file.read_exact(&mut bytes).map_err(io)?;
        Ok(bytes)
    };
    let artifact = FlashArtifact::open(file_len, &mut read).expect("the real artifact is valid");
    let report = artifact.report();
    let tokenizer = artifact
        .tokenizer()
        .expect("the real artifact carries a tokenizer section");
    let tokenizer_bytes: usize = euhedral_inference_edrl::TOKENIZER_ENTRIES
        .iter()
        .map(|name| tokenizer.file(name).expect("every tokenizer entry").len())
        .sum();
    println!(
        "flash-next artifact: {} fixed objects, {} expert banks, {} expert records, {} layers, content end {}, tokenizer {} bytes",
        report.tensors,
        report.banks,
        report.expert_records,
        artifact.config().text.num_layers,
        artifact.header().file_size(),
        tokenizer_bytes,
    );
}
