//! The reference logits file: the header, the tokens, and the records, on a synthetic file.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test writes the synthetic file"
)]

use std::fs;

use euhedral_inference_bench::kld::{self, KldError, Reference, Row};

const CONTEXT: usize = 8;
const VOCABULARY: usize = 5;

fn tokens() -> Vec<Vec<u32>> {
    (0..3)
        .map(|chunk| (0..CONTEXT as u32).map(|i| 10 * chunk + i).collect())
        .collect()
}

/// Record `step` of `chunk`: scale 1/4, minimum -16 + chunk, codes `100 * step + token`.
fn file() -> Vec<u8> {
    kld::encode(CONTEXT, VOCABULARY, &tokens(), |chunk, step| {
        (
            0.25,
            -16.0 + chunk as f32,
            (0..VOCABULARY).map(|token| (100 * step + token) as u16).collect(),
        )
    })
}

#[test]
fn kld_reader_parses_header_and_rows() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.kld");
    let bytes = file();
    fs::write(&path, &bytes).unwrap();

    let (reference, _file) = Reference::open(&path).unwrap();
    assert_eq!(
        (reference.context, reference.vocabulary, reference.chunks()),
        (CONTEXT, VOCABULARY, 3)
    );
    assert_eq!(reference.tokens, tokens());
    // The scored half of a chunk of 8: the prefix is 8 / 2 + 1 = 5 tokens, the 3 that follow are forced, and each is
    // the target of one record.
    assert_eq!((reference.prefix_len(), reference.records()), (5, 3));
    assert_eq!(reference.prefix(1), &[10, 11, 12, 13, 14]);
    assert_eq!(reference.forced(1), &[15, 16, 17]);
    // Two f32 and 6 codes (5 padded to an even count) of 2 bytes.
    assert_eq!(reference.row_bytes(), 8 + 12);
    assert_eq!(reference.row_offset(0, 0), 20 + 4 * 3 * 8);
    assert_eq!(reference.row_offset(2, 1), 20 + 4 * 3 * 8 + 20 * (2 * 3 + 1));
    assert_eq!(bytes.len() as u64, reference.row_offset(2, 3));

    let at = reference.row_offset(2, 1) as usize;
    let row = Row::new(&bytes[at..at + reference.row_bytes()], VOCABULARY);
    // chunk 2: minimum -14; step 1: codes 100 + token
    assert_eq!(row.log_p(0), 0.25 * 100.0 - 14.0);
    assert_eq!(row.log_p(4), 0.25 * 104.0 - 14.0);
}

#[test]
fn a_file_that_is_not_a_reference_or_is_cut_short_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("reference.kld");
    let bytes = file();

    let mut wrong = bytes.clone();
    wrong[0] = b'x';
    fs::write(&path, &wrong).unwrap();
    assert_eq!(Reference::open(&path).unwrap_err(), KldError::NotReference);

    fs::write(&path, &bytes[..30]).unwrap();
    assert!(matches!(
        Reference::open(&path).unwrap_err(),
        KldError::Truncated {
            what: "the chunks' tokens",
            ..
        }
    ));

    fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(matches!(
        Reference::open(&path).unwrap_err(),
        KldError::Truncated {
            what: "the records",
            ..
        }
    ));

    let mut negative = bytes;
    negative[8..12].copy_from_slice(&(-1i32).to_le_bytes());
    assert!(matches!(Reference::parse(&negative).unwrap_err(), KldError::Invalid(_)));
}
