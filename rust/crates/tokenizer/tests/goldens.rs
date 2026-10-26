//! The Java tokenizer's reference vectors (`QwenTokenizerTest`), checked against the tokenizer section of a real
//! artifact. Needs `euhedral.qwen.rco-artifact` (in `~/.gradle/gradle.properties`, or
//! `EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`); reads only the header, the footer and the tokenizer body, never a payload.
//!
//! ```text
//! cargo test -p euhedral-inference-tokenizer -- --ignored
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test reads a fixture artifact"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test prints its skip line when the machine has no artifact"
)]

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;

use euhedral_inference_edrl::{ReadPlan, Region, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::Tokenizer;

const KEY: &str = "euhedral.qwen.rco-artifact";

fn read_range(file: &mut File, range: Range<u64>) -> Vec<u8> {
    file.seek(SeekFrom::Start(range.start)).unwrap();
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

/// The artifact's tokenizer, or `None` (with a skip line) when the machine has no artifact.
fn real_tokenizer() -> Option<Tokenizer> {
    let path = PathBuf::from(test_support::require_property(KEY)?);
    if !path.is_file() {
        println!("skipped: {KEY} names no file");
        return None;
    }
    let mut file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len();
    let probe = ReadPlan::probe(file_len).unwrap();
    let footer = TokenizerFooter::parse(
        read_range(&mut file, probe.range(Region::TokenizerFooter).unwrap())
            .as_slice()
            .try_into()
            .unwrap(),
    )
    .unwrap()
    .expect("the artifact carries its tokenizer");
    let body = read_range(&mut file, footer.body_range(file_len).unwrap());
    let section = TokenizerSection::parse(body, &footer).unwrap();
    Some(Tokenizer::from_section(&section).unwrap())
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn tokenizer_matches_java_goldens() {
    let Some(tokenizer) = real_tokenizer() else { return };
    let vectors: [(&str, &[u32]); 14] = [
        ("Hello, world!", &[9419, 11, 1814, 0]),
        ("  repeated   spaces", &[220, 11173, 256, 12258]),
        ("line one\nline two\n", &[1021, 799, 198, 1021, 1330, 198]),
        ("café 中文 😀", &[895, 56868, 220, 99986, 87209]),
        ("A🙂B", &[32, 169171, 33]),
        ("", &[]),
        ("naïve é", &[3267, 36328, 571, 3825]),
        // The gap after the ÿ is a no-break space (U+00A0).
        ("ÿ\u{a0}\t\r\n", &[127, 123, 3966, 4326]),
        ("a...!!!??? 12345", &[64, 1076, 11726, 31921, 220, 16, 17, 18, 19, 20]),
        ("don't We're I'll", &[14572, 914, 1165, 2224, 353, 3172]),
        ("한국어 العربية русский", &[174675, 30061, 171405, 211075]),
        ("<think>thinking</think>", &[248068, 79420, 248069]),
        ("<|im_start|>user\nHi<|im_end|>", &[248045, 846, 198, 12675, 248046]),
        (
            "The quick brown fox jumps over the lazy dog.",
            &[760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13],
        ),
    ];
    for (text, expected) in &vectors {
        assert_eq!(tokenizer.encode(text).unwrap(), *expected, "{text:?}");
    }
    // The checkpoint enables neither BOS nor EOS, so model special tokens add nothing.
    assert_eq!(
        tokenizer.encode_with_model_special_tokens("hello").unwrap(),
        tokenizer.encode("hello").unwrap()
    );
    assert_eq!(tokenizer.decode(&[1814]).unwrap(), " world");
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn special_tokens_match_java_goldens() {
    let Some(tokenizer) = real_tokenizer() else { return };
    assert_eq!(tokenizer.control_token_id("<|im_start|>"), Some(248045));
    assert_eq!(tokenizer.control_token_id("<|im_end|>"), Some(248046));
    assert_eq!(tokenizer.control_token_id("<think>"), Some(248068));
    assert_eq!(tokenizer.eos_id(), 248046);
    assert_eq!(tokenizer.eos_ids(), [248044, 248046]);
    assert_eq!(tokenizer.generation_bos_id(), Some(248044));
    assert_eq!(tokenizer.decode(&[248046]).unwrap(), "<|im_end|>");
    assert_eq!(tokenizer.decode(&[248044]).unwrap(), "<|endoftext|>");
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn decoding_matches_java_goldens() {
    let Some(tokenizer) = real_tokenizer() else { return };
    assert_eq!(
        tokenizer.decode(&[248045, 846, 198, 12675, 248046]).unwrap(),
        "<|im_start|>user\nHi<|im_end|>"
    );
    for text in ["Hello, world!", "café 中文 😀"] {
        assert_eq!(tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(), text);
    }

    // An emoji split over four tokens is held back until the last completes it.
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.push(172).unwrap(), "");
    assert_eq!(stream.push(253).unwrap(), "");
    assert_eq!(stream.push(246).unwrap(), "");
    assert_eq!(stream.push(222).unwrap(), "😀");
    assert_eq!(stream.finish(), "");
    assert_eq!(tokenizer.decode(&[172, 253, 246, 222]).unwrap(), "😀");

    // A truncated suffix becomes U+FFFD at the end, before a following character, or before a control token.
    assert_eq!(tokenizer.decode(&[172]).unwrap(), "\u{fffd}");
    assert_eq!(tokenizer.decode(&[172, 32]).unwrap(), "\u{fffd}A");
    assert_eq!(tokenizer.decode(&[172, 248046]).unwrap(), "\u{fffd}<|im_end|>");
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.push(172).unwrap(), "");
    assert_eq!(stream.push(248046).unwrap(), "\u{fffd}<|im_end|>");
    assert_eq!(stream.finish(), "");
}
