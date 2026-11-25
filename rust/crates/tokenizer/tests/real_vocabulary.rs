//! The tokenizer's behaviour on the real vocabulary of an artifact: text round-trips, every control token encodes to
//! its own single id, encoding is repeatable and a split character is held back. Needs `euhedral.qwen.rco-artifact`
//! (in `~/.gradle/gradle.properties`, or `EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`); reads only the header, the footer and the
//! tokenizer body, never a payload.
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
    clippy::print_stdout,
    reason = "a test prints its skip line when the machine has no artifact"
)]

use std::path::PathBuf;

use euhedral_inference_tokenizer::{Tokenizer, read_artifact_tokenizer};

const KEY: &str = "euhedral.qwen.rco-artifact";

/// Text of every kind the engine meets: multilingual, emoji, no-break spaces, runs of whitespace, control
/// characters, numbers, contractions and the chat markers spelled out.
const TEXTS: [&str; 16] = [
    "Hello, world!",
    "  repeated   spaces",
    "line one\nline two\n",
    "café 中文 😀",
    "A🙂B",
    "naïve é",
    "ÿ\u{a0}\t\r\n",
    "a...!!!??? 12345",
    "don't We're I'll",
    "한국어 العربية русский",
    "zero-width joiners 👨\u{200d}👩\u{200d}👧 and \u{1f1e9}\u{1f1ea}",
    "\u{0}\u{1}\u{7}\u{1b}[0m\u{7f}",
    "<think>thinking</think>",
    "<|im_start|>user\nHi<|im_end|>",
    "The quick brown fox jumps over the lazy dog.",
    "tab\there\u{2028}separator\u{3000}ideographic space",
];

/// The artifact's tokenizer, or `None` (with a skip line) when the machine has no artifact.
fn real_tokenizer() -> Option<Tokenizer> {
    let path = PathBuf::from(test_support::require_property(KEY)?);
    if !path.is_file() {
        println!("skipped: {KEY} names no file");
        return None;
    }
    Some(read_artifact_tokenizer(&path).unwrap())
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn text_of_every_kind_round_trips() {
    let Some(tokenizer) = real_tokenizer() else { return };
    for text in TEXTS {
        let ids = tokenizer.encode(text).unwrap();
        assert!(!ids.is_empty(), "{text:?} encodes to something");
        assert_eq!(tokenizer.decode(&ids).unwrap(), text, "{text:?}");
        // The stream gives the same text as the whole decode, whatever the split.
        let mut stream = tokenizer.decode_stream();
        let mut streamed = String::new();
        for id in &ids {
            streamed.push_str(&stream.push(*id).unwrap());
        }
        streamed.push_str(&stream.finish());
        assert_eq!(streamed, text, "{text:?} streamed");
    }
    assert!(tokenizer.encode("").unwrap().is_empty());
    assert_eq!(tokenizer.decode(&[]).unwrap(), "");
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn each_control_token_encodes_to_its_single_id() {
    let Some(tokenizer) = real_tokenizer() else { return };
    let mut seen = 0;
    for id in 0..tokenizer.vocabulary_size() as u32 {
        let Some(text) = tokenizer.control_token_text(id) else {
            continue;
        };
        seen += 1;
        assert!(tokenizer.is_control_token(id));
        assert_eq!(tokenizer.control_token_id(text), Some(id), "{text}");
        assert_eq!(tokenizer.encode(text).unwrap(), [id], "{text} encodes to one id");
        assert_eq!(tokenizer.decode(&[id]).unwrap(), text, "{text} decodes to its spelling");
    }
    assert!(seen > 0, "the vocabulary has control tokens");
    // The chat markers are among them, and the stop tokens are control tokens too.
    for marker in ["<|im_start|>", "<|im_end|>", "<think>", "</think>"] {
        assert!(tokenizer.control_token_id(marker).is_some(), "{marker}");
    }
    for eos in tokenizer.eos_ids() {
        assert!(tokenizer.is_control_token(*eos), "stop id {eos}");
    }
    assert!(tokenizer.eos_ids().contains(&tokenizer.eos_id()));
    // The checkpoint enables neither BOS nor EOS, so model special tokens add nothing.
    assert_eq!(
        tokenizer.encode_with_model_special_tokens("hello").unwrap(),
        tokenizer.encode("hello").unwrap()
    );
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn encoding_is_the_same_on_a_second_call() {
    let Some(tokenizer) = real_tokenizer() else { return };
    for text in TEXTS {
        assert_eq!(
            tokenizer.encode(text).unwrap(),
            tokenizer.encode(text).unwrap(),
            "{text:?}"
        );
    }
}

/// The id of the ordinary token that is exactly the byte `byte`.
fn single_byte_token(tokenizer: &Tokenizer, byte: u8) -> u32 {
    (0..tokenizer.vocabulary_size() as u32)
        .find(|id| tokenizer.token_bytes(*id) == Some(&[byte]))
        .expect("a byte-level vocabulary has every single byte")
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn a_character_split_over_tokens_is_held_back_until_complete() {
    let Some(tokenizer) = real_tokenizer() else { return };
    let emoji = "😀";
    let pieces: Vec<u32> = emoji.bytes().map(|byte| single_byte_token(&tokenizer, byte)).collect();
    assert_eq!(pieces.len(), 4);

    // Fed one byte at a time, the stream holds the character back until the last byte completes it.
    let mut stream = tokenizer.decode_stream();
    for id in &pieces[..3] {
        assert_eq!(stream.push(*id).unwrap(), "");
    }
    assert_eq!(stream.push(pieces[3]).unwrap(), emoji);
    assert_eq!(stream.finish(), "");
    assert_eq!(tokenizer.decode(&pieces).unwrap(), emoji);

    // A truncated suffix becomes U+FFFD at the end, before a following character, or before a control token.
    let letter = single_byte_token(&tokenizer, b'A');
    let end = tokenizer.control_token_id("<|im_end|>").unwrap();
    assert_eq!(tokenizer.decode(&pieces[..1]).unwrap(), "\u{fffd}");
    assert_eq!(tokenizer.decode(&[pieces[0], letter]).unwrap(), "\u{fffd}A");
    assert_eq!(tokenizer.decode(&[pieces[0], end]).unwrap(), "\u{fffd}<|im_end|>");
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.push(pieces[0]).unwrap(), "");
    assert_eq!(stream.push(end).unwrap(), "\u{fffd}<|im_end|>");
    assert_eq!(stream.finish(), "");
}
