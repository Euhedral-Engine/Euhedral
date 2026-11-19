//! The `quantization package-tokenizer` command, run as a process on a tiny artifact and a synthetic checkpoint.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use euhedral_inference_edrl::tiny::TinyArtifact;
use euhedral_inference_edrl::{FOOTER_BYTES, TokenizerFooter};
use serde_json::json;

const TEMPLATE: &str = include_str!("../../api-protocol/tests/data/qwen-chat-template.jinja");

/// The expression the Qwen checkpoints split with.
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The byte-level spelling of every byte.
fn byte_chars() -> Vec<char> {
    let mut next = 256;
    (0..=255u32)
        .map(|byte| {
            if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                char::from_u32(byte).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            }
        })
        .collect()
}

fn tokenizer_json(kind: &str) -> String {
    let mut vocab = serde_json::Map::new();
    for (byte, character) in byte_chars().iter().enumerate() {
        vocab.insert(character.to_string(), json!(byte));
    }
    vocab.insert("he".into(), json!(256));
    let added = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
               "normalized": false, "special": true})
    };
    json!({
        "version": "1.0",
        "added_tokens": [added(257, "<|im_start|>"), added(258, "<|im_end|>"), added(259, "<|endoftext|>")],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        ]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {
            "type": kind, "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": ["h e"],
        },
    })
    .to_string()
}

fn write_checkpoint(directory: &Path) {
    fs::create_dir_all(directory).unwrap();
    fs::write(directory.join("tokenizer.json"), tokenizer_json("BPE")).unwrap();
    fs::write(
        directory.join("tokenizer_config.json"),
        json!({"eos_token": "<|im_end|>", "bos_token": null, "add_bos_token": false}).to_string(),
    )
    .unwrap();
    fs::write(
        directory.join("generation_config.json"),
        json!({"bos_token_id": 259, "eos_token_id": [258, 259], "pad_token_id": 259}).to_string(),
    )
    .unwrap();
    fs::write(directory.join("chat_template.jinja"), TEMPLATE).unwrap();
}

fn quantization(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_quantization"))
        .args(args)
        .env_remove("EUHEDRAL_CONFIG")
        .output()
        .expect("the tool starts")
}

#[test]
fn the_command_packages_in_place_and_to_an_output_with_the_same_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = directory.path().join("checkpoint");
    write_checkpoint(&checkpoint);
    // The tiny artifact carries a tokenizer section of its own; the content is what precedes it.
    let mut content = TinyArtifact::new(3, true).bytes().unwrap();
    let footer = content.split_off(content.len() - FOOTER_BYTES);
    let footer = TokenizerFooter::parse(footer.as_slice().try_into().unwrap())
        .unwrap()
        .expect("a footer");
    content.truncate(content.len() - footer.body_length() as usize);
    let artifact = directory.path().join("a.edrl");
    let twin = directory.path().join("twin.edrl");
    let out = directory.path().join("out.edrl");
    fs::write(&artifact, &content).unwrap();
    fs::write(&twin, &content).unwrap();

    let checkpoint = checkpoint.to_str().unwrap();
    let output = quantization(&[
        "package-tokenizer",
        "--artifact",
        artifact.to_str().unwrap(),
        "--tokenizer",
        checkpoint,
        "--out",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(fs::read(&artifact).unwrap(), content, "the source is left alone");

    let output = quantization(&[
        "package-tokenizer",
        "--artifact",
        twin.to_str().unwrap(),
        "--tokenizer",
        checkpoint,
    ]);
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let packaged = fs::read(&out).unwrap();
    assert!(packaged.len() > content.len());
    assert_eq!(packaged, fs::read(&twin).unwrap());
    assert!(packaged.starts_with(&content));
    assert!(!directory.path().join("out.edrl.part").exists());
}

#[test]
fn a_checkpoint_without_a_tokenizer_fails_with_a_message_and_a_nonzero_code() {
    let directory = tempfile::tempdir().unwrap();
    let artifact = directory.path().join("a.edrl");
    fs::write(&artifact, TinyArtifact::new(3, true).bytes().unwrap()).unwrap();
    let before = fs::read(&artifact).unwrap();
    let output = quantization(&[
        "package-tokenizer",
        "--artifact",
        artifact.to_str().unwrap(),
        "--tokenizer",
        directory.path().to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("package-tokenizer failed"));
    assert_eq!(fs::read(&artifact).unwrap(), before);
}
