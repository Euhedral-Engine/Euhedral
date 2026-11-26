//! The prompts at the large sizes, built with the tokenizer of a real artifact: a prompt has the token count asked for
//! (a chat prompt at most the target and most of it, a words prompt exactly the target), its ids are what the tokenizer
//! makes of its text, its digest is the digest of that text, and building it twice gives the same text. Needs
//! `euhedral.qwen.rco-artifact` (in `~/.config/euhedral/test.properties` or `~/.gradle/gradle.properties`, or `EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`); reads only
//! the tokenizer section.
//!
//! ```text
//! cargo test -p euhedral-inference-bench -- --ignored
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

use euhedral_inference_bench::prompt::words::DEFAULT_SEED;
use euhedral_inference_bench::prompt::{Prompt, PromptCorpus, chat, sha256_hex};
use euhedral_inference_tokenizer::{Tokenizer, read_artifact_tokenizer};

const KEY: &str = "euhedral.qwen.rco-artifact";

fn real_tokenizer() -> Option<Tokenizer> {
    let path = PathBuf::from(test_support::require_property(KEY)?);
    if !path.is_file() {
        println!("skipped: {KEY} names no file");
        return None;
    }
    Some(read_artifact_tokenizer(&path).unwrap())
}

/// What every prompt is, whatever its size: ids that are its text's encoding, and the digest of its text.
fn assert_consistent(prompt: &Prompt, tokenizer: &Tokenizer, what: &str) {
    assert_eq!(
        prompt.ids,
        tokenizer.encode(&prompt.text).unwrap(),
        "{what}: ids are the text's encoding"
    );
    assert_eq!(
        prompt.sha256,
        sha256_hex(&prompt.text),
        "{what}: the digest is of the text"
    );
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn chat_prompts_fill_most_of_a_large_target_and_are_repeatable() {
    let Some(tokenizer) = real_tokenizer() else { return };
    for target in [128, 2000, 4096, 16384, 32000] {
        let prompts = PromptCorpus::Chat.prompts(target, DEFAULT_SEED, &tokenizer).unwrap();
        assert_eq!(prompts.len(), chat::TASKS);
        for (task, prompt) in prompts.iter().enumerate() {
            let what = format!("chat target {target}, task {task}");
            assert!(prompt.ids.len() <= target, "{what}: {} tokens", prompt.ids.len());
            assert!(
                prompt.ids.len() * 10 >= target * 9,
                "{what}: {} tokens fill most of it",
                prompt.ids.len()
            );
            assert_consistent(prompt, &tokenizer, &what);
        }
        assert_eq!(
            PromptCorpus::Chat.prompts(target, DEFAULT_SEED, &tokenizer).unwrap(),
            prompts,
            "chat target {target}: a second build"
        );
        // The iteration picks the task, and wraps.
        let wrapped = PromptCorpus::Chat
            .prompt(target, chat::TASKS + 1, DEFAULT_SEED, &tokenizer)
            .unwrap();
        assert_eq!(wrapped, prompts[1]);
    }
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn words_prompts_reach_their_target_exactly_and_are_repeatable() {
    let Some(tokenizer) = real_tokenizer() else { return };
    for target in [32, 1024, 4096] {
        let prompt = PromptCorpus::Words.prompt(target, 0, DEFAULT_SEED, &tokenizer).unwrap();
        assert_eq!(prompt.ids.len(), target);
        assert_consistent(&prompt, &tokenizer, &format!("words target {target}"));
        let again = PromptCorpus::Words.prompt(target, 0, DEFAULT_SEED, &tokenizer).unwrap();
        assert_eq!(again, prompt, "words target {target}: a second build");
    }
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact)"]
fn a_words_prompt_of_128000_tokens_reaches_its_target_and_is_repeatable() {
    let Some(tokenizer) = real_tokenizer() else { return };
    let prompt = PromptCorpus::Words
        .prompt(128_000, 0, DEFAULT_SEED, &tokenizer)
        .unwrap();
    assert_eq!(prompt.ids.len(), 128_000);
    assert_consistent(&prompt, &tokenizer, "words target 128000");
    let again = PromptCorpus::Words
        .prompt(128_000, 0, DEFAULT_SEED, &tokenizer)
        .unwrap();
    assert_eq!(again, prompt);
}
