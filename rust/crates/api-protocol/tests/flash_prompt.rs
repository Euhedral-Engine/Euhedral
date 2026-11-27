//! The chat prompts the Java greedy references were recorded from, against what the server renders and encodes for
//! the same messages. Needs `euhedral.qwen4.artifact` (or `EUHEDRAL_TEST_QWEN4_ARTIFACT`); reads only the tokenizer
//! section, and skips without the key.

#![allow(clippy::expect_used, clippy::panic, clippy::indexing_slicing)]
#![allow(clippy::print_stdout, reason = "a test prints what it rendered and its skip line")]

use std::path::PathBuf;

use euhedral_inference_api_protocol::template::{Message, TemplateInput, TemplateOptions, render};
use euhedral_inference_tokenizer::{Tokenizer, read_artifact_section};

const KEY: &str = "euhedral.qwen4.artifact";

/// The user turn the Java greedy reference test sends, and the ids it recorded for its raw prompt.
const GOLDEN_USER: &str = "Explain, in a few short paragraphs, how a language model produces text one token at a time.";
const GOLDEN_IDS: [u32; 29] = [
    248_045, 846, 198, 814, 20139, 11, 303, 264, 2342, 2716, 41228, 11, 1204, 264, 3992, 1558, 18082, 1414, 799, 3817,
    506, 264, 854, 13, 248_046, 198, 248_045, 74455, 198,
];

/// The reference fixture's chat prompt "What is 17 times 23?", rendered with thinking off, and its ids.
const REFERENCE_USER: &str = "What is 17 times 23?";
const REFERENCE_IDS: [u32; 22] = [
    248_045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 248_046, 198, 248_045, 74_455, 198, 248_068, 271,
    248_069, 271,
];

fn tokenizer() -> Option<Tokenizer> {
    let path = PathBuf::from(test_support::require_property(KEY)?);
    let section = read_artifact_section(&path).expect("the artifact's tokenizer section");
    Some(Tokenizer::from_section(&section).expect("the artifact's tokenizer"))
}

fn rendered(text: &str, options: &TemplateOptions) -> String {
    let messages = vec![Message::user(text)];
    render(
        &TemplateInput {
            messages,
            tools: Vec::new(),
        },
        options,
    )
    .expect("renders")
}

fn encoded(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode_with_model_special_tokens(text).expect("encodes")
}

#[test]
fn the_reference_chat_message_renders_and_encodes_to_the_reference_prompt_ids() {
    let Some(tokenizer) = tokenizer() else { return };
    let text = rendered(REFERENCE_USER, &TemplateOptions::THINKING_DISABLED);
    assert_eq!(
        text,
        "<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    assert_eq!(encoded(&tokenizer, &text), REFERENCE_IDS);
}

#[test]
fn the_golden_prompt_is_the_templated_user_turn_without_the_think_block_the_template_adds() {
    let Some(tokenizer) = tokenizer() else { return };
    // Java encodes this text as it is; the template adds the empty think block of a thinking-off turn after it.
    let raw = format!("<|im_start|>user\n{GOLDEN_USER}<|im_end|>\n<|im_start|>assistant\n");
    assert_eq!(encoded(&tokenizer, &raw), GOLDEN_IDS);
    let templated = rendered(GOLDEN_USER, &TemplateOptions::THINKING_DISABLED);
    assert_eq!(templated, format!("{raw}<think>\n\n</think>\n\n"));
    let mut ids = GOLDEN_IDS.to_vec();
    ids.extend([248_068, 271, 248_069, 271]);
    assert_eq!(encoded(&tokenizer, &templated), ids);
}
