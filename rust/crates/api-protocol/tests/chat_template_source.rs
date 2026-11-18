//! Which text of a tokenizer section is the chat template: the `chat_template.jinja` entry over the `chat_template`
//! string of `tokenizer_config.json`, and an error when neither holds one. The chosen text goes through the check
//! the server makes of its template.

#![allow(clippy::expect_used, clippy::panic)]

use euhedral_inference_api_protocol::template::check_template_source;
use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::{TokenizerError, chat_template};
use serde_json::json;

const CHECKPOINT_TEMPLATE: &str = include_str!("data/qwen-chat-template.jinja");

/// A section whose `tokenizer_config.json` is `config` and whose `chat_template.jinja` is `jinja`.
fn section(config: &str, jinja: &str) -> TokenizerSection {
    let contents = ["{}", config, jinja, "{}"];
    let mut body = 4i32.to_be_bytes().to_vec();
    for (name, data) in TOKENIZER_ENTRIES.iter().zip(contents) {
        body.extend(i32::try_from(name.len()).expect("a short name").to_be_bytes());
        body.extend(name.as_bytes());
        body.extend(i32::try_from(data.len()).expect("a short file").to_be_bytes());
        body.extend(data.as_bytes());
    }
    let mut footer = i64::try_from(body.len()).expect("a short body").to_be_bytes().to_vec();
    footer.extend(crc32fast::hash(&body).to_be_bytes());
    footer.extend(1i32.to_be_bytes());
    footer.extend(b"EDRLTOKN");
    let footer: [u8; FOOTER_BYTES] = footer.try_into().expect("the footer's size");
    let footer = TokenizerFooter::parse(&footer).expect("a footer").expect("the magic");
    TokenizerSection::parse(body, &footer).expect("a section")
}

fn config_with_template(template: &str) -> String {
    json!({ "chat_template": template }).to_string()
}

#[test]
fn the_jinja_entry_wins_over_the_config_template() {
    let other = "{{ messages }}";
    let section = section(&config_with_template(other), CHECKPOINT_TEMPLATE);
    let source = chat_template(&section).expect("a template");
    assert_eq!(source, CHECKPOINT_TEMPLATE);
    assert_eq!(check_template_source(&source), Ok(()));
}

#[test]
fn the_jinja_entry_wins_even_when_it_is_not_the_checkpoint_template() {
    let section = section(&config_with_template(CHECKPOINT_TEMPLATE), "{{ messages }}");
    let source = chat_template(&section).expect("a template");
    assert_eq!(source, "{{ messages }}");
    assert!(check_template_source(&source).is_err());
}

#[test]
fn the_config_template_is_used_without_a_jinja_text() {
    let section = section(&config_with_template(CHECKPOINT_TEMPLATE), "");
    let source = chat_template(&section).expect("a template");
    assert_eq!(source, CHECKPOINT_TEMPLATE);
    assert_eq!(check_template_source(&source), Ok(()));
}

#[test]
fn no_template_anywhere_is_an_error() {
    for config in [
        "{}",
        r#"{"chat_template":null}"#,
        r#"{"chat_template":["a"]}"#,
        r#"{"chat_template":7}"#,
    ] {
        let failure = chat_template(&section(config, "")).expect_err("no template");
        assert!(matches!(failure, TokenizerError::NoChatTemplate), "{config}: {failure}");
    }
    let failure = chat_template(&section("[", "")).expect_err("not JSON");
    assert!(matches!(failure, TokenizerError::Json { .. }), "{failure}");
}
