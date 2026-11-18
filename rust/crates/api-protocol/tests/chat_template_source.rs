//! Which text of a tokenizer section is the chat template: the `chat_template.jinja` entry, whatever
//! `tokenizer_config.json` holds. The text goes through the check the server makes of its template.

#![allow(clippy::expect_used, clippy::panic)]

use euhedral_inference_api_protocol::template::check_template_source;
use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::chat_template;
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

fn section_without_config_template(jinja: &str) -> TokenizerSection {
    section("{}", jinja)
}

fn config_with_template(template: &str) -> String {
    json!({ "chat_template": template }).to_string()
}

#[test]
fn the_jinja_entry_is_the_template_whatever_the_config_holds() {
    let section = section(&config_with_template("{{ messages }}"), CHECKPOINT_TEMPLATE);
    let source = chat_template(&section).expect("a template");
    assert_eq!(source, CHECKPOINT_TEMPLATE);
    assert_eq!(check_template_source(source), Ok(()));
    let section = section_without_config_template(CHECKPOINT_TEMPLATE);
    assert_eq!(chat_template(&section).expect("a template"), CHECKPOINT_TEMPLATE);
}

#[test]
fn a_jinja_entry_that_is_not_the_checkpoint_template_is_refused_even_when_the_config_has_it() {
    let section = section(&config_with_template(CHECKPOINT_TEMPLATE), "{{ messages }}");
    let source = chat_template(&section).expect("a template");
    assert_eq!(source, "{{ messages }}");
    assert!(check_template_source(source).is_err());
}

#[test]
fn an_empty_jinja_entry_is_refused_not_replaced_by_the_config_template() {
    let section = section(&config_with_template(CHECKPOINT_TEMPLATE), "");
    let source = chat_template(&section).expect("an entry");
    assert_eq!(source, "");
    assert!(check_template_source(source).is_err());
}
