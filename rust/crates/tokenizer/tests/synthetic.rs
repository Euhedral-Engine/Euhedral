//! A tiny byte-level BPE tokenizer built in memory, packaged as an artifact's tokenizer section. No artifact is
//! needed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::{Tokenizer, TokenizerError};
use serde_json::{Value, json};

/// The expression the Qwen checkpoints split with.
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

const HE: u32 = 256;
const LL: u32 = 257;
const HELL: u32 = 258;
const HELLO: u32 = 259;
const SPACE_W: u32 = 260;
const IM_START: u32 = 261;
const IM_END: u32 = 262;
const END_OF_TEXT: u32 = 263;
const THINK: u32 = 264;

/// The byte-level spelling of every byte: printable Latin-1 as itself, the rest from U+0100 up in byte order.
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

fn tokenizer_json() -> Value {
    let chars = byte_chars();
    let mut vocab = serde_json::Map::new();
    for (byte, character) in chars.iter().enumerate() {
        vocab.insert(character.to_string(), json!(byte));
    }
    for (token, id) in [
        ("he", HE),
        ("ll", LL),
        ("hell", HELL),
        ("hello", HELLO),
        ("\u{120}w", SPACE_W),
    ] {
        vocab.insert(token.to_string(), json!(id));
    }
    let added = |id: u32, content: &str, special: bool| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
               "normalized": false, "special": special})
    };
    json!({
        "version": "1.0",
        "added_tokens": [
            added(IM_START, "<|im_start|>", true),
            added(IM_END, "<|im_end|>", true),
            added(END_OF_TEXT, "<|endoftext|>", true),
            added(THINK, "<think>", false),
        ],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        ]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {
            "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab,
            "merges": ["h e", "l l", "he ll", "hell o", "\u{120} w"],
        },
    })
}

fn config() -> Value {
    json!({"eos_token": "<|im_end|>", "bos_token": null, "add_bos_token": false})
}

fn special_tokens() -> Value {
    json!({"bos_token_id": END_OF_TEXT, "eos_token_id": [IM_END, END_OF_TEXT], "pad_token_id": END_OF_TEXT})
}

/// The section as an artifact stores it: the four files in order, then the footer with the body's length and CRC.
fn section(files: [&Value; 3]) -> TokenizerSection {
    let contents: [Vec<u8>; 4] = [
        serde_json::to_vec(files[0]).unwrap(),
        serde_json::to_vec(files[1]).unwrap(),
        b"{{ template }}".to_vec(),
        serde_json::to_vec(files[2]).unwrap(),
    ];
    let mut body = 4i32.to_be_bytes().to_vec();
    for (name, data) in TOKENIZER_ENTRIES.iter().zip(&contents) {
        body.extend((name.len() as i32).to_be_bytes());
        body.extend(name.as_bytes());
        body.extend((data.len() as i32).to_be_bytes());
        body.extend(data);
    }
    let mut footer = (body.len() as i64).to_be_bytes().to_vec();
    footer.extend(crc32fast::hash(&body).to_be_bytes());
    footer.extend(1i32.to_be_bytes());
    footer.extend(b"EDRLTOKN");
    let footer: [u8; FOOTER_BYTES] = footer.try_into().unwrap();
    let footer = TokenizerFooter::parse(&footer).unwrap().unwrap();
    TokenizerSection::parse(body, &footer).unwrap()
}

fn load(tokenizer: &Value, config: &Value, special: &Value) -> Result<Tokenizer, TokenizerError> {
    Tokenizer::from_section(&section([tokenizer, config, special]))
}

fn tokenizer() -> Tokenizer {
    load(&tokenizer_json(), &config(), &special_tokens()).unwrap()
}

fn byte_ids(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

#[test]
fn bpe_merges_apply_within_a_pre_token() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.encode("hello").unwrap(), [HELLO]);
    assert_eq!(tokenizer.encode("hell").unwrap(), [HELL]);
    assert_eq!(tokenizer.encode("he").unwrap(), [HE]);
    // " world": the space merges with `w` only; the pre-token is the space plus the word.
    let mut expected = vec![SPACE_W];
    expected.extend(byte_ids("orld"));
    assert_eq!(tokenizer.encode("hello world").unwrap()[1..], expected[..]);
    assert_eq!(tokenizer.encode("").unwrap(), Vec::<u32>::new());
}

#[test]
fn the_split_expression_keeps_pre_tokens_apart() {
    let tokenizer = tokenizer();
    // Punctuation and digits are pre-tokens of their own, so "hello," never merges across the comma.
    let mut expected = vec![HELLO];
    expected.extend(byte_ids(",12"));
    assert_eq!(tokenizer.encode("hello,12").unwrap(), expected);
    assert_eq!(tokenizer.encode("a\n\nb").unwrap(), byte_ids("a\n\nb"));
}

#[test]
fn added_tokens_encode_to_their_own_ids_wherever_they_appear() {
    let tokenizer = tokenizer();
    let mut expected = vec![IM_START];
    expected.extend(byte_ids("user\n"));
    expected.push(HELLO);
    expected.push(IM_END);
    expected.push(u32::from(b'\n'));
    expected.push(THINK);
    assert_eq!(
        tokenizer.encode("<|im_start|>user\nhello<|im_end|>\n<think>").unwrap(),
        expected
    );
}

#[test]
fn text_is_nfc_normalized_before_it_is_split() {
    let tokenizer = tokenizer();
    // "e" followed by a combining acute accent composes to U+00E9.
    assert_eq!(
        tokenizer.encode("e\u{301}").unwrap(),
        tokenizer.encode("\u{e9}").unwrap()
    );
    assert_eq!(tokenizer.encode("\u{e9}").unwrap(), byte_ids("\u{e9}"));
}

#[test]
fn model_special_tokens_follow_the_config() {
    let plain = tokenizer();
    assert_eq!(
        plain.encode_with_model_special_tokens("hello").unwrap(),
        plain.encode("hello").unwrap()
    );

    let config = json!({"eos_token": "<|im_end|>", "bos_token": "<|endoftext|>", "add_bos_token": true,
                        "add_eos_token": true});
    let both = load(&tokenizer_json(), &config, &special_tokens()).unwrap();
    assert_eq!(
        both.encode_with_model_special_tokens("hello").unwrap(),
        [END_OF_TEXT, HELLO, IM_END]
    );
    // Encoding alone never adds them.
    assert_eq!(both.encode("hello").unwrap(), [HELLO]);
}

#[test]
fn special_ids_are_exposed() {
    let tokenizer = tokenizer();
    assert_eq!(tokenizer.eos_id(), IM_END);
    assert_eq!(tokenizer.eos_ids(), [IM_END, END_OF_TEXT]);
    assert_eq!(tokenizer.generation_bos_id(), Some(END_OF_TEXT));
    assert_eq!(tokenizer.control_token_id("<think>"), Some(THINK));
    assert_eq!(tokenizer.control_token_id("hello"), None);
    assert_eq!(tokenizer.vocabulary_size(), THINK as usize + 1);
}

#[test]
fn decoding_round_trips_text_and_spells_control_tokens() {
    let tokenizer = tokenizer();
    for text in ["hello world", "caf\u{e9} \u{4e2d}\u{6587} \u{1f600}", "a\n\n  b"] {
        assert_eq!(tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(), text);
    }
    assert_eq!(
        tokenizer.decode(&[IM_START, HELLO, IM_END]).unwrap(),
        "<|im_start|>hello<|im_end|>"
    );
    assert!(matches!(
        tokenizer.decode(&[1_000_000]),
        Err(TokenizerError::UnknownToken(1_000_000))
    ));
}

#[test]
fn the_stream_holds_back_a_split_character() {
    let tokenizer = tokenizer();
    // U+1F600 is the four bytes f0 9f 98 80, one token each in this vocabulary.
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.push(0xf0).unwrap(), "");
    assert_eq!(stream.push(0x9f).unwrap(), "");
    assert_eq!(stream.push(0x98).unwrap(), "");
    assert_eq!(stream.push(0x80).unwrap(), "\u{1f600}");
    assert_eq!(stream.finish(), "");

    let mut truncated = tokenizer.decode_stream();
    assert_eq!(truncated.push(0xf0).unwrap(), "");
    assert_eq!(truncated.push(IM_END).unwrap(), "\u{fffd}<|im_end|>");
}

#[test]
fn settings_only_the_library_honours_are_refused() {
    let load_with = |edit: &dyn Fn(&mut Value)| {
        let mut json = tokenizer_json();
        edit(&mut json);
        load(&json, &config(), &special_tokens())
    };
    let refused = |edit: &dyn Fn(&mut Value)| {
        let error = load_with(edit).unwrap_err();
        assert!(matches!(error, TokenizerError::Unsupported(_)), "{error}");
    };
    refused(&|json| json["model"]["dropout"] = json!(0.1));
    refused(&|json| json["model"]["ignore_merges"] = json!(true));
    refused(&|json| json["model"]["continuing_subword_prefix"] = json!("##"));
    refused(&|json| json["model"]["end_of_word_suffix"] = json!("</w>"));
    refused(&|json| json["truncation"] = json!({"max_length": 8, "stride": 0}));
    refused(&|json| json["padding"] = json!({"strategy": "BatchLongest"}));
    for flag in ["lstrip", "rstrip", "single_word", "normalized"] {
        refused(&|json| json["added_tokens"][1][flag] = json!(true));
    }
    // Unset spellings are accepted: null or empty affixes, false flags, null truncation and padding, no dropout.
    assert!(load_with(&|json| json["model"]["continuing_subword_prefix"] = json!("")).is_ok());
    assert!(load_with(&|json| json["model"]["end_of_word_suffix"] = json!("")).is_ok());
    assert!(load_with(&|json| json["truncation"] = Value::Null).is_ok());
    assert!(load_with(&|json| json["padding"] = Value::Null).is_ok());
    assert!(load_with(&|json| json["model"]["dropout"] = Value::Null).is_ok());
}

#[test]
fn a_tokenizer_that_is_not_byte_level_bpe_is_refused() {
    let refused = |edit: &dyn Fn(&mut Value)| {
        let mut json = tokenizer_json();
        edit(&mut json);
        let error = load(&json, &config(), &special_tokens()).unwrap_err();
        assert!(matches!(error, TokenizerError::Unsupported(_)), "{error}");
    };
    refused(&|json| json["model"]["type"] = json!("Unigram"));
    refused(&|json| json["normalizer"] = json!({"type": "NFD"}));
    refused(&|json| json["normalizer"] = Value::Null);
    refused(&|json| json["pre_tokenizer"]["pretokenizers"][0]["behavior"] = json!("Removed"));
    refused(&|json| json["pre_tokenizer"]["pretokenizers"][0]["invert"] = json!(true));
    refused(&|json| json["pre_tokenizer"]["pretokenizers"][1]["use_regex"] = json!(true));
    refused(&|json| json["pre_tokenizer"]["pretokenizers"][1]["add_prefix_space"] = json!(true));
    refused(&|json| {
        json["pre_tokenizer"]["pretokenizers"].as_array_mut().unwrap().pop();
    });
    // An added token on a vocabulary ID, or listed twice.
    refused(&|json| json["added_tokens"][0]["id"] = json!(5));
    refused(&|json| json["added_tokens"][1]["content"] = json!("<|im_start|>"));
}

#[test]
fn special_token_metadata_must_resolve() {
    let tokenizer_json = tokenizer_json();
    let unsupported = |config: Value, special: Value| {
        let error = load(&tokenizer_json, &config, &special).unwrap_err();
        assert!(matches!(error, TokenizerError::Unsupported(_)), "{error}");
    };
    unsupported(json!({"eos_token": "<think>"}), special_tokens());
    unsupported(json!({"eos_token": "<missing>"}), special_tokens());
    unsupported(json!({}), special_tokens());
    unsupported(
        json!({"eos_token": "<|im_end|>", "add_bos_token": true}),
        special_tokens(),
    );
    unsupported(config(), json!({"bos_token_id": END_OF_TEXT}));
    unsupported(config(), json!({"eos_token_id": []}));
    unsupported(config(), json!({"eos_token_id": 999}));
    unsupported(config(), json!({"eos_token_id": IM_END, "bos_token_id": 999}));
    // A single EOS ID is accepted as well as a list.
    let single = load(&tokenizer_json, &config(), &json!({"eos_token_id": IM_END})).unwrap();
    assert_eq!(single.eos_ids(), [IM_END]);
}

#[test]
fn a_file_of_the_wrong_shape_is_refused() {
    let error = load(&tokenizer_json(), &json!("not an object"), &special_tokens()).unwrap_err();
    assert!(
        matches!(
            error,
            TokenizerError::Json {
                file: "tokenizer_config.json",
                ..
            }
        ),
        "{error}"
    );
    let error = load(&json!({"model": {"type": "BPE"}}), &config(), &special_tokens()).unwrap_err();
    assert!(
        matches!(
            error,
            TokenizerError::Json {
                file: "tokenizer.json",
                ..
            }
        ),
        "{error}"
    );
}
