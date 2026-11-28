//! The grammar matcher over the real Qwen vocabulary: the checkpoint tokenizer's token bytes and end tokens, a
//! logits row as wide as the model's. Needs `euhedral.qwen.tokenizer-dir` (in `~/.config/euhedral/test.properties` or `~/.gradle/gradle.properties`, or
//! `EUHEDRAL_TEST_QWEN_TOKENIZER_DIR`); each test is skipped, with a line `--nocapture` shows, without it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test file: a failed unwrap is the failure, and a fixture is read with blocking I/O"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::str::from_utf8;
use std::sync::OnceLock;

use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection};
use euhedral_inference_tokenizer::Tokenizer;
use serde_json::Value;

use super::{GrammarChecker, GrammarMatcher, end_ids, token_bytes};

const KEY: &str = "euhedral.qwen.tokenizer-dir";
/// The width of the model's logits row, past the tokenizer's own IDs.
const VOCABULARY: usize = 248_320;
const SPECIAL_TOKEN_KEYS: [&str; 3] = ["bos_token_id", "eos_token_id", "pad_token_id"];
const PERSON: &str = r#"{"type":"object","properties":{
  "name":{"type":"string","minLength":1,"maxLength":12},
  "age":{"type":"integer","minimum":0,"maximum":150},
  "role":{"enum":["admin","user"]},
  "tags":{"type":"array","items":{"type":"string","maxLength":5},"maxItems":3},
  "email":{"anyOf":[{"type":"string","pattern":"^[a-z]+@[a-z]+\\.com$"},{"type":"null"}]}},
 "required":["name","age","role"],"additionalProperties":false}"#;

struct Fixture {
    tokenizer: Tokenizer,
    table: Vec<Vec<u8>>,
    ends: Vec<u32>,
}

impl Fixture {
    fn checker(&self) -> GrammarChecker {
        GrammarChecker::new(&self.table, &self.ends).expect("the real vocabulary")
    }

    fn person(&self) -> GrammarMatcher {
        self.checker()
            .constraint(&format!("start: %json {PERSON}"))
            .expect("grammar")
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        self.tokenizer.encode(text).expect("encodes")
    }
}

/// The checkpoint's tokenizer packaged as the artifact's tokenizer section, as a loader reads a
/// checkpoint directory: the generation config's special-token IDs only.
fn load(directory: &Path) -> Tokenizer {
    let read = |name: &str| fs::read(directory.join(name)).unwrap_or_else(|_| panic!("the checkpoint has {name}"));
    let generation: Value = serde_json::from_slice(&read("generation_config.json")).unwrap();
    let special: serde_json::Map<String, Value> = SPECIAL_TOKEN_KEYS
        .iter()
        .filter_map(|key| generation.get(*key).map(|value| ((*key).to_owned(), value.clone())))
        .collect();
    let contents = [
        read("tokenizer.json"),
        read("tokenizer_config.json"),
        b"{{ template }}".to_vec(),
        serde_json::to_vec(&special).unwrap(),
    ];
    let mut body = i32::try_from(contents.len()).unwrap().to_be_bytes().to_vec();
    for (name, data) in TOKENIZER_ENTRIES.iter().zip(&contents) {
        body.extend(i32::try_from(name.len()).unwrap().to_be_bytes());
        body.extend(name.as_bytes());
        body.extend(i32::try_from(data.len()).unwrap().to_be_bytes());
        body.extend(data);
    }
    let mut footer = i64::try_from(body.len()).unwrap().to_be_bytes().to_vec();
    footer.extend(crc32fast::hash(&body).to_be_bytes());
    footer.extend(1i32.to_be_bytes());
    footer.extend(b"EDRLTOKN");
    let footer: [u8; FOOTER_BYTES] = footer.try_into().unwrap();
    let footer = TokenizerFooter::parse(&footer).unwrap().unwrap();
    Tokenizer::from_section(&TokenizerSection::parse(body, &footer).unwrap()).unwrap()
}

/// The shared fixture, or `None` (with a skip line) on a machine without the checkpoint's tokenizer.
fn fixture() -> Option<&'static Fixture> {
    static FIXTURE: OnceLock<Option<Fixture>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let directory = PathBuf::from(test_support::require_property(KEY)?);
            if !directory.join("tokenizer.json").is_file() {
                println!("skipped: {KEY} names no checkpoint");
                return None;
            }
            let tokenizer = load(&directory);
            let table = token_bytes(&tokenizer, VOCABULARY);
            let ends = end_ids(&tokenizer);
            Some(Fixture { tokenizer, table, ends })
        })
        .as_ref()
}

fn allowed_by_mask(mask: &[u32], id: u32) -> bool {
    mask.get((id >> 5) as usize)
        .is_some_and(|word| word & (1 << (id & 31)) != 0)
}

/// The IDs a mask allows, in order.
fn allowed(mask: &[u32]) -> Vec<u32> {
    let mut ids = Vec::new();
    for (index, word) in mask.iter().enumerate() {
        let mut rest = *word;
        while rest != 0 {
            ids.push(u32::try_from(index).unwrap() * 32 + rest.trailing_zeros());
            rest &= rest - 1;
        }
    }
    ids
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(bound).unwrap()).unwrap()
    }
}

#[test]
fn the_table_is_the_tokens_bytes_and_names_for_the_rest() {
    let Some(fixture) = fixture() else { return };
    let tokenizer = &fixture.tokenizer;
    assert_eq!(fixture.table.len(), VOCABULARY);
    assert!(VOCABULARY > tokenizer.vocabulary_size());
    let mut partial = 0;
    for (index, bytes) in fixture.table.iter().enumerate() {
        let id = u32::try_from(index).unwrap();
        assert!(!bytes.is_empty(), "token {id}");
        if let Some(exact) = tokenizer.token_bytes(id) {
            assert_eq!(bytes, exact, "token {id}");
            partial += usize::from(from_utf8(exact).is_err());
        } else {
            assert_eq!(bytes[0], 0xFF, "token {id}");
            assert!(from_utf8(&bytes[1..]).is_ok(), "token {id}");
        }
    }
    assert!(
        partial > 100,
        "byte-level tokens that end inside a character: {partial}"
    );
    for name in [
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<think>",
        "</think>",
        "<tool_call>",
    ] {
        let id = tokenizer.control_token_id(name).expect(name);
        assert!(tokenizer.is_control_token(id), "{name}");
        assert_eq!(
            fixture.table[id as usize],
            [&[0xFF][..], name.as_bytes()].concat(),
            "{name}"
        );
    }
    let past = tokenizer.vocabulary_size();
    assert_eq!(
        fixture.table[past],
        [&[0xFF][..], format!("<[unused {past}]>").as_bytes()].concat()
    );
    assert_eq!(fixture.ends[0], tokenizer.eos_id());
    assert!(tokenizer.eos_ids().iter().all(|id| fixture.ends.contains(id)));
}

#[test]
fn the_bulk_mask_agrees_with_token_queries_and_never_allows_control_tokens() {
    let Some(fixture) = fixture() else { return };
    let grammar = "start: %json {\"type\":\"string\"}";
    let prefix = fixture.encode("\"inside");
    let mut checker = fixture.checker();
    let mut matcher = checker.constraint(grammar).unwrap();
    for id in &prefix {
        matcher.consume(*id).unwrap();
    }
    let mask = matcher.compute_mask().unwrap().to_vec();
    for id in 0..u32::try_from(VOCABULARY).unwrap() {
        assert_eq!(matcher.allows(id).unwrap(), allowed_by_mask(&mask, id), "token {id}");
    }
    assert!(!allowed(&mask).is_empty());
    // Past the vocabulary no bit is set, and no id there is allowed.
    assert!(allowed(&mask).iter().all(|id| (*id as usize) < VOCABULARY));
    assert!(!matcher.allows(u32::try_from(VOCABULARY).unwrap()).unwrap());
    for name in ["</think>", "<think>", "<tool_call>", "<|im_start|>"] {
        let id = fixture.tokenizer.control_token_id(name).expect(name);
        assert!(!matcher.allows(id).unwrap(), "{name}");
    }
    assert!(
        !matcher.allows(fixture.tokenizer.eos_id()).unwrap(),
        "the string is unfinished"
    );
    // A fresh matcher at the same position takes a token exactly when the mask allows it.
    let probe = fixture
        .encode("side")
        .into_iter()
        .chain([u32::try_from(VOCABULARY - 1).unwrap()]);
    for id in probe.chain((0..VOCABULARY as u32).step_by(997)) {
        let mut fresh = checker.constraint(grammar).unwrap();
        for earlier in &prefix {
            fresh.consume(*earlier).unwrap();
        }
        assert_eq!(fresh.consume(id).is_ok(), allowed_by_mask(&mask, id), "token {id}");
    }
}

#[test]
fn an_end_token_is_allowed_only_once_the_document_is_whole() {
    let Some(fixture) = fixture() else { return };
    let mut matcher = fixture.person();
    for id in fixture.encode(r#"{"name":"Ann","age":42,"role":"user""#) {
        assert!(matcher.allows(id).unwrap(), "token {id}");
        matcher.consume(id).unwrap();
    }
    assert!(!matcher.is_accepting());
    for end in &fixture.ends {
        assert!(!matcher.allows(*end).unwrap(), "end {end}");
    }
    let close = fixture.encode("}")[0];
    matcher.consume(close).unwrap();
    assert!(matcher.is_accepting());
    for end in &fixture.ends {
        assert!(matcher.allows(*end).unwrap(), "end {end}");
    }
    let space = fixture.encode(" ")[0];
    assert!(!matcher.allows(space).unwrap(), "nothing follows the document");
    // Generation commits the terminator it selected; the finished grammar takes it without consuming.
    matcher.consume(fixture.tokenizer.eos_id()).unwrap();
}

/// Whether a token is short punctuation or digits, the tokens a walk prefers so that it leaves strings.
fn structural(fixture: &Fixture, id: u32) -> bool {
    let Some(bytes) = fixture.tokenizer.token_bytes(id) else {
        return false;
    };
    bytes.len() <= 2 && bytes.iter().all(|byte| b"\"{}[],:-0123456789aeu".contains(byte))
}

fn assert_person(person: &Value, text: &str) {
    let object = person.as_object().unwrap_or_else(|| panic!("{text}"));
    assert!(
        object
            .keys()
            .all(|key| ["name", "age", "role", "tags", "email"].contains(&key.as_str())),
        "{text}"
    );
    let name = object["name"].as_str().expect(text).chars().count();
    assert!((1..=12).contains(&name), "{text}");
    let age = object["age"].as_u64().unwrap_or_else(|| panic!("{text}"));
    assert!(age <= 150, "{text}");
    assert!(
        ["admin", "user"].contains(&object["role"].as_str().expect(text)),
        "{text}"
    );
    if let Some(tags) = object.get("tags") {
        let tags = tags.as_array().unwrap_or_else(|| panic!("{text}"));
        assert!(tags.len() <= 3, "{text}");
        assert!(
            tags.iter()
                .all(|tag| tag.as_str().is_some_and(|tag| tag.chars().count() <= 5)),
            "{text}"
        );
    }
    if let Some(email) = object.get("email").filter(|email| !email.is_null()) {
        let email = email.as_str().unwrap_or_else(|| panic!("{text}"));
        let (user, domain) = email.split_once('@').unwrap_or_else(|| panic!("{text}"));
        let host = domain.strip_suffix(".com").unwrap_or_else(|| panic!("{text}"));
        let lower = |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_lowercase());
        assert!(lower(user) && lower(host), "{text}");
    }
}

/// Whatever tokens a generation picks among those the mask allows, a finished text satisfies the schema. A uniform
/// walk almost never leaves a string, so walks prefer short punctuation and digit tokens while still taking
/// arbitrary allowed tokens three times in ten.
#[test]
fn random_walks_through_the_mask_only_finish_valid_documents() {
    let Some(fixture) = fixture() else { return };
    let mut checker = fixture.checker();
    let grammar = format!("start: %json {PERSON}");
    let mut random = SplitMix(7);
    let mut finished = 0;
    for _ in 0..40 {
        let mut matcher = checker.constraint(&grammar).unwrap();
        let mut chosen = Vec::new();
        for _ in 0..200 {
            let options = allowed(matcher.compute_mask().unwrap());
            assert!(!options.is_empty(), "a live grammar always allows a token");
            if matcher.is_accepting() && random.below(3) == 0 {
                break;
            }
            let punctuation: Vec<u32> = options.iter().copied().filter(|id| structural(fixture, *id)).collect();
            let pick = if !punctuation.is_empty() && random.below(10) < 7 {
                punctuation[random.below(punctuation.len())]
            } else {
                options[random.below(options.len())]
            };
            if fixture.ends.contains(&pick) {
                break;
            }
            matcher.consume(pick).unwrap();
            chosen.push(pick);
        }
        if !matcher.is_accepting() {
            continue;
        }
        finished += 1;
        let text = fixture.tokenizer.decode(&chosen).unwrap();
        let person: Value = serde_json::from_str(&text).unwrap_or_else(|error| panic!("{error}: {text}"));
        assert_person(&person, &text);
    }
    assert!(finished > 10, "too few walks finished: {finished}");
}
