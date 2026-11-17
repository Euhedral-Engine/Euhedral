use llguidance::TokenParser;
use llguidance::panic_utils::catch_unwind;

use super::*;
use crate::json::JsonValue;

const THINK: u32 = 256;
const OPEN: u32 = 257;
const END: u32 = 258;
const END_TEXT: u32 = 259;
const SIZE: u32 = 260;

const PERSON: &str = r#"{"type":"object","properties":{
  "name":{"type":"string","minLength":1,"maxLength":12},
  "age":{"type":"integer","minimum":0,"maximum":150},
  "role":{"enum":["admin","user"]},
  "tags":{"type":"array","items":{"type":"string","maxLength":5},"maxItems":3},
  "email":{"anyOf":[{"type":"string","pattern":"^[a-z]+@[a-z]+\\.com$"},{"type":"null"}]}},
 "required":["name","age","role"],"additionalProperties":false}"#;

/// One token per byte, two control tokens, and two end tokens.
fn tokens() -> Vec<Vec<u8>> {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
    tokens.push(b"\xFF<think>".to_vec());
    tokens.push(b"\xFF<tool_call>".to_vec());
    tokens.push(b"\xFF<|im_end|>".to_vec());
    tokens.push(b"\xFF<|endoftext|>".to_vec());
    tokens
}

fn checker() -> GrammarChecker {
    GrammarChecker::new(&tokens(), &[END, END_TEXT]).expect("vocabulary")
}

fn constraint(checker: &mut GrammarChecker, schema: &str) -> GrammarMatcher {
    checker
        .constraint(&format!("start: %json {schema}"))
        .expect("grammar compiles")
}

fn encode(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

fn allowed(matcher: &mut GrammarMatcher) -> Vec<u32> {
    (0..SIZE).filter(|id| matcher.allows(*id).expect("mask")).collect()
}

/// Feeds `text` while the mask allows it; whether every token was allowed.
fn feed(matcher: &mut GrammarMatcher, text: &str) -> bool {
    for id in encode(text) {
        if !matcher.allows(id).expect("mask") {
            return false;
        }
        matcher.consume(id).expect("allowed");
    }
    true
}

/// A small deterministic generator (xorshift), so a walk is reproducible.
struct Random(u64);

impl Random {
    fn below(&mut self, bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        usize::try_from(self.0 % bound as u64).expect("fits")
    }
}

#[test]
fn a_matcher_can_move_between_threads() {
    fn send<T: Send>() {}
    send::<GrammarMatcher>();
    send::<GrammarChecker>();
}

#[test]
fn random_walks_through_the_mask_only_finish_valid_documents() {
    // A uniform walk almost never leaves a string, so walks prefer short punctuation and digit tokens while still
    // taking arbitrary allowed tokens three times in ten.
    let mut checker = checker();
    let mut random = Random(0x9E37_79B9_7F4A_7C15);
    let mut finished = 0;
    for _ in 0..40 {
        let mut matcher = constraint(&mut checker, PERSON);
        let mut chosen = Vec::new();
        for _ in 0..300 {
            let allowed = allowed(&mut matcher);
            assert!(!allowed.is_empty(), "a live grammar always allows a token");
            if allowed.contains(&END) && random.below(3) == 0 {
                break;
            }
            let structural: Vec<u32> = allowed
                .iter()
                .copied()
                .filter(|id| *id < 128 && "\"{}[],:-0123456789aeu".contains(char::from(*id as u8)))
                .collect();
            let choice = if !structural.is_empty() && random.below(10) < 7 {
                structural[random.below(structural.len())]
            } else {
                allowed[random.below(allowed.len())]
            };
            if choice == END || choice == END_TEXT {
                break;
            }
            matcher.consume(choice).expect("allowed");
            chosen.push(u8::try_from(choice).expect("a byte token"));
        }
        if !matcher.is_accepting() {
            continue;
        }
        finished += 1;
        let text = String::from_utf8(chosen).expect("the schema's strings are UTF-8");
        assert_person(&text);
    }
    assert!(finished > 10, "too few walks finished: {finished}");
}

fn assert_person(text: &str) {
    let person = JsonValue::parse(text).unwrap_or_else(|| panic!("not JSON: {text}"));
    let JsonValue::Object(members) = &person else {
        panic!("not an object: {text}");
    };
    for (key, _) in members {
        assert!(
            ["name", "age", "role", "tags", "email"].contains(&key.as_str()),
            "{text}"
        );
    }
    let JsonValue::String(name) = person.get("name").expect("name") else {
        panic!("name: {text}");
    };
    assert!((1..=12).contains(&name.chars().count()), "{text}");
    let JsonValue::Integer(age) = person.get("age").expect("age") else {
        panic!("age: {text}");
    };
    assert!(
        (0..=150).contains(&age.as_str().parse::<i128>().expect("integer")),
        "{text}"
    );
    let JsonValue::String(role) = person.get("role").expect("role") else {
        panic!("role: {text}");
    };
    assert!(["admin", "user"].contains(&role.as_str()), "{text}");
    if let Some(JsonValue::Array(tags)) = person.get("tags") {
        assert!(tags.len() <= 3, "{text}");
        for tag in tags {
            let JsonValue::String(tag) = tag else {
                panic!("tag: {text}");
            };
            assert!(tag.chars().count() <= 5, "{text}");
        }
    }
    match person.get("email") {
        Some(JsonValue::String(email)) => {
            let (user, host) = email.split_once('@').expect("an address");
            let domain = host.strip_suffix(".com").expect(".com");
            let lower = |part: &str| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_lowercase());
            assert!(lower(user) && lower(domain), "{text}");
        }
        None | Some(JsonValue::Null) => {}
        Some(_) => panic!("email: {text}"),
    }
}

#[test]
fn the_bulk_mask_agrees_with_token_queries_and_never_allows_control_tokens() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, r#"{"type":"string"}"#);
    assert!(feed(&mut matcher, "\"inside"));
    let words = matcher.compute_mask().expect("mask").to_vec();
    assert_eq!(words.len(), SIZE.div_ceil(32) as usize);
    for id in 0..SIZE {
        let bit = words[(id >> 5) as usize] & (1 << (id & 31)) != 0;
        assert_eq!(matcher.allows(id).expect("mask"), bit, "token {id}");
    }
    // Bits past the vocabulary, up to the last word's end, are never set.
    for id in SIZE..SIZE.div_ceil(32) * 32 {
        assert_eq!(words[(id >> 5) as usize] & (1 << (id & 31)), 0, "padding bit {id}");
    }
    for control in [THINK, OPEN] {
        assert!(!matcher.allows(control).expect("mask"), "control {control}");
    }
    assert!(!matcher.allows(END).expect("mask"), "the string is unfinished");
    assert!(!matcher.allows(END_TEXT).expect("mask"), "the string is unfinished");
}

#[test]
fn ids_at_or_above_the_vocabulary_are_never_allowed() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, r#"{"type":"string"}"#);
    for id in [SIZE, SIZE + 1, 1 << 20, u32::MAX] {
        assert!(!matcher.allows(id).expect("mask"), "{id}");
        assert_eq!(matcher.consume(id), Err(MatcherError::Violation(id)));
    }
    assert!(
        matcher.allows(u32::from(b'"')).expect("mask"),
        "the matcher is unchanged"
    );
}

#[test]
fn an_end_token_is_allowed_only_once_the_document_is_whole() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, PERSON);
    assert!(feed(&mut matcher, r#"{"name":"Ann","age":42,"role":"user""#));
    assert!(!matcher.is_accepting());
    for end in [END, END_TEXT] {
        assert!(!matcher.allows(end).expect("mask"));
    }
    assert!(feed(&mut matcher, "}"));
    assert!(matcher.is_accepting());
    for end in [END, END_TEXT] {
        assert!(matcher.allows(end).expect("mask"));
    }
    assert!(
        !matcher.allows(u32::from(b' ')).expect("mask"),
        "nothing follows the document"
    );
    // Generation commits the terminator it selected; the finished grammar takes it without consuming.
    matcher.consume(END).expect("the end token");
    assert!(matcher.is_accepting());
    assert!(matcher.allows(END_TEXT).expect("mask"));
}

#[test]
fn a_token_the_grammar_forbids_is_refused_and_changes_nothing() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, r#"{"type":"boolean"}"#);
    assert_eq!(
        matcher.consume(u32::from(b'x')),
        Err(MatcherError::Violation(u32::from(b'x')))
    );
    assert!(matcher.is_usable());
    assert!(feed(&mut matcher, "true"));
    assert!(matcher.is_accepting());
}

#[test]
fn schema_violations_cannot_be_generated() {
    let mut checker = checker();
    for invalid in [
        r#"{"name":"Ann","age":151"#,
        r#"{"name":"Ann","age":-1"#,
        r#"{"name":"Ann","age":4.5"#,
        r#"{"name":"Ann","age":42,"role":"root""#,
        r#"{"name":"Ann","age":42,"role":"user","extra":1"#,
        r#"{"name":"","#,
        r#"{"name":"Ann","age":42,"role":"user","email":"Ann@x.com""#,
        r#"{"age":42}"#,
        r#"{"name":"Ann","age":42,"role":"user"}}"#,
    ] {
        let mut matcher = constraint(&mut checker, PERSON);
        let refused = !feed(&mut matcher, invalid);
        assert!(refused || !matcher.is_accepting(), "accepted {invalid}");
    }
}

#[test]
fn unsupported_schemas_are_refused_with_llguidances_reason() {
    let mut checker = checker();
    let refused = |schema: &str| {
        let error = checker
            .check_json_schema(&JsonValue::parse(schema).expect("JSON"), "schema")
            .expect_err(schema);
        format!("{error:?}")
    };
    assert!(refused(r#"{"not":{}}"#).contains(r#"Unimplemented keys: [\"not\"]"#));
    refused(r#"{"type":"string","format":"x"}"#);
    refused(r#"{"type":"integer","minimum":5,"maximum":1}"#);
    refused(r#"{"oneOf":[{"type":"integer"},{"type":"number"}]}"#);
    assert!(checker.constraint("start: nothing").is_err());
    checker
        .check_json_schema(&JsonValue::parse(PERSON).expect("JSON"), "schema")
        .expect("compiles");
    checker
        .check_json_schema(
            &JsonValue::parse(r#"{"oneOf":[{"type":"string"},{"type":"integer"}]}"#).expect("JSON"),
            "schema",
        )
        .expect("compiles");
}

#[test]
fn properties_appear_in_the_schemas_order() {
    let mut checker = checker();
    let ordered =
        r#"{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}"#;
    let mut matcher = constraint(&mut checker, ordered);
    assert!(
        !feed(&mut matcher, r#"{"b":1,"a":2}"#),
        "llguidance generates properties in the schema's order"
    );
    let mut matcher = constraint(&mut checker, ordered);
    assert!(feed(&mut matcher, r#"{"a":1,"b":2}"#));
    assert!(matcher.is_accepting());
}

#[test]
fn matchers_from_one_cached_grammar_are_independent() {
    let mut checker = checker();
    let mut first = constraint(&mut checker, r#"{"type":"boolean"}"#);
    let mut second = constraint(&mut checker, r#"{"type":"boolean"}"#);
    assert!(feed(&mut first, "true"));
    assert!(first.is_accepting());
    assert!(!second.is_accepting());
    assert!(second.allows(u32::from(b'f')).expect("mask"));
    assert!(feed(&mut second, "false"));
    assert!(first.is_accepting() && second.is_accepting());
}

#[test]
fn a_failure_in_the_middle_of_a_generation_leaves_the_matcher_unusable() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, PERSON);
    assert!(feed(&mut matcher, r#"{"name":"A"#));
    matcher.compute_mask().expect("mask");
    assert!(matcher.mask().is_some());
    matcher
        .matcher
        .test_trigger_lexer_error()
        .expect_err("the hook fails the matcher");
    // A mask computed before the failure is still the step's; the next computation reports the failure.
    matcher.mask = None;
    let error = matcher.compute_mask().expect_err("a failed matcher has no mask");
    assert!(matches!(error, MatcherError::Failed(_)), "{error}");
    assert!(!matcher.is_usable());
    assert!(matcher.mask().is_none());
    assert!(matches!(matcher.allows(u32::from(b'a')), Err(MatcherError::Failed(_))));
    assert!(matches!(matcher.consume(u32::from(b'a')), Err(MatcherError::Failed(_))));
    assert!(!matcher.is_accepting());
}

#[test]
fn a_failure_found_by_the_acceptance_query_makes_the_matcher_unusable() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, PERSON);
    assert!(feed(&mut matcher, r#"{"name":"A"#));
    matcher
        .matcher
        .test_trigger_lexer_error()
        .expect_err("the hook fails the matcher");
    assert!(matcher.is_usable(), "nothing has noticed yet");
    assert!(!matcher.is_accepting());
    assert!(!matcher.is_usable());
}

#[test]
fn a_matcher_that_failed_when_built_reports_the_failure() {
    let panicked = catch_unwind::<_, TokenParser>(AssertUnwindSafe(|| panic!("boom")));
    let mut matcher = GrammarMatcher::new(Matcher::new(panicked), SIZE, Arc::from([END]));
    let error = matcher.compute_mask().expect_err("a failed matcher has no mask");
    assert!(
        matches!(&error, MatcherError::Failed(message) if message.contains("boom")),
        "{error}"
    );
    assert!(!matcher.is_usable());
}

#[test]
fn the_mask_is_read_without_computing_and_dropped_by_consume() {
    let mut checker = checker();
    let mut matcher = constraint(&mut checker, r#"{"type":"boolean"}"#);
    assert!(matcher.mask().is_none());
    let computed = matcher.compute_mask().expect("mask").to_vec();
    assert_eq!(matcher.mask(), Some(&computed[..]));
    matcher.consume(u32::from(b't')).expect("allowed");
    assert!(matcher.mask().is_none());
    matcher.compute_mask().expect("mask");
    assert!(matcher.mask().is_some());
}

fn grammar(number: usize) -> String {
    format!("start: \"g{number}\"")
}

#[test]
fn the_cache_keeps_a_bounded_number_of_grammars_and_evicts_the_least_recently_used() {
    let mut checker = GrammarChecker::with_cache_size(&tokens(), &[END], 2).expect("vocabulary");
    for number in 0..2 {
        checker.constraint(&grammar(number)).expect("compiles");
    }
    assert_eq!(checker.compiled.len(), 2);
    // Reusing the first makes the second the least recently used.
    checker.constraint(&grammar(0)).expect("cached");
    checker.constraint(&grammar(2)).expect("compiles");
    assert_eq!(checker.compiled.len(), 2);
    assert!(checker.compiled.holds(&grammar(0)));
    assert!(!checker.compiled.holds(&grammar(1)), "evicted");
    assert!(checker.compiled.holds(&grammar(2)));
    // An evicted grammar compiles again.
    checker.constraint(&grammar(1)).expect("compiles again");
    assert!(!checker.compiled.holds(&grammar(0)));
}

#[test]
fn the_default_bound_is_thirty_two() {
    assert_eq!(DEFAULT_CACHED_GRAMMARS, 32);
    let mut checker = checker();
    for number in 0..=DEFAULT_CACHED_GRAMMARS {
        checker.constraint(&grammar(number)).expect("compiles");
    }
    assert_eq!(checker.compiled.len(), DEFAULT_CACHED_GRAMMARS);
    assert!(!checker.compiled.holds(&grammar(0)));
    assert!(checker.compiled.holds(&grammar(1)));
}

#[test]
fn a_refused_grammar_is_not_cached_and_a_zero_bound_keeps_nothing() {
    let mut checker = checker();
    assert!(checker.constraint("start: nothing").is_err());
    assert_eq!(checker.compiled.len(), 0);
    let mut none = GrammarChecker::with_cache_size(&tokens(), &[END], 0).expect("vocabulary");
    let mut matcher = none.constraint(&grammar(0)).expect("compiles");
    assert!(feed(&mut matcher, "g0"));
    assert!(matcher.is_accepting());
    assert_eq!(none.compiled.len(), 0);
}
