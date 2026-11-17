use toktrie::TokenId;

use super::*;
use crate::tool_definition::ToolDefinition;

const WEATHER: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"units":{"enum":["metric","imperial"]}},
 "required":["city"],"additionalProperties":false}"#;
const OSLO: &str = "<parameter=city>\nOslo\n</parameter>\n";
const OPEN: TokenId = 256;
const CLOSE: TokenId = 257;
const END: TokenId = 258;

fn json(text: &str) -> JsonValue {
    JsonValue::parse(text).expect("test schema is JSON")
}

/// One token per byte, the two call tags as control tokens, and the end token.
fn checker() -> GrammarChecker {
    let mut tokens: Vec<Vec<u8>> = (0..=255u8).map(|byte| vec![byte]).collect();
    tokens.push(b"\xFF<tool_call>".to_vec());
    tokens.push(b"\xFF</tool_call>".to_vec());
    tokens.push(b"\xFF<|im_end|>".to_vec());
    GrammarChecker::new(&tokens, &[END]).expect("vocabulary")
}

fn tool(name: &str, schema: Option<&str>, strict: bool) -> FunctionTool {
    FunctionTool::new(
        ToolDefinition {
            name: name.to_owned(),
            description: None,
            parameters: schema.map(json),
        },
        strict,
    )
}

fn calling_of(tools: Vec<FunctionTool>, choice: ToolChoice, parallel: bool) -> ToolCalling {
    ToolCalling {
        tools,
        choice,
        parallel,
    }
}

fn weather(choice: ToolChoice, parallel: bool, strict: bool) -> ToolCalling {
    calling_of(vec![tool("get_weather", Some(WEATHER), strict)], choice, parallel)
}

fn call(name: &str, parameters: &str) -> String {
    format!("<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>")
}

/// The token ids of `text`: the two call tags are control tokens, every other byte its own token.
fn encode(text: &str) -> Vec<TokenId> {
    let mut ids = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let (token, length) = if rest.starts_with("<tool_call>") {
            (OPEN, "<tool_call>".len())
        } else if rest.starts_with("</tool_call>") {
            (CLOSE, "</tool_call>".len())
        } else {
            (TokenId::from(rest.as_bytes()[0]), 1)
        };
        ids.push(token);
        rest = &rest[length..];
    }
    ids
}

/// Whether a generation could produce exactly `text` and then end.
fn finishes(grammar: &str, text: &str) -> bool {
    let mut checker = checker();
    let mut matcher = checker.constraint(grammar).expect(grammar);
    for id in encode(text) {
        if !matcher.allows(id).expect("mask") {
            return false;
        }
        matcher.consume(id).expect("allowed");
    }
    matcher.allows(END).expect("mask")
}

fn tools(calling: &ToolCalling, answer: Option<&JsonValue>, reasoning: bool) -> String {
    let grammar = OutputGrammar::for_tools(calling, answer, reasoning).expect("grammar");
    checker().check_grammar(&grammar).expect("compiles");
    grammar
}

#[test]
fn auto_with_a_json_answer_allows_a_call_or_the_document_and_nothing_else() {
    let answer = json(r#"{"type":"object","properties":{"answer":{"type":"integer"}},"required":["answer"]}"#);
    let calling = weather(ToolChoice::Auto, true, false);
    let grammar = tools(&calling, Some(&answer), false);
    assert!(finishes(&grammar, &call("get_weather", OSLO)));
    assert!(finishes(&grammar, "{\"answer\":42}"));
    assert!(!finishes(&grammar, "{\"answer\":\"42\"}"));
    assert!(!finishes(&grammar, "It is cold."));
    assert!(!finishes(&grammar, &call("get_time", "")), "only offered functions");
    assert!(
        !finishes(&grammar, " {\"answer\":42}"),
        "without reasoning the answer starts at once"
    );
    let reasoning = tools(&calling, Some(&answer), true);
    assert!(finishes(&reasoning, &format!("\n\n{}", call("get_weather", OSLO))));
    assert!(finishes(&reasoning, "\n\n{\"answer\":1}"));
}

#[test]
fn a_required_call_offers_no_answer() {
    let grammar = tools(&weather(ToolChoice::Required, true, false), None, false);
    assert!(finishes(&grammar, &call("get_weather", OSLO)));
    assert!(!finishes(&grammar, "It is cold."));
    assert!(!finishes(&grammar, ""));
}

#[test]
fn a_non_strict_function_takes_any_names_and_text_values() {
    let grammar = tools(&weather(ToolChoice::Required, true, false), None, false);
    assert!(finishes(&grammar, &call("get_weather", "")));
    let more = format!("{OSLO}<parameter=x-1>\n[1, 2]\n</parameter>\n");
    assert!(finishes(&grammar, &call("get_weather", &more)));
    assert!(finishes(
        &grammar,
        &call("get_weather", "<parameter=a>\n\n</parameter>\n")
    ));
    assert!(finishes(
        &grammar,
        &call("get_weather", "<parameter=a>\nline\nline two\n</parameter>\n")
    ));
    assert!(
        !finishes(&grammar, &call("get_weather", "<parameter=a b>\nx\n</parameter>\n")),
        "key characters"
    );
    assert!(
        !finishes(&grammar, "<tool_call>\n<function=get_weather>\n</function>"),
        "unclosed call"
    );
}

#[test]
fn a_string_value_cannot_contain_the_closing_tag() {
    let grammar = tools(&weather(ToolChoice::Required, true, false), None, false);
    for (text, allowed) in [
        ("<parameter=a>\nx</parameter>y\n</parameter>\n", false),
        ("<parameter=a>\nx</parameter>\ny\n</parameter>\n", false),
        ("<parameter=a>\nx</paramete\n</parameter>\n", true),
        ("<parameter=a>\nif a < b && c</d> then\n</parameter>\n", true),
        ("<parameter=a>\n</parameter\n</parameter>\n", true),
    ] {
        assert_eq!(finishes(&grammar, &call("get_weather", text)), allowed, "{text}");
    }
    let strict = tools(&weather(ToolChoice::Required, true, true), None, false);
    let closed = "<parameter=city>\nOs</parameter>lo\n</parameter>\n";
    assert!(!finishes(&strict, &call("get_weather", closed)));
    let open = "<parameter=city>\nOs</paramete lo<\n</parameter>\n";
    assert!(finishes(&strict, &call("get_weather", open)));
}

#[test]
fn strict_parameters_follow_the_schema_in_order() {
    let grammar = tools(&weather(ToolChoice::Required, true, true), None, false);
    let units = "<parameter=units>\nmetric\n</parameter>\n";
    assert!(finishes(&grammar, &call("get_weather", &format!("{OSLO}{units}"))));
    assert!(finishes(&grammar, &call("get_weather", OSLO)));
    assert!(
        !finishes(&grammar, &call("get_weather", units)),
        "required property missing"
    );
    assert!(
        !finishes(&grammar, &call("get_weather", &format!("{units}{OSLO}"))),
        "schema order"
    );
    let kelvin = format!("{OSLO}<parameter=units>\nkelvin\n</parameter>\n");
    assert!(!finishes(&grammar, &call("get_weather", &kelvin)), "enum");
    let wind = format!("{OSLO}<parameter=wind>\n1\n</parameter>\n");
    assert!(!finishes(&grammar, &call("get_weather", &wind)), "extra");
    assert!(
        !finishes(&grammar, &call("get_weather", &format!("{OSLO}{OSLO}"))),
        "repeated"
    );
}

#[test]
fn strict_non_string_parameters_are_json_of_their_schema() {
    let schema = r##"{"type":"object","properties":{"count":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}},
 "note":{"type":["string","null"]},"free":{},"mode":{"$ref":"#/$defs/mode"}},
 "required":["count","mode"],"additionalProperties":false,
 "$defs":{"mode":{"type":"integer","enum":[1,2]}}}"##;
    let calling = calling_of(
        vec![tool("record", Some(schema), true)],
        ToolChoice::Function("record".to_owned()),
        true,
    );
    let grammar = tools(&calling, None, false);
    let mode = "<parameter=mode>\n2\n</parameter>\n";
    let count = "<parameter=count>\n3\n</parameter>\n";
    assert!(finishes(&grammar, &call("record", &format!("{count}{mode}"))));
    let everything = format!(
        "{count}<parameter=tags>\n[\"a\", \"b\"]\n</parameter>\n<parameter=note>\nfree text\n</parameter>\n\
         <parameter=free>\nx y\n</parameter>\n{mode}"
    );
    assert!(finishes(&grammar, &call("record", &everything)));
    let three = format!("<parameter=count>\nthree\n</parameter>\n{mode}");
    assert!(!finishes(&grammar, &call("record", &three)));
    let five = format!("{count}<parameter=mode>\n5\n</parameter>\n");
    assert!(!finishes(&grammar, &call("record", &five)), "$ref enum");
    assert!(!finishes(&grammar, &call("record", count)), "required property missing");
}

#[test]
fn a_named_function_is_the_only_choice() {
    let functions = || vec![tool("get_weather", Some(WEATHER), true), tool("get_time", None, false)];
    let named = calling_of(functions(), ToolChoice::Function("get_time".to_owned()), true);
    let grammar = tools(&named, None, false);
    assert!(finishes(&grammar, &call("get_time", "")));
    assert!(!finishes(&grammar, &call("get_weather", OSLO)));
    let either = tools(&calling_of(functions(), ToolChoice::Required, true), None, false);
    assert!(finishes(&either, &call("get_weather", OSLO)));
    assert!(finishes(&either, &call("get_time", "")));
}

#[test]
fn parallel_calls_are_separated_by_a_newline() {
    let one = call("get_weather", OSLO);
    let single = tools(&weather(ToolChoice::Required, false, true), None, false);
    assert!(finishes(&single, &one));
    assert!(!finishes(&single, &format!("{one}\n{one}")));
    let parallel = tools(&weather(ToolChoice::Required, true, true), None, false);
    assert!(finishes(&parallel, &format!("{one}\n{one}")));
    assert!(!finishes(&parallel, &format!("{one}{one}")));
}

#[test]
fn strict_schemas_keep_their_own_references() {
    let recursive = r##"{"type":"object","$defs":{"node":{"type":"object","properties":{"v":{"type":"integer"},
 "next":{"$ref":"#/$defs/node"}},"required":["v"],"additionalProperties":false}},
 "properties":{"head":{"$ref":"#/$defs/node"}},"required":["head"],"additionalProperties":false}"##;
    let calling = calling_of(vec![tool("walk", Some(recursive), true)], ToolChoice::Required, true);
    let grammar = tools(&calling, None, false);
    let good = "<parameter=head>\n{\"v\":1,\"next\":{\"v\":2}}\n</parameter>\n";
    assert!(finishes(&grammar, &call("walk", good)));
    let bad = "<parameter=head>\n{\"v\":1,\"next\":{\"w\":2}}\n</parameter>\n";
    assert!(!finishes(&grammar, &call("walk", bad)));
}

#[test]
fn a_string_keyword_the_grammar_cannot_enforce_is_refused() {
    for keyword in [
        "\"pattern\":\"^a+$\"",
        "\"format\":\"uuid\"",
        "\"minLength\":1",
        "\"maxLength\":9",
    ] {
        let schema = format!(
            "{{\"type\":\"object\",\"properties\":{{\"id\":{{\"type\":\"string\",{keyword}}}}},\"required\":[\"id\"]}}"
        );
        let calling = calling_of(vec![tool("lookup", Some(&schema), true)], ToolChoice::Required, true);
        let refused = OutputGrammar::for_tools(&calling, None, false).expect_err("refused");
        let name = keyword.split('"').nth(1).unwrap_or_default();
        assert_eq!(refused.status, 400);
        assert_eq!(refused.code, Some("unsupported_schema"));
        assert_eq!(refused.param.as_deref(), Some("parameters"));
        assert_eq!(
            refused.message,
            format!(
                "The JSON Schema cannot be enforced: a string parameter of a native tool call cannot use \
                 '{name}'; parameter 'id' of lookup."
            )
        );
    }
}

#[test]
fn after_reasoning_the_answer_may_follow_a_blank_line() {
    let grammar = OutputGrammar::json(&json(r#"{"type":"array","items":{"type":"number"}}"#), true);
    assert!(finishes(&grammar, "\n\n[1, 2.5]"));
    assert!(
        !finishes(&grammar, &format!("{}[1]", " ".repeat(9))),
        "the gap is bounded"
    );
    let plain = OutputGrammar::json(&json(r#"{"type":"array"}"#), false);
    assert!(!finishes(&plain, "\n[1]"));
    assert!(finishes(&plain, "[1,\n  2]"));
    assert!(
        !finishes(&plain, &format!("[1,{}2]", " ".repeat(41))),
        "whitespace runs are bounded"
    );
}

#[test]
fn the_grammar_text_of_a_json_answer() {
    let format = ResponseFormat::JsonObject;
    assert_eq!(
        OutputGrammar::for_response_format(&format, false).as_deref(),
        Some(
            "start: answer\nanswer: %json {\"x-guidance\":{\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,40}\"},\
             \"type\":\"object\"}\n"
        )
    );
    assert_eq!(
        OutputGrammar::for_response_format(&format, true).as_deref(),
        Some(
            "start: GAP? answer\nGAP: /[ \\t\\r\\n]{1,8}/\nanswer: %json \
             {\"x-guidance\":{\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,40}\"},\"type\":\"object\"}\n"
        )
    );
    assert_eq!(OutputGrammar::for_response_format(&ResponseFormat::Text, false), None);
}

#[test]
fn the_grammar_text_of_tool_calls() {
    let guidance = "{\"x-guidance\":{\"whitespace_pattern\":\"[\\\\x20\\\\x0A\\\\x0D\\\\x09]{1,40}\"}";
    let text =
        "TEXT: (ANY & ~(ANY \"</parameter>\" ANY)) \"\\n</parameter>\\n\"\nANY: /[\\s\\S]*/\nKEY: /[a-zA-Z0-9_-]+/\n";
    let open = weather(ToolChoice::Required, false, false);
    assert_eq!(
        OutputGrammar::for_tools(&open, None, false).expect("grammar"),
        format!(
            "start: calls\ncalls: call\ncall: <tool_call> \"\\n\" (function_0) </tool_call>\n\
             function_0: \"<function=get_weather>\\n\" parameter_0* \"</function>\\n\"\n\
             parameter_0: \"<parameter=\" KEY \">\\n\" TEXT\n{text}"
        )
    );
    let strict = weather(ToolChoice::Auto, true, true);
    let answer = json(r#"{"type":"integer"}"#);
    assert_eq!(
        OutputGrammar::for_tools(&strict, Some(&answer), true).expect("grammar"),
        format!(
            "start: GAP? (calls | answer)\nGAP: /[ \\t\\r\\n]{{1,8}}/\ncalls: call (\"\\n\" call)*\n\
             call: <tool_call> \"\\n\" (function_0) </tool_call>\n\
             function_0: \"<function=get_weather>\\n\" parameter_0_0 parameter_0_1? \"</function>\\n\"\n\
             parameter_0_0: \"<parameter=\" \"city\" \">\\n\" TEXT\n\
             parameter_0_1: \"<parameter=\" \"units\" \">\\n\" (\"metric\" | \"imperial\") \"\\n</parameter>\\n\"\n\
             answer: %json {guidance},\"type\":\"integer\"}}\n{text}"
        )
    );
    // The schema of a parameter that is not text is a rule of its own, with the function's definitions.
    let schema = r##"{"type":"object","properties":{"n":{"$ref":"#/$defs/n"}},"required":["n","extra"],"$defs":{"n":{"type":"integer"}},"additionalProperties":false}"##;
    let calling = calling_of(vec![tool("f", Some(schema), true)], ToolChoice::Required, true);
    assert_eq!(
        OutputGrammar::for_tools(&calling, None, false).expect("grammar"),
        format!(
            "start: calls\ncalls: call (\"\\n\" call)*\ncall: <tool_call> \"\\n\" (function_0) </tool_call>\n\
             function_0: \"<function=f>\\n\" parameter_0_0 parameter_0_1 \"</function>\\n\"\n\
             parameter_0_0: \"<parameter=\" \"n\" \">\\n\" parameter_0_0_json \"\\n</parameter>\\n\"\n\
             parameter_0_0_json: %json {guidance},\"$ref\":\"#/$defs/n\",\"$defs\":{{\"n\":{{\"type\":\"integer\"}}}}}}\n\
             parameter_0_1: \"<parameter=\" \"extra\" \">\\n\" TEXT\n{text}"
        )
    );
}

#[test]
fn only_auto_takes_an_answer_schema() {
    let answer = json(r#"{"type":"integer"}"#);
    assert!(OutputGrammar::for_tools(&weather(ToolChoice::Required, true, true), Some(&answer), false).is_err());
    assert!(OutputGrammar::for_tools(&weather(ToolChoice::Auto, true, true), None, false).is_err());
}

#[test]
fn a_string_enum_skips_literals_that_are_not_strings() {
    let schema = r#"{"type":"object","properties":{"mode":{"type":"string","enum":["a",1,"b"]}},"required":["mode"],"additionalProperties":false}"#;
    let calling = calling_of(vec![tool("f", Some(schema), true)], ToolChoice::Required, true);
    let grammar = tools(&calling, None, false);
    assert!(grammar.contains("(\"a\" | \"b\") \"\\n</parameter>\\n\""), "{grammar}");
}

#[test]
fn a_request_gets_the_grammar_of_its_format_and_tools() {
    let mut checker = checker();
    let plain = ToolCalling::disabled();
    let schema = ResponseFormat::JsonSchema {
        name: Some("p".to_owned()),
        schema: json(r#"{"type":"object","properties":{"age":{"type":"integer"}},"required":["age"]}"#),
        schema_param: "response_format.json_schema.schema".to_owned(),
    };
    let grammar = OutputGrammar::for_request(&plain, &schema, false, &mut checker).expect("planned");
    let grammar = grammar.expect("a grammar");
    assert!(
        grammar.starts_with("start: answer\nanswer: %json {\"x-guidance\":"),
        "{grammar}"
    );
    assert!(grammar.contains("\"required\":[\"age\"]"), "{grammar}");
    let text = OutputGrammar::for_request(&plain, &ResponseFormat::Text, false, &mut checker);
    assert_eq!(text.expect("planned"), None);
    // Auto calls under free text are split by the parser, not constrained.
    let auto = weather(ToolChoice::Auto, true, true);
    assert_eq!(
        OutputGrammar::for_request(&auto, &ResponseFormat::Text, false, &mut checker).expect("planned"),
        None
    );
    let with_answer = OutputGrammar::for_request(&auto, &schema, false, &mut checker)
        .expect("planned")
        .expect("a grammar");
    assert!(with_answer.contains("start: (calls | answer)"), "{with_answer}");
    assert!(with_answer.contains("answer: %json {\"x-guidance\":"), "{with_answer}");
    // A strict tool's schema is compiled as it is.
    let bounded = r#"{"type":"object","properties":{"n":{"type":"integer","minimum":0}},"required":["n"]}"#;
    let strict = calling_of(vec![tool("f", Some(bounded), true)], ToolChoice::Required, true);
    let grammar = OutputGrammar::for_request(&strict, &ResponseFormat::Text, false, &mut checker)
        .expect("planned")
        .expect("a grammar");
    assert!(grammar.contains("\"minimum\":0"), "{grammar}");
}

#[test]
fn a_schema_llguidance_refuses_is_an_unenforceable_schema_at_its_parameter() {
    let mut checker = checker();
    let param = "response_format.json_schema.schema";
    for (schema, message) in [
        (
            r#"{"type":"object","if":{}}"#,
            "The JSON Schema cannot be enforced: Unimplemented keys: [\"if\"]",
        ),
        (
            r#"{"type":"integer","minimum":3,"maximum":2}"#,
            "The JSON Schema cannot be enforced: Unsatisfiable schema: minimum (3) is greater than maximum (2)",
        ),
    ] {
        let format = ResponseFormat::JsonSchema {
            name: Some("p".to_owned()),
            schema: json(schema),
            schema_param: param.to_owned(),
        };
        let refused =
            OutputGrammar::for_request(&ToolCalling::disabled(), &format, false, &mut checker).expect_err("refused");
        assert_eq!(refused.status, 400);
        assert_eq!(refused.code, Some("unsupported_schema"));
        assert_eq!(refused.param.as_deref(), Some(param));
        assert_eq!(refused.message, message);
    }
}

#[test]
fn an_invalid_grammar_is_a_bad_request_without_a_parameter() {
    let refused = checker().check_grammar("start: missing_rule\n").expect_err("refused");
    assert_eq!(refused.status, 400);
    assert_eq!(refused.code, None);
    assert_eq!(refused.param, None);
    assert_eq!(
        refused.message,
        "The requested output cannot be enforced: unknown name: \"missing_rule\""
    );
    assert!(checker().check_grammar("start: \"a\"\n").is_ok());
}

#[test]
fn a_string_parameter_with_no_string_literal_is_refused() {
    for property in [
        r#"{"type":"string","enum":[1,2]}"#,
        r#"{"type":"string","const":null}"#,
        r#"{"type":"string","const":"x","enum":[1]}"#,
        r#"{"type":"string","enum":[]}"#,
    ] {
        let schema = format!(r#"{{"type":"object","properties":{{"mode":{property}}},"required":["mode"]}}"#);
        let calling = calling_of(vec![tool("f", Some(&schema), true)], ToolChoice::Required, true);
        let refused = OutputGrammar::for_tools(&calling, None, false).expect_err(property);
        assert_eq!(refused.status, 400, "{property}");
        assert_eq!(refused.code, Some("unsupported_schema"), "{property}");
        assert_eq!(refused.param.as_deref(), Some("parameters"), "{property}");
        assert_eq!(
            refused.message,
            "The JSON Schema cannot be enforced: a string parameter of a native tool call needs a string in its \
             'enum' or 'const'; parameter 'mode' of f.",
            "{property}"
        );
    }
}

#[test]
fn a_summary_drops_the_position_and_the_quoted_grammar() {
    assert_eq!(summary("at 1(8): unknown rule\n  start: x"), "unknown rule");
    assert_eq!(summary("no position here\nmore"), "no position here");
    assert_eq!(summary("at x(1): kept"), "at x(1): kept");
}

#[test]
fn a_vocabulary_without_an_end_token_or_bytes_is_refused() {
    let tokens = vec![b"a".to_vec(), b"b".to_vec()];
    assert!(GrammarChecker::new(&tokens, &[]).is_err());
    assert!(GrammarChecker::new(&tokens, &[2]).is_err());
    assert!(GrammarChecker::new(&[b"a".to_vec(), Vec::new()], &[0]).is_err());
    assert!(GrammarChecker::new(&[], &[0]).is_err());
}
