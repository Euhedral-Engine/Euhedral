//! The llguidance grammars (Lark) that constrain an answer: a JSON document for `response_format`, or the
//! checkpoint's native tool calls, and the checks that llguidance accepts them.
//!
//! llguidance compiles each `%json` schema in strict mode, so a schema keyword it cannot enforce is refused rather
//! than ignored.
//!
//! - Whitespace between JSON tokens is bounded to runs of 40 bytes, so a model cannot pad forever.
//! - After reasoning the answer may begin with up to 8 whitespace bytes, the blank line the template puts after
//!   `</think>`; without reasoning it begins at its first byte.
//! - A call is spelled as the template renders it,
//!   `<tool_call>\n<function=NAME>\n(<parameter=KEY>\nVALUE\n</parameter>\n)*</function>\n</tool_call>`, and
//!   consecutive calls are separated by `\n`. A string VALUE is a terminal that includes its closing
//!   `\n</parameter>\n` and cannot contain `</parameter>`, so any other text, `<` and `</paramete` included, is
//!   admitted. A strict function's parameters follow its schema's `properties` in schema order: required ones are
//!   present, optional ones may be omitted, and a parameter the schema types as a string is text (an `enum` or
//!   `const` of strings, its literals). A parameter of another type is a `%json` of its schema, which keeps the
//!   function's `$defs`; an untyped one, or a union that includes a string, is any text. Names outside `properties`
//!   are generated only when the schema does not set `additionalProperties: false`, or for a required name it does
//!   not declare. A function that is not strict takes any parameter names and any text values.

use std::collections::BTreeSet;
use std::fmt::{Debug, Formatter, Result as FmtResult, Write};
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use llguidance::api::{GrammarInit, ParserLimits, TopLevelGrammar};
use llguidance::earley::SlicedBiasComputer;
use llguidance::panic_utils::catch_unwind;
use llguidance::{Logger, Matcher, ParserFactory};
use thiserror::Error;
use toktrie::{ApproximateTokEnv, InferenceCapabilities, TokEnv, TokRxInfo, TokTrie};

use crate::error::ApiError;
use crate::json::{JsonValue, put};
use crate::pipeline::ToolChoice;
use crate::response_format::ResponseFormat;
use crate::tool_calling::{FunctionTool, ToolCalling};

mod matcher;
mod vocabulary;

use matcher::CompiledGrammars;
pub use matcher::{DEFAULT_CACHED_GRAMMARS, GrammarMatcher, MatcherError};
pub use vocabulary::{end_ids, token_bytes};

/// The skipped whitespace between JSON tokens.
pub const WHITESPACE: &str = "[\\x20\\x0A\\x0D\\x09]{1,40}";

const ANNOTATIONS: [&str; 5] = ["description", "title", "default", "examples", "$comment"];
const UNENFORCEABLE_STRING_KEYWORDS: [&str; 4] = ["pattern", "format", "minLength", "maxLength"];

/// Text up to the first `</parameter>`, which it consumes together with the newline before it and the one after.
const TEXT: &str = "TEXT: (ANY & ~(ANY \"</parameter>\" ANY)) \"\\n</parameter>\\n\"\nANY: /[\\s\\S]*/\n";

/// Builds the grammars of an answer.
#[derive(Debug)]
pub struct OutputGrammar;

impl OutputGrammar {
    /// A JSON document of `schema`.
    pub fn json(schema: &JsonValue, reasoning: bool) -> String {
        format!(
            "{}answer: %json {}\n",
            start(reasoning, "answer"),
            Self::schema_text(schema)
        )
    }

    /// The grammar of a JSON response format; `None` for free text.
    pub fn for_response_format(format: &ResponseFormat, reasoning: bool) -> Option<String> {
        format.answer_schema().map(|schema| Self::json(&schema, reasoning))
    }

    /// The native calls of `calling`'s callable functions. `answer` is the schema of a direct JSON answer that may
    /// replace the calls: required for `ToolChoice::Auto` and refused for any other choice, which must call.
    pub fn for_tools(calling: &ToolCalling, answer: Option<&JsonValue>, reasoning: bool) -> Result<String, ApiError> {
        let auto = calling.choice == ToolChoice::Auto;
        if auto != answer.is_some() {
            // A caller bug, never a request's fault.
            return Err(ApiError::server_error());
        }
        let callable = calling.callable();
        let mut grammar = start(reasoning, if auto { "(calls | answer)" } else { "calls" });
        grammar.push_str("calls: call");
        if calling.parallel {
            grammar.push_str(" (\"\\n\" call)*");
        }
        grammar.push('\n');
        grammar.push_str("call: <tool_call> \"\\n\" (");
        for index in 0..callable.len() {
            if index > 0 {
                grammar.push_str(" | ");
            }
            let _ = write!(grammar, "function_{index}");
        }
        grammar.push_str(") </tool_call>\n");
        let mut rules = String::new();
        for (index, tool) in callable.iter().enumerate() {
            function(tool, index, &mut grammar, &mut rules)?;
        }
        grammar.push_str(&rules);
        if let Some(answer) = answer {
            let _ = writeln!(grammar, "answer: %json {}", Self::schema_text(answer));
        }
        grammar.push_str(TEXT);
        grammar.push_str("KEY: /[a-zA-Z0-9_-]+/\n");
        Ok(grammar)
    }

    /// The grammar that constrains a request's answer, or `None` for an answer that is free text, after checking
    /// every schema the request carries (the response schema and each callable strict tool's parameters, in that
    /// order) and the grammar itself with `checker`.
    pub fn for_request(
        calling: &ToolCalling,
        format: &ResponseFormat,
        reasoning: bool,
        checker: &dyn GrammarCheck,
    ) -> Result<Option<String>, ApiError> {
        if let ResponseFormat::JsonSchema {
            schema, schema_param, ..
        } = format
        {
            checker.check_json_schema(schema, schema_param)?;
        }
        // Free text under auto, which the parser splits; a call is forced by a required or named choice, and a JSON
        // answer beside auto calls is constrained as a whole.
        let auto = calling.choice == ToolChoice::Auto;
        for tool in calling.callable() {
            if tool.strict {
                checker.check_json_schema(&tool.parameters_or_empty(), &tool.parameters_param)?;
            }
        }
        let grammar = if calling.parses_output() && (!auto || format.is_json()) {
            let answer = if auto { format.answer_schema() } else { None };
            Self::for_tools(calling, answer.as_ref(), reasoning)?
        } else if calling.parses_output() {
            return Ok(None);
        } else if let Some(grammar) = Self::for_response_format(format, reasoning) {
            grammar
        } else {
            return Ok(None);
        };
        checker.check_grammar(&grammar)?;
        Ok(Some(grammar))
    }

    /// `schema` as llguidance reads it: compact JSON with the bounded whitespace option.
    pub fn schema_text(schema: &JsonValue) -> String {
        let mut options = vec![(
            "x-guidance".to_owned(),
            JsonValue::Object(vec![(
                "whitespace_pattern".to_owned(),
                JsonValue::String(WHITESPACE.to_owned()),
            )]),
        )];
        if let JsonValue::Object(members) = schema {
            for (key, value) in members {
                put(&mut options, key.clone(), value.clone());
            }
        }
        JsonValue::Object(options).to_compact_string()
    }
}

fn function(tool: &FunctionTool, index: usize, grammar: &mut String, rules: &mut String) -> Result<(), ApiError> {
    // Function names are [a-zA-Z0-9_-]{1,64}: nothing in them needs escaping.
    let _ = write!(grammar, "function_{index}: \"<function={}>\\n\"", tool.name());
    if !tool.strict {
        let _ = writeln!(grammar, " parameter_{index}* \"</function>\\n\"");
        let _ = writeln!(rules, "parameter_{index}: \"<parameter=\" KEY \">\\n\" TEXT");
        return Ok(());
    }
    let schema = tool.parameters_or_empty();
    let properties: &[(String, JsonValue)] = match schema.get("properties") {
        Some(JsonValue::Object(members)) => members,
        _ => &[],
    };
    let mut required: Vec<String> = Vec::new();
    for name in tool.definition.required_parameters() {
        if !required.contains(&name) {
            required.push(name);
        }
    }
    let mut parameter = 0;
    for (key, property) in properties {
        if *property == JsonValue::Bool(false) {
            continue;
        }
        let name = format!("parameter_{index}_{parameter}");
        parameter += 1;
        let optional = if required.contains(key) { "" } else { "?" };
        let _ = write!(grammar, " {name}{optional}");
        let _ = writeln!(
            rules,
            "{name}: \"<parameter=\" {} \">\\n\" {}",
            escaped(key),
            value(tool, &schema, key, property, &name)?
        );
    }
    for key in &required {
        if properties.iter().any(|(name, _)| name == key) {
            continue;
        }
        let name = format!("parameter_{index}_{parameter}");
        parameter += 1;
        let _ = write!(grammar, " {name}");
        let _ = writeln!(rules, "{name}: \"<parameter=\" {} \">\\n\" TEXT", escaped(key));
    }
    if schema.get("additionalProperties") != Some(&JsonValue::Bool(false)) {
        let _ = write!(grammar, " parameter_{index}*");
        let _ = writeln!(rules, "parameter_{index}: \"<parameter=\" KEY \">\\n\" TEXT");
    }
    grammar.push_str(" \"</function>\\n\"\n");
    Ok(())
}

/// The right-hand side of a parameter after its opening tag, including its closing tag.
fn value(
    tool: &FunctionTool,
    root: &JsonValue,
    key: &str,
    property: &JsonValue,
    rule: &str,
) -> Result<String, ApiError> {
    let JsonValue::Object(members) = property else {
        return Ok("TEXT".to_owned());
    };
    let types = tool.definition.parameter_types(key);
    if types == BTreeSet::from(["string".to_owned()]) {
        for keyword in UNENFORCEABLE_STRING_KEYWORDS {
            if property.get(keyword).is_some() {
                return Err(ApiError::unenforceable_schema(
                    format!(
                        "The JSON Schema cannot be enforced: a string parameter of a native tool call cannot use \
                         '{keyword}'; parameter '{key}' of {}.",
                        tool.name()
                    ),
                    Some(&tool.parameters_param),
                ));
            }
        }
        let single;
        let literals: &[JsonValue] = match (property.get("enum"), property.get("const")) {
            (Some(JsonValue::Array(values)), _) => values,
            (_, Some(constant)) => {
                single = [constant.clone()];
                &single
            }
            _ => return Ok("TEXT".to_owned()),
        };
        // Only a string can be the value of a string parameter: another literal is not offered, and a parameter
        // left with no literal at all cannot be enforced (an empty choice would admit an empty value).
        let choices: Vec<String> = literals
            .iter()
            .filter_map(|literal| match literal {
                JsonValue::String(text) => Some(escaped(text)),
                _ => None,
            })
            .collect();
        if choices.is_empty() {
            return Err(ApiError::unenforceable_schema(
                format!(
                    "The JSON Schema cannot be enforced: a string parameter of a native tool call needs a string \
                     in its 'enum' or 'const'; parameter '{key}' of {}.",
                    tool.name()
                ),
                Some(&tool.parameters_param),
            ));
        }
        return Ok(format!("({}) \"\\n</parameter>\\n\"", choices.join(" | ")));
    }
    let annotation_only = members.iter().all(|(name, _)| ANNOTATIONS.contains(&name.as_str()));
    if types.contains("string") || (types.is_empty() && annotation_only) {
        return Ok("TEXT".to_owned());
    }
    // The parameter's own schema, with the function's definitions so its references resolve.
    let mut standalone = members.clone();
    for definitions in ["$defs", "definitions"] {
        if let Some(found) = root.get(definitions)
            && !standalone.iter().any(|(name, _)| name == definitions)
        {
            standalone.push((definitions.to_owned(), found.clone()));
        }
    }
    Ok(format!(
        "{rule}_json \"\\n</parameter>\\n\"\n{rule}_json: %json {}",
        OutputGrammar::schema_text(&JsonValue::Object(standalone))
    ))
}

fn escaped(text: &str) -> String {
    JsonValue::String(text.to_owned()).to_compact_string()
}

fn start(reasoning: bool, rule: &str) -> String {
    if reasoning {
        format!("start: GAP? {rule}\nGAP: /[ \\t\\r\\n]{{1,8}}/\n")
    } else {
        format!("start: {rule}\n")
    }
}

/// A vocabulary llguidance cannot be built over.
#[derive(Debug, Error)]
#[error("llguidance refused the vocabulary: {0}")]
pub struct VocabularyError(String);

/// What compiles a request's grammar and schemas: the checker over the vocabulary, or the backend that holds it.
pub trait GrammarCheck {
    /// Whether the Lark grammar compiles; a refusal is a 400 without a parameter.
    fn check_grammar(&self, grammar: &str) -> Result<(), ApiError>;

    /// Whether `schema` compiles on its own; a refusal is an unenforceable schema at `param`.
    fn check_json_schema(&self, schema: &JsonValue, param: &str) -> Result<(), ApiError>;
}

impl GrammarCheck for GrammarChecker {
    fn check_grammar(&self, grammar: &str) -> Result<(), ApiError> {
        GrammarChecker::check_grammar(self, grammar)
    }

    fn check_json_schema(&self, schema: &JsonValue, param: &str) -> Result<(), ApiError> {
        GrammarChecker::check_json_schema(self, schema, param)
    }
}

/// Checks that grammars and JSON schemas compile over one vocabulary, the way the generation will compile them.
pub struct GrammarChecker {
    factory: ParserFactory,
    size: u32,
    end_ids: Arc<[u32]>,
    compiled: CompiledGrammars,
}

impl Debug for GrammarChecker {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("GrammarChecker").finish_non_exhaustive()
    }
}

impl GrammarChecker {
    /// A checker over the vocabulary whose token `id` is `token_bytes[id]`: its text for an ordinary token, `0xFF`
    /// and its name for a token a grammar may only name explicitly (a control token). Every ID needs bytes. A
    /// grammar ends with any of `end_ids`, the first of which is the main end token.
    ///
    /// What a check decides depends on the vocabulary: llguidance precomputes lexemes and enforces its lexer-state
    /// limits against the whole token trie, and `<tool_call>` is only a name in a grammar when its token is a `0xFF`
    /// control token. The caller therefore builds the vocabulary the generation will sample over, padded to the
    /// logits row's width with unused-token names as the native side does, and checks with that very checker.
    ///
    /// Compiling a grammar costs enough (a large schema or tool set takes noticeable CPU) that a caller runs a check
    /// as its own frame, never inline in a request handler.
    pub fn new(token_bytes: &[Vec<u8>], end_ids: &[u32]) -> Result<GrammarChecker, VocabularyError> {
        Self::with_cache_size(token_bytes, end_ids, DEFAULT_CACHED_GRAMMARS)
    }

    /// As [`GrammarChecker::new`], keeping up to `cached_grammars` compiled grammars for reuse by
    /// [`GrammarChecker::constraint`]; zero keeps none.
    pub fn with_cache_size(
        token_bytes: &[Vec<u8>],
        end_ids: &[u32],
        cached_grammars: usize,
    ) -> Result<GrammarChecker, VocabularyError> {
        let Some(&end) = end_ids.first() else {
            return Err(VocabularyError("a grammar needs an end token".to_owned()));
        };
        if token_bytes.is_empty() {
            return Err(VocabularyError("the vocabulary is empty".to_owned()));
        }
        if let Some(id) = token_bytes.iter().position(Vec::is_empty) {
            return Err(VocabularyError(format!("token {id} has no bytes")));
        }
        let size = u32::try_from(token_bytes.len()).map_err(|_| VocabularyError("too many tokens".to_owned()))?;
        if let Some(id) = end_ids.iter().find(|id| **id >= size) {
            return Err(VocabularyError(format!(
                "end token {id} is out of range (vocabulary size {size})"
            )));
        }
        let mut trie = TokTrie::from(&TokRxInfo::new(size, end), token_bytes);
        if end_ids.len() > 1 {
            trie = trie.with_eos_tokens(end_ids);
        }
        // Masks come from the token trie; tokenization only serves features this server does not use (forced
        // tokens, prompt healing), so llguidance's own greedy tokenization stands in for it.
        let environment: TokEnv = Arc::new(ApproximateTokEnv::new(trie));
        let factory = ParserFactory::new(
            &environment,
            InferenceCapabilities::default(),
            &SlicedBiasComputer::general_slices(),
        )
        .map_err(|error| VocabularyError(error.to_string()))?;
        Ok(GrammarChecker {
            factory,
            size,
            end_ids: end_ids.into(),
            compiled: CompiledGrammars::new(cached_grammars),
        })
    }

    /// Whether llguidance compiles the Lark grammar; the refusal is a 400 without a parameter.
    pub fn check_grammar(&self, grammar: &str) -> Result<(), ApiError> {
        self.compile(grammar, "lark").map_err(|refused| {
            ApiError::invalid_request(format!("The requested output cannot be enforced: {refused}"), None)
        })
    }

    /// Whether llguidance compiles `schema` on its own, so that a refusal names the schema's problem rather than a
    /// grammar around it; the refusal is an unenforceable schema at `param`.
    pub fn check_json_schema(&self, schema: &JsonValue, param: &str) -> Result<(), ApiError> {
        self.compile(&OutputGrammar::schema_text(schema), "json_schema")
            .map_err(|refused| {
                ApiError::unenforceable_schema(format!("The JSON Schema cannot be enforced: {refused}"), Some(param))
            })
    }

    /// Compiles `grammar` of the given tagged type as a matcher does, with nothing logged.
    fn compile(&self, grammar: &str, kind: &str) -> Result<(), String> {
        match self.matcher(grammar, kind).get_error() {
            None => Ok(()),
            Some(message) => Err(summary(&message)),
        }
    }

    fn matcher(&self, grammar: &str, kind: &str) -> Matcher {
        // A panic inside llguidance is a refusal ("panic: ..."), as on the native side.
        let compiled = catch_unwind(AssertUnwindSafe(|| {
            let parser = TopLevelGrammar::from_tagged_str(kind, grammar).and_then(|top| {
                self.factory.create_parser_from_init_ext(
                    GrammarInit::Serialized(top),
                    Logger::new(0, 0),
                    InferenceCapabilities::default(),
                    ParserLimits::default(),
                )
            });
            Ok(Matcher::new(parser))
        }));
        compiled.unwrap_or_else(|panicked| Matcher::new(Err(panicked)))
    }
}

/// llguidance's first line without its source position ("at 1(8): "); the rest quotes the grammar.
fn summary(message: &str) -> String {
    let first = message.lines().next().unwrap_or(message).trim();
    let rest = first.strip_prefix("at ").and_then(|position| {
        let (digits, rest) = position.split_once("): ")?;
        let (line, column) = digits.split_once('(')?;
        let numeric = |text: &str| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
        (numeric(line) && numeric(column)).then_some(rest)
    });
    rest.unwrap_or(first).to_owned()
}

#[cfg(test)]
mod real_vocabulary;
#[cfg(test)]
mod tests;
