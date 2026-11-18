use super::*;
use crate::json::JsonInteger;
use crate::template::Role;

fn defaults() -> Defaults {
    Defaults {
        model: "m".to_owned(),
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        max_tokens: 4096,
    }
}

fn map_body(body: &str) -> Result<GenerationRequest, ApiError> {
    map_messages(MessagesRequest::parse(body).expect("the body parses"), &defaults())
}

fn count_body(body: &str) -> Result<GenerationRequest, ApiError> {
    map_count_tokens(MessagesRequest::parse(body).expect("the body parses"), &defaults())
}

fn accepted(body: &str) -> GenerationRequest {
    map_body(body).unwrap_or_else(|error| panic!("{body} was rejected: {}", error.message))
}

/// A request of one user message with `rest` appended to its members (`,"key":value`).
fn req(rest: &str) -> String {
    format!(r#"{{"model":"m","max_tokens":64,"messages":[{{"role":"user","content":"hi"}}]{rest}}}"#)
}

/// A request with the given `messages` array.
fn msgs(messages: &str) -> String {
    format!(r#"{{"model":"m","max_tokens":64,"messages":{messages}}}"#)
}

/// A request whose messages are one user message made of `content`.
fn user(content: &str) -> String {
    msgs(&format!(r#"[{{"role":"user","content":{content}}}]"#))
}

/// A request that follows an assistant message of `assistant` content with the user message of `content`.
fn after(assistant: &str, content: &str) -> String {
    msgs(&format!(
        r#"[{{"role":"user","content":"q"}},{{"role":"assistant","content":{assistant}}},{{"role":"user","content":{content}}}]"#
    ))
}

const USE: &str = r#"[{"type":"tool_use","id":"t1","name":"f","input":{"a":1}}]"#;

fn with_tool(tool: &str) -> String {
    req(&format!(r#","tools":[{tool}]"#))
}

fn with_choice(choice: &str) -> String {
    req(&format!(
        r#","tools":[{{"name":"f","input_schema":{{}}}}],"tool_choice":{choice}"#
    ))
}

struct Row {
    name: &'static str,
    body: String,
    count: bool,
    code: Option<&'static str>,
    status: u16,
    param: Option<String>,
    message: String,
}

fn bad(name: &'static str, body: String, param: &str, message: &str) -> Row {
    Row {
        name,
        body,
        count: false,
        code: None,
        status: 400,
        param: Some(param.to_owned()),
        message: message.to_owned(),
    }
}

fn unsupported(name: &'static str, body: String, param: &str) -> Row {
    Row {
        name,
        body,
        count: false,
        code: Some("unsupported_parameter"),
        status: 400,
        param: Some(param.to_owned()),
        message: format!("Unsupported parameter: '{param}' is not supported by this server."),
    }
}

fn unrecognized(name: &'static str, body: String, param: &str) -> Row {
    Row {
        name,
        body,
        count: false,
        code: Some("unrecognized_argument"),
        status: 400,
        param: Some(param.to_owned()),
        message: format!("Unrecognized request argument supplied: {param}"),
    }
}

impl Row {
    fn counting(mut self) -> Row {
        self.count = true;
        self
    }
}

fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    // Top level.
    for field in ["mcp_servers", "container", "context_management"] {
        rows.push(unsupported(
            "unsupported field",
            req(&format!(r#","{field}":[]"#)),
            field,
        ));
    }
    rows.push(unrecognized("unknown field", req(r#","bogus":1"#), "bogus"));
    rows.push(unrecognized("unknown null field", req(r#","bogus":null"#), "bogus"));
    rows.push(bad(
        "model missing",
        r#"{"max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
        "model",
        "model: Field required",
    ));
    rows.push(bad(
        "model blank",
        r#"{"model":" \t","max_tokens":5,"messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
        "model",
        "model: Field required",
    ));
    rows.push(Row {
        name: "model unknown",
        body: r#"{"model":"claude","max_tokens":5,"messages":[]}"#.to_owned(),
        count: false,
        code: Some("model_not_found"),
        status: 404,
        param: Some("model".to_owned()),
        message: "The model 'claude' does not exist or you do not have access to it.".to_owned(),
    });
    rows.push(bad(
        "max_tokens missing",
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
        "max_tokens",
        "max_tokens: Field required",
    ));
    rows.push(bad(
        "max_tokens zero",
        r#"{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
        "max_tokens",
        "max_tokens: must be at least 1",
    ));
    rows.push(unrecognized("count tokens takes no max_tokens", req(""), "max_tokens").counting());
    rows.push(
        unrecognized(
            "count tokens takes no stream",
            r#"{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
            "stream",
        )
        .counting(),
    );
    // Tools.
    rows.push(bad(
        "tools not a list",
        req(r#","tools":{}"#),
        "tools",
        "tools: must be a list",
    ));
    rows.push(bad(
        "tool not an object",
        req(r#","tools":[1]"#),
        "tools[0]",
        "tools[0]: must be an object",
    ));
    rows.push(unsupported(
        "tool of a server type",
        with_tool(r#"{"type":"web_search_20250305","name":"w"}"#),
        "tools[0].type=web_search_20250305",
    ));
    rows.push(unsupported(
        "tool type that is not text",
        with_tool(r#"{"type":["a",1,{"k":true}],"name":"w"}"#),
        "tools[0].type=[a, 1, {k=true}]",
    ));
    rows.push(unrecognized(
        "tool key",
        with_tool(r#"{"name":"f","input_schema":{},"x":1}"#),
        "tools[0].x",
    ));
    let name = "tools[0].name";
    let name_rule = "tools[0].name: must match ^[a-zA-Z0-9_-]{1,128}$";
    rows.push(bad(
        "tool name missing",
        with_tool(r#"{"input_schema":{}}"#),
        name,
        name_rule,
    ));
    rows.push(bad(
        "tool name with a space",
        with_tool(r#"{"name":"a b","input_schema":{}}"#),
        name,
        name_rule,
    ));
    rows.push(bad(
        "tool name too long",
        with_tool(&format!(r#"{{"name":"{}","input_schema":{{}}}}"#, "a".repeat(129))),
        name,
        name_rule,
    ));
    rows.push(bad(
        "tool name twice",
        req(r#","tools":[{"name":"f","input_schema":{}},{"name":"f","input_schema":{}}]"#),
        "tools[1].name",
        "tools[1].name: tool names must be unique",
    ));
    rows.push(bad(
        "tool description",
        with_tool(r#"{"name":"f","description":1,"input_schema":{}}"#),
        "tools[0].description",
        "tools[0].description: must be a string",
    ));
    rows.push(bad(
        "tool strict",
        with_tool(r#"{"name":"f","strict":"yes","input_schema":{}}"#),
        "tools[0].strict",
        "tools[0].strict: must be a boolean",
    ));
    rows.push(bad(
        "input_schema missing",
        with_tool(r#"{"name":"f"}"#),
        "tools[0].input_schema",
        "tools[0].input_schema: Field required",
    ));
    let schema = "tools[0].input_schema";
    let schema_row = |name: &'static str, schema_json: &str, param: &str, message: &str| {
        bad(
            name,
            with_tool(&format!(r#"{{"name":"f","input_schema":{schema_json}}}"#)),
            param,
            message,
        )
    };
    rows.push(schema_row(
        "input_schema not an object",
        "[]",
        schema,
        "Function 'parameters' must be a JSON Schema object.",
    ));
    rows.push(schema_row(
        "input_schema type",
        r#"{"type":"string"}"#,
        "tools[0].input_schema.type",
        "Function 'parameters' must describe an object; tool calls pass named arguments.",
    ));
    rows.push(schema_row(
        "input_schema properties",
        r#"{"properties":[]}"#,
        "tools[0].input_schema.properties",
        "'properties' must be an object.",
    ));
    rows.push(schema_row(
        "input_schema property name",
        r#"{"properties":{"a<b":{}}}"#,
        "tools[0].input_schema.properties",
        "Parameter names must be non-empty and free of '<', '>' and line breaks.",
    ));
    rows.push(schema_row(
        "input_schema required",
        r#"{"required":"a"}"#,
        "tools[0].input_schema.required",
        "'required' must be an array of strings.",
    ));
    rows.push(schema_row(
        "input_schema required name",
        r#"{"required":["a","a"]}"#,
        "tools[0].input_schema.required",
        "Required parameter names must be unique, non-empty and free of '<', '>' and line breaks.",
    ));
    rows.push(schema_row(
        "input_schema required undeclared",
        r#"{"required":["a"],"additionalProperties":false}"#,
        "tools[0].input_schema.required",
        "Required parameters must be declared when additional properties are forbidden.",
    ));
    // Tool choice.
    rows.push(bad(
        "tool_choice not an object",
        with_choice(r#""auto""#),
        "tool_choice",
        "tool_choice: must be an object",
    ));
    rows.push(unrecognized(
        "tool_choice key",
        with_choice(r#"{"type":"auto","x":1}"#),
        "tool_choice.x",
    ));
    rows.push(bad(
        "disable_parallel_tool_use",
        with_choice(r#"{"type":"auto","disable_parallel_tool_use":"yes"}"#),
        "tool_choice.disable_parallel_tool_use",
        "tool_choice.disable_parallel_tool_use: must be a boolean",
    ));
    rows.push(bad(
        "tool_choice without tools",
        req(r#","tool_choice":{"type":"auto"}"#),
        "tool_choice",
        "tool_choice: requires tools",
    ));
    rows.push(bad(
        "tool_choice with an empty tool list",
        req(r#","tools":[],"tool_choice":{"type":"any"}"#),
        "tool_choice",
        "tool_choice: requires tools",
    ));
    rows.push(bad(
        "tool_choice name",
        with_choice(r#"{"type":"tool","name":"g"}"#),
        "tool_choice.name",
        "tool_choice.name: must name one of the tools",
    ));
    rows.push(bad(
        "tool_choice name missing",
        with_choice(r#"{"type":"tool"}"#),
        "tool_choice.name",
        "tool_choice.name: must name one of the tools",
    ));
    rows.push(bad(
        "tool_choice type missing",
        with_choice(r#"{}"#),
        "tool_choice.type",
        "tool_choice.type: Field required",
    ));
    rows.push(bad(
        "tool_choice type",
        with_choice(r#"{"type":"some"}"#),
        "tool_choice.type",
        "tool_choice.type: must be auto, any, tool or none",
    ));
    // Thinking.
    let effort_needs = "output_config.effort: selects how much the model thinks, which needs thinking enabled";
    rows.push(bad(
        "effort without thinking",
        req(r#","output_config":{"effort":"low"}"#),
        "output_config.effort",
        effort_needs,
    ));
    rows.push(bad(
        "effort with thinking disabled",
        req(r#","thinking":{"type":"disabled"},"output_config":{"effort":"low"}"#),
        "output_config.effort",
        effort_needs,
    ));
    rows.push(bad(
        "effort value",
        req(r#","thinking":{"type":"adaptive"},"output_config":{"effort":"extreme"}"#),
        "output_config.effort",
        "output_config.effort: must be low, medium, high or max",
    ));
    rows.push(bad(
        "effort that is no string",
        req(r#","thinking":{"type":"adaptive"},"output_config":{"effort":1}"#),
        "output_config.effort",
        "output_config.effort: must be low, medium, high or max",
    ));
    rows.push(bad(
        "thinking not an object",
        req(r#","thinking":"on""#),
        "thinking",
        "thinking: must be an object",
    ));
    for (name, thinking) in [
        ("disabled", r#"{"type":"disabled","budget_tokens":5}"#),
        ("adaptive", r#"{"type":"adaptive","budget_tokens":5}"#),
        ("enabled", r#"{"type":"enabled","budget_tokens":5,"x":1}"#),
    ] {
        let key = if name == "enabled" { "x" } else { "budget_tokens" };
        rows.push(unrecognized(
            "thinking key",
            req(&format!(r#","thinking":{thinking}"#)),
            &format!("thinking.{key}"),
        ));
    }
    for (name, budget) in [
        ("budget missing", ""),
        ("budget zero", r#","budget_tokens":0"#),
        ("budget fraction", r#","budget_tokens":5.5"#),
        ("budget text", r#","budget_tokens":"5""#),
        ("budget beyond an int", r#","budget_tokens":3000000000"#),
    ] {
        rows.push(bad(
            name,
            req(&format!(r#","thinking":{{"type":"enabled"{budget}}}"#)),
            "thinking.budget_tokens",
            "thinking.budget_tokens: must be a positive integer",
        ));
    }
    rows.push(bad(
        "budget not below max_tokens",
        req(r#","thinking":{"type":"enabled","budget_tokens":64}"#),
        "thinking.budget_tokens",
        "thinking.budget_tokens: must be less than max_tokens",
    ));
    rows.push(bad(
        "thinking type missing",
        req(r#","thinking":{}"#),
        "thinking.type",
        "thinking.type: Field required",
    ));
    rows.push(bad(
        "thinking type that is no string",
        req(r#","thinking":{"type":1}"#),
        "thinking.type",
        "thinking.type: Field required",
    ));
    rows.push(bad(
        "thinking type",
        req(r#","thinking":{"type":"on"}"#),
        "thinking.type",
        "thinking.type: must be enabled, adaptive or disabled",
    ));
    // Output format.
    rows.push(bad(
        "output_config not an object",
        req(r#","output_config":[]"#),
        "output_config",
        "output_config: must be an object",
    ));
    rows.push(unrecognized(
        "output_config key",
        req(r#","output_config":{"x":1}"#),
        "output_config.x",
    ));
    let schema_format = r#"{"type":"json_schema","schema":{}}"#;
    rows.push(bad(
        "two output formats",
        req(&format!(
            r#","output_config":{{"format":{schema_format}}},"output_format":{schema_format}"#
        )),
        "output_format",
        "output_format: conflicts with output_config.format; send only one",
    ));
    rows.push(bad(
        "format type",
        req(r#","output_config":{"format":{"type":"text"}}"#),
        "output_config.format",
        "output_config.format: type must be json_schema",
    ));
    rows.push(bad(
        "format not an object",
        req(r#","output_format":"json""#),
        "output_format",
        "output_format: type must be json_schema",
    ));
    rows.push(unrecognized(
        "format key",
        req(r#","output_format":{"type":"json_schema","schema":{},"strict":true}"#),
        "output_format.strict",
    ));
    rows.push(bad(
        "format schema missing",
        req(r#","output_config":{"format":{"type":"json_schema"}}"#),
        "output_config.format.schema",
        "output_config.format.schema: must be a JSON Schema object",
    ));
    rows.push(bad(
        "format schema not an object",
        req(r#","output_format":{"type":"json_schema","schema":[]}"#),
        "output_format.schema",
        "output_format.schema: must be a JSON Schema object",
    ));
    rows.push(unsupported(
        "format schema guidance",
        req(r#","output_format":{"type":"json_schema","schema":{"x-guidance":null}}"#),
        "output_format.schema.x-guidance",
    ));
    // Stop sequences.
    rows.push(bad(
        "stop_sequences not a list",
        req(r#","stop_sequences":"a""#),
        "stop_sequences",
        "stop_sequences: must be a list of strings",
    ));
    rows.push(bad(
        "stop_sequences count",
        req(&format!(r#","stop_sequences":[{}]"#, vec![r#""a""#; 33].join(","))),
        "stop_sequences",
        "stop_sequences: at most 32 sequences",
    ));
    rows.push(bad(
        "stop sequence empty",
        req(r#","stop_sequences":["a",""]"#),
        "stop_sequences",
        "stop_sequences: each sequence must be a non-empty string",
    ));
    rows.push(bad(
        "stop sequence that is no string",
        req(r#","stop_sequences":[1]"#),
        "stop_sequences",
        "stop_sequences: each sequence must be a non-empty string",
    ));
    rows.push(bad(
        "stop_sequences with a JSON format",
        req(&format!(r#","stop_sequences":["a"],"output_format":{schema_format}"#)),
        "stop_sequences",
        "stop_sequences cannot be combined with a JSON output format: a stop inside the JSON would end it invalid.",
    ));
    // System and messages.
    rows.push(bad(
        "system that is no string or list",
        req(r#","system":1"#),
        "system",
        "system: must be a string or a list of content blocks",
    ));
    rows.push(bad(
        "system block not an object",
        req(r#","system":["a"]"#),
        "system[0]",
        "system[0]: must be an object",
    ));
    rows.push(bad(
        "system block type missing",
        req(r#","system":[{"text":"a"}]"#),
        "system[0].type",
        "system[0].type: Field required",
    ));
    rows.push(unsupported(
        "system block of another type",
        req(r#","system":[{"type":"image"}]"#),
        "system[0].type=image",
    ));
    rows.push(bad(
        "no messages",
        r#"{"model":"m","max_tokens":5}"#.to_owned(),
        "messages",
        "messages: at least one message is required",
    ));
    rows.push(bad(
        "empty messages",
        msgs("[]"),
        "messages",
        "messages: at least one message is required",
    ));
    rows.push(bad(
        "message not an object",
        msgs("[1]"),
        "messages[0]",
        "messages[0]: must be an object",
    ));
    rows.push(unrecognized(
        "message key",
        msgs(r#"[{"role":"user","content":"a","name":"n"}]"#),
        "messages[0].name",
    ));
    rows.push(bad(
        "message content missing",
        msgs(r#"[{"role":"user"}]"#),
        "messages[0].content",
        "messages[0].content: Field required",
    ));
    rows.push(bad(
        "message role missing",
        msgs(r#"[{"content":"a"}]"#),
        "messages[0].role",
        "messages[0].role: Field required",
    ));
    rows.push(bad(
        "message role",
        msgs(r#"[{"role":"system","content":"a"}]"#),
        "messages[0].role",
        "messages[0].role: must be 'user' or 'assistant'",
    ));
    rows.push(bad(
        "message role that is no string",
        msgs(r#"[{"role":1,"content":"a"}]"#),
        "messages[0].role",
        "messages[0].role: Field required",
    ));
    rows.push(bad(
        "content that is no string or list",
        user("1"),
        "messages[0].content",
        "messages[0].content: must be a string or a list of content blocks",
    ));
    rows.push(bad(
        "content block not an object",
        user(r#"["a"]"#),
        "messages[0].content[0]",
        "messages[0].content[0]: must be an object",
    ));
    rows.push(bad(
        "block type missing",
        user(r#"[{"text":"a"}]"#),
        "messages[0].content[0].type",
        "messages[0].content[0].type: Field required",
    ));
    rows.push(unrecognized(
        "text block key",
        user(r#"[{"type":"text","text":"a","x":1}]"#),
        "messages[0].content[0].x",
    ));
    rows.push(unsupported(
        "text block citations",
        user(r#"[{"type":"text","text":"a","citations":[{}]}]"#),
        "messages[0].content[0].citations",
    ));
    rows.push(bad(
        "text block text missing",
        user(r#"[{"type":"text"}]"#),
        "messages[0].content[0].text",
        "messages[0].content[0].text: Field required",
    ));
    rows.push(unsupported(
        "user block of another type",
        user(r#"[{"type":"image","source":{}}]"#),
        "messages[0].content[0].type=image",
    ));
    rows.push(unsupported(
        "user thinking block",
        user(r#"[{"type":"thinking","thinking":"a"}]"#),
        "messages[0].content[0].type=thinking",
    ));
    let result = r#"{"type":"tool_result","tool_use_id":"t1","content":"r"}"#;
    rows.push(bad(
        "tool_result after text",
        after(USE, &format!(r#"[{{"type":"text","text":"a"}},{result}]"#)),
        "messages[2].content[1]",
        "messages[2].content[1]: tool_result blocks must come before text in a user message",
    ));
    rows.push(unrecognized(
        "tool_result key",
        after(USE, r#"[{"type":"tool_result","tool_use_id":"t1","x":1}]"#),
        "messages[2].content[0].x",
    ));
    rows.push(bad(
        "tool_result is_error",
        after(USE, r#"[{"type":"tool_result","tool_use_id":"t1","is_error":"no"}]"#),
        "messages[2].content[0].is_error",
        "messages[2].content[0].is_error: must be a boolean",
    ));
    rows.push(unsupported(
        "tool_result content block",
        after(
            USE,
            r#"[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"image"}]}]"#,
        ),
        "messages[2].content[0].content[0].type=image",
    ));
    rows.push(bad(
        "tool_result content",
        after(USE, r#"[{"type":"tool_result","tool_use_id":"t1","content":1}]"#),
        "messages[2].content[0].content",
        "messages[2].content[0].content: must be a string or a list of content blocks",
    ));
    rows.push(bad(
        "tool_result without a tool_use",
        user(&format!("[{result}]")),
        "messages[0]",
        "messages[0]: tool_result blocks must answer tool_use blocks of the preceding assistant message",
    ));
    rows.push(bad(
        "tool_result id missing",
        after(USE, r#"[{"type":"tool_result","content":"r"}]"#),
        "messages[2].content[0].tool_use_id",
        "Tool results require 'tool_use_id'.",
    ));
    rows.push(bad(
        "tool_result id unknown",
        after(USE, r#"[{"type":"tool_result","tool_use_id":"zz"}]"#),
        "messages[2].content[0].tool_use_id",
        "'tool_use_id' zz does not match a tool call of the preceding assistant message.",
    ));
    rows.push(bad(
        "tool_result twice",
        after(USE, &format!("[{result},{result}]")),
        "messages[2].content[1].tool_use_id",
        "Tool call t1 already has a response.",
    ));
    let missing = "An assistant message with tool calls must be followed by a result for each call, matched by \
                   'tool_use_id'. Missing responses: t1.";
    rows.push(bad(
        "tool_use unanswered by text",
        after(USE, r#""next""#),
        "messages[1].content",
        missing,
    ));
    rows.push(bad(
        "tool_use unanswered by an assistant message",
        msgs(&format!(
            r#"[{{"role":"user","content":"q"}},{{"role":"assistant","content":{USE}}},{{"role":"assistant","content":"b"}},{{"role":"user","content":"c"}}]"#
        )),
        "messages[1].content",
        missing,
    ));
    rows.push(bad(
        "tool_use partly answered",
        after(
            r#"[{"type":"tool_use","id":"t1","name":"f","input":{}},{"type":"tool_use","id":"t2","name":"f","input":{}}]"#,
            r#"[{"type":"tool_result","tool_use_id":"t2"}]"#,
        ),
        "messages[1].content",
        missing,
    ));
    rows.push(unsupported(
        "final assistant message",
        msgs(r#"[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]"#),
        "messages[1] (a final assistant message to continue)",
    ));
    let assistant = |name: &'static str, content: &str, param: &str, message: &str| {
        bad(name, after(content, r#""x""#), param, message)
    };
    rows.push(unrecognized(
        "thinking block key",
        after(r#"[{"type":"thinking","thinking":"a","x":1}]"#, r#""x""#),
        "messages[1].content[0].x",
    ));
    rows.push(assistant(
        "thinking block text",
        r#"[{"type":"thinking"}]"#,
        "messages[1].content[0].thinking",
        "messages[1].content[0].thinking: Field required",
    ));
    rows.push(unrecognized(
        "tool_use key",
        after(
            r#"[{"type":"tool_use","id":"t","name":"f","input":{},"x":1}]"#,
            r#""x""#,
        ),
        "messages[1].content[0].x",
    ));
    rows.push(assistant(
        "tool_use id missing",
        r#"[{"type":"tool_use","name":"f","input":{}}]"#,
        "messages[1].content[0].id",
        "messages[1].content[0].id: Field required",
    ));
    rows.push(assistant(
        "tool_use id empty",
        r#"[{"type":"tool_use","id":"","name":"f","input":{}}]"#,
        "messages[1].content[0].id",
        "messages[1].content[0].id: Field required",
    ));
    rows.push(assistant(
        "tool_use id twice",
        r#"[{"type":"tool_use","id":"t","name":"f","input":{}},{"type":"tool_use","id":"t","name":"f","input":{}}]"#,
        "messages[1].content[1].id",
        "messages[1].content[1].id: tool_use IDs must be unique",
    ));
    rows.push(assistant(
        "tool_use name",
        r#"[{"type":"tool_use","id":"t","name":"a b","input":{}}]"#,
        "messages[1].content[0].name",
        "messages[1].content[0].name: must match ^[a-zA-Z0-9_-]{1,128}$",
    ));
    rows.push(assistant(
        "tool_use input missing",
        r#"[{"type":"tool_use","id":"t","name":"f"}]"#,
        "messages[1].content[0].input",
        "messages[1].content[0].input: must be an object",
    ));
    rows.push(assistant(
        "tool_use input name",
        r#"[{"type":"tool_use","id":"t","name":"f","input":{"a>":1}}]"#,
        "messages[1].content[0].input",
        "messages[1].content[0].input: argument names must be non-empty and free of '<', '>' and line breaks",
    ));
    rows.push(unsupported(
        "assistant block of another type",
        after(r#"[{"type":"redacted_thinking","data":"x"}]"#, r#""x""#),
        "messages[1].content[0].type=redacted_thinking",
    ));
    // Sampling.
    rows.push(bad(
        "temperature above 1",
        req(r#","temperature":1.5"#),
        "temperature",
        "temperature: must be between 0 and 1",
    ));
    rows.push(bad(
        "temperature below 0",
        req(r#","temperature":-0.5"#),
        "temperature",
        "temperature: must be between 0 and 1",
    ));
    rows.push(bad(
        "top_p zero",
        req(r#","top_p":0"#),
        "top_p",
        "top_p: must be greater than 0 and at most 1",
    ));
    rows.push(bad(
        "top_p above 1",
        req(r#","top_p":1.5"#),
        "top_p",
        "top_p: must be greater than 0 and at most 1",
    ));
    rows.push(bad(
        "top_k zero",
        req(r#","top_k":0"#),
        "top_k",
        "top_k: must be at least 1",
    ));
    rows
}

#[test]
fn every_rejection_is_reported_with_its_status_param_and_message() {
    let rows = rows();
    println!("{} rows", rows.len());
    let mut failures = Vec::new();
    for row in rows {
        let result = if row.count {
            count_body(&row.body)
        } else {
            map_body(&row.body)
        };
        match result {
            Err(error)
                if error.status == row.status
                    && error.code == row.code
                    && error.param == row.param
                    && error.message == row.message => {}
            Err(error) => failures.push(format!("{}: got {error:?}", row.name)),
            Ok(_) => failures.push(format!("{}: accepted", row.name)),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Of two faults the one checked first is the one reported.
#[test]
fn the_first_check_decides() {
    let ordered: [(&str, &str, &str); 14] = [
        // Unknown fields come before the model.
        (r#"{"bogus":1,"messages":[]}"#, "unrecognized_argument", "bogus"),
        // The model before max_tokens.
        (r#"{"model":"x","messages":[]}"#, "model_not_found", "model"),
        // max_tokens before tools and messages.
        (r#"{"model":"m","tools":1}"#, "invalid", "max_tokens"),
        // Tools before thinking.
        (
            r#"{"model":"m","max_tokens":5,"tools":1,"thinking":1,"messages":[]}"#,
            "invalid",
            "tools",
        ),
        // Thinking before the output format.
        (
            r#"{"model":"m","max_tokens":5,"thinking":1,"output_config":1,"messages":[]}"#,
            "invalid",
            "thinking",
        ),
        // The effort before the thinking object.
        (
            r#"{"model":"m","max_tokens":5,"thinking":1,"output_config":{"effort":"x"},"messages":[]}"#,
            "invalid",
            "output_config.effort",
        ),
        // The output format before stop sequences.
        (
            r#"{"model":"m","max_tokens":5,"output_config":1,"stop_sequences":1,"messages":[]}"#,
            "invalid",
            "output_config",
        ),
        // Stop sequences before messages.
        (
            r#"{"model":"m","max_tokens":5,"stop_sequences":1,"messages":[]}"#,
            "invalid",
            "stop_sequences",
        ),
        // The system prompt before the empty message list.
        (
            r#"{"model":"m","max_tokens":5,"system":1,"messages":[]}"#,
            "invalid",
            "system",
        ),
        // Messages before sampling.
        (
            r#"{"model":"m","max_tokens":5,"temperature":9,"messages":[]}"#,
            "invalid",
            "messages",
        ),
        // The unknown key before the missing content.
        (
            r#"{"model":"m","max_tokens":5,"messages":[{"x":1}]}"#,
            "unrecognized_argument",
            "messages[0].x",
        ),
        // The content shape before the role.
        (
            r#"{"model":"m","max_tokens":5,"messages":[{"role":"bad","content":1}]}"#,
            "invalid",
            "messages[0].content",
        ),
        // A final assistant message after the results of the call before it.
        (
            r#"{"model":"m","max_tokens":5,"messages":[{"role":"user","content":"a"},{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"f","input":{}}]},{"role":"assistant","content":"b"}]}"#,
            "invalid",
            "messages[1].content",
        ),
        // Sampling checks temperature, then top_p, then top_k.
        (
            r#"{"model":"m","max_tokens":5,"messages":[{"role":"user","content":"a"}],"top_k":0,"top_p":0,"temperature":2}"#,
            "invalid",
            "temperature",
        ),
    ];
    for (body, kind, param) in ordered {
        let error = map_body(body).expect_err(body);
        let expected = match kind {
            "unrecognized_argument" => Some("unrecognized_argument"),
            "model_not_found" => Some("model_not_found"),
            _ => None,
        };
        assert_eq!(error.code, expected, "{body}");
        assert_eq!(error.param.as_deref(), Some(param), "{body}");
    }
}

fn texts(request: &GenerationRequest) -> Vec<(Role, String)> {
    request
        .template
        .messages
        .iter()
        .map(|message| {
            let text = match message {
                Message::System(parts) | Message::User(parts) | Message::Tool(parts) => parts,
                Message::Assistant(assistant) => &assistant.content,
            }
            .iter()
            .map(|ContentPart::Text(text)| text.as_str())
            .collect::<String>();
            (message.role(), text)
        })
        .collect()
}

#[test]
fn a_plain_request_takes_the_defaults() {
    let request = accepted(&req(""));
    assert_eq!(request.model, "m");
    assert_eq!(request.template.messages, vec![Message::user("hi")]);
    assert!(request.template.tools.is_empty());
    assert_eq!(request.thinking, TemplateOptions::THINKING_DISABLED);
    assert_eq!(request.reasoning_budget, None);
    assert_eq!(request.tools, ToolCalling::disabled());
    assert_eq!(request.format, ResponseFormat::Text);
    assert!(request.stops.is_empty());
    assert_eq!(
        request.sampling,
        Sampling {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed: None,
            greedy: false
        }
    );
    assert_eq!(request.max_tokens, Some(64));
    assert_eq!(request.max_tokens_param, "max_tokens");
    assert!(!request.stream && !request.return_token_ids);
}

#[test]
fn sampling_fields_override_the_defaults() {
    let request = accepted(&req(r#","temperature":0.25,"top_p":0.5,"top_k":7,"stream":true"#));
    assert_eq!(
        request.sampling,
        Sampling {
            temperature: 0.25,
            top_k: 7,
            top_p: 0.5,
            seed: None,
            greedy: false
        }
    );
    assert!(request.stream);
    let greedy = accepted(&req(r#","temperature":0,"top_p":0.5"#));
    assert!(greedy.sampling.greedy);
    assert_eq!(greedy.sampling.top_p, 1.0);
    assert!(accepted(&req(r#","temperature":1,"top_p":1"#)).sampling.top_p == 1.0);
}

#[test]
fn metadata_and_null_fields_are_accepted() {
    let request = accepted(&req(
        r#","metadata":{"user_id":"u"},"service_tier":"auto","system":null,"tools":null,"tool_choice":null,"thinking":null,"output_config":null,"output_format":null,"stop_sequences":null,"temperature":null"#,
    ));
    assert_eq!(request.template.messages, vec![Message::user("hi")]);
    assert_eq!(request.tools, ToolCalling::disabled());
}

#[test]
fn a_system_prompt_is_a_string_or_text_blocks() {
    let plain = accepted(&req(r#","system":"Be brief.""#));
    assert_eq!(
        texts(&plain),
        vec![(Role::System, "Be brief.".to_owned()), (Role::User, "hi".to_owned())]
    );
    let blocks = accepted(&req(
        r#","system":[{"type":"text","text":"Be ","cache_control":{"type":"ephemeral"}},{"type":"text","text":"brief.","citations":[]}]"#,
    ));
    assert_eq!(texts(&blocks), texts(&plain));
    let empty = accepted(&req(r#","system":[]"#));
    assert_eq!(texts(&empty)[0], (Role::System, String::new()));
}

#[test]
fn content_blocks_concatenate_into_one_user_turn() {
    let request = accepted(&user(
        r#"[{"type":"text","text":"a","cache_control":{"type":"ephemeral"}},{"type":"text","text":"b"}]"#,
    ));
    assert_eq!(request.template.messages, vec![Message::user("ab")]);
    assert_eq!(accepted(&user("[]")).template.messages, vec![Message::user("")]);
}

#[test]
fn tool_use_and_tool_results_continue_the_conversation() {
    let request = accepted(
        r#"{"model":"m","max_tokens":64,"tools":[{"name":"f","input_schema":{}}],"messages":[
            {"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"thinking","thinking":"hm","signature":"s"},{"type":"text","text":"ok"},
                {"type":"tool_use","id":"t1","name":"f","input":{"a":1}},
                {"type":"tool_use","id":"t2","name":"f","input":{}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"r"},{"type":"text","text":"2"}]},
                {"type":"tool_result","tool_use_id":"t1","is_error":true},
                {"type":"text","text":"and then"}]}]}"#,
    );
    let messages = &request.template.messages;
    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0], Message::user("q"));
    let Message::Assistant(assistant) = &messages[1] else {
        panic!("an assistant turn");
    };
    assert_eq!(assistant.content, vec![ContentPart::Text("ok".to_owned())]);
    assert_eq!(assistant.reasoning, "hm");
    assert_eq!(assistant.tool_calls.len(), 2);
    assert_eq!(assistant.tool_calls[0].name, "f");
    assert_eq!(
        assistant.tool_calls[0].arguments,
        vec![("a".to_owned(), JsonValue::Integer(JsonInteger::from(1i128)))]
    );
    // The results follow in the order of the calls.
    assert_eq!(messages[2], Message::tool(""));
    assert_eq!(messages[3], Message::tool("r2"));
    assert_eq!(messages[4], Message::user("and then"));
}

#[test]
fn a_user_turn_without_text_after_results_is_not_added() {
    let request = accepted(&after(
        USE,
        r#"[{"type":"tool_result","tool_use_id":"t1","content":"r"}]"#,
    ));
    assert_eq!(
        texts(&request),
        vec![
            (Role::User, "q".to_owned()),
            (Role::Assistant, String::new()),
            (Role::Tool, "r".to_owned())
        ]
    );
    // A call followed by a user message that answers it later is refused, one answered in a single message is not.
    let two = accepted(&msgs(&format!(
        r#"[{{"role":"user","content":"q"}},{{"role":"assistant","content":{USE}}},{{"role":"user","content":[{{"type":"tool_result","tool_use_id":"t1"}}]}},{{"role":"assistant","content":"a"}},{{"role":"user","content":"z"}}]"#
    )));
    assert_eq!(two.template.messages.len(), 5);
}

#[test]
fn a_final_user_message_with_only_results_is_accepted() {
    let request = accepted(&after(
        USE,
        r#"[{"type":"tool_result","tool_use_id":"t1","content":""}]"#,
    ));
    assert_eq!(request.template.messages.last(), Some(&Message::tool("")));
}

#[test]
fn tools_become_prompt_tools_and_a_choice() {
    let tools = r#","tools":[{"name":"f","description":"d","input_schema":{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]},"cache_control":{"type":"ephemeral"},"strict":true},{"type":"custom","name":"g","input_schema":{}}]"#;
    let auto = accepted(&req(tools));
    assert_eq!(auto.tools.choice, ToolChoice::Auto);
    assert!(auto.tools.parallel);
    assert_eq!(auto.template.tools.len(), 2);
    assert_eq!(auto.template.tools[0].name, "f");
    assert_eq!(auto.template.tools[0].description.as_deref(), Some("d"));
    assert!(auto.tools.tools[0].strict);
    assert!(!auto.tools.tools[1].strict);
    assert_eq!(auto.tools.tools[0].parameters_param, "tools[0].input_schema");
    assert_eq!(auto.tools.tools[1].parameters_param, "tools[1].input_schema");

    let choice = |choice: &str| accepted(&req(&format!(r#"{tools},"tool_choice":{choice}"#)));
    assert_eq!(choice(r#"{"type":"auto"}"#).tools.choice, ToolChoice::Auto);
    assert_eq!(choice(r#"{"type":"any"}"#).tools.choice, ToolChoice::Required);
    let named = choice(r#"{"type":"tool","name":"g","disable_parallel_tool_use":true}"#);
    assert_eq!(named.tools.choice, ToolChoice::Function("g".to_owned()));
    assert!(!named.tools.parallel);
    let none = choice(r#"{"type":"none"}"#);
    assert_eq!(none.tools.choice, ToolChoice::None);
    assert!(
        none.template.tools.is_empty(),
        "a none choice keeps the tools out of the prompt"
    );
    assert_eq!(none.tools.tools.len(), 2);
    assert!(
        choice(r#"{"type":"auto","disable_parallel_tool_use":false}"#)
            .tools
            .parallel
    );
}

#[test]
fn a_choice_without_tools_is_none_only() {
    let none = accepted(&req(
        r#","tool_choice":{"type":"none","disable_parallel_tool_use":true}"#,
    ));
    assert!(none.tools.tools.is_empty());
    assert_eq!(none.tools.choice, ToolChoice::None);
    assert!(!none.tools.parallel);
    let empty = accepted(&req(r#","tools":[]"#));
    assert_eq!(empty.tools, ToolCalling::disabled());
}

#[test]
fn thinking_options_resolve_to_the_templates_variables() {
    let thinking = |rest: &str| accepted(&req(rest));
    let enabled = thinking(r#","thinking":{"type":"enabled","budget_tokens":10}"#);
    assert_eq!(
        enabled.thinking,
        TemplateOptions {
            effort: Some(Effort::XHigh),
            preserve_thinking: true
        }
    );
    assert_eq!(enabled.reasoning_budget, Some(10));
    assert!(enabled.reasoning());
    let adaptive = thinking(r#","thinking":{"type":"adaptive"},"output_config":{"effort":"low"}"#);
    assert_eq!(adaptive.thinking.effort, Some(Effort::Low));
    assert_eq!(adaptive.reasoning_budget, None);
    for (name, effort) in [
        ("medium", Effort::Medium),
        ("high", Effort::XHigh),
        ("max", Effort::XHigh),
    ] {
        let request = thinking(&format!(
            r#","thinking":{{"type":"enabled","budget_tokens":3}},"output_config":{{"effort":"{name}"}}"#
        ));
        assert_eq!(request.thinking.effort, Some(effort), "{name}");
    }
    let disabled = thinking(r#","thinking":{"type":"disabled"}"#);
    assert_eq!(disabled.thinking, TemplateOptions::THINKING_DISABLED);
    assert!(!disabled.reasoning());
    // The effort is a cap's partner: a null effort names none.
    assert_eq!(
        thinking(r#","thinking":{"type":"adaptive"},"output_config":{"effort":null}"#)
            .thinking
            .effort,
        Some(Effort::XHigh)
    );
}

#[test]
fn a_thinking_budget_is_checked_against_max_tokens_only_when_generating() {
    let body =
        r#"{"model":"m","thinking":{"type":"enabled","budget_tokens":100},"messages":[{"role":"user","content":"a"}]}"#;
    let counted = count_body(body).expect("a count states no max_tokens");
    assert_eq!(counted.max_tokens, None);
    assert_eq!(counted.reasoning_budget, Some(100));
    assert!(!counted.stream);
}

#[test]
fn stop_sequences_and_json_formats() {
    let request = accepted(&req(r#","stop_sequences":["END","STOP"]"#));
    assert_eq!(request.stops, vec!["END".to_owned(), "STOP".to_owned()]);
    let many = accepted(&req(&format!(
        r#","stop_sequences":[{}]"#,
        vec![r#""a""#; 32].join(",")
    )));
    assert_eq!(many.stops.len(), 32);

    let schema = r#"{"type":"object","properties":{"answer":{"type":"integer"}}}"#;
    for (key, path) in [
        ("output_config", "output_config.format.schema"),
        ("output_format", "output_format.schema"),
    ] {
        let format = format!(r#"{{"type":"json_schema","schema":{schema}}}"#);
        let body = if key == "output_config" {
            req(&format!(r#","output_config":{{"format":{format}}}"#))
        } else {
            req(&format!(r#","output_format":{format}"#))
        };
        let request = accepted(&body);
        let ResponseFormat::JsonSchema {
            schema: parsed,
            schema_param,
            ..
        } = request.format
        else {
            panic!("a JSON schema format");
        };
        assert_eq!(schema_param, path);
        assert_eq!(parsed, JsonValue::parse(schema).unwrap());
    }
    let both = accepted(&req(
        r#","thinking":{"type":"adaptive"},"output_config":{"effort":"medium","format":{"type":"json_schema","schema":{}}}"#,
    ));
    assert!(both.format.is_json());
    assert_eq!(both.thinking.effort, Some(Effort::Medium));
}

#[test]
fn a_count_tokens_request_maps_like_a_message_without_a_budget() {
    let request = count_body(
        r#"{"model":"m","system":"s","messages":[{"role":"user","content":"hi"}],"tools":[{"name":"f","input_schema":{}}]}"#,
    )
    .expect("accepted");
    assert_eq!(request.max_tokens, None);
    assert_eq!(request.max_tokens_param, "max_tokens");
    assert_eq!(texts(&request).len(), 2);
    assert_eq!(request.template.tools.len(), 1);
    // The rest of the checks are the same.
    let error = count_body(r#"{"model":"m","messages":[]}"#).expect_err("no messages");
    assert_eq!(error.param.as_deref(), Some("messages"));
}

#[test]
fn values_print_as_error_parameters() {
    let value = |text: &str| JsonValue::parse(text).unwrap();
    assert_eq!(value("1.0").param_text(), "1.0");
    assert_eq!(value("5").param_text(), "5");
    assert_eq!(value("0.5").param_text(), "0.5");
    assert_eq!(value("1e10").param_text(), "1.0E10");
    assert_eq!(value("1.5e-5").param_text(), "1.5E-5");
    assert_eq!(value("true").param_text(), "true");
    assert_eq!(value("null").param_text(), "null");
    assert_eq!(&value(r#"{"a":[1,"b"],"c":{}}"#).param_text(), "{a=[1, b], c={}}");
}

#[test]
fn numbers_in_tool_use_input_reach_the_template_exactly() {
    let request = accepted(&msgs(
        r#"[{"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"f","input":{"n":123456789012345678901234567890,"e":1e400}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"r"}]}]"#,
    ));
    let Message::Assistant(assistant) = &request.template.messages[1] else {
        panic!("an assistant turn");
    };
    let written: Vec<String> = assistant.tool_calls[0]
        .arguments
        .iter()
        .map(|(_, value)| value.to_python_json())
        .collect();
    assert_eq!(written, ["123456789012345678901234567890", "Infinity"]);
}
