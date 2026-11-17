use super::*;
use crate::json::JsonInteger;
use crate::template::{Effort, TemplateInput, TemplateOptions};

fn defaults() -> Defaults {
    Defaults {
        model: "m".to_owned(),
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        max_tokens: 4096,
    }
}

fn map(body: &str) -> Result<MappedResponses, ApiError> {
    let request = ResponsesRequest::parse(body).expect("the body parses");
    map_responses(request, &defaults())
}

fn accepted(body: &str) -> GenerationRequest {
    accepted_mapped(body).generation
}

fn accepted_mapped(body: &str) -> MappedResponses {
    map(body).unwrap_or_else(|error| panic!("{body} was rejected: {}", error.message))
}

/// A request of the string input "hi" with `rest` appended to its members (`,"key":value`).
fn req(rest: &str) -> String {
    format!(r#"{{"model":"m","input":"hi"{rest}}}"#)
}

/// A request with the given `items` array as its input.
fn items(items: &str) -> String {
    format!(r#"{{"model":"m","input":{items}}}"#)
}

/// A request with one input item made of `members`.
fn item(members: &str) -> String {
    items(&format!("[{{{members}}}]"))
}

struct Row {
    name: &'static str,
    body: String,
    status: u16,
    code: Option<&'static str>,
    param: Option<String>,
    message: String,
}

fn bad(name: &'static str, body: String, param: Option<&str>, message: &str) -> Row {
    Row {
        name,
        body,
        status: 400,
        code: None,
        param: param.map(str::to_owned),
        message: message.to_owned(),
    }
}

fn unsupported(name: &'static str, body: String, param: &str) -> Row {
    Row {
        name,
        body,
        status: 400,
        code: Some("unsupported_parameter"),
        param: Some(param.to_owned()),
        message: format!("Unsupported parameter: '{param}' is not supported by this server."),
    }
}

fn unrecognized(name: &'static str, body: String, param: &str) -> Row {
    Row {
        name,
        body,
        status: 400,
        code: Some("unrecognized_argument"),
        param: Some(param.to_owned()),
        message: format!("Unrecognized request argument supplied: {param}"),
    }
}

const TOOL: &str = r#"{"type":"function","name":"a"}"#;
const NAME_MESSAGE: &str = "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.";

/// Two calls, `c1` and `c2`, after a user item, then `after` items (`,{...}`).
fn two_calls(after: &str) -> String {
    items(&format!(
        r#"[{{"role":"user","content":"x"}},{{"type":"function_call","call_id":"c1","name":"f","arguments":"{{}}"}},{{"type":"function_call","call_id":"c2","name":"f","arguments":"{{}}"}}{after}]"#
    ))
}

fn output(id: &str) -> String {
    format!(r#",{{"type":"function_call_output","call_id":"{id}","output":"r"}}"#)
}

/// A request with one user item whose content is `content`.
fn with_content(content: &str) -> String {
    item(&format!(r#""role":"user","content":{content}"#))
}

fn rows() -> Vec<Row> {
    let tool_choice = |choice: &str| req(&format!(r#","tools":[{TOOL}],"tool_choice":{choice}"#));
    let tool = |members: &str| req(&format!(r#","tools":[{members}]"#));
    let function = |members: &str| tool(&format!(r#"{{"type":"function",{members}}}"#));
    let parameters = |schema: &str| function(&format!(r#""name":"a","parameters":{schema}"#));
    let reasoning = |options: &str| req(&format!(r#","reasoning":{options}"#));
    let text = |text: &str| req(&format!(r#","text":{text}"#));
    let format = |format: &str| text(&format!(r#"{{"format":{format}}}"#));
    let schema = |members: &str| format(&format!(r#"{{"type":"json_schema",{members}}}"#));
    let content = |content: &str| item(&format!(r#""role":"user","content":{content}"#));
    let part = |part: &str| content(&format!("[{part}]"));
    let call = |members: &str| item(&format!(r#""type":"function_call",{members}"#));
    let reasoning_item = |members: &str| item(&format!(r#""type":"reasoning",{members}"#));
    let no_model = r#"{"input":"hi"}"#.to_owned();
    let model_message = "You must provide a model parameter.";
    let choice_without_tools = "'tool_choice' is only allowed when 'tools' are specified.";
    let choice_message = "'tool_choice' must be 'none', 'auto', 'required', or a function choice.";
    let results_message = |ids: &str| {
        format!(
            "An assistant message with tool calls must be followed by a result for each call, matched by 'call_id'. \
             Missing responses: {ids}."
        )
    };
    let effort_message = |param: &str| format!("'{param}' must be one of none, low, medium, high (or xhigh).");
    vec![
        // Other fields.
        unsupported("truncation auto", req(r#","truncation":"auto""#), "truncation"),
        unsupported("background true", req(r#","background":true"#), "background"),
        unsupported(
            "previous_response_id",
            req(r#","previous_response_id":"r""#),
            "previous_response_id",
        ),
        unsupported("conversation", req(r#","conversation":"c""#), "conversation"),
        unsupported("prompt", req(r#","prompt":{}"#), "prompt"),
        unsupported("max_tool_calls", req(r#","max_tool_calls":1"#), "max_tool_calls"),
        unsupported("top_logprobs", req(r#","top_logprobs":0"#), "top_logprobs"),
        unrecognized("unknown field", req(r#","bogus":1"#), "bogus"),
        unrecognized("unknown null field", req(r#","bogus":null"#), "bogus"),
        unrecognized("the first unknown field", req(r#","a":1,"b":2"#), "a"),
        // The model.
        bad("no model", no_model, Some("model"), model_message),
        bad(
            "blank model",
            r#"{"model":" \t","input":"hi"}"#.to_owned(),
            Some("model"),
            model_message,
        ),
        Row {
            name: "unknown model",
            body: r#"{"model":"zzz","input":"hi"}"#.to_owned(),
            status: 404,
            code: Some("model_not_found"),
            param: Some("model".to_owned()),
            message: "The model 'zzz' does not exist or you do not have access to it.".to_owned(),
        },
        // include.
        bad(
            "include not a list",
            req(r#","include":"x""#),
            Some("include"),
            "'include' must be a list.",
        ),
        unsupported(
            "include entry",
            req(r#","include":["reasoning.encrypted_content","message.output_text.logprobs"]"#),
            "include=message.output_text.logprobs",
        ),
        unsupported("include number entry", req(r#","include":[1.5]"#), "include=1.5"),
        unsupported(
            "include list entry",
            req(r#","include":[["a",2,null]]"#),
            "include=[a, 2, null]",
        ),
        unsupported(
            "include object entry",
            req(r#","include":[{"k":"v"}]"#),
            "include={k=v}",
        ),
        // tools.
        bad(
            "tools not a list",
            req(r#","tools":5"#),
            Some("tools"),
            "'tools' must be a list.",
        ),
        bad(
            "tool not an object",
            req(r#","tools":[5]"#),
            Some("tools[0]"),
            "Each tool must be an object.",
        ),
        unsupported(
            "tool type",
            tool(r#"{"type":"web_search"}"#),
            "tools[0].type=web_search",
        ),
        unsupported("tool without type", tool(r#"{"name":"a"}"#), "tools[0].type=null"),
        unrecognized("tool key", function(r#""name":"a","x":1"#), "tools[0].x"),
        bad(
            "tool name missing",
            function(r#""strict":true"#),
            Some("tools[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "tool name invalid",
            function(r#""name":"a b""#),
            Some("tools[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "tool name not a string",
            function(r#""name":5"#),
            Some("tools[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "tool name too long",
            function(&format!(r#""name":"{}""#, "a".repeat(65))),
            Some("tools[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "duplicate tool",
            req(r#","tools":[{"type":"function","name":"a"},{"type":"function","name":"a"}]"#),
            Some("tools[1].name"),
            "Tool names must be unique.",
        ),
        bad(
            "tool description",
            function(r#""name":"a","description":1"#),
            Some("tools[0].description"),
            "'description' must be a string.",
        ),
        bad(
            "tool strict",
            function(r#""name":"a","strict":"yes""#),
            Some("tools[0].strict"),
            "'strict' must be a boolean.",
        ),
        bad(
            "parameters not an object",
            parameters("[]"),
            Some("tools[0].parameters"),
            "Function 'parameters' must be a JSON Schema object.",
        ),
        bad(
            "parameters type",
            parameters(r#"{"type":"string"}"#),
            Some("tools[0].parameters.type"),
            "Function 'parameters' must describe an object; tool calls pass named arguments.",
        ),
        bad(
            "properties not an object",
            parameters(r#"{"properties":[]}"#),
            Some("tools[0].parameters.properties"),
            "'properties' must be an object.",
        ),
        bad(
            "property name",
            parameters(r#"{"properties":{"a<b":{}}}"#),
            Some("tools[0].parameters.properties"),
            "Parameter names must be non-empty and free of '<', '>' and line breaks.",
        ),
        bad(
            "required not strings",
            parameters(r#"{"required":[1]}"#),
            Some("tools[0].parameters.required"),
            "'required' must be an array of strings.",
        ),
        bad(
            "required name duplicate",
            parameters(r#"{"required":["a","a"]}"#),
            Some("tools[0].parameters.required"),
            "Required parameter names must be unique, non-empty and free of '<', '>' and line breaks.",
        ),
        bad(
            "required undeclared when closed",
            parameters(r#"{"required":["a"],"additionalProperties":false}"#),
            Some("tools[0].parameters.required"),
            "Required parameters must be declared when additional properties are forbidden.",
        ),
        // tool_choice.
        bad(
            "choice without tools",
            req(r#","tool_choice":"required""#),
            Some("tool_choice"),
            choice_without_tools,
        ),
        bad(
            "named choice without tools",
            req(r#","tools":[],"tool_choice":{"type":"function","name":"a"}"#),
            Some("tool_choice"),
            choice_without_tools,
        ),
        bad(
            "unknown choice string",
            tool_choice(r#""sometimes""#),
            Some("tool_choice"),
            choice_message,
        ),
        bad("choice number", tool_choice("5"), Some("tool_choice"), choice_message),
        bad("choice array", tool_choice("[]"), Some("tool_choice"), choice_message),
        unsupported(
            "choice type",
            tool_choice(r#"{"type":"allowed_tools"}"#),
            "tool_choice.type=allowed_tools",
        ),
        unsupported(
            "choice without type",
            tool_choice(r#"{"name":"a"}"#),
            "tool_choice.type=null",
        ),
        unrecognized(
            "choice key",
            tool_choice(r#"{"type":"function","name":"a","x":1}"#),
            "tool_choice.x",
        ),
        bad(
            "choice name missing",
            tool_choice(r#"{"type":"function"}"#),
            Some("tool_choice.name"),
            "'tool_choice.name' must name one of the tools.",
        ),
        bad(
            "choice name unknown",
            tool_choice(r#"{"type":"function","name":"b"}"#),
            Some("tool_choice.name"),
            "'tool_choice.name' must name one of the tools.",
        ),
        // reasoning.
        bad(
            "reasoning not an object",
            reasoning("5"),
            Some("reasoning"),
            "'reasoning' must be an object.",
        ),
        unrecognized("reasoning key", reasoning(r#"{"x":1}"#), "reasoning.x"),
        bad(
            "reasoning summary",
            reasoning(r#"{"summary":"long"}"#),
            Some("reasoning.summary"),
            "'reasoning.summary' must be auto, concise or detailed.",
        ),
        bad(
            "reasoning generate_summary",
            reasoning(r#"{"generate_summary":5}"#),
            Some("reasoning.generate_summary"),
            "'reasoning.generate_summary' must be auto, concise or detailed.",
        ),
        bad(
            "reasoning effort minimal",
            reasoning(r#"{"effort":"minimal"}"#),
            Some("reasoning_effort"),
            "'reasoning_effort' 'minimal' is not supported: this model's reasoning levels are none, low, medium and \
             high.",
        ),
        bad(
            "reasoning effort unknown",
            reasoning(r#"{"effort":"max"}"#),
            Some("reasoning_effort"),
            &effort_message("reasoning_effort"),
        ),
        bad(
            "reasoning effort not a string",
            reasoning(r#"{"effort":1}"#),
            Some("reasoning_effort"),
            &effort_message("reasoning_effort"),
        ),
        // text.
        bad(
            "text not an object",
            text("[]"),
            Some("text"),
            "'text' must be an object.",
        ),
        unrecognized("text key", text(r#"{"x":1}"#), "text.x"),
        unsupported("text verbosity", text(r#"{"verbosity":"low"}"#), "text.verbosity"),
        bad(
            "format not an object",
            format("5"),
            Some("text.format.type"),
            "'text.format.type' is required.",
        ),
        bad(
            "format without type",
            format(r#"{"name":"n"}"#),
            Some("text.format.type"),
            "'text.format.type' is required.",
        ),
        bad(
            "format type not a string",
            format(r#"{"type":1}"#),
            Some("text.format.type"),
            "'text.format.type' is required.",
        ),
        unrecognized("text format key", format(r#"{"type":"text","x":1}"#), "text.format.x"),
        unrecognized(
            "json_object key",
            format(r#"{"type":"json_object","x":1}"#),
            "text.format.x",
        ),
        unrecognized("json_schema key", schema(r#""name":"n","x":1"#), "text.format.x"),
        unsupported(
            "format type",
            format(r#"{"type":"grammar"}"#),
            "text.format.type=grammar",
        ),
        bad(
            "schema name missing",
            schema(r#""schema":{}"#),
            Some("response_format.json_schema.name"),
            "'response_format.json_schema.name' must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
        ),
        bad(
            "schema name invalid",
            schema(r#""name":"a b""#),
            Some("response_format.json_schema.name"),
            "'response_format.json_schema.name' must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
        ),
        bad(
            "schema description",
            schema(r#""name":"n","description":1"#),
            Some("response_format.json_schema.description"),
            "'response_format.json_schema.description' must be a string.",
        ),
        bad(
            "schema strict",
            schema(r#""name":"n","strict":"yes""#),
            Some("response_format.json_schema.strict"),
            "'response_format.json_schema.strict' must be a boolean.",
        ),
        bad(
            "schema not an object",
            schema(r#""name":"n","schema":[]"#),
            Some("response_format.json_schema.schema"),
            "'response_format.json_schema.schema' must be a JSON Schema object.",
        ),
        unsupported(
            "schema x-guidance",
            schema(r#""name":"n","schema":{"x-guidance":null}"#),
            "response_format.json_schema.schema.x-guidance",
        ),
        // input.
        bad(
            "no input",
            r#"{"model":"m"}"#.to_owned(),
            Some("input"),
            "'input' is required.",
        ),
        bad(
            "null input",
            r#"{"model":"m","input":null}"#.to_owned(),
            Some("input"),
            "'input' is required.",
        ),
        bad(
            "input not a string or list",
            r#"{"model":"m","input":5}"#.to_owned(),
            Some("input"),
            "'input' must be a string or a list of items.",
        ),
        bad("empty input", items("[]"), Some("input"), "'input' must not be empty."),
        bad(
            "item not an object",
            items(r#"[{"role":"user","content":"x"},"y"]"#),
            Some("input[1]"),
            "Each input item must be an object.",
        ),
        bad(
            "item without type",
            item(r#""content":"x""#),
            Some("input[0].type"),
            "Input item 'type' is required.",
        ),
        bad(
            "item type not a string",
            item(r#""type":5"#),
            Some("input[0].type"),
            "Input item 'type' is required.",
        ),
        unrecognized(
            "message key",
            item(r#""role":"user","content":"x","x":1"#),
            "input[0].x",
        ),
        bad(
            "message without role",
            item(r#""type":"message","content":"x""#),
            Some("input[0].role"),
            "Message 'role' is required.",
        ),
        bad(
            "message role not a string",
            item(r#""type":"message","role":5"#),
            Some("input[0].role"),
            "Message 'role' is required.",
        ),
        bad(
            "message role invalid",
            item(r#""role":"tool","content":"x""#),
            Some("input[0].role"),
            "Invalid message role 'tool'.",
        ),
        bad(
            "system after the conversation",
            items(r#"[{"role":"user","content":"x"},{"role":"system","content":"s"}]"#),
            Some("input[1]"),
            "System and developer messages must come before the conversation.",
        ),
        bad(
            "developer after an assistant message",
            items(r#"[{"role":"assistant","content":"a"},{"role":"developer","content":"s"}]"#),
            Some("input[1]"),
            "System and developer messages must come before the conversation.",
        ),
        unrecognized("reasoning item key", reasoning_item(r#""x":1"#), "input[0].x"),
        unrecognized("function_call key", call(r#""x":1"#), "input[0].x"),
        bad(
            "call_id missing",
            call(r#""name":"f","arguments":"{}""#),
            Some("input[0].call_id"),
            "'call_id' is required.",
        ),
        bad(
            "call_id empty",
            call(r#""call_id":"","name":"f","arguments":"{}""#),
            Some("input[0].call_id"),
            "'call_id' is required.",
        ),
        bad(
            "call_id not a string",
            call(r#""call_id":5,"name":"f","arguments":"{}""#),
            Some("input[0].call_id"),
            "'call_id' is required.",
        ),
        bad(
            "call_id repeated",
            items(
                r#"[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"}]"#,
            ),
            Some("input[1].call_id"),
            "Call IDs must be unique within a turn.",
        ),
        bad(
            "call name missing",
            call(r#""call_id":"c","arguments":"{}""#),
            Some("input[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "call name invalid",
            call(r#""call_id":"c","name":"a b","arguments":"{}""#),
            Some("input[0].name"),
            NAME_MESSAGE,
        ),
        bad(
            "arguments missing",
            call(r#""call_id":"c","name":"f""#),
            Some("input[0].arguments"),
            "'arguments' must be a JSON string.",
        ),
        bad(
            "arguments not a string",
            call(r#""call_id":"c","name":"f","arguments":{}"#),
            Some("input[0].arguments"),
            "'arguments' must be a JSON string.",
        ),
        bad(
            "arguments invalid",
            call(r##""call_id":"c","name":"f","arguments":"{""##),
            Some("input[0].arguments"),
            "'arguments' must be valid JSON.",
        ),
        bad(
            "arguments with trailing text",
            call(r##""call_id":"c","name":"f","arguments":"{\"a\":1} trailing""##),
            Some("input[0].arguments"),
            "'arguments' must be valid JSON.",
        ),
        bad(
            "arguments of two objects",
            call(r#""call_id":"c","name":"f","arguments":"{} {}""#),
            Some("input[0].arguments"),
            "'arguments' must be valid JSON.",
        ),
        bad(
            "arguments blank",
            call(r#""call_id":"c","name":"f","arguments":" ""#),
            Some("input[0].arguments"),
            "'arguments' must be valid JSON.",
        ),
        bad(
            "arguments not an object",
            call(r#""call_id":"c","name":"f","arguments":"[1]""#),
            Some("input[0].arguments"),
            "'arguments' must encode a JSON object.",
        ),
        bad(
            "arguments null",
            call(r#""call_id":"c","name":"f","arguments":"null""#),
            Some("input[0].arguments"),
            "'arguments' must encode a JSON object.",
        ),
        unrecognized(
            "output key",
            item(r#""type":"function_call_output","x":1"#),
            "input[0].x",
        ),
        bad(
            "output without a call",
            item(r#""type":"function_call_output","call_id":"c","output":"r""#),
            Some("input[0]"),
            "A function_call_output must answer a function_call of the preceding turn.",
        ),
        bad(
            "output after a user item",
            items(
                r#"[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"role":"user","content":"x"},{"type":"function_call_output","call_id":"c","output":"r"}]"#,
            ),
            Some("input[0]"),
            &results_message("c"),
        ),
        bad(
            "output call_id missing",
            two_calls(r#",{"type":"function_call_output","output":"r"}"#),
            Some("input[3].call_id"),
            "Tool results require 'call_id'.",
        ),
        bad(
            "output call_id empty",
            two_calls(r#",{"type":"function_call_output","call_id":"","output":"r"}"#),
            Some("input[3].call_id"),
            "Tool results require 'call_id'.",
        ),
        bad(
            "output call_id unknown",
            two_calls(&output("zz")),
            Some("input[3].call_id"),
            "'call_id' zz does not match a tool call of the preceding assistant message.",
        ),
        bad(
            "output answered twice",
            two_calls(&format!("{}{}", output("c1"), output("c1"))),
            Some("input[4].call_id"),
            "Tool call c1 already has a response.",
        ),
        bad(
            "output content",
            two_calls(r#",{"type":"function_call_output","call_id":"c1","output":5}"#),
            Some("input[3].output"),
            "Content must be a string or a list of content parts.",
        ),
        bad(
            "a call missing its output before the next user item",
            two_calls(&format!(r#"{},{{"role":"user","content":"y"}}"#, output("c1"))),
            Some("input[1]"),
            &results_message("c2"),
        ),
        bad(
            "a call missing its output at the end",
            two_calls(&output("c2")),
            Some("input[1]"),
            &results_message("c1"),
        ),
        bad(
            "a call missing every output at the end",
            two_calls(""),
            Some("input[1]"),
            &results_message("c1, c2"),
        ),
        bad(
            "a call missing its output before a user item",
            items(
                r#"[{"role":"user","content":"x"},{"type":"function_call","call_id":"c1","name":"f","arguments":"{}"},{"role":"user","content":"y"}]"#,
            ),
            Some("input[1]"),
            &results_message("c1"),
        ),
        unsupported(
            "item type",
            item(r#""type":"web_search_call""#),
            "input[0].type=web_search_call",
        ),
        unsupported(
            "item type item_reference",
            item(r#""type":"item_reference","id":"i""#),
            "input[0].type=item_reference",
        ),
        // Content.
        bad(
            "content not a string or list",
            content("5"),
            Some("input[0].content"),
            "Content must be a string or a list of content parts.",
        ),
        bad(
            "content an object",
            content(r#"{"type":"input_text","text":"x"}"#),
            Some("input[0].content"),
            "Content must be a string or a list of content parts.",
        ),
        bad(
            "part not an object",
            content(r#"[{"type":"input_text","text":"x"},"y"]"#),
            Some("input[0].content[1]"),
            "Content part must be an object.",
        ),
        unsupported(
            "part type",
            part(r#"{"type":"input_image","image_url":"u"}"#),
            "input[0].content[0].type=input_image",
        ),
        unsupported(
            "part without type",
            part(r#"{"text":"x"}"#),
            "input[0].content[0].type=null",
        ),
        unsupported("part type number", part(r#"{"type":7}"#), "input[0].content[0].type=7"),
        unrecognized(
            "part key",
            part(r#"{"type":"input_text","text":"x","x":1}"#),
            "input[0].content[0].x",
        ),
        unsupported(
            "part annotations",
            part(r#"{"type":"output_text","text":"x","annotations":[{}]}"#),
            "input[0].content[0].annotations",
        ),
        bad(
            "part without text",
            part(r#"{"type":"input_text"}"#),
            Some("input[0].content[0].text"),
            "Text content part requires 'text'.",
        ),
        bad(
            "part text not a string",
            part(r#"{"type":"input_text","text":5}"#),
            Some("input[0].content[0].text"),
            "Text content part requires 'text'.",
        ),
        // Reasoning items.
        bad(
            "reasoning content part type",
            reasoning_item(r#""content":[{"type":"output_text","text":"x"}]"#),
            Some("input[0].content"),
            "Reasoning content must be reasoning_text parts.",
        ),
        bad(
            "reasoning content part text",
            reasoning_item(r#""content":[{"type":"reasoning_text"}]"#),
            Some("input[0].content"),
            "Reasoning content must be reasoning_text parts.",
        ),
        bad(
            "reasoning content part not an object",
            reasoning_item(r#""content":["x"]"#),
            Some("input[0].content"),
            "Reasoning content must be reasoning_text parts.",
        ),
        bad(
            "encrypted_content from elsewhere",
            reasoning_item(r#""encrypted_content":"gAAAA""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' was not issued by this server.",
        ),
        bad(
            "encrypted_content with a bad character",
            reasoning_item(r#""encrypted_content":"euhedral:a+b/""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' is malformed.",
        ),
        bad(
            "encrypted_content with a lone last character",
            reasoning_item(r#""encrypted_content":"euhedral:QUJDR""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' is malformed.",
        ),
        bad(
            "encrypted_content with missing padding character",
            reasoning_item(r#""encrypted_content":"euhedral:QQ=""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' is malformed.",
        ),
        bad(
            "encrypted_content with padding too early",
            reasoning_item(r#""encrypted_content":"euhedral:=""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' is malformed.",
        ),
        bad(
            "encrypted_content with text after padding",
            reasoning_item(r#""encrypted_content":"euhedral:QQ==QQ==""#),
            Some("input[0].encrypted_content"),
            "'encrypted_content' is malformed.",
        ),
        // Sampling.
        bad(
            "temperature",
            req(r#","temperature":2.5"#),
            Some("temperature"),
            "'temperature' must be between 0 and 2.",
        ),
        bad(
            "negative temperature",
            req(r#","temperature":-0.5"#),
            Some("temperature"),
            "'temperature' must be between 0 and 2.",
        ),
        bad(
            "top_p",
            req(r#","top_p":1.5"#),
            Some("top_p"),
            "'top_p' must be between 0 and 1.",
        ),
        bad(
            "negative top_p",
            req(r#","top_p":-1"#),
            Some("top_p"),
            "'top_p' must be between 0 and 1.",
        ),
    ]
}

#[test]
fn every_rejection_site_has_its_error() {
    let rows = rows();
    assert!(rows.len() >= 130, "{} rows", rows.len());
    for row in rows {
        let error = map(&row.body).expect_err(row.name);
        assert_eq!(error.status, row.status, "{}: status", row.name);
        assert_eq!(error.code, row.code, "{}: code", row.name);
        assert_eq!(error.param, row.param, "{}: param", row.name);
        assert_eq!(error.message, row.message, "{}: message", row.name);
        assert_eq!(error.kind, "invalid_request_error", "{}: type", row.name);
    }
}

fn first_error(body: &str) -> Option<String> {
    map(body).expect_err(body).param
}

#[test]
fn the_first_failing_check_decides_the_error() {
    // Each pair breaks the earlier-checked rule and a later one; the earlier one is reported.
    let unknown_before_model = r#"{"foo":1,"input":"x"}"#;
    assert_eq!(first_error(unknown_before_model).as_deref(), Some("foo"));
    let refused_before_unknown = r#"{"bogus":1,"background":true,"input":"x"}"#;
    assert_eq!(first_error(refused_before_unknown).as_deref(), Some("bogus"));
    let truncation_before_unknown = r#"{"truncation":"auto","bogus":1,"input":"x"}"#;
    assert_eq!(first_error(truncation_before_unknown).as_deref(), Some("truncation"));
    let model_before_include = r#"{"include":5,"input":"x"}"#;
    assert_eq!(first_error(model_before_include).as_deref(), Some("model"));
    let unknown_model_before_include = r#"{"model":"zzz","include":5,"input":"x"}"#;
    assert_eq!(first_error(unknown_model_before_include).as_deref(), Some("model"));
    let include_before_tools = req(r#","include":5,"tools":5,"reasoning":5,"text":5"#);
    assert_eq!(first_error(&include_before_tools).as_deref(), Some("include"));
    let tools_before_reasoning = req(r#","tools":5,"reasoning":5,"text":5"#);
    assert_eq!(first_error(&tools_before_reasoning).as_deref(), Some("tools"));
    let tools_before_choice = req(r#","tools":[5],"tool_choice":5"#);
    assert_eq!(first_error(&tools_before_choice).as_deref(), Some("tools[0]"));
    let tool_order =
        req(r#","tools":[{"type":"function","name":"a","description":1},{"type":"function","name":"a"},{"type":"x"}]"#);
    assert_eq!(first_error(&tool_order).as_deref(), Some("tools[0].description"));
    let tool_keys_before_name = req(r#","tools":[{"type":"function","name":"a b","x":1}]"#);
    assert_eq!(first_error(&tool_keys_before_name).as_deref(), Some("tools[0].x"));
    let duplicate_before_description =
        req(r#","tools":[{"type":"function","name":"a"},{"type":"function","name":"a","description":1}]"#);
    assert_eq!(
        first_error(&duplicate_before_description).as_deref(),
        Some("tools[1].name")
    );
    let strict_before_parameters = req(r#","tools":[{"type":"function","name":"a","strict":1,"parameters":5}]"#);
    assert_eq!(
        first_error(&strict_before_parameters).as_deref(),
        Some("tools[0].strict")
    );
    let choice_before_reasoning = req(r#","tool_choice":"required","reasoning":5"#);
    assert_eq!(first_error(&choice_before_reasoning).as_deref(), Some("tool_choice"));
    let reasoning_before_effort = req(r#","reasoning":{"x":1,"effort":"bogus","summary":"bogus"},"text":5"#);
    assert_eq!(first_error(&reasoning_before_effort).as_deref(), Some("reasoning.x"));
    let summary_before_effort = req(r#","reasoning":{"effort":"bogus","summary":"bogus"},"text":5"#);
    assert_eq!(
        first_error(&summary_before_effort).as_deref(),
        Some("reasoning.summary")
    );
    let summary_before_generate = req(r#","reasoning":{"generate_summary":"x","summary":"x"}"#);
    assert_eq!(
        first_error(&summary_before_generate).as_deref(),
        Some("reasoning.summary")
    );
    let effort_before_text = req(r#","reasoning":{"effort":"bogus"},"text":5,"temperature":9"#);
    assert_eq!(first_error(&effort_before_text).as_deref(), Some("reasoning_effort"));
    let text_before_input = r#"{"model":"m","text":5,"temperature":9}"#;
    assert_eq!(first_error(text_before_input).as_deref(), Some("text"));
    let text_key_before_verbosity = req(r#","text":{"verbosity":"low","x":1}"#);
    assert_eq!(first_error(&text_key_before_verbosity).as_deref(), Some("text.x"));
    let verbosity_before_format = req(r#","text":{"verbosity":"low","format":5}"#);
    assert_eq!(first_error(&verbosity_before_format).as_deref(), Some("text.verbosity"));
    let format_before_input = r#"{"model":"m","text":{"format":{"type":"nope"}},"temperature":9}"#;
    assert_eq!(
        first_error(format_before_input).as_deref(),
        Some("text.format.type=nope")
    );
    let input_before_sampling = r#"{"model":"m","temperature":9,"top_p":9}"#;
    assert_eq!(first_error(input_before_sampling).as_deref(), Some("input"));
    let sampling_after_input = r#"{"model":"m","input":[],"temperature":9}"#;
    assert_eq!(first_error(sampling_after_input).as_deref(), Some("input"));
    let temperature_before_top_p = req(r#","temperature":9,"top_p":9"#);
    assert_eq!(first_error(&temperature_before_top_p).as_deref(), Some("temperature"));
    // Within an item: unknown key, then role, then content, then the role's own rule.
    let key_before_role = item(r#""role":"robot","x":1"#);
    assert_eq!(first_error(&key_before_role).as_deref(), Some("input[0].x"));
    let role_before_content = item(r#""type":"message","content":5"#);
    assert_eq!(first_error(&role_before_content).as_deref(), Some("input[0].role"));
    let content_before_role_name = item(r#""role":"robot","content":5"#);
    assert_eq!(
        first_error(&content_before_role_name).as_deref(),
        Some("input[0].content")
    );
    let content_before_leading = items(r#"[{"role":"user","content":"x"},{"role":"system","content":5}]"#);
    assert_eq!(
        first_error(&content_before_leading).as_deref(),
        Some("input[1].content")
    );
    // The first of two bad items.
    let two_items = items(r#"[{"role":"user","content":"x"},{"role":"user","content":5},{"role":"robot"}]"#);
    assert_eq!(first_error(&two_items).as_deref(), Some("input[1].content"));
    // A part's type before its keys before its annotations and text.
    let part_type_before_keys = with_content(r#"[{"type":"nope","x":1}]"#);
    assert_eq!(
        first_error(&part_type_before_keys).as_deref(),
        Some("input[0].content[0].type=nope")
    );
    let part_keys_before_annotations = with_content(r#"[{"type":"input_text","x":1,"annotations":[1]}]"#);
    assert_eq!(
        first_error(&part_keys_before_annotations).as_deref(),
        Some("input[0].content[0].x")
    );
    let annotations_before_text = with_content(r#"[{"type":"input_text","annotations":[1]}]"#);
    assert_eq!(
        first_error(&annotations_before_text).as_deref(),
        Some("input[0].content[0].annotations")
    );
    // A call: id, then repeat, then name, then arguments.
    let id_before_name = item(r#""type":"function_call","name":"a b","arguments":5"#);
    assert_eq!(first_error(&id_before_name).as_deref(), Some("input[0].call_id"));
    let name_before_arguments = item(r#""type":"function_call","call_id":"c","name":"a b","arguments":5"#);
    assert_eq!(first_error(&name_before_arguments).as_deref(), Some("input[0].name"));
    let repeat_before_name = items(
        r#"[{"type":"function_call","call_id":"c","name":"f","arguments":""},{"type":"function_call","call_id":"c","name":"a b"}]"#,
    );
    assert_eq!(first_error(&repeat_before_name).as_deref(), Some("input[1].call_id"));
    // An output: its place, then its content, then its id.
    let place_before_content = item(r#""type":"function_call_output","call_id":"c","output":5"#);
    assert_eq!(first_error(&place_before_content).as_deref(), Some("input[0]"));
    let content_before_id = two_calls(r#",{"type":"function_call_output","call_id":"zz","output":5}"#);
    assert_eq!(first_error(&content_before_id).as_deref(), Some("input[3].output"));
    // Missing results are reported when the next item arrives, before that item is looked at.
    let missing_before_item =
        items(r#"[{"type":"function_call","call_id":"c","name":"f","arguments":""},{"role":"user","content":5}]"#);
    assert_eq!(first_error(&missing_before_item).as_deref(), Some("input[0]"));
    let missing_before_bad_type =
        items(r#"[{"type":"function_call","call_id":"c","name":"f","arguments":""},{"type":"nope"}]"#);
    assert_eq!(first_error(&missing_before_bad_type).as_deref(), Some("input[0]"));
    // A reasoning item's text is read before the reasoning that follows.
    let reasoning_items =
        items(r#"[{"type":"reasoning","encrypted_content":"x"},{"type":"reasoning","content":["y"]}]"#);
    assert_eq!(
        first_error(&reasoning_items).as_deref(),
        Some("input[0].encrypted_content")
    );
}

fn text_of(message: &Message) -> String {
    match message {
        Message::System(parts) | Message::User(parts) | Message::Tool(parts) => {
            parts.iter().map(|ContentPart::Text(text)| text.as_str()).collect()
        }
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|ContentPart::Text(text)| text.as_str())
            .collect(),
    }
}

fn assistant_of(message: &Message) -> &AssistantMessage {
    match message {
        Message::Assistant(assistant) => assistant,
        other => panic!("{other:?} is not an assistant message"),
    }
}

#[test]
fn a_minimal_request_takes_the_defaults() {
    let mapped = accepted(r#"{"model":"m","input":"hi"}"#);
    assert_eq!(
        mapped,
        GenerationRequest {
            model: "m".to_owned(),
            template: TemplateInput {
                messages: vec![Message::user("hi")],
                tools: Vec::new(),
            },
            thinking: TemplateOptions::default(),
            reasoning_budget: None,
            tools: ToolCalling {
                tools: Vec::new(),
                choice: ToolChoice::None,
                parallel: true,
            },
            format: ResponseFormat::Text,
            stops: Vec::new(),
            sampling: Sampling {
                temperature: 0.6,
                top_k: 20,
                top_p: 0.95,
                seed: None,
                greedy: false,
            },
            max_tokens: None,
            max_tokens_param: "max_output_tokens".to_owned(),
            stream: false,
            return_token_ids: false,
        }
    );
    assert!(mapped.reasoning());
}

#[test]
fn options_override_the_defaults() {
    let mapped = accepted(&req(
        r#","temperature":0.2,"top_p":0.5,"max_output_tokens":77,"stream":true,"parallel_tool_calls":false"#,
    ));
    assert_eq!(
        mapped.sampling,
        Sampling {
            temperature: 0.2,
            top_k: 20,
            top_p: 0.5,
            seed: None,
            greedy: false,
        }
    );
    assert_eq!(mapped.max_tokens, Some(77));
    assert!(mapped.stream);
    assert!(!mapped.tools.parallel, "the flag is kept without tools");
    assert!(mapped.tools.tools.is_empty());
    let greedy = accepted(&req(r#","temperature":0"#));
    assert!(greedy.sampling.greedy);
    assert_eq!(greedy.sampling.top_p, 1.0);
    let greedy_top_p = accepted(&req(r#","top_p":0"#));
    assert!(greedy_top_p.sampling.greedy);
    assert_eq!(greedy_top_p.sampling.top_p, 1.0);
    assert_eq!(
        accepted(&req(r#","temperature":2,"top_p":1"#)).sampling.temperature,
        2.0
    );
    // The budget is the planner's to check.
    assert_eq!(accepted(&req(r#","max_output_tokens":0"#)).max_tokens, Some(0));
    assert_eq!(accepted(&req(r#","max_output_tokens":-5"#)).max_tokens, Some(-5));
}

#[test]
fn ignored_and_neutral_fields_are_accepted() {
    for extra in [
        r#""user":"u""#,
        r#""metadata":{"a":"b"}"#,
        r#""store":false"#,
        r#""service_tier":"auto""#,
        r#""safety_identifier":"s""#,
        r#""prompt_cache_key":"k""#,
        r#""truncation":"disabled""#,
        r#""truncation":null"#,
        r#""background":false"#,
        r#""background":null"#,
        r#""previous_response_id":null"#,
        r#""conversation":null"#,
        r#""prompt":null"#,
        r#""max_tool_calls":null"#,
        r#""top_logprobs":null"#,
        r#""include":null"#,
        r#""include":[]"#,
        r#""include":["reasoning.encrypted_content"]"#,
        r#""tools":null"#,
        r#""tool_choice":null"#,
        r#""reasoning":null"#,
        r#""text":null"#,
        r#""text":{}"#,
        r#""text":{"verbosity":null,"format":null}"#,
        r#""reasoning":{"summary":"auto","generate_summary":"detailed"}"#,
        r#""tools":[],"tool_choice":"none""#,
        r#""tools":[],"tool_choice":"auto""#,
    ] {
        accepted(&req(&format!(",{extra}")));
    }
}

#[test]
fn input_as_a_string_or_a_list_maps_to_the_same_turns() {
    let expected = vec![Message::user("hi")];
    assert_eq!(accepted(&req("")).template.messages, expected);
    assert_eq!(
        accepted(&items(r#"[{"role":"user","content":"hi"}]"#))
            .template
            .messages,
        expected
    );
    assert_eq!(
        accepted(&items(
            r#"[{"type":"message","role":"user","content":"hi","id":"i","status":"completed"}]"#
        ))
        .template
        .messages,
        expected
    );
    assert_eq!(
        accepted(&items(
            r#"[{"role":"user","content":[{"type":"input_text","text":"h"},{"type":"input_text","text":"i"}]}]"#
        ))
        .template
        .messages,
        expected
    );
    assert_eq!(
        accepted(&items(
            r#"[{"role":"user","content":[{"type":"input_text","text":"hi","annotations":[],"logprobs":[]}]}]"#
        ))
        .template
        .messages,
        expected
    );
    // A null role still makes an item a message, whose role is then missing; a null content is empty.
    assert_eq!(
        accepted(&items(r#"[{"role":"user","content":null}]"#))
            .template
            .messages,
        vec![Message::user("")]
    );
    assert_eq!(
        accepted(&items(r#"[{"role":"user"}]"#)).template.messages,
        vec![Message::user("")]
    );
}

#[test]
fn instructions_and_system_items_join_into_one_leading_system_turn() {
    let mapped = accepted(
        r#"{"model":"m","instructions":"A.","input":[{"role":"developer","content":"B."},{"role":"system","content":[{"type":"input_text","text":"C."}]},{"role":"user","content":"x"}]}"#,
    );
    assert_eq!(
        mapped.template.messages,
        vec![Message::system("A.\n\nB.\n\nC."), Message::user("x")]
    );
    let only_items = accepted(&items(
        r#"[{"role":"developer","content":"B."},{"role":"user","content":"x"}]"#,
    ));
    assert_eq!(
        only_items.template.messages,
        vec![Message::system("B."), Message::user("x")]
    );
    let only_instructions = accepted(r#"{"model":"m","instructions":"Be brief.","input":"x"}"#);
    assert_eq!(
        only_instructions.template.messages,
        vec![Message::system("Be brief."), Message::user("x")]
    );
    // Empty instructions add no turn and no separator.
    let empty = accepted(
        r#"{"model":"m","instructions":"","input":[{"role":"system","content":"S"},{"role":"user","content":"x"}]}"#,
    );
    assert_eq!(empty.template.messages, vec![Message::system("S"), Message::user("x")]);
    let nothing = accepted(r#"{"model":"m","instructions":"","input":"x"}"#);
    assert_eq!(nothing.template.messages, vec![Message::user("x")]);
    // An empty system text after text still adds the separator.
    let separator =
        accepted(r#"{"model":"m","instructions":"A","input":[{"role":"system"},{"role":"user","content":"x"}]}"#);
    assert_eq!(separator.template.messages[0], Message::system("A\n\n"));
}

#[test]
fn assistant_items_make_one_turn_with_reasoning_text_and_calls() {
    let mapped = accepted(&items(
        r#"[{"role":"user","content":"Hi"},
        {"type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Think"},{"type":"reasoning_text","text":"ing"}]},
        {"type":"message","role":"assistant","content":[{"type":"output_text","text":"Do"}]},
        {"type":"message","role":"assistant","content":"ne."},
        {"role":"user","content":"Again"}]"#,
    ));
    let messages = &mapped.template.messages;
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0], Message::user("Hi"));
    assert_eq!(text_of(&messages[1]), "Done.");
    assert_eq!(assistant_of(&messages[1]).reasoning, "Thinking");
    assert!(assistant_of(&messages[1]).tool_calls.is_empty());
    assert_eq!(messages[2], Message::user("Again"));
    // An assistant message alone, and reasoning alone, are turns of their own.
    let alone = accepted(&items(r#"[{"role":"assistant","content":"a"}]"#));
    assert_eq!(alone.template.messages, vec![Message::assistant("a")]);
    let reasoning_alone = accepted(&items(
        r#"[{"type":"reasoning","summary":[{"text":"S"},{"type":"x"},5,{"text":"T"}]}]"#,
    ));
    assert_eq!(assistant_of(&reasoning_alone.template.messages[0]).reasoning, "ST");
    assert_eq!(text_of(&reasoning_alone.template.messages[0]), "");
}

#[test]
fn a_reasoning_item_reads_its_content_then_its_token_then_its_summary() {
    let reasoning = |members: &str| {
        let mapped = accepted(&item(&format!(r#""type":"reasoning",{members}"#)));
        assistant_of(&mapped.template.messages[0]).reasoning.clone()
    };
    assert_eq!(
        reasoning(
            r#""content":[{"type":"reasoning_text","text":"C"}],"encrypted_content":"junk","summary":[{"text":"S"}]"#
        ),
        "C"
    );
    assert_eq!(
        reasoning(&format!(
            r#""content":[],"encrypted_content":"{}","summary":[{{"text":"S"}}]"#,
            encrypt("T")
        )),
        "T"
    );
    assert_eq!(reasoning(r#""content":[],"summary":[{"text":"S"}]"#), "S");
    assert_eq!(
        reasoning(r#""content":null,"encrypted_content":null,"summary":null"#),
        ""
    );
    assert_eq!(reasoning(r#""encrypted_content":5,"summary":[{"text":"S"}]"#), "S");
    assert_eq!(reasoning(r#""content":"text","summary":[{"text":"S"}]"#), "S");
}

#[test]
fn the_opaque_reasoning_token_round_trips() {
    for text in [
        "",
        "a",
        "ab",
        "abc",
        "abcd",
        "Thinking it over.",
        "naive \u{e9}\u{4e2d}\u{1f600}\n\"quoted\"",
        "\u{3fff}?>",
    ] {
        let token = encrypt(text);
        assert!(token.starts_with("euhedral:"));
        let encoded = token.strip_prefix("euhedral:").expect("the prefix");
        assert!(!encoded.contains('='), "{token}");
        assert_eq!(decode_url_base64(encoded), Some(text.as_bytes().to_vec()), "{text}");
        let mapped = accepted(&item(&format!(r#""type":"reasoning","encrypted_content":"{token}""#)));
        assert_eq!(assistant_of(&mapped.template.messages[0]).reasoning, text);
    }
    assert_eq!(encrypt("Thinking it over."), "euhedral:VGhpbmtpbmcgaXQgb3Zlci4");
    assert_eq!(encrypt("\u{fb}\u{ff}"), "euhedral:w7vDvw");
}

#[test]
fn the_token_decoder_follows_the_runtimes_url_decoder() {
    let decode = |text: &str| decode_url_base64(text);
    assert_eq!(decode(""), Some(Vec::new()));
    assert_eq!(decode("QQ"), Some(b"A".to_vec()));
    assert_eq!(decode("QQ=="), Some(b"A".to_vec()));
    assert_eq!(decode("QUI"), Some(b"AB".to_vec()));
    assert_eq!(decode("QUI="), Some(b"AB".to_vec()));
    assert_eq!(decode("QUJD"), Some(b"ABC".to_vec()));
    assert_eq!(decode("-_-_"), Some(vec![0xfb, 0xff, 0xbf]));
    // Trailing bits need not be zero.
    assert_eq!(decode("QR"), Some(b"A".to_vec()));
    for refused in [
        "Q",
        "QUJDR",
        "QQ=",
        "QQ=A",
        "Q===",
        "=",
        "====",
        "QQ==Q",
        "QUJD=",
        "QUJD====",
        "a+b/",
        "QQ Q",
        "QQ\n",
        "\u{e9}\u{e9}",
    ] {
        assert_eq!(decode(refused), None, "{refused:?}");
    }
}

#[test]
fn function_calls_and_their_outputs_follow_call_order() {
    let mapped = accepted(&items(
        r#"[{"role":"user","content":"Weather?"},
        {"type":"function_call","id":"fc_1","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"Oslo\",\"n\":[1,2.5]}","status":"completed"},
        {"type":"function_call","call_id":"c2","name":"other","arguments":""},
        {"type":"function_call_output","call_id":"c2","output":"second"},
        {"type":"function_call_output","id":"fo","call_id":"c1","output":[{"type":"input_text","text":"-3"},{"type":"output_text","text":"C"}],"status":"completed"},
        {"role":"user","content":"Thanks"}]"#,
    ));
    let messages = &mapped.template.messages;
    assert_eq!(messages.len(), 5);
    let turn = assistant_of(&messages[1]);
    assert_eq!(text_of(&messages[1]), "");
    assert_eq!(turn.tool_calls.len(), 2);
    assert_eq!(turn.tool_calls[0].name, "get_weather");
    assert_eq!(
        turn.tool_calls[0].arguments,
        vec![
            ("city".to_owned(), JsonValue::String("Oslo".to_owned())),
            (
                "n".to_owned(),
                JsonValue::Array(vec![
                    JsonValue::Integer(JsonInteger::parse("1").expect("an integer")),
                    JsonValue::Float(2.5)
                ])
            ),
        ]
    );
    assert_eq!(turn.tool_calls[1].name, "other");
    assert!(turn.tool_calls[1].arguments.is_empty());
    assert_eq!(messages[2], Message::tool("-3C"));
    assert_eq!(messages[3], Message::tool("second"));
    assert_eq!(messages[4], Message::user("Thanks"));
    // Results close at the end of the input too, and an assistant text before the calls stays in the turn.
    let at_end = accepted(&items(
        r#"[{"role":"user","content":"x"},{"role":"assistant","content":"Let me look."},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":null}]"#,
    ));
    assert_eq!(at_end.template.messages.len(), 3);
    assert_eq!(text_of(&at_end.template.messages[1]), "Let me look.");
    assert_eq!(assistant_of(&at_end.template.messages[1]).tool_calls.len(), 1);
    assert_eq!(at_end.template.messages[2], Message::tool(""));
    // A new turn's call may reuse an earlier turn's id.
    accepted(&items(
        r#"[{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"r"},{"role":"user","content":"x"},{"type":"function_call","call_id":"c","name":"f","arguments":"{}"},{"type":"function_call_output","call_id":"c","output":"r"}]"#,
    ));
}

#[test]
fn arguments_are_one_json_object() {
    let arguments = |text: &str| {
        let mapped = accepted(&items(&format!(
            r#"[{{"type":"function_call","call_id":"c","name":"f","arguments":{}}},{{"type":"function_call_output","call_id":"c","output":""}}]"#,
            JsonValue::String(text.to_owned()).to_compact_string()
        )));
        assistant_of(&mapped.template.messages[0]).tool_calls[0]
            .arguments
            .clone()
    };
    assert_eq!(arguments(r#"  {"a":1}  "#).len(), 1);
    assert!(arguments("{}").is_empty());
    // The argument names are not checked.
    assert_eq!(arguments(r#"{"a<b":1}"#).len(), 1);
}

#[test]
fn include_asks_for_the_encrypted_reasoning_token() {
    assert!(!accepted_mapped(&req("")).include_encrypted_content);
    assert!(!accepted_mapped(&req(r#","include":[]"#)).include_encrypted_content);
    assert!(accepted_mapped(&req(r#","include":["reasoning.encrypted_content"]"#)).include_encrypted_content);
}

#[test]
fn tools_map_with_their_choice_and_parallel_flag() {
    let mapped = accepted(&req(
        r#","tools":[{"type":"function","name":"a","description":"d","parameters":{"type":"object","properties":{"x":{"type":"string"}}}},{"type":"function","name":"b","strict":false},{"type":"function","name":"c","strict":null,"description":null,"parameters":null}]"#,
    ));
    assert_eq!(mapped.tools.choice, ToolChoice::Auto);
    assert!(mapped.tools.parallel);
    let tools = &mapped.tools.tools;
    assert_eq!(tools.len(), 3);
    assert_eq!(tools[0].name(), "a");
    assert_eq!(tools[0].definition.description.as_deref(), Some("d"));
    assert!(tools[0].definition.parameters.is_some());
    assert!(tools[0].strict, "tools are strict unless they say otherwise");
    assert_eq!(tools[0].parameters_param, "tools[0].parameters");
    assert!(!tools[1].strict);
    assert_eq!(tools[1].parameters_param, "tools[1].parameters");
    assert!(tools[2].strict);
    assert_eq!(tools[2].definition.description, None);
    assert_eq!(tools[2].definition.parameters, None);
    assert_eq!(mapped.template.tools.len(), 3);
    let choice = |choice: &str| {
        accepted(&req(&format!(
            r#","tools":[{TOOL}],"parallel_tool_calls":false,"tool_choice":{choice}"#
        )))
    };
    for (given, expected) in [
        (r#""auto""#, ToolChoice::Auto),
        (r#""required""#, ToolChoice::Required),
        (r#""none""#, ToolChoice::None),
        (
            r#"{"type":"function","name":"a"}"#,
            ToolChoice::Function("a".to_owned()),
        ),
        ("null", ToolChoice::Auto),
    ] {
        let mapped = choice(given);
        assert_eq!(mapped.tools.choice, expected, "{given}");
        assert!(!mapped.tools.parallel, "{given}");
        let offered = if expected == ToolChoice::None { 0 } else { 1 };
        assert_eq!(mapped.template.tools.len(), offered, "{given}");
    }
}

#[test]
fn no_tools_and_empty_tools_are_disabled() {
    for body in [
        req(""),
        req(r#","tools":[]"#),
        req(r#","tools":null,"tool_choice":"none""#),
    ] {
        let mapped = accepted(&body);
        assert!(mapped.tools.tools.is_empty());
        assert_eq!(mapped.tools.choice, ToolChoice::None);
        assert!(mapped.template.tools.is_empty());
        assert!(!mapped.tools.parses_output());
    }
}

#[test]
fn reasoning_options_resolve_to_template_variables() {
    let options = |effort: &str| accepted(&req(&format!(r#","reasoning":{effort}"#))).thinking;
    let on = |effort| TemplateOptions {
        effort: Some(effort),
        preserve_thinking: true,
    };
    assert_eq!(accepted(&req("")).thinking, TemplateOptions::default());
    assert_eq!(options("{}"), on(Effort::XHigh));
    assert_eq!(options(r#"{"effort":null}"#), on(Effort::XHigh));
    assert_eq!(options(r#"{"effort":"low"}"#), on(Effort::Low));
    assert_eq!(options(r#"{"effort":"medium"}"#), on(Effort::Medium));
    assert_eq!(options(r#"{"effort":"high"}"#), on(Effort::XHigh));
    assert_eq!(options(r#"{"effort":"xhigh"}"#), on(Effort::XHigh));
    assert_eq!(options(r#"{"effort":"none"}"#), TemplateOptions::THINKING_DISABLED);
    assert_eq!(options(r#"{"effort":"low","summary":"concise"}"#), on(Effort::Low));
    assert!(!accepted(&req(r#","reasoning":{"effort":"none"}"#)).reasoning());
}

#[test]
fn response_formats_map() {
    let format = |format: &str| accepted(&req(&format!(r#","text":{{"format":{format}}}"#))).format;
    assert_eq!(format(r#"{"type":"text"}"#), ResponseFormat::Text);
    assert_eq!(format(r#"{"type":"json_object"}"#), ResponseFormat::JsonObject);
    let ResponseFormat::JsonSchema {
        name,
        schema,
        schema_param,
    } = format(
        r#"{"type":"json_schema","name":"n","description":"d","strict":true,"schema":{"type":"object","properties":{"n":{"type":"integer"}}}}"#,
    )
    else {
        panic!("a JSON schema format");
    };
    assert_eq!(name.as_deref(), Some("n"));
    assert_eq!(
        schema.to_compact_string(),
        r#"{"type":"object","properties":{"n":{"type":"integer"}}}"#
    );
    assert_eq!(schema_param, "text.format.schema");
    assert_eq!(
        format(r#"{"type":"json_schema","name":"n"}"#),
        ResponseFormat::JsonSchema {
            name: Some("n".to_owned()),
            schema: JsonValue::Object(Vec::new()),
            schema_param: "text.format.schema".to_owned(),
        }
    );
    assert_eq!(
        accepted(&req(r#","text":{"verbosity":null}"#)).format,
        ResponseFormat::Text
    );
}

#[test]
fn a_json_format_does_not_refuse_the_conversation_around_it() {
    // Unlike Chat Completions this surface has no stop sequences to combine with.
    let mapped = accepted(&req(r#","text":{"format":{"type":"json_object"}}"#));
    assert!(mapped.stops.is_empty());
    assert!(mapped.format.is_json());
}

#[test]
fn value_text_writes_values_as_a_param_spells_them() {
    let text = |json: &str| value_text(&JsonValue::parse(json).expect("json"));
    assert_eq!(text("null"), "null");
    assert_eq!(text("true"), "true");
    assert_eq!(text("12"), "12");
    assert_eq!(text("1.5"), "1.5");
    assert_eq!(text("1e10"), "1.0E10");
    assert_eq!(text(r#""s""#), "s");
    assert_eq!(text(r#"[1,"a",[],{}]"#), "[1, a, [], {}]");
    assert_eq!(text(r#"{"a":1,"b":[true]}"#), "{a=1, b=[true]}");
}
