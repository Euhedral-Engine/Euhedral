use super::*;
use crate::json::JsonInteger;
use crate::pipeline::ToolChoice;
use crate::template::{Effort, TemplateOptions};
use crate::tool_calling::ToolCalling;

fn defaults() -> Defaults {
    Defaults {
        model: "m".to_owned(),
        temperature: 0.6,
        top_p: 0.95,
        top_k: 20,
        max_tokens: 4096,
    }
}

fn map(body: &str) -> Result<GenerationRequest, ApiError> {
    map_request(body).map(|mapped| mapped.generation)
}

fn map_request(body: &str) -> Result<ChatRequest, ApiError> {
    let request = ChatCompletionRequest::parse(body).expect("the body parses");
    map_chat(request, &defaults())
}

fn accepted(body: &str) -> GenerationRequest {
    map(body).unwrap_or_else(|error| panic!("{body} was rejected: {}", error.message))
}

fn accepted_request(body: &str) -> ChatRequest {
    map_request(body).unwrap_or_else(|error| panic!("{body} was rejected: {}", error.message))
}

/// A request of one user message with `rest` appended to its members (`,"key":value`).
fn req(rest: &str) -> String {
    format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}]{rest}}}"#)
}

/// A request with the given `messages` array.
fn msgs(messages: &str) -> String {
    format!(r#"{{"model":"m","messages":{messages}}}"#)
}

/// A request with one message made of `members` (`"role":"user",...`).
fn msg(members: &str) -> String {
    msgs(&format!("[{{{members}}}]"))
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

const TOOL: &str = r#"{"type":"function","function":{"name":"a"}}"#;

fn with_tools(tools: &str, rest: &str) -> String {
    req(&format!(r#","tools":{tools}{rest}"#))
}

fn with_tool(function: &str) -> String {
    with_tools(&format!(r#"[{{"type":"function","function":{function}}}]"#), "")
}

fn assistant_calls(calls: &str) -> String {
    msgs(&format!(
        r#"[{{"role":"user","content":"x"}},{{"role":"assistant","tool_calls":{calls}}}]"#
    ))
}

fn call(members: &str) -> String {
    assistant_calls(&format!("[{{{members}}}]"))
}

fn call_function(function: &str) -> String {
    call(&format!(r#""id":"c1","type":"function","function":{function}"#))
}

fn rows() -> Vec<Row> {
    let tool_choice = |choice: &str| with_tools(&format!("[{TOOL}]"), &format!(r#","tool_choice":{choice}"#));
    let kwargs = |kwargs: &str| req(&format!(r#","chat_template_kwargs":{kwargs}"#));
    let format = |format: &str| req(&format!(r#","response_format":{format}"#));
    let schema = |definition: &str| format(&format!(r#"{{"type":"json_schema","json_schema":{definition}}}"#));
    let content = |content: &str| msg(&format!(r#""role":"user","content":{content}"#));
    let part = |part: &str| content(&format!("[{part}]"));
    let results = |after: &str| {
        msgs(&format!(
            r#"[{{"role":"user","content":"x"}},{{"role":"assistant","tool_calls":[{{"id":"c1","type":"function","function":{{"name":"f","arguments":"{{}}"}}}},{{"id":"c2","type":"function","function":{{"name":"f","arguments":"{{}}"}}}}]}}{after}]"#
        ))
    };
    let tool_result = |id: &str| format!(r#",{{"role":"tool","content":"r"{id}}}"#);
    let tool_msg = |id: &str| tool_result(&format!(r#","tool_call_id":"{id}""#));
    let no_model = r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_owned();
    let model_message = "You must provide a model parameter.";
    let tool_choice_message = "'tool_choice' must be 'none', 'auto', 'required', or a function choice.";
    let name_message = "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.";
    let choice_without_tools = "'tool_choice' is only allowed when 'tools' are specified.";
    let schema_name =
        "'response_format.json_schema.name' must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.";
    let param_names = "Parameter names must be non-empty and free of '<', '>' and line breaks.";
    let required_names = "Required parameter names must be unique, non-empty and free of '<', '>' and line breaks.";
    vec![
        // The request's other fields.
        unrecognized("unknown field", req(r#","foo":1"#), "foo"),
        unrecognized("unknown field with null", req(r#","foo":null"#), "foo"),
        unsupported("n above one", req(r#","n":2"#), "n"),
        unsupported("logprobs on", req(r#","logprobs":true"#), "logprobs"),
        unsupported("top_logprobs", req(r#","top_logprobs":1"#), "top_logprobs"),
        unsupported(
            "frequency penalty",
            req(r#","frequency_penalty":0.5"#),
            "frequency_penalty",
        ),
        unsupported(
            "presence penalty",
            req(r#","presence_penalty":0.5"#),
            "presence_penalty",
        ),
        unsupported("repeat penalty", req(r#","repeat_penalty":1.1"#), "repeat_penalty"),
        unsupported(
            "repetition penalty",
            req(r#","repetition_penalty":1.1"#),
            "repetition_penalty",
        ),
        unsupported("min_p", req(r#","min_p":0.1"#), "min_p"),
        unsupported("logit bias", req(r#","logit_bias":{"1":2}"#), "logit_bias"),
        unsupported("functions", req(r#","functions":[{}]"#), "functions"),
        unsupported("function_call", req(r#","function_call":"auto""#), "function_call"),
        unsupported("modalities", req(r#","modalities":["text","audio"]"#), "modalities"),
        unsupported("audio", req(r#","audio":{}"#), "audio"),
        unsupported("prediction", req(r#","prediction":{}"#), "prediction"),
        unsupported("web search", req(r#","web_search_options":{}"#), "web_search_options"),
        unsupported("verbosity", req(r#","verbosity":"low""#), "verbosity"),
        // Streaming options.
        bad(
            "stream_options without stream",
            req(r#","stream_options":{}"#),
            Some("stream_options"),
            "The 'stream_options' parameter is only allowed when 'stream' is enabled.",
        ),
        unrecognized(
            "unknown stream option",
            req(r#","stream":true,"stream_options":{"x":1}"#),
            "stream_options.x",
        ),
        bad(
            "token ids while streaming",
            req(r#","stream":true,"return_token_ids":true"#),
            Some("return_token_ids"),
            "'return_token_ids' is available only without streaming.",
        ),
        // The model.
        bad("model missing", no_model, Some("model"), model_message),
        bad(
            "model blank",
            r#"{"model":" \t","messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
            Some("model"),
            model_message,
        ),
        Row {
            name: "model unknown",
            body: r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#.to_owned(),
            status: 404,
            code: Some("model_not_found"),
            param: Some("model".to_owned()),
            message: "The model 'other' does not exist or you do not have access to it.".to_owned(),
        },
        // Tools.
        bad(
            "tool_choice without tools",
            req(r#","tool_choice":"required""#),
            Some("tool_choice"),
            choice_without_tools,
        ),
        bad(
            "tool_choice object without tools",
            req(r#","tool_choice":{"type":"function"}"#),
            Some("tool_choice"),
            choice_without_tools,
        ),
        bad(
            "tool_choice with empty tools",
            with_tools("[]", r#","tool_choice":"required""#),
            Some("tool_choice"),
            choice_without_tools,
        ),
        bad(
            "tools not an array",
            with_tools("{}", ""),
            Some("tools"),
            "'tools' must be an array of tool definitions.",
        ),
        bad(
            "tool not an object",
            with_tools("[1]", ""),
            Some("tools[0]"),
            "Each tool must be an object.",
        ),
        bad(
            "tool without type",
            with_tools("[{}]", ""),
            Some("tools[0].type"),
            "Tool 'type' is required.",
        ),
        bad(
            "tool type not a string",
            with_tools(r#"[{"type":1}]"#, ""),
            Some("tools[0].type"),
            "Tool 'type' is required.",
        ),
        unsupported(
            "tool of another type",
            with_tools(r#"[{"type":"custom"}]"#, ""),
            "tools[0].type",
        ),
        unsupported(
            "tool of another type with an unknown key",
            with_tools(r#"[{"type":"custom","x":1}]"#, ""),
            "tools[0].type",
        ),
        unrecognized(
            "tool with unknown key",
            with_tools(r#"[{"type":"function","function":{"name":"a"},"x":1}]"#, ""),
            "tools[0].x",
        ),
        bad(
            "tool without function",
            with_tools(r#"[{"type":"function"}]"#, ""),
            Some("tools[0].function"),
            "Function tools require a 'function' object.",
        ),
        unrecognized(
            "function with unknown key",
            with_tool(r#"{"name":"a","x":1}"#),
            "tools[0].function.x",
        ),
        bad(
            "function name missing",
            with_tool("{}"),
            Some("tools[0].function.name"),
            name_message,
        ),
        bad(
            "function name with space",
            with_tool(r#"{"name":"a b"}"#),
            Some("tools[0].function.name"),
            name_message,
        ),
        bad(
            "function name too long",
            with_tool(&format!(r#"{{"name":"{}"}}"#, "n".repeat(65))),
            Some("tools[0].function.name"),
            name_message,
        ),
        bad(
            "function description not a string",
            with_tool(r#"{"name":"a","description":1}"#),
            Some("tools[0].function.description"),
            "Function 'description' must be a string.",
        ),
        bad(
            "function strict not a boolean",
            with_tool(r#"{"name":"a","strict":"yes"}"#),
            Some("tools[0].function.strict"),
            "Function 'strict' must be a boolean.",
        ),
        bad(
            "tool names repeated",
            with_tools(&format!("[{TOOL},{TOOL}]"), ""),
            Some("tools[1].function.name"),
            "Tool names must be unique; 'a' is defined twice.",
        ),
        bad(
            "parameters not an object",
            with_tool(r#"{"name":"a","parameters":[]}"#),
            Some("tools[0].function.parameters"),
            "Function 'parameters' must be a JSON Schema object.",
        ),
        bad(
            "parameters not of type object",
            with_tool(r#"{"name":"a","parameters":{"type":"string"}}"#),
            Some("tools[0].function.parameters.type"),
            "Function 'parameters' must describe an object; tool calls pass named arguments.",
        ),
        bad(
            "properties not an object",
            with_tool(r#"{"name":"a","parameters":{"properties":[]}}"#),
            Some("tools[0].function.parameters.properties"),
            "'properties' must be an object.",
        ),
        bad(
            "property name with bracket",
            with_tool(r#"{"name":"a","parameters":{"properties":{"p<q":{}}}}"#),
            Some("tools[0].function.parameters.properties"),
            param_names,
        ),
        bad(
            "required not an array",
            with_tool(r#"{"name":"a","parameters":{"required":"p"}}"#),
            Some("tools[0].function.parameters.required"),
            "'required' must be an array of strings.",
        ),
        bad(
            "required with a non-string",
            with_tool(r#"{"name":"a","parameters":{"required":["p",1]}}"#),
            Some("tools[0].function.parameters.required"),
            "'required' must be an array of strings.",
        ),
        bad(
            "required name repeated",
            with_tool(r#"{"name":"a","parameters":{"required":["p","p"]}}"#),
            Some("tools[0].function.parameters.required"),
            required_names,
        ),
        bad(
            "required name empty",
            with_tool(r#"{"name":"a","parameters":{"required":[""]}}"#),
            Some("tools[0].function.parameters.required"),
            required_names,
        ),
        bad(
            "required undeclared in a closed schema",
            with_tool(
                r#"{"name":"a","parameters":{"properties":{"q":{}},"required":["p"],"additionalProperties":false}}"#,
            ),
            Some("tools[0].function.parameters.required"),
            "Required parameters must be declared when additional properties are forbidden.",
        ),
        bad(
            "tool_choice unknown mode",
            tool_choice(r#""bogus""#),
            Some("tool_choice"),
            tool_choice_message,
        ),
        bad(
            "tool_choice number",
            tool_choice("5"),
            Some("tool_choice"),
            tool_choice_message,
        ),
        bad(
            "tool_choice array",
            tool_choice("[]"),
            Some("tool_choice"),
            tool_choice_message,
        ),
        bad(
            "tool_choice without type",
            tool_choice("{}"),
            Some("tool_choice.type"),
            "'tool_choice.type' is required.",
        ),
        unsupported(
            "tool_choice of another type",
            tool_choice(r#"{"type":"custom"}"#),
            "tool_choice.type",
        ),
        unsupported(
            "tool_choice of another type with an unknown key",
            tool_choice(r#"{"type":"custom","x":1}"#),
            "tool_choice.type",
        ),
        unrecognized(
            "tool_choice with unknown key",
            tool_choice(r#"{"type":"function","function":{"name":"a"},"x":1}"#),
            "tool_choice.x",
        ),
        bad(
            "tool_choice without function",
            tool_choice(r#"{"type":"function"}"#),
            Some("tool_choice.function"),
            "'tool_choice.function' is required.",
        ),
        unrecognized(
            "tool_choice function with unknown key",
            tool_choice(r#"{"type":"function","function":{"name":"a","y":1}}"#),
            "tool_choice.function.y",
        ),
        bad(
            "tool_choice function without name",
            tool_choice(r#"{"type":"function","function":{}}"#),
            Some("tool_choice.function.name"),
            "'tool_choice.function.name' is required.",
        ),
        bad(
            "tool_choice names an absent function",
            tool_choice(r#"{"type":"function","function":{"name":"zzz"}}"#),
            Some("tool_choice"),
            "'tool_choice' names function 'zzz', which is not in 'tools'.",
        ),
        // Reasoning.
        bad(
            "template kwargs not an object",
            kwargs("[]"),
            Some("chat_template_kwargs"),
            "'chat_template_kwargs' must be an object.",
        ),
        unsupported("unknown template kwarg", kwargs(r#"{"x":1}"#), "chat_template_kwargs.x"),
        bad(
            "enable_thinking not a boolean",
            kwargs(r#"{"enable_thinking":"yes"}"#),
            Some("chat_template_kwargs.enable_thinking"),
            "'chat_template_kwargs.enable_thinking' must be a boolean.",
        ),
        bad(
            "preserve_thinking not a boolean",
            kwargs(r#"{"preserve_thinking":1}"#),
            Some("chat_template_kwargs.preserve_thinking"),
            "'chat_template_kwargs.preserve_thinking' must be a boolean.",
        ),
        bad(
            "reasoning_effort minimal",
            req(r#","reasoning_effort":"minimal""#),
            Some("reasoning_effort"),
            "'reasoning_effort' 'minimal' is not supported: this model's reasoning levels are none, low, medium and high.",
        ),
        bad(
            "reasoning_effort unknown",
            req(r#","reasoning_effort":"extreme""#),
            Some("reasoning_effort"),
            "'reasoning_effort' must be one of none, low, medium, high (or xhigh).",
        ),
        bad(
            "reasoning_effort not a string",
            req(r#","reasoning_effort":3"#),
            Some("reasoning_effort"),
            "'reasoning_effort' must be one of none, low, medium, high (or xhigh).",
        ),
        bad(
            "template reasoning_effort unknown",
            kwargs(r#"{"reasoning_effort":"extreme"}"#),
            Some("chat_template_kwargs.reasoning_effort"),
            "'chat_template_kwargs.reasoning_effort' must be one of none, low, medium, high (or xhigh).",
        ),
        bad(
            "template reasoning_effort minimal",
            kwargs(r#"{"reasoning_effort":"minimal"}"#),
            Some("chat_template_kwargs.reasoning_effort"),
            "'chat_template_kwargs.reasoning_effort' 'minimal' is not supported: this model's reasoning levels are none, low, medium and high.",
        ),
        bad(
            "template reasoning_effort none",
            kwargs(r#"{"reasoning_effort":"none"}"#),
            Some("chat_template_kwargs.reasoning_effort"),
            "'chat_template_kwargs.reasoning_effort' must be low, medium or xhigh; use enable_thinking to turn thinking off.",
        ),
        bad(
            "reasoning efforts conflict",
            req(r#","reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"medium"}"#),
            Some("reasoning_effort"),
            "'reasoning_effort' and 'chat_template_kwargs.reasoning_effort' conflict; send only one.",
        ),
        bad(
            "thinking off with an effort",
            req(r#","reasoning_effort":"high","chat_template_kwargs":{"enable_thinking":false}"#),
            Some("reasoning_effort"),
            "'chat_template_kwargs.enable_thinking' conflicts with 'reasoning_effort'.",
        ),
        bad(
            "thinking on with effort none",
            req(r#","reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true}"#),
            Some("reasoning_effort"),
            "'chat_template_kwargs.enable_thinking' conflicts with 'reasoning_effort'.",
        ),
        // The response format.
        bad(
            "response_format not an object",
            format(r#""json""#),
            Some("response_format"),
            "'response_format' must be an object.",
        ),
        bad(
            "response_format without type",
            format("{}"),
            Some("response_format.type"),
            "'response_format.type' is required.",
        ),
        unsupported(
            "response_format of another type",
            format(r#"{"type":"xml"}"#),
            "response_format.type=xml",
        ),
        unrecognized(
            "text format with unknown key",
            format(r#"{"type":"text","x":1}"#),
            "response_format.x",
        ),
        unrecognized(
            "json_object format with unknown key",
            format(r#"{"type":"json_object","x":1}"#),
            "response_format.x",
        ),
        unrecognized(
            "json_schema format with unknown key",
            format(r#"{"type":"json_schema","json_schema":{"name":"n"},"x":1}"#),
            "response_format.x",
        ),
        bad(
            "json_schema missing",
            format(r#"{"type":"json_schema"}"#),
            Some("response_format.json_schema"),
            "'response_format.json_schema' is required.",
        ),
        unrecognized(
            "json_schema with unknown key",
            schema(r#"{"name":"n","x":1}"#),
            "response_format.json_schema.x",
        ),
        bad(
            "json_schema name missing",
            schema("{}"),
            Some("response_format.json_schema.name"),
            schema_name,
        ),
        bad(
            "json_schema name invalid",
            schema(r#"{"name":"a.b"}"#),
            Some("response_format.json_schema.name"),
            schema_name,
        ),
        bad(
            "json_schema description not a string",
            schema(r#"{"name":"n","description":1}"#),
            Some("response_format.json_schema.description"),
            "'response_format.json_schema.description' must be a string.",
        ),
        bad(
            "json_schema strict not a boolean",
            schema(r#"{"name":"n","strict":"yes"}"#),
            Some("response_format.json_schema.strict"),
            "'response_format.json_schema.strict' must be a boolean.",
        ),
        bad(
            "json_schema schema not an object",
            schema(r#"{"name":"n","schema":[]}"#),
            Some("response_format.json_schema.schema"),
            "'response_format.json_schema.schema' must be a JSON Schema object.",
        ),
        unsupported(
            "json_schema with x-guidance",
            schema(r#"{"name":"n","schema":{"x-guidance":{}}}"#),
            "response_format.json_schema.schema.x-guidance",
        ),
        bad(
            "stop with json_object",
            req(r#","response_format":{"type":"json_object"},"stop":"x""#),
            Some("stop"),
            "'stop' cannot be combined with a JSON 'response_format': a stop inside the JSON would end it invalid.",
        ),
        bad(
            "stop list with json_schema",
            req(r#","response_format":{"type":"json_schema","json_schema":{"name":"n"}},"stop":[]"#),
            Some("stop"),
            "'stop' cannot be combined with a JSON 'response_format': a stop inside the JSON would end it invalid.",
        ),
        // The messages.
        bad(
            "messages missing",
            r#"{"model":"m"}"#.to_owned(),
            Some("messages"),
            "'messages' must contain at least one message.",
        ),
        bad(
            "messages empty",
            msgs("[]"),
            Some("messages"),
            "'messages' must contain at least one message.",
        ),
        bad(
            "message null",
            msgs("[null]"),
            Some("messages[0]"),
            "Message must be an object.",
        ),
        unrecognized(
            "message with unknown key",
            msg(r#""role":"user","content":"x","foo":1"#),
            "messages[0].foo",
        ),
        unsupported(
            "message with refusal",
            msg(r#""role":"user","content":"x","refusal":"no""#),
            "messages[0].refusal",
        ),
        unsupported(
            "message with function_call",
            msg(r#""role":"user","content":"x","function_call":{}"#),
            "messages[0].function_call",
        ),
        unsupported(
            "message with audio",
            msg(r#""role":"user","content":"x","audio":{}"#),
            "messages[0].audio",
        ),
        bad(
            "role missing",
            msg(r#""content":"x""#),
            Some("messages[0].role"),
            "Message role is required.",
        ),
        bad(
            "role unknown",
            msg(r#""role":"robot","content":"x""#),
            Some("messages[0].role"),
            "Invalid message role 'robot'.",
        ),
        unsupported(
            "role function",
            msg(r#""role":"function","content":"x""#),
            "messages[0].role",
        ),
        bad(
            "tool_calls on a user message",
            msg(r#""role":"user","content":"x","tool_calls":[]"#),
            Some("messages[0].tool_calls"),
            "Only assistant messages may contain 'tool_calls'.",
        ),
        bad(
            "tool_call_id on a user message",
            msg(r#""role":"user","content":"x","tool_call_id":"c""#),
            Some("messages[0].tool_call_id"),
            "Only tool messages may contain 'tool_call_id'.",
        ),
        bad(
            "reasoning_content on a user message",
            msg(r#""role":"user","content":"x","reasoning_content":"r""#),
            Some("messages[0].reasoning_content"),
            "Only assistant messages may contain 'reasoning_content'.",
        ),
        bad(
            "reasoning_content not a string",
            msg(r#""role":"assistant","content":"x","reasoning_content":1"#),
            Some("messages[0].reasoning_content"),
            "'reasoning_content' must be a string.",
        ),
        bad(
            "user content missing",
            msg(r#""role":"user""#),
            Some("messages[0].content"),
            "Message content is required.",
        ),
        bad(
            "user content null",
            msg(r#""role":"user","content":null"#),
            Some("messages[0].content"),
            "Message content is required.",
        ),
        bad(
            "content not a string or array",
            content("1"),
            Some("messages[0].content"),
            "Message content must be a string or an array of content parts.",
        ),
        bad(
            "content part not an object",
            part("1"),
            Some("messages[0].content[0]"),
            "Content part must be an object.",
        ),
        unsupported(
            "content part of another type",
            part(r#"{"type":"image_url","image_url":{}}"#),
            "messages[0].content[0].type=image_url",
        ),
        bad(
            "content part without type",
            part("{}"),
            Some("messages[0].content[0].type"),
            "Content part type is required.",
        ),
        bad(
            "content part type not a string",
            part(r#"{"type":1}"#),
            Some("messages[0].content[0].type"),
            "Content part type is required.",
        ),
        bad(
            "text part without text",
            part(r#"{"type":"text"}"#),
            Some("messages[0].content[0].text"),
            "Text content part requires 'text'.",
        ),
        bad(
            "text part without text and with an unknown key",
            part(r#"{"type":"text","cache":1}"#),
            Some("messages[0].content[0].text"),
            "Text content part requires 'text'.",
        ),
        unrecognized(
            "text part with unknown key",
            part(r#"{"type":"text","text":"x","cache":1}"#),
            "messages[0].content[0].cache",
        ),
        bad(
            "tool message first",
            msg(r#""role":"tool","content":"x","tool_call_id":"c""#),
            Some("messages[0].role"),
            "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'.",
        ),
        bad(
            "tool message after a plain assistant",
            msgs(
                r#"[{"role":"user","content":"x"},{"role":"assistant","content":"y"},{"role":"tool","content":"r","tool_call_id":"c"}]"#,
            ),
            Some("messages[2].role"),
            "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'.",
        ),
        // Tool calls of assistant messages.
        bad(
            "tool_calls not an array",
            assistant_calls("{}"),
            Some("messages[1].tool_calls"),
            "'tool_calls' must be an array.",
        ),
        bad(
            "tool call not an object",
            assistant_calls("[1]"),
            Some("messages[1].tool_calls[0]"),
            "Each tool call must be an object.",
        ),
        unrecognized(
            "tool call with unknown key",
            call(r#""id":"c1","x":1"#),
            "messages[1].tool_calls[0].x",
        ),
        bad(
            "tool call without id",
            call(r#""type":"function""#),
            Some("messages[1].tool_calls[0].id"),
            "Tool call 'id' is required.",
        ),
        bad(
            "tool call with empty id",
            call(r#""id":"""#),
            Some("messages[1].tool_calls[0].id"),
            "Tool call 'id' is required.",
        ),
        bad(
            "tool call ids repeated",
            assistant_calls(r#"[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}},{"id":"c1"}]"#),
            Some("messages[1].tool_calls[1].id"),
            "Tool call IDs must be unique within a message.",
        ),
        bad(
            "tool call without type",
            call(r#""id":"c1""#),
            Some("messages[1].tool_calls[0].type"),
            "Tool call 'type' is required.",
        ),
        unsupported(
            "tool call of another type",
            call(r#""id":"c1","type":"custom""#),
            "messages[1].tool_calls[0].type",
        ),
        bad(
            "tool call without function",
            call(r#""id":"c1","type":"function""#),
            Some("messages[1].tool_calls[0].function"),
            "Tool call 'function' is required.",
        ),
        unrecognized(
            "tool call function with unknown key",
            call_function(r#"{"name":"f","x":1}"#),
            "messages[1].tool_calls[0].function.x",
        ),
        bad(
            "tool call function name missing",
            call_function(r#"{"arguments":"{}"}"#),
            Some("messages[1].tool_calls[0].function.name"),
            name_message,
        ),
        bad(
            "tool call function name invalid",
            call_function(r#"{"name":"f g"}"#),
            Some("messages[1].tool_calls[0].function.name"),
            name_message,
        ),
        bad(
            "tool call arguments missing",
            call_function(r#"{"name":"f"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must be a JSON string.",
        ),
        bad(
            "tool call arguments not a string",
            call_function(r#"{"name":"f","arguments":{}}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must be a JSON string.",
        ),
        bad(
            "tool call arguments not JSON",
            call_function(r#"{"name":"f","arguments":"{"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must be valid JSON.",
        ),
        bad(
            "tool call arguments with trailing text",
            call_function(r#"{"name":"f","arguments":"{\"a\":1} trailing"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must be valid JSON.",
        ),
        bad(
            "tool call arguments with a second value",
            call_function(r#"{"name":"f","arguments":"{} {}"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must be valid JSON.",
        ),
        bad(
            "tool call arguments not an object",
            call_function(r#"{"name":"f","arguments":"[1]"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Tool call 'arguments' must encode a JSON object.",
        ),
        bad(
            "tool call argument name with bracket",
            call_function(r#"{"name":"f","arguments":"{\"a>b\":1}"}"#),
            Some("messages[1].tool_calls[0].function.arguments"),
            "Argument names must be non-empty and free of '<', '>' and line breaks.",
        ),
        // Tool results.
        bad(
            "tool result without id",
            results(&tool_result("")),
            Some("messages[2].tool_call_id"),
            "Tool results require 'tool_call_id'.",
        ),
        bad(
            "tool result with unknown id",
            results(&tool_msg("zz")),
            Some("messages[2].tool_call_id"),
            "'tool_call_id' zz does not match a tool call of the preceding assistant message.",
        ),
        bad(
            "tool result repeated",
            results(&format!("{}{}", tool_msg("c1"), tool_msg("c1"))),
            Some("messages[3].tool_call_id"),
            "Tool call c1 already has a response.",
        ),
        bad(
            "conversation continues before every call is answered",
            results(&format!(r#"{},{{"role":"user","content":"y"}}"#, tool_msg("c1"))),
            Some("messages[1].tool_calls"),
            "An assistant message with tool calls must be followed by a result for each call, matched by 'tool_call_id'. Missing responses: c2.",
        ),
        bad(
            "conversation ends before every call is answered",
            results(""),
            Some("messages[1].tool_calls"),
            "An assistant message with tool calls must be followed by a result for each call, matched by 'tool_call_id'. Missing responses: c1, c2.",
        ),
        // Stops, sampling and length.
        bad(
            "stop a number",
            req(r#","stop":5"#),
            Some("stop"),
            "'stop' must be a string or an array of strings.",
        ),
        bad(
            "stop with five sequences",
            req(r#","stop":["a","b","c","d","e"]"#),
            Some("stop"),
            "'stop' accepts at most 4 sequences.",
        ),
        bad(
            "stop with an empty sequence",
            req(r#","stop":["a",""]"#),
            Some("stop"),
            "Each stop sequence must be a non-empty string.",
        ),
        bad(
            "stop empty string",
            req(r#","stop":"" "#),
            Some("stop"),
            "Each stop sequence must be a non-empty string.",
        ),
        bad(
            "stop with a number",
            req(r#","stop":["a",1]"#),
            Some("stop"),
            "Each stop sequence must be a non-empty string.",
        ),
        bad(
            "temperature above two",
            req(r#","temperature":2.5"#),
            Some("temperature"),
            "'temperature' must be between 0 and 2.",
        ),
        bad(
            "temperature below zero",
            req(r#","temperature":-0.1"#),
            Some("temperature"),
            "'temperature' must be between 0 and 2.",
        ),
        bad(
            "top_p above one",
            req(r#","top_p":1.5"#),
            Some("top_p"),
            "'top_p' must be between 0 and 1.",
        ),
        bad(
            "top_p below zero",
            req(r#","top_p":-0.5"#),
            Some("top_p"),
            "'top_p' must be between 0 and 1.",
        ),
        bad(
            "top_k below minus one",
            req(r#","top_k":-2"#),
            Some("top_k"),
            "'top_k' must be at least 1, or 0 or -1 for no limit.",
        ),
        bad(
            "max_tokens differs from max_completion_tokens",
            req(r#","max_tokens":5,"max_completion_tokens":6"#),
            Some("max_tokens"),
            "'max_tokens' and 'max_completion_tokens' conflict; send only one.",
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

fn first_error(body: &str) -> (Option<String>, String) {
    let error = map(body).expect_err(body);
    (error.param, error.message)
}

#[test]
fn the_first_failing_check_decides_the_error() {
    // Each pair breaks the earlier-checked rule and a later one; the earlier one is reported.
    let no_model_unknown_field = r#"{"foo":1,"messages":[{"role":"user","content":"x"}]}"#;
    assert_eq!(first_error(no_model_unknown_field).0.as_deref(), Some("foo"));
    let tokens_and_model = r#"{"stream":true,"return_token_ids":true,"messages":[{"role":"user","content":"x"}]}"#;
    assert_eq!(first_error(tokens_and_model).0.as_deref(), Some("return_token_ids"));
    let stream_options_and_tokens =
        r#"{"stream_options":{},"return_token_ids":true,"stream":false,"model":"m","messages":[]}"#;
    assert_eq!(
        first_error(stream_options_and_tokens).0.as_deref(),
        Some("stream_options")
    );
    let model_and_tools = r#"{"tools":5,"messages":[{"role":"user","content":"x"}]}"#;
    assert_eq!(first_error(model_and_tools).0.as_deref(), Some("model"));
    let unknown_model_and_tools = r#"{"model":"zzz","tools":5,"messages":[]}"#;
    assert_eq!(first_error(unknown_model_and_tools).0.as_deref(), Some("model"));
    let tools_and_reasoning = req(r#","tools":5,"reasoning_effort":"bogus","response_format":5,"stop":5"#);
    assert_eq!(first_error(&tools_and_reasoning).0.as_deref(), Some("tools"));
    let reasoning_and_format = req(r#","reasoning_effort":"bogus","response_format":5,"stop":5"#);
    assert_eq!(
        first_error(&reasoning_and_format).0.as_deref(),
        Some("reasoning_effort")
    );
    let format_and_stop = req(r#","response_format":5,"stop":5"#);
    assert_eq!(first_error(&format_and_stop).0.as_deref(), Some("response_format"));
    let json_stop_and_messages = r#"{"model":"m","messages":[],"response_format":{"type":"json_object"},"stop":"x"}"#;
    assert_eq!(first_error(json_stop_and_messages).0.as_deref(), Some("stop"));
    let messages_and_stops = r#"{"model":"m","messages":[],"stop":5,"temperature":9}"#;
    assert_eq!(first_error(messages_and_stops).0.as_deref(), Some("messages"));
    let stop_and_sampling = req(r#","stop":5,"temperature":9,"top_k":-5,"max_tokens":1,"max_completion_tokens":2"#);
    assert_eq!(first_error(&stop_and_sampling).0.as_deref(), Some("stop"));
    let sampling_and_length = req(r#","temperature":9,"top_p":9,"top_k":-5,"max_tokens":1,"max_completion_tokens":2"#);
    assert_eq!(first_error(&sampling_and_length).0.as_deref(), Some("temperature"));
    let top_p_and_top_k = req(r#","top_p":9,"top_k":-5,"max_tokens":1,"max_completion_tokens":2"#);
    assert_eq!(first_error(&top_p_and_top_k).0.as_deref(), Some("top_p"));
    let top_k_and_length = req(r#","top_k":-5,"max_tokens":1,"max_completion_tokens":2"#);
    assert_eq!(first_error(&top_k_and_length).0.as_deref(), Some("top_k"));
    // Within the messages: unknown key, then role, then the misplaced fields, then content.
    let key_and_role = msg(r#""role":"robot","foo":1"#);
    assert_eq!(first_error(&key_and_role).0.as_deref(), Some("messages[0].foo"));
    let role_and_tool_calls = msg(r#""role":"robot","tool_calls":[]"#);
    assert_eq!(first_error(&role_and_tool_calls).0.as_deref(), Some("messages[0].role"));
    let tool_calls_and_content = msg(r#""role":"user","tool_calls":[],"tool_call_id":"c""#);
    assert_eq!(
        first_error(&tool_calls_and_content).0.as_deref(),
        Some("messages[0].tool_calls")
    );
    let id_and_reasoning = msg(r#""role":"user","tool_call_id":"c","reasoning_content":"r""#);
    assert_eq!(
        first_error(&id_and_reasoning).0.as_deref(),
        Some("messages[0].tool_call_id")
    );
    let reasoning_and_content = msg(r#""role":"assistant","reasoning_content":1"#);
    assert_eq!(
        first_error(&reasoning_and_content).0.as_deref(),
        Some("messages[0].reasoning_content")
    );
    let content_and_tool_role = msgs(r#"[{"role":"tool","tool_call_id":"c"}]"#);
    assert_eq!(
        first_error(&content_and_tool_role).0.as_deref(),
        Some("messages[0].content")
    );
    // The first of two bad messages.
    let two_messages = msgs(r#"[{"role":"user","content":"x"},{"role":"user"},{"role":"robot"}]"#);
    assert_eq!(first_error(&two_messages).0.as_deref(), Some("messages[1].content"));
    // A bad later call is not reached when an earlier call is bad.
    let two_calls = assistant_calls(r#"[{"id":"a","type":"x"},{"id":""}]"#);
    assert_eq!(
        first_error(&two_calls).0.as_deref(),
        Some("messages[1].tool_calls[0].type")
    );
    // The id check of a call comes after its unknown-key check.
    let keys_and_id = call(r#""x":1"#);
    assert_eq!(
        first_error(&keys_and_id).0.as_deref(),
        Some("messages[1].tool_calls[0].x")
    );
}

#[test]
fn a_minimal_request_takes_the_defaults() {
    let mapped = accepted(&req(""));
    assert_eq!(mapped.model, "m");
    assert_eq!(mapped.template.messages, vec![Message::user("hi")]);
    assert!(mapped.template.tools.is_empty());
    assert_eq!(mapped.thinking, TemplateOptions::default());
    assert!(mapped.reasoning());
    assert_eq!(mapped.reasoning_budget, None);
    assert_eq!(mapped.tools, ToolCalling::disabled());
    assert_eq!(mapped.format, ResponseFormat::Text);
    assert!(mapped.stops.is_empty());
    assert_eq!(
        mapped.sampling,
        Sampling {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed: None,
            greedy: false,
        }
    );
    assert_eq!(mapped.max_tokens, None);
    assert_eq!(mapped.max_tokens_param, "max_completion_tokens");
    assert!(!mapped.stream);
    assert!(!mapped.return_token_ids);
}

#[test]
fn usage_is_not_asked_for_unless_the_stream_options_say_so() {
    assert!(!accepted_request(&req("")).include_usage);
}

#[test]
fn sampling_options_override_the_defaults() {
    let mapped = accepted(&req(
        r#","temperature":1.5,"top_p":0.5,"top_k":7,"seed":-3,"max_tokens":9"#,
    ));
    assert_eq!(
        mapped.sampling,
        Sampling {
            temperature: 1.5,
            top_k: 7,
            top_p: 0.5,
            seed: Some(-3),
            greedy: false,
        }
    );
    assert_eq!(mapped.max_tokens, Some(9));
    assert_eq!(mapped.max_tokens_param, "max_tokens");
    for (top_k, expected) in [(0, 0), (-1, 0), (1, 1)] {
        let mapped = accepted(&req(&format!(r#","top_k":{top_k}"#)));
        assert_eq!(mapped.sampling.top_k, expected);
    }
    // The same budget under both names is not a conflict; either name alone is the budget.
    let both = accepted(&req(r#","max_tokens":5,"max_completion_tokens":5"#));
    assert_eq!(
        (both.max_tokens, both.max_tokens_param.as_str()),
        (Some(5), "max_tokens")
    );
    let completion = accepted(&req(r#","max_completion_tokens":6"#));
    assert_eq!(
        (completion.max_tokens, completion.max_tokens_param.as_str()),
        (Some(6), "max_completion_tokens")
    );
}

#[test]
fn zero_temperature_or_top_p_is_greedy_with_top_p_one() {
    for rest in [
        r#","temperature":0"#,
        r#","top_p":0"#,
        r#","temperature":0,"top_p":0.3"#,
    ] {
        let sampling = accepted(&req(rest)).sampling;
        assert!(sampling.greedy, "{rest}");
        assert_eq!(sampling.top_p, 1.0, "{rest}");
    }
    let sampling = accepted(&req(r#","temperature":0"#)).sampling;
    assert_eq!(sampling.temperature, 0.0);
    assert_eq!(sampling.top_k, 20);
}

#[test]
fn stops_are_a_string_or_an_array() {
    assert_eq!(accepted(&req(r#","stop":"END""#)).stops, ["END"]);
    assert_eq!(
        accepted(&req(r#","stop":["a","b","c","d"]"#)).stops,
        ["a", "b", "c", "d"]
    );
    assert!(accepted(&req(r#","stop":[]"#)).stops.is_empty());
    assert!(accepted(&req(r#","stop":null"#)).stops.is_empty());
}

#[test]
fn content_parts_are_joined_and_roles_are_mapped() {
    let mapped = accepted(&msgs(
        r#"[{"role":"developer","content":"be brief"},
            {"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]},
            {"role":"assistant","content":null},
            {"role":"user","content":[]},
            {"role":"system","content":"s","name":"ignored"}]"#,
    ));
    assert_eq!(
        mapped.template.messages,
        vec![
            Message::system("be brief"),
            Message::user("ab"),
            Message::assistant(""),
            Message::user(""),
            Message::system("s"),
        ]
    );
}

#[test]
fn an_assistant_message_replays_its_reasoning() {
    let mapped = accepted(&msgs(
        r#"[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":"why"}]"#,
    ));
    assert_eq!(
        mapped.template.messages[1],
        Message::Assistant(AssistantMessage {
            content: vec![ContentPart::Text("a".to_owned())],
            tool_calls: Vec::new(),
            reasoning: "why".to_owned(),
        })
    );
}

fn number(digits: &str) -> JsonValue {
    JsonValue::Integer(JsonInteger::parse(digits).expect("digits"))
}

#[test]
fn tool_calls_and_their_results_follow_call_order() {
    let mapped = accepted(&msgs(
        r#"[{"role":"user","content":"x"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"f","arguments":"{\"b\":1,\"a\":[true,null]}"}},
                {"id":"c2","type":"function","function":{"name":"g","arguments":""}}]},
            {"role":"tool","tool_call_id":"c2","content":"second"},
            {"role":"tool","tool_call_id":"c1","content":[{"type":"text","text":"fi"},{"type":"text","text":"rst"}]},
            {"role":"user","content":"thanks"}]"#,
    ));
    let calls = vec![
        ToolCall {
            name: "f".to_owned(),
            arguments: vec![
                ("b".to_owned(), number("1")),
                (
                    "a".to_owned(),
                    JsonValue::Array(vec![JsonValue::Bool(true), JsonValue::Null]),
                ),
            ],
        },
        ToolCall {
            name: "g".to_owned(),
            arguments: Vec::new(),
        },
    ];
    assert_eq!(
        mapped.template.messages,
        vec![
            Message::user("x"),
            Message::Assistant(AssistantMessage {
                content: vec![ContentPart::Text(String::new())],
                tool_calls: calls,
                reasoning: String::new(),
            }),
            Message::tool("first"),
            Message::tool("second"),
            Message::user("thanks"),
        ]
    );
}

#[test]
fn results_close_at_the_end_of_the_conversation() {
    let mapped = accepted(&msgs(
        r#"[{"role":"user","content":"x"},
            {"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"r"}]"#,
    ));
    assert_eq!(mapped.template.messages.len(), 3);
    assert_eq!(mapped.template.messages[2], Message::tool("r"));
}

#[test]
fn tools_map_with_their_choice_and_parallel_flag() {
    let both = r#"[{"type":"function","function":{"name":"a","description":"d","parameters":{"type":"object","properties":{"p":{"type":"string"}},"required":["p"]},"strict":true}},
                   {"type":"function","function":{"name":"b"}}]"#;
    let mapped = accepted(&with_tools(both, ""));
    assert_eq!(mapped.tools.choice, ToolChoice::Auto);
    assert!(mapped.tools.parallel);
    assert_eq!(mapped.tools.tools.len(), 2);
    assert!(mapped.tools.tools[0].strict);
    assert!(!mapped.tools.tools[1].strict);
    assert_eq!(mapped.tools.tools[0].parameters_param, "tools[0].function.parameters");
    assert_eq!(mapped.tools.tools[0].definition.description.as_deref(), Some("d"));
    assert_eq!(mapped.tools.tools[1].definition.parameters, None);
    assert_eq!(mapped.template.tools.len(), 2);
    assert_eq!(mapped.template.tools[0].name, "a");
    assert_eq!(mapped.template.tools, mapped.tools.prompt_tools());

    let required = accepted(&with_tools(
        both,
        r#","tool_choice":"required","parallel_tool_calls":false"#,
    ));
    assert_eq!(required.tools.choice, ToolChoice::Required);
    assert!(!required.tools.parallel);

    let named = accepted(&with_tools(
        both,
        r#","tool_choice":{"type":"function","function":{"name":"b"}}"#,
    ));
    assert_eq!(named.tools.choice, ToolChoice::Function("b".to_owned()));
    assert_eq!(named.tools.callable().len(), 1);

    // `none` keeps the tools but offers none of them to the model.
    let none = accepted(&with_tools(both, r#","tool_choice":"none""#));
    assert_eq!(none.tools.choice, ToolChoice::None);
    assert_eq!(none.tools.tools.len(), 2);
    assert!(none.template.tools.is_empty());
    assert_eq!(none.template.tools, none.tools.prompt_tools());
    assert!(none.tools.pipeline_options().is_none());
}

#[test]
fn no_tools_and_empty_tools_are_disabled() {
    for rest in [
        r#","tools":[]"#,
        r#","tools":[],"tool_choice":"auto""#,
        r#","tools":null,"tool_choice":"none""#,
        r#","tool_choice":"auto""#,
        r#","parallel_tool_calls":false"#,
    ] {
        assert_eq!(accepted(&req(rest)).tools, ToolCalling::disabled(), "{rest}");
    }
}

#[test]
fn reasoning_options_resolve_to_template_variables() {
    let cases = [
        ("", Some(Effort::XHigh), true),
        (r#","reasoning_effort":"none""#, None, true),
        (r#","reasoning_effort":"low""#, Some(Effort::Low), true),
        (r#","reasoning_effort":"medium""#, Some(Effort::Medium), true),
        (r#","reasoning_effort":"high""#, Some(Effort::XHigh), true),
        (r#","reasoning_effort":"xhigh""#, Some(Effort::XHigh), true),
        (r#","chat_template_kwargs":{"enable_thinking":false}"#, None, true),
        (
            r#","chat_template_kwargs":{"enable_thinking":true}"#,
            Some(Effort::XHigh),
            true,
        ),
        (
            r#","chat_template_kwargs":{"reasoning_effort":"low"}"#,
            Some(Effort::Low),
            true,
        ),
        (
            r#","chat_template_kwargs":{"preserve_thinking":false}"#,
            Some(Effort::XHigh),
            false,
        ),
        (
            r#","reasoning_effort":"low","chat_template_kwargs":{"reasoning_effort":"low","enable_thinking":true}"#,
            Some(Effort::Low),
            true,
        ),
        (
            r#","reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":false,"preserve_thinking":false}"#,
            None,
            false,
        ),
        (
            r#","reasoning_effort":null,"chat_template_kwargs":null"#,
            Some(Effort::XHigh),
            true,
        ),
    ];
    for (rest, effort, preserve) in cases {
        let thinking = accepted(&req(rest)).thinking;
        assert_eq!(thinking.effort, effort, "{rest}");
        assert_eq!(thinking.preserve_thinking, preserve, "{rest}");
    }
}

#[test]
fn response_formats_map() {
    assert_eq!(
        accepted(&req(r#","response_format":{"type":"text"}"#)).format,
        ResponseFormat::Text
    );
    assert_eq!(
        accepted(&req(r#","response_format":{"type":"json_object"}"#)).format,
        ResponseFormat::JsonObject
    );
    let mapped = accepted(&req(
        r#","response_format":{"type":"json_schema","json_schema":{"name":"n","description":"d","strict":false,"schema":{"type":"object","z":1,"a":2}}}"#,
    ));
    let ResponseFormat::JsonSchema {
        name,
        schema,
        schema_param,
    } = &mapped.format
    else {
        panic!("a schema format");
    };
    assert_eq!(name.as_deref(), Some("n"));
    assert_eq!(schema_param, "response_format.json_schema.schema");
    assert_eq!(schema.to_compact_string(), r#"{"type":"object","z":1,"a":2}"#);
    assert!(mapped.format.is_json());
    let without_schema = accepted(&req(
        r#","response_format":{"type":"json_schema","json_schema":{"name":"n"}}"#,
    ));
    assert_eq!(
        without_schema.format.answer_schema(),
        Some(JsonValue::Object(Vec::new()))
    );
    assert_eq!(
        accepted(&req(r#","response_format":{"type":"json_object"}"#))
            .format
            .answer_schema()
            .map(|schema| schema.to_compact_string()),
        Some(r#"{"type":"object"}"#.to_owned())
    );
}

#[test]
fn stream_options_and_token_ids() {
    let mapped = accepted_request(&req(
        r#","stream":true,"stream_options":{"include_usage":true,"include_obfuscation":false}"#,
    ));
    assert!(mapped.generation.stream);
    assert!(mapped.include_usage);
    let without_usage = accepted_request(&req(r#","stream":true,"stream_options":{"include_usage":false}"#));
    assert!(!without_usage.include_usage);
    let ids = accepted(&req(r#","return_token_ids":true"#));
    assert!(ids.return_token_ids);
    assert!(!accepted(&req(r#","stream":true,"return_token_ids":false"#)).return_token_ids);
}

#[test]
fn ignored_metadata_and_neutral_values_are_accepted() {
    accepted(&req(
        r#","user":"u","metadata":{"a":1},"store":false,"service_tier":"auto","safety_identifier":"s","prompt_cache_key":"k","prompt_cache_retention":"24h""#,
    ));
    accepted(&req(
        r#","n":1,"n":1.0,"logprobs":false,"top_logprobs":0,"frequency_penalty":0,"presence_penalty":0.0,"repeat_penalty":1,"repetition_penalty":1.0,"min_p":0,"logit_bias":{},"functions":[],"function_call":"none","modalities":["text"]"#,
    ));
    accepted(&req(
        r#","audio":null,"prediction":null,"web_search_options":null,"verbosity":null,"logprobs":null,"n":null,"modalities":null"#,
    ));
    accepted(&msg(
        r#""role":"user","content":"x","refusal":null,"function_call":null,"audio":null"#,
    ));
}

/// The prompt a request renders to.
fn prompt_of(body: &str) -> String {
    let request = accepted(body);
    crate::template::render(&request.template, &request.thinking).expect("the conversation renders")
}

/// The tool-call arguments of an assistant turn with the one JSON literal `literal` as the argument `n`.
fn call_with_number(literal: &str) -> String {
    msgs(&format!(
        r#"[{{"role":"user","content":"x"}},{{"role":"assistant","content":null,"tool_calls":[
            {{"id":"c1","type":"function","function":{{"name":"f","arguments":"{{\"n\":{literal}}}"}}}}]}},
            {{"role":"tool","tool_call_id":"c1","content":"r"}}]"#
    ))
}

#[test]
fn numbers_in_tool_call_arguments_reach_the_prompt_as_python_writes_them() {
    for (literal, written) in [
        ("123456789012345678901234567890", "123456789012345678901234567890"),
        ("-98765432109876543210987654321", "-98765432109876543210987654321"),
        ("0.1", "0.1"),
        ("2.5e-7", "2.5e-07"),
        ("1e400", "Infinity"),
        ("-1e400", "-Infinity"),
    ] {
        let prompt = prompt_of(&call_with_number(literal));
        let expected = format!("<parameter=n>\n{written}\n</parameter>");
        assert!(prompt.contains(&expected), "{literal} came out as: {prompt}");
    }
}

#[test]
fn numbers_in_tool_schemas_reach_the_prompt_as_python_writes_them() {
    let body = with_tool(
        r#"{"name":"f","parameters":{"type":"object","properties":{"a":{"type":"integer","default":123456789012345678901234567890},
            "b":{"type":"number","default":0.1},"c":{"type":"number","maximum":1e400},"d":{"type":"number","minimum":-1e400}}}}"#,
    );
    let prompt = prompt_of(&body);
    for written in [
        r#""default": 123456789012345678901234567890"#,
        r#""default": 0.1"#,
        r#""maximum": Infinity"#,
        r#""minimum": -Infinity"#,
    ] {
        assert!(prompt.contains(written), "{written} is missing from: {prompt}");
    }
}
