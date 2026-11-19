//! The API's errors: one status, type, code and message per failure, and the two JSON bodies that carry them.

use crate::json::JsonValue;
use crate::pipeline::PipelineError;
use crate::template::TemplateError;
use crate::wire::{ToJson, message_text, object, optional_text, text};

/// The API surface an error is answered on; each has its own error format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Surface {
    /// OpenAI's error object, used for every path except the messages paths.
    OpenAi,
    /// Anthropic's error object, used for paths under `/v1/messages`.
    Anthropic,
}

impl Surface {
    /// The surface that answers a request path: the messages paths are Anthropic's, all others OpenAI's.
    pub fn of_path(path: &str) -> Surface {
        if path.starts_with("/v1/messages") {
            Surface::Anthropic
        } else {
            Surface::OpenAi
        }
    }
}

/// An API failure with its HTTP status and error fields, in OpenAI's vocabulary (`type`, `param`, `code`). Messages are
/// safe to show a client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub status: u16,
    pub kind: &'static str,
    pub code: Option<&'static str>,
    pub param: Option<String>,
    pub message: String,
}

/// OpenAI's error envelope: `{"error": {"message", "type", "param", "code"}}`, every field always present.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiErrorEnvelope {
    pub error: OpenAiErrorBody,
}

/// The object inside OpenAI's error envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenAiErrorBody {
    pub message: String,
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Anthropic's error envelope: `{"type": "error", "error": {"type", "message"}}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnthropicErrorEnvelope {
    pub error: AnthropicErrorBody,
}

/// The object inside Anthropic's error envelope, its type named after the status.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AnthropicErrorBody {
    pub kind: &'static str,
    pub message: String,
}

impl ToJson for OpenAiErrorBody {
    fn to_json(&self) -> JsonValue {
        object([
            ("message", message_text(&self.message)),
            ("type", text(&self.kind)),
            ("param", optional_text(self.param.as_deref())),
            ("code", optional_text(self.code.as_deref())),
        ])
    }
}

impl ToJson for OpenAiErrorEnvelope {
    fn to_json(&self) -> JsonValue {
        object([("error", self.error.to_json())])
    }
}

impl ToJson for AnthropicErrorBody {
    fn to_json(&self) -> JsonValue {
        object([("type", text(self.kind)), ("message", message_text(&self.message))])
    }
}

impl ToJson for AnthropicErrorEnvelope {
    fn to_json(&self) -> JsonValue {
        object([("type", text("error")), ("error", self.error.to_json())])
    }
}

const INVALID_REQUEST: &str = "invalid_request_error";
const SERVER_ERROR: &str = "server_error";
const UNAVAILABLE: &str = "service_unavailable_error";

impl ApiError {
    fn new(
        status: u16,
        kind: &'static str,
        message: impl Into<String>,
        param: Option<&str>,
        code: Option<&'static str>,
    ) -> ApiError {
        ApiError {
            status,
            kind,
            code,
            param: param.map(str::to_owned),
            message: message.into(),
        }
    }

    /// A protocol error the web layer raised (405, 415, ...), with its status.
    pub fn protocol(status: u16, message: impl Into<String>) -> ApiError {
        ApiError::new(status, INVALID_REQUEST, message, None, None)
    }

    /// A JSON Schema or grammar that cannot be enforced while sampling.
    pub fn unenforceable_schema(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError::new(400, INVALID_REQUEST, message, param, Some("unsupported_schema"))
    }

    pub fn invalid_request(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError::new(400, INVALID_REQUEST, message, param, None)
    }

    pub fn request_too_large() -> ApiError {
        ApiError::new(
            413,
            INVALID_REQUEST,
            "The chat completion request body exceeds the configured size limit.",
            None,
            Some("request_too_large"),
        )
    }

    pub fn unsupported_parameter(param: &str) -> ApiError {
        ApiError::new(
            400,
            INVALID_REQUEST,
            format!("Unsupported parameter: '{param}' is not supported by this server."),
            Some(param),
            Some("unsupported_parameter"),
        )
    }

    pub fn unrecognized_argument(param: &str) -> ApiError {
        ApiError::new(
            400,
            INVALID_REQUEST,
            format!("Unrecognized request argument supplied: {param}"),
            Some(param),
            Some("unrecognized_argument"),
        )
    }

    pub fn context_length_exceeded(message: impl Into<String>, param: Option<&str>) -> ApiError {
        ApiError::new(400, INVALID_REQUEST, message, param, Some("context_length_exceeded"))
    }

    pub fn model_not_found(model: &str) -> ApiError {
        ApiError::new(
            404,
            INVALID_REQUEST,
            format!("The model '{model}' does not exist or you do not have access to it."),
            Some("model"),
            Some("model_not_found"),
        )
    }

    pub fn not_found(message: impl Into<String>) -> ApiError {
        ApiError::new(404, INVALID_REQUEST, message, None, None)
    }

    pub fn unavailable(message: impl Into<String>) -> ApiError {
        ApiError::new(503, UNAVAILABLE, message, None, Some("engine_unavailable"))
    }

    pub fn server_error() -> ApiError {
        ApiError::new(
            500,
            SERVER_ERROR,
            "The server had an error while processing your request.",
            None,
            None,
        )
    }

    /// The model's output began a tool call that cannot be returned for the offered tools. The reason describes only
    /// the generated text and the request's own definitions.
    pub fn invalid_tool_call(reason: &str) -> ApiError {
        ApiError::new(
            500,
            SERVER_ERROR,
            format!("The model produced an invalid tool call: {reason}."),
            None,
            Some("invalid_tool_call"),
        )
    }

    /// The model's output under a JSON response format is not a JSON document.
    pub fn invalid_structured_output() -> ApiError {
        ApiError::new(
            500,
            SERVER_ERROR,
            "The model's output is not valid JSON for the requested response format.",
            None,
            Some("invalid_structured_output"),
        )
    }

    /// The client closed its connection; nobody reads this, but it completes the request.
    pub fn client_closed() -> ApiError {
        ApiError::new(
            503,
            UNAVAILABLE,
            "The client closed the connection.",
            None,
            Some("client_closed"),
        )
    }

    pub fn timeout() -> ApiError {
        ApiError::new(
            503,
            UNAVAILABLE,
            "The request timed out before generation completed.",
            None,
            Some("timeout"),
        )
    }

    /// Every generation slot is busy and the queue is full.
    pub fn at_capacity() -> ApiError {
        ApiError::unavailable("The server is at generation capacity; retry later.")
    }

    /// The server is shutting down and takes no more generations.
    pub fn shutting_down() -> ApiError {
        ApiError::unavailable("The server is shutting down.")
    }

    /// A running generation was stopped because the server is shutting down.
    pub fn interrupted_by_shutdown() -> ApiError {
        ApiError::unavailable("Generation was interrupted by server shutdown.")
    }

    /// The engine cannot take work; `reason` is the engine's own text.
    pub fn engine_unavailable(reason: &str) -> ApiError {
        ApiError::unavailable(format!("The inference engine is unavailable: {reason}"))
    }

    /// A request body that is not valid JSON of the documented shape.
    pub fn unparsable_body() -> ApiError {
        ApiError::invalid_request(
            "We could not parse the JSON body of your request. Check that it is valid JSON and that each field has \
             the documented type.",
            None,
        )
    }

    /// A request with a method the URL does not take; Spring's wording.
    pub fn method_not_supported(method: &str) -> ApiError {
        ApiError::protocol(405, format!("Method '{method}' is not supported."))
    }

    /// A request whose content type the URL does not take; Spring's wording.
    pub fn unsupported_media_type(content_type: &str) -> ApiError {
        ApiError::protocol(415, format!("Content-Type '{content_type}' is not supported."))
    }

    /// A request whose content type header cannot be read; Spring's wording.
    pub fn unreadable_content_type() -> ApiError {
        ApiError::protocol(415, "Could not parse Content-Type.")
    }

    /// A request whose `Accept` header allows none of the types the URL produces; Spring's wording, naming the types
    /// the URL produces.
    pub fn not_acceptable(produces: &[&str]) -> ApiError {
        ApiError::protocol(406, format!("Acceptable representations: [{}].", produces.join(", ")))
    }

    /// A request whose `Accept` header cannot be read; Spring's wording.
    pub fn unreadable_accept() -> ApiError {
        ApiError::protocol(406, "Could not parse Accept header.")
    }

    /// Any other client error the web layer raised, with the detail it gave; `Invalid request.` when it gave none.
    pub fn framework(status: u16, detail: Option<&str>) -> ApiError {
        ApiError::protocol(status, detail.unwrap_or("Invalid request."))
    }

    /// A request for a URL the server does not serve.
    pub fn unknown_url(method: &str, path: &str) -> ApiError {
        ApiError::not_found(format!("Unknown request URL: {method} {path}."))
    }

    /// The HTTP status the error is answered with on a surface: Anthropic's clients retry 529 as overloaded, so its
    /// surface answers 529 where the server says 503.
    pub fn status_on(&self, surface: Surface) -> u16 {
        if surface == Surface::Anthropic && self.status == 503 {
            529
        } else {
            self.status
        }
    }

    /// Anthropic's error type for an HTTP status.
    pub fn anthropic_type(status: u16) -> &'static str {
        match status {
            400 | 405 | 406 | 415 => "invalid_request_error",
            401 => "authentication_error",
            403 => "permission_error",
            404 => "not_found_error",
            413 => "request_too_large",
            429 => "rate_limit_error",
            503 => "overloaded_error",
            _ => "api_error",
        }
    }

    /// The error as OpenAI's error object.
    pub fn openai_envelope(&self) -> OpenAiErrorEnvelope {
        OpenAiErrorEnvelope {
            error: OpenAiErrorBody {
                message: self.message.clone(),
                kind: self.kind.to_owned(),
                param: self.param.clone(),
                code: self.code.map(str::to_owned),
            },
        }
    }

    /// The error as Anthropic's error object.
    pub fn anthropic_envelope(&self) -> AnthropicErrorEnvelope {
        AnthropicErrorEnvelope {
            error: AnthropicErrorBody {
                kind: ApiError::anthropic_type(self.status),
                message: self.message.clone(),
            },
        }
    }

    /// The JSON body of the error on a surface.
    pub fn body(&self, surface: Surface) -> Vec<u8> {
        let text = match surface {
            Surface::OpenAi => self.openai_envelope().to_json_string(),
            Surface::Anthropic => self.anthropic_envelope().to_json_string(),
        };
        text.into_bytes()
    }
}

/// A conversation the template refuses is the client's: a bad request about its messages.
impl From<TemplateError> for ApiError {
    fn from(error: TemplateError) -> ApiError {
        ApiError::invalid_request(error.to_string(), Some("messages"))
    }
}

/// Output the pipeline cannot accept is a server error: the model produced an invalid tool call.
impl From<PipelineError> for ApiError {
    fn from(error: PipelineError) -> ApiError {
        ApiError::invalid_tool_call(error.message())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Piece;

    struct Row {
        error: ApiError,
        status: u16,
        kind: &'static str,
        param: Option<&'static str>,
        code: Option<&'static str>,
        message: &'static str,
        anthropic_status: u16,
        openai_body: &'static str,
        anthropic_body: &'static str,
    }

    fn text(error: &ApiError, surface: Surface) -> String {
        String::from_utf8(error.body(surface)).expect("utf-8")
    }

    #[test]
    fn every_constructor_has_its_status_type_code_and_bodies() {
        let rows = vec![
            Row {
                error: ApiError::protocol(405, "Method not allowed."),
                status: 405,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Method not allowed."##,
                anthropic_status: 405,
                openai_body: r##"{"error":{"message":"Method not allowed.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Method not allowed."}}"##,
            },
            Row {
                error: ApiError::protocol(429, "Slow down."),
                status: 429,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Slow down."##,
                anthropic_status: 429,
                openai_body: r##"{"error":{"message":"Slow down.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"rate_limit_error","message":"Slow down."}}"##,
            },
            Row {
                error: ApiError::unenforceable_schema("bad schema", Some("response_format")),
                status: 400,
                kind: "invalid_request_error",
                param: Some("response_format"),
                code: Some("unsupported_schema"),
                message: r##"bad schema"##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"bad schema","type":"invalid_request_error","param":"response_format","code":"unsupported_schema"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"bad schema"}}"##,
            },
            Row {
                error: ApiError::invalid_request("nope", Some("messages")),
                status: 400,
                kind: "invalid_request_error",
                param: Some("messages"),
                code: None,
                message: r##"nope"##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"nope","type":"invalid_request_error","param":"messages","code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"nope"}}"##,
            },
            Row {
                error: ApiError::invalid_request("nope", None),
                status: 400,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"nope"##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"nope","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"nope"}}"##,
            },
            Row {
                error: ApiError::request_too_large(),
                status: 413,
                kind: "invalid_request_error",
                param: None,
                code: Some("request_too_large"),
                message: r##"The chat completion request body exceeds the configured size limit."##,
                anthropic_status: 413,
                openai_body: r##"{"error":{"message":"The chat completion request body exceeds the configured size limit.","type":"invalid_request_error","param":null,"code":"request_too_large"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"request_too_large","message":"The chat completion request body exceeds the configured size limit."}}"##,
            },
            Row {
                error: ApiError::unsupported_parameter("n"),
                status: 400,
                kind: "invalid_request_error",
                param: Some("n"),
                code: Some("unsupported_parameter"),
                message: r##"Unsupported parameter: 'n' is not supported by this server."##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"Unsupported parameter: 'n' is not supported by this server.","type":"invalid_request_error","param":"n","code":"unsupported_parameter"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Unsupported parameter: 'n' is not supported by this server."}}"##,
            },
            Row {
                error: ApiError::unrecognized_argument("foo"),
                status: 400,
                kind: "invalid_request_error",
                param: Some("foo"),
                code: Some("unrecognized_argument"),
                message: r##"Unrecognized request argument supplied: foo"##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"Unrecognized request argument supplied: foo","type":"invalid_request_error","param":"foo","code":"unrecognized_argument"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Unrecognized request argument supplied: foo"}}"##,
            },
            Row {
                error: ApiError::context_length_exceeded("too long", Some("messages")),
                status: 400,
                kind: "invalid_request_error",
                param: Some("messages"),
                code: Some("context_length_exceeded"),
                message: r##"too long"##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"too long","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"too long"}}"##,
            },
            Row {
                error: ApiError::model_not_found("gpt-x"),
                status: 404,
                kind: "invalid_request_error",
                param: Some("model"),
                code: Some("model_not_found"),
                message: r##"The model 'gpt-x' does not exist or you do not have access to it."##,
                anthropic_status: 404,
                openai_body: r##"{"error":{"message":"The model 'gpt-x' does not exist or you do not have access to it.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"not_found_error","message":"The model 'gpt-x' does not exist or you do not have access to it."}}"##,
            },
            Row {
                error: ApiError::not_found("gone"),
                status: 404,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"gone"##,
                anthropic_status: 404,
                openai_body: r##"{"error":{"message":"gone","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"not_found_error","message":"gone"}}"##,
            },
            Row {
                error: ApiError::unavailable("busy"),
                status: 503,
                kind: "service_unavailable_error",
                param: None,
                code: Some("engine_unavailable"),
                message: r##"busy"##,
                anthropic_status: 529,
                openai_body: r##"{"error":{"message":"busy","type":"service_unavailable_error","param":null,"code":"engine_unavailable"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"##,
            },
            Row {
                error: ApiError::server_error(),
                status: 500,
                kind: "server_error",
                param: None,
                code: None,
                message: r##"The server had an error while processing your request."##,
                anthropic_status: 500,
                openai_body: r##"{"error":{"message":"The server had an error while processing your request.","type":"server_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"api_error","message":"The server had an error while processing your request."}}"##,
            },
            Row {
                error: ApiError::invalid_tool_call("unknown tool"),
                status: 500,
                kind: "server_error",
                param: None,
                code: Some("invalid_tool_call"),
                message: r##"The model produced an invalid tool call: unknown tool."##,
                anthropic_status: 500,
                openai_body: r##"{"error":{"message":"The model produced an invalid tool call: unknown tool.","type":"server_error","param":null,"code":"invalid_tool_call"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"api_error","message":"The model produced an invalid tool call: unknown tool."}}"##,
            },
            Row {
                error: ApiError::invalid_structured_output(),
                status: 500,
                kind: "server_error",
                param: None,
                code: Some("invalid_structured_output"),
                message: r##"The model's output is not valid JSON for the requested response format."##,
                anthropic_status: 500,
                openai_body: r##"{"error":{"message":"The model's output is not valid JSON for the requested response format.","type":"server_error","param":null,"code":"invalid_structured_output"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"api_error","message":"The model's output is not valid JSON for the requested response format."}}"##,
            },
            Row {
                error: ApiError::client_closed(),
                status: 503,
                kind: "service_unavailable_error",
                param: None,
                code: Some("client_closed"),
                message: r##"The client closed the connection."##,
                anthropic_status: 529,
                openai_body: r##"{"error":{"message":"The client closed the connection.","type":"service_unavailable_error","param":null,"code":"client_closed"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"overloaded_error","message":"The client closed the connection."}}"##,
            },
            Row {
                error: ApiError::timeout(),
                status: 503,
                kind: "service_unavailable_error",
                param: None,
                code: Some("timeout"),
                message: r##"The request timed out before generation completed."##,
                anthropic_status: 529,
                openai_body: r##"{"error":{"message":"The request timed out before generation completed.","type":"service_unavailable_error","param":null,"code":"timeout"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"overloaded_error","message":"The request timed out before generation completed."}}"##,
            },
            Row {
                error: ApiError::engine_unavailable("starting"),
                status: 503,
                kind: "service_unavailable_error",
                param: None,
                code: Some("engine_unavailable"),
                message: r##"The inference engine is unavailable: starting"##,
                anthropic_status: 529,
                openai_body: r##"{"error":{"message":"The inference engine is unavailable: starting","type":"service_unavailable_error","param":null,"code":"engine_unavailable"}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"overloaded_error","message":"The inference engine is unavailable: starting"}}"##,
            },
            Row {
                error: ApiError::unparsable_body(),
                status: 400,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type."##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type."}}"##,
            },
            Row {
                error: ApiError::unknown_url("GET", "/v1/nothing"),
                status: 404,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Unknown request URL: GET /v1/nothing."##,
                anthropic_status: 404,
                openai_body: r##"{"error":{"message":"Unknown request URL: GET /v1/nothing.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"not_found_error","message":"Unknown request URL: GET /v1/nothing."}}"##,
            },
            Row {
                error: ApiError::method_not_supported("PUT"),
                status: 405,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Method 'PUT' is not supported."##,
                anthropic_status: 405,
                openai_body: r##"{"error":{"message":"Method 'PUT' is not supported.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Method 'PUT' is not supported."}}"##,
            },
            Row {
                error: ApiError::unsupported_media_type("text/plain"),
                status: 415,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Content-Type 'text/plain' is not supported."##,
                anthropic_status: 415,
                openai_body: r##"{"error":{"message":"Content-Type 'text/plain' is not supported.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Content-Type 'text/plain' is not supported."}}"##,
            },
            Row {
                error: ApiError::unreadable_content_type(),
                status: 415,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Could not parse Content-Type."##,
                anthropic_status: 415,
                openai_body: r##"{"error":{"message":"Could not parse Content-Type.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Could not parse Content-Type."}}"##,
            },
            Row {
                error: ApiError::not_acceptable(&["application/json", "text/event-stream"]),
                status: 406,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Acceptable representations: [application/json, text/event-stream]."##,
                anthropic_status: 406,
                openai_body: r##"{"error":{"message":"Acceptable representations: [application/json, text/event-stream].","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Acceptable representations: [application/json, text/event-stream]."}}"##,
            },
            Row {
                error: ApiError::unreadable_accept(),
                status: 406,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Could not parse Accept header."##,
                anthropic_status: 406,
                openai_body: r##"{"error":{"message":"Could not parse Accept header.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Could not parse Accept header."}}"##,
            },
            Row {
                error: ApiError::framework(400, None),
                status: 400,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Invalid request."##,
                anthropic_status: 400,
                openai_body: r##"{"error":{"message":"Invalid request.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Invalid request."}}"##,
            },
            Row {
                error: ApiError::framework(406, Some("Detail.")),
                status: 406,
                kind: "invalid_request_error",
                param: None,
                code: None,
                message: r##"Detail."##,
                anthropic_status: 406,
                openai_body: r##"{"error":{"message":"Detail.","type":"invalid_request_error","param":null,"code":null}}"##,
                anthropic_body: r##"{"type":"error","error":{"type":"invalid_request_error","message":"Detail."}}"##,
            },
        ];
        for row in rows {
            let message = row.message;
            assert_eq!(row.error.status, row.status, "{message}");
            assert_eq!(row.error.kind, row.kind, "{message}");
            assert_eq!(row.error.param.as_deref(), row.param, "{message}");
            assert_eq!(row.error.code, row.code, "{message}");
            assert_eq!(row.error.message, message);
            assert_eq!(row.error.status_on(Surface::OpenAi), row.status);
            assert_eq!(
                row.error.status_on(Surface::Anthropic),
                row.anthropic_status,
                "{message}"
            );
            assert_eq!(text(&row.error, Surface::OpenAi), row.openai_body, "{message}");
            assert_eq!(text(&row.error, Surface::Anthropic), row.anthropic_body, "{message}");
        }
    }

    #[test]
    fn other_statuses_get_their_anthropic_types() {
        for (status, kind) in [
            (401, "authentication_error"),
            (403, "permission_error"),
            (500, "api_error"),
            (418, "api_error"),
        ] {
            assert_eq!(ApiError::anthropic_type(status), kind);
        }
    }

    #[test]
    fn messages_paths_are_the_anthropic_surface() {
        assert_eq!(Surface::of_path("/v1/messages"), Surface::Anthropic);
        assert_eq!(Surface::of_path("/v1/messages/count_tokens"), Surface::Anthropic);
        assert_eq!(Surface::of_path("/v1/chat/completions"), Surface::OpenAi);
        assert_eq!(Surface::of_path("/v1/models"), Surface::OpenAi);
    }

    #[test]
    fn quotes_and_control_characters_are_escaped_with_capital_hex() {
        let error = ApiError::invalid_request("a \"b\"\n\u{1b}\u{b}\u{1f}\u{1}\u{8}\u{c}é", None);
        assert_eq!(
            text(&error, Surface::Anthropic),
            r##"{"type":"error","error":{"type":"invalid_request_error","message":"a \"b\"\n\u001B\u000B\u001F\u0001\b\fé"}}"##
        );
        assert_eq!(
            text(&error, Surface::OpenAi),
            r##"{"error":{"message":"a \"b\"\n\u001B\u000B\u001F\u0001\b\fé","type":"invalid_request_error","param":null,"code":null}}"##
        );
    }

    #[test]
    fn a_template_refusal_is_a_bad_request_about_the_messages() {
        for (error, message) in [
            (TemplateError::NoMessages, "No messages provided."),
            (TemplateError::NoUserQuery, "No user query found in messages."),
            (
                TemplateError::SystemNotFirst,
                "System message must be at the beginning.",
            ),
        ] {
            let converted = ApiError::from(error);
            assert_eq!(converted, ApiError::invalid_request(message, Some("messages")));
            assert_eq!((converted.status, converted.code), (400, None));
            assert_eq!(converted.param.as_deref(), Some("messages"));
        }
    }

    #[test]
    fn a_malformed_tool_call_is_an_internal_error_with_the_reason() {
        let failure = PipelineError::MalformedToolCall {
            message: "unknown function 'x'".to_owned(),
            emitted: vec![Piece::Text("kept".to_owned())],
        };
        let converted = ApiError::from(failure);
        assert_eq!(converted, ApiError::invalid_tool_call("unknown function 'x'"));
        assert_eq!(converted.status, 500);
        assert_eq!(converted.code, Some("invalid_tool_call"));
        assert_eq!(
            converted.message,
            "The model produced an invalid tool call: unknown function 'x'."
        );
    }

    #[test]
    fn admission_refusals_are_503_with_the_engine_unavailable_code() {
        for (error, message) in [
            (
                ApiError::at_capacity(),
                "The server is at generation capacity; retry later.",
            ),
            (ApiError::shutting_down(), "The server is shutting down."),
            (
                ApiError::interrupted_by_shutdown(),
                "Generation was interrupted by server shutdown.",
            ),
        ] {
            assert_eq!(error.status, 503);
            assert_eq!(error.kind, "service_unavailable_error");
            assert_eq!(error.code, Some("engine_unavailable"));
            assert_eq!(error.message, message);
        }
    }
}
