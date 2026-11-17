//! The API's errors: one status, type, code and message per failure, and the two JSON bodies that carry them.

use serde::Serialize;

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
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OpenAiErrorEnvelope {
    pub error: OpenAiErrorBody,
}

/// The object inside OpenAI's error envelope.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct OpenAiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// Anthropic's error envelope: `{"type": "error", "error": {"type", "message"}}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AnthropicErrorEnvelope {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub error: AnthropicErrorBody,
}

/// The object inside Anthropic's error envelope, its type named after the status.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub message: String,
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
            kind: "error",
            error: AnthropicErrorBody {
                kind: ApiError::anthropic_type(self.status),
                message: self.message.clone(),
            },
        }
    }

    /// The JSON body of the error on a surface.
    pub fn body(&self, surface: Surface) -> Vec<u8> {
        let bytes = match surface {
            Surface::OpenAi => serde_json::to_vec(&self.openai_envelope()),
            Surface::Anthropic => serde_json::to_vec(&self.anthropic_envelope()),
        };
        // Both envelopes hold only strings and nulls, which always serialise.
        bytes.unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(error: &ApiError, surface: Surface) -> String {
        String::from_utf8(error.body(surface)).expect("utf-8")
    }

    #[test]
    fn every_constructor_has_its_status_type_code_and_bodies() {
        // (error, status, kind, param, code, message, anthropic status, anthropic type)
        type Row = (
            ApiError,
            u16,
            &'static str,
            Option<&'static str>,
            Option<&'static str>,
            &'static str,
            u16,
            &'static str,
        );
        let rows: Vec<Row> = vec![
            (
                ApiError::protocol(405, "Method not allowed."),
                405,
                "invalid_request_error",
                None,
                None,
                "Method not allowed.",
                405,
                "invalid_request_error",
            ),
            (
                ApiError::protocol(415, "Unsupported media type."),
                415,
                "invalid_request_error",
                None,
                None,
                "Unsupported media type.",
                415,
                "invalid_request_error",
            ),
            (
                ApiError::protocol(429, "Slow down."),
                429,
                "invalid_request_error",
                None,
                None,
                "Slow down.",
                429,
                "rate_limit_error",
            ),
            (
                ApiError::unenforceable_schema("bad schema", Some("response_format")),
                400,
                "invalid_request_error",
                Some("response_format"),
                Some("unsupported_schema"),
                "bad schema",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::invalid_request("nope", Some("messages")),
                400,
                "invalid_request_error",
                Some("messages"),
                None,
                "nope",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::invalid_request("nope", None),
                400,
                "invalid_request_error",
                None,
                None,
                "nope",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::request_too_large(),
                413,
                "invalid_request_error",
                None,
                Some("request_too_large"),
                "The chat completion request body exceeds the configured size limit.",
                413,
                "request_too_large",
            ),
            (
                ApiError::unsupported_parameter("n"),
                400,
                "invalid_request_error",
                Some("n"),
                Some("unsupported_parameter"),
                "Unsupported parameter: 'n' is not supported by this server.",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::unrecognized_argument("foo"),
                400,
                "invalid_request_error",
                Some("foo"),
                Some("unrecognized_argument"),
                "Unrecognized request argument supplied: foo",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::context_length_exceeded("too long", Some("messages")),
                400,
                "invalid_request_error",
                Some("messages"),
                Some("context_length_exceeded"),
                "too long",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::model_not_found("gpt-x"),
                404,
                "invalid_request_error",
                Some("model"),
                Some("model_not_found"),
                "The model 'gpt-x' does not exist or you do not have access to it.",
                404,
                "not_found_error",
            ),
            (
                ApiError::not_found("gone"),
                404,
                "invalid_request_error",
                None,
                None,
                "gone",
                404,
                "not_found_error",
            ),
            (
                ApiError::unavailable("busy"),
                503,
                "service_unavailable_error",
                None,
                Some("engine_unavailable"),
                "busy",
                529,
                "overloaded_error",
            ),
            (
                ApiError::server_error(),
                500,
                "server_error",
                None,
                None,
                "The server had an error while processing your request.",
                500,
                "api_error",
            ),
            (
                ApiError::invalid_tool_call("unknown tool"),
                500,
                "server_error",
                None,
                Some("invalid_tool_call"),
                "The model produced an invalid tool call: unknown tool.",
                500,
                "api_error",
            ),
            (
                ApiError::invalid_structured_output(),
                500,
                "server_error",
                None,
                Some("invalid_structured_output"),
                "The model's output is not valid JSON for the requested response format.",
                500,
                "api_error",
            ),
            (
                ApiError::client_closed(),
                503,
                "service_unavailable_error",
                None,
                Some("client_closed"),
                "The client closed the connection.",
                529,
                "overloaded_error",
            ),
            (
                ApiError::timeout(),
                503,
                "service_unavailable_error",
                None,
                Some("timeout"),
                "The request timed out before generation completed.",
                529,
                "overloaded_error",
            ),
            (
                ApiError::engine_unavailable("starting"),
                503,
                "service_unavailable_error",
                None,
                Some("engine_unavailable"),
                "The inference engine is unavailable: starting",
                529,
                "overloaded_error",
            ),
            (
                ApiError::unparsable_body(),
                400,
                "invalid_request_error",
                None,
                None,
                "We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type.",
                400,
                "invalid_request_error",
            ),
            (
                ApiError::unknown_url("GET", "/v1/nothing"),
                404,
                "invalid_request_error",
                None,
                None,
                "Unknown request URL: GET /v1/nothing.",
                404,
                "not_found_error",
            ),
        ];
        for (error, status, kind, param, code, message, anthropic_status, anthropic_type) in rows {
            assert_eq!(error.status, status, "{message}");
            assert_eq!(error.kind, kind, "{message}");
            assert_eq!(error.param.as_deref(), param, "{message}");
            assert_eq!(error.code, code, "{message}");
            assert_eq!(error.message, message);
            assert_eq!(error.status_on(Surface::OpenAi), status);
            assert_eq!(error.status_on(Surface::Anthropic), anthropic_status, "{message}");
            let openai = format!(
                "{{\"error\":{{\"message\":{},\"type\":{},\"param\":{},\"code\":{}}}}}",
                serde_json::to_string(message).unwrap(),
                serde_json::to_string(kind).unwrap(),
                serde_json::to_string(&param).unwrap(),
                serde_json::to_string(&code).unwrap(),
            );
            assert_eq!(text(&error, Surface::OpenAi), openai);
            let anthropic = format!(
                "{{\"type\":\"error\",\"error\":{{\"type\":\"{anthropic_type}\",\"message\":{}}}}}",
                serde_json::to_string(message).unwrap(),
            );
            assert_eq!(text(&error, Surface::Anthropic), anthropic);
        }
    }

    #[test]
    fn a_full_openai_body_has_the_documented_bytes() {
        assert_eq!(
            text(&ApiError::model_not_found("m"), Surface::OpenAi),
            r#"{"error":{"message":"The model 'm' does not exist or you do not have access to it.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#
        );
        assert_eq!(
            text(&ApiError::server_error(), Surface::OpenAi),
            r#"{"error":{"message":"The server had an error while processing your request.","type":"server_error","param":null,"code":null}}"#
        );
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
    fn quotes_and_control_characters_are_escaped() {
        let error = ApiError::invalid_request("a \"b\"\n\u{1}é", None);
        assert_eq!(
            text(&error, Surface::Anthropic),
            "{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"a \\\"b\\\"\\n\\u0001é\"}}"
        );
    }
}
