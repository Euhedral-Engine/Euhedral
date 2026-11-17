//! The routes that need no generation, the errors, the limits and the stop, over loopback.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use common::{Fixture, assert_json, headers, text};
use euhedral_net::Method;

const MODEL: &str = "euhedral-test-model";
const JSON: Option<&str> = Some("application/json");

const OPENAI_LIST: &str = r#"{"object":"list","data":[{"id":"euhedral-test-model","object":"model","created":1715935800,"owned_by":"euhedral"}]}"#;
const OPENAI_MODEL: &str =
    r#"{"id":"euhedral-test-model","object":"model","created":1715935800,"owned_by":"euhedral"}"#;
const ANTHROPIC_MODEL: &str = r#"{"type":"model","id":"euhedral-test-model","display_name":"euhedral-test-model","created_at":"2024-05-17T08:50:00Z"}"#;

#[test]
fn health_reports_the_engine_and_the_model() {
    let fixture = Fixture::start();
    let up = r#"{"status":"up","engine":"ready","model":"euhedral-test-model"}"#;
    assert_json(&fixture.get("/health"), 200, up);
    fixture.backend.set_available(false);
    let closed = r#"{"status":"up","engine":"closed","model":"euhedral-test-model"}"#;
    assert_json(&fixture.get("/health"), 503, closed);
    fixture.finish();
}

#[test]
fn health_answers_head_with_the_head_only_and_refuses_an_accept_without_json() {
    let fixture = Fixture::start();
    let head = fixture.call(&Fixture::req(Method::Head, "/health"));
    assert_eq!(head.status, 200);
    assert!(head.body.is_empty());
    let accepts_json = Fixture::req(Method::Get, "/health").header("Accept", "text/html, application/json;q=0.5");
    assert_eq!(fixture.call(&accepts_json).status, 200);
    let not_acceptable = Fixture::req(Method::Get, "/health").header("Accept", "text/plain");
    let body = r#"{"error":{"message":"Acceptable representations: [application/json].","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.call(&not_acceptable), 406, body);
    let unreadable = Fixture::req(Method::Get, "/health").header("Accept", "nonsense");
    let body = r#"{"error":{"message":"Could not parse Accept header.","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.call(&unreadable), 406, body);
    fixture.finish();
}

#[test]
fn models_are_listed_in_openai_form() {
    let fixture = Fixture::start();
    assert_json(&fixture.get("/v1/models"), 200, OPENAI_LIST);
    assert_json(&fixture.get(&format!("/v1/models/{MODEL}")), 200, OPENAI_MODEL);
    assert_json(&fixture.get("/v1/models?limit=3"), 200, OPENAI_LIST);
    fixture.finish();
}

#[test]
fn models_are_listed_in_anthropic_form_for_a_client_that_sends_its_version() {
    let fixture = Fixture::start();
    let list = format!(r#"{{"data":[{ANTHROPIC_MODEL}],"has_more":false,"first_id":"{MODEL}","last_id":"{MODEL}"}}"#);
    let get = |target: &str| Fixture::req(Method::Get, target).header("anthropic-version", "2023-06-01");
    assert_json(&fixture.call(&get("/v1/models")), 200, &list);
    assert_json(
        &fixture.call(&get(&format!("/v1/models/{MODEL}"))),
        200,
        ANTHROPIC_MODEL,
    );
    fixture.finish();
}

#[test]
fn an_unknown_model_is_not_found_in_openai_form_whichever_client_asks() {
    let fixture = Fixture::start();
    let body = r#"{"error":{"message":"The model 'other' does not exist or you do not have access to it.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#;
    assert_json(&fixture.get("/v1/models/other"), 404, body);
    let anthropic = Fixture::req(Method::Get, "/v1/models/other").header("anthropic-version", "2023-06-01");
    assert_json(&fixture.call(&anthropic), 404, body);
    let escaped = r#"{"error":{"message":"The model 'a b' does not exist or you do not have access to it.","type":"invalid_request_error","param":"model","code":"model_not_found"}}"#;
    assert_json(&fixture.get("/v1/models/a%20b"), 404, escaped);
    fixture.finish();
}

#[test]
fn an_unknown_url_is_not_found_in_the_format_of_its_surface() {
    let fixture = Fixture::start();
    let openai = r#"{"error":{"message":"Unknown request URL: GET /v1/nothing.","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.get("/v1/nothing"), 404, openai);
    let anthropic = r#"{"type":"error","error":{"type":"not_found_error","message":"Unknown request URL: GET /v1/messages/nothing."}}"#;
    assert_json(&fixture.get("/v1/messages/nothing"), 404, anthropic);
    let root = r#"{"error":{"message":"Unknown request URL: GET /.","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.get("/"), 404, root);
    // A path with a trailing slash or a deeper model path is not a route.
    assert_eq!(fixture.get("/v1/models/").status, 404);
    assert_eq!(fixture.get("/v1/models/a/b").status, 404);
    assert_eq!(fixture.get("/health/").status, 404);
    fixture.finish();
}

#[test]
fn a_method_the_url_does_not_take_is_405_with_the_error_of_its_surface() {
    let fixture = Fixture::start();
    let openai = |method: &str| {
        format!(
            r#"{{"error":{{"message":"Method '{method}' is not supported.","type":"invalid_request_error","param":null,"code":null}}}}"#
        )
    };
    assert_json(
        &fixture.call(&Fixture::req(Method::Delete, "/v1/models")),
        405,
        &openai("DELETE"),
    );
    assert_json(
        &fixture.call(&Fixture::req(Method::Post, "/health")),
        405,
        &openai("POST"),
    );
    assert_json(&fixture.get("/v1/chat/completions"), 405, &openai("GET"));
    assert_json(&fixture.get("/v1/responses"), 405, &openai("GET"));
    let anthropic =
        r#"{"type":"error","error":{"type":"invalid_request_error","message":"Method 'GET' is not supported."}}"#;
    assert_json(&fixture.get("/v1/messages"), 405, anthropic);
    assert_json(&fixture.get("/v1/messages/count_tokens"), 405, anthropic);
    // Any other method on an unknown URL is also 405.
    assert_json(
        &fixture.call(&Fixture::req(Method::Post, "/nothing")),
        405,
        &openai("POST"),
    );
    fixture.finish();
}

#[test]
fn options_answers_with_the_methods_of_the_url() {
    let fixture = Fixture::start();
    let allow = |target: &str| {
        let response = fixture.call(&Fixture::req(Method::Options, target));
        assert_eq!(response.status, 200);
        assert!(response.body.is_empty());
        response.headers.get("allow").map(str::to_owned)
    };
    assert_eq!(allow("/v1/models").as_deref(), Some("GET, HEAD, OPTIONS"));
    assert_eq!(allow("/v1/chat/completions").as_deref(), Some("POST, OPTIONS"));
    assert_eq!(allow("/nothing").as_deref(), Some("GET,HEAD,OPTIONS"));
    fixture.finish();
}

#[test]
fn a_generation_route_wants_json() {
    let fixture = Fixture::start();
    let wrong = |content_type: &str| {
        format!(
            r#"{{"error":{{"message":"Content-Type '{content_type}' is not supported.","type":"invalid_request_error","param":null,"code":null}}}}"#
        )
    };
    let route = "/v1/chat/completions";
    assert_json(
        &fixture.post(route, Some("text/plain"), "hi"),
        415,
        &wrong("text/plain"),
    );
    assert_json(
        &fixture.post(route, Some("Text/Plain; charset=UTF-8"), "hi"),
        415,
        &wrong("text/plain;charset=UTF-8"),
    );
    assert_json(
        &fixture.post("/v1/responses", None, "{}"),
        415,
        &wrong("application/octet-stream"),
    );
    let unreadable = r#"{"error":{"message":"Could not parse Content-Type.","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.post(route, Some("garbage"), "{}"), 415, unreadable);
    let anthropic = r#"{"type":"error","error":{"type":"invalid_request_error","message":"Content-Type 'text/plain' is not supported."}}"#;
    assert_json(&fixture.post("/v1/messages", Some("text/plain"), "hi"), 415, anthropic);
    assert_json(
        &fixture.post("/v1/messages/count_tokens", Some("text/plain"), "hi"),
        415,
        anthropic,
    );
    fixture.finish();
}

const OPENAI_TOO_LARGE: &str = r#"{"error":{"message":"The chat completion request body exceeds the configured size limit.","type":"invalid_request_error","param":null,"code":"request_too_large"}}"#;
const ANTHROPIC_TOO_LARGE: &str = r#"{"type":"error","error":{"type":"request_too_large","message":"The chat completion request body exceeds the configured size limit."}}"#;

#[test]
fn a_body_over_the_limit_is_413_with_the_json_error() {
    let fixture = Fixture::start();
    // The limit is 64 bytes; the transport reads 1024 more so that it can be answered.
    for length in [65, 500, 64 + 1024] {
        let body = "x".repeat(length);
        assert_json(
            &fixture.post("/v1/chat/completions", JSON, &body),
            413,
            OPENAI_TOO_LARGE,
        );
        assert_json(&fixture.post("/v1/responses", JSON, &body), 413, OPENAI_TOO_LARGE);
        assert_json(&fixture.post("/v1/messages", JSON, &body), 413, ANTHROPIC_TOO_LARGE);
        assert_json(
            &fixture.post("/v1/messages/count_tokens", JSON, &body),
            413,
            ANTHROPIC_TOO_LARGE,
        );
    }
    // A body at the limit passes it.
    let at_limit = "x".repeat(64);
    assert_eq!(fixture.post("/v1/chat/completions", JSON, &at_limit).status, 501);
    fixture.finish();
}

#[test]
fn a_body_far_over_the_limit_is_refused_by_the_transport_with_no_body() {
    let fixture = Fixture::start();
    let response = fixture.post("/v1/chat/completions", JSON, &"x".repeat(64 + 1024 + 1));
    assert_eq!(response.status, 413);
    assert!(response.body.is_empty());
    fixture.finish();
}

#[test]
fn the_content_type_is_checked_before_the_size() {
    let fixture = Fixture::start();
    let response = fixture.post("/v1/chat/completions", Some("text/plain"), &"x".repeat(500));
    assert_eq!(response.status, 415);
    fixture.finish();
}

#[test]
fn without_a_ready_backend_each_generation_route_is_unavailable_in_its_surface_format() {
    let fixture = Fixture::start();
    fixture.backend.set_available(false);
    let openai = r#"{"error":{"message":"The inference engine is unavailable: inference engine is shutting down","type":"service_unavailable_error","param":null,"code":"engine_unavailable"}}"#;
    let anthropic = r#"{"type":"error","error":{"type":"overloaded_error","message":"The inference engine is unavailable: inference engine is shutting down"}}"#;
    assert_json(&fixture.post("/v1/chat/completions", JSON, "{}"), 503, openai);
    assert_json(&fixture.post("/v1/responses", JSON, "{}"), 503, openai);
    assert_json(&fixture.post("/v1/messages", JSON, "{}"), 529, anthropic);
    assert_json(&fixture.post("/v1/messages/count_tokens", JSON, "{}"), 529, anthropic);
    // The model lists do not need the engine.
    assert_json(&fixture.get("/v1/models"), 200, OPENAI_LIST);
    fixture.finish();
}

#[test]
fn with_a_ready_backend_each_generation_route_answers_the_placeholder() {
    let fixture = Fixture::start();
    let openai = r#"{"error":{"message":"This route is not implemented yet.","type":"server_error","param":null,"code":"not_implemented"}}"#;
    let anthropic = r#"{"type":"error","error":{"type":"api_error","message":"This route is not implemented yet."}}"#;
    assert_json(
        &fixture.post("/v1/chat/completions", Some("application/json; charset=utf-8"), "{}"),
        501,
        openai,
    );
    assert_json(&fixture.post("/v1/responses", JSON, "{}"), 501, openai);
    assert_json(&fixture.post("/v1/messages", JSON, "{}"), 501, anthropic);
    assert_json(&fixture.post("/v1/messages/count_tokens", JSON, "{}"), 501, anthropic);
    fixture.finish();
}

#[test]
fn an_error_has_a_content_type_and_nothing_else_of_its_own() {
    let fixture = Fixture::start();
    let response = fixture.get("/v1/nothing");
    assert_eq!(headers(&response).len(), 2, "{:?}", headers(&response));
    assert!(text(&response).starts_with("{\"error\""));
    fixture.finish();
}

#[test]
fn stop_ends_the_server_and_refuses_new_connections() {
    let fixture = Fixture::start();
    assert_eq!(fixture.get("/health").status, 200);
    fixture.stop();
    assert!(fixture.try_call(&Fixture::req(Method::Get, "/health")).is_err());
    assert_eq!(fixture.server.counters().open, 0);
    fixture.finish_after_stop();
}
