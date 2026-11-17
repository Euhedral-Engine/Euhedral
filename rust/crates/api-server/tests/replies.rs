//! The JSON replies of the generating routes.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits with a bound"
)]

mod common;

use common::{Fixture, text};
use euhedral_inference_api_server::scripted::Script;

fn limits() -> euhedral_inference_api_server::Limits {
    let mut limits = common::small_limits();
    limits.max_body_bytes = 1 << 20;
    limits
}

fn tokens(chunks: &[&str], stop_token: bool) -> Script {
    Script::Tokens {
        chunks: chunks.iter().map(|chunk| (*chunk).to_owned()).collect(),
        stop_token,
    }
}

#[test]
fn dump() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let r = fx.post(
        "/v1/chat/completions",
        Some("application/json"),
        r#"{"model":"euhedral-test-model","reasoning_effort":"none","messages":[{"role":"user","content":"Hi"}]}"#,
    );
    println!("CHAT {} {:?} {}", r.status, r.headers, text(&r));
    let r = fx.post(
        "/v1/messages",
        Some("application/json"),
        r#"{"model":"euhedral-test-model","max_tokens":16,"messages":[{"role":"user","content":"Hi"}]}"#,
    );
    println!("MSG {} {}", r.status, text(&r));
    let r = fx.post(
        "/v1/responses",
        Some("application/json"),
        r#"{"model":"euhedral-test-model","input":"Hi"}"#,
    );
    println!("RESP {} {}", r.status, text(&r));
    fx.finish();
}
