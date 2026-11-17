//! `GET /health`: liveness and engine readiness, outside `/v1`.

use euhedral_inference_api_protocol::json::JsonValue;

/// The status and the JSON body: `up` and `ready` with 200 while the engine takes work, `up` and `closed` with 503 once
/// shutdown began. Only the public model name is reported.
pub fn report(available: bool, model: &str) -> (u16, String) {
    let (status, engine) = if available { (200, "ready") } else { (503, "closed") };
    let member = |key: &str, value: &str| (key.to_owned(), JsonValue::String(value.to_owned()));
    let body = JsonValue::Object(vec![
        member("status", "up"),
        member("engine", engine),
        member("model", model),
    ]);
    (status, body.to_compact_string())
}
