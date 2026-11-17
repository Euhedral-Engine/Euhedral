//! The chain from a request body to a planned generation, through its outcome, and the routes that start it.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, PoisonError};

use common::{Fixture, PATIENCE, assert_json, lattice, small_limits};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_server::scripted::{MODEL_ID, ScriptedBackend};
use euhedral_inference_api_server::{Endpoint, Extras, Limits, Planned, Planner, Publisher, RequestDefaults};
use euhedral_net::Method;

const JSON: Option<&str> = Some("application/json");
const HI: &str = r#"[{"role":"user","content":"Hi"}]"#;
/// What the scripted backend counts for `HI` rendered with thinking off: one token per character.
const HI_PROMPT: &str = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const UNPARSABLE: &str = "We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type.";

/// Offers frames to a sink as the server does, and records what it was given. After `allowed` frames it refuses, as a
/// lattice that is closing does.
struct Recording {
    sink: Arc<QueueIngestSink>,
    published: AtomicUsize,
    allowed: usize,
    hashes: Mutex<Vec<u64>>,
}

impl Publisher for Recording {
    fn publish(&self, frame: Box<dyn Frame>) {
        self.hashes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(frame.header().routing_hash());
        if self.published.fetch_add(1, Ordering::AcqRel) < self.allowed {
            drop(self.sink.offer(frame));
        }
    }
}

struct Rig {
    lattice: Arc<ControlPlaneLattice>,
    backend: Arc<ScriptedBackend>,
    publisher: Arc<Recording>,
    planner: Planner,
}

impl Rig {
    fn new() -> Rig {
        Rig::with(RequestDefaults::default(), usize::MAX)
    }

    fn with(defaults: RequestDefaults, allowed: usize) -> Rig {
        let lattice = lattice(Some(2));
        let backend = ScriptedBackend::on(&lattice);
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(sink.delegate()).expect("sink attached");
        let publisher = Arc::new(Recording {
            sink,
            published: AtomicUsize::new(0),
            allowed,
            hashes: Mutex::new(Vec::new()),
        });
        let planner = Planner::new(backend.clone(), defaults, publisher.clone());
        Rig {
            lattice,
            backend,
            publisher,
            planner,
        }
    }

    /// Plans `body`; the frames published for it are counted from here on.
    fn plan(&self, endpoint: Endpoint, body: &str) -> Result<Planned, ApiError> {
        self.publisher.published.store(0, Ordering::Release);
        self.publisher
            .hashes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        let (tx, rx) = channel();
        let first = self.planner.plan(
            endpoint,
            body.as_bytes().to_vec(),
            Box::new(move |outcome| {
                tx.send(outcome).ok();
            }),
        );
        self.publisher.publish(first);
        rx.recv_timeout(PATIENCE).expect("the chain ended")
    }

    fn count(&self, body: &[u8]) -> Result<usize, ApiError> {
        self.publisher.published.store(0, Ordering::Release);
        let (tx, rx) = channel();
        let first = self.planner.count_tokens(
            body.to_vec(),
            Box::new(move |outcome| {
                tx.send(outcome).ok();
            }),
        );
        self.publisher.publish(first);
        rx.recv_timeout(PATIENCE).expect("the chain ended")
    }

    /// How many frames the last chain published, its first included.
    fn frames(&self) -> usize {
        self.publisher.published.load(Ordering::Acquire)
    }

    fn finish(self) {
        self.backend.release();
        self.publisher.sink.complete_gracefully();
        self.lattice.close();
    }
}

fn chat(extra: &str) -> String {
    format!(r#"{{"model":"{MODEL_ID}","messages":{HI}{extra}}}"#)
}

fn messages(extra: &str) -> String {
    format!(r#"{{"model":"{MODEL_ID}","max_tokens":8,"messages":{HI}{extra}}}"#)
}

fn expect_error(result: Result<Planned, ApiError>) -> ApiError {
    result.expect_err("the chain should end in an error")
}

#[test]
fn a_chat_request_runs_as_five_frames_and_ends_in_a_plan() {
    let rig = Rig::new();
    let planned = rig
        .plan(Endpoint::ChatCompletions, &chat(r#","reasoning_effort":"none""#))
        .expect("a plan");
    // The first frame, the map, the grammar, the template and the plan: one frame each, each with its own hash. The
    // encoding is the backend's frame.
    assert_eq!(rig.frames(), 5);
    let hashes: HashSet<u64> = rig
        .publisher
        .hashes
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .iter()
        .copied()
        .collect();
    assert_eq!(hashes.len(), 5);
    assert_eq!(rig.backend.encoded(), 1);
    assert_eq!(planned.endpoint, Endpoint::ChatCompletions);
    assert_eq!(planned.extras, Extras::Chat { include_usage: false });
    assert_eq!(planned.prompt.text, HI_PROMPT);
    assert_eq!(planned.prompt.token_count(), HI_PROMPT.chars().count());
    assert_eq!(planned.id.len(), 32);
    assert!(planned.id.bytes().all(|byte| byte.is_ascii_hexdigit()));
    assert_eq!(planned.max_new_tokens, 4096 - HI_PROMPT.chars().count());
    assert!(!planned.output.reasoning);
    assert_eq!(planned.output.grammar, None);
    assert!(rig.backend.generations().is_empty(), "planning opens no generation");
    rig.finish();
}

#[test]
fn the_default_thinking_opens_the_think_block_and_the_output_reasons() {
    let rig = Rig::new();
    let planned = rig.plan(Endpoint::ChatCompletions, &chat("")).expect("a plan");
    assert!(planned.output.reasoning);
    assert!(planned.prompt.text.ends_with("<think>\n"), "{}", planned.prompt.text);
    rig.finish();
}

#[test]
fn a_messages_request_plans_with_its_own_extras() {
    let rig = Rig::new();
    let planned = rig.plan(Endpoint::Messages, &messages("")).expect("a plan");
    assert_eq!(
        (planned.endpoint, planned.extras.clone()),
        (Endpoint::Messages, Extras::Messages)
    );
    assert_eq!(planned.max_new_tokens, 8);
    assert_eq!(rig.frames(), 5);
    rig.finish();
}

#[test]
fn a_responses_request_carries_what_its_reply_repeats() {
    let rig = Rig::new();
    let body = format!(r#"{{"model":"{MODEL_ID}","input":"Hi","include":["reasoning.encrypted_content"]}}"#);
    let planned = rig.plan(Endpoint::Responses, &body).expect("a plan");
    assert_eq!(planned.endpoint, Endpoint::Responses);
    let Extras::Responses {
        include_encrypted_content,
        echo,
    } = planned.extras
    else {
        panic!("the extras of a Responses request");
    };
    assert!(include_encrypted_content);
    assert!(matches!(echo, JsonValue::Object(_)));
    rig.finish();
}

#[test]
fn the_default_max_tokens_is_the_setting_capped_by_what_the_context_leaves() {
    let defaults = RequestDefaults {
        max_tokens: 50,
        ..RequestDefaults::default()
    };
    let rig = Rig::with(defaults, usize::MAX);
    let body = chat(r#","reasoning_effort":"none""#);
    let prompt = HI_PROMPT.chars().count();
    assert_eq!(
        rig.plan(Endpoint::ChatCompletions, &body)
            .expect("a plan")
            .max_new_tokens,
        50
    );
    // A context with fewer than the default left: the rest of the context.
    rig.backend.set_context_length(prompt + 7);
    assert_eq!(
        rig.plan(Endpoint::ChatCompletions, &body)
            .expect("a plan")
            .max_new_tokens,
        7
    );
    // A request that states its budget has it, if it fits.
    rig.backend.set_context_length(prompt + 3);
    let stated = chat(r#","reasoning_effort":"none","max_completion_tokens":3"#);
    assert_eq!(
        rig.plan(Endpoint::ChatCompletions, &stated)
            .expect("a plan")
            .max_new_tokens,
        3
    );
    rig.finish();
}

#[test]
fn a_budget_over_the_context_is_refused_with_the_count_in_the_message() {
    let rig = Rig::new();
    let prompt = HI_PROMPT.chars().count();
    rig.backend.set_context_length(80);
    let error = expect_error(rig.plan(
        Endpoint::ChatCompletions,
        &chat(r#","reasoning_effort":"none","max_tokens":50"#),
    ));
    assert_eq!(
        error,
        ApiError::context_length_exceeded(
            format!(
                "This model's maximum context length is 80 tokens. However, you requested {} tokens ({prompt} in the messages, 50 in the completion).",
                prompt + 50
            ),
            Some("max_tokens"),
        )
    );
    assert_eq!((error.status, error.code), (400, Some("context_length_exceeded")));
    // The parameter is the one the request used.
    let error = expect_error(rig.plan(
        Endpoint::ChatCompletions,
        &chat(r#","reasoning_effort":"none","max_completion_tokens":50"#),
    ));
    assert_eq!(error.param.as_deref(), Some("max_completion_tokens"));
    let error = expect_error(rig.plan(Endpoint::Messages, &messages("").replace(":8,", ":50,")));
    assert_eq!(error.param.as_deref(), Some("max_tokens"));
    assert_eq!(
        rig.frames(),
        5,
        "the budget is checked by the last frame, once the prompt's tokens are known"
    );
    // One token too many, and exactly the room that is left.
    rig.backend.set_context_length(prompt + 5);
    let fits = chat(r#","reasoning_effort":"none","max_tokens":5"#);
    assert_eq!(
        rig.plan(Endpoint::ChatCompletions, &fits)
            .expect("a plan")
            .max_new_tokens,
        5
    );
    let over = chat(r#","reasoning_effort":"none","max_tokens":6"#);
    assert!(
        expect_error(rig.plan(Endpoint::ChatCompletions, &over))
            .message
            .contains("you requested")
    );
    rig.finish();
}

#[test]
fn a_prompt_that_leaves_no_room_is_refused_about_the_messages() {
    let rig = Rig::new();
    let prompt = HI_PROMPT.chars().count();
    rig.backend.set_context_length(prompt);
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat(r#","reasoning_effort":"none""#)));
    assert_eq!(
        error,
        ApiError::context_length_exceeded(
            format!("This model's maximum context length is {prompt} tokens, but the messages use {prompt} tokens."),
            Some("messages"),
        )
    );
    rig.finish();
}

#[test]
fn a_body_that_is_not_json_ends_the_chain_at_its_first_frame() {
    let rig = Rig::new();
    for (endpoint, body) in [
        (Endpoint::ChatCompletions, "{\"model\":"),
        (Endpoint::Messages, "not json"),
        (Endpoint::Responses, "[1]"),
    ] {
        let error = expect_error(rig.plan(endpoint, body));
        assert_eq!(error, ApiError::invalid_request(UNPARSABLE, None));
        assert_eq!(rig.frames(), 1);
    }
    let error = expect_error(rig.planner_plan_bytes(&[0xff, 0xfe]));
    assert_eq!(error, ApiError::unparsable_body());
    rig.finish();
}

impl Rig {
    fn planner_plan_bytes(&self, body: &[u8]) -> Result<Planned, ApiError> {
        let (tx, rx) = channel();
        let first = self.planner.plan(
            Endpoint::ChatCompletions,
            body.to_vec(),
            Box::new(move |outcome| {
                tx.send(outcome).ok();
            }),
        );
        self.publisher.publish(first);
        rx.recv_timeout(PATIENCE).expect("the chain ended")
    }
}

#[test]
fn the_mapping_ends_the_chain_at_its_frame() {
    let rig = Rig::new();
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &format!(r#"{{"messages":{HI}}}"#)));
    assert_eq!(error.param.as_deref(), Some("model"));
    assert_eq!(rig.frames(), 2);
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat("").replace(MODEL_ID, "gpt-4o")));
    assert_eq!(error, ApiError::model_not_found("gpt-4o"));
    let error = expect_error(rig.plan(Endpoint::Messages, &messages("").replace(r#""max_tokens":8,"#, "")));
    assert_eq!(
        error,
        ApiError::invalid_request("max_tokens: Field required", Some("max_tokens"))
    );
    assert_eq!(rig.frames(), 2);
    rig.finish();
}

#[test]
fn a_refused_grammar_or_schema_ends_the_chain_at_the_grammar_frame() {
    let rig = Rig::new();
    let schema = r#","reasoning_effort":"none","response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"#;
    // A JSON answer is constrained by a grammar, which the backend compiled.
    let planned = rig.plan(Endpoint::ChatCompletions, &chat(schema)).expect("a plan");
    let grammar = planned.output.grammar.expect("a grammar");
    assert!(
        grammar.starts_with("start: answer\nanswer: %json {\"x-guidance\":"),
        "{grammar}"
    );
    assert_eq!(rig.frames(), 5);
    let object = r#","reasoning_effort":"none","response_format":{"type":"json_object"}"#;
    assert!(
        rig.plan(Endpoint::ChatCompletions, &chat(object))
            .expect("a plan")
            .output
            .grammar
            .is_some()
    );

    rig.backend.refuse_schemas(Some("no if"));
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat(schema)));
    assert_eq!(error.status, 400);
    assert_eq!(error.message, "The JSON Schema cannot be enforced: no if");
    assert_eq!(error.param.as_deref(), Some("response_format.json_schema.schema"));
    assert_eq!(rig.frames(), 3);

    rig.backend.refuse_schemas(None);
    rig.backend.refuse_grammars(Some("too large"));
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat(object)));
    assert_eq!(
        error,
        ApiError::invalid_request("The requested output cannot be enforced: too large", None)
    );
    assert_eq!(rig.frames(), 3);
    // Free text needs no grammar, so nothing is refused.
    assert!(rig.plan(Endpoint::ChatCompletions, &chat("")).is_ok());
    rig.finish();
}

#[test]
fn a_conversation_the_template_refuses_ends_the_chain_at_the_template_frame() {
    let rig = Rig::new();
    let body = format!(r#"{{"model":"{MODEL_ID}","messages":[{{"role":"system","content":"s"}}]}}"#);
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &body));
    assert_eq!(
        error,
        ApiError::invalid_request("No user query found in messages.", Some("messages"))
    );
    assert_eq!(rig.frames(), 4);
    assert_eq!(rig.backend.encoded(), 0, "a refused conversation is never encoded");
    rig.finish();
}

#[test]
fn an_engine_that_cannot_encode_ends_the_chain_as_unavailable() {
    let rig = Rig::new();
    rig.backend.refuse_work();
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat("")));
    assert_eq!(error, ApiError::engine_unavailable("the lattice refused the work"));
    assert_eq!(error.status, 503);
    let counted = rig.count(messages("").replace(r#""max_tokens":8,"#, "").as_bytes());
    assert_eq!(
        counted,
        Err(ApiError::engine_unavailable("the lattice refused the work"))
    );
    rig.finish();
}

#[test]
fn a_frame_the_lattice_refuses_answers_that_the_server_is_shutting_down() {
    // The first frame and the map are accepted; the grammar frame is refused.
    let rig = Rig::with(RequestDefaults::default(), 2);
    let error = expect_error(rig.plan(Endpoint::ChatCompletions, &chat("")));
    assert_eq!(error, ApiError::shutting_down());
    // The very first frame refused.
    let rig_none = Rig::with(RequestDefaults::default(), 0);
    let error = expect_error(rig_none.plan(Endpoint::ChatCompletions, &chat("")));
    assert_eq!(error, ApiError::shutting_down());
    let counted = rig_none.count(b"{}");
    assert_eq!(counted, Err(ApiError::shutting_down()));
    rig.finish();
    rig_none.finish();
}

#[test]
fn sampling_reaches_the_plan_and_a_missing_seed_is_drawn() {
    let rig = Rig::new();
    let base = r#","reasoning_effort":"none""#;
    let plain = rig.plan(Endpoint::ChatCompletions, &chat(base)).expect("a plan");
    assert_eq!(
        (plain.config.temperature, plain.config.top_k, plain.config.top_p),
        (1.0, 20, 0.95)
    );
    assert!(!plain.config.greedy);
    let again = rig.plan(Endpoint::ChatCompletions, &chat(base)).expect("a plan");
    assert_ne!(plain.config.seed, again.config.seed);
    assert_ne!(plain.id, again.id);
    let stated = format!(r#"{base},"temperature":0.7,"top_p":0.5,"top_k":5,"seed":42"#);
    let stated = rig.plan(Endpoint::ChatCompletions, &chat(&stated)).expect("a plan");
    assert_eq!(
        (
            stated.config.temperature,
            stated.config.top_p,
            stated.config.top_k,
            stated.config.seed
        ),
        (0.7, 0.5, 5, 42)
    );
    let off = rig
        .plan(Endpoint::ChatCompletions, &chat(&format!(r#"{base},"top_k":-1"#)))
        .expect("a plan");
    assert_eq!(off.config.top_k, 0);
    let greedy = rig
        .plan(Endpoint::ChatCompletions, &chat(&format!(r#"{base},"temperature":0"#)))
        .expect("a plan");
    assert!(greedy.config.greedy);
    let defaults = RequestDefaults {
        temperature: 0.5,
        top_k: 3,
        top_p: 0.9,
        max_tokens: 10,
    };
    let rig_set = Rig::with(defaults, usize::MAX);
    let planned = rig_set.plan(Endpoint::ChatCompletions, &chat(base)).expect("a plan");
    assert_eq!(
        (planned.config.temperature, planned.config.top_k, planned.config.top_p),
        (0.5, 3, 0.9)
    );
    rig.finish();
    rig_set.finish();
}

#[test]
fn count_tokens_runs_the_chain_without_the_grammar_and_the_plan() {
    let rig = Rig::new();
    let body = format!(r#"{{"model":"{MODEL_ID}","messages":{HI}}}"#);
    assert_eq!(rig.count(body.as_bytes()), Ok(HI_PROMPT.chars().count()));
    // The first frame, the map and the template; the count is the backend's.
    assert_eq!(rig.frames(), 3);
    assert!(rig.backend.generations().is_empty());
    // Count takes no max_tokens and no stream, and the context is not checked.
    rig.backend.set_context_length(1);
    assert_eq!(rig.count(body.as_bytes()), Ok(HI_PROMPT.chars().count()));
    let error = rig.count(messages("").as_bytes()).expect_err("max_tokens");
    assert_eq!(error, ApiError::unrecognized_argument("max_tokens"));
    assert_eq!(rig.frames(), 2);
    assert_eq!(rig.count(b"nope"), Err(ApiError::unparsable_body()));
    assert_eq!(rig.frames(), 1);
    rig.finish();
}

const OPENAI_UNPARSABLE: &str = r#"{"error":{"message":"We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type.","type":"invalid_request_error","param":null,"code":null}}"#;
const ANTHROPIC_UNPARSABLE: &str = r#"{"type":"error","error":{"type":"invalid_request_error","message":"We could not parse the JSON body of your request. Check that it is valid JSON and that each field has the documented type."}}"#;

fn roomy() -> Fixture {
    let limits = Limits {
        max_body_bytes: 4096,
        ..small_limits()
    };
    Fixture::with(limits, Some(2))
}

#[test]
fn a_malformed_body_is_answered_in_the_format_of_each_surface() {
    let fixture = roomy();
    for path in ["/v1/chat/completions", "/v1/responses"] {
        assert_json(&fixture.post(path, JSON, "{\"model\":"), 400, OPENAI_UNPARSABLE);
    }
    for path in ["/v1/messages", "/v1/messages/count_tokens"] {
        assert_json(&fixture.post(path, JSON, "{\"model\":"), 400, ANTHROPIC_UNPARSABLE);
    }
    // A body that is not UTF-8 is as unreadable.
    let bytes = Fixture::req(Method::Post, "/v1/chat/completions")
        .header("Content-Type", "application/json")
        .with_body(vec![b'{', 0xff, b'}']);
    assert_json(&fixture.call(&bytes), 400, OPENAI_UNPARSABLE);
    fixture.finish();
}

#[test]
fn count_tokens_answers_the_prompts_tokens_over_http() {
    let fixture = roomy();
    let body = format!(r#"{{"model":"{MODEL_ID}","messages":{HI}}}"#);
    let expected = format!(r#"{{"input_tokens":{}}}"#, HI_PROMPT.chars().count());
    assert_json(&fixture.post("/v1/messages/count_tokens", JSON, &body), 200, &expected);
    assert!(fixture.backend.generations().is_empty());
    let unknown = body.replace(MODEL_ID, "claude");
    let missing = r#"{"type":"error","error":{"type":"not_found_error","message":"The model 'claude' does not exist or you do not have access to it."}}"#;
    assert_json(&fixture.post("/v1/messages/count_tokens", JSON, &unknown), 404, missing);
    fixture.finish();
}

#[test]
fn errors_of_each_step_reach_the_client_in_their_surface_format() {
    let fixture = roomy();
    let prompt = HI_PROMPT.chars().count();
    fixture.backend.set_context_length(prompt);
    let context = format!(
        r#"{{"error":{{"message":"This model's maximum context length is {prompt} tokens, but the messages use {prompt} tokens.","type":"invalid_request_error","param":"messages","code":"context_length_exceeded"}}}}"#
    );
    let body = chat(r#","reasoning_effort":"none""#);
    assert_json(&fixture.post("/v1/chat/completions", JSON, &body), 400, &context);
    let anthropic = format!(
        r#"{{"type":"error","error":{{"type":"invalid_request_error","message":"This model's maximum context length is {prompt} tokens. However, you requested {} tokens ({prompt} in the messages, 50 in the completion)."}}}}"#,
        prompt + 50
    );
    assert_json(
        &fixture.post("/v1/messages", JSON, &messages("").replace(":8,", ":50,")),
        400,
        &anthropic,
    );
    fixture.finish();
}
