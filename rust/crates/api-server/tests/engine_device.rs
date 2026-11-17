//! A chat completion on the device, end to end: the engine starts through the backend's start function, the server
//! runs on its lattice on a loopback port, and a client asks for greedy completions. The tests are `#[ignore]`d and
//! run locally, through the system arbiter (the engine loads the 12 GB artifact):
//! `system-run --gpu shared:16 --mem 24G --est 400 --who <name> -- cargo test -p euhedral-inference-api-server --test engine_device -- --ignored --nocapture`.
//!
//! The artifact comes from `euhedral.qwen.rco-artifact` and the library from `euhedral.cuda.library`.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    clippy::print_stdout,
    reason = "a device test: the test thread waits on hand-offs and channels with a bound, and says why it skipped"
)]

mod common;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use common::{CREATED, Raw, text};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_server::engine_backend::start;
use euhedral_inference_api_server::{ApiConfig, Limits, RequestDefaults, once, start as start_server};
use euhedral_inference_engine::Handoff;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_inference_tokenizer::read_artifact_tokenizer;
use euhedral_io::AsyncIo;
use euhedral_net::{ClientConfig, Method, Request, Response, request};

/// The longest the test waits for the engine to load or for one answer.
const WAIT: Duration = Duration::from_secs(20 * 60);
const QUESTION: &str = "What is the capital of France? Answer in one word.";
const NEW_TOKENS: u32 = 24;

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.rco-artifact names no file");
        return None;
    }
    Some(Machine { artifact, library })
}

/// What a reference generation on the engine selected.
struct Reference {
    tokens: Arc<Mutex<Vec<u32>>>,
    done: Arc<Handoff<FinishReason>>,
}

impl GenerationListener for Reference {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.lock().unwrap_or_else(PoisonError::into_inner).push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.done.send(outcome.reason);
    }
}

fn integers(value: &JsonValue) -> Vec<u32> {
    let JsonValue::Array(items) = value else {
        panic!("not an array: {value:?}");
    };
    items
        .iter()
        .map(|item| item.to_compact_string().parse().expect("an id"))
        .collect()
}

fn string(value: Option<&JsonValue>) -> String {
    match value {
        Some(JsonValue::String(text)) => text.clone(),
        other => panic!("not a string: {other:?}"),
    }
}

fn number(value: Option<&JsonValue>) -> usize {
    value
        .expect("a number")
        .to_compact_string()
        .parse()
        .expect("an integer")
}

fn body(extra: &str, tokens: u32) -> String {
    ask(QUESTION, extra, tokens)
}

fn ask(question: &str, extra: &str, tokens: u32) -> String {
    format!(
        r#"{{"model":"qwen","messages":[{{"role":"user","content":"{question}"}}],"max_tokens":{tokens},"temperature":0,"reasoning_effort":"none"{extra}}}"#
    )
}

/// A chat completions request as a raw socket sends it.
fn post_head(body: &str) -> String {
    format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

struct Client {
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    addr: SocketAddr,
}

impl Client {
    fn post(&self, body: &str) -> Response {
        let post = Request::new(Method::Post, "/v1/chat/completions")
            .header("Host", "t")
            .header("Content-Type", "application/json")
            .with_body(body);
        let (tx, rx) = channel();
        let config = ClientConfig {
            time_limit: Some(WAIT),
            ..ClientConfig::default()
        };
        request(
            &self.io,
            &self.sink,
            self.addr,
            &post,
            config,
            Box::new(move |outcome| {
                drop(tx.send(outcome));
                once(|| {})
            }),
        );
        rx.recv_timeout(WAIT + Duration::from_secs(5))
            .expect("the client reported")
            .expect("an answer")
    }
}

#[test]
#[ignore = "needs the device: run through the system arbiter"]
fn a_greedy_chat_completion_matches_the_engine_and_a_cancel_frees_the_sequence() {
    let Some(machine) = machine() else {
        return;
    };
    let mut settings = Settings::with_model(ModelSettings {
        artifact: machine.artifact.clone(),
        name: "qwen".to_owned(),
    });
    settings.context.max_tokens = 4096;
    let host = start(&settings, &machine.library, WAIT).expect("the engine starts");
    let backend = host.backend();

    let config = ApiConfig {
        address: "127.0.0.1:0".parse().expect("an address"),
        limits: Limits::default(),
        backend,
        defaults: RequestDefaults::default(),
        io: None,
        created_at: CREATED,
    };
    let server = start_server(host.lattice(), config).expect("a started server");
    let io = AsyncIo::open().expect("an AsyncIo");
    let sink = Arc::new(QueueIngestSink::new());
    host.lattice().add_upstream(io.clone()).expect("io attached");
    host.lattice().add_upstream(sink.delegate()).expect("sink attached");
    let client = Client {
        io,
        sink: Arc::clone(&sink),
        addr: server.local_addr(),
    };

    // The model and its limits come from the engine.
    let tokenizer = read_artifact_tokenizer(&machine.artifact).expect("the artifact's tokenizer");

    // A non-streamed completion.
    let response = client.post(&body(r#","return_token_ids":true"#, NEW_TOKENS));
    assert_eq!(response.status, 200, "{}", text(&response));
    let json = JsonValue::parse(text(&response)).expect("a JSON body");
    let choice = match json.get("choices") {
        Some(JsonValue::Array(choices)) => choices[0].clone(),
        other => panic!("no choices: {other:?}"),
    };
    let content = string(choice.get("message").and_then(|message| message.get("content")));
    let finish = string(choice.get("finish_reason"));
    assert_eq!(finish, "stop", "{content:?}");
    assert!(!content.trim().is_empty());
    let prompt_ids = integers(json.get("prompt_token_ids").expect("prompt ids"));
    let completion_ids = integers(choice.get("token_ids").expect("completion ids"));
    let usage = json.get("usage").expect("usage");
    assert_eq!(number(usage.get("prompt_tokens")), prompt_ids.len());
    assert_eq!(number(usage.get("completion_tokens")), completion_ids.len());
    assert_eq!(
        number(usage.get("total_tokens")),
        prompt_ids.len() + completion_ids.len()
    );

    // The same prompt, run straight on the engine as the benchmark runs it: greedy, the artifact's speculation.
    let runtime = host.runtime();
    let tokens = Arc::new(Mutex::new(Vec::new()));
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt: prompt_ids.clone(),
            max_tokens: NEW_TOKENS,
            stop_tokens: tokenizer.eos_ids().to_vec(),
            speculation: if runtime.plan().draft_depth > 0 {
                Speculation::Artifact
            } else {
                Speculation::Off
            },
            sampling: SamplingConfig::greedy(0),
            listener: Box::new(Reference {
                tokens: Arc::clone(&tokens),
                done: Arc::clone(&done),
            }),
        })
        .expect("the engine takes the reference generation");
    let reason = done.wait(WAIT).expect("the reference generation ended");
    assert_eq!(reason, FinishReason::Stop);
    let reference = tokens.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(
        completion_ids, reference,
        "the server's tokens are the engine's greedy tokens"
    );
    let visible: Vec<u32> = reference
        .iter()
        .copied()
        .filter(|id| !tokenizer.eos_ids().contains(id))
        .collect();
    let decoded = tokenizer.decode_stream().push_all(&visible).expect("decodable");
    assert_eq!(content, decoded, "the text is the decoded tokens");

    // The streamed completion says the same, with its usage.
    let mut raw = Raw::connect(server.local_addr());
    raw.send(&post_head(&body(
        r#","stream":true,"stream_options":{"include_usage":true}"#,
        NEW_TOKENS,
    )));
    let streamed = raw.read_until("data:[DONE]");
    assert!(streamed.starts_with("HTTP/1.1 200"), "{streamed}");
    drop(raw);
    let mut streamed_text = String::new();
    let mut streamed_finish = None;
    let mut streamed_usage = None;
    let mut done_marker = false;
    for line in streamed.lines() {
        let Some(data) = line.trim_end().strip_prefix("data:") else {
            continue;
        };
        if data.trim_start() == "[DONE]" {
            done_marker = true;
            continue;
        }
        let chunk = JsonValue::parse(data).expect("a JSON chunk");
        if let Some(JsonValue::Array(choices)) = chunk.get("choices")
            && let Some(choice) = choices.first()
        {
            if let Some(JsonValue::String(piece)) = choice.get("delta").and_then(|delta| delta.get("content")) {
                streamed_text.push_str(piece);
            }
            if let Some(JsonValue::String(reason)) = choice.get("finish_reason") {
                streamed_finish = Some(reason.clone());
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|usage| **usage != JsonValue::Null) {
            streamed_usage = Some((
                number(usage.get("prompt_tokens")),
                number(usage.get("completion_tokens")),
            ));
        }
    }
    assert!(done_marker, "the stream ends with [DONE]");
    assert_eq!(streamed_text, content);
    assert_eq!(streamed_finish.as_deref(), Some("stop"));
    assert_eq!(streamed_usage, Some((prompt_ids.len(), completion_ids.len())));

    // A request cancelled mid-generation (its client hangs up) frees the one sequence: the next request runs, and in
    // far less time than the cancelled generation's budget would take.
    let mut raw = Raw::connect(server.local_addr());
    raw.send(&post_head(&ask(
        "Write a very long story about a dragon.",
        r#","stream":true"#,
        3000,
    )));
    raw.read_until(r#""delta":{"content":""#);
    drop(raw);
    let before = Instant::now();
    let again = client.post(&body(r#","return_token_ids":true"#, NEW_TOKENS));
    assert_eq!(again.status, 200, "{}", text(&again));
    let again = JsonValue::parse(text(&again)).expect("a JSON body");
    let again_ids = match again.get("choices") {
        Some(JsonValue::Array(choices)) => integers(choices[0].get("token_ids").expect("completion ids")),
        other => panic!("no choices: {other:?}"),
    };
    assert_eq!(again_ids, reference, "the sequence after a cancel starts clean");
    println!("the request after the cancel was answered in {:?}", before.elapsed());

    // Stop the server, then the engine.
    let (tx, rx) = channel();
    server.stop(once(move || {
        tx.send(()).ok();
    }));
    rx.recv_timeout(Duration::from_secs(30)).expect("the server stopped");
    server.release();
    sink.complete_gracefully();
    host.shutdown().expect("the engine stops");
}
