//! Constrained requests on the device, end to end: the engine starts through the backend's start function, the server
//! runs on its lattice on a loopback port, and a client asks for a JSON answer by schema and for a reasoning budget.
//! The tests are `#[ignore]`d and run locally, through the system arbiter, one engine at a time (the engine loads the 12 GB
//! artifact):
//! `system-run --gpu shared:14 --mem 24G --est 600 --who <name> -- <test binary> --ignored --test-threads=1 <test>`.
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
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::{CREATED, Raw, text};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_server::engine_backend::{EngineHost, start};
use euhedral_inference_api_server::{ApiConfig, ApiServer, Limits, RequestDefaults, once, start as start_server};
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;
use euhedral_net::{ClientConfig, Method, Request, Response, request};

/// The longest the test waits for the engine to load or for one answer.
const WAIT: Duration = Duration::from_secs(9 * 60);

fn machine() -> Option<(PathBuf, PathBuf)> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    if !artifact.is_file() {
        println!("skipped: euhedral.qwen.rco-artifact names no file");
        return None;
    }
    Some((artifact, library))
}

/// The card, held by one test at a time: every engine of the model fills it, so two never exist side by side.
static CARD: Mutex<()> = Mutex::new(());

/// The engine, a server on its lattice and a client for it.
struct Served {
    host: EngineHost,
    server: ApiServer,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    addr: SocketAddr,
    /// Released last, after the engine is gone, also when the test panics.
    _card: MutexGuard<'static, ()>,
}

impl Served {
    fn start() -> Option<Served> {
        let (artifact, library) = machine()?;
        let card = CARD.lock().unwrap_or_else(PoisonError::into_inner);
        let mut settings = Settings::with_model(ModelSettings {
            artifact,
            name: "qwen".to_owned(),
        });
        settings.context.max_tokens = 4096;
        let host = start(&settings, &library, WAIT, None).expect("the engine starts");
        let config = ApiConfig {
            address: "127.0.0.1:0".parse().expect("an address"),
            limits: Limits::default(),
            backend: host.backend(),
            defaults: RequestDefaults::default(),
            io: None,
            clock: None,
            created_at: CREATED,
        };
        let server = start_server(host.lattice(), config).expect("a started server");
        let io = AsyncIo::open().expect("an AsyncIo");
        let sink = Arc::new(QueueIngestSink::new());
        host.lattice().add_upstream(io.clone()).expect("io attached");
        host.lattice().add_upstream(sink.delegate()).expect("sink attached");
        let addr = server.local_addr();
        Some(Served {
            host,
            server,
            io,
            sink,
            addr,
            _card: card,
        })
    }

    fn post(&self, target: &str, body: &str) -> Response {
        let post = Request::new(Method::Post, target)
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

    /// The value of a counter series of the server's metrics, summed over its labels.
    fn counter(&self, name: &str) -> f64 {
        let mut raw = Raw::connect(self.addr);
        raw.send("GET /metrics HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
        let end = Instant::now() + Duration::from_secs(30);
        let mut seen = String::new();
        while Instant::now() < end {
            match raw.read_for(Duration::from_millis(200)) {
                Some(more) => seen.push_str(&more),
                None => break,
            }
        }
        let total: f64 = seen
            .lines()
            .filter(|line| line.starts_with(name) && line[name.len()..].starts_with('{'))
            .filter_map(|line| line.rsplit_once(' ')?.1.parse::<f64>().ok())
            .sum();
        assert!(seen.contains(name), "no {name} in the metrics: {seen:?}");
        total
    }

    fn stop(self) {
        let (tx, rx) = channel();
        self.server.stop(once(move || {
            tx.send(()).ok();
        }));
        rx.recv_timeout(Duration::from_secs(30)).expect("the server stopped");
        self.server.release();
        self.sink.complete_gracefully();
        self.host.shutdown(Duration::from_secs(30)).expect("the engine stops");
    }
}

#[test]
#[ignore = "needs the device: run through the system arbiter"]
fn a_greedy_json_schema_answer_parses_and_matches_the_schema() {
    let Some(served) = Served::start() else {
        return;
    };
    let before = served.counter("euhedral_reasoning_tokens_total");
    let schema = r#"{"type":"object","properties":{"city":{"type":"string"},"population_millions":{"type":"integer"}},"required":["city","population_millions"],"additionalProperties":false}"#;
    let body = format!(
        r#"{{"model":"qwen","messages":[{{"role":"user","content":"Name one large city in France and its population in millions."}}],"max_tokens":1500,"temperature":0,"response_format":{{"type":"json_schema","json_schema":{{"name":"city","strict":true,"schema":{schema}}}}}}}"#
    );
    let response = served.post("/v1/chat/completions", &body);
    let answer = text(&response);
    assert_eq!(response.status, 200, "{answer}");
    let reasoned = served.counter("euhedral_reasoning_tokens_total") - before;
    println!("the reasoning tokens before the constrained answer: {reasoned}");
    assert!(reasoned > 0.0, "reasoning was on: {answer}");
    let json = JsonValue::parse(answer).expect("a JSON body");
    let Some(JsonValue::Array(choices)) = json.get("choices") else {
        panic!("no choices: {answer}");
    };
    let message = choices[0].get("message").expect("a message");
    let Some(JsonValue::String(content)) = message.get("content") else {
        panic!("no content: {answer}");
    };
    println!("the constrained answer: {content}");
    let document = JsonValue::parse(content).expect("the content is JSON");
    let JsonValue::Object(members) = &document else {
        panic!("not an object: {content}");
    };
    assert_eq!(members.len(), 2, "no member beyond the schema's: {content}");
    assert!(matches!(document.get("city"), Some(JsonValue::String(city)) if !city.is_empty()));
    let population = document.get("population_millions").expect("the population");
    assert!(
        population.to_compact_string().parse::<i64>().is_ok(),
        "an integer: {content}"
    );
    served.stop();
}

#[test]
#[ignore = "needs the device: run through the system arbiter"]
fn a_reasoning_budget_of_eight_ends_the_think_block_within_eight_tokens() {
    let Some(served) = Served::start() else {
        return;
    };
    let before = served.counter("euhedral_reasoning_tokens_total");
    let body = r#"{"model":"qwen","max_tokens":300,"temperature":0,"thinking":{"type":"enabled","budget_tokens":8},"messages":[{"role":"user","content":"Think about it step by step, then say which is larger, 9.11 or 9.9."}]}"#;
    let response = served.post("/v1/messages", body);
    let answer = text(&response);
    assert_eq!(response.status, 200, "{answer}");
    let reasoned = served.counter("euhedral_reasoning_tokens_total") - before;
    println!("the reasoning tokens of the answer: {reasoned}");
    assert!(reasoned > 0.0, "the model reasoned: {answer}");
    assert!(
        reasoned <= 8.0,
        "the think block ended within the budget: {reasoned} tokens"
    );
    let json = JsonValue::parse(answer).expect("a JSON body");
    let Some(JsonValue::Array(blocks)) = json.get("content") else {
        panic!("no content: {answer}");
    };
    assert!(
        blocks
            .iter()
            .any(|block| matches!(block.get("type"), Some(JsonValue::String(kind)) if kind == "text")),
        "an answer follows the think block: {answer}"
    );
    served.stop();
}
