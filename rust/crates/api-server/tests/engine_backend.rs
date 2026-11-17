//! The backend over the engine, against a fake engine and a fake tokenizer: how a planned request becomes the engine's,
//! how the engine's events become the listener's, that the end is told exactly once on every path, and which requests
//! are refused on each surface.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use common::{CREATED, PATIENCE, lattice, text};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_server::backend::{
    Backend, BackendError, EncodedPrompt, EngineReport, Generation, GenerationConfig, GenerationResult, OutputSpec,
    TextListener,
};
use euhedral_inference_api_server::{
    ApiConfig, Codec, Engine, EngineBackend, Limits, RequestDefaults, TextStream, once, start,
};
use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, FinishReason, GenerationListener, GenerationOutcome,
    GenerationRequest as EngineRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_io::AsyncIo;
use euhedral_net::{ClientConfig, Method, Request, Response, request};

const STOP: u32 = 1;
const THINK_END: u32 = 2;
const BAD: u32 = u32::MAX;

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One token per character; ids `STOP`, `THINK_END` and `BAD` are a terminator, the end of thinking and an id that
/// cannot be decoded.
struct Chars;

struct CharStream;

impl TextStream for CharStream {
    fn push(&mut self, id: u32) -> Result<String, String> {
        match char::from_u32(id) {
            Some(letter) if id != BAD => Ok(letter.to_string()),
            _ => Err(format!("token {id} has no text")),
        }
    }

    fn finish(self: Box<Self>) -> String {
        "!".to_owned()
    }
}

impl Codec for Chars {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn stream(&self) -> Box<dyn TextStream> {
        Box::new(CharStream)
    }

    fn stop_tokens(&self) -> Vec<u32> {
        vec![STOP]
    }

    fn think_end(&self) -> Option<u32> {
        Some(THINK_END)
    }
}

/// What the fake engine does with the next request.
enum Mode {
    /// Runs the request at once, on the offering thread: these tokens, then this ending.
    Run(Vec<u32>, FinishReason),
    /// Keeps the request for the test to drive.
    Hold,
    /// Refuses it, dropping its listener untold.
    Refuse(AdmissionError),
    /// Takes it and drops it untold.
    Lose,
}

struct Seen {
    prompt: Vec<u32>,
    max_tokens: u32,
    stop_tokens: Vec<u32>,
    speculation: Speculation,
    seed: u64,
    greedy: bool,
}

struct FakeEngine {
    mode: Mutex<Mode>,
    seen: Mutex<Vec<Seen>>,
    held: Mutex<VecDeque<(Box<dyn GenerationListener>, CancelHandle)>>,
    speculation: Speculation,
}

impl FakeEngine {
    fn new(mode: Mode) -> Arc<FakeEngine> {
        Arc::new(FakeEngine {
            mode: Mutex::new(mode),
            seen: Mutex::new(Vec::new()),
            held: Mutex::new(VecDeque::new()),
            speculation: Speculation::Artifact,
        })
    }

    fn set(&self, mode: Mode) {
        *locked(&self.mode) = mode;
    }
}

fn quantum(kind: QuantumKind) -> QuantumEvent {
    let now = Instant::now();
    QuantumEvent {
        kind,
        rows: 1,
        start_position: 0,
        admitted: now,
        retired: now,
        replayed: false,
        recorded: false,
    }
}

fn outcome(reason: FinishReason, generated: usize) -> GenerationOutcome {
    GenerationOutcome {
        reason,
        prompt_tokens: 3,
        generated,
        elapsed: Duration::ZERO,
    }
}

fn play(listener: &mut dyn GenerationListener, tokens: &[u32], reason: FinishReason) {
    let now = Instant::now();
    listener.quantum(quantum(QuantumKind::Prefill));
    listener.prefill_done(now);
    for id in tokens {
        listener.quantum(quantum(QuantumKind::Decode));
        listener.token(*id, now);
    }
    listener.finished(outcome(reason, tokens.len()));
}

impl Engine for FakeEngine {
    fn generate(&self, request: EngineRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        locked(&self.seen).push(Seen {
            prompt: request.prompt.clone(),
            max_tokens: request.max_tokens,
            stop_tokens: request.stop_tokens.clone(),
            speculation: request.speculation,
            seed: request.sampling.seed(),
            greedy: request.sampling.is_greedy(),
        });
        let mut listener = request.listener;
        let mut mode = locked(&self.mode);
        match &mut *mode {
            Mode::Run(tokens, reason) => {
                play(listener.as_mut(), tokens, reason.clone());
                Ok(())
            }
            Mode::Hold => {
                locked(&self.held).push_back((listener, cancel));
                Ok(())
            }
            Mode::Refuse(error) => Err(match error {
                AdmissionError::Busy => AdmissionError::Busy,
                AdmissionError::Engine(text) => AdmissionError::Engine(text.clone()),
                _ => AdmissionError::EmptyPrompt,
            }),
            Mode::Lose => Ok(()),
        }
    }

    fn context_length(&self) -> usize {
        4096
    }

    fn sequences(&self) -> usize {
        1
    }

    fn speculation(&self) -> Speculation {
        self.speculation
    }

    fn report(&self) -> EngineReport {
        EngineReport {
            workers: 3,
            speculative_depth: 2,
            ..EngineReport::default()
        }
    }
}

/// Everything a listener heard.
#[derive(Default)]
struct Heard {
    events: Vec<String>,
    results: Vec<Result<GenerationResult, BackendError>>,
}

struct Recorder(Arc<Mutex<Heard>>, Sender<()>);

impl TextListener for Recorder {
    fn text(&mut self, text: &str) {
        locked(&self.0).events.push(format!("text:{text}"));
    }

    fn prefilled(&mut self) {
        locked(&self.0).events.push("prefilled".to_owned());
    }

    fn finished(&mut self, result: Result<GenerationResult, BackendError>) {
        locked(&self.0).results.push(result);
        self.1.send(()).ok();
    }
}

struct Rig {
    lattice: Arc<ControlPlaneLattice>,
    engine: Arc<FakeEngine>,
    backend: Arc<EngineBackend>,
}

impl Rig {
    fn new(mode: Mode) -> Rig {
        let lattice = lattice(Some(2));
        let engine = FakeEngine::new(mode);
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(sink.delegate()).expect("sink attached");
        let backend = Arc::new(EngineBackend::new(
            engine.clone(),
            Arc::new(Chars),
            sink,
            "fake-model".to_owned(),
        ));
        Rig {
            lattice,
            engine,
            backend,
        }
    }

    fn open(&self, output: OutputSpec) -> Arc<dyn Generation> {
        self.backend
            .open_generation(GenerationConfig::greedy(7), output)
            .expect("a generation")
    }

    /// Runs one generation of the prompt `abc` to the listener's end, returning what it heard and how often it ended.
    fn run(&self, generation: &Arc<dyn Generation>) -> (Arc<Mutex<Heard>>, Receiver<()>) {
        let heard = Arc::new(Mutex::new(Heard::default()));
        let (tx, rx) = channel();
        generation.generate(
            EncodedPrompt {
                text: "abc".to_owned(),
                token_ids: vec![97, 98, 99],
            },
            5,
            Box::new(Recorder(heard.clone(), tx)),
        );
        (heard, rx)
    }

    fn finish(self) {
        self.backend.close();
        self.lattice.close();
    }
}

fn ended(rx: &Receiver<()>) {
    rx.recv_timeout(PATIENCE).expect("the listener was told the end");
    assert!(
        rx.recv_timeout(Duration::from_millis(100)).is_err(),
        "the end was told twice"
    );
}

#[test]
fn a_request_reaches_the_engine_as_prompt_budget_stops_and_seed() {
    let rig = Rig::new(Mode::Run(vec![u32::from('x'), STOP], FinishReason::Stop));
    let generation = rig.open(OutputSpec::text());
    let (_, rx) = rig.run(&generation);
    ended(&rx);
    let seen = locked(&rig.engine.seen);
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].prompt, vec![97, 98, 99]);
    assert_eq!(seen[0].max_tokens, 5);
    assert_eq!(seen[0].stop_tokens, vec![STOP]);
    assert_eq!(seen[0].speculation, Speculation::Artifact);
    assert_eq!(seen[0].seed, 7);
    assert!(seen[0].greedy);
    drop(seen);
    rig.finish();
}

#[test]
fn tokens_arrive_in_order_as_text_then_the_result_with_usage() {
    let tokens = vec![u32::from('h'), u32::from('i'), STOP];
    let rig = Rig::new(Mode::Run(tokens.clone(), FinishReason::Stop));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    let heard = locked(&heard);
    assert_eq!(heard.events, ["prefilled", "text:h", "text:i", "text:!"]);
    let result = heard.results[0].as_ref().expect("a result");
    assert_eq!(result.completion_tokens, 3, "the terminator counts");
    assert!(result.stop_token_reached);
    assert_eq!(result.token_ids, tokens);
    assert_eq!(result.reasoning_tokens, 0);
    drop(heard);
    generation.close();
    rig.finish();
}

#[test]
fn a_run_to_the_budget_has_not_reached_a_stop_token() {
    let rig = Rig::new(Mode::Run(vec![u32::from('a'), u32::from('b')], FinishReason::Length));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    let result = locked(&heard).results[0].clone().expect("a result");
    assert_eq!(result.completion_tokens, 2);
    assert!(!result.stop_token_reached);
    rig.finish();
}

#[test]
fn reasoning_tokens_are_those_before_the_end_of_thinking() {
    let rig = Rig::new(Mode::Run(
        vec![u32::from('a'), u32::from('b'), THINK_END, u32::from('c'), STOP],
        FinishReason::Stop,
    ));
    let reasoning = OutputSpec {
        reasoning: true,
        ..OutputSpec::text()
    };
    let generation = rig.open(reasoning.clone());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert_eq!(locked(&heard).results[0].clone().expect("a result").reasoning_tokens, 2);

    // Thinking that never ends is all the tokens but the terminator.
    rig.engine.set(Mode::Run(
        vec![u32::from('a'), u32::from('b'), STOP],
        FinishReason::Stop,
    ));
    let generation = rig.open(reasoning);
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert_eq!(locked(&heard).results[0].clone().expect("a result").reasoning_tokens, 2);
    rig.finish();
}

#[test]
fn a_cancel_stops_the_engine_and_the_partial_result_is_told_once() {
    let rig = Rig::new(Mode::Hold);
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    let (mut listener, cancel) = locked(&rig.engine.held)
        .pop_front()
        .expect("the engine holds the request");
    assert!(!generation.is_cancelled());
    generation.cancel();
    assert!(cancel.is_cancelled(), "the engine's handle is the generation's");
    assert!(generation.is_cancelled());
    play(listener.as_mut(), &[u32::from('z')], FinishReason::Cancelled);
    ended(&rx);
    let result = locked(&heard).results[0]
        .clone()
        .expect("a cancelled generation has its partial result");
    assert_eq!(result.completion_tokens, 1);
    assert!(!result.stop_token_reached);
    generation.close();
    generation.close();
    rig.finish();
}

#[test]
fn a_failure_is_told_once_after_the_text_before_it() {
    let rig = Rig::new(Mode::Run(
        vec![u32::from('a')],
        FinishReason::Failed("device lost".to_owned()),
    ));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    let heard = locked(&heard);
    assert_eq!(heard.events, ["prefilled", "text:a", "text:!"]);
    assert_eq!(heard.results, [Err(BackendError::Failed("device lost".to_owned()))]);
    drop(heard);
    rig.finish();
}

#[test]
fn a_token_that_cannot_be_decoded_cancels_and_fails_the_generation() {
    let rig = Rig::new(Mode::Run(
        vec![u32::from('a'), BAD, u32::from('b')],
        FinishReason::Cancelled,
    ));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert!(generation.is_cancelled(), "the engine was told to stop");
    let heard = locked(&heard);
    assert_eq!(heard.events, ["prefilled", "text:a"]);
    assert!(
        matches!(heard.results[0], Err(BackendError::Failed(_))),
        "{:?}",
        heard.results
    );
    drop(heard);
    rig.finish();
}

#[test]
fn a_refused_begin_ends_the_listener_once_with_the_reason() {
    let rig = Rig::new(Mode::Refuse(AdmissionError::Busy));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert!(matches!(locked(&heard).results[0], Err(BackendError::Failed(_))));
    assert!(locked(&heard).events.is_empty());
    generation.close();

    rig.engine
        .set(Mode::Refuse(AdmissionError::Engine("the engine is closed".to_owned())));
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert_eq!(
        locked(&heard).results,
        [Err(BackendError::Unavailable("the engine is closed".to_owned()))]
    );
    generation.close();
    rig.finish();
}

#[test]
fn a_request_the_engine_takes_and_drops_ends_the_listener_once() {
    let rig = Rig::new(Mode::Lose);
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    ended(&rx);
    assert!(matches!(locked(&heard).results[0], Err(BackendError::Failed(_))));
    rig.finish();
}

#[test]
fn a_request_the_engine_drops_later_ends_the_listener_once() {
    let rig = Rig::new(Mode::Hold);
    let generation = rig.open(OutputSpec::text());
    let (heard, rx) = rig.run(&generation);
    drop(locked(&rig.engine.held).pop_front());
    ended(&rx);
    assert!(matches!(locked(&heard).results[0], Err(BackendError::Failed(_))));
    rig.finish();
}

#[test]
fn a_generation_runs_once() {
    let rig = Rig::new(Mode::Run(vec![STOP], FinishReason::Stop));
    let generation = rig.open(OutputSpec::text());
    let (_, first) = rig.run(&generation);
    ended(&first);
    let (heard, second) = rig.run(&generation);
    ended(&second);
    assert!(matches!(locked(&heard).results[0], Err(BackendError::Failed(_))));
    assert_eq!(locked(&rig.engine.seen).len(), 1, "the engine ran one");
    rig.finish();
}

#[test]
fn opening_is_refused_when_the_engine_cannot_run_the_output() {
    let rig = Rig::new(Mode::Hold);
    let open = |output| rig.backend.open_generation(GenerationConfig::greedy(1), output).err();
    assert!(matches!(
        open(OutputSpec {
            grammar: Some("start: \"a\"".to_owned()),
            ..OutputSpec::text()
        }),
        Some(BackendError::Failed(_))
    ));
    assert!(matches!(
        open(OutputSpec {
            reasoning: true,
            reasoning_budget: Some(8),
            grammar: None
        }),
        Some(BackendError::Failed(_))
    ));
    rig.backend.close();
    assert!(!rig.backend.is_available());
    assert!(matches!(open(OutputSpec::text()), Some(BackendError::Unavailable(_))));
    rig.finish();
}

#[test]
fn the_backend_reports_the_engine() {
    let rig = Rig::new(Mode::Hold);
    assert_eq!(rig.backend.model_id(), "fake-model");
    assert_eq!(rig.backend.context_length(), 4096);
    assert_eq!(rig.backend.max_active_generations(), 1);
    assert_eq!(rig.backend.engine_report().workers, 3);
    assert_eq!(rig.backend.engine_report().speculative_depth, 2);
    rig.finish();
}

#[test]
fn a_prompt_is_encoded_on_the_lattice_and_refused_once_the_sink_closes() {
    let rig = Rig::new(Mode::Hold);
    let (tx, rx) = channel();
    let first = tx.clone();
    rig.backend
        .encode_prompt("hi".to_owned(), Box::new(move |encoded| drop(first.send(encoded))));
    let encoded = rx.recv_timeout(PATIENCE).expect("encoded").expect("an encoding");
    assert_eq!(encoded.token_ids, vec![104, 105]);
    assert_eq!(encoded.text, "hi");
    rig.backend.close();
    rig.backend
        .encode_prompt("hi".to_owned(), Box::new(move |encoded| drop(tx.send(encoded))));
    let refused = rx.recv_timeout(PATIENCE).expect("told");
    assert!(matches!(refused, Err(BackendError::Unavailable(_))), "{refused:?}");
    rig.finish();
}

/// A server on the fake engine's backend, and a client for it.
struct Served {
    rig: Rig,
    server: euhedral_inference_api_server::ApiServer,
    addr: SocketAddr,
    io: Arc<AsyncIo>,
    client: Arc<QueueIngestSink>,
}

impl Served {
    fn new(mode: Mode) -> Served {
        let rig = Rig::new(mode);
        let config = ApiConfig {
            address: "127.0.0.1:0".parse().expect("an address"),
            limits: Limits::default(),
            backend: rig.backend.clone(),
            defaults: RequestDefaults::default(),
            io: None,
            created_at: CREATED,
        };
        let server = start(&rig.lattice, config).expect("a started server");
        let io = AsyncIo::open().expect("an AsyncIo");
        let client = Arc::new(QueueIngestSink::new());
        rig.lattice.add_upstream(io.clone()).expect("io attached");
        rig.lattice.add_upstream(client.delegate()).expect("client attached");
        let addr = server.local_addr();
        Served {
            rig,
            server,
            addr,
            io,
            client,
        }
    }

    fn post(&self, target: &str, body: &str) -> Response {
        let post = Request::new(Method::Post, target)
            .header("Host", "t")
            .header("Content-Type", "application/json")
            .with_body(body);
        let (tx, rx) = channel();
        let config = ClientConfig {
            time_limit: Some(PATIENCE),
            ..ClientConfig::default()
        };
        request(
            &self.io,
            &self.client,
            self.addr,
            &post,
            config,
            Box::new(move |outcome| {
                drop(tx.send(outcome));
                once(|| {})
            }),
        );
        rx.recv_timeout(PATIENCE + Duration::from_secs(5))
            .expect("the client reported")
            .expect("an answer")
    }

    fn finish(self) {
        let (tx, rx) = channel();
        self.server.stop(once(move || {
            tx.send(()).ok();
        }));
        rx.recv_timeout(PATIENCE).expect("the server stopped");
        self.server.release();
        self.client.complete_gracefully();
        self.rig.finish();
    }
}

const CHAT: &str = "/v1/chat/completions";
const MESSAGES: &str = "/v1/messages";
const RESPONSES: &str = "/v1/responses";

fn chat(extra: &str) -> String {
    format!(r#"{{"model":"fake-model","messages":[{{"role":"user","content":"Hi"}}],"max_tokens":8{extra}}}"#)
}

fn messages(extra: &str) -> String {
    format!(r#"{{"model":"fake-model","max_tokens":64,"messages":[{{"role":"user","content":"Hi"}}]{extra}}}"#)
}

fn responses(extra: &str) -> String {
    format!(r#"{{"model":"fake-model","input":"Hi","max_output_tokens":8{extra}}}"#)
}

/// Asserts the answer is an `unsupported_parameter` refusal naming `param`, in the surface's error format.
fn assert_refused(response: &Response, param: &str, anthropic: bool) {
    let body = text(response);
    assert_eq!(response.status, 400, "{body}");
    let named = format!("Unsupported parameter: '{param}' is not supported by this server.");
    assert!(body.contains(&named), "{body}");
    if anthropic {
        assert!(body.starts_with(r#"{"type":"error""#), "{body}");
    } else {
        assert!(body.contains(r#""code":"unsupported_parameter""#), "{body}");
        assert!(body.contains(&format!(r#""param":"{param}""#)), "{body}");
    }
}

#[test]
fn a_greedy_chat_completion_runs_on_the_engine() {
    let served = Served::new(Mode::Run(
        vec![u32::from('H'), u32::from('i'), STOP],
        FinishReason::Stop,
    ));
    let response = served.post(CHAT, &chat(r#","temperature":0,"reasoning_effort":"none""#));
    let body = text(&response);
    assert_eq!(response.status, 200, "{body}");
    assert!(body.contains(r#""content":"Hi!""#), "{body}");
    assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
    assert!(body.contains(r#""completion_tokens":3"#), "{body}");
    assert_eq!(locked(&served.rig.engine.seen).len(), 1);
    served.finish();
}

#[test]
fn sampled_requests_are_refused_on_every_surface() {
    let served = Served::new(Mode::Hold);
    // A request that states nothing takes the server's sampling defaults, which sample.
    assert_refused(&served.post(CHAT, &chat("")), "temperature", false);
    assert_refused(&served.post(CHAT, &chat(r#","temperature":0.7"#)), "temperature", false);
    assert_refused(
        &served.post(MESSAGES, &messages(r#","temperature":0.7"#)),
        "temperature",
        true,
    );
    assert_refused(
        &served.post(RESPONSES, &responses(r#","temperature":0.7"#)),
        "temperature",
        false,
    );
    assert!(locked(&served.rig.engine.seen).is_empty(), "nothing reached the engine");
    served.finish();
}

#[test]
fn structured_output_is_refused_naming_the_surfaces_field() {
    let served = Served::new(Mode::Hold);
    let json = r#","reasoning_effort":"none","temperature":0,"response_format":{"type":"json_object"}"#;
    assert_refused(&served.post(CHAT, &chat(json)), "response_format", false);
    let format = r#","temperature":0,"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}"#;
    assert_refused(&served.post(MESSAGES, &messages(format)), "output_config.format", true);
    let format = r#","temperature":0,"text":{"format":{"type":"json_object"}}"#;
    assert_refused(&served.post(RESPONSES, &responses(format)), "text.format", false);
    assert!(locked(&served.rig.engine.seen).is_empty());
    served.finish();
}

#[test]
fn a_forced_tool_call_is_refused_and_an_automatic_one_is_not() {
    let served = Served::new(Mode::Run(
        vec![u32::from('o'), u32::from('k'), STOP],
        FinishReason::Stop,
    ));
    let tool =
        r#","temperature":0,"tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]"#;
    let forced = format!(r#"{tool},"tool_choice":"required""#);
    assert_refused(&served.post(CHAT, &chat(&forced)), "tool_choice", false);
    let auto = format!(r#"{tool},"tool_choice":"auto""#);
    assert_eq!(served.post(CHAT, &chat(&auto)).status, 200);
    served.finish();
}

#[test]
fn a_reasoning_budget_is_refused() {
    let served = Served::new(Mode::Hold);
    let thinking = r#","temperature":0,"thinking":{"type":"enabled","budget_tokens":32}"#;
    assert_refused(
        &served.post(MESSAGES, &messages(thinking)),
        "thinking.budget_tokens",
        true,
    );
    assert!(locked(&served.rig.engine.seen).is_empty());
    served.finish();
}
