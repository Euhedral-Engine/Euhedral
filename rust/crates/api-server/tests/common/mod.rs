//! A server on a lattice of its own and a client that talks to it through `euhedral-net`'s client.
#![allow(
    dead_code,
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::str::from_utf8;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_hardware_utils::affinity;
use euhedral_inference_api_server::scripted::ScriptedBackend;
use euhedral_inference_api_server::{ApiConfig, ApiServer, Clock, EngineReport, Limits, RequestDefaults, once, start};
use euhedral_io::AsyncIo;
use euhedral_net::{ClientConfig, Method, NetError, Request, Response, request};

/// How long a test waits for anything before it fails.
pub const PATIENCE: Duration = Duration::from_secs(20);

/// A lattice on at most `cpus` of the CPUs this process may use (`None`: the default worker count).
pub fn lattice(cpus: Option<usize>) -> Arc<ControlPlaneLattice> {
    let cloneable = BaseCloneableObject::new(FragmentConfig::default(), DefaultExecutor::new());
    let mut config = LatticeConfig::new(Arc::new(cloneable));
    config.name = "api-server-test".into();
    config.shard_name = "api-server-test".into();
    if let Some(count) = cpus {
        config.allowed_cpus = affinity::base_mask().iter().take(count).collect();
    }
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let lattice = ControlPlaneLattice::new(config).expect("a lattice");
    lattice.start().expect("a started lattice");
    lattice
}

/// The time admission sees, which only the test moves: the moments are a base plus what was [advanced](Self::advance),
/// and a timer fires, as a frame into the lattice, when the test moves the clock to its deadline or past it.
pub struct ManualClock {
    base: Instant,
    sink: Arc<QueueIngestSink>,
    state: Mutex<ClockState>,
}

struct ClockState {
    moved: Duration,
    timers: Vec<(Instant, Box<dyn Frame>)>,
}

impl ManualClock {
    pub fn new() -> Arc<ManualClock> {
        Arc::new(ManualClock {
            base: Instant::now(),
            sink: Arc::new(QueueIngestSink::new()),
            state: Mutex::new(ClockState {
                moved: Duration::ZERO,
                timers: Vec::new(),
            }),
        })
    }

    /// Attaches the clock's timers to `lattice`, which emits them when they fire.
    pub fn attach(&self, lattice: &ControlPlaneLattice) {
        lattice
            .add_upstream(self.sink.delegate())
            .expect("the timers are attached");
    }

    /// Completes the clock's timers: call it before the lattice closes.
    pub fn complete(&self) {
        self.sink.complete_gracefully();
    }

    fn lock(&self) -> MutexGuard<'_, ClockState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Moves the clock on by `by` and emits the timers that are due.
    pub fn advance(&self, by: Duration) {
        let due = {
            let mut state = self.lock();
            state.moved += by;
            let now = self.base + state.moved;
            let (due, later): (Vec<_>, Vec<_>) = state.timers.drain(..).partition(|(deadline, _)| *deadline <= now);
            state.timers = later;
            due
        };
        for (_, frame) in due {
            drop(self.sink.offer(frame));
        }
    }

    /// The earliest deadline of a timer that has not fired.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.lock().timers.iter().map(|(deadline, _)| *deadline).min()
    }

    /// The moment `after` from the start of the clock.
    pub fn at_offset(&self, after: Duration) -> Instant {
        self.base + after
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Instant {
        self.base + self.lock().moved
    }

    fn at(&self, deadline: Instant, frame: Box<dyn Frame>) {
        let mut state = self.lock();
        if deadline <= self.base + state.moved {
            drop(state);
            drop(self.sink.offer(frame));
        } else {
            state.timers.push((deadline, frame));
        }
    }
}

/// Limits small enough to cross in a test.
pub fn small_limits() -> Limits {
    Limits {
        max_body_bytes: 64,
        over_limit_read_bytes: 1024,
        max_connections: 16,
        read_limit: None,
        idle_limit: None,
        write_limit: None,
        ..Limits::default()
    }
}

pub const CREATED: i64 = 1_715_935_800;

pub struct Fixture {
    pub lattice: Arc<ControlPlaneLattice>,
    pub backend: Arc<ScriptedBackend>,
    pub server: ApiServer,
    pub addr: SocketAddr,
    io: Arc<AsyncIo>,
    sink: Arc<QueueIngestSink>,
    clock: Option<Arc<ManualClock>>,
}

impl Fixture {
    pub fn start() -> Fixture {
        Fixture::with(small_limits(), Some(2))
    }

    pub fn with(limits: Limits, cpus: Option<usize>) -> Fixture {
        Fixture::reporting(limits, cpus, EngineReport::default())
    }

    /// A server whose backend reports `report` for the metrics.
    pub fn reporting(limits: Limits, cpus: Option<usize>, report: EngineReport) -> Fixture {
        Fixture::build(limits, cpus, report, None)
    }

    /// A server on a clock the test moves: its limits pass when [`ManualClock::advance`] reaches them.
    pub fn timed(limits: Limits) -> (Fixture, Arc<ManualClock>) {
        let clock = ManualClock::new();
        let fixture = Fixture::build(limits, Some(2), EngineReport::default(), Some(clock.clone()));
        (fixture, clock)
    }

    fn build(limits: Limits, cpus: Option<usize>, report: EngineReport, clock: Option<Arc<ManualClock>>) -> Fixture {
        let lattice = lattice(cpus);
        let backend = ScriptedBackend::on(&lattice);
        backend.set_engine_report(report);
        let config = ApiConfig {
            address: "127.0.0.1:0".parse().expect("an address"),
            limits,
            backend: backend.clone(),
            defaults: RequestDefaults::default(),
            io: None,
            clock: clock.clone().map(|clock| clock as Arc<dyn Clock>),
            created_at: CREATED,
        };
        let server = start(&lattice, config).expect("a started server");
        let io = AsyncIo::open().expect("an AsyncIo");
        if let Some(clock) = &clock {
            clock.attach(&lattice);
        }
        let sink = Arc::new(QueueIngestSink::new());
        lattice.add_upstream(io.clone()).expect("io attached");
        lattice.add_upstream(sink.delegate()).expect("sink attached");
        let addr = server.local_addr();
        Fixture {
            lattice,
            backend,
            server,
            addr,
            io,
            sink,
            clock,
        }
    }

    /// One request, and the answer or the client's error.
    pub fn try_call(&self, request_to_send: &Request) -> Result<Response, NetError> {
        let (tx, rx): (Sender<Result<Response, NetError>>, _) = channel();
        let config = ClientConfig {
            time_limit: Some(PATIENCE),
            ..ClientConfig::default()
        };
        request(
            &self.io,
            &self.sink,
            self.addr,
            request_to_send,
            config,
            Box::new(move |outcome| {
                drop(tx.send(outcome));
                once(|| {})
            }),
        );
        rx.recv_timeout(PATIENCE + Duration::from_secs(5))
            .expect("the client reported")
    }

    pub fn call(&self, request_to_send: &Request) -> Response {
        self.try_call(request_to_send).expect("an answer")
    }

    /// A request with the `Host` the client needs.
    pub fn req(method: Method, target: &str) -> Request {
        Request::new(method, target).header("Host", "t")
    }

    pub fn get(&self, target: &str) -> Response {
        self.call(&Fixture::req(Method::Get, target))
    }

    pub fn post(&self, target: &str, content_type: Option<&str>, body: &str) -> Response {
        let mut post = Fixture::req(Method::Post, target);
        if let Some(content_type) = content_type {
            post = post.header("Content-Type", content_type);
        }
        self.call(&post.with_body(body))
    }

    /// Stops the server, waits for the report and releases everything.
    pub fn finish(self) {
        self.stop();
        self.finish_after_stop();
    }

    /// Releases everything of a server that was stopped.
    pub fn finish_after_stop(self) {
        let counters = self.server.counters();
        assert_eq!(counters.open, 0, "a connection is still counted");
        self.server.release();
        if let Some(clock) = &self.clock {
            clock.complete();
        }
        self.backend.release();
        self.sink.complete_gracefully();
        self.lattice.close();
    }

    /// Stops the server and waits (bounded) for it to say so.
    pub fn stop(&self) {
        let (tx, rx) = channel();
        self.server.stop(once(move || {
            tx.send(()).ok();
        }));
        rx.recv_timeout(PATIENCE)
            .expect("the server did not report that it stopped");
    }
}

/// The response's headers as written, without the transport's own `Connection`.
pub fn headers(response: &Response) -> Vec<(String, String)> {
    response
        .headers
        .iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("connection"))
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect()
}

/// The headers of a JSON answer with a body of `length` bytes.
pub fn json_headers(length: usize) -> Vec<(String, String)> {
    vec![
        ("Content-Type".to_owned(), "application/json".to_owned()),
        ("Content-Length".to_owned(), length.to_string()),
    ]
}

pub fn text(response: &Response) -> &str {
    from_utf8(&response.body).expect("a UTF-8 body")
}

/// Asserts a JSON answer: status, the headers the API sets and the exact body.
pub fn assert_json(response: &Response, status: u16, body: &str) {
    assert_eq!(response.status, status, "{}", text(response));
    assert_eq!(headers(response), json_headers(body.len()));
    assert_eq!(text(response), body);
}

/// A plain socket to the server, for what the client cannot do: send a request in parts, hold a connection open.
pub struct Raw {
    stream: TcpStream,
    /// The start of a multi-byte character whose end has not arrived yet.
    held: Vec<u8>,
}

/// The text of `held` followed by `fresh`, without a character cut by the end of a read: the cut start stays in `held`
/// for the next read, so the text has the bytes of the wire, whatever the sizes of the reads.
pub fn utf8_text(held: &mut Vec<u8>, fresh: &[u8]) -> String {
    held.extend_from_slice(fresh);
    let whole = match from_utf8(held) {
        Ok(text) => held.len().min(text.len()),
        Err(error) if error.error_len().is_none() => error.valid_up_to(),
        Err(_) => held.len(),
    };
    let rest = held.split_off(whole);
    let text = String::from_utf8_lossy(held).into_owned();
    *held = rest;
    text
}

impl Raw {
    pub fn connect(addr: SocketAddr) -> Raw {
        let stream = TcpStream::connect_timeout(&addr, PATIENCE).expect("a connection");
        Raw {
            stream,
            held: Vec::new(),
        }
    }

    pub fn send(&mut self, bytes: &str) {
        self.stream.write_all(bytes.as_bytes()).expect("a write");
    }

    /// Closes the sending side only: the client still reads, as one that has sent its whole request does.
    pub fn half_close(&mut self) {
        self.stream.shutdown(Shutdown::Write).expect("a half close");
    }

    /// What arrives within `wait`, possibly nothing; `None` once the server has closed and nothing is left.
    pub fn read_for(&mut self, wait: Duration) -> Option<String> {
        self.stream.set_read_timeout(Some(wait)).expect("a read timeout");
        let mut buffer = [0u8; 4096];
        let read = loop {
            match self.stream.read(&mut buffer) {
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                other => break other,
            }
        };
        match read {
            Ok(0) => None,
            Ok(count) => Some(utf8_text(&mut self.held, &buffer[..count])),
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => Some(String::new()),
            Err(error) if error.kind() == ErrorKind::ConnectionReset => None,
            Err(error) => panic!("a read failed: {error}"),
        }
    }

    /// Reads until `needle` has arrived, within the patience of a test.
    pub fn read_until(&mut self, needle: &str) -> String {
        let end = Instant::now() + PATIENCE;
        let mut seen = String::new();
        while !seen.contains(needle) {
            assert!(Instant::now() < end, "no {needle:?} in {seen:?}");
            match self.read_for(Duration::from_millis(200)) {
                Some(more) => seen.push_str(&more),
                None => panic!("the server closed before {needle:?}; seen {seen:?}"),
            }
        }
        seen
    }

    /// Whether the server closes the connection within the patience of a test.
    pub fn closed_by_server(&mut self) -> bool {
        let end = Instant::now() + PATIENCE;
        while Instant::now() < end {
            if self.read_for(Duration::from_millis(200)).is_none() {
                return true;
            }
        }
        false
    }
}

impl Fixture {
    /// Waits, within the patience of a test, until the server counts `open` connections.
    pub fn wait_open(&self, open: u64) {
        let end = Instant::now() + PATIENCE;
        while self.server.counters().open != open {
            assert!(Instant::now() < end, "{:?}", self.server.counters());
            sleep(Duration::from_millis(10));
        }
    }
}
