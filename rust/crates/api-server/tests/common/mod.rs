//! A server on a lattice of its own and a client that talks to it through `euhedral-net`'s client.
#![allow(
    dead_code,
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::str::from_utf8;
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_hardware_utils::affinity;
use euhedral_inference_api_server::scripted::ScriptedBackend;
use euhedral_inference_api_server::{ApiConfig, ApiServer, Limits, once, start};
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
}

impl Fixture {
    pub fn start() -> Fixture {
        Fixture::with(small_limits(), Some(2))
    }

    pub fn with(limits: Limits, cpus: Option<usize>) -> Fixture {
        let lattice = lattice(cpus);
        let backend = ScriptedBackend::on(&lattice);
        let config = ApiConfig {
            address: "127.0.0.1:0".parse().expect("an address"),
            limits,
            backend: backend.clone(),
            io: None,
            created_at: CREATED,
        };
        let server = start(&lattice, config).expect("a started server");
        let io = AsyncIo::open().expect("an AsyncIo");
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
}

impl Raw {
    pub fn connect(addr: SocketAddr) -> Raw {
        let stream = TcpStream::connect_timeout(&addr, PATIENCE).expect("a connection");
        Raw { stream }
    }

    pub fn send(&mut self, bytes: &str) {
        self.stream.write_all(bytes.as_bytes()).expect("a write");
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
            Ok(count) => Some(String::from_utf8_lossy(&buffer[..count]).into_owned()),
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
