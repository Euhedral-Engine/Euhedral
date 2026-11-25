//! Shared by the tests: a bounded raw HTTP client and the settings of a test server.
#![allow(
    dead_code,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the test thread is not a lattice worker; it reads sockets with a bound"
)]

use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_inference_api_server::scripted::MODEL_ID;

/// How long a test waits for anything before it fails.
pub const PATIENCE: Duration = Duration::from_secs(30);

/// The address a client connects to for a server that listens on `addr`: the loopback address where it listens on every
/// address.
pub fn reach(addr: SocketAddr) -> SocketAddr {
    if addr.ip().is_unspecified() {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), addr.port())
    } else {
        addr
    }
}

/// Waits until the port accepts connections, then requires the first exchange to be answered: a connection made in the
/// first moments after the port opens must not be reset or dropped.
pub fn ready(addr: SocketAddr) {
    let deadline = Instant::now() + PATIENCE;
    loop {
        match try_exchange(addr, "GET", "/health", None) {
            Err(error) if error.starts_with("connecting") => {
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for the port to open: {error}"
                );
                sleep(Duration::from_millis(10));
            }
            Ok(answer) => {
                assert_eq!(status(&answer), 200, "the first answer: {answer}");
                return;
            }
            Err(error) => panic!("the first connection after the port opened failed: {error}"),
        }
    }
}

/// The whole of one exchange: `request` sent, the answer read until the server closes (`Connection: close`) or the
/// patience ends; what was read either way.
pub fn exchange(addr: SocketAddr, method: &str, target: &str, body: Option<&str>) -> String {
    try_exchange(addr, method, target, body).unwrap_or_else(|error| panic!("{method} {target}: {error}"))
}

/// [`exchange`], reporting a connection that fails or an empty answer instead of panicking.
pub fn try_exchange(addr: SocketAddr, method: &str, target: &str, body: Option<&str>) -> Result<String, String> {
    let mut stream =
        TcpStream::connect_timeout(&reach(addr), PATIENCE).map_err(|error| format!("connecting: {error}"))?;
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .map_err(|error| error.to_string())?;
    let body = body.unwrap_or("");
    let request = format!(
        "{method} {target} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| format!("writing: {error}"))?;
    let mut text = Vec::new();
    let deadline = Instant::now() + PATIENCE;
    let mut chunk = [0_u8; 4096];
    while Instant::now() < deadline {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(count) => text.extend_from_slice(&chunk[..count]),
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted
                ) => {}
            Err(error) => return Err(format!("reading: {error}")),
        }
    }
    if text.is_empty() {
        return Err("no answer".to_owned());
    }
    Ok(String::from_utf8_lossy(&text).into_owned())
}

/// A connection that asked for `/health`, was answered and was left open: the server has accepted it and holds it idle.
pub fn idle_connection(addr: SocketAddr) -> TcpStream {
    let mut stream = TcpStream::connect_timeout(&reach(addr), PATIENCE).expect("a connection");
    stream
        .set_read_timeout(Some(Duration::from_millis(200)))
        .expect("a read timeout");
    stream
        .write_all(b"GET /health HTTP/1.1\r\nHost: t\r\n\r\n")
        .expect("a request");
    let mut head = Vec::new();
    let mut chunk = [0_u8; 4096];
    let deadline = Instant::now() + PATIENCE;
    while !head.windows(4).any(|window| window == b"\r\n\r\n") {
        assert!(Instant::now() < deadline, "no answer on the idle connection");
        match stream.read(&mut chunk) {
            Ok(0) => panic!("the server closed the connection before it answered"),
            Ok(count) => head.extend_from_slice(&chunk[..count]),
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(error) => panic!("reading: {error}"),
        }
    }
    stream
}

/// Waits until the server closes `stream`, for at most the patience. The server closes an idle connection when it
/// stops, so this is the sign that a stop has begun.
pub fn closed_by_server(mut stream: TcpStream) -> bool {
    let mut chunk = [0_u8; 4096];
    let deadline = Instant::now() + PATIENCE;
    while Instant::now() < deadline {
        match stream.read(&mut chunk) {
            Ok(0) => return true,
            Ok(_) => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
            Err(_) => return true,
        }
    }
    false
}

pub fn status(answer: &str) -> u16 {
    answer
        .split(' ')
        .nth(1)
        .and_then(|code| code.parse().ok())
        .unwrap_or_else(|| panic!("a status line in {answer:?}"))
}

/// A chat request body for the scripted model.
pub fn chat(stream: bool, max_tokens: u32) -> String {
    format!(
        "{{\"model\":\"{MODEL_ID}\",\"reasoning_effort\":\"none\",\"stream\":{stream},\"max_tokens\":{max_tokens},\
         \"messages\":[{{\"role\":\"user\",\"content\":\"Hi\"}}]}}"
    )
}

/// The value of the gauge `name` on the metrics page.
pub fn gauge(addr: SocketAddr, name: &str) -> Option<f64> {
    let page = exchange(addr, "GET", "/metrics", None);
    page.lines()
        .filter(|line| line.starts_with(name) && !line.starts_with('#'))
        .find_map(|line| line.rsplit(' ').next()?.parse().ok())
}

/// Calls `check` until it holds, for at most the patience.
pub fn until(what: &str, mut check: impl FnMut() -> bool) {
    let deadline = Instant::now() + PATIENCE;
    while !check() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10));
    }
}
