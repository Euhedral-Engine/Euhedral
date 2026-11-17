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

/// Waits until the server answers `/health`: a connection made in the first moments after the port opens is
/// sometimes reset, and the test must not fail on that.
pub fn ready(addr: SocketAddr) {
    until("the server to answer", || {
        try_exchange(addr, "GET", "/health", None).is_ok_and(|answer| status(&answer) == 200)
    });
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
