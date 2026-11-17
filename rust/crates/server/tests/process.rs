//! The built binary as a process: the exit codes and what it prints, and (on Unix) the stop by signal.
#![allow(
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the test thread spawns a process and polls it with a bound"
)]

mod common;

use std::fs::write;
use std::io::Read;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio, id};
use std::sync::atomic::{AtomicU16, Ordering};
use std::thread::{sleep, spawn as spawn_thread};
use std::time::{Duration, Instant};

use common::{PATIENCE, chat, exchange, gauge, status, try_exchange, until};

const SERVER: &str = env!("CARGO_BIN_EXE_euhedral-server");

/// A child that is killed when the test ends, whatever happened.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        self.0.kill().ok();
        self.0.wait().ok();
    }
}

/// A port nothing listens on, below the range the system hands out to the sockets of clients: a port from that range
/// can be taken by a client of another test between the choice and the server's bind, or connect to itself.
fn free_port() -> u16 {
    static NEXT: AtomicU16 = AtomicU16::new(0);
    let start = u16::try_from(id() % 2000).expect("a small number") * 5;
    loop {
        let port = 20_000 + (start + NEXT.fetch_add(7, Ordering::Relaxed)) % 10_000;
        if TcpListener::bind(("0.0.0.0", port)).is_ok() {
            return port;
        }
    }
}

fn config(directory: &Path, port: u16, extra: &str) -> PathBuf {
    let file = directory.join("config.yaml");
    write(
        &file,
        format!(
            "model:\n  artifact: /m.edrl\n  name: n\nserver:\n  port: {port}\n  requests:\n    max-queued: 1\n{extra}"
        ),
    )
    .expect("a file");
    file
}

fn spawn(arguments: &[&str]) -> Server {
    let child = Command::new(SERVER)
        .args(arguments)
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the server binary");
    Server(child)
}

/// The exit code of the child, waiting at most `limit`.
fn exit_code(server: &mut Server, limit: Duration) -> i32 {
    let deadline = Instant::now() + limit;
    loop {
        if let Some(status) = server.0.try_wait().expect("a status") {
            return status.code().expect("an exit code");
        }
        assert!(Instant::now() < deadline, "the process did not exit within {limit:?}");
        sleep(Duration::from_millis(10));
    }
}

fn stderr_of(server: &mut Server) -> String {
    let mut text = String::new();
    server
        .0
        .stderr
        .take()
        .expect("a piped stderr")
        .read_to_string(&mut text)
        .expect("the error stream");
    text
}

#[test]
fn a_bad_file_exits_with_two_and_lists_every_problem_on_its_own_line() {
    let directory = tempfile::tempdir().expect("a directory");
    let file = directory.path().join("config.yaml");
    write(
        &file,
        "model:\n  artifact: /m.edrl\n  name: n\nserver:\n  port: 0\n  colour: red\ncontext:\n  max-tokens: many\n",
    )
    .expect("a file");
    let mut server = spawn(&["--config", file.to_str().expect("a path")]);
    assert_eq!(exit_code(&mut server, PATIENCE), 2);
    let text = stderr_of(&mut server);
    assert_eq!(text.lines().count(), 3, "{text}");
    for key in ["server.port", "server.colour", "context.max-tokens"] {
        assert!(text.contains(key), "{key}: {text}");
    }
}

#[test]
fn settings_that_name_no_model_exit_with_two() {
    let mut server = spawn(&[]);
    assert_eq!(exit_code(&mut server, PATIENCE), 2);
    assert!(stderr_of(&mut server).contains("model.artifact"));
}

#[test]
fn a_backend_that_does_not_start_exits_with_one_and_says_why() {
    let directory = tempfile::tempdir().expect("a directory");
    let file = config(directory.path(), free_port(), "");
    let mut server = spawn(&["--config", file.to_str().expect("a path")]);
    assert_eq!(exit_code(&mut server, PATIENCE), 1);
    let text = stderr_of(&mut server);
    assert!(text.contains("APPLICATION FAILED TO START"), "{text}");
    assert!(text.contains("Starting the engine failed"), "{text}");
}

#[test]
fn a_port_in_use_exits_with_one_and_names_the_port() {
    let taken = TcpListener::bind("0.0.0.0:0").expect("a listener");
    let port = taken.local_addr().expect("an address").port();
    let directory = tempfile::tempdir().expect("a directory");
    let file = config(directory.path(), port, "");
    let mut server = spawn(&["--config", file.to_str().expect("a path"), "--scripted-backend"]);
    assert_eq!(exit_code(&mut server, PATIENCE), 1);
    let text = stderr_of(&mut server);
    assert!(text.contains(&format!("Port {port} was already in use.")), "{text}");
}

/// Starts a scripted server and waits until it answers.
fn serving(directory: &Path, script: &str, extra: &str) -> (Server, SocketAddr) {
    let port = free_port();
    let file = config(directory, port, extra);
    let server = spawn(&[
        "--config",
        file.to_str().expect("a path"),
        &format!("--scripted-backend={script}"),
    ]);
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    // A connection made in the first moments after the port opens is sometimes reset: ask until the server answers.
    until("the server to answer", || {
        try_exchange(addr, "GET", "/health", None).is_ok_and(|answer| status(&answer) == 200)
    });
    (server, addr)
}

#[cfg(unix)]
fn signal(server: &Server, name: &str) {
    let sent = Command::new("kill")
        .arg(format!("-{name}"))
        .arg(server.0.id().to_string())
        .status()
        .expect("kill");
    assert!(sent.success());
}

#[cfg(unix)]
#[test]
fn sigterm_stops_a_serving_process_with_143() {
    let directory = tempfile::tempdir().expect("a directory");
    let (mut server, _) = serving(directory.path(), "hello", "");
    signal(&server, "TERM");
    assert_eq!(exit_code(&mut server, PATIENCE), 143);
}

#[cfg(unix)]
#[test]
fn sigint_stops_a_serving_process_with_130() {
    let directory = tempfile::tempdir().expect("a directory");
    let (mut server, _) = serving(directory.path(), "hello", "");
    signal(&server, "INT");
    assert_eq!(exit_code(&mut server, PATIENCE), 130);
}

#[cfg(unix)]
#[test]
fn a_second_signal_ends_the_process_at_once_with_the_code_of_the_signal() {
    let directory = tempfile::tempdir().expect("a directory");
    // A generation that never ends keeps the stop waiting for the whole shutdown timeout.
    let (mut server, addr) = serving(directory.path(), "hung", "  shutdown-timeout: 60s\n");
    spawn_thread(move || exchange(addr, "POST", "/v1/chat/completions", Some(&chat(true, 10))));
    until("the request to run", || {
        gauge(addr, "euhedral_generations_active") == Some(1.0)
    });
    signal(&server, "TERM");
    // The server stopped accepting: the first signal was taken, and the stop is waiting for the generation.
    until("the listener to close", || {
        TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_err()
    });
    assert!(
        server.0.try_wait().expect("a status").is_none(),
        "the stop ended by itself"
    );
    signal(&server, "INT");
    assert_eq!(exit_code(&mut server, Duration::from_secs(10)), 130);
}

#[cfg(unix)]
#[test]
fn a_stop_by_signal_interrupts_a_running_request_and_refuses_the_queued_one() {
    let directory = tempfile::tempdir().expect("a directory");
    let (mut server, addr) = serving(directory.path(), "hung", "  shutdown-timeout: 2s\n");
    let running = spawn_thread(move || exchange(addr, "POST", "/v1/chat/completions", Some(&chat(true, 10))));
    until("the request to run", || {
        gauge(addr, "euhedral_generations_active") == Some(1.0)
    });
    let queued = spawn_thread(move || exchange(addr, "POST", "/v1/chat/completions", Some(&chat(false, 10))));
    until("the request to wait", || {
        gauge(addr, "euhedral_generations_queued") == Some(1.0)
    });
    signal(&server, "TERM");
    assert_eq!(exit_code(&mut server, PATIENCE), 143);
    let queued = queued.join().expect("a client");
    assert_eq!(status(&queued), 503, "{queued}");
    assert!(queued.contains("shutting down"), "{queued}");
    let running = running.join().expect("a client");
    assert!(running.contains("interrupted by server shutdown"), "{running}");
}
