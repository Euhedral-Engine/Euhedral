//! The server binary on the real engine: it starts, answers `/health` and a chat request, and stops with 143 on
//! SIGTERM. Ignored by default (it needs the card and the model artifact): run it with `--ignored` under a shared
//! device grant, with `euhedral.cuda.library` and `euhedral.qwen.rco-artifact` set.
#![cfg(unix)]
#![allow(
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the test thread spawns a process and polls it with a bound"
)]

mod common;

use std::fs::write;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use common::{status, try_exchange};

const SERVER: &str = env!("CARGO_BIN_EXE_euhedral-server");
/// The longest the engine may take to load, and one answer to come.
const LOAD: Duration = Duration::from_secs(9 * 60);
/// The settings' stop window (`server.shutdown-timeout`) and the margin the process gets to exit within it.
const STOP: Duration = Duration::from_secs(20);

#[test]
#[ignore = "needs the card, the CUDA library and the model artifact"]
fn the_server_serves_the_engine_and_stops_on_sigterm() {
    let Some(library) = test_support::require_property("euhedral.cuda.library") else {
        return;
    };
    let Some(artifact) = test_support::require_property("euhedral.qwen.rco-artifact") else {
        return;
    };
    // The native root holds `lib/<library>`.
    let root = PathBuf::from(library)
        .parent()
        .and_then(|lib| lib.parent())
        .expect("a native root")
        .to_path_buf();
    let port = TcpListener::bind("127.0.0.1:0")
        .expect("a port")
        .local_addr()
        .expect("an address")
        .port();
    let directory = tempfile::tempdir().expect("a directory");
    let file = directory.path().join("config.yaml");
    write(
        &file,
        format!(
            "model:\n  artifact: {artifact}\n  name: m\nserver:\n  port: {port}\n  shutdown-timeout: {}s\nhost:\n  \
             native-cache: {}\n",
            STOP.as_secs(),
            root.display()
        ),
    )
    .expect("a file");
    let mut child = Command::new(SERVER)
        .args(["--config", file.to_str().expect("a path")])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("the server binary");
    let addr = SocketAddr::from(([127, 0, 0, 1], port));

    // The port opens once the engine has loaded; a connection in the first moments can be reset, so retry.
    let deadline = Instant::now() + LOAD;
    loop {
        if try_exchange(addr, "GET", "/health", None).is_ok_and(|answer| status(&answer) == 200) {
            break;
        }
        if let Some(code) = child.try_wait().expect("a status") {
            panic!("the server exited with {code} before it served");
        }
        if Instant::now() > deadline {
            child.kill().ok();
            child.wait().ok();
            panic!("the server did not serve within {LOAD:?}");
        }
        sleep(Duration::from_millis(500));
    }

    let body = "{\"model\":\"m\",\"reasoning_effort\":\"none\",\"max_tokens\":24,\"messages\":[{\"role\":\"user\",\
                \"content\":\"What is the capital of France? Answer in one word.\"}]}";
    let answer = try_exchange(addr, "POST", "/v1/chat/completions", Some(body));
    let stop = Command::new("kill")
        .arg("-TERM")
        .arg(child.id().to_string())
        .status()
        .expect("kill");
    assert!(stop.success());
    let stopped_at = Instant::now();
    let code = loop {
        if let Some(status) = child.try_wait().expect("a status") {
            break status.code();
        }
        if stopped_at.elapsed() > STOP * 2 {
            child.kill().ok();
            child.wait().ok();
            panic!("the server did not exit within the stop window");
        }
        sleep(Duration::from_millis(50));
    };
    let answer = answer.expect("a chat answer");
    assert_eq!(status(&answer), 200, "{answer}");
    let content = answer.split("\"content\":\"").nth(1).expect("a content field");
    assert!(!content.starts_with('"'), "the answer is empty: {answer}");
    assert_eq!(code, Some(143));
}
