//! The server's life through [`serve`], with a scripted backend and a closure for the stop request: what the
//! application test and the shutdown test of the Java server assert. Each test starts a lattice of its own.
#![allow(
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the test thread is not a lattice worker; it waits with a bound"
)]

mod common;

use std::cell::Cell;
use std::fs::write;
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::mpsc::channel;
use std::thread::spawn;
use std::time::{Duration, Instant};

use common::{PATIENCE, chat, exchange, gauge, status, until};
use euhedral_inference_api_server::scripted::{MODEL_ID, Script};
use euhedral_inference_server::scripted::{start, start_with};
use euhedral_inference_server::{
    EXIT_SIGINT, EXIT_SIGTERM, EXIT_START_FAILED, StartError, defaults, limits, load_settings, serve,
};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

const CHAT: &str = "/v1/chat/completions";

fn settings(shutdown_timeout: Duration) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: "/m.edrl".into(),
        name: "n".to_owned(),
    });
    settings.server.port = 0;
    settings.server.shutdown_timeout = shutdown_timeout;
    settings.server.requests.max_queued = 1;
    settings
}

#[test]
fn the_routes_answer_and_the_model_is_listed() {
    let code = serve(&settings(Duration::from_secs(5)), start, |addr| {
        let health = exchange(addr, "GET", "/health", None);
        assert_eq!(status(&health), 200, "{health}");
        let models = exchange(addr, "GET", "/v1/models", None);
        assert_eq!(status(&models), 200, "{models}");
        assert!(models.contains(MODEL_ID), "{models}");
        let reply = exchange(addr, "POST", CHAT, Some(&chat(false, 20)));
        assert_eq!(status(&reply), 200, "{reply}");
        assert!(reply.contains("Hello, world"), "{reply}");
        EXIT_SIGINT
    });
    assert_eq!(code, EXIT_SIGINT);
}

#[test]
fn a_stop_with_nothing_running_returns_promptly() {
    let started = Instant::now();
    let code = serve(&settings(Duration::from_secs(30)), start, |_| EXIT_SIGTERM);
    assert_eq!(code, EXIT_SIGTERM);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "stopping took {:?}",
        started.elapsed()
    );
}

#[test]
fn a_stop_refuses_the_queued_request_and_interrupts_the_running_one_within_the_shutdown_timeout() {
    let shutdown_timeout = Duration::from_secs(2);
    let (clients, answers) = channel();
    let started = Instant::now();
    let code = serve(
        &settings(shutdown_timeout),
        |_| start_with(Script::Hung),
        |addr| {
            let running = clients.clone();
            spawn(move || {
                running
                    .send(("running", exchange(addr, "POST", CHAT, Some(&chat(true, 3000)))))
                    .ok()
            });
            until("the first request to run", || {
                gauge(addr, "euhedral_generations_active") == Some(1.0)
            });
            let queued = clients.clone();
            spawn(move || {
                queued
                    .send(("queued", exchange(addr, "POST", CHAT, Some(&chat(false, 3000)))))
                    .ok()
            });
            until("the second request to wait", || {
                gauge(addr, "euhedral_generations_queued") == Some(1.0)
            });
            EXIT_SIGTERM
        },
    );
    assert_eq!(code, EXIT_SIGTERM);
    // The generation never ends, so the stop waits for it up to the shutdown timeout and no longer.
    let took = started.elapsed();
    assert!(
        took < shutdown_timeout + Duration::from_secs(10),
        "stopping took {took:?}"
    );
    drop(clients);
    let mut running = String::new();
    let mut queued = String::new();
    while let Ok((which, answer)) = answers.recv_timeout(PATIENCE) {
        if which == "running" {
            running = answer;
        } else {
            queued = answer;
        }
        if !running.is_empty() && !queued.is_empty() {
            break;
        }
    }
    assert_eq!(status(&queued), 503, "{queued}");
    assert!(queued.contains("shutting down"), "{queued}");
    assert_eq!(status(&running), 200, "{running}");
    assert!(running.contains("interrupted by server shutdown"), "{running}");
}

#[test]
fn a_stop_with_a_generation_that_never_ends_returns_after_the_shutdown_timeout() {
    let shutdown_timeout = Duration::from_secs(1);
    let started = Instant::now();
    let code = serve(
        &settings(shutdown_timeout),
        |_| start_with(Script::Hung),
        |addr| {
            spawn(move || exchange(addr, "POST", CHAT, Some(&chat(true, 10))));
            until("the request to run", || {
                gauge(addr, "euhedral_generations_active") == Some(1.0)
            });
            EXIT_SIGTERM
        },
    );
    assert_eq!(code, EXIT_SIGTERM);
    let took = started.elapsed();
    assert!(took >= shutdown_timeout, "stopping took only {took:?}");
    assert!(took < PATIENCE, "stopping took {took:?}");
}

#[test]
fn a_backend_that_does_not_start_is_a_failed_start_and_nothing_waits_for_a_stop() {
    let code = serve(
        &settings(Duration::from_secs(5)),
        |_| Err(StartError::new("Loading the model", ["no such file"])),
        |_| panic!("nothing is serving"),
    );
    assert_eq!(code, EXIT_START_FAILED);
}

#[test]
fn a_port_in_use_is_a_failed_start_and_the_backend_is_shut_down() {
    let taken = TcpListener::bind("0.0.0.0:0").expect("a listener");
    let mut settings = settings(Duration::from_secs(5));
    settings.server.port = taken.local_addr().expect("an address").port();
    let closed = Rc::new(Cell::new(false));
    let seen = closed.clone();
    let code = serve(
        &settings,
        move |settings| {
            let mut started = start(settings)?;
            let shutdown = started.shutdown;
            started.shutdown = Box::new(move || {
                seen.set(true);
                shutdown()
            });
            Ok(started)
        },
        |_| panic!("nothing is serving"),
    );
    assert_eq!(code, EXIT_START_FAILED);
    assert!(closed.get(), "the backend was left running");
}

#[test]
fn the_report_of_a_failed_start_names_the_step_the_causes_and_the_action() {
    let mut error = StartError::new("Loading the model", ["cannot open", "no such file"]);
    let report = error.report();
    assert!(report.contains("APPLICATION FAILED TO START"), "{report}");
    assert!(
        report.contains("Loading the model failed:\n\n    cannot open\n    caused by: no such file"),
        "{report}"
    );
    assert!(
        report.contains("Action:\n\nCorrect what the message names."),
        "{report}"
    );
    error.device = true;
    assert!(error.report().contains("nvidia-smi"));
}

#[test]
fn the_limits_and_defaults_come_from_the_settings() {
    let mut settings = settings(Duration::from_secs(7));
    settings.server.requests.max_body_size = ByteSize(2048);
    settings.server.requests.timeout = Duration::from_secs(300);
    settings.server.requests.default_max_tokens = 99;
    let limits = limits(&settings);
    assert_eq!(limits.max_body_bytes, 2048);
    assert_eq!(limits.max_queued, 1);
    assert_eq!(limits.request_limit, Duration::from_secs(300));
    assert_eq!(limits.shutdown_limit, Duration::from_secs(7));
    assert_eq!(limits.drain_limit, Duration::ZERO);
    assert_eq!(defaults(&settings).max_tokens, 99);
}

#[test]
fn a_bad_file_lists_every_problem_on_its_own_line() {
    let directory = tempfile::tempdir().expect("a directory");
    let file = directory.path().join("config.yaml");
    write(
        &file,
        "model:\n  artifact: /m.edrl\n  name: n\nserver:\n  port: 0\n  colour: red\ncontext:\n  max-tokens: many\n",
    )
    .expect("a file");
    let problems = load_settings(&["--config".into(), file.into_os_string()], &|_| None).expect_err("a bad file");
    assert_eq!(problems.len(), 3, "{problems:?}");
    for key in ["server.port", "server.colour", "context.max-tokens"] {
        assert!(problems.iter().any(|line| line.contains(key)), "{key}: {problems:?}");
    }
    let missing = load_settings(&[], &|_| None).expect_err("no model");
    assert!(
        missing.iter().any(|line| line.contains("model.artifact")),
        "{missing:?}"
    );
}
