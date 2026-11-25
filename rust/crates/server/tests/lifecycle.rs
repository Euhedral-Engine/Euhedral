//! The server's life through [`serve`], with a scripted backend and a closure for the stop request: that the routes
//! answer and the model is listed, and how a stop treats the requests in flight. Each test starts a lattice of its own.
#![allow(
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the test thread is not a lattice worker; it waits with a bound"
)]

mod common;

use std::cell::Cell;
use std::fs::write;
use std::net::{TcpListener, TcpStream};
use std::rc::Rc;
use std::sync::mpsc::channel;
use std::thread::spawn;
use std::time::{Duration, Instant};

use common::{PATIENCE, chat, exchange, gauge, reach, ready, status, until};
use euhedral_inference_api_server::scripted::{MODEL_ID, Script};
use euhedral_inference_server::scripted::{start, start_scripted, start_with};
use euhedral_inference_server::{
    DRAIN_WINDOW, EXIT_SIGINT, EXIT_SIGTERM, EXIT_START_FAILED, SHUTDOWN_WINDOW, StartError, StopWindows, defaults,
    limits, load_settings, serve,
};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

const CHAT: &str = "/v1/chat/completions";

/// The windows of a stop that waits for the requests in flight not at all, and for the generations `shutdown`.
fn at_once(shutdown: Duration) -> StopWindows {
    StopWindows {
        drain: Duration::ZERO,
        shutdown,
    }
}

/// What a test's wait for the stop returns when the stop is asked for: the moment is noted, so that a test times the stop
/// and not the start of the backend before it.
fn stop_asked(asked: &Cell<Option<Instant>>) -> u8 {
    asked.set(Some(Instant::now()));
    EXIT_SIGTERM
}

/// How long the stop took, from the moment [`stop_asked`] noted to now.
fn stopped_after(asked: &Cell<Option<Instant>>) -> Duration {
    asked.get().expect("the stop was asked for").elapsed()
}

fn settings() -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: "/m.edrl".into(),
        name: "n".to_owned(),
    });
    settings.server.port = 0;
    settings.server.requests.max_queued = 1;
    settings
}

#[test]
fn the_routes_answer_and_the_model_is_listed() {
    let code = serve(&settings(), StopWindows::default(), start, |addr| {
        ready(addr);
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
    let asked = Rc::new(Cell::new(None));
    let code = serve(&settings(), StopWindows::default(), start, {
        let asked = asked.clone();
        move |_| stop_asked(&asked)
    });
    assert_eq!(code, EXIT_SIGTERM);
    let took = stopped_after(&asked);
    assert!(took < Duration::from_secs(10), "stopping took {took:?}");
}

#[test]
fn a_stop_refuses_the_queued_request_and_interrupts_the_running_one_within_the_shutdown_timeout() {
    let windows = StopWindows {
        drain: Duration::from_millis(300),
        shutdown: Duration::from_secs(2),
    };
    let (clients, answers) = channel();
    let asked = Rc::new(Cell::new(None));
    let code = serve(
        &settings(),
        windows,
        |_| start_with(Script::Hung),
        |addr| {
            ready(addr);
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
            stop_asked(&asked)
        },
    );
    assert_eq!(code, EXIT_SIGTERM);
    // The generation never ends, so the stop waits for it up to the drain and the shutdown windows and no longer.
    let took = stopped_after(&asked);
    assert!(took >= windows.drain + windows.shutdown, "stopping took only {took:?}");
    assert!(
        took < windows.drain + windows.shutdown + Duration::from_secs(10),
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
fn a_stop_with_a_generation_that_never_ends_returns_after_the_shutdown_window() {
    let shutdown_timeout = Duration::from_secs(1);
    let asked = Rc::new(Cell::new(None));
    let code = serve(
        &settings(),
        at_once(shutdown_timeout),
        |_| start_with(Script::Hung),
        |addr| {
            ready(addr);
            spawn(move || exchange(addr, "POST", CHAT, Some(&chat(true, 10))));
            until("the request to run", || {
                gauge(addr, "euhedral_generations_active") == Some(1.0)
            });
            stop_asked(&asked)
        },
    );
    assert_eq!(code, EXIT_SIGTERM);
    let took = stopped_after(&asked);
    assert!(took >= shutdown_timeout, "stopping took only {took:?}");
    assert!(took < PATIENCE, "stopping took {took:?}");
}

#[test]
fn a_backend_that_does_not_start_is_a_failed_start_and_nothing_waits_for_a_stop() {
    let code = serve(
        &settings(),
        StopWindows::default(),
        |_| Err(StartError::new("Loading the model", ["no such file"])),
        |_| panic!("nothing is serving"),
    );
    assert_eq!(code, EXIT_START_FAILED);
}

#[test]
fn a_port_in_use_is_a_failed_start_and_the_backend_is_shut_down() {
    let taken = TcpListener::bind("0.0.0.0:0").expect("a listener");
    let mut settings = settings();
    settings.server.port = taken.local_addr().expect("an address").port();
    let closed = Rc::new(Cell::new(false));
    let seen = closed.clone();
    let code = serve(
        &settings,
        StopWindows::default(),
        move |settings| {
            let mut started = start(settings)?;
            let shutdown = started.shutdown;
            started.shutdown = Box::new(move |timeout| {
                seen.set(true);
                shutdown(timeout)
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
    let mut settings = settings();
    settings.server.shutdown_timeout = Duration::from_secs(7);
    settings.server.requests.max_body_size = ByteSize(2048);
    settings.server.requests.timeout = Duration::from_secs(300);
    settings.server.requests.default_max_tokens = 99;
    let windows = StopWindows {
        drain: Duration::from_secs(3),
        shutdown: Duration::from_secs(4),
    };
    let limits = limits(&settings, windows);
    assert_eq!(limits.max_body_bytes, 2048);
    assert_eq!(limits.max_queued, 1);
    assert_eq!(limits.request_limit, Duration::from_secs(300));
    // `server.shutdown-timeout` is the backend's own, not admission's.
    assert_eq!(limits.shutdown_limit, Duration::from_secs(4));
    assert_eq!(limits.drain_limit, Duration::from_secs(3));
    assert_eq!(StopWindows::default().drain, DRAIN_WINDOW);
    assert_eq!(StopWindows::default().shutdown, SHUTDOWN_WINDOW);
    assert_eq!(DRAIN_WINDOW, Duration::from_secs(30));
    assert_eq!(SHUTDOWN_WINDOW, Duration::from_secs(30));
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

#[test]
fn the_backend_is_shut_down_within_the_shutdown_timeout_of_the_settings() {
    let mut settings = settings();
    settings.server.shutdown_timeout = Duration::from_secs(7);
    let given = Rc::new(Cell::new(None));
    let seen = given.clone();
    let code = serve(
        &settings,
        StopWindows::default(),
        move |settings| {
            let mut started = start(settings)?;
            let shutdown = started.shutdown;
            started.shutdown = Box::new(move |timeout| {
                seen.set(Some(timeout));
                shutdown(timeout)
            });
            Ok(started)
        },
        |_| EXIT_SIGTERM,
    );
    assert_eq!(code, EXIT_SIGTERM);
    assert_eq!(given.get(), Some(Duration::from_secs(7)));
}

#[test]
fn running_and_queued_requests_finish_within_the_drain_window() {
    let windows = StopWindows {
        drain: Duration::from_secs(20),
        shutdown: Duration::from_secs(5),
    };
    let (started, backend) = start_scripted(Script::Hung).expect("a scripted backend");
    let (clients, answers) = channel();
    let begun = Instant::now();
    let code = serve(
        &settings(),
        windows,
        |_| Ok(started),
        |addr| {
            ready(addr);
            let running = clients.clone();
            spawn(move || {
                running
                    .send(("running", exchange(addr, "POST", CHAT, Some(&chat(true, 20)))))
                    .ok()
            });
            until("the first request to run", || {
                gauge(addr, "euhedral_generations_active") == Some(1.0)
            });
            // What the queued request will get once it runs.
            backend.set_script(Script::hello());
            until("the first token", || {
                backend
                    .generations()
                    .first()
                    .is_some_and(|generation| generation.emit_held("tok0 "))
            });
            let queued = clients.clone();
            spawn(move || {
                queued
                    .send(("queued", exchange(addr, "POST", CHAT, Some(&chat(false, 20)))))
                    .ok()
            });
            until("the second request to wait", || {
                gauge(addr, "euhedral_generations_queued") == Some(1.0)
            });
            // The running generation ends on its own, once the stop has begun and no longer accepts.
            let running = backend.generations().remove(0);
            spawn(move || {
                until("the server to stop accepting", || {
                    TcpStream::connect_timeout(&reach(addr), Duration::from_millis(200)).is_err()
                });
                assert!(running.finish_held_ok(20, false), "the generation was not held");
            });
            EXIT_SIGTERM
        },
    );
    assert_eq!(code, EXIT_SIGTERM);
    assert!(
        begun.elapsed() < windows.drain,
        "the stop waited for the whole drain window"
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
    assert_eq!(status(&running), 200, "{running}");
    assert!(running.contains("\"finish_reason\":\"length\""), "{running}");
    assert!(running.contains("[DONE]"), "{running}");
    assert_eq!(status(&queued), 200, "{queued}");
    assert!(queued.contains("Hello, world"), "{queued}");
    assert_eq!(backend.generations().len(), 2);
}
