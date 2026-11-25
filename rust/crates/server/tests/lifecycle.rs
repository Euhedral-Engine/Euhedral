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
use std::net::TcpListener;
use std::rc::Rc;
use std::sync::mpsc::{Receiver, channel};
use std::thread::spawn;
use std::time::Duration;

use common::{PATIENCE, chat, closed_by_server, exchange, gauge, idle_connection, ready, status, until};
use euhedral_inference_api_server::scripted::{MODEL_ID, Script};
use euhedral_inference_server::scripted::{start, start_scripted, start_with};
use euhedral_inference_server::{
    DRAIN_WINDOW, EXIT_SIGINT, EXIT_SIGTERM, EXIT_START_FAILED, SHUTDOWN_WINDOW, StartError, StopWindows, defaults,
    limits, load_settings, serve,
};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

const CHAT: &str = "/v1/chat/completions";

/// A window no test waits out: a stop that has to wait for it never returns.
const FOR_EVER: Duration = Duration::from_secs(3600);

/// The windows of a stop that waits for the requests in flight not at all, and for the generations `shutdown`.
fn at_once(shutdown: Duration) -> StopWindows {
    StopWindows {
        drain: Duration::ZERO,
        shutdown,
    }
}

/// Runs `body`, which calls `serve`, on a thread of its own and returns what it returned. A `serve` that does not
/// return within the patience is a stop that waits for something that does not end: the test fails, saying so.
fn returns<T: Send + 'static>(body: impl FnOnce() -> T + Send + 'static) -> T {
    let (done, returned) = channel();
    spawn(move || done.send(body()).ok());
    returned
        .recv_timeout(PATIENCE)
        .expect("serve never returned: the stop is waiting for something that does not end")
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
fn a_stop_with_nothing_running_returns_without_waiting_for_the_windows() {
    let windows = StopWindows {
        drain: FOR_EVER,
        shutdown: FOR_EVER,
    };
    let code = returns(move || serve(&settings(), windows, start, |_| EXIT_SIGTERM));
    assert_eq!(code, EXIT_SIGTERM);
}

/// The answers of the clients of a test that ended: the running request's and the queued one's, each when it came.
fn answered(answers: &Receiver<(&'static str, String)>) -> (String, String) {
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
    (running, queued)
}

#[test]
fn a_stop_refuses_the_queued_request_and_interrupts_the_running_one() {
    let (clients, answers) = channel();
    let code = returns({
        let clients = clients.clone();
        move || {
            serve(
                &settings(),
                // The generation never ends, so the stop has to give up on it at the end of the shutdown window.
                StopWindows {
                    drain: Duration::from_millis(1),
                    shutdown: Duration::from_millis(1),
                },
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
                    EXIT_SIGTERM
                },
            )
        }
    });
    assert_eq!(code, EXIT_SIGTERM);
    drop(clients);
    let (running, queued) = answered(&answers);
    assert_eq!(status(&queued), 503, "{queued}");
    assert!(queued.contains("shutting down"), "{queued}");
    assert_eq!(status(&running), 200, "{running}");
    assert!(running.contains("interrupted by server shutdown"), "{running}");
}

#[test]
fn a_stop_with_a_generation_that_never_ends_returns_once_the_shutdown_window_passed() {
    let code = returns(|| {
        serve(
            &settings(),
            at_once(Duration::from_millis(1)),
            |_| start_with(Script::Hung),
            |addr| {
                ready(addr);
                spawn(move || exchange(addr, "POST", CHAT, Some(&chat(true, 10))));
                until("the request to run", || {
                    gauge(addr, "euhedral_generations_active") == Some(1.0)
                });
                EXIT_SIGTERM
            },
        )
    });
    assert_eq!(code, EXIT_SIGTERM);
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
    // The windows are not waited out: the stop returns as soon as both requests finished, and a stop that did not
    // wait for them would answer the queued one with a refusal.
    let windows = StopWindows {
        drain: FOR_EVER,
        shutdown: FOR_EVER,
    };
    let (clients, answers) = channel();
    let generations = returns({
        let clients = clients.clone();
        move || {
            let (started, backend) = start_scripted(Script::Hung).expect("a scripted backend");
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
                    // The running generation ends on its own, once the stop has begun: the stop closes the idle
                    // connection.
                    let running = backend.generations().remove(0);
                    let idle = idle_connection(addr);
                    spawn(move || {
                        assert!(closed_by_server(idle), "the stop did not begin");
                        assert!(running.finish_held_ok(20, false), "the generation was not held");
                    });
                    EXIT_SIGTERM
                },
            );
            assert_eq!(code, EXIT_SIGTERM);
            backend.generations().len()
        }
    });
    drop(clients);
    let (running, queued) = answered(&answers);
    assert_eq!(status(&running), 200, "{running}");
    assert!(running.contains("\"finish_reason\":\"length\""), "{running}");
    assert!(running.contains("[DONE]"), "{running}");
    assert_eq!(status(&queued), 200, "{queued}");
    assert!(queued.contains("Hello, world"), "{queued}");
    assert_eq!(generations, 2);
}
