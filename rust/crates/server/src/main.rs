//! `euhedral-server`: serves the OpenAI-, Anthropic- and Responses-compatible API of one model.
//!
//! ```text
//! euhedral-server [--config FILE]
//! ```
//!
//! The settings come from the file named by `--config` (or `EUHEDRAL_CONFIG`) and the environment. Exit codes: 0 never
//! (the server runs until it is stopped), 1 the backend or the server did not start, 2 the settings are invalid, 128
//! plus the signal's number after a stop by SIGTERM (143) or SIGINT (130). A second signal during the stop ends the
//! process at once with the same code.
//!
//! This file is the process's edge: it reads the arguments and the environment, installs the logger and the signals,
//! and chooses the backend.

use std::env;
#[cfg(feature = "scripted")]
use std::ffi::OsString;
use std::process::ExitCode;
#[cfg(feature = "scripted")]
use std::time::Duration;

#[cfg(feature = "scripted")]
use euhedral_inference_server::scripted::{script_named, start_with};
use euhedral_inference_server::signals::install;
#[cfg(feature = "scripted")]
use euhedral_inference_server::{DRAIN_WINDOW, SHUTDOWN_WINDOW};
use euhedral_inference_server::{
    EXIT_INVALID_SETTINGS, EXIT_START_FAILED, StartError, Started, StopWindows, ignored_arguments, install_logger,
    load_settings, say, serve,
};
use euhedral_inference_settings::{Settings, process_env};

/// The backend this binary serves.
///
/// THE PLACE TO PASS THE ENGINE ADAPTER: until the adapter exists the start function fails, naming why. Replace its
/// body with the adapter's start (it returns a [`Started`]: the lattice, the backend and the engine's shutdown).
fn start_backend(_settings: &Settings) -> Result<Started, StartError> {
    Err(StartError::new(
        "Starting the engine",
        ["this binary has no engine adapter built in"],
    ))
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let settings = match load_settings(&args, &process_env) {
        Ok(settings) => settings,
        Err(problems) => {
            for problem in problems {
                say(&problem);
            }
            return ExitCode::from(EXIT_INVALID_SETTINGS);
        }
    };
    // A failed logger install only loses the log lines.
    if install_logger().is_err() {
        say("the log could not be installed");
    }
    for argument in ignored_arguments(&args) {
        log::warn!("ignoring the argument {argument}");
    }
    // The handlers come before the backend starts: a signal that arrives while it starts is kept, and stops the server
    // in order as soon as it serves.
    let mut signals = match install() {
        Ok(signals) => signals,
        Err(error) => {
            say(&format!("the signal handlers could not be installed: {error}"));
            return ExitCode::from(EXIT_START_FAILED);
        }
    };
    let wait = move |_| signals.wait();
    #[cfg(feature = "scripted")]
    if let Some(name) = scripted_name(&args) {
        let Some(script) = script_named(&name) else {
            say(&format!("unknown scripted backend {name}"));
            return ExitCode::from(EXIT_INVALID_SETTINGS);
        };
        let windows = StopWindows {
            drain: scripted_millis(&args, "--scripted-drain-ms=").unwrap_or(DRAIN_WINDOW),
            shutdown: scripted_millis(&args, "--scripted-shutdown-ms=").unwrap_or(SHUTDOWN_WINDOW),
        };
        return ExitCode::from(serve(&settings, windows, |_| start_with(script), wait));
    }
    ExitCode::from(serve(&settings, StopWindows::default(), start_backend, wait))
}

/// The duration in milliseconds that `--scripted-...-ms=N` states, if the argument is there.
#[cfg(feature = "scripted")]
fn scripted_millis(args: &[OsString], prefix: &str) -> Option<Duration> {
    args.iter()
        .find_map(|argument| argument.to_str()?.strip_prefix(prefix)?.parse().ok())
        .map(Duration::from_millis)
}

/// The script named by `--scripted-backend` or `--scripted-backend=NAME` (`hello` when no name is given).
#[cfg(feature = "scripted")]
fn scripted_name(args: &[OsString]) -> Option<String> {
    args.iter().find_map(|argument| {
        let text = argument.to_str()?;
        if text == "--scripted-backend" {
            Some("hello".to_owned())
        } else {
            text.strip_prefix("--scripted-backend=").map(str::to_owned)
        }
    })
}
