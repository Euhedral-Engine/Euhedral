//! The server's life: start the backend, serve the API, wait for the request to stop, stop in order, and say which
//! exit code the process ends with.
//!
//! The functions here are the whole of `main` apart from the process's edge (arguments, the logger, the signals): the
//! backend comes in through a start function and the stop request through a wait function, so a test drives them with
//! a scripted backend and a closure, and the binary drives them with the engine and the signals.
//!
//! The shutdown order of [`serve`]:
//! 1. The wait function returns: the stop was requested.
//! 2. [`ApiServer::stop`] stops accepting at once, then admission refuses the requests that wait with 503 "shutting
//!    down", interrupts the running generations with "Generation was interrupted by server shutdown." and waits for
//!    them up to `server.shutdown-timeout`, after which it tells them to give up.
//! 3. The server is released (its sink, its timers and its sockets' engine are completed).
//! 4. The backend's own shutdown (`Started::shutdown`) closes the engine and its lattice.
//!
//! This crate is a binary: its main thread may block, and does, only to wait for the stop request and for the server's
//! report that it stopped.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a binary: the main thread waits for the stop request and for the server's report that it stopped"
)]

mod logger;
#[cfg(feature = "scripted")]
pub mod scripted;
pub mod signals;

use std::ffi::OsString;
use std::fmt::{self, Debug, Formatter};
use std::io;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::sync::mpsc::channel;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_inference_api_server::{ApiConfig, Backend, Limits, RequestDefaults, once, start};
use euhedral_inference_settings::Settings;

pub use euhedral_inference_api_server::ApiServer;
pub use logger::install_logger;

/// The exit code when the settings cannot be used.
pub const EXIT_INVALID_SETTINGS: u8 = 2;
/// The exit code when the backend or the server did not start.
pub const EXIT_START_FAILED: u8 = 1;
/// The exit code of a stop by SIGTERM (128 plus the signal's number), and by SIGINT.
pub const EXIT_SIGTERM: u8 = 143;
pub const EXIT_SIGINT: u8 = 130;

const SETTINGS_HELP: &str = "The settings and their environment variables are listed in docs/OPERATIONS.md.";

/// A started backend and what stops it.
#[must_use = "the backend is running: its shutdown must run"]
pub struct Started {
    /// The lattice the server's frames run on.
    pub lattice: Arc<ControlPlaneLattice>,
    /// What the generating routes use.
    pub backend: Arc<dyn Backend>,
    /// Stops the backend and closes the lattice, once the server is released. Runs on the main thread.
    pub shutdown: Box<dyn FnOnce() -> Result<(), String>>,
}

impl Debug for Started {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Started").finish_non_exhaustive()
    }
}

/// Why the backend did not start: the step that failed and the messages of its causes, outermost first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StartError {
    /// What was being done, such as `Loading the model`.
    pub step: String,
    pub causes: Vec<String>,
    /// The device is what failed (no driver, no card, no memory): the report then names what the engine needs.
    pub device: bool,
}

impl StartError {
    /// A failed `step` for the reasons `causes`.
    pub fn new(step: impl Into<String>, causes: impl IntoIterator<Item = impl Into<String>>) -> StartError {
        StartError {
            step: step.into(),
            causes: causes.into_iter().map(Into::into).collect(),
            device: false,
        }
    }

    /// The text printed on the error stream: what failed, in the messages of its causes, and what to change.
    pub fn report(&self) -> String {
        let action = if self.device {
            format!(
                "The engine needs the NVIDIA driver, the CUDA 13 runtime libraries, and a Blackwell GPU (compute \
                 capability 12.x) this process can see: check nvidia-smi, and in a container pass the GPU \
                 (docker run --gpus all). {SETTINGS_HELP}"
            )
        } else {
            format!("Correct what the message names. {SETTINGS_HELP}")
        };
        failure_report(
            &format!("{} failed:\n\n    {}", self.step, self.causes.join("\n    caused by: ")),
            &action,
        )
    }
}

fn failure_report(description: &str, action: &str) -> String {
    format!(
        "\n***************************\nAPPLICATION FAILED TO START\n***************************\n\nDescription:\n\n\
         {description}\n\nAction:\n\n{action}\n"
    )
}

/// The report of a server that could not listen on `port`.
fn bind_report(port: u16, error: &io::Error) -> String {
    if error.kind() == io::ErrorKind::AddrInUse {
        failure_report(
            &format!("Web server failed to start. Port {port} was already in use."),
            &format!(
                "Identify and stop the process that's listening on port {port} or configure this application to \
                 listen on another port."
            ),
        )
    } else {
        failure_report(
            &format!("Web server failed to start on port {port}:\n\n    {error}"),
            &format!("Correct what the message names. {SETTINGS_HELP}"),
        )
    }
}

/// The limits of the server the settings state. The drain window is zero: a stop interrupts the requests in flight at
/// once, and `server.shutdown-timeout` bounds the wait for the interrupted generations.
pub fn limits(settings: &Settings) -> Limits {
    let requests = &settings.server.requests;
    Limits {
        max_body_bytes: usize::try_from(requests.max_body_size.bytes()).unwrap_or(usize::MAX),
        max_queued: usize::try_from(requests.max_queued).unwrap_or(usize::MAX),
        request_limit: requests.timeout,
        shutdown_limit: settings.server.shutdown_timeout,
        drain_limit: Duration::ZERO,
        ..Limits::default()
    }
}

/// What a request that states nothing takes, from the settings.
pub fn defaults(settings: &Settings) -> RequestDefaults {
    RequestDefaults {
        temperature: settings.sampling.temperature,
        top_k: i32::try_from(settings.sampling.top_k).unwrap_or(i32::MAX),
        top_p: settings.sampling.top_p,
        max_tokens: i32::try_from(settings.server.requests.default_max_tokens).unwrap_or(i32::MAX),
    }
}

/// The name of the signal a stop exit code stands for (128 plus its number), for the log.
pub fn signal_name(code: u8) -> String {
    match code {
        EXIT_SIGTERM => "SIGTERM".to_owned(),
        EXIT_SIGINT => "SIGINT".to_owned(),
        other => format!("signal {}", other.saturating_sub(128)),
    }
}

/// Starts the backend with `start`, serves the API on `settings.server.port`, calls `wait_for_stop` with the address
/// the server listens on and, when it returns, stops in order (see the crate documentation). Returns the process's exit
/// code: what `wait_for_stop` returned after a stop, [`EXIT_START_FAILED`] when the backend or the server did not start
/// (the report is on the error stream).
///
/// `wait_for_stop` is where the caller installs its signals and blocks until the first one; it returns the exit code
/// of that stop (128 plus the signal's number).
pub fn serve(
    settings: &Settings,
    start_backend: impl FnOnce(&Settings) -> Result<Started, StartError>,
    wait_for_stop: impl FnOnce(SocketAddr) -> u8,
) -> u8 {
    let started = match start_backend(settings) {
        Ok(started) => started,
        Err(error) => {
            report(&error.report());
            return EXIT_START_FAILED;
        }
    };
    let Started {
        lattice,
        backend,
        shutdown,
    } = started;
    let port = settings.server.port;
    let model = backend.model_id().to_owned();
    let config = ApiConfig {
        address: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port),
        limits: limits(settings),
        backend,
        created_at: now_seconds(),
        defaults: defaults(settings),
        io: None,
    };
    let drain = config.limits.drain_limit;
    let patience = drain.saturating_add(config.limits.shutdown_limit);
    let server = match start(&lattice, config) {
        Ok(server) => server,
        Err(error) => {
            report(&bind_report(port, &error));
            stop_backend(shutdown);
            return EXIT_START_FAILED;
        }
    };
    log::info!(
        "Listening on port {}, serving model {model}",
        server.local_addr().port()
    );
    let code = wait_for_stop(server.local_addr());
    log::info!("{}: shutting down", signal_name(code));
    let (stopped, reported) = channel();
    server.stop(once(move || {
        stopped.send(()).ok();
    }));
    // Admission waits for the generations up to the shutdown limit and then tells them to give up; the margin lets its
    // own timer report before this wait gives up first.
    match reported.recv_timeout(patience.saturating_add(patience.max(Duration::from_secs(1)))) {
        Ok(()) => server.release(),
        Err(_) => log::error!("the server did not report that it stopped; closing the backend anyway"),
    }
    stop_backend(shutdown);
    log::info!("Stopped");
    code
}

fn stop_backend(shutdown: Box<dyn FnOnce() -> Result<(), String>>) {
    if let Err(error) = shutdown() {
        log::error!("closing the backend failed: {error}");
    }
}

fn now_seconds() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| i64::try_from(elapsed.as_secs()).unwrap_or(i64::MAX))
}

/// Prints `text` on the error stream.
#[allow(
    clippy::print_stderr,
    reason = "the report of a failed start is the process's output"
)]
fn report(text: &str) {
    eprintln!("{text}");
}

/// The settings the file named by `--config` (or `--config=`, else `EUHEDRAL_CONFIG`) and the environment give; on
/// failure every problem, each as one line.
pub fn load_settings(args: &[OsString], env: &dyn Fn(&str) -> Option<String>) -> Result<Settings, Vec<String>> {
    euhedral_inference_settings::load(args, env)
        .map_err(|errors| errors.0.iter().map(|problem| problem.message().to_owned()).collect())
}

/// The arguments of `args` that are neither `--config` with its value nor `--config=...`.
pub fn ignored_arguments(args: &[OsString]) -> Vec<String> {
    let mut ignored = Vec::new();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if arg == "--config" {
            rest.next();
        } else if !arg
            .to_str()
            .is_some_and(|text| text.starts_with("--config=") || text.starts_with("--scripted-backend"))
        {
            ignored.push(arg.to_string_lossy().into_owned());
        }
    }
    ignored
}
