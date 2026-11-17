//! The engine as the server's backend: where its CUDA library is, how long its start may take, and how its start
//! errors read.

use std::env::current_exe;
use std::path::{Path, PathBuf};
use std::time::Duration;

use euhedral_inference_api_server::engine_backend::start;
use euhedral_inference_api_server::{EngineHost, StartError as EngineStartError};
use euhedral_inference_settings::{Auto, Settings};

use crate::{StartError, Started};

/// What the engine's start waits for the engine to load: no limit in practice, as the Java server's start has none (the
/// operator stops a start that hangs). A bounded value because the wait takes a duration.
pub const START_LIMIT: Duration = Duration::from_secs(u32::MAX as u64);

/// The file name of the CUDA library on this platform.
fn library_name() -> &'static str {
    if cfg!(windows) {
        "euhedral_cuda.dll"
    } else {
        "libeuhedral_cuda.so"
    }
}

/// The CUDA library: `lib/` under `host.native-cache` when that is set, else `lib/` beside the executable. Both are
/// the layout the Java server unpacks its native files into (`lib/`, `share/` and `cuda/` under one root).
pub fn library_path(settings: &Settings, executable: Option<&Path>) -> PathBuf {
    let root = match &settings.host.native_cache {
        Auto::Value(root) => root.clone(),
        Auto::Auto => executable
            .and_then(Path::parent)
            .map_or_else(PathBuf::new, Path::to_path_buf),
    };
    root.join("lib").join(library_name())
}

/// The server's start error for an engine start error: the messages are the engine's.
fn start_error(error: &EngineStartError) -> StartError {
    let mut mapped = StartError::new("Starting the engine", [error.to_string()]);
    mapped.device = matches!(error, EngineStartError::Driver(_) | EngineStartError::Engine(_));
    mapped
}

/// Starts the engine of `settings` and waits for it. The CUDA library is checked first, so a missing library reports
/// the driver message before the artifact is read.
pub fn start_engine(settings: &Settings) -> Result<Started, StartError> {
    let library = library_path(settings, current_exe().ok().as_deref());
    if !library.is_file() {
        let missing = EngineStartError::Driver(format!("{} is not there", library.display()));
        return Err(start_error(&missing));
    }
    let host = start(settings, &library, START_LIMIT).map_err(|error| start_error(&error))?;
    Ok(started(host))
}

fn started(host: EngineHost) -> Started {
    Started {
        lattice: host.lattice().clone(),
        backend: host.backend(),
        // The engine's shutdown closes the backend, then the lattice, and returns when the device memory is freed or
        // `server.shutdown-timeout` is spent, saying what it left.
        shutdown: Box::new(move |window| host.shutdown(window).map_err(|error| error.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use euhedral_inference_settings::ModelSettings;

    use super::*;

    fn settings() -> Settings {
        Settings::with_model(ModelSettings {
            artifact: PathBuf::from("m.edrl"),
            name: "n".to_owned(),
        })
    }

    #[test]
    fn the_library_is_under_the_native_cache_when_one_is_set() {
        let mut settings = settings();
        settings.host.native_cache = Auto::Value(PathBuf::from("native"));
        let path = library_path(&settings, Some(Path::new("bin/euhedral-server")));
        assert_eq!(path, Path::new("native").join("lib").join(library_name()));
    }

    #[test]
    fn the_library_is_beside_the_executable_otherwise() {
        let path = library_path(&settings(), Some(Path::new("bin/euhedral-server")));
        assert_eq!(path, Path::new("bin").join("lib").join(library_name()));
    }

    #[test]
    fn a_driver_error_keeps_its_message_and_names_the_device() {
        let error = start_error(&EngineStartError::Driver("no library".to_owned()));
        assert!(error.device);
        assert!(error.causes[0].starts_with("the NVIDIA driver is required"));
    }
}
