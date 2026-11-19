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

/// The places the CUDA library may be, in order: `lib/` under `host.native-cache` when that is set; else `lib/` beside
/// the executable and, when the executable's directory is named `bin` and `share/euhedral_cuda` is a directory beside
/// it (the bundle: `bin/euhedral-server` with `lib/` beside `bin/`), `lib/` beside that directory. These are the layout
/// the Java server unpacks its native files into (`lib/`, `share/` and `cuda/` under one root).
pub fn library_candidates(settings: &Settings, executable: Option<&Path>) -> Vec<PathBuf> {
    let library = |root: &Path| root.join("lib").join(library_name());
    if let Auto::Value(root) = &settings.host.native_cache {
        return vec![library(root)];
    }
    let directory = executable
        .and_then(Path::parent)
        .map_or_else(PathBuf::new, Path::to_path_buf);
    let mut candidates = vec![library(&directory)];
    if directory.file_name().is_some_and(|name| name == "bin")
        && let Some(root) = directory.parent()
        && root.join("share").join("euhedral_cuda").is_dir()
    {
        candidates.push(library(root));
    }
    candidates
}

/// The CUDA library: the first of [`library_candidates`] that is a file, else the first of them.
pub fn library_path(settings: &Settings, executable: Option<&Path>) -> PathBuf {
    let candidates = library_candidates(settings, executable);
    candidates
        .iter()
        .find(|candidate| candidate.is_file())
        .or_else(|| candidates.first())
        .cloned()
        .unwrap_or_default()
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
    let executable = current_exe().ok();
    let library = library_path(settings, executable.as_deref());
    if !library.is_file() {
        let tried: Vec<String> = library_candidates(settings, executable.as_deref())
            .iter()
            .map(|candidate| candidate.display().to_string())
            .collect();
        let missing = EngineStartError::Driver(format!("{} is not there", tried.join(" or ")));
        return Err(start_error(&missing));
    }
    let host = start(settings, &library, START_LIMIT, Some(settings.server.shutdown_timeout))
        .map_err(|error| start_error(&error))?;
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
    use std::env::{temp_dir, var_os};
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::process::id;

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
    fn a_bundled_layout_finds_the_library_beside_bin_without_a_native_cache() -> io::Result<()> {
        let base = var_os("EUHEDRAL_SCRATCH").map_or_else(temp_dir, PathBuf::from);
        let bundle = base.join(format!("engine-bundle-{}", id()));
        let lib = bundle.join("lib");
        fs::create_dir_all(&lib)?;
        fs::create_dir_all(bundle.join("bin"))?;
        fs::write(lib.join(library_name()), "stand-in")?;
        // Without `share/euhedral_cuda` the directory above `bin` is not a bundle (`/usr/bin` never picks `/usr/lib`).
        let path = library_path(&settings(), Some(&bundle.join("bin").join("euhedral-server")));
        assert_eq!(path, bundle.join("bin/lib").join(library_name()));
        fs::create_dir_all(bundle.join("share/euhedral_cuda"))?;
        // Beside a directory not named `bin` it is not a bundle either.
        fs::create_dir_all(bundle.join("sbin"))?;
        let path = library_path(&settings(), Some(&bundle.join("sbin").join("euhedral-server")));
        assert_eq!(path, bundle.join("sbin/lib").join(library_name()));
        let path = library_path(&settings(), Some(&bundle.join("bin").join("euhedral-server")));
        assert_eq!(path, lib.join(library_name()));
        assert_eq!(
            library_candidates(&settings(), Some(&bundle.join("bin").join("euhedral-server"))).len(),
            2
        );
        assert!(path.is_file());
        // A library beside the executable still wins.
        fs::create_dir_all(bundle.join("bin/lib"))?;
        fs::write(bundle.join("bin/lib").join(library_name()), "x")?;
        let path = library_path(&settings(), Some(&bundle.join("bin").join("euhedral-server")));
        assert_eq!(path, bundle.join("bin/lib").join(library_name()));
        drop(fs::remove_dir_all(bundle));
        Ok(())
    }

    #[test]
    fn a_driver_error_keeps_its_message_and_names_the_device() {
        let error = start_error(&EngineStartError::Driver("no library".to_owned()));
        assert!(error.device);
        assert!(error.causes[0].starts_with("the NVIDIA driver is required"));
    }
}
