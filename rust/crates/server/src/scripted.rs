//! A backend that follows a script, on a lattice of its own: for testing the process without a device. The binary
//! offers it behind the hidden `--scripted-backend[=hello|endless|hung]` argument when built with the `scripted` feature.

use std::sync::Arc;
use std::time::Duration;

use euhedral_core::config::{FragmentConfig, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_inference_api_server::scripted::{Script, ScriptedBackend};
use euhedral_inference_settings::Settings;

use crate::{StartError, Started};

/// A lattice with the default worker count.
fn lattice() -> Result<Arc<ControlPlaneLattice>, StartError> {
    let cloneable = BaseCloneableObject::new(FragmentConfig::default(), DefaultExecutor::new());
    let mut config = LatticeConfig::new(Arc::new(cloneable));
    config.name = "scripted-server".into();
    config.shard_name = "scripted-server".into();
    config.shutdown_timeout = Duration::from_secs(5);
    config.startup_timeout = Duration::from_secs(10);
    let failed = |error: String| StartError::new("Starting the lattice", [error]);
    let lattice = ControlPlaneLattice::new(config).map_err(|error| failed(format!("{error:?}")))?;
    lattice.start().map_err(|error| failed(format!("{error:?}")))?;
    Ok(lattice)
}

/// A scripted backend that follows `script`.
pub fn start_with(script: Script) -> Result<Started, StartError> {
    start_scripted(script).map(|(started, _)| started)
}

/// A scripted backend that follows `script`, and the backend itself, for a test that steers its generations.
pub fn start_scripted(script: Script) -> Result<(Started, Arc<ScriptedBackend>), StartError> {
    let lattice = lattice()?;
    let backend = ScriptedBackend::on(&lattice);
    backend.set_script(script);
    let stop = (backend.clone(), lattice.clone());
    let started = Started {
        lattice,
        backend: backend.clone(),
        // The lattice of a scripted backend closes within its own bound.
        shutdown: Box::new(move |_timeout| {
            let (backend, lattice) = stop;
            backend.release();
            lattice.close();
            Ok(())
        }),
    };
    Ok((started, backend))
}

/// A scripted backend that says `Hello, world`.
pub fn start(_settings: &Settings) -> Result<Started, StartError> {
    start_with(Script::hello())
}

/// The script a name stands for: `hello`, `endless` (until cancelled) or `hung` (never ends, even when cancelled).
pub fn script_named(name: &str) -> Option<Script> {
    match name {
        "hello" => Some(Script::hello()),
        "endless" => Some(Script::Endless),
        "hung" => Some(Script::Hung),
        _ => None,
    }
}
