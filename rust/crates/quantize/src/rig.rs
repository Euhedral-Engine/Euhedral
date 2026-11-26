//! The host runtime of the commands that move files and need no device: a lattice with the asynchronous I/O as a
//! source and one host lake. The calling thread only starts a run and waits for its one completion.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_settings::Settings;
use euhedral_io::AsyncIo;

/// A running lattice, its asynchronous I/O and its host lake.
pub struct Rig {
    lattice: Arc<ControlPlaneLattice>,
    pub io: Arc<AsyncIo>,
    pub host: Arc<Lake>,
}

impl Rig {
    /// Starts the runtime the settings describe.
    pub fn start(settings: &Settings) -> Result<Rig, String> {
        let lattice = ControlPlaneLattice::new(lattice_config(settings)).map_err(|error| format!("{error:?}"))?;
        let io = AsyncIo::open().map_err(|error| error.to_string())?;
        lattice
            .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
            .map_err(|error| format!("{error:?}"))?;
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(settings.host.task_parallelism as usize).unwrap_or(NonZeroUsize::MIN),
            settings.host.idle_release.max(Duration::from_millis(1)),
        );
        Ok(Rig { lattice, io, host })
    }

    /// Stops the runtime once its work has ended.
    pub fn stop(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}
