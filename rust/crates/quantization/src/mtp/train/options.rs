//! What a run is asked to do and what it runs on.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_io::AsyncIo;

use super::settings::Settings;

/// What a run is asked to do.
#[derive(Clone, Debug)]
pub struct Options {
    /// The corpus `generate` wrote.
    pub corpus: PathBuf,
    /// The base model's checkpoint directory: the drafter's starting parameters.
    pub model: PathBuf,
    /// The artifact: the draft head and the token embedding.
    pub artifact: PathBuf,
    /// Where the best parameters and the state go.
    pub out: PathBuf,
    pub settings: Settings,
    /// The engine's CUDA library.
    pub library: PathBuf,
    /// The directory of the CUDA math libraries.
    pub math: PathBuf,
    /// Ends the run after this many state saves, leaving the state to resume from; 0 never stops early.
    pub stop_after_saves: u32,
}

/// The runtime a run goes on: the asynchronous I/O, the host lake and the way to have the lattice take the lane's
/// completions.
pub struct Host<'a> {
    pub io: &'a Arc<AsyncIo>,
    pub lake: &'a Arc<Lake>,
    pub attach: &'a dyn Fn(Arc<dyn LatticeSource>) -> Result<(), String>,
}

/// Where the run's lines go.
pub type Log = Arc<dyn Fn(&str) + Send + Sync>;

impl Debug for Host<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Host").finish_non_exhaustive()
    }
}
