//! The Qwen3.8 engine on RustEE: the runtime, the residency planner, the weight loader, the workspace and the
//! sequence state.

pub mod error;
pub mod formats;
mod frames;
pub mod generation;
pub mod handoff;
pub mod kernels;
pub mod model;
pub mod plan;
pub mod quantum;
pub mod residency;
pub mod runtime;
pub mod stages;
mod startup;
pub mod state;
mod warmup;
pub mod workspace;

pub use handoff::{Handoff, TimedOut};
pub use runtime::{
    EngineRuntime, ReadyHandoff, RuntimeOwner, StartupReport, Teardown, fragment_config, lattice_config,
};
pub use warmup::Warmed;
