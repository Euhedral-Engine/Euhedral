//! The HTTP server of the API on RustEE's `euhedral-net`: it routes a request, answers the routes that need no
//! generation (health and the model lists), answers every failure with the JSON error of the surface the request
//! addressed, and hands the generating routes to a [`Backend`].
//!
//! Nothing here owns a thread or waits: the server is frames on the caller's lattice, and the backend's work is
//! reported through continuations and listeners.

pub mod admission;
pub mod answer;
pub mod backend;
pub mod clock;
pub mod engine_backend;
mod frames;
pub mod health;
pub mod limits;
pub mod metrics;
pub mod models;
pub mod planning;
pub mod refusals;
pub mod router;
#[cfg(any(test, feature = "test-support"))]
pub mod scripted;
mod server;

pub use admission::{Admission, Counts, Entry, Handle, Interruption, Slot};
pub use answer::Answerer;
pub use backend::{
    Backend, BackendError, EncodedPrompt, EngineReport, Generation, GenerationConfig, GenerationResult, Next,
    OutputSpec, PrefixCacheReport, TextListener,
};
pub use clock::{Clock, Timers};
pub use engine_backend::{Codec, Engine, EngineBackend, EngineHost, EngineStarting, StartError, TextStream};
pub use frames::once;
pub use limits::Limits;
pub use planning::{DefaultsError, Endpoint, Extras, Outcome, Planned, Planner, Publisher, RequestDefaults};
pub use server::{ApiConfig, ApiServer, start, start_on};
