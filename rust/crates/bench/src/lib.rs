//! The benchmark: the Java benchmark's configuration files and result rows, the deterministic prompt corpora
//! ([`prompt`]) that reproduce its prompts byte for byte, and the runner that measures this engine on them.
//!
//! The crate is a library and a thin binary. The library carries the no-wait rules of the workspace: nothing in it
//! prints, exits or reads the environment, and the few blocking file calls (the configuration, the tokenizer section,
//! the result file, the teacher-forced report) are named exceptions at their own definitions. The binary (`main.rs`)
//! parses the command line, prints, and is the only place that waits for a generation (`run`); its main thread only
//! starts requests and waits for them, as the engine's own tests do.

pub mod artifact;
pub mod config;
pub mod dump;
pub mod metrics;
pub mod prompt;
pub mod result;
pub mod run;
