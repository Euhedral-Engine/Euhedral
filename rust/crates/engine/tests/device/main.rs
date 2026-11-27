//! The engine's device tests: one binary, so one process holds the engines of the whole suite, and a test that needs the
//! engine another test left running reuses it (see `common::engine`). Each module is one area; a test that runs on the
//! card takes it through `common`.

#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes a lane to read results"
)]
#![allow(
    clippy::disallowed_types,
    reason = "the reduced model of the rig tests is shared under a lock of the test harness, no engine state"
)]

#[path = "../common/mod.rs"]
mod common;
#[path = "../common/linear_gpu.rs"]
mod linear_gpu;
#[path = "../common/typed.rs"]
mod typed;

mod batched;
mod cancel;
mod capture;
mod capture_exact;
mod capture_fault;
mod composite;
mod constraint;
mod flash;
mod formats;
mod formats_oracle;
mod generation;
mod hidden;
mod loader;
mod mtp;
mod prefix_copy;
mod prefix_mtp;
mod prefix_restore;
mod reserve;
mod retire;
mod sampled_mtp;
mod stages;
mod state;
