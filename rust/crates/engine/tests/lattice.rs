//! The engine's lattice, started from settings and run with frames. Each test starts a lattice of its own.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the frames it published"
)]

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use euhedral_core::config::FragmentConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::Lake;
use euhedral_inference_engine::{Handoff, RuntimeOwner, Teardown, fragment_config, lattice_config};
use euhedral_inference_settings::{Auto, ModelSettings, Settings};

/// The longest a test waits for its frames; past it the test fails, never hangs.
const WAIT: Duration = Duration::from_secs(60);

struct Counted {
    header: FrameHeader,
    ran: Arc<AtomicUsize>,
    /// Frames still to run; the one that brings it to zero sends `total` to `done`.
    pending: Arc<AtomicUsize>,
    total: usize,
    done: Arc<Handoff<usize>>,
}

impl Frame for Counted {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.ran.fetch_add(1, Ordering::AcqRel);
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.done.send(self.total);
        }
        Ok(())
    }
}

/// Runs `frames` frames through a lake of a lattice made from `settings`, and returns how many ran.
fn run_frames(settings: &Settings, frames: usize) -> usize {
    let lattice = ControlPlaneLattice::new(lattice_config(settings)).expect("the lattice starts");
    let lake = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    let unit = lake.admit().unwrap();
    let finished = Handoff::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let pending = Arc::new(AtomicUsize::new(frames));
    for part in 0..frames {
        let frame = Counted {
            header: FrameHeader::new(part as u64 + 1),
            ran: Arc::clone(&ran),
            pending: Arc::clone(&pending),
            total: frames,
            done: Arc::clone(&finished),
        };
        assert!(lake.offer(Box::new(frame)).is_ok());
    }
    assert_eq!(finished.wait(WAIT), Ok(frames));
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
    ran.load(Ordering::Acquire)
}

fn settings(max_batch_size: Auto<u32>) -> Settings {
    let mut settings = Settings::with_model(ModelSettings::default());
    settings.host.max_batch_size = max_batch_size;
    settings
}

/// A worker that sits out cycles parks while frames wait in its cache and nothing wakes it; the engine's workers enter a
/// handle in every cycle, and keep RustEE's other timings.
#[test]
fn workers_never_sit_out_a_cycle() {
    let policy = fragment_config(&settings(Auto::Auto)).idle_policy().clone();
    let default = FragmentConfig::default().idle_policy().clone();
    assert!(policy.probe_interval().is_zero());
    assert_eq!(policy.idle_park(), default.idle_park());
    assert_eq!(policy.unproductive_park(), default.unproductive_park());
}

#[test]
fn the_batch_size_follows_the_settings_and_defaults_to_the_runtimes() {
    let default = fragment_config(&settings(Auto::Auto));
    assert_eq!(default.max_batch_size(), FragmentConfig::default().max_batch_size());
    assert!(!default.polls());
    assert!(fragment_config(&settings(Auto::Value(1))).polls());
    assert!(
        fragment_config(&settings(Auto::Value(2))).polls(),
        "2 is the largest size that polls"
    );
    assert!(!fragment_config(&settings(Auto::Value(3))).polls());
    assert_eq!(fragment_config(&settings(Auto::Value(64))).max_batch_size().get(), 64);
}

#[test]
fn a_lattice_in_poll_mode_runs_frames() {
    assert_eq!(run_frames(&settings(Auto::Value(2)), 256), 256);
}

#[test]
fn a_lattice_in_drain_mode_runs_frames() {
    assert_eq!(run_frames(&settings(Auto::Auto), 256), 256);
}

#[test]
fn dropping_the_owner_closes_the_lakes_and_the_lattice_whatever_handles_are_alive() {
    let settings = settings(Auto::Auto);
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let lakes: Vec<Arc<Lake>> = (0..2)
        .map(|_| {
            Lake::new(
                Arc::clone(&lattice),
                NonZeroUsize::new(1).unwrap(),
                Duration::from_secs(1),
            )
        })
        .collect();
    let teardown = Teardown::new(Arc::clone(&lattice), lakes.clone());
    let owner = RuntimeOwner::over(Arc::clone(&teardown));

    // The engine is working: a frame has run on the lattice.
    let unit = lakes[0].admit().unwrap();
    let finished = Handoff::new();
    let frame = Counted {
        header: FrameHeader::new(1),
        ran: Arc::new(AtomicUsize::new(0)),
        pending: Arc::new(AtomicUsize::new(1)),
        total: 1,
        done: Arc::clone(&finished),
    };
    assert!(lakes[0].offer(Box::new(frame)).is_ok());
    assert_eq!(finished.wait(WAIT), Ok(1));
    drop(unit);

    // `teardown` stands for a handle a frame still holds (the runtime's, in an engine): the drop closes anyway.
    drop(owner);
    for lake in &lakes {
        assert!(lake.admit().is_err(), "a closed lake takes no unit");
    }
    assert!(lattice.start().is_err(), "a closed lattice does not start again");
    teardown.close();
}
