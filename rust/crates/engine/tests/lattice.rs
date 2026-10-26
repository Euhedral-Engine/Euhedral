//! The engine's lattice, started from settings and run with frames. Each test starts a lattice of its own.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the latch for the frames it published"
)]

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use euhedral_core::config::FragmentConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::{Join, Lake, Latch};
use euhedral_inference_engine::{RuntimeOwner, Teardown, fragment_config, lattice_config};
use euhedral_inference_settings::{Auto, ModelSettings, Settings};

struct Counted {
    header: FrameHeader,
    ran: Arc<AtomicUsize>,
    join: Arc<Join<usize>>,
    part: usize,
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
        self.join.arrive(self.part, self.part);
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
    let (done, finished) = Latch::new();
    let ran = Arc::new(AtomicUsize::new(0));
    let join = Join::new(frames, move |parts| done.set(parts.len()));
    for part in 0..frames {
        let frame = Counted {
            header: FrameHeader::new(part as u64 + 1),
            ran: Arc::clone(&ran),
            join: Arc::clone(&join),
            part,
        };
        assert!(lake.offer(Box::new(frame)).is_ok());
    }
    assert_eq!(finished.wait(), frames);
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
    let (done, finished) = Latch::new();
    let join = Join::new(1, move |parts: Vec<Option<usize>>| done.set(parts.len()));
    let frame = Counted {
        header: FrameHeader::new(1),
        ran: Arc::new(AtomicUsize::new(0)),
        join,
        part: 0,
    };
    assert!(lakes[0].offer(Box::new(frame)).is_ok());
    assert_eq!(finished.wait(), 1);
    drop(unit);

    // `teardown` stands for a handle a frame still holds (the runtime's, in an engine): the drop closes anyway.
    drop(owner);
    for lake in &lakes {
        assert!(lake.admit().is_err(), "a closed lake takes no unit");
    }
    assert!(lattice.start().is_err(), "a closed lattice does not start again");
    teardown.close();
}
