//! The weight cache on a real lattice: engines that ask for the same object through the `host` lake share one load.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a channel carries each answer to the waiting test thread, under a deadline that names what never arrived"
)]

use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_graph::Lake;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::weight_cache::{Backend, HostFrames, Key, LoadError, WeightCache};
use euhedral_inference_settings::{ModelSettings, Settings};

const DEADLINE: Duration = Duration::from_secs(30);

struct Reads(Arc<AtomicUsize>);

impl Backend for Reads {
    type Handle = u64;

    fn load(&self, key: &Key, _bytes: u64) -> Result<u64, LoadError> {
        self.0.fetch_add(1, Ordering::AcqRel);
        Ok(key.offset)
    }
}

#[test]
fn engines_asking_through_the_host_lake_share_one_load() {
    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let host = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    let frames = HostFrames::new(Arc::clone(&host));
    let reads = Arc::new(AtomicUsize::new(0));
    let cache = WeightCache::new(Reads(Arc::clone(&reads)));
    let key = Key {
        file: PathBuf::from("base.edrl"),
        name: "norm".to_string(),
        offset: 7,
    };

    let (sender, receiver) = mpsc::channel();
    for _ in 0..4 {
        let sender = sender.clone();
        cache.acquire(key.clone(), 64, &frames, move |outcome| {
            sender.send(outcome).ok();
        });
    }
    let leases: Vec<_> = (0..4)
        .map(|_| {
            receiver
                .recv_timeout(DEADLINE)
                .expect("an engine's answer never arrived")
                .expect("the object loaded")
        })
        .collect();
    assert_eq!(reads.load(Ordering::Acquire), 1);
    assert!(leases.iter().all(|lease| ptr::eq(lease.handle(), leases[0].handle())));
    assert_eq!(*leases[0].handle(), 7);
    drop(leases);
    assert_eq!(cache.size(), 0);
    assert_eq!(cache.resident_bytes(), 0);

    host.complete_gracefully();
    lattice.close();
}
