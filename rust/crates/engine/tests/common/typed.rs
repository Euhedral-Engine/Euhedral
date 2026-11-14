//! RustEE lanes and the kernels loaded on them, for the device tests that run stages the way the runtime does.

#![allow(dead_code, reason = "each test file uses some of these helpers")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test helper: a failed unwrap is the failure"
)]

use std::sync::Arc;

use euhedral_gpu::cuda::{CudaGpu, CudaQueue};
use euhedral_gpu::{StreamOptions, open_shared_lanes};
use euhedral_graph::{LaneId, LanePool};
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_cuda::launch::load::load;
use euhedral_inference_engine::kernels::kernel_sources;
use euhedral_inference_engine::runtime::{open_device, open_lane_set};
use euhedral_inference_engine::stages::LaneTable;
use euhedral_inference_settings::{ModelSettings, Settings};

/// The lanes of a test: `compute` lanes with ids from 0, then the capture and the owner's lanes, which graphs are not
/// homed on, and the kernels loaded on the device.
pub struct Typed {
    pub gpu: Arc<CudaGpu>,
    pub table: LaneTable,
    /// The lanes as queues, for the graph runtime and the completions.
    pub queues: Arc<LanePool>,
    pub library: Arc<CudaLibrary>,
}

impl Typed {
    /// Opens the device `library` bound, loads the kernels and opens the lanes, each with `ring` markers.
    pub fn open(library: &Arc<CudaLibrary>, compute: usize, ring: usize) -> Typed {
        let gpu = open_device(library).expect("the device opens through RustEE");
        let settings = Settings::with_model(ModelSettings::default());
        let sources = kernel_sources(library, &settings).expect("kernel sources");
        let kernels = Arc::new(load(&gpu, &sources).expect("the kernels load"));
        let set = open_lane_set(&gpu, compute, ring, 1).expect("the lanes open");
        Typed {
            table: LaneTable::new(set.typed, kernels, Arc::clone(library)),
            queues: set.queues,
            gpu,
            library: Arc::clone(library),
        }
    }

    /// Opens `total` lanes of `ring` markers each, none reserved, with ids from 0: for a test that picks each lane's use.
    pub fn plain(library: &Arc<CudaLibrary>, total: usize, ring: usize) -> Typed {
        let gpu = open_device(library).expect("the device opens through RustEE");
        let settings = Settings::with_model(ModelSettings::default());
        let sources = kernel_sources(library, &settings).expect("kernel sources");
        let kernels = Arc::new(load(&gpu, &sources).expect("the kernels load"));
        let options = StreamOptions {
            markers: ring as u32,
            spans: 0,
            ..StreamOptions::default()
        };
        let specs: Vec<(&str, StreamOptions)> = (0..total).map(|_| ("test", options)).collect();
        let shared = open_shared_lanes(&gpu, &specs).expect("the lanes open");
        Typed {
            table: LaneTable::new(shared.pool, kernels, Arc::clone(library)),
            queues: shared.queues,
            gpu,
            library: Arc::clone(library),
        }
    }

    /// The lane `id`.
    pub fn lane(&self, id: u16) -> &Arc<CudaQueue> {
        self.table.get(LaneId(id)).expect("the lane exists")
    }

    /// Waits for all the work queued on the device.
    pub fn sync(&self) {
        self.library.synchronize().unwrap();
    }
}
