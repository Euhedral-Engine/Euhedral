//! The device rig of the trellis quantizer tests: one lane, the kernels and the math libraries, and a pump that stands
//! in for the lattice's worker, delivering the lane's retirements until a chain's `done` call has run.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the pump is the worker, and a failed unwrap is the failure"
)]
#![allow(dead_code, reason = "each test file uses part of the rig")]

use std::f64::consts::TAU;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::{Completions, LanePool};
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer};
use euhedral_inference_quantization::kernels::{ModuleOptions, Modules};
use euhedral_inference_quantization::trellis::{
    Finished, QuantizeError, Quantized, Refused, Request, TrellisLane, TrellisSettings,
};

pub struct Rig {
    pub library: Arc<CudaLibrary>,
    pub completions: Arc<Completions>,
    lane: Option<TrellisLane>,
    // The device and the lane pool stay open for the lane.
    _gpu: Arc<CudaGpu>,
    _pool: Arc<LanePool>,
}

/// The rig, or a panic naming the missing setting: a device test that skips is no check.
pub fn rig(settings: TrellisSettings) -> Rig {
    let library = test_support::gradle_property("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY)");
    let directory = test_support::gradle_property("euhedral.quantization.cuda-libraries")
        .expect("set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES)");
    let library = CudaLibrary::load(&PathBuf::from(library)).expect("the CUDA library loads");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    let kernels = TrellisLane::kernels(&mut modules).expect("the trellis kernels compile");
    let (pool, mut streams) = gpu
        .open_lanes(&["trellis"], StreamOptions::default())
        .expect("a lane opens");
    let mut stream = streams.remove(0);
    let completions = Completions::new(&pool);
    let mut handles = None;
    euhedral_gpu::Stream::submit(&mut stream, |_| {
        handles = Some(math.lane_handles());
        Ok(())
    })
    .expect("the handles are made on the lane");
    let handles = handles.unwrap().expect("the math handles");
    let lane = TrellisLane::new(
        stream,
        handles,
        kernels,
        Arc::clone(&library),
        Arc::clone(&completions),
        settings,
    )
    .expect("the lane");
    Rig {
        library,
        completions,
        lane: Some(lane),
        _gpu: gpu,
        _pool: pool,
    }
}

type Slot = Arc<Mutex<Option<Finished<TrellisLane, Quantized, QuantizeError>>>>;

impl Rig {
    /// Quantizes and pumps the lane's completions until the chain finishes (two minutes at most).
    pub fn quantize(&mut self, request: &Request) -> Result<Quantized, QuantizeError> {
        let slot: Slot = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&slot);
        let lane = self.lane.take().expect("the lane is idle");
        let begun = lane.quantize(request, move |finished| {
            *sink.lock().unwrap() = Some(finished);
        });
        if let Err(refused) = begun {
            let Refused { lane, error } = *refused;
            self.lane = Some(lane);
            return Err(error);
        }
        let deadline = Instant::now() + Duration::from_secs(120);
        let finished = loop {
            self.completions.reap();
            self.completions.pull(
                &mut |mut frame| {
                    frame.execute().unwrap();
                    frame.do_finally();
                },
                &mut |_| false,
                u64::MAX,
            );
            if let Some(finished) = slot.lock().unwrap().take() {
                break finished;
            }
            assert!(Instant::now() < deadline, "the chain did not finish in two minutes");
        };
        let Finished { lane, scratch, result } = finished;
        self.lane = Some(lane);
        scratch.release();
        result
    }

    pub fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, bytes.len()).expect("device allocation");
        buffer.copy_from_host(0, bytes).expect("upload");
        buffer
    }

    pub fn upload_floats(&self, values: &[f32]) -> DeviceBuffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(&bytes)
    }

    pub fn upload_halves(&self, values: &[u16]) -> DeviceBuffer {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.upload(&bytes)
    }

    pub fn download(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        let mut host = vec![0u8; bytes];
        buffer.copy_to_host(0, &mut host).expect("download");
        host
    }
}

/// The values of a buffer of little-endian 16-bit words.
pub fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes(*pair))
        .collect()
}

/// A bfloat16 word's value.
pub fn bf16(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

/// The bfloat16 word of a value, by truncation.
pub fn to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// A half-precision word's value.
pub fn f16(word: u16) -> f32 {
    let sign = if word & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((word >> 10) & 0x1f);
    let fraction = f32::from(word & 0x3ff);
    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 => {
            if fraction == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + fraction / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// Deterministic standard-normal values.
pub fn gaussians(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| ((-2.0 * uniform().ln()).sqrt() * (TAU * uniform()).cos()) as f32)
        .collect()
}

/// `X^T X / rows` for `rows` standard-normal rows of `k` features: a symmetric positive-definite Hessian.
pub fn hessian(k: usize, rows: usize, seed: u64) -> Vec<f32> {
    let x = gaussians(rows * k, seed);
    let mut h = vec![0f32; k * k];
    for r in 0..rows {
        let row = &x[r * k..(r + 1) * k];
        for i in 0..k {
            for j in 0..k {
                h[i * k + j] += row[i] * row[j];
            }
        }
    }
    h.iter_mut().for_each(|v| *v /= rows as f32);
    h
}

/// A request over the buffers.
pub fn request(
    weight: &DeviceBuffer,
    hessian: Option<&DeviceBuffer>,
    n: u32,
    k: u32,
    count: u64,
    bits: f64,
) -> Request {
    Request {
        weight: weight.addr(),
        n,
        k,
        hessian: hessian.map(DeviceBuffer::addr),
        count,
        bits,
        seed: 0,
        apply_out_scales: true,
    }
}
