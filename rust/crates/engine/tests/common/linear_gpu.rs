//! What the linear format tests on a GPU share: the library and a lane, objects uploaded from the RCO artifact or made
//! synthetic, activations, and one linear run and read back.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or
//! `gradle.properties`) and the artifact from `euhedral.qwen.rco-artifact`; a test skips with a message when it is
//! absent.

#![allow(dead_code, reason = "each test file uses some of these helpers")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test helper: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads the artifact, copies device memory and synchronizes"
)]

use std::f32::consts::TAU;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::LaneId;
use euhedral_inference_cuda::{CudaLane, CudaLibrary, DeviceBuffer, Submission, SubmitMode};
use euhedral_inference_edrl::{
    Header, ReadPlan, Region, TensorDataType, TensorTable, TokenizerFooter, TrellisCompanions, WeightFormat,
    WeightLayout, content_end,
};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, ScratchRegion};
use euhedral_inference_engine::model::{TrellisParts, WeightRef};

pub struct Rig {
    pub library: Arc<CudaLibrary>,
    pub lane: Arc<CudaLane>,
}

/// The library and a lane, or `None` (with a message) on a machine without them.
pub fn rig() -> Option<Rig> {
    let path = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let library = CudaLibrary::load(&path).expect("the CUDA library loads");
    let lane = CudaLane::new(&library, LaneId(0), 16).expect("a lane");
    Some(Rig { library, lane })
}

/// Splitmix64: the same stream for the same seed, on any machine.
pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Standard normal (Box-Muller).
    pub fn gaussian(&mut self) -> f32 {
        let (a, b) = (self.uniform().max(1e-7), self.uniform());
        (-2.0 * a.ln()).sqrt() * (TAU * b).cos()
    }
}

/// BF16 of `value`, rounded to nearest even.
pub fn bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

pub fn bf16_value(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// FP16 of a normal `value`, rounded to nearest even.
pub fn fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value == 0.0 {
        return sign;
    }
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exponent), "{value} is not a normal FP16");
    let mantissa = bits & 0x007f_ffff;
    let mut half = ((exponent as u32) << 10) | (mantissa >> 13);
    let rest = mantissa & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half as u16
}

pub fn upload(library: &Arc<CudaLibrary>, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.len().max(256)).unwrap();
    buffer.copy_from_host(0, bytes).unwrap();
    buffer
}

pub fn empty(library: &Arc<CudaLibrary>, bytes: usize) -> DeviceBuffer {
    DeviceBuffer::alloc(library, bytes.max(256)).unwrap()
}

/// `rows` rows of `k` Gaussian activations as BF16 bytes.
pub fn activations(random: &mut Random, rows: usize, k: usize) -> Vec<u8> {
    (0..rows * k)
        .flat_map(|_| bf16(random.gaussian()).to_le_bytes())
        .collect()
}

/// One trellis object on the device: the buffers, and the weight that addresses them.
pub struct Trellis {
    pub weight: WeightRef,
    pub n: usize,
    pub k: usize,
    _buffers: Vec<DeviceBuffer>,
}

/// The weight addressing `buffers` = [tile words, suh, svh, segments], which the result keeps alive.
fn trellis_weight(
    name: &str,
    shape: Vec<u64>,
    ends: &[u32],
    words_bytes: usize,
    buffers: Vec<DeviceBuffer>,
) -> Trellis {
    let k = shape[0] as usize * 16;
    let n = shape[1] as usize * 16;
    Trellis {
        weight: WeightRef {
            name: name.to_string(),
            shape,
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Trellis,
            layout: WeightLayout::TrellisTile16V1,
            ptr: buffers[0].ptr(),
            bytes: words_bytes as u64,
            trellis: Some(TrellisParts {
                suh: buffers[1].ptr(),
                svh: buffers[2].ptr(),
                segments: buffers[3].ptr(),
                segment_ends: Arc::from(ends),
                suh_rows: ends.len() as u64,
            }),
        },
        n,
        k,
        _buffers: buffers,
    }
}

/// A synthetic trellis object (random tile words, signs and output scales) of `n` outputs and `k` inputs at `w` words
/// per tile, cut into segments ending at `ends`.
pub fn synthetic_trellis(
    library: &Arc<CudaLibrary>,
    random: &mut Random,
    n: usize,
    k: usize,
    w: usize,
    ends: &[u32],
) -> Trellis {
    let words = k / 16 * (n / 16) * w;
    let payload: Vec<u8> = (0..words)
        .flat_map(|_| (random.next_u64() as u16).to_le_bytes())
        .collect();
    let suh: Vec<u8> = (0..ends.len() * k)
        .flat_map(|_| fp16(if random.next_u64() & 1 == 1 { 1.0 } else { -1.0 }).to_le_bytes())
        .collect();
    let svh: Vec<u8> = (0..n)
        .flat_map(|_| fp16(0.5 + random.uniform()).to_le_bytes())
        .collect();
    let segments: Vec<u8> = ends.iter().flat_map(|end| end.to_le_bytes()).collect();
    let buffers = vec![
        upload(library, &payload),
        upload(library, &suh),
        upload(library, &svh),
        upload(library, &segments),
    ];
    let shape = vec![(k / 16) as u64, (n / 16) as u64, w as u64];
    trellis_weight("synthetic", shape, ends, payload.len(), buffers)
}

/// The RCO artifact's table, and the file its payloads are read from.
pub struct Artifact {
    file: File,
    table: TensorTable,
}

fn read_range(file: &mut File, range: Range<u64>) -> Vec<u8> {
    file.seek(SeekFrom::Start(range.start)).unwrap();
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

impl Artifact {
    /// The artifact named by `euhedral.qwen.rco-artifact`, or `None` (with a message) when there is none.
    #[allow(clippy::print_stdout, reason = "a test reports why it skipped")]
    pub fn open() -> Option<Artifact> {
        let path = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
        if !path.is_file() {
            println!("skipped: euhedral.qwen.rco-artifact names no file");
            return None;
        }
        let mut file = File::open(&path).unwrap();
        let file_len = file.metadata().unwrap().len();
        let probe = ReadPlan::probe(file_len).unwrap();
        let header = Header::parse(
            read_range(&mut file, probe.range(Region::Header).unwrap())
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let footer = TokenizerFooter::parse(
            read_range(&mut file, probe.range(Region::TokenizerFooter).unwrap())
                .as_slice()
                .try_into()
                .unwrap(),
        )
        .unwrap();
        let plan = ReadPlan::new(&header, file_len, footer.as_ref()).unwrap();
        let table = TensorTable::parse(
            &read_range(&mut file, plan.range(Region::Table).unwrap()),
            &header,
            content_end(file_len, footer.as_ref()).unwrap(),
        )
        .unwrap();
        Some(Artifact { file, table })
    }

    pub fn table(&self) -> &TensorTable {
        &self.table
    }

    /// The payload of the tensor `name`.
    pub fn bytes(&mut self, name: &str) -> Vec<u8> {
        let range = self
            .table
            .get(name)
            .unwrap_or_else(|| panic!("no tensor {name}"))
            .range();
        read_range(&mut self.file, range)
    }

    /// The trellis object `name` with its companions, uploaded.
    pub fn trellis(&mut self, library: &Arc<CudaLibrary>, name: &str) -> Trellis {
        let entry = self.table.get(name).unwrap();
        assert_eq!(entry.format(), WeightFormat::Trellis, "{name}");
        let shape = entry.shape().to_vec();
        let payload = self.bytes(name);
        let suh = self.bytes(&format!("{name}.suh"));
        let svh = self.bytes(&format!("{name}.svh"));
        let segments = self.bytes(&format!("{name}.segments"));
        let ends = TrellisCompanions::parse(&segments).unwrap();
        let buffers = vec![
            upload(library, &payload),
            upload(library, &suh),
            upload(library, &svh),
            upload(library, &segments),
        ];
        trellis_weight(name, shape, &ends, payload.len(), buffers)
    }

    /// A weight stored whole in one format (an NVFP4 projection, the Q3 embedding), uploaded.
    pub fn plain(&mut self, library: &Arc<CudaLibrary>, name: &str) -> (DeviceBuffer, WeightRef) {
        let entry = self.table.get(name).unwrap();
        let (shape, dtype, format, layout) = (entry.shape().to_vec(), entry.dtype(), entry.format(), entry.layout());
        let payload = self.bytes(name);
        let buffer = upload(library, &payload);
        let weight = WeightRef {
            name: name.to_string(),
            shape,
            dtype,
            format,
            layout,
            ptr: buffer.ptr(),
            bytes: payload.len() as u64,
            trellis: None,
        };
        (buffer, weight)
    }
}

impl Rig {
    /// Runs `format` over `rows` rows of `x` (BF16, `k` wide) against `weight` and returns the BF16 output bits,
    /// `rows` rows of `n`. The scratch is the format's own size for a prefill-capable launch.
    #[allow(
        clippy::too_many_arguments,
        reason = "a test helper that names every dimension of the launch"
    )]
    pub fn linear(
        &self,
        format: &dyn LinearFormat,
        weight: &WeightRef,
        x: &DeviceBuffer,
        x_offset_rows: usize,
        k: usize,
        n: usize,
        rows: usize,
        mode: LinearMode,
    ) -> Vec<u16> {
        let scratch_bytes = format.scratch_bytes(weight, rows, LinearMode { prefill: true, ..mode });
        let scratch = empty(&self.library, scratch_bytes);
        let y = empty(&self.library, rows * n * 2);
        let input = x.ptr().offset((x_offset_rows * k * 2) as u64);
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        format
            .submit(
                &submission,
                weight,
                input,
                y.ptr(),
                rows,
                ScratchRegion {
                    ptr: scratch.ptr(),
                    bytes: scratch.bytes(),
                },
                mode,
            )
            .unwrap();
        submission.finish().unwrap();
        self.lane.synchronize().unwrap();
        let mut host = vec![0u8; rows * n * 2];
        y.copy_to_host(0, &mut host).unwrap();
        host.as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect()
    }
}

pub fn values(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|bits| bf16_value(*bits)).collect()
}

/// The rms of `actual - wanted` relative to the rms of `wanted`.
pub fn relative_rms(wanted: &[f32], actual: &[f32]) -> f64 {
    let (mut error, mut energy) = (0.0f64, 0.0f64);
    for (wanted, actual) in wanted.iter().zip(actual) {
        error += f64::from(actual - wanted).powi(2);
        energy += f64::from(*wanted).powi(2);
    }
    (error / energy).sqrt()
}

/// `actual` is within `relative` of `wanted` per element, plus `relative` of the row's rms.
pub fn assert_close(wanted: &[f32], actual: &[f32], n: usize, relative: f64, label: &str) {
    assert_eq!(wanted.len(), actual.len(), "{label}");
    for (row, (wanted_row, actual_row)) in wanted.chunks(n).zip(actual.chunks(n)).enumerate() {
        let rms = (wanted_row.iter().map(|value| f64::from(*value).powi(2)).sum::<f64>() / n as f64).sqrt();
        assert!(rms > 0.0 && rms.is_finite(), "{label} row {row} has no energy");
        for (column, (wanted, actual)) in wanted_row.iter().zip(actual_row).enumerate() {
            let allowed = relative * (f64::from(wanted.abs()) + rms);
            let error = f64::from((wanted - actual).abs());
            assert!(
                error <= allowed,
                "{label} row {row} column {column}: {actual} against {wanted}"
            );
        }
    }
}
