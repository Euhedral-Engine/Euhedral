//! The device side of the `cost` command: one object at a time, loaded from a source artifact as copies on the card,
//! and its linear launched through the engine's own format for it, timed by device events.
//!
//! The launches of a timed run are queued behind a gate, a long device copy, so the host finishes queueing them while
//! the card is still busy and the time between the span's two events holds the launches and no host gap. A run is
//! valid only if the gate had not ended when the last launch was queued; a run that fails this is repeated behind a
//! longer gate.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::time::Duration;

use euhedral_gpu::cuda::{CudaGpu, CudaStream};
use euhedral_gpu::{Gpu, GpuError, Stream, StreamOptions};
use euhedral_graph::DeviceQueue;
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::{KernelTable, load};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, KernelOps};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, Nvfp4Linear, ScratchRegion, TrellisLinear};
use euhedral_inference_engine::kernels::kernel_sources;
use euhedral_inference_engine::model::{Residence, TrellisParts, WeightRef};
use euhedral_inference_engine::runtime::open_device;
use euhedral_inference_quantization::assemble::{SourceSet, TensorEntry, TrellisCompanions, WeightFormat};
use euhedral_inference_quantization::mixed::{Bench, MixedError, QuantOption, Shape, launch_mode};
use euhedral_inference_settings::Settings;

/// BF16 1.0: what the input is filled with, so the launches read real values and not denormals or zeros.
const BF16_ONE: u16 = 0x3C00;
/// One gate copy moves this many bytes each way.
const GATE_CHUNK_BYTES: usize = 64 << 20;
/// The gate's first length, in copies.
const GATE_START_COPIES: usize = 4;
/// The gate stops growing here: a card whose launches cannot be hidden behind this many copies is refused.
const GATE_MAX_COPIES: usize = 8192;
/// Tries to queue a timed run's marker and span events before the ring is known to be wedged; the ring frees as the
/// card works, and a run that never gets a slot is an error, not a wait.
const BUSY_TRIES: usize = 1 << 20;

fn failed(what: &str, error: impl Display) -> MixedError {
    MixedError::Invalid(format!("{what}: {error}"))
}

/// The engine's format for the objects `option` stores.
fn format_of<'f>(option: QuantOption, nvfp4: &'f Nvfp4Linear, trellis: &'f TrellisLinear) -> &'f dyn LinearFormat {
    if option == QuantOption::Nvfp4 { nvfp4 } else { trellis }
}

/// The object on the card: its copies, each a weight and the buffers that keep it.
struct Loaded {
    option: QuantOption,
    weights: Vec<WeightRef>,
    // Dropped with the weights, after the card is idle: see `DeviceBench::load`.
    _buffers: Vec<DeviceBuffer>,
}

/// The input, output and scratch of the launches, sized for the most rows a run takes.
struct Operands {
    x: DeviceBuffer,
    y: DeviceBuffer,
    scratch: DeviceBuffer,
}

/// The card, its kernels and the stream the launches run on.
pub struct DeviceBench<'a> {
    sources: &'a SourceSet,
    files: BTreeMap<QuantOption, File>,
    library: Arc<CudaLibrary>,
    table: Arc<KernelTable>,
    stream: CudaStream,
    queue: Arc<dyn DeviceQueue>,
    trellis: TrellisLinear,
    nvfp4: Nvfp4Linear,
    gate: [DeviceBuffer; 2],
    gate_copies: usize,
    loaded: Option<Loaded>,
    operands: Option<Operands>,
    shape: Shape,
}

impl<'a> DeviceBench<'a> {
    /// Opens the card `library` bound and loads the kernels the engine loads there under `settings`.
    pub fn open(
        library: &Arc<CudaLibrary>,
        settings: &Settings,
        sources: &'a SourceSet,
    ) -> Result<DeviceBench<'a>, MixedError> {
        let gpu: Arc<CudaGpu> = open_device(library).map_err(|error| failed("the device", error))?;
        let kernel_sources = kernel_sources(library, settings).map_err(|error| failed("the kernel sources", error))?;
        let table = Arc::new(load(&gpu, &kernel_sources).map_err(|error| failed("the kernels", error))?);
        let options = StreamOptions::default();
        let (_pool, mut streams) = gpu
            .open_lanes(&["cost"], options)
            .map_err(|error| failed("the stream", error))?;
        let stream = streams.pop().ok_or_else(|| failed("the stream", "none was opened"))?;
        let queue = stream.queue();
        let gate = [
            DeviceBuffer::alloc(library, GATE_CHUNK_BYTES).map_err(|error| failed("the gate buffer", error))?,
            DeviceBuffer::alloc(library, GATE_CHUNK_BYTES).map_err(|error| failed("the gate buffer", error))?,
        ];
        Ok(DeviceBench {
            sources,
            files: BTreeMap::new(),
            library: Arc::clone(library),
            table,
            stream,
            queue,
            trellis: TrellisLinear::from_settings(&settings.kernels.trellis, library.device().sm_count),
            nvfp4: Nvfp4Linear::new(library),
            gate,
            gate_copies: GATE_START_COPIES,
            loaded: None,
            operands: None,
            shape: Shape { input: 0, output: 0 },
        })
    }

    /// The bytes of `entry` in `option`'s source.
    fn read(&mut self, option: QuantOption, entry: &TensorEntry) -> Result<Vec<u8>, MixedError> {
        if !self.files.contains_key(&option) {
            let path = self.sources.path(option).map_err(|error| failed("the source", error))?;
            let file = File::open(path).map_err(|error| failed("the source", error))?;
            self.files.insert(option, file);
        }
        let range = entry.range();
        let length = usize::try_from(range.end - range.start).map_err(|_| failed(entry.name(), "too large to read"))?;
        let mut bytes = vec![0; length];
        let file = self
            .files
            .get(&option)
            .ok_or_else(|| failed("the source", "not open"))?;
        file.read_exact_at(&mut bytes, range.start)
            .map_err(|error| failed(entry.name(), error))?;
        Ok(bytes)
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceBuffer, MixedError> {
        let buffer =
            DeviceBuffer::alloc(&self.library, bytes.len().max(256)).map_err(|error| failed("a buffer", error))?;
        buffer
            .copy_from_host(0, bytes)
            .map_err(|error| failed("an upload", error))?;
        Ok(buffer)
    }

    /// Waits for every launch queued so far.
    fn settle(&self) -> Result<(), MixedError> {
        self.library.synchronize().map_err(|error| failed("the card", error))
    }

    /// The weights of one copy of `object` from the bytes already read, uploaded into `buffers`.
    fn copy_of(
        &self,
        entry: &TensorEntry,
        payload: &[u8],
        companions: Option<[&[u8]; 3]>,
        buffers: &mut Vec<DeviceBuffer>,
    ) -> Result<WeightRef, MixedError> {
        let body = self.upload(payload)?;
        let (ptr, bytes) = (body.ptr(), payload.len() as u64);
        buffers.push(body);
        let trellis = match companions {
            None => None,
            Some([suh, svh, segments]) => {
                let ends = TrellisCompanions::parse(segments).map_err(|error| failed(entry.name(), error))?;
                let [suh, svh, segments_buffer] = [suh, svh, segments].map(|bytes| self.upload(bytes));
                let (suh, svh, segments_buffer) = (suh?, svh?, segments_buffer?);
                let parts = TrellisParts {
                    suh: suh.ptr(),
                    svh: svh.ptr(),
                    segments: segments_buffer.ptr(),
                    suh_rows: ends.len() as u64,
                    segment_ends: Arc::from(ends),
                };
                buffers.extend([suh, svh, segments_buffer]);
                Some(parts)
            }
        };
        Ok(WeightRef {
            name: entry.name().to_owned(),
            shape: entry.shape().to_vec(),
            dtype: entry.dtype(),
            format: entry.format(),
            layout: entry.layout(),
            ptr,
            bytes,
            residence: Residence::Device,
            trellis,
        })
    }

    /// Queues one launch of the linear of each copy in `order`, `rows` rows each.
    fn queue_launches(&mut self, rows: u32, order: &[usize]) -> Result<(), MixedError> {
        let (Some(loaded), Some(operands)) = (self.loaded.as_ref(), self.operands.as_ref()) else {
            return Err(failed("a launch", "no object is loaded"));
        };
        let format = format_of(loaded.option, &self.nvfp4, &self.trellis);
        let mode = launch_mode(rows);
        let modes = Modes {
            row_exact: mode.row_exact,
            ..Modes::default()
        };
        let scratch = ScratchRegion {
            ptr: operands.scratch.ptr(),
            bytes: operands.scratch.bytes(),
        };
        let (x, y, table) = (operands.x.ptr(), operands.y.ptr(), &self.table);
        self.stream
            .submit(|queue| {
                let mut ops = RusteeOps::new(queue, table, modes);
                for &copy in order {
                    let weight = &loaded.weights[copy];
                    format
                        .submit(&mut ops, weight, x, y, rows as usize, scratch, mode)
                        .map_err(|error| GpuError::Invalid(error.to_string()))?;
                }
                Ok(())
            })
            .map_err(|error| failed("the launches", error))
    }

    /// Queues the gate, then the marker that tells when it ended.
    fn queue_gate(&mut self) -> Result<u64, MixedError> {
        let (copies, source, destination) = (self.gate_copies, self.gate[0].ptr(), self.gate[1].ptr());
        let table = &self.table;
        self.stream
            .submit(|queue| {
                let mut ops = RusteeOps::new(queue, table, Modes::default());
                for _ in 0..copies {
                    ops.copy_device_to_device(destination, source, GATE_CHUNK_BYTES as u64)
                        .map_err(|error| GpuError::Invalid(error.to_string()))?;
                }
                Ok(())
            })
            .map_err(|error| failed("the gate", error))?;
        for _ in 0..BUSY_TRIES {
            match self.stream.mark() {
                Ok(marker) => return Ok(marker.value),
                Err(GpuError::Busy) => {}
                Err(error) => return Err(failed("the gate marker", error)),
            }
        }
        Err(failed("the gate marker", "the marker ring stayed full"))
    }

    /// One timed run behind the current gate: the time of the launches, or `None` when the gate ended before the last
    /// launch was queued.
    fn timed_once(&mut self, rows: u32, order: &[usize]) -> Result<Option<Duration>, MixedError> {
        let gate = self.queue_gate()?;
        let span = self.stream.span_begin().map_err(|error| failed("the span", error))?;
        self.queue_launches(rows, order)?;
        self.stream.span_end(span).map_err(|error| failed("the span", error))?;
        let hidden = self.queue.completed() < gate;
        self.settle()?;
        let elapsed = self
            .stream
            .span_poll(span)
            .map_err(|error| failed("the span", error))?
            .ok_or_else(|| failed("the span", "not read after the card was idle"))?;
        Ok(hidden.then_some(elapsed))
    }
}

impl Bench for DeviceBench<'_> {
    fn load(
        &mut self,
        option: QuantOption,
        object: &str,
        shape: Shape,
        copies: usize,
        max_rows: u32,
    ) -> Result<(), MixedError> {
        self.settle()?;
        self.loaded = None;
        self.operands = None;
        let parts = self
            .sources
            .object_with_companions(option, object)
            .map_err(|error| failed(object, error))?;
        let entries: Vec<TensorEntry> = parts.into_iter().cloned().collect();
        let mut bytes = Vec::new();
        for entry in &entries {
            bytes.push(self.read(option, entry)?);
        }
        let main = entries.first().ok_or_else(|| failed(object, "no descriptor"))?;
        let companions = match (main.format(), bytes.as_slice()) {
            (WeightFormat::Trellis, [_, suh, svh, segments]) => {
                Some([suh.as_slice(), svh.as_slice(), segments.as_slice()])
            }
            (WeightFormat::Trellis, _) => return Err(failed(object, "a trellis object lacks a companion")),
            _ => None,
        };
        let mut buffers = Vec::new();
        let mut weights = Vec::with_capacity(copies);
        for _ in 0..copies {
            weights.push(self.copy_of(main, &bytes[0], companions, &mut buffers)?);
        }
        let rows = max_rows as usize;
        let (input, output) = (shape.input as usize, shape.output as usize);
        let format = format_of(option, &self.nvfp4, &self.trellis);
        let scratch = [
            LinearMode::default(),
            LinearMode {
                row_exact: true,
                prefill: false,
            },
            LinearMode {
                row_exact: false,
                prefill: true,
            },
        ]
        .into_iter()
        .map(|mode| format.scratch_bytes(&weights[0], rows, mode))
        .max()
        .unwrap_or(0);
        let ones: Vec<u8> = (0..rows * input).flat_map(|_| BF16_ONE.to_le_bytes()).collect();
        let operands = Operands {
            x: self.upload(&ones)?,
            y: DeviceBuffer::alloc(&self.library, (rows * output * 2).max(256))
                .map_err(|error| failed("the output", error))?,
            scratch: DeviceBuffer::alloc(&self.library, scratch.max(256))
                .map_err(|error| failed("the scratch", error))?,
        };
        self.settle()?;
        self.shape = shape;
        self.loaded = Some(Loaded {
            option,
            weights,
            _buffers: buffers,
        });
        self.operands = Some(operands);
        Ok(())
    }

    fn run(&mut self, rows: u32, order: &[usize]) -> Result<(), MixedError> {
        self.queue_launches(rows, order)?;
        self.settle()
    }

    fn time(&mut self, rows: u32, order: &[usize]) -> Result<Duration, MixedError> {
        loop {
            if let Some(elapsed) = self.timed_once(rows, order)? {
                return Ok(elapsed);
            }
            if self.gate_copies >= GATE_MAX_COPIES {
                return Err(failed(
                    "the timed run",
                    format!(
                        "the launches could not be queued behind a gate of {} copies",
                        self.gate_copies
                    ),
                ));
            }
            self.gate_copies *= 2;
        }
    }
}
