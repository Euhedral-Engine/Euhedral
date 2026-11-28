//! The device side of the `cost` command: one object at a time, loaded from a source artifact as copies on the card,
//! and its linear launched through the engine's own format for it, timed by device events.
//!
//! The launches of a timed run are queued behind a gate, a long device copy, so the host finishes queueing them while
//! the card is still busy and the time between the span's two events holds the launches and no host gap. A run is
//! valid only if the gate had not ended when the last launch was queued; a run that fails this is repeated behind a
//! longer gate ([`hide_launches`]).
//!
//! An object's bytes are read through the asynchronous I/O of the rig the command keeps up while it measures
//! ([`fetch`]), and uploaded once for each copy.

use std::fmt::Display;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use euhedral_gpu::cuda::{CudaGpu, CudaStream};
use euhedral_gpu::{Gpu, GpuError, Stream, StreamOptions};
use euhedral_graph::DeviceQueue;
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::{KernelTable, load};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, DevicePtr, KernelOps};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, Nvfp4Linear, ScratchRegion, TrellisLinear};
use euhedral_inference_engine::kernels::kernel_sources;
use euhedral_inference_engine::model::{Residence, TrellisParts, WeightRef};
use euhedral_inference_engine::runtime::open_device;
use euhedral_inference_quantization::assemble::{ObjectBytes, SourceSet, TrellisBytes, read_object};
use euhedral_inference_quantization::mixed::{Bench, MixedError, QuantOption, Shape, hide_launches, launch_mode};
use euhedral_inference_settings::Settings;
use euhedral_io::AsyncIo;

/// BF16 0.0078125 (2^-7), the bits the input is filled with: a normal, nonzero value, so the launches read real values
/// and not denormals or zeros.
const BF16_INPUT_FILL: u16 = 0x3C00;
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

/// Reads the object `name` of `option`'s source through `io`, from a thread that is not one of the lattice's workers,
/// and returns when it was read.
pub fn fetch(
    io: &Arc<AsyncIo>,
    sources: &SourceSet,
    option: QuantOption,
    name: &str,
) -> Result<ObjectBytes, MixedError> {
    let artifact = sources.artifact(option).map_err(|error| failed(name, error))?;
    let (sender, receiver) = mpsc::channel();
    read_object(
        io,
        artifact,
        name,
        Arc::new(move |read| {
            // The receiver is gone only when the run was abandoned.
            let _ = sender.send(read);
        }),
    )
    .map_err(|error| failed(name, error))?;
    receiver
        .recv()
        .map_err(|_| failed(name, "the read did not finish"))?
        .map_err(|error| failed(name, error))
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
    io: Arc<AsyncIo>,
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
}

impl<'a> DeviceBench<'a> {
    /// Opens the card `library` bound and loads the kernels the engine loads there under `settings`.
    pub fn open(
        library: &Arc<CudaLibrary>,
        settings: &Settings,
        sources: &'a SourceSet,
        io: &Arc<AsyncIo>,
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
            io: Arc::clone(io),
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
        })
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

    /// The weights of one copy of `object`, uploaded into `buffers`: `weight` is told the addresses of the copy.
    fn copy_of(
        &self,
        weight: &dyn Fn(DevicePtr, Option<TrellisParts>) -> WeightRef,
        object: &ObjectBytes,
        buffers: &mut Vec<DeviceBuffer>,
    ) -> Result<WeightRef, MixedError> {
        let body = self.upload(&object.bytes)?;
        let ptr = body.ptr();
        buffers.push(body);
        let trellis = match &object.trellis {
            None => None,
            Some(TrellisBytes {
                suh,
                svh,
                segments,
                segment_ends,
                suh_rows,
            }) => {
                let [suh, svh, segments] = [suh, svh, segments].map(|bytes| self.upload(bytes));
                let (suh, svh, segments) = (suh?, svh?, segments?);
                let parts = TrellisParts {
                    suh: suh.ptr(),
                    svh: svh.ptr(),
                    segments: segments.ptr(),
                    suh_rows: *suh_rows,
                    segment_ends: Arc::clone(segment_ends),
                };
                buffers.extend([suh, svh, segments]);
                Some(parts)
            }
        };
        Ok(weight(ptr, trellis))
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
        name: &str,
        shape: Shape,
        copies: usize,
        max_rows: u32,
    ) -> Result<(), MixedError> {
        self.settle()?;
        self.loaded = None;
        self.operands = None;
        let object = fetch(&self.io, self.sources, option, name)?;
        let sources = self.sources;
        let entry = sources.descriptor(option, name).map_err(|error| failed(name, error))?;
        let weight = |ptr: DevicePtr, trellis: Option<TrellisParts>| WeightRef {
            name: entry.name().to_owned(),
            shape: entry.shape().to_vec(),
            dtype: entry.dtype(),
            format: entry.format(),
            layout: entry.layout(),
            ptr,
            bytes: object.bytes.len() as u64,
            residence: Residence::Device,
            trellis,
        };
        let mut buffers = Vec::new();
        let mut weights = Vec::with_capacity(copies);
        for _ in 0..copies {
            weights.push(self.copy_of(&weight, &object, &mut buffers)?);
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
        let ones: Vec<u8> = (0..rows * input).flat_map(|_| BF16_INPUT_FILL.to_le_bytes()).collect();
        let operands = Operands {
            x: self.upload(&ones)?,
            y: DeviceBuffer::alloc(&self.library, (rows * output * 2).max(256))
                .map_err(|error| failed("the output", error))?,
            scratch: DeviceBuffer::alloc(&self.library, scratch.max(256))
                .map_err(|error| failed("the scratch", error))?,
        };
        self.settle()?;
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
        let (elapsed, copies) = hide_launches(self.gate_copies, GATE_MAX_COPIES, |copies| {
            self.gate_copies = copies;
            self.timed_once(rows, order)
        })?;
        self.gate_copies = copies;
        Ok(elapsed)
    }
}
