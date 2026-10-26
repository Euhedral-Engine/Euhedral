//! The settings the engine reads, one struct per group of `config/euhedral.example.yaml`.
//!
//! A value that is always present has its default here. A value the YAML may leave as `auto` is an [`Auto`]: the
//! settings crate only represents `auto`; the engine computes the value at run time.

use std::ops::RangeInclusive;
use std::path::PathBuf;
use std::time::Duration;

use crate::sizes::ByteSize;

/// A setting that is a value or `auto`, which the engine resolves at run time from the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Auto<T> {
    /// Computed at run time.
    Auto,
    /// Given in the file or the environment.
    Value(T),
}

// Derived, it would require `T: Default`; `auto` needs no value.
#[allow(clippy::derivable_impls)]
impl<T> Default for Auto<T> {
    fn default() -> Self {
        Self::Auto
    }
}

impl<T> Auto<T> {
    /// The given value, or `None` for `auto`.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Auto => None,
            Self::Value(value) => Some(value),
        }
    }

    /// The given value, or `None` for `auto`.
    pub fn into_value(self) -> Option<T> {
        match self {
            Self::Auto => None,
            Self::Value(value) => Some(value),
        }
    }

    /// Whether the value is left to run time.
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
}

/// Every deployer setting, read once at startup.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The model to serve.
    pub model: ModelSettings,
    /// Sampling for the fields a request leaves out.
    pub sampling: SamplingSettings,
    /// The HTTP server.
    pub server: ServerSettings,
    /// The context a sequence may hold.
    pub context: ContextSettings,
    /// The prompt prefix cache.
    pub prefix_cache: PrefixCacheSettings,
    /// How sequences share passes.
    pub batching: BatchingSettings,
    /// Memory limits and where layers run.
    pub memory: MemorySettings,
    /// CPUs and native files.
    pub host: HostSettings,
    /// Kernel thresholds.
    pub kernels: KernelSettings,
}

impl Settings {
    /// The defaults of every optional key, serving `model`.
    pub fn with_model(model: ModelSettings) -> Self {
        Self {
            model,
            sampling: SamplingSettings::default(),
            server: ServerSettings::default(),
            context: ContextSettings::default(),
            prefix_cache: PrefixCacheSettings::default(),
            batching: BatchingSettings::default(),
            memory: MemorySettings::default(),
            host: HostSettings::default(),
            kernels: KernelSettings::default(),
        }
    }
}

/// The model to serve (`model`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelSettings {
    /// The model artifact file.
    pub artifact: PathBuf,
    /// The model name clients send as `model`.
    pub name: String,
}

/// Sampling for the fields a request leaves out (`sampling`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingSettings {
    /// Softmax temperature; 0 is greedy decoding.
    pub temperature: f32,
    /// The most likely tokens kept; 0 keeps every token.
    pub top_k: u32,
    /// The probability mass kept, above 0 and at most 1.
    pub top_p: f32,
}

impl Default for SamplingSettings {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
        }
    }
}

/// The HTTP server (`server`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServerSettings {
    /// The port to listen on.
    pub port: u16,
    /// How long closing the engine waits for its workers to stop.
    pub shutdown_timeout: Duration,
    /// Limits on one request.
    pub requests: RequestSettings,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            port: 1738,
            shutdown_timeout: Duration::from_secs(10),
            requests: RequestSettings::default(),
        }
    }
}

/// Limits on one request (`server.requests`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestSettings {
    /// Tokens generated when a request names no limit.
    pub default_max_tokens: u32,
    /// Requests waiting for the engine before new ones are refused; 0 rejects any request that would wait.
    pub max_queued: u32,
    /// Largest request body.
    pub max_body_size: ByteSize,
    /// Longest a request may run.
    pub timeout: Duration,
}

impl Default for RequestSettings {
    fn default() -> Self {
        Self {
            default_max_tokens: 4096,
            max_queued: 16,
            max_body_size: ByteSize::mib(1),
            timeout: Duration::from_secs(30 * 60),
        }
    }
}

/// The format a sequence's key/value cache is stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvFormat {
    /// 4-bit floating point with block scales.
    Nvfp4,
    /// 16-bit brain floating point.
    Bf16,
}

/// The context a sequence may hold (`context`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSettings {
    /// Longest context of one sequence.
    pub max_tokens: u32,
    /// Format of the key/value cache; `auto` is the artifact's.
    pub kv_cache_format: Auto<KvFormat>,
    /// Tokens of a sequence kept on the GPU before its cache spills to the host.
    pub kv_cache_on_gpu_tokens: u32,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            max_tokens: 32768,
            kv_cache_format: Auto::Auto,
            kv_cache_on_gpu_tokens: 65536,
        }
    }
}

/// The prompt prefix cache (`prefix-cache`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrefixCacheSettings {
    /// Host memory for shared prompt prefixes; 0 disables the cache.
    pub size: ByteSize,
    /// Tokens between stored checkpoints; a positive multiple of 512.
    pub checkpoint_tokens: u32,
}

impl Default for PrefixCacheSettings {
    fn default() -> Self {
        Self {
            size: ByteSize::gib(4),
            checkpoint_tokens: 2048,
        }
    }
}

/// How sequences share passes (`batching`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchingSettings {
    /// Sequences held at once; 0 holds as many as fit.
    pub max_sequences: u32,
    /// Sequences in one shared decode or verification pass; `auto` is no bound.
    pub max_sequences_per_pass: Auto<u32>,
    /// Rows in one shared pass; `auto` is `prefill_chunk_tokens`.
    pub max_tokens_per_pass: Auto<u32>,
    /// Rows of the longest prefill chunk.
    pub prefill_chunk_tokens: u32,
}

impl Default for BatchingSettings {
    fn default() -> Self {
        Self {
            max_sequences: 1,
            max_sequences_per_pass: Auto::Auto,
            max_tokens_per_pass: Auto::Auto,
            prefill_chunk_tokens: 512,
        }
    }
}

/// Memory the engine may use (`memory`). An `auto` limit is resolved at run time from the machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemorySettings {
    /// Host memory the engine may use; 0 leaves no RAM tier.
    pub host_limit: Auto<ByteSize>,
    /// Pinned host memory for host-backed weights.
    pub host_weights_limit: Auto<ByteSize>,
    /// Device memory the engine may use; `auto` is what is free.
    pub gpu_limit: Auto<ByteSize>,
    /// Device memory left for other programs.
    pub gpu_reserve: ByteSize,
    /// Device memory the dense model's plan keeps beside the weights and the cache for the first sequence's state, the
    /// workspaces and the kernel modules; `auto` is 1280 MiB.
    pub runtime_reserve: Auto<ByteSize>,
    /// Released pinned buffers kept for reuse.
    pub pinned_buffer_pool: Auto<ByteSize>,
    /// Streaming of weights from disk.
    pub weight_streaming: WeightStreamingSettings,
    /// Layers whose weights stay in host memory.
    pub host_layers: HostLayersSettings,
}

impl Default for MemorySettings {
    fn default() -> Self {
        Self {
            host_limit: Auto::Auto,
            host_weights_limit: Auto::Auto,
            gpu_limit: Auto::Auto,
            gpu_reserve: ByteSize::mib(700),
            runtime_reserve: Auto::Auto,
            pinned_buffer_pool: Auto::Auto,
            weight_streaming: WeightStreamingSettings::default(),
            host_layers: HostLayersSettings::default(),
        }
    }
}

/// Streaming of weights from disk (`memory.weight-streaming`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightStreamingSettings {
    /// Artifact records read at once.
    pub concurrent_reads: u32,
    /// Parts one record read is split into.
    pub read_split: u32,
    /// CUDA streams copying expert weights to the GPU.
    pub copy_streams: u32,
    /// GPU streams a step runs on.
    pub gpu_streams: u32,
}

impl Default for WeightStreamingSettings {
    fn default() -> Self {
        Self {
            concurrent_reads: 8,
            read_split: 1,
            copy_streams: 4,
            gpu_streams: 2,
        }
    }
}

/// Where a layer whose weights live in host memory is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostLayerCompute {
    /// On the CPU, reading the weights in place.
    Cpu,
    /// On the GPU, with the weights streamed over PCIe.
    Stream,
}

/// Layers whose weights stay in host memory (`memory.host-layers`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HostLayersSettings {
    /// Where host-resident layers are computed; `auto` is chosen at run time.
    pub compute: Auto<HostLayerCompute>,
    /// Rows of one CPU GEMV chunk; `auto` is sized at run time from the cache and the row's byte size.
    pub gemv_rows: Auto<u32>,
    /// Staging slots for prefill of host-resident layers; `auto` is sized at run time.
    pub prefill_slots: Auto<u32>,
}

/// A list of processor IDs and inclusive ranges, such as `2-5,8`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuList(Vec<RangeInclusive<u32>>);

impl CpuList {
    /// Parses comma-separated processor IDs and ranges such as `2-5,8`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let invalid = || format!("must be a CPU list such as 2-5,8: {text}");
        let mut ranges = Vec::new();
        for part in text.split(',') {
            let (first, last) = part.split_once('-').unwrap_or((part, part));
            let number = |digits: &str| {
                if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                    return Err(invalid());
                }
                digits.parse::<u32>().map_err(|_| invalid())
            };
            let (first, last) = (number(first)?, number(last)?);
            if last < first {
                return Err(invalid());
            }
            ranges.push(first..=last);
        }
        Ok(Self(ranges))
    }

    /// The ranges as written, in order.
    pub fn ranges(&self) -> &[RangeInclusive<u32>] {
        &self.0
    }

    /// Every listed processor ID, in the order written.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().flat_map(|range| range.clone())
    }
}

/// The host CPUs and files the engine uses (`host`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostSettings {
    /// The CPUs workers run on; `auto` is all of them.
    pub cpus: Auto<CpuList>,
    /// Workers taking host work at once.
    pub task_parallelism: u32,
    /// Idle time before workers are released; zero never releases them.
    pub idle_release: Duration,
    /// The most frames a lattice worker takes from its cache at once; `auto` is the runtime's default. 1 or 2 makes the
    /// workers poll their caches one frame at a time, so that other workers can steal what waits behind a heavy frame.
    pub max_batch_size: Auto<u32>,
    /// Where native files are unpacked; `auto` is the user's cache directory.
    pub native_cache: Auto<PathBuf>,
}

impl Default for HostSettings {
    fn default() -> Self {
        Self {
            cpus: Auto::Auto,
            task_parallelism: 8,
            idle_release: Duration::from_secs(1),
            max_batch_size: Auto::Auto,
            native_cache: Auto::Auto,
        }
    }
}

/// Kernel thresholds (`kernels`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelSettings {
    /// The trellis kernels.
    pub trellis: TrellisKernelSettings,
    /// The CUDA graphs of captured quanta.
    pub graphs: GraphKernelSettings,
}

/// The CUDA graphs of captured quanta (`kernels.graphs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphKernelSettings {
    /// Graphs a sequence keeps for each view (decode, verification, draft) before the one admitted longest ago, or
    /// the one whose attention geometry the sequence has passed, is released.
    pub captures_per_view: u32,
}

impl Default for GraphKernelSettings {
    fn default() -> Self {
        Self { captures_per_view: 8 }
    }
}

/// The fewest rows of a launch kept on the tensor-core route.
pub const MIN_TENSOR_CORE_ROWS: u32 = 8;

/// Trellis kernel thresholds (`kernels.trellis`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrellisKernelSettings {
    /// Most rows of a non-verification launch kept on the tensor-core route; at least [`MIN_TENSOR_CORE_ROWS`].
    pub tensor_core_max_tokens: u32,
    /// Thread blocks the K split aims for; `auto` is 8 per streaming multiprocessor.
    pub target_blocks: Auto<u32>,
    /// Fewest rows for blocks that run every K split of their columns themselves.
    pub whole_k_min_tokens: u32,
    /// Fewest rows for blocks of 64 rows instead of 32.
    pub wide_block_min_tokens: u32,
}

impl Default for TrellisKernelSettings {
    fn default() -> Self {
        Self {
            tensor_core_max_tokens: 512,
            target_blocks: Auto::Auto,
            whole_k_min_tokens: 64,
            wide_block_min_tokens: 128,
        }
    }
}
