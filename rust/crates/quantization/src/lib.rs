//! The quantization library: everything the quantization commands share, apart from argument parsing, which the
//! `euhedral-inference-quantize` binary owns.
//!
//! - [`java_random`]: seeded generators in the style of `java.util.Random` and `java.util.SplittableRandom`. A
//!   consumer that needs the exact sequence of a seed adds its own test of it.
//! - [`kernels`]: the quantization kernels as NVRTC modules, their kernel names and the loader that compiles them.
//! - [`training`]: the drafter's training operations: launch planners for the kernels and the FP32 products, and
//!   their enqueue on a lane.
//! - [`trellis`]: the trellis kernels (rates, launch planners and the enqueue of their steps) and the quantizer that
//!   chains them on a lane: scales, LDLQ through the Viterbi encode and the refit.
//! - [`npy`] and [`npz`]: the NumPy array and archive formats the calibration state and the fixtures use. Both work
//!   on bytes the caller supplies or takes; neither opens a file.
//! - [`forward`]: one decoder layer in BF16 on the engine's kernels, with the weights loaded for it and an observer of
//!   each quantized linear's input.
//! - [`memory`]: stream-ordered device buffers, pooled pinned staging and the claims that account for them, which
//!   the GPTQ rounding and the trellis quantizer share.
//! - [`kld`]: the reference logits file the scorer and the reference writer share, opened and read through `AsyncIo`.
//! - [`rows`]: the work on the logits rows of one teacher-forced run, as frames of the `host` lake.
//! - [`mixed`]: the mixed allocation search's data: options, allocations, priors, the cost table, the knapsack, the
//!   Pareto front and the residency check.
//! - [`assemble`]: the source artifacts of a mixed artifact, opened and checked, and the artifact an allocation makes of
//!   them, written through `AsyncIo`.
//! - [`checkpoint`]: the safetensors reader and writer and the fused objects of a Qwen 3.8 checkpoint, on `AsyncIo`.
//! - [`recipe`]: the storage format of each quantized object of an artifact.
//! - [`exl3`]: the bit allocation of the trellis quantizer and the writer of its EXL3 checkpoint.
//! - [`resume`]: the file a long run overwrites in place and resumes from, and [`durable`], the durability barrier,
//!   truncation and removal it is built on.
//! - [`package_tokenizer`]: gives an artifact a checkpoint's tokenizer, in place or as a copy.

pub mod assemble;
mod chain;
pub mod checkpoint;
pub mod durable;
pub mod exl3;
pub mod forward;
pub mod gptq;
pub mod java_random;
pub mod kernels;
pub mod kld;
pub mod memory;
pub mod mixed;
pub mod npy;
pub mod npz;
pub mod package_error;
pub mod package_tokenizer;
pub mod recipe;
pub mod reference;
pub mod resume;
pub mod rows;
pub mod step;
pub mod tokenizer_files;
pub mod training;
pub mod trellis;
