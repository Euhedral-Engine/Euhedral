//! The quantization library: everything the quantization commands share, apart from argument parsing, which the
//! `euhedral-inference-quantize` binary owns.
//!
//! - [`java_random`]: `java.util.Random` and `java.util.SplittableRandom`, drawn exactly as Java draws them, so a seed
//!   picks the same numbers in both.
//! - [`npy`] and [`npz`]: the NumPy array and archive formats the calibration state and the fixtures use. Both work
//!   on bytes the caller supplies or takes; neither opens a file.
//! - [`kld`]: the reference logits file the scorer and the reference writer share, opened and read through `AsyncIo`.
//! - [`rows`]: the work on the logits rows of one teacher-forced run, as frames of the `host` lake.

pub mod java_random;
pub mod kld;
pub mod npy;
pub mod npz;
pub mod rows;
