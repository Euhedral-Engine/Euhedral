//! The quantization library: everything the quantization commands share, apart from argument parsing, which the
//! `euhedral-inference-quantize` binary owns.
//!
//! - [`java_random`]: `java.util.Random` and `java.util.SplittableRandom`, drawn exactly as Java draws them, so a seed
//!   picks the same numbers in both.
//! - [`kernels`]: the quantization kernels as NVRTC modules, their kernel names and the loader that compiles them.
//! - [`npy`] and [`npz`]: the NumPy array and archive formats the calibration state and the fixtures use. Both work
//!   on bytes the caller supplies or takes; neither opens a file.

pub mod java_random;
pub mod kernels;
pub mod npy;
pub mod npz;
