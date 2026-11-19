//! The quantization library: everything the quantization commands share, apart from argument parsing, which the
//! `euhedral-inference-quantize` binary owns.
//!
//! - [`java_random`]: seeded generators in the style of `java.util.Random` and `java.util.SplittableRandom`. A
//!   consumer that needs the exact sequence of a seed adds its own test of it.
//! - [`npy`] and [`npz`]: the NumPy array and archive formats the calibration state and the fixtures use. Both work
//!   on bytes the caller supplies or takes; neither opens a file.

pub mod java_random;
pub mod npy;
pub mod npz;
