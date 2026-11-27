//! The queue of buffers a cache has given up, owned by the cuda crate and used by the engine and the GPTQ lane alike.

pub use euhedral_inference_cuda::released::{Released, ReleasedBuffers};
