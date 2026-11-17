//! Choosing a token from one row of logits on the host: temperature, top-k, top-p, a seed keyed by the token's
//! position, greedy selection and selection under a filter that says which tokens may be chosen.
//!
//! The result is fixed by the order of the floating-point operations and by the layout of the heaps, so the loops
//! here are written to be followed line by line against the reference selection they reproduce. Nothing in this
//! module touches the device, a thread or a lock; a sampler is owned by one request and its scratch rows are
//! allocated once, when it is made.

mod config;
mod sampler;

#[cfg(test)]
mod random;
#[cfg(test)]
mod reference;
#[cfg(test)]
mod tests;

pub use config::{SamplingConfig, SamplingError};
pub use sampler::{TokenFilter, TokenSampler, unit_draw};
