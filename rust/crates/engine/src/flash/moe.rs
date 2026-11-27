//! The mixture of experts of a Flash-Next layer, on the device.
//!
//! One layer's block is these stages, laid out in [`crate::flash::plan`]: [`stages::MoeRoute`] runs the router and
//! copies its choice to the host; [`stages::MoePlan`], behind a device-completion edge from it, groups the pairs by
//! expert and copies the description to the device; for each expert the block can name, [`fetch::MoeFetch`] holds its
//! place until the expert is in the device cache and [`expert::MoeExpert`] runs its kernels and closes its lease
//! behind a marker of its lane; [`stages::MoeShared`] runs the shared expert beside them; and [`stages::MoeFinish`]
//! sums the experts' outputs in order and combines them with the shared expert's gated output. [`layer::MoeLayer`]
//! holds what they share (shape, weights, scratch, the source of experts) and lists each step's operations;
//! [`block::MoeBlock`] is what a quantum carries between them.

pub mod block;
pub mod expert;
pub mod fetch;
pub mod layer;
pub mod ops;
pub mod routing;
pub mod source;
pub mod stages;

#[cfg(test)]
mod testing;
