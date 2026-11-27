//! The dense part of a Flash-Next layer on the device: the hyper-connection mix and inject around each block, the GDN
//! block with the Flash-Next convolution, the embedding over the residual streams, and the per-layer n-gram embedding
//! with its host roots.
//!
//! This is the port of `core/.../model/qwen4/{Ops,HyperConnection,GdnLayer,Ple,NgramIds,Stages}.java`. [`ops`] holds the
//! launches of the Flash-Next kernel table, one function per Java `Ops` entry with the same grid, block and argument
//! order; [`hyper`], [`gdn`], [`embed`] and [`ple`] are the stages; [`plan`] lays them out as the nodes and edges of one
//! layer. The stages run on the dense [`crate::stages::Quantum`] and its lanes: they read its rows and tokens and hold
//! the device addresses of their own weights, scratch and state ([`buffers`]), which a session supplies.

/// Makes `$stage` a `Stage<Quantum>` that runs eagerly, as the dense stages do.
macro_rules! eager {
    ($($stage:ty),+ $(,)?) => {$(
        impl euhedral_graph::Stage<crate::stages::Quantum> for $stage {
            fn run(
                &self,
                quantum: &crate::stages::Quantum,
                cx: &euhedral_graph::StageCx<'_>,
            ) -> Result<euhedral_graph::Submitted, euhedral_graph::StageError> {
                crate::stages::run_eager(self, quantum, cx)
            }
        }
    )+};
}

pub mod block;
pub mod buffers;
pub mod embed;
pub mod gdn;
pub mod hyper;
pub mod kernels;
pub mod linear;
pub mod ops;
pub mod plan;
pub mod ple;
pub mod sequence;

use euhedral_inference_cuda::DevicePtr;
use thiserror::Error;

/// A launch or a stage the Flash-Next kernels refuse, with the value that broke the rule.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FlashError {
    #[error("{op}: {what}")]
    Refused { op: &'static str, what: String },
}

impl FlashError {
    pub(crate) fn refused(op: &'static str, what: impl Into<String>) -> Self {
        Self::Refused { op, what: what.into() }
    }
}

/// A weight tensor on the device: its address and its size in bytes (the NVFP4 linears take both).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weight {
    pub address: DevicePtr,
    pub bytes: u64,
}
