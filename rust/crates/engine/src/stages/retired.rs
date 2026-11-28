//! What a quantum or a prefix-cache copy hands to its place in its sequence's order of conclusion when it ends.
//!
//! The place ([`Slot`]) travels with the work: a [`Quantum`] holds one in each segment, and a copy job in its last
//! group. The frame that ends the work completes the place with its [`Retirement`]; work that is dropped without
//! having ended passes its place, so the work behind it still concludes.

use euhedral_gpu::Ran;
use euhedral_graph::{Marker, QuantumOutcome};

use super::sampling::SelectionLease;
use super::{Outcome, Quantum, QuantumKind};
use crate::state::FlightGuard;
use std::sync::Arc;
use std::time::Instant;

/// How a place in the order is completed.
pub(crate) enum Retirement {
    /// A quantum, or this sequence's part of a pass, retired.
    Quantum(Box<Retired>),
    /// A prefix-cache copy job ended.
    Copy,
}

/// What an owner keeps of a retired quantum until the work queued on its lane has run.
pub(crate) enum Kept {
    /// A quantum of the sequence alone.
    Whole(Box<Quantum>),
    /// A pass of several sequences, which every member's owner keeps (the last to let go frees its input record), and the
    /// selection lease of this member's part.
    Part {
        batch: Arc<Quantum>,
        _lease: Option<SelectionLease>,
        /// The part's count in flight on its sequence, which is its own: the sequence settles when this part is let go
        /// of, whatever the other members do.
        _flight: Option<FlightGuard>,
    },
}

impl Kept {
    /// The marker of the upload of the quantum's input record.
    pub(crate) fn prepared(&self) -> Option<Marker> {
        match self {
            Kept::Whole(quantum) => quantum.prepared,
            Kept::Part { batch, .. } => batch.prepared,
        }
    }
}

/// A retired quantum (or this sequence's part of a pass) and how it ended for this sequence.
pub(crate) struct Retired {
    /// The number of the quantum that ran, which for a part is the pass's.
    pub(crate) serial: u64,
    pub(crate) start: u64,
    pub(crate) kind: QuantumKind,
    pub(crate) rows: usize,
    /// The rows of the quantum this sequence commits when it completed.
    pub(crate) committed: usize,
    pub(crate) ran: Option<Ran>,
    /// Whether the quantum is a pass of several sequences, which the owner of the workspace was told retired before this.
    pub(crate) pass: bool,
    pub(crate) kept: Kept,
    pub(crate) outcome: QuantumOutcome,
    /// When the quantum retired.
    pub(crate) at: Instant,
}

impl Retired {
    /// A quantum of one sequence that retired.
    pub(crate) fn whole(quantum: Quantum, outcome: QuantumOutcome) -> Retired {
        Retired {
            serial: quantum.id(),
            start: quantum.start_position(),
            kind: quantum.kind(),
            rows: quantum.rows(),
            committed: quantum.rows(),
            ran: quantum.graph_run(),
            pass: false,
            kept: Kept::Whole(Box::new(quantum)),
            outcome,
            at: Instant::now(),
        }
    }

    /// Segment `segment` of the pass `batch` that retired as `outcome`: how it ended for its own sequence (a member that
    /// was cancelled concludes cancelled, whatever the pass did), with the lease its owner gives back.
    pub(crate) fn part(
        batch: &Arc<Quantum>,
        segment: usize,
        lease: Option<SelectionLease>,
        flight: Option<FlightGuard>,
        outcome: &QuantumOutcome,
    ) -> Retired {
        let concluded = match batch.conclusion_for(segment, outcome) {
            Ok(Outcome::Completed) => QuantumOutcome::Completed,
            Ok(Outcome::Cancelled) => QuantumOutcome::Killed,
            Ok(Outcome::Failed(cause)) => QuantumOutcome::DeviceFailed { cause },
            Err(error) => QuantumOutcome::DeviceFailed {
                cause: error.to_string(),
            },
        };
        let (start, rows) = batch
            .segments()
            .get(segment)
            .map_or((0, 0), |part| (part.start_position(), part.rows()));
        Retired {
            serial: batch.id(),
            start,
            kind: batch.kind(),
            rows,
            committed: batch.committed_rows(segment).unwrap_or(0),
            ran: None,
            pass: true,
            kept: Kept::Part {
                batch: Arc::clone(batch),
                _lease: lease,
                _flight: flight,
            },
            outcome: concluded,
            at: Instant::now(),
        }
    }
}
