//! The views of the execution plan: which stage graph a quantum of a kind and a row count runs
//! (`ExecutionPlan.forExecution`).

use crate::stages::QuantumKind;

/// Most rows a quantum runs on the decode kernels, which read quantized weights in place without scratch; a draft
/// quantum of more rows is a catch-up (`ExecutionPlan.DECODE_MAX_ROWS`).
pub const DECODE_MAX_ROWS: usize = 8;

/// Most rows of a quantum that is captured into a CUDA graph (`Quantum.MAX_CAPTURED_ROWS`).
pub const MAX_CAPTURED_ROWS: usize = 8;

/// The view a quantum runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum View {
    /// A chunk of the prompt: the fused topology with the multi-row scratch rules, whatever its rows.
    PrefillChunk,
    /// One row of the sequence.
    Decode,
    /// Speculative verification of `rows` rows. Up to [`DECODE_MAX_ROWS`] it runs the decode view; more rows use the
    /// view whose quantized linears take the expansion scratch.
    Verify { rows: usize },
    /// An MTP draft over `rows` rows, at most [`DECODE_MAX_ROWS`]: a recursive draft row, or a catch-up that fits the
    /// decode kernels.
    Draft { rows: usize },
    /// An MTP catch-up over more than [`DECODE_MAX_ROWS`] rows, whose quantized linears take the expansion scratch.
    CatchUp { rows: usize },
}

impl View {
    /// The view of a quantum of `kind` over `rows` rows.
    pub fn of(kind: QuantumKind, rows: usize) -> View {
        match kind {
            QuantumKind::Prefill => View::PrefillChunk,
            QuantumKind::Decode => View::Decode,
            QuantumKind::Verify => View::Verify { rows },
            QuantumKind::Draft if rows <= DECODE_MAX_ROWS => View::Draft { rows },
            QuantumKind::Draft => View::CatchUp { rows },
        }
    }

    /// The rows of the view, if it names them.
    pub fn rows(self) -> Option<usize> {
        match self {
            View::PrefillChunk => None,
            View::Decode => Some(1),
            View::Verify { rows } | View::Draft { rows } | View::CatchUp { rows } => Some(rows),
        }
    }

    /// The stage graph the view runs.
    pub(crate) fn variant(self) -> Variant {
        match self {
            View::PrefillChunk => Variant::Prefill,
            View::Decode => Variant::Decode,
            View::Verify { rows } if rows <= DECODE_MAX_ROWS => Variant::Verify,
            View::Verify { .. } => Variant::VerifyWide,
            View::Draft { .. } => Variant::Draft,
            View::CatchUp { .. } => Variant::CatchUp,
        }
    }

    /// Whether quanta of this view are captured into graphs (when nothing else rules a quantum out): decode,
    /// verification and draft quanta of at most [`MAX_CAPTURED_ROWS`] rows.
    pub fn capturable(self) -> bool {
        match self {
            View::PrefillChunk | View::CatchUp { .. } => false,
            View::Decode => true,
            View::Verify { rows } | View::Draft { rows } => rows <= MAX_CAPTURED_ROWS,
        }
    }
}

/// A stage graph of the plan. Decode and verification up to eight rows share their topology and their scratch rules,
/// as Java's one decode view does; they differ in the rows their selection reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Variant {
    Decode,
    Verify,
    VerifyWide,
    Prefill,
    Draft,
    CatchUp,
}

impl Variant {
    /// Every variant, in the order the plan stores them.
    pub(crate) const ALL: [Variant; 6] = [
        Variant::Decode,
        Variant::Verify,
        Variant::VerifyWide,
        Variant::Prefill,
        Variant::Draft,
        Variant::CatchUp,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// A view that runs this variant.
    pub(crate) fn representative(self) -> View {
        match self {
            Variant::Decode => View::Decode,
            Variant::Verify => View::Verify { rows: 2 },
            Variant::VerifyWide => View::Verify {
                rows: DECODE_MAX_ROWS + 1,
            },
            Variant::Prefill => View::PrefillChunk,
            Variant::Draft => View::Draft { rows: 1 },
            Variant::CatchUp => View::CatchUp {
                rows: DECODE_MAX_ROWS + 1,
            },
        }
    }

    /// Whether the variant is the MTP layer's (the unfused topology of `mtpDraft`), not the base model's.
    pub(crate) fn mtp(self) -> bool {
        matches!(self, Variant::Draft | Variant::CatchUp)
    }

    /// Whether a quantized linear of a format that takes scratch only on many rows declares it: every view but the
    /// decode and MTP draft views (`Shape.scratchUse`'s `multiRow`).
    pub(crate) fn multi_row(self) -> bool {
        !matches!(self, Variant::Decode | Variant::Verify | Variant::Draft)
    }

    /// Whether the variant runs beside decode on the tensor-core routes only, which gives the value/z projection its
    /// own scratch region (`Shape.scratchRegion`).
    pub(crate) fn decode_type(self) -> bool {
        matches!(
            self,
            Variant::Decode | Variant::Verify | Variant::VerifyWide | Variant::Draft
        )
    }

    /// Whether the variant's selection covers every logits row (a verification), not the last one.
    pub(crate) fn selects_each(self) -> bool {
        matches!(self, Variant::Verify | Variant::VerifyWide)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quantum_runs_the_view_java_selects() {
        assert_eq!(View::of(QuantumKind::Prefill, 512), View::PrefillChunk);
        assert_eq!(View::of(QuantumKind::Prefill, 3), View::PrefillChunk);
        assert_eq!(View::of(QuantumKind::Decode, 1), View::Decode);
        assert_eq!(View::of(QuantumKind::Verify, 5), View::Verify { rows: 5 });
        assert_eq!(View::of(QuantumKind::Draft, 8), View::Draft { rows: 8 });
        assert_eq!(View::of(QuantumKind::Draft, 9), View::CatchUp { rows: 9 });
    }

    #[test]
    fn verification_up_to_eight_rows_runs_the_decode_topology() {
        assert_eq!(View::Verify { rows: 2 }.variant(), Variant::Verify);
        assert_eq!(View::Verify { rows: 8 }.variant(), Variant::Verify);
        assert_eq!(View::Verify { rows: 9 }.variant(), Variant::VerifyWide);
        assert!(!Variant::Decode.multi_row() && !Variant::Verify.multi_row() && !Variant::Draft.multi_row());
        assert!(Variant::VerifyWide.multi_row() && Variant::Prefill.multi_row() && Variant::CatchUp.multi_row());
    }

    #[test]
    fn only_decode_verification_and_small_drafts_are_capturable() {
        assert!(View::Decode.capturable());
        assert!(View::Verify { rows: 8 }.capturable());
        assert!(!View::Verify { rows: 9 }.capturable());
        assert!(View::Draft { rows: 1 }.capturable());
        assert!(!View::CatchUp { rows: 9 }.capturable());
        assert!(!View::PrefillChunk.capturable());
    }
}
