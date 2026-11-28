//! Which quantum of a prompt chunk carries a checkpoint, and what a restore sets the speculative state to. Pure
//! functions of the request's draft kind and step.

use super::mtp::Speculating;
use super::source::Step;
use crate::prefix::Speculation as CheckpointKind;

/// How a request drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DraftKind {
    /// Does not speculate.
    Plain,
    /// Speculates on the MTP route without drafting.
    Idle,
    /// Drafts with the MTP layer.
    Mtp,
    /// Speculates on the DFlash2 route; `drafting` tells whether it drafts.
    Dflash2 { drafting: bool },
}

impl DraftKind {
    /// The kind of draft state the request's checkpoints hold beside the base state; `None` when it does not draft.
    pub(super) fn caching(self) -> Option<CheckpointKind> {
        match self {
            DraftKind::Mtp => Some(CheckpointKind::Mtp),
            DraftKind::Dflash2 { drafting: true } => Some(CheckpointKind::Dflash2),
            DraftKind::Plain | DraftKind::Idle | DraftKind::Dflash2 { drafting: false } => None,
        }
    }

    /// Whether the checkpoints hold the draft's state.
    fn drafting(self) -> bool {
        self.caching().is_some()
    }
}

/// The kind of the request whose speculative state is `state`, `None` for a request that does not speculate.
pub(super) fn kind_of(state: Option<&Speculating>) -> DraftKind {
    state.map_or(DraftKind::Plain, Speculating::draft_kind)
}

/// The quantum of a prompt chunk that carries the checkpoint at its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Carrier {
    /// The base prefill.
    Prefill,
    /// The draft update (the MTP catch-up or the DFlash2 context update).
    DraftUpdate,
    /// No quantum.
    Nothing,
}

/// The carrier of a prompt chunk; `last` is whether it ends the prompt.
pub(super) fn carrier(kind: DraftKind, last: bool) -> Carrier {
    match (kind, last) {
        (DraftKind::Plain, _) => Carrier::Prefill,
        (_, true) => Carrier::Nothing,
        (kind, false) if kind.drafting() => Carrier::DraftUpdate,
        (_, false) => Carrier::Prefill,
    }
}

/// Whether `step` of a request of `kind` carries the checkpoint at its end: a prompt chunk whose carrier is the
/// prefill.
pub(super) fn prefill_carries(kind: DraftKind, step: Step) -> bool {
    match step {
        Step::Prefill { last, .. } => carrier(kind, last) == Carrier::Prefill,
        _ => false,
    }
}

/// Whether an MTP catch-up captures: the checkpoint is `wanted` and the catch-up is not the re-pairing of a restore
/// (`resuming`).
pub(super) fn catch_up_captures(wanted: bool, resuming: bool) -> bool {
    wanted && !resuming
}

/// What a restore sets the speculative state to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Resume {
    /// Nothing.
    Nothing,
    /// The MTP cache is restored; `whole` when its last row is paired with the prompt's token.
    Mtp { whole: bool },
    /// The DFlash2 drafter's context resumes at the position.
    Drafter,
}

/// The state a restore leaves: `paired` is the token the checkpoint's last row was paired with, `token` the prompt's
/// token at the checkpoint's position.
pub(super) fn resume(kind: DraftKind, paired: Option<u32>, token: Option<u32>) -> Resume {
    match kind {
        DraftKind::Plain | DraftKind::Idle => Resume::Nothing,
        DraftKind::Mtp => Resume::Mtp {
            whole: paired.is_some() && token == paired,
        },
        DraftKind::Dflash2 { .. } => Resume::Drafter,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::mtp::Speculating;

    const DRAFTING: [DraftKind; 2] = [DraftKind::Mtp, DraftKind::Dflash2 { drafting: true }];
    const NOT_DRAFTING: [DraftKind; 2] = [DraftKind::Idle, DraftKind::Dflash2 { drafting: false }];

    #[test]
    fn a_request_caches_the_draft_state_of_the_kind_it_drafts_with_and_none_otherwise() {
        assert_eq!(DraftKind::Mtp.caching(), Some(CheckpointKind::Mtp));
        assert_eq!(
            DraftKind::Dflash2 { drafting: true }.caching(),
            Some(CheckpointKind::Dflash2)
        );
        for kind in NOT_DRAFTING.into_iter().chain([DraftKind::Plain]) {
            assert_eq!(kind.caching(), None, "{kind:?}");
        }
    }

    #[test]
    fn a_plain_request_captures_every_chunk_after_its_prefill() {
        assert_eq!(carrier(DraftKind::Plain, false), Carrier::Prefill);
        assert_eq!(carrier(DraftKind::Plain, true), Carrier::Prefill);
    }

    #[test]
    fn a_drafting_request_captures_after_the_draft_update_never_after_the_prefill() {
        for kind in DRAFTING {
            assert_eq!(carrier(kind, false), Carrier::DraftUpdate, "{kind:?}");
        }
    }

    #[test]
    fn a_speculative_request_never_captures_its_last_chunk() {
        for kind in DRAFTING.into_iter().chain(NOT_DRAFTING) {
            assert_eq!(carrier(kind, true), Carrier::Nothing, "{kind:?}");
        }
    }

    #[test]
    fn a_speculative_request_that_does_not_draft_captures_after_the_prefill() {
        for kind in NOT_DRAFTING {
            assert_eq!(carrier(kind, false), Carrier::Prefill, "{kind:?}");
        }
    }

    #[test]
    fn the_catch_up_that_re_pairs_a_restored_row_captures_nothing() {
        assert!(catch_up_captures(true, false));
        assert!(!catch_up_captures(true, true));
        assert!(!catch_up_captures(false, false));
        assert!(!catch_up_captures(false, true));
    }

    #[test]
    fn a_restore_resumes_the_dflash2_drafter_drafting_or_not() {
        for drafting in [true, false] {
            let kind = DraftKind::Dflash2 { drafting };
            assert_eq!(resume(kind, None, Some(7)), Resume::Drafter);
            assert_eq!(resume(kind, Some(7), Some(7)), Resume::Drafter);
        }
    }

    #[test]
    fn a_restore_keeps_the_mtp_last_row_only_when_the_prompt_has_the_paired_token() {
        assert_eq!(resume(DraftKind::Mtp, Some(7), Some(7)), Resume::Mtp { whole: true });
        assert_eq!(resume(DraftKind::Mtp, Some(7), Some(8)), Resume::Mtp { whole: false });
        assert_eq!(resume(DraftKind::Mtp, Some(7), None), Resume::Mtp { whole: false });
        assert_eq!(resume(DraftKind::Mtp, None, None), Resume::Mtp { whole: false });
        assert_eq!(resume(DraftKind::Mtp, None, Some(7)), Resume::Mtp { whole: false });
    }

    #[test]
    fn a_restore_sets_nothing_for_a_request_without_a_draft() {
        for kind in [DraftKind::Plain, DraftKind::Idle] {
            assert_eq!(resume(kind, Some(7), Some(7)), Resume::Nothing, "{kind:?}");
        }
    }

    const CHUNK: Step = Step::Prefill { rows: 512, last: false };
    const LAST: Step = Step::Prefill { rows: 40, last: true };

    #[test]
    fn only_a_prompt_chunk_whose_carrier_is_the_prefill_carries_at_the_prefill() {
        for kind in [
            DraftKind::Plain,
            DraftKind::Idle,
            DraftKind::Dflash2 { drafting: false },
        ] {
            assert!(prefill_carries(kind, CHUNK), "{kind:?}");
        }
        for kind in DRAFTING {
            assert!(!prefill_carries(kind, CHUNK), "{kind:?}");
        }
        assert!(prefill_carries(DraftKind::Plain, LAST));
        for kind in DRAFTING.into_iter().chain(NOT_DRAFTING) {
            assert!(!prefill_carries(kind, LAST), "{kind:?}");
        }
    }

    #[test]
    fn a_step_that_is_not_a_prompt_chunk_carries_nothing_at_the_prefill() {
        for step in [
            Step::Decode,
            Step::PromptCatchUp { rows: 512 },
            Step::Speculate { drafts: 3 },
        ] {
            assert!(!prefill_carries(DraftKind::Plain, step), "{step:?}");
        }
    }

    #[test]
    fn the_kind_of_a_request_is_plain_without_speculative_state() {
        assert_eq!(kind_of(None), DraftKind::Plain);
        assert_eq!(kind_of(Some(&Speculating::new(3, 4096, 3000, 64))), DraftKind::Mtp);
        assert_eq!(
            kind_of(Some(&Speculating::new(3, 4096, 3000, 64).with_drafter())),
            DraftKind::Dflash2 { drafting: true }
        );
    }
}
