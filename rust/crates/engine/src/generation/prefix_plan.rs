//! The owner's choices about the prefix cache that depend on how the request kind, as pure functions of the run's
//! mode and step: which quantum of a prompt chunk carries the checkpoint, and what a restore sets the speculative state
//! to. The owner calls them where it admits a chunk and where it restores a hit.

/// How the run speculates, as far as its checkpoints are concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DraftKind {
    /// The request does not speculate.
    Plain,
    /// The request speculates on the MTP route but does not draft (one token wanted, or no room for a row).
    Idle,
    /// The MTP layer kind: its cache is part of the checkpoint.
    Mtp,
    /// The route is DFlash2; `drafting` tells whether the drafter's rings are part of the checkpoint.
    Dflash2 { drafting: bool },
}

impl DraftKind {
    /// Whether the checkpoints of the run hold the draft's state.
    fn drafting(self) -> bool {
        matches!(self, DraftKind::Mtp | DraftKind::Dflash2 { drafting: true })
    }
}

/// The quantum of a prompt chunk that carries the checkpoint at the chunk's end.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Carrier {
    /// The chunk's base prefill.
    Prefill,
    /// The drafter's update of the chunk (the MTP catch-up, or the DFlash2 context update), once its rows are in the
    /// draft's state.
    DraftUpdate,
    /// No quantum of the chunk: the prompt's last chunk of a speculative request is never captured (its first step
    /// follows at once).
    Nothing,
}

/// The carrier of a prompt chunk, `last` when it is the prompt's last.
pub(super) fn carrier(kind: DraftKind, last: bool) -> Carrier {
    match (kind, last) {
        (DraftKind::Plain, _) => Carrier::Prefill,
        (_, true) => Carrier::Nothing,
        (kind, false) if kind.drafting() => Carrier::DraftUpdate,
        (_, false) => Carrier::Prefill,
    }
}

/// Whether an MTP catch-up that completes a chunk ending at a wanted checkpoint captures it. The catch-up that
/// re-pairs a restored checkpoint's last row (`resuming`) completes no chunk.
pub(super) fn catch_up_captures(wanted: bool, resuming: bool) -> bool {
    wanted && !resuming
}

/// What a restore sets the speculative state to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Resume {
    /// The state has nothing of the draft's to set.
    Nothing,
    /// The MTP layer's cache was restored: whole, with the checkpoint's last row paired with the prompt's token, or
    /// short of that row, which a catch-up of one row re-pairs.
    Mtp { whole: bool },
    /// The DFlash2 drafter's rings were restored whole; the drafter's context resumes at the position.
    Drafter,
}

/// The state a restore of a checkpoint whose last row was paired with `paired` leaves, when the prompt has `token` at
/// the checkpoint's position.
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

    const DRAFTING: [DraftKind; 2] = [DraftKind::Mtp, DraftKind::Dflash2 { drafting: true }];
    const NOT_DRAFTING: [DraftKind; 2] = [DraftKind::Idle, DraftKind::Dflash2 { drafting: false }];

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
}
