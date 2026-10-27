//! Exact-match acceptance of one speculative verification (`SpeculativeAcceptance`, `docs/speculation/MTP_CONTRACT.md`
//! section 4).
//!
//! The verified rows are `[t0, d1 .. dn]`: the last committed token, then the drafts. Row `j`'s selected token `g_j` is
//! the model's next token after rows `0..=j`, and a draft is accepted while `d_{j+1} == g_j`. The outputs are the
//! accepted drafts followed by the first `g` that no draft matched, which is exactly the tokens ordinary decode
//! produces. They are cut after the first end-of-generation token and at the budget.
//!
//! The rows the step commits equal the output count: row 0 carries `t0` and rows `1..count` the first `count - 1`
//! outputs, which one-row decode would have fed. The last output (the next `t0`, or the end token) is not fed yet.

use crate::stages::sampling::token_of;

/// How a verification resolved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Accepted {
    /// Tokens the step commits to the output, and so rows it commits to the cache: the accepted drafts and the
    /// verifier's token, cut after the first end token and at the budget.
    pub count: usize,
    /// The index of the end token among the outputs when the cut fell on one (always `count - 1`).
    pub eos_at: Option<usize>,
    /// Drafts the verifier confirmed (0 to the number of drafts), before any end-token or budget cut.
    pub accepted_drafts: usize,
    /// The tokens committed, in order (`count` of them).
    pub outputs: Vec<u32>,
    /// Whether the verifier selected no token where the step needed one (every logit of the row was NaN or negative
    /// infinity): the outputs end before it.
    pub unselected: bool,
}

/// Resolves a verification of `drafts` from the argmax key of each verified row (`verified_keys[j]` is row `j`'s, so
/// there is one more key than drafts): `eos` are the ids that end a generation and `budget` caps the outputs.
///
/// # Panics
///
/// If `verified_keys` is not one longer than `drafts`, or `budget` is 0 (a step is admitted only while tokens are
/// wanted).
pub fn accept(drafts: &[u32], verified_keys: &[u64], eos: &[u32], budget: u32) -> Accepted {
    assert_eq!(
        verified_keys.len(),
        drafts.len() + 1,
        "one verified key per row: the current token and each draft"
    );
    assert!(budget > 0, "a step is admitted while tokens are wanted");
    let mut accepted = 0;
    while accepted < drafts.len() && token_of(verified_keys[accepted]) == Some(drafts[accepted]) {
        accepted += 1;
    }
    let mut outputs: Vec<u32> = drafts[..accepted].to_vec();
    let unselected = match token_of(verified_keys[accepted]) {
        Some(token) => {
            outputs.push(token);
            false
        }
        None => true,
    };
    outputs.truncate(budget as usize);
    let mut eos_at = None;
    if let Some(at) = outputs.iter().position(|token| eos.contains(token)) {
        outputs.truncate(at + 1);
        eos_at = Some(at);
    }
    // A verifier row with no token ends the outputs only when nothing cut them before it.
    let unselected = unselected && outputs.len() == accepted && eos_at.is_none() && accepted < budget as usize;
    Accepted {
        count: outputs.len(),
        eos_at,
        accepted_drafts: accepted,
        outputs,
        unselected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 99;

    /// The key the device's argmax writes for `token`.
    fn key(token: u32) -> u64 {
        (0x3f80 << 32) | u64::from(0xFFFF_FFFF - token)
    }

    fn keys(tokens: &[u32]) -> Vec<u64> {
        tokens.iter().map(|token| key(*token)).collect()
    }

    /// `SpeculativeAcceptanceTest.resolve`: rows `[t0, d1..]` and the selected token of each row.
    fn resolve(rows: &[u32], selected: &[u32], budget: u32) -> Accepted {
        accept(&rows[1..], &keys(selected), &[EOS], budget)
    }

    #[test]
    fn acceptance_counts_prefix_matches() {
        // Rows [t0, d1, d2, d3]; selections g0..g3.
        let none = resolve(&[5, 6, 7, 8], &[1, 2, 3, 4], 16);
        assert_eq!(none.accepted_drafts, 0);
        assert_eq!(none.outputs, [1]);
        assert_eq!(none.count, 1);

        let two = resolve(&[5, 6, 7, 8], &[6, 7, 3, 4], 16);
        assert_eq!(two.accepted_drafts, 2);
        assert_eq!(two.outputs, [6, 7, 3]);
        assert_eq!(two.count, 3);

        let all = resolve(&[5, 6, 7, 8], &[6, 7, 8, 9], 16);
        assert_eq!(all.accepted_drafts, 3);
        assert_eq!(all.outputs, [6, 7, 8, 9], "the full step adds the verifier's token");
        assert_eq!(all.count, 4);
        assert!(!all.unselected && all.eos_at.is_none());

        // A draft that matches only after a mismatch is not accepted.
        let late = resolve(&[5, 6, 7, 8], &[6, 0, 8, 9], 16);
        assert_eq!(late.accepted_drafts, 1);
        assert_eq!(late.outputs, [6, 0]);

        // No drafts: one row, one output.
        let single = accept(&[], &keys(&[3]), &[EOS], 16);
        assert_eq!((single.accepted_drafts, single.outputs.as_slice()), (0, [3].as_slice()));
    }

    #[test]
    fn acceptance_cuts_at_eos() {
        let accepted_end = resolve(&[5, EOS, 7, 8], &[EOS, 7, 8, 9], 16);
        assert_eq!(
            accepted_end.accepted_drafts, 3,
            "the cut comes after the count of accepted drafts"
        );
        assert_eq!(accepted_end.outputs, [EOS], "the end token is never fed back");
        assert_eq!((accepted_end.count, accepted_end.eos_at), (1, Some(0)));

        let base_end = resolve(&[5, 6, 7, 8], &[6, EOS, 1, 2], 16);
        assert_eq!(base_end.outputs, [6, EOS]);
        assert_eq!((base_end.count, base_end.eos_at), (2, Some(1)));

        // Only the first end token counts.
        let twice = resolve(&[5, 6, EOS, 8], &[6, EOS, EOS, 2], 16);
        assert_eq!(twice.outputs, [6, EOS]);
        assert_eq!(twice.eos_at, Some(1));
    }

    #[test]
    fn acceptance_respects_budget() {
        let budget = resolve(&[5, 6, 7, 8], &[6, 7, 8, 9], 2);
        assert_eq!(budget.outputs, [6, 7]);
        assert_eq!(budget.count, 2);
        assert_eq!(
            budget.accepted_drafts, 3,
            "the budget cuts the outputs, not the verifier's count"
        );
        // An end token past the budget is not reached.
        let past = resolve(&[5, 6, 7, 8], &[6, 7, EOS, 9], 2);
        assert_eq!(past.outputs, [6, 7]);
        assert_eq!(past.eos_at, None);
        // An end token at the budget is.
        let at = resolve(&[5, 6, 7, 8], &[6, EOS, 8, 9], 2);
        assert_eq!(at.outputs, [6, EOS]);
        assert_eq!(at.eos_at, Some(1));
        // A budget of one commits the verifier's first token only.
        let one = resolve(&[5, 6, 7, 8], &[6, 7, 8, 9], 1);
        assert_eq!(one.outputs, [6]);
    }

    #[test]
    fn a_row_with_no_selectable_token_ends_the_outputs() {
        // The verifier's token after the accepted drafts is missing.
        let drafts = [6, 7];
        let mut selected = keys(&[6, 7]);
        selected.push(0);
        let missing = accept(&drafts, &selected, &[EOS], 16);
        assert_eq!(missing.outputs, [6, 7]);
        assert!(missing.unselected);
        // A missing row past a mismatch is never needed.
        let mut after = keys(&[1]);
        after.extend([0, 0]);
        let fine = accept(&drafts, &after, &[EOS], 16);
        assert_eq!(fine.outputs, [1]);
        assert!(!fine.unselected);
        // The budget cut before the missing row hides it.
        let cut = accept(&drafts, &selected, &[EOS], 2);
        assert_eq!(cut.outputs, [6, 7]);
        assert!(!cut.unselected);
    }
}
