//! What a window's device inputs and the host's checks are, from the sequence's tokens and the base model's choices:
//! the input token of every draft, the draft-head row each draft is trained towards, and the row it is checked against.

use super::progress::{NO_TARGET, OUTSIDE_HEAD};
use super::settings::Window;

/// One sequence of the corpus.
#[derive(Clone, Copy, Debug)]
pub struct Sequence<'a> {
    /// Its tokens, the prompt's and the answer's.
    pub tokens: &'a [i32],
    /// The base model's greedy choice at each position.
    pub labels: &'a [i32],
    /// How many of the tokens are the prompt.
    pub prompt: usize,
}

/// A window's inputs, each depth-major `[depths][count]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Staged {
    /// The input token of each draft: the token at the draft's position plus its depth, 0 past the sequence.
    pub ids: Vec<i32>,
    /// The draft-head row of the base model's choice at the draft's target, -1 where the target is not an answer token or
    /// is outside the head: the rows the device trains towards.
    pub labels: Vec<i32>,
    /// The row each draft is checked against: the labels when training, the text's own tokens (what a verification of a
    /// sampled answer accepts) when `evaluation`; [`NO_TARGET`] for a prompt token or a position past the sequence,
    /// [`OUTSIDE_HEAD`] for an answer token outside the head (a draft it rejects).
    pub checked: Vec<i32>,
}

/// The row of `token` in the draft head, or -1 outside it.
fn head_row(row_of_token: &[i32], token: i32) -> i32 {
    usize::try_from(token)
        .ok()
        .and_then(|token| row_of_token.get(token).copied())
        .unwrap_or(-1)
}

/// Stages `window` of `sequence` for `depths` drafts. `row_of_token` maps a token to its row of the draft head, -1 for a
/// token the head does not hold.
pub fn stage(sequence: &Sequence<'_>, window: &Window, depths: u32, row_of_token: &[i32], evaluation: bool) -> Staged {
    let (count, length) = (window.count as usize, sequence.tokens.len());
    let size = depths as usize * count;
    let mut staged = Staged {
        ids: Vec::with_capacity(size),
        labels: Vec::with_capacity(size),
        checked: Vec::with_capacity(size),
    };
    for k in 1..=depths as usize {
        for r in 0..count {
            let position = window.a as usize + r;
            let (input, target) = (position + k, position + k + 1);
            staged.ids.push(if input < length { sequence.tokens[input] } else { 0 });
            let answer = target < length && target >= sequence.prompt;
            let label_row = if answer {
                head_row(row_of_token, sequence.labels[target])
            } else {
                -1
            };
            staged.labels.push(label_row);
            let row = match (answer, evaluation) {
                (false, _) => NO_TARGET,
                (true, true) => head_row(row_of_token, sequence.tokens[target]),
                (true, false) => label_row,
            };
            staged.checked.push(if answer && row < 0 { OUTSIDE_HEAD } else { row });
        }
    }
    staged
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens 10..20, labels 110..120; the head holds the tokens 12, 13 and 15 as rows 0, 1 and 2 (labels 112 and 113
    /// as rows 3 and 4).
    fn fixture() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let tokens: Vec<i32> = (10..20).collect();
        let labels: Vec<i32> = (110..120).collect();
        let mut rows = vec![-1; 200];
        for (token, row) in [(12, 0), (13, 1), (15, 2), (112, 3), (113, 4)] {
            rows[token] = row;
        }
        (tokens, labels, rows)
    }

    fn window(a: u32, count: u32) -> Window {
        Window {
            a,
            count,
            first_key: 0,
            labelled: true,
        }
    }

    #[test]
    fn a_draft_reads_the_token_at_its_depth_and_targets_the_one_after() {
        let (tokens, labels, rows) = fixture();
        let sequence = Sequence { tokens: &tokens, labels: &labels, prompt: 3 };
        let staged = stage(&sequence, &window(0, 2), 2, &rows, false);
        // Depth 1 inputs are tokens 1 and 2, depth 2 inputs tokens 2 and 3.
        assert_eq!(staged.ids, [11, 12, 12, 13]);
        // Targets: depth 1 positions 2 (a prompt token) and 3; depth 2 positions 3 and 4. Labels 113 is row 4, 114 and
        // 113's neighbours are outside the head.
        assert_eq!(staged.labels, [-1, 4, 4, -1]);
        assert_eq!(staged.checked, [NO_TARGET, 4, 4, OUTSIDE_HEAD]);
    }

    #[test]
    fn evaluation_checks_against_the_texts_own_tokens() {
        let (tokens, labels, rows) = fixture();
        let sequence = Sequence { tokens: &tokens, labels: &labels, prompt: 3 };
        let staged = stage(&sequence, &window(0, 2), 2, &rows, true);
        // What trains is unchanged; the token at position 3 is 13 (row 1), at position 4 is 14 (outside the head).
        assert_eq!(staged.labels, [-1, 4, 4, -1]);
        assert_eq!(staged.checked, [NO_TARGET, 1, 1, OUTSIDE_HEAD]);
    }

    #[test]
    fn positions_past_the_sequence_read_token_zero_and_have_no_target() {
        let (tokens, labels, rows) = fixture();
        let sequence = Sequence { tokens: &tokens, labels: &labels, prompt: 3 };
        // The last window of 9 lineages: positions 8 only (a = 8, count 1), depths 3 reach 11, 12, 13.
        let staged = stage(&sequence, &window(8, 1), 3, &rows, false);
        assert_eq!(staged.ids, [19, 0, 0]);
        assert_eq!(staged.labels, [-1, -1, -1]);
        assert_eq!(staged.checked, [NO_TARGET, NO_TARGET, NO_TARGET]);
    }

    #[test]
    fn a_label_or_token_outside_the_table_is_outside_the_head() {
        let (tokens, labels, _) = fixture();
        let sequence = Sequence { tokens: &tokens, labels: &labels, prompt: 0 };
        let staged = stage(&sequence, &window(0, 1), 1, &[], false);
        assert_eq!((staged.labels, staged.checked), (vec![-1], vec![OUTSIDE_HEAD]));
        let negative = Sequence { tokens: &tokens, labels: &[-5; 10], prompt: 0 };
        let rows = vec![0; 200];
        let staged = stage(&negative, &window(0, 1), 1, &rows, false);
        assert_eq!(staged.checked, [OUTSIDE_HEAD]);
    }

    #[test]
    fn every_depth_has_every_lineage_in_depth_major_order() {
        let (tokens, labels, rows) = fixture();
        let sequence = Sequence { tokens: &tokens, labels: &labels, prompt: 0 };
        let staged = stage(&sequence, &window(1, 3), 4, &rows, false);
        assert_eq!((staged.ids.len(), staged.labels.len(), staged.checked.len()), (12, 12, 12));
        // Depth 2, lineage 1: position 2, input position 4.
        assert_eq!(staged.ids[3 + 1], 14);
    }
}
