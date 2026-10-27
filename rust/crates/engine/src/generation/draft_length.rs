//! How many tokens an MTP step drafts before its verification (`DraftLength`, `docs/speculation/MTP_DRAFT_CONFIDENCE.md`):
//! at least `least`, at most `most`, and past `least` in blocks of `step` drafts, each block only while the running sum
//! of the drafts' log-probabilities under the draft head is at or above `threshold`. The verification checks every
//! draft, so the output is greedy decode's whatever the length; the length only decides how many tokens one
//! verification can commit.

use thiserror::Error;

/// Most tokens a step drafts (`MtpDecoder.MAX_DRAFTS`).
pub const MAX_DRAFTS: usize = 16;

/// Most drafts one verification checks (`MtpDecoder.MAX_VERIFIED`).
pub const MAX_VERIFIED: usize = 7;

/// A draft length that breaks the rules of [`DraftLength::new`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DraftLengthError {
    #[error("draft length must satisfy 1 <= least <= most <= {MAX_DRAFTS}, not {least}..{most}")]
    Bounds { least: usize, most: usize },
    #[error("step must be positive")]
    Step,
    #[error("threshold must be a number")]
    Threshold,
}

/// How many tokens an MTP step drafts.
///
/// The engine runs the fixed length of the artifact's depth ([`DraftLength::fixed`]), which its chained step admits
/// whole. A gated length (`least < most`) needs the host between the drafts and the draft head's log-probabilities;
/// it is ported here and not wired into the engine yet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftLength {
    least: usize,
    most: usize,
    step: usize,
    threshold: f32,
}

impl DraftLength {
    /// A length of `least` to `most` drafts, past `least` in blocks of `step`, while the summed log-probability is at
    /// least `threshold`.
    pub fn new(least: usize, most: usize, step: usize, threshold: f32) -> Result<DraftLength, DraftLengthError> {
        if least < 1 || most < least || most > MAX_DRAFTS {
            return Err(DraftLengthError::Bounds { least, most });
        }
        if step < 1 {
            return Err(DraftLengthError::Step);
        }
        if threshold.is_nan() {
            return Err(DraftLengthError::Threshold);
        }
        Ok(DraftLength {
            least,
            most,
            step,
            threshold,
        })
    }

    /// Exactly `drafts` drafts per step. `drafts` must be 1 to [`MAX_DRAFTS`].
    pub fn fixed(drafts: usize) -> Result<DraftLength, DraftLengthError> {
        DraftLength::new(drafts, drafts, 1, f32::NEG_INFINITY)
    }

    pub fn least(&self) -> usize {
        self.least
    }

    pub fn most(&self) -> usize {
        self.most
    }

    /// Whether the length depends on the drafts' log-probabilities.
    pub fn gated(&self) -> bool {
        self.least < self.most
    }

    /// Whether a step that has drafted `drafted` tokens whose log-probabilities sum to `log_probability` drafts
    /// another. The sum is checked at `least` and at every block boundary after it; a sum that is not a number (an
    /// unscored draft) stops the step there.
    pub fn continues(&self, drafted: usize, log_probability: f64) -> bool {
        if drafted >= self.most {
            return false;
        }
        if drafted < self.least || !(drafted - self.least).is_multiple_of(self.step) {
            return true;
        }
        log_probability >= f64::from(self.threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drafts a step makes when its drafts carry these log-probabilities (`DraftLengthTest.drafted`).
    fn drafted(length: &DraftLength, log_probabilities: &[f64]) -> usize {
        let mut drafted = 1;
        let mut sum = log_probabilities[0];
        while length.continues(drafted, sum) {
            sum += log_probabilities[drafted];
            drafted += 1;
        }
        drafted
    }

    #[test]
    fn draft_length_matches_java_table() {
        // A fixed length ignores confidence.
        let four = DraftLength::fixed(4).unwrap();
        assert!(!four.gated());
        assert_eq!(drafted(&four, &[-9.0; 5]), 4);
        assert_eq!(drafted(&four, &[0.0; 5]), 4);

        // A gated length drafts the least, then while the sum holds.
        let gated = DraftLength::new(3, 7, 1, -0.5).unwrap();
        assert!(gated.gated());
        assert_eq!(
            drafted(&gated, &[-2.0; 7]),
            3,
            "unconfident drafts still reach the least"
        );
        assert_eq!(drafted(&gated, &[-0.01; 8]), 7, "confident drafts run to the most");
        assert_eq!(
            drafted(&gated, &[-0.1, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1]),
            5,
            "the sum crosses the threshold with the fifth draft, which is still made; the sixth is not"
        );
        assert_eq!(
            drafted(&gated, &[-0.25, -0.25, 0.0, -0.1, 0.0, 0.0, 0.0]),
            4,
            "a sum exactly at the threshold continues"
        );

        // A block step checks only at block boundaries.
        let blocks = DraftLength::new(3, 6, 3, -0.5).unwrap();
        assert_eq!(
            drafted(&blocks, &[-0.1, -0.1, -0.1, -5.0, -5.0, -5.0, -5.0]),
            6,
            "a confident first block drafts the whole second block however unconfident it is"
        );
        assert_eq!(
            drafted(&blocks, &[-0.1, -0.1, -0.4, 0.0, 0.0, 0.0, 0.0]),
            3,
            "an unconfident first block stops at the least"
        );
        assert_eq!(
            drafted(&DraftLength::new(3, 7, 3, -0.5).unwrap(), &[0.0; 8]),
            7,
            "the cap cuts a block short"
        );

        // An unscored draft stops at the least.
        assert_eq!(drafted(&DraftLength::new(3, 7, 1, -0.5).unwrap(), &[f64::NAN; 4]), 3);
    }

    #[test]
    fn invalid_lengths_are_refused() {
        assert!(DraftLength::new(0, 3, 1, -1.0).is_err());
        assert!(DraftLength::new(4, 3, 1, -1.0).is_err());
        assert!(DraftLength::new(1, MAX_DRAFTS + 1, 1, -1.0).is_err());
        assert_eq!(DraftLength::new(1, 3, 1, f32::NAN), Err(DraftLengthError::Threshold));
        assert_eq!(DraftLength::new(1, 3, 0, -1.0), Err(DraftLengthError::Step));
        assert!(DraftLength::fixed(0).is_err());
        assert_eq!(DraftLength::fixed(MAX_DRAFTS).unwrap().most(), MAX_DRAFTS);
    }
}
