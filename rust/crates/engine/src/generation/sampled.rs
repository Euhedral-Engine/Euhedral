//! The host's part of a sampled request: the frame that selects a token from a logits row, and the bookkeeping that
//! tells the owner which selection it is waiting for.
//!
//! A sampled quantum copies its logits row to pinned memory inside its graph. When the quantum has retired the owner
//! publishes one [`Kind::Sample`](crate::frames::Kind::Sample) frame, which owns the row's lease, the request's sampler
//! and the step and position it selects for. The frame runs [`TokenSampler::select_bf16`], lets go of the row, and posts
//! the answer to the owner. The owner admits nothing until the answer arrives: the token is the next decode's input.
//! A frame that is dropped without running, or whose selection panics, still posts, with an error, so the request ends
//! and never waits for a selection that will not come.

use std::panic::{self, AssertUnwindSafe};

use euhedral_core::frames::Frame;

use crate::frames::{Kind, step};
use crate::sampling::TokenSampler;
use crate::stages::sampling::RowReadbackLease;

/// The logits row a [`Sample`] frame selects from.
pub(crate) trait Row: Send {
    /// The row's BF16 bytes, exactly the vocabulary wide; `None` when there is no such row.
    fn bytes(&self) -> Option<&[u8]>;
}

impl Row for RowReadbackLease {
    fn bytes(&self) -> Option<&[u8]> {
        self.readback().row(0)
    }
}

/// What the frame tells the owner.
pub(crate) struct Sampled {
    /// The selection the answer is for.
    pub(crate) step: u64,
    /// The token and the sampler, which the request keeps for its next selection, or why there is no token.
    pub(crate) result: Result<(u32, Box<TokenSampler>), String>,
}

/// The one selection of a sampled request that is waiting for its frame.
#[derive(Debug, Default)]
pub(crate) struct Pending {
    issued: u64,
    waiting: Option<u64>,
}

impl Pending {
    /// Starts a selection and returns its step. One at a time: a request keeps one quantum in flight.
    pub(crate) fn begin(&mut self) -> u64 {
        let step = self.issued;
        self.issued += 1;
        self.waiting = Some(step);
        step
    }

    /// Whether `step` is the selection awaited: it then no longer is. Any other step is stale.
    pub(crate) fn take(&mut self, step: u64) -> bool {
        if self.waiting == Some(step) {
            self.waiting = None;
            true
        } else {
            false
        }
    }

    /// Whether a selection is awaited.
    pub(crate) fn waiting(&self) -> bool {
        self.waiting.is_some()
    }
}

/// What the frame owns, and posts the answer however it ends.
struct Sample<R: Row> {
    step: u64,
    position: u64,
    row: Option<R>,
    sampler: Option<Box<TokenSampler>>,
    post: Option<Box<dyn FnOnce(Sampled) + Send>>,
}

impl<R: Row> Sample<R> {
    fn run(mut self) {
        let (Some(row), Some(mut sampler)) = (self.row.take(), self.sampler.take()) else {
            return;
        };
        let position = self.position;
        let selected = panic::catch_unwind(AssertUnwindSafe(|| {
            let bytes = row
                .bytes()
                .ok_or_else(|| "the logits row is not in the readback".to_string())?;
            sampler
                .select_bf16(bytes, position, None)
                .map_err(|error| error.to_string())
        }))
        .unwrap_or_else(|_| Err("the sampler panicked".to_string()));
        // The row goes back to its pool before the owner hears: its next quantum may take it at once.
        drop(row);
        self.answer(selected.map(|token| (token, sampler)));
    }

    fn answer(&mut self, result: Result<(u32, Box<TokenSampler>), String>) {
        if let Some(post) = self.post.take() {
            post(Sampled {
                step: self.step,
                result,
            });
        }
    }
}

impl<R: Row> Drop for Sample<R> {
    fn drop(&mut self) {
        if self.post.is_some() {
            // Dropped before it ran (the lattice ended, or the frame was discarded): the request must still end.
            self.row = None;
            self.sampler = None;
            self.answer(Err("the selection was dropped before it ran".to_string()));
        }
    }
}

/// The frame that selects the token at `position` from `row` with `sampler` and calls `post` with the answer.
pub(crate) fn sample_frame<R: Row + 'static>(
    step_number: u64,
    position: u64,
    sampler: Box<TokenSampler>,
    row: R,
    post: impl FnOnce(Sampled) + Send + 'static,
) -> Box<dyn Frame> {
    let sample = Sample {
        step: step_number,
        position,
        row: Some(row),
        sampler: Some(sampler),
        post: Some(Box::new(post)),
    };
    step(Kind::Sample, 0, move || sample.run())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, OnceLock};

    use super::*;
    use crate::sampling::SamplingConfig;

    /// What the frames under test posted: the first answer and how many came.
    #[derive(Clone, Default)]
    struct Answers {
        first: Arc<OnceLock<Sampled>>,
        count: Arc<AtomicUsize>,
    }

    impl Answers {
        fn poster(&self) -> impl FnOnce(Sampled) + Send + 'static {
            let (first, count) = (Arc::clone(&self.first), Arc::clone(&self.count));
            move |sampled| {
                count.fetch_add(1, Ordering::SeqCst);
                let _ = first.set(sampled);
            }
        }

        /// The answer posted, which must be the only one.
        fn only(&self) -> &Sampled {
            assert_eq!(self.count.load(Ordering::SeqCst), 1, "one answer");
            self.first.get().unwrap()
        }
    }

    struct Bytes(Vec<u8>);

    impl Row for Bytes {
        fn bytes(&self) -> Option<&[u8]> {
            Some(&self.0)
        }
    }

    struct Panics;

    impl Row for Panics {
        fn bytes(&self) -> Option<&[u8]> {
            panic!("a row that cannot be read")
        }
    }

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn sampler(vocabulary: usize, top_k: u32) -> Box<TokenSampler> {
        let config = SamplingConfig::new(1.0, top_k, 1.0, 7, false).unwrap();
        Box::new(TokenSampler::new(config, vocabulary).unwrap())
    }

    fn run(frame: &mut Box<dyn Frame>) {
        frame.execute().unwrap();
    }

    #[test]
    fn a_frame_selects_the_row_and_hands_the_sampler_back() {
        let answers = Answers::default();
        let row = Bytes(bf16(&[0.5, 3.0, -1.0, 2.0]));
        let mut frame = sample_frame(4, 9, sampler(4, 1), row, answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert_eq!(sampled.step, 4);
        let (token, sampler) = sampled.result.as_ref().unwrap();
        assert_eq!(*token, 1, "the highest logit under top-k 1");
        assert_eq!(sampler.vocabulary(), 4);
    }

    #[test]
    fn a_row_of_the_wrong_width_is_an_error() {
        let answers = Answers::default();
        let mut frame = sample_frame(0, 0, sampler(4, 1), Bytes(bf16(&[1.0, 2.0])), answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert!(sampled.result.is_err());
    }

    #[test]
    fn a_frame_dropped_before_it_ran_still_answers_once_with_an_error() {
        let answers = Answers::default();
        let frame = sample_frame(2, 0, sampler(4, 1), Bytes(bf16(&[1.0; 4])), answers.poster());
        drop(frame);
        let sampled = answers.only();
        assert_eq!(sampled.step, 2);
        assert!(sampled.result.is_err());
    }

    #[test]
    fn a_selection_that_panics_answers_with_an_error() {
        let answers = Answers::default();
        let mut frame = sample_frame(1, 0, sampler(4, 1), Panics, answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert_eq!(
            sampled.result.as_ref().err().map(String::as_str),
            Some("the sampler panicked")
        );
    }

    #[test]
    fn only_the_awaited_step_is_taken_and_a_stale_one_is_ignored() {
        let mut pending = Pending::default();
        assert!(!pending.waiting());
        let first = pending.begin();
        assert!(pending.waiting());
        assert!(!pending.take(first + 1), "a step not issued");
        assert!(pending.waiting(), "a stale answer leaves the wait");
        assert!(pending.take(first));
        assert!(!pending.take(first), "an answer comes once");
        assert!(!pending.waiting());
        let second = pending.begin();
        assert_ne!(first, second);
        assert!(!pending.take(first), "the answer of an earlier selection is stale");
        assert!(pending.take(second));
    }
}
