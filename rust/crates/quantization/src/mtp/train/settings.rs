//! What a run is asked for, and the host arithmetic that follows from it: the windows of a sequence, the shuffle of
//! an epoch and the learning-rate schedule. Nothing here touches a device or a file.

use std::f64::consts::PI;

use thiserror::Error;

use crate::java_random::Random;

/// Rows of draft-head logits computed at a time.
pub const HEAD_ROWS: u32 = 128;
/// The longest sequence of the corpus the key buffers hold (`generate` prompts are at most 4096 tokens, answers its
/// `--max-tokens`).
pub const MAX_POSITIONS: u32 = 16384;
/// The most drafts one lineage unrolls: the statistics of a run keep this many depths.
pub const MOST_DEPTHS: u32 = 16;
/// The seed of epoch 0's shuffle; epoch `n` takes `SHUFFLE_SEED + n`.
const SHUFFLE_SEED: i64 = 20_261_019;
/// The rate the schedule decays to, as a share of the set rate.
const RATE_FLOOR: f64 = 0.1;

/// Why the settings are refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SettingsError {
    #[error("{what} {value} is not valid: {rule}")]
    Invalid {
        what: &'static str,
        value: String,
        rule: &'static str,
    },
}

fn invalid(what: &'static str, value: impl ToString, rule: &'static str) -> SettingsError {
    SettingsError::Invalid {
        what,
        value: value.to_string(),
        rule,
    }
}

/// The settings of a run, without its paths.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// The drafts unrolled.
    pub depths: u32,
    /// The lineages of a window.
    pub rows: u32,
    /// The keys a window attends, at least `rows`.
    pub keys: u32,
    /// The trained lineages between two updates.
    pub tokens: u32,
    pub epochs: u32,
    pub projection_rate: f32,
    pub norm_rate: f32,
    /// The updates of the linear warm-up.
    pub warmup: u32,
    /// Every `holdout`-th sequence is held out.
    pub holdout: u32,
    /// An evaluation follows every this many updates; 0 evaluates only before and after training.
    pub evaluate_every: u32,
    /// The most sequences of the corpus to read; 0 reads them all.
    pub limit: usize,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            depths: 4,
            rows: 1024,
            keys: 4096,
            tokens: 16384,
            epochs: 1,
            projection_rate: 2e-5,
            norm_rate: 2e-5,
            warmup: 10,
            holdout: 50,
            evaluate_every: 50,
            limit: 0,
        }
    }
}

impl Settings {
    /// Refuses a setting the run cannot work with, such as a held-out set of none, depths past its statistics or no
    /// epochs.
    pub fn check(&self) -> Result<(), SettingsError> {
        if !(1..=MOST_DEPTHS).contains(&self.depths) {
            return Err(invalid("--depths", self.depths, "1 to 16 drafts"));
        }
        if self.rows < 1 {
            return Err(invalid("--rows", self.rows, "at least 1"));
        }
        if self.keys < self.rows || self.keys > MAX_POSITIONS {
            return Err(invalid("--keys", self.keys, "at least --rows and at most 16384"));
        }
        if self.tokens < 1 {
            return Err(invalid("--tokens", self.tokens, "at least 1"));
        }
        if self.epochs < 1 {
            return Err(invalid("--epochs", self.epochs, "at least 1"));
        }
        if self.holdout < 1 {
            return Err(invalid(
                "--holdout",
                self.holdout,
                "at least 1: an evaluation needs held-out sequences",
            ));
        }
        for (what, rate) in [
            ("--projection-rate", self.projection_rate),
            ("--norm-rate", self.norm_rate),
        ] {
            if !rate.is_finite() || rate <= 0.0 {
                return Err(invalid(what, rate, "a positive number"));
            }
        }
        Ok(())
    }

    /// Whether sequence `index` of the corpus is held out.
    pub fn held_out(&self, index: usize) -> bool {
        index.is_multiple_of(self.holdout as usize)
    }
}

/// One window of a sequence: lineages `[a, a + count)`, keys from `first_key`, and whether any of its drafts has a
/// label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub a: u32,
    pub count: u32,
    pub first_key: u32,
    pub labelled: bool,
}

/// The windows of a sequence of `length` tokens whose first `prompt` are its prompt, in order. A sequence of fewer than
/// two tokens has none.
pub fn windows(length: u32, prompt: u32, settings: &Settings) -> Vec<Window> {
    let lineages = length.saturating_sub(1);
    let rows = settings.rows;
    let mut out = Vec::new();
    let mut a = 0;
    while a < lineages {
        let count = rows.min(lineages - a);
        let end = a + count;
        // The last lineage's last draft targets token `end + depths`; it is labelled when that is an answer token.
        out.push(Window {
            a,
            count,
            first_key: end.saturating_sub(settings.keys),
            labelled: end + settings.depths >= prompt,
        });
        a = end;
    }
    out
}

/// The training sequences in the order of `epoch`: a Fisher-Yates shuffle over a `java.util.Random`-compatible
/// generator, so that an epoch's order follows from its number.
pub fn shuffled(training: &[usize], epoch: u32) -> Vec<usize> {
    let mut order = training.to_vec();
    let mut random = Random::new(SHUFFLE_SEED + i64::from(epoch));
    for i in (1..order.len()).rev() {
        let bound = i32::try_from(i + 1).unwrap_or(i32::MAX);
        let j = random.next_int_bounded(bound).unwrap_or(0) as usize;
        order.swap(i, j);
    }
    order
}

/// The updates one epoch is expected to make: its answer-bearing lineages over `tokens`, at least 1. `lengths` are the
/// `(length, prompt)` of the training sequences.
pub fn expected_steps(lengths: impl Iterator<Item = (u64, u64)>, tokens: u32) -> u64 {
    let lineages: u64 = lengths
        .map(|(length, prompt)| length.saturating_sub(1).saturating_sub(prompt))
        .sum();
    (lineages / u64::from(tokens)).max(1)
}

/// The projection and norm rates of update `step` (from 1): a linear warm-up, then a cosine decay to a tenth.
/// `expected` is [`expected_steps`].
pub fn rates(settings: &Settings, step: u32, expected: u64) -> (f32, f32) {
    let total = (f64::from(settings.epochs) * expected as f64).max(1.0);
    let scale = if step <= settings.warmup {
        f64::from(step) / f64::from(settings.warmup)
    } else {
        let progress = (f64::from(step - settings.warmup) / total).min(1.0);
        RATE_FLOOR + (1.0 - RATE_FLOOR) * 0.5 * (1.0 + (PI * progress).cos())
    };
    (
        (f64::from(settings.projection_rate) * scale) as f32,
        (f64::from(settings.norm_rate) * scale) as f32,
    )
}

#[cfg(test)]
mod tests {
    use std::iter::empty;

    use super::*;

    fn small(rows: u32, keys: u32, depths: u32) -> Settings {
        Settings {
            rows,
            keys,
            depths,
            ..Settings::default()
        }
    }

    #[test]
    fn defaults_pass_their_own_check() {
        assert_eq!(Settings::default().check(), Ok(()));
    }

    #[test]
    fn refuses_each_setting_java_met_as_a_crash_or_a_loop() {
        let base = Settings::default();
        let refused = [
            Settings { depths: 0, ..base },
            Settings { depths: 17, ..base },
            Settings { rows: 0, ..base },
            Settings { keys: 1023, ..base },
            Settings { keys: 16385, ..base },
            Settings { tokens: 0, ..base },
            Settings { epochs: 0, ..base },
            Settings { holdout: 0, ..base },
            Settings {
                projection_rate: 0.0,
                ..base
            },
            Settings {
                projection_rate: f32::NAN,
                ..base
            },
            Settings {
                norm_rate: -1.0,
                ..base
            },
            Settings {
                norm_rate: f32::INFINITY,
                ..base
            },
        ];
        for settings in refused {
            assert!(settings.check().is_err(), "{settings:?} was accepted");
        }
        for settings in [
            Settings { depths: 16, ..base },
            Settings { keys: 16384, ..base },
            Settings { warmup: 0, ..base },
        ] {
            assert_eq!(settings.check(), Ok(()), "{settings:?} was refused");
        }
    }

    #[test]
    fn holds_out_every_holdout_th_sequence_from_the_first() {
        let settings = Settings {
            holdout: 3,
            ..Settings::default()
        };
        let held: Vec<usize> = (0..7).filter(|&s| settings.held_out(s)).collect();
        assert_eq!(held, [0, 3, 6]);
    }

    #[test]
    fn a_sequence_of_fewer_than_two_tokens_has_no_window() {
        let settings = small(4, 8, 2);
        assert!(windows(0, 0, &settings).is_empty());
        assert!(windows(1, 0, &settings).is_empty());
    }

    #[test]
    fn windows_tile_the_lineages_and_the_last_is_short() {
        let windows = windows(11, 4, &small(4, 8, 2));
        let tiles: Vec<(u32, u32)> = windows.iter().map(|w| (w.a, w.count)).collect();
        assert_eq!(tiles, [(0, 4), (4, 4), (8, 2)]);
    }

    #[test]
    fn keys_start_late_only_once_the_window_ends_past_the_key_count() {
        let windows = windows(21, 4, &small(4, 8, 2));
        let firsts: Vec<u32> = windows.iter().map(|w| w.first_key).collect();
        assert_eq!(firsts, [0, 0, 4, 8, 12]);
    }

    #[test]
    fn a_window_is_labelled_when_its_last_target_is_an_answer_token() {
        // 3 + 2 >= 5 labels the window ending at 3; the one ending at 2 has its deepest target at token 4 < 5.
        let windows = windows(12, 5, &small(1, 4, 2));
        let first_labelled = windows.iter().position(|w| w.labelled);
        assert_eq!(first_labelled, Some(2));
        assert!(windows[2..].iter().all(|w| w.labelled));
    }

    #[test]
    fn shuffle_follows_the_java_generator() {
        let items: Vec<usize> = (0..10).map(|i| i * 3 + 1).collect();
        assert_eq!(shuffled(&items, 0), [4, 13, 7, 19, 25, 1, 22, 28, 10, 16]);
        assert_eq!(shuffled(&items, 1), [4, 25, 22, 1, 16, 28, 10, 13, 19, 7]);
    }

    #[test]
    fn shuffle_of_none_and_one_is_itself() {
        assert!(shuffled(&[], 3).is_empty());
        assert_eq!(shuffled(&[1], 0), [1]);
        assert_eq!(shuffled(&[1, 4], 5), [1, 4]);
    }

    #[test]
    fn expected_steps_count_answer_lineages_and_never_fall_below_one() {
        // (10 - 1 - 4) + (20 - 1 - 4) = 20 lineages; 20 / 8 = 2.
        assert_eq!(expected_steps([(10, 4), (20, 4)].into_iter(), 8), 2);
        // A sequence that is all prompt counts none.
        assert_eq!(expected_steps([(4, 4), (0, 0)].into_iter(), 8), 1);
        assert_eq!(expected_steps(empty(), 8), 1);
    }

    #[test]
    fn rates_warm_up_linearly_then_decay_to_a_tenth() {
        let settings = Settings {
            warmup: 4,
            epochs: 1,
            projection_rate: 1e-3,
            norm_rate: 2e-3,
            ..Settings::default()
        };
        let at = |step| rates(&settings, step, 10);
        assert_eq!(at(2), (5e-4, 1e-3));
        assert_eq!(at(4), (1e-3, 2e-3));
        // Half way through the decay the cosine is 0 and the rate is 0.55 of the set one.
        let half = at(9).0 as f64;
        assert!((half - 0.55e-3).abs() < 1e-9, "{half}");
        // At and past the end the rate stays at a tenth.
        assert_eq!(at(14), at(1000));
        assert!(((at(14).0 as f64) - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn rates_without_warm_up_start_at_the_full_rate() {
        let settings = Settings {
            warmup: 0,
            projection_rate: 1e-3,
            norm_rate: 1e-3,
            ..Settings::default()
        };
        assert_eq!(
            rates(&settings, 1, 10).0,
            1e-3 * (0.1 + 0.9 * 0.5 * (1.0 + (PI * 0.1).cos())) as f32
        );
    }
}
