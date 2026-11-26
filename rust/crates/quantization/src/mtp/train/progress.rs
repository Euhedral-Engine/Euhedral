//! Where a run is, and what each outcome of the device does to it: the sequence order, the trained lineages waiting for
//! an update, the held-out evaluation's tallies, the best evaluation so far, and the header of the saved state. Nothing
//! here touches a device or a file: the run's frames feed it what the device returned and do what it decides.

use std::fmt::Write;
use std::mem::take;
use std::str::FromStr;

use thiserror::Error;

use super::settings::{MOST_DEPTHS, Settings, rates, shuffled};
use crate::resume::Header;

const DEPTH_SLOTS: usize = MOST_DEPTHS as usize;
/// The rows a draft is checked against that mean "no target": a prompt token or a position past the sequence.
pub const NO_TARGET: i32 = -1;
/// The checked row of an answer token that is outside the draft head: a draft of it is rejected.
pub const OUTSIDE_HEAD: i32 = -2;
/// How often a training report is made, in updates.
const REPORT_EVERY: u32 = 10;

/// The headers' keys. The identity has one key per part, so that a refusal can name which part differs.
const CORPUS: &str = "identity.corpus";
const MODEL: &str = "identity.model";
const ARTIFACT: &str = "identity.artifact";
const SETTINGS: &str = "identity.settings";
const IDENTITY_KEYS: [&str; 4] = [CORPUS, MODEL, ARTIFACT, SETTINGS];
const EPOCH: &str = "epoch";
const CURSOR: &str = "cursor";
const STEP: &str = "step";
const BEST: &str = "best";
const PENDING: &str = "pending";
/// The header key of the best parameters' accepted drafts.
const ACCEPTED: &str = "accepted";

/// Why a saved state is refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum StateError {
    /// The state is the run's own but a field is missing or does not parse.
    #[error("the saved state has no valid {0}")]
    Field(&'static str),
    /// The file is there but holds no run's header: it was cut short, or it is not a state.
    #[error("the saved state has no header of a run: it is torn or not a state; --fresh starts over")]
    Torn,
    /// The state is another run's: the parts of the identity that differ are named.
    #[error("the saved state is of another run: its {} differ; --fresh starts over", .differs.join(", "))]
    Foreign { differs: Vec<&'static str> },
    /// The best parameters of a run are there but its state is not.
    #[error("best.bin is there but no saved state to continue from; --fresh starts over")]
    Leftover,
    /// The state is further on than the run can be.
    #[error("the saved state is at epoch {epoch} cursor {cursor}, past the {epochs} epochs of {sequences} sequences")]
    Past {
        epoch: u32,
        cursor: usize,
        epochs: u32,
        sequences: usize,
    },
}

/// What a run is: the content of its corpus, of the model's starting parameters and of the artifact, and the settings
/// that shape its state. A saved state continues only a run of the same identity; where the files lie does not matter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// A digest of the corpus's content.
    pub corpus: String,
    /// A digest of the drafter's starting parameters.
    pub model: String,
    /// A digest of the artifact's draft head and embedding.
    pub artifact: String,
    settings: String,
}

impl Identity {
    /// The identity of a run over `sequences` sequences of the corpus, with the digests of its inputs.
    pub fn new(corpus: String, model: String, artifact: String, settings: &Settings, sequences: usize) -> Identity {
        Identity {
            corpus,
            model,
            artifact,
            settings: format!(
                "sequences={sequences} depths={} rows={} keys={} tokens={} epochs={} rates={}/{} warmup={} holdout={}",
                settings.depths,
                settings.rows,
                settings.keys,
                settings.tokens,
                settings.epochs,
                settings.projection_rate,
                settings.norm_rate,
                settings.warmup,
                settings.holdout,
            ),
        }
    }

    fn parts(&self) -> [(&'static str, &str); 4] {
        [
            (CORPUS, &self.corpus),
            (MODEL, &self.model),
            (ARTIFACT, &self.artifact),
            (SETTINGS, &self.settings),
        ]
    }

    fn write(&self, header: &mut Header) {
        for (key, value) in self.parts() {
            header.set(key, value);
        }
    }

    /// The names of the parts that `header` does not hold the same as this identity; none when it is this run's.
    fn differing(&self, header: &Header) -> Vec<&'static str> {
        self.parts()
            .into_iter()
            .filter(|(key, value)| header.get(key) != Some(*value))
            .map(|(key, _)| key.strip_prefix("identity.").unwrap_or(key))
            .collect()
    }
}

/// What the run does next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Next {
    /// Train on this sequence of the corpus.
    Sequence(usize),
    /// The last epoch is done: the final evaluation.
    Final,
}

/// The device's answer for one training window.
#[derive(Debug)]
pub struct Trained<'a> {
    /// Whether the window had labels, and so was trained.
    pub labelled: bool,
    /// The lineages of the window.
    pub count: usize,
    /// What each draft is checked against, depth-major `[depths][count]`.
    pub checked: &'a [i32],
    /// The draft head's argmax of each draft, laid out as `checked`.
    pub predicted: &'a [i32],
    /// The window's summed loss of each depth.
    pub losses: &'a [f64],
}

/// An optimiser step the run owes the device.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Update {
    /// The step, from 1.
    pub step: u32,
    pub projection_rate: f32,
    pub norm_rate: f32,
}

/// The training statistics since the last report.
#[derive(Clone, Debug, Default)]
struct Tally {
    loss: [f64; DEPTH_SLOTS],
    correct: [u64; DEPTH_SLOTS],
    count: [u64; DEPTH_SLOTS],
}

/// The saved state's file as the start found it.
#[derive(Clone, Copy, Debug)]
pub enum Found<'a> {
    /// There is no file.
    Missing,
    /// The file's header: empty when the file is shorter than a header.
    Header(&'a Header),
}

/// How a run continues from a saved state.
#[derive(Debug)]
pub enum Resume {
    /// There is no state of this run: start from the checkpoint.
    Fresh,
    /// Continue from the state.
    Continue(Progress),
    /// The state is of a run that finished: there is nothing to train.
    Complete(Progress),
}

/// The run's progress.
#[derive(Clone, Debug)]
pub struct Progress {
    settings: Settings,
    training: Vec<usize>,
    expected: u64,
    order: Vec<usize>,
    epoch: u32,
    cursor: usize,
    step: u32,
    /// Trained lineages waiting for an update.
    pending: u32,
    best: f64,
    evaluation_due: bool,
    tally: Tally,
}

impl Progress {
    /// A run at its start over `training` (indices of the corpus's sequences), whose epochs are `expected` updates long.
    pub fn new(settings: Settings, training: Vec<usize>, expected: u64) -> Progress {
        let order = shuffled(&training, 0);
        Progress {
            settings,
            training,
            expected,
            order,
            epoch: 0,
            cursor: 0,
            step: 0,
            pending: 0,
            best: -1.0,
            evaluation_due: false,
            tally: Tally::default(),
        }
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn epoch(&self) -> u32 {
        self.epoch
    }

    pub fn best(&self) -> f64 {
        self.best
    }

    /// The training sequences of the epoch, with the cursor: how far through it the run is.
    pub fn position(&self) -> (usize, usize) {
        (self.cursor, self.order.len())
    }

    /// What follows: the next sequence of the epoch, the first of the next one (reshuffled), or, once the last epoch
    /// ends, the final evaluation.
    pub fn advance(&mut self) -> Next {
        if self.cursor >= self.order.len() {
            self.epoch += 1;
            self.cursor = 0;
            if self.epoch >= self.settings.epochs {
                return Next::Final;
            }
            self.order = shuffled(&self.training, self.epoch);
        }
        let sequence = self.order[self.cursor];
        self.cursor += 1;
        Next::Sequence(sequence)
    }

    /// Takes in a trained window. Returns the update the run owes once `tokens` lineages are waiting; the run applies it
    /// before its next window.
    pub fn trained(&mut self, window: &Trained<'_>) -> Option<Update> {
        if window.labelled {
            self.tally(window);
            let first = window.checked.get(..window.count).unwrap_or_default();
            self.pending += first.iter().filter(|&&row| row != NO_TARGET).count() as u32;
        }
        if self.pending < self.settings.tokens {
            return None;
        }
        self.step += 1;
        self.pending = 0;
        if self.settings.evaluate_every > 0 && self.step.is_multiple_of(self.settings.evaluate_every) {
            self.evaluation_due = true;
        }
        let (projection_rate, norm_rate) = rates(&self.settings, self.step, self.expected);
        Some(Update {
            step: self.step,
            projection_rate,
            norm_rate,
        })
    }

    fn tally(&mut self, window: &Trained<'_>) {
        let count = window.count;
        for k in 0..self.settings.depths as usize {
            self.tally.loss[k] += window.losses.get(k).copied().unwrap_or(0.0);
            let (checked, predicted) = (
                window.checked.get(k * count..(k + 1) * count).unwrap_or_default(),
                window.predicted.get(k * count..(k + 1) * count).unwrap_or_default(),
            );
            for (&row, &draft) in checked.iter().zip(predicted) {
                if row < 0 {
                    continue;
                }
                self.tally.count[k] += 1;
                self.tally.correct[k] += u64::from(draft == row);
            }
        }
    }

    /// The training report of an update, every tenth (and then the statistics start again).
    pub fn report(&mut self, update: &Update) -> Option<String> {
        if !update.step.is_multiple_of(REPORT_EVERY) {
            return None;
        }
        let mut line = format!(
            "step {} (epoch {}, {}/{}):",
            update.step,
            self.epoch,
            self.cursor,
            self.order.len()
        );
        for k in 0..self.settings.depths as usize {
            let count = self.tally.count[k].max(1) as f64;
            let _ = write!(
                line,
                " d{} loss {:.3} top-1 {:.3}",
                k + 1,
                self.tally.loss[k] / count,
                self.tally.correct[k] as f64 / count
            );
        }
        self.tally = Tally::default();
        Some(line)
    }

    /// Whether a sequence has just ended with an evaluation due; the due flag is cleared.
    pub fn take_evaluation_due(&mut self) -> bool {
        take(&mut self.evaluation_due)
    }

    /// Takes in a finished evaluation. `initial` is the one before training, `last` the one after the final epoch.
    pub fn evaluated(&mut self, evaluation: &Evaluation, initial: bool, last: bool) -> Verdict {
        let mean = evaluation.mean();
        let improved = mean > self.best;
        if improved {
            self.best = mean;
        }
        let name = if initial {
            "before training".to_owned()
        } else {
            format!("step {}", self.step)
        };
        Verdict {
            mean,
            improved,
            save_state: !initial,
            finished: last,
            line: format!("{name}: {}", evaluation.line(mean)),
        }
    }

    /// The header of the state: the identity of the run and where it is.
    pub fn header(&self, identity: &Identity) -> Header {
        let mut header = Header::new();
        identity.write(&mut header);
        header.set(EPOCH, self.epoch.to_string());
        header.set(CURSOR, self.cursor.to_string());
        header.set(STEP, self.step.to_string());
        header.set(BEST, self.best.to_string());
        header.set(PENDING, self.pending.to_string());
        header
    }

    /// The header of the best parameters: the step they were found at and their accepted drafts per lineage.
    pub fn best_header(&self, identity: &Identity) -> Header {
        let mut header = Header::new();
        identity.write(&mut header);
        header.set(STEP, self.step.to_string());
        header.set(ACCEPTED, self.best.to_string());
        header
    }

    /// What the saved state `found` means for a run of `settings` over `training`, whose identity is `identity`.
    ///
    /// Nothing that a run has saved is ever started over by accident: a state of another run, a torn one, or best
    /// parameters without a state refuse and say what differs. Only `fresh`, which the caller sets when the user asked
    /// to start over, takes the checkpoint again.
    pub fn resume(
        settings: Settings,
        training: Vec<usize>,
        expected: u64,
        identity: &Identity,
        found: Found<'_>,
        best_exists: bool,
        fresh: bool,
    ) -> Result<Resume, StateError> {
        if fresh {
            return Ok(Resume::Fresh);
        }
        let header = match found {
            Found::Missing if best_exists => return Err(StateError::Leftover),
            Found::Missing => return Ok(Resume::Fresh),
            Found::Header(header) => header,
        };
        let differs = identity.differing(header);
        if differs.len() == IDENTITY_KEYS.len() {
            return Err(StateError::Torn);
        }
        if !differs.is_empty() {
            return Err(StateError::Foreign { differs });
        }
        fn field<T: FromStr>(header: &Header, key: &'static str) -> Result<T, StateError> {
            header
                .get(key)
                .and_then(|text| text.parse().ok())
                .ok_or(StateError::Field(key))
        }
        let (epoch, cursor, step): (u32, usize, u32) =
            (field(header, EPOCH)?, field(header, CURSOR)?, field(header, STEP)?);
        let (best, pending): (f64, u32) = (field(header, BEST)?, field(header, PENDING)?);
        if !best.is_finite() {
            return Err(StateError::Field(BEST));
        }
        let mut progress = Progress::new(settings, training, expected);
        let sequences = progress.training.len();
        let finished = epoch == settings.epochs && cursor == 0;
        if epoch > settings.epochs || cursor > sequences || (epoch == settings.epochs && !finished) {
            return Err(StateError::Past {
                epoch,
                cursor,
                epochs: settings.epochs,
                sequences,
            });
        }
        progress.epoch = epoch;
        progress.cursor = cursor;
        progress.step = step;
        progress.best = best;
        progress.pending = pending;
        if !finished {
            progress.order = shuffled(&progress.training, epoch);
        }
        Ok(if finished {
            Resume::Complete(progress)
        } else {
            Resume::Continue(progress)
        })
    }
}

/// What an evaluation decided.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    /// The accepted drafts per lineage.
    pub mean: f64,
    /// Whether it is the best so far: the parameters are saved as the best.
    pub improved: bool,
    /// Whether the state is saved (every evaluation after training began).
    pub save_state: bool,
    /// Whether the run ends here.
    pub finished: bool,
    /// What to tell.
    pub line: String,
}

/// The tallies of an evaluation over the held-out sequences: of the lineages whose every draft has a target, how many
/// leading drafts the base model's own tokens confirm.
#[derive(Clone, Debug)]
pub struct Evaluation {
    depths: usize,
    lineages: u64,
    accepted: [u64; DEPTH_SLOTS + 1],
    reached: [u64; DEPTH_SLOTS],
    kind_lineages: [u64; 2],
    kind_accepted: [u64; 2],
}

impl Evaluation {
    pub fn new(depths: u32) -> Evaluation {
        Evaluation {
            depths: depths as usize,
            lineages: 0,
            accepted: [0; DEPTH_SLOTS + 1],
            reached: [0; DEPTH_SLOTS],
            kind_lineages: [0; 2],
            kind_accepted: [0; 2],
        }
    }

    /// Takes in a window of a sequence (`sampled` or greedy) whose drafts were checked against `checked` and drafted
    /// `predicted`, both depth-major `[depths][count]`.
    pub fn add(&mut self, sampled: bool, count: usize, checked: &[i32], predicted: &[i32]) {
        let at = |k: usize, r: usize| {
            (
                checked.get(k * count + r).copied(),
                predicted.get(k * count + r).copied(),
            )
        };
        for r in 0..count {
            let complete = (0..self.depths).all(|k| at(k, r).0.is_some_and(|row| row != NO_TARGET));
            if !complete {
                continue;
            }
            self.lineages += 1;
            let mut run = 0;
            while run < self.depths {
                let (row, draft) = at(run, r);
                match row {
                    Some(row) if row >= 0 && Some(row) == draft => run += 1,
                    _ => break,
                }
            }
            self.accepted[run] += 1;
            for reached in &mut self.reached[..run] {
                *reached += 1;
            }
            self.kind_lineages[usize::from(sampled)] += 1;
            self.kind_accepted[usize::from(sampled)] += run as u64;
        }
    }

    /// The lineages counted.
    pub fn lineages(&self) -> u64 {
        self.lineages
    }

    /// The accepted drafts per lineage.
    pub fn mean(&self) -> f64 {
        let sum: f64 = (0..=self.depths)
            .map(|run| run as f64 * self.accepted[run] as f64)
            .sum();
        sum / self.lineages.max(1) as f64
    }

    fn line(&self, mean: f64) -> String {
        let mut line = format!(
            "held-out accepted drafts {mean:.4} per lineage ({} lineages), conditional",
            self.lineages
        );
        for k in 0..self.depths {
            let before = if k == 0 { self.lineages } else { self.reached[k - 1] };
            let _ = write!(line, " d{} {:.3}", k + 1, self.reached[k] as f64 / before.max(1) as f64);
        }
        let rate = |kind: usize| self.kind_accepted[kind] as f64 / self.kind_lineages[kind].max(1) as f64;
        let _ = write!(
            line,
            "; greedy {:.4} ({}), sampled {:.4} ({})",
            rate(0),
            self.kind_lineages[0],
            rate(1),
            self.kind_lineages[1]
        );
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(corpus: &str) -> Identity {
        Identity::new(corpus.to_owned(), "m".to_owned(), "a".to_owned(), &settings(), 3)
    }

    /// The decision for a state `found`, of the run `identity("a")`.
    fn decide(found: Found<'_>, best_exists: bool, fresh: bool) -> Result<Resume, StateError> {
        Progress::resume(settings(), vec![1, 2, 4], 5, &identity("a"), found, best_exists, fresh)
    }

    fn settings() -> Settings {
        Settings {
            depths: 2,
            rows: 4,
            keys: 8,
            tokens: 3,
            epochs: 2,
            warmup: 2,
            evaluate_every: 2,
            ..Settings::default()
        }
    }

    fn progress() -> Progress {
        Progress::new(settings(), vec![1, 2, 4], 5)
    }

    /// A trained window of `count` lineages whose depth-1 targets are `first` and depth-2 are all answers.
    fn window(first: &[i32], predicted_first: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let count = first.len();
        let mut checked = first.to_vec();
        checked.extend(vec![5; count]);
        let mut predicted = predicted_first.to_vec();
        predicted.extend(vec![5; count]);
        (checked, predicted)
    }

    fn train(progress: &mut Progress, checked: &[i32], predicted: &[i32]) -> Option<Update> {
        let count = checked.len() / 2;
        progress.trained(&Trained {
            labelled: true,
            count,
            checked,
            predicted,
            losses: &[1.0, 2.0],
        })
    }

    #[test]
    fn an_epoch_visits_every_training_sequence_once_then_reshuffles_and_the_last_ends_in_the_final_evaluation() {
        let mut progress = progress();
        let mut seen = Vec::new();
        for _ in 0..3 {
            let Next::Sequence(sequence) = progress.advance() else {
                panic!("the epoch ended early")
            };
            seen.push(sequence);
        }
        assert_eq!(seen, shuffled(&[1, 2, 4], 0));
        let Next::Sequence(first) = progress.advance() else {
            panic!("the second epoch did not start")
        };
        assert_eq!((progress.epoch(), first), (1, shuffled(&[1, 2, 4], 1)[0]));
        progress.advance();
        progress.advance();
        assert_eq!(progress.advance(), Next::Final);
        assert_eq!(progress.epoch(), 2);
    }

    #[test]
    fn lineages_wait_for_tokens_then_an_update_at_the_scheduled_rate_is_owed() {
        let mut progress = progress();
        let (checked, predicted) = window(&[5, 5], &[5, 5]);
        assert_eq!(train(&mut progress, &checked, &predicted), None);
        let update = train(&mut progress, &checked, &predicted).unwrap();
        assert_eq!(update.step, 1);
        // Warm-up of two updates: the first is half the set rate.
        assert_eq!(update.projection_rate, 1e-5);
        assert_eq!(progress.pending, 0);
    }

    #[test]
    fn a_lineage_with_no_depth_one_target_is_not_waiting_for_an_update() {
        let mut progress = progress();
        let (checked, predicted) = window(&[NO_TARGET, OUTSIDE_HEAD, 5], &[0, 0, 5]);
        assert_eq!(train(&mut progress, &checked, &predicted), None);
        assert_eq!(progress.pending, 2);
    }

    #[test]
    fn an_unlabelled_window_trains_nothing() {
        let mut progress = progress();
        let update = progress.trained(&Trained {
            labelled: false,
            count: 4,
            checked: &[],
            predicted: &[],
            losses: &[],
        });
        assert_eq!((update, progress.pending), (None, 0));
    }

    #[test]
    fn an_evaluation_falls_due_every_evaluate_every_updates_and_is_taken_once() {
        let mut progress = progress();
        let (checked, predicted) = window(&[5, 5, 5], &[5, 5, 5]);
        assert!(train(&mut progress, &checked, &predicted).is_some());
        assert!(!progress.take_evaluation_due());
        assert!(train(&mut progress, &checked, &predicted).is_some());
        assert!(progress.take_evaluation_due());
        assert!(!progress.take_evaluation_due());
    }

    #[test]
    fn no_evaluate_every_means_no_evaluation_between_updates() {
        let mut progress = Progress::new(
            Settings {
                evaluate_every: 0,
                ..settings()
            },
            vec![1],
            5,
        );
        let (checked, predicted) = window(&[5, 5, 5], &[5, 5, 5]);
        assert!(train(&mut progress, &checked, &predicted).is_some());
        assert!(!progress.take_evaluation_due());
    }

    #[test]
    fn every_tenth_update_reports_the_mean_loss_and_top_1_of_the_labelled_drafts_and_starts_over() {
        let mut progress = Progress::new(
            Settings {
                tokens: 1,
                ..settings()
            },
            vec![1],
            5,
        );
        // Ten windows: summed losses 1 and 2 over 20 and 30 labelled drafts; depth 1 is right for half of them. A draft without a target is not counted.
        let (checked, predicted) = window(&[5, 6, NO_TARGET], &[5, 0, 0]);
        let mut reports = Vec::new();
        for _ in 0..10 {
            let update = train(&mut progress, &checked, &predicted).unwrap();
            reports.extend(progress.report(&update));
        }
        assert_eq!(reports.len(), 1);
        assert_eq!(
            reports[0],
            "step 10 (epoch 0, 0/1): d1 loss 0.500 top-1 0.500 d2 loss 0.667 top-1 1.000"
        );
        let update = train(&mut progress, &checked, &predicted).unwrap();
        assert_eq!(progress.report(&update), None);
        assert_eq!(progress.tally.count[0], 2);
    }

    #[test]
    fn the_first_evaluation_is_not_saved_as_state_and_a_better_one_replaces_the_best() {
        let mut progress = progress();
        let mut evaluation = Evaluation::new(2);
        evaluation.add(false, 1, &[5, 5], &[5, 5]);
        let initial = progress.evaluated(&evaluation, true, false);
        assert_eq!(
            (initial.mean, initial.improved, initial.save_state, initial.finished),
            (2.0, true, false, false)
        );
        assert!(
            initial
                .line
                .starts_with("before training: held-out accepted drafts 2.0000 per lineage (1 lineages)")
        );
        let worse = Evaluation::new(2);
        let later = progress.evaluated(&worse, false, true);
        assert_eq!(
            (later.mean, later.improved, later.save_state, later.finished),
            (0.0, false, true, true)
        );
        assert_eq!(progress.best(), 2.0);
    }

    #[test]
    fn an_evaluation_counts_leading_accepted_drafts_of_lineages_whose_every_draft_has_a_target() {
        let mut evaluation = Evaluation::new(2);
        // Lineage 0: both accepted. 1: the first only. 2: a rejected token outside the head. 3: no target at depth 2.
        evaluation.add(
            false,
            4,
            &[5, 6, OUTSIDE_HEAD, 7, 5, 6, 6, NO_TARGET],
            &[5, 6, 5, 7, 5, 0, 6, 0],
        );
        evaluation.add(true, 1, &[3, 3], &[3, 4]);
        assert_eq!(evaluation.lineages(), 4);
        assert_eq!(evaluation.mean(), (2.0 + 1.0 + 0.0 + 1.0) / 4.0);
        assert_eq!(evaluation.kind_lineages, [3, 1]);
        assert_eq!(evaluation.kind_accepted, [3, 1]);
        assert_eq!(evaluation.reached[..2], [3, 1]);
    }

    #[test]
    fn an_evaluation_of_no_lineages_is_zero() {
        let evaluation = Evaluation::new(3);
        assert_eq!((evaluation.lineages(), evaluation.mean()), (0, 0.0));
    }

    fn saved(progress: &Progress) -> Header {
        progress.header(&identity("a"))
    }

    #[test]
    fn a_state_resumes_where_it_was_with_the_waiting_lineages_and_the_order_of_its_epoch() {
        let mut progress = progress();
        for _ in 0..4 {
            progress.advance();
        }
        progress.pending = 2;
        progress.step = 7;
        progress.best = 1.25;
        let resumed = decide(Found::Header(&saved(&progress)), true, false).unwrap();
        let Resume::Continue(mut resumed) = resumed else {
            panic!("not a state to continue")
        };
        assert_eq!(
            (
                resumed.epoch(),
                resumed.position(),
                resumed.step(),
                resumed.pending,
                resumed.best()
            ),
            (1, (1, 3), 7, 2, 1.25)
        );
        assert_eq!(resumed.advance(), progress.advance());
    }

    #[test]
    fn a_state_of_another_run_is_refused_naming_the_parts_that_differ() {
        let other = Progress::new(settings(), vec![1, 2, 4], 5).header(&identity("b"));
        assert_eq!(
            decide(Found::Header(&other), true, false).err(),
            Some(StateError::Foreign {
                differs: vec!["corpus"]
            })
        );
        let moved = Identity::new("a".to_owned(), "m2".to_owned(), "a".to_owned(), &settings(), 4);
        let other = Progress::new(settings(), vec![1, 2, 4], 5).header(&moved);
        assert_eq!(
            decide(Found::Header(&other), true, false).err(),
            Some(StateError::Foreign {
                differs: vec!["model", "settings"]
            })
        );
    }

    #[test]
    fn a_header_with_no_identity_is_a_torn_state_and_is_refused() {
        assert_eq!(
            decide(Found::Header(&Header::new()), true, false).err(),
            Some(StateError::Torn)
        );
        assert_eq!(
            decide(Found::Header(&Header::new()), false, false).err(),
            Some(StateError::Torn)
        );
    }

    #[test]
    fn nothing_saved_starts_fresh_but_best_parameters_without_a_state_are_refused() {
        assert!(matches!(decide(Found::Missing, false, false), Ok(Resume::Fresh)));
        assert_eq!(decide(Found::Missing, true, false).err(), Some(StateError::Leftover));
    }

    #[test]
    fn a_fresh_start_is_taken_over_whatever_is_saved_and_only_when_asked() {
        let torn = Header::new();
        let foreign = Progress::new(settings(), vec![1, 2, 4], 5).header(&identity("b"));
        for found in [Found::Missing, Found::Header(&torn), Found::Header(&foreign)] {
            assert!(matches!(decide(found, true, true), Ok(Resume::Fresh)));
        }
        assert!(decide(Found::Header(&foreign), true, false).is_err());
    }

    #[test]
    fn the_identity_is_the_digests_and_settings_not_where_the_files_lie() {
        let settings = settings();
        let same = Identity::new("c".to_owned(), "m".to_owned(), "a".to_owned(), &settings, 3);
        assert_eq!(
            same,
            Identity::new("c".to_owned(), "m".to_owned(), "a".to_owned(), &settings, 3)
        );
        let mut other = settings;
        other.tokens += 1;
        assert_ne!(
            same,
            Identity::new("c".to_owned(), "m".to_owned(), "a".to_owned(), &other, 3)
        );
        assert_ne!(
            same,
            Identity::new("c".to_owned(), "m".to_owned(), "a".to_owned(), &settings, 4)
        );
    }

    #[test]
    fn a_state_of_a_finished_run_is_complete_and_trains_nothing() {
        let mut progress = progress();
        progress.epoch = 2;
        let resumed = decide(Found::Header(&saved(&progress)), true, false).unwrap();
        assert!(matches!(resumed, Resume::Complete(ref done) if done.epoch() == 2));
    }

    #[test]
    fn a_state_the_run_cannot_have_reached_or_cannot_read_is_refused() {
        let mut header = saved(&progress());
        header.set("cursor", "4");
        assert!(matches!(
            decide(Found::Header(&header), true, false),
            Err(StateError::Past { cursor: 4, .. })
        ));
        let mut header = saved(&progress());
        header.set("epoch", "3");
        assert!(matches!(
            decide(Found::Header(&header), true, false),
            Err(StateError::Past { .. })
        ));
        let mut header = saved(&progress());
        header.set("epoch", "2");
        header.set("cursor", "1");
        assert!(matches!(
            decide(Found::Header(&header), true, false),
            Err(StateError::Past { .. })
        ));
        for key in ["epoch", "cursor", "step", "best", "pending"] {
            let mut header = saved(&progress());
            header.set(key, "x");
            assert_eq!(
                decide(Found::Header(&header), true, false).err(),
                Some(StateError::Field(match key {
                    "epoch" => EPOCH,
                    "cursor" => CURSOR,
                    "step" => STEP,
                    "best" => BEST,
                    _ => PENDING,
                })),
                "{key}"
            );
            header.remove(key);
            assert!(decide(Found::Header(&header), true, false).is_err(), "{key} missing");
        }
    }

    #[test]
    fn the_best_header_names_the_step_and_the_accepted_drafts() {
        let mut progress = progress();
        progress.step = 4;
        progress.best = 0.5;
        let header = progress.best_header(&identity("a"));
        assert_eq!((header.get("step"), header.get("accepted")), (Some("4"), Some("0.5")));
    }
}
