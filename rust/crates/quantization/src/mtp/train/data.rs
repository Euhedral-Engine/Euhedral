//! The corpus a run trains on: its sequences (tokens and the base model's choices, in memory) and where each one's hidden
//! rows are in `hidden.bin`, with the split into training and held-out sequences. Checked once, before the device is
//! touched, so that a window never reads outside a file.

use crate::corpus::Entry;

use super::error::RunError;
use super::settings::{MAX_POSITIONS, Settings};
use super::stage::Sequence;

/// The checked sequences of a corpus.
#[derive(Debug)]
pub struct Data {
    entries: Vec<Entry>,
    tokens: Vec<i32>,
    labels: Vec<i32>,
    training: Vec<usize>,
    held_out: Vec<usize>,
}

/// The files of a corpus as the run needs them.
#[derive(Debug)]
pub struct Files {
    /// The index lines, as many as the run uses.
    pub entries: Vec<Entry>,
    /// Every token of `tokens.bin`, and the base model's choice at each.
    pub tokens: Vec<i32>,
    pub labels: Vec<i32>,
    /// The rows `hidden.bin` holds.
    pub hidden_rows: u64,
    /// The vocabulary of the base model: every token and label is below it.
    pub vocabulary: u64,
}

fn refuse(message: String) -> RunError {
    RunError::Input(message)
}

impl Data {
    /// Checks `files` for a run of `settings` and splits the sequences: those whose index is a multiple of `holdout` are
    /// held out. Refuses a corpus the run cannot use, naming the sequence.
    pub fn new(files: Files, settings: &Settings) -> Result<Data, RunError> {
        let Files {
            entries,
            tokens,
            labels,
            hidden_rows,
            vocabulary,
        } = files;
        if entries.is_empty() {
            return Err(refuse("the corpus holds no sequences".to_owned()));
        }
        if tokens.len() != labels.len() {
            return Err(refuse(format!(
                "tokens.bin holds {} tokens and labels.bin {}",
                tokens.len(),
                labels.len()
            )));
        }
        if let Some(bad) = tokens
            .iter()
            .chain(&labels)
            .find(|&&token| u64::try_from(token).map_or(true, |token| token >= vocabulary))
        {
            return Err(refuse(format!(
                "the corpus holds token {bad}, outside the vocabulary of {vocabulary}"
            )));
        }
        let (mut training, mut held_out) = (Vec::new(), Vec::new());
        for (index, entry) in entries.iter().enumerate() {
            let place = || format!("sequence {}", entry.id);
            if entry.length > u64::from(MAX_POSITIONS) {
                return Err(refuse(format!(
                    "{} is too long: {} tokens, at most {MAX_POSITIONS}",
                    place(),
                    entry.length
                )));
            }
            if entry.length < 2 || entry.prompt > entry.length {
                return Err(refuse(format!(
                    "{} has {} tokens of which {} are the prompt",
                    place(),
                    entry.length,
                    entry.prompt
                )));
            }
            let tokens_end = entry.token.checked_add(entry.length);
            let rows_end = entry.row.checked_add(entry.length - 1);
            if tokens_end.is_none_or(|end| end > tokens.len() as u64) || rows_end.is_none_or(|end| end > hidden_rows) {
                return Err(refuse(format!("{} reaches past the end of the corpus files", place())));
            }
            if settings.held_out(index) {
                held_out.push(index);
            } else {
                training.push(index);
            }
        }
        if training.is_empty() || held_out.is_empty() {
            return Err(refuse(format!(
                "{} sequences with --holdout {} leave {} to train on and {} to evaluate; both are needed",
                entries.len(),
                settings.holdout,
                training.len(),
                held_out.len()
            )));
        }
        Ok(Data {
            entries,
            tokens,
            labels,
            training,
            held_out,
        })
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The indices of the training sequences, in corpus order.
    pub fn training(&self) -> &[usize] {
        &self.training
    }

    /// The indices of the held-out sequences, in corpus order.
    pub fn held_out(&self) -> &[usize] {
        &self.held_out
    }

    /// Sequence `index`: its tokens and labels. `index` is one the corpus has.
    pub fn sequence(&self, index: usize) -> Result<Sequence<'_>, RunError> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| RunError::Broken(format!("the corpus has no sequence {index}")))?;
        let range = entry.token as usize..(entry.token + entry.length) as usize;
        match (self.tokens.get(range.clone()), self.labels.get(range)) {
            (Some(tokens), Some(labels)) => Ok(Sequence {
                tokens,
                labels,
                prompt: entry.prompt as usize,
            }),
            _ => Err(RunError::Broken(format!(
                "sequence {} is outside the corpus files",
                entry.id
            ))),
        }
    }

    /// The byte range of sequence `index`'s hidden rows from window row `a`, `count` rows of `width` BF16 values.
    pub fn hidden_range(&self, index: usize, a: u32, count: u32, width: u32) -> Result<(u64, usize), RunError> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| RunError::Broken(format!("the corpus has no sequence {index}")))?;
        let row_bytes = 2 * u64::from(width);
        let first = entry.row + u64::from(a);
        if u64::from(a) + u64::from(count) > entry.length - 1 {
            return Err(RunError::Broken(format!(
                "a window past the rows of sequence {}",
                entry.id
            )));
        }
        Ok((first * row_bytes, (u64::from(count) * row_bytes) as usize))
    }

    /// Whether sequence `index` is one of the sampled answers.
    pub fn sampled(&self, index: usize) -> bool {
        self.entries.get(index).is_some_and(|entry| entry.sampled)
    }

    /// For the schedule: each training sequence's length and prompt.
    pub fn lengths(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.training
            .iter()
            .filter_map(|&index| self.entries.get(index))
            .map(|entry| (entry.length, entry.prompt))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn entry(id: &str, token: u64, row: u64, length: u64, prompt: u64) -> Entry {
        Entry {
            id: id.to_owned(),
            token,
            row,
            length,
            prompt,
            sampled: false,
        }
    }

    /// `count` sequences of 5 tokens (2 of them the prompt), back to back.
    fn files(count: u64) -> Files {
        Files {
            entries: (0..count)
                .map(|i| entry(&format!("s{i}"), 5 * i, 4 * i, 5, 2))
                .collect(),
            tokens: (0..5 * count as i32).map(|token| token % 7).collect(),
            labels: (0..5 * count as i32).map(|token| (token + 1) % 7).collect(),
            hidden_rows: 4 * count,
            vocabulary: 8,
        }
    }

    fn settings(holdout: u32) -> Settings {
        Settings {
            holdout,
            ..Settings::default()
        }
    }

    #[test]
    fn every_holdout_th_sequence_is_held_out_and_the_rest_train() {
        let data = Data::new(files(7), &settings(3)).unwrap();
        assert_eq!(data.held_out(), [0, 3, 6]);
        assert_eq!(data.training(), [1, 2, 4, 5]);
    }

    #[test]
    fn a_sequence_gives_its_own_tokens_and_choices() {
        let data = Data::new(files(3), &settings(2)).unwrap();
        let second = data.sequence(1).unwrap();
        assert_eq!(second.tokens, [5, 6, 0, 1, 2]);
        assert_eq!(second.labels, [6, 0, 1, 2, 3]);
        assert_eq!(second.prompt, 2);
        assert!(data.sequence(3).is_err());
    }

    #[test]
    fn a_windows_hidden_rows_start_at_its_sequence_row_plus_its_first_lineage() {
        let data = Data::new(files(3), &settings(2)).unwrap();
        // Sequence 1's rows start at row 4; a window from lineage 1 of 2 rows of width 8 is bytes 5 * 16 .. + 32.
        assert_eq!(data.hidden_range(1, 1, 2, 8).unwrap(), (80, 32));
        // A sequence of 5 tokens has 4 rows: a window may end at the fourth and no later.
        assert!(data.hidden_range(1, 2, 2, 8).is_ok());
        assert!(data.hidden_range(1, 2, 3, 8).is_err());
    }

    #[test]
    fn a_corpus_the_run_cannot_use_is_refused() {
        let long = |length: u64| {
            let mut files = files(2);
            files.entries[1] = entry("long", 5, 4, length, 2);
            files.tokens = vec![0; 5 + length as usize];
            files.labels = vec![0; 5 + length as usize];
            files.hidden_rows = 4 + length;
            files
        };
        let cases: Vec<(Files, &str)> = vec![
            (long(u64::from(MAX_POSITIONS) + 1), "too long"),
            (
                Files {
                    entries: Vec::new(),
                    ..files(1)
                },
                "no sequences",
            ),
            (
                Files {
                    labels: vec![0; 3],
                    ..files(2)
                },
                "labels.bin",
            ),
            (
                Files {
                    tokens: vec![8; 10],
                    ..files(2)
                },
                "outside the vocabulary",
            ),
            (
                Files {
                    hidden_rows: 7,
                    ..files(2)
                },
                "past the end",
            ),
            (
                Files {
                    tokens: vec![0; 9],
                    labels: vec![0; 9],
                    ..files(2)
                },
                "past the end",
            ),
        ];
        for (case, wanted) in cases {
            let message = Data::new(case, &settings(2)).unwrap_err().to_string();
            assert!(message.contains(wanted), "{message}");
        }
        let mut negative = files(2);
        negative.tokens[3] = -1;
        assert!(Data::new(negative, &settings(2)).is_err());
        let mut short = files(2);
        short.entries[0] = entry("short", 0, 0, 1, 0);
        assert!(Data::new(short, &settings(2)).is_err());
    }

    #[test]
    fn a_split_with_nothing_to_train_or_nothing_to_evaluate_is_refused() {
        // One sequence is held out and leaves nothing to train on; a holdout of 1 holds out all.
        assert!(
            Data::new(files(1), &settings(50))
                .unwrap_err()
                .to_string()
                .contains("both are needed")
        );
        assert!(
            Data::new(files(4), &settings(1))
                .unwrap_err()
                .to_string()
                .contains("both are needed")
        );
        assert!(Data::new(files(2), &settings(2)).is_ok());
    }

    #[test]
    fn the_schedule_counts_the_training_sequences_lengths_and_prompts() {
        let data = Data::new(files(3), &settings(3)).unwrap();
        assert_eq!(data.lengths().collect::<Vec<_>>(), [(5, 2), (5, 2)]);
    }
}
