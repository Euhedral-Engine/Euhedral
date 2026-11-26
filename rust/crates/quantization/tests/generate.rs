//! The drafter's training corpus and the run that writes it: the file format, the rule that picks greedy or sampled
//! answers, and the run on a real lattice with a stand-in for the engine.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_types,
    reason = "a test file: the stand-in engine keeps the requests it was given in a list the test thread takes from"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the frames it published, and reads and damages files"
)]

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_graph::Lake;
use euhedral_inference_engine::generation::{
    CancelHandle, FinishReason, GenerationOutcome, GenerationRequest, HiddenSink, Speculation,
};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::corpus::{self, Entry, Sequence, Writer};
use euhedral_inference_quantization::generate::run::{Answerer, Options, start};
use euhedral_inference_quantization::generate::{GenerateError, Prompt, Summary, answers_greedily, seed};
use euhedral_inference_settings::{ModelSettings, SamplingSettings, Settings};

const WIDTH: usize = 2;

fn le(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// `rows` rows of a sequence's hidden: bytes that say which sequence and which row they are.
fn hidden(tag: u8, rows: usize) -> Vec<u8> {
    (0..rows * WIDTH * 2)
        .map(|index| tag.wrapping_mul(31).wrapping_add(index as u8))
        .collect()
}

fn sequence(id: &str, prompt: usize, answer: &[u32], sampled: bool, tag: u8) -> Sequence {
    let tokens: Vec<u32> = (1..=prompt as u32).chain(answer.iter().copied()).collect();
    let labels = if sampled {
        (1..=prompt as u32)
            .chain(answer.iter().map(|token| token + 100))
            .collect()
    } else {
        tokens.clone()
    };
    Sequence {
        id: id.to_owned(),
        hidden: hidden(tag, tokens.len() - 1),
        tokens,
        labels,
        prompt,
        sampled,
    }
}

fn four_sequences() -> Vec<Sequence> {
    vec![
        sequence("a\"b\\c", 3, &[4, 5], true, 1),
        // An empty answer: the sequence is its prompt.
        sequence("empty", 3, &[], false, 2),
        // An answer that ended at its token budget of four: four tokens run and one more chosen.
        sequence("capped", 2, &[7, 8, 9, 10, 11], false, 3),
        sequence("tab\there", 2, &[6], true, 4),
    ]
}

fn write_all(directory: &Path, sequences: &[Sequence]) {
    let mut writer = Writer::open(directory, WIDTH).unwrap();
    for sequence in sequences {
        writer.append(sequence).unwrap();
    }
}

fn files(directory: &Path) -> Vec<Vec<u8>> {
    ["tokens.bin", "labels.bin", "hidden.bin", "index.jsonl"]
        .iter()
        .map(|name| fs::read(directory.join(name)).unwrap())
        .collect()
}

#[test]
fn a_corpus_is_four_files_in_the_format_the_java_tools_read() {
    let directory = tempfile::tempdir().unwrap();
    let sequences = four_sequences();
    write_all(directory.path(), &sequences);
    let [tokens, labels, hidden_bytes, index] = files(directory.path()).try_into().unwrap();
    let all = |pick: fn(&Sequence) -> &[u32]| sequences.iter().flat_map(pick).copied().collect::<Vec<u32>>();
    assert_eq!(tokens, le(&all(|sequence| &sequence.tokens)));
    assert_eq!(labels, le(&all(|sequence| &sequence.labels)));
    assert_eq!(
        hidden_bytes,
        sequences.iter().flat_map(|s| s.hidden.clone()).collect::<Vec<u8>>()
    );
    let expected = concat!(
        "{\"id\":\"a\\\"b\\\\c\",\"token\":0,\"row\":0,\"length\":5,\"prompt\":3,\"sampled\":true}\n",
        "{\"id\":\"empty\",\"token\":5,\"row\":4,\"length\":3,\"prompt\":3,\"sampled\":false}\n",
        "{\"id\":\"capped\",\"token\":8,\"row\":6,\"length\":7,\"prompt\":2,\"sampled\":false}\n",
        "{\"id\":\"tab\\u0009here\",\"token\":15,\"row\":12,\"length\":3,\"prompt\":2,\"sampled\":true}\n",
    );
    assert_eq!(String::from_utf8(index).unwrap(), expected);
}

#[test]
fn the_index_reads_back_as_written() {
    let directory = tempfile::tempdir().unwrap();
    assert!(corpus::entries(directory.path()).unwrap().is_empty());
    write_all(directory.path(), &four_sequences());
    let entries = corpus::entries(directory.path()).unwrap();
    let expected = [
        ("a\"b\\c", 0, 0, 5, 3, true),
        ("empty", 5, 4, 3, 3, false),
        ("capped", 8, 6, 7, 2, false),
        ("tab\there", 15, 12, 3, 2, true),
    ]
    .map(|(id, token, row, length, prompt, sampled)| Entry {
        id: id.to_owned(),
        token,
        row,
        length,
        prompt,
        sampled,
    });
    assert_eq!(entries, expected);
}

#[test]
fn reopening_cuts_a_torn_tail_back_to_the_index_and_carries_on_as_if_it_never_happened() {
    let torn = tempfile::tempdir().unwrap();
    let clean = tempfile::tempdir().unwrap();
    let sequences = four_sequences();
    write_all(torn.path(), &sequences[..3]);
    // A stopped run left rows past the last index line.
    for name in ["tokens.bin", "labels.bin", "hidden.bin"] {
        let mut file = OpenOptions::new().append(true).open(torn.path().join(name)).unwrap();
        file.write_all(&[0xEE; 37]).unwrap();
    }
    let mut writer = Writer::open(torn.path(), WIDTH).unwrap();
    assert_eq!(writer.ids().len(), 3);
    assert!(writer.ids().contains("capped") && !writer.ids().contains("tab\there"));
    writer.append(&sequences[3]).unwrap();
    write_all(clean.path(), &sequences);
    assert_eq!(files(torn.path()), files(clean.path()));
}

#[test]
fn reopening_a_corpus_without_labels_makes_its_labels_its_tokens() {
    let directory = tempfile::tempdir().unwrap();
    let sequences = [sequence("x", 3, &[4, 5], false, 1), sequence("y", 2, &[6, 7], false, 2)];
    write_all(directory.path(), &sequences);
    let labels = directory.path().join("labels.bin");
    // Only the first sequence's labels survive.
    fs::OpenOptions::new()
        .write(true)
        .open(&labels)
        .unwrap()
        .set_len(4 * 5)
        .unwrap();
    drop(Writer::open(directory.path(), WIDTH).unwrap());
    assert_eq!(
        fs::read(labels).unwrap(),
        fs::read(directory.path().join("tokens.bin")).unwrap()
    );
}

#[test]
fn opening_for_the_ids_alone_leaves_the_hidden_rows_as_they_are() {
    let directory = tempfile::tempdir().unwrap();
    write_all(directory.path(), &four_sequences()[..2]);
    let before = files(directory.path());
    let writer = Writer::open(directory.path(), 0).unwrap();
    assert_eq!(writer.ids().len(), 2);
    assert_eq!(files(directory.path()), before);
}

#[test]
fn a_sequence_whose_hidden_is_not_a_row_for_each_run_token_is_refused_and_writes_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(directory.path(), WIDTH).unwrap();
    let mut short = sequence("short", 3, &[4], false, 1);
    short.hidden.pop();
    let mut labels = sequence("labels", 3, &[4], false, 1);
    labels.labels.pop();
    assert!(writer.append(&short).is_err());
    assert!(writer.append(&labels).is_err());
    assert!(files(directory.path()).iter().all(Vec::is_empty));
}

/// The greedy-or-sampled draw as Java states it, computed here with unsigned arithmetic on the 64-bit FNV-1a hash.
#[test]
fn the_kind_of_answer_follows_the_hash_of_the_prompt_id() {
    assert_eq!(seed(""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(seed("a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(seed("foobar"), 0x8594_4171_f739_67e8);
    // Fractions 0.0345 and 0.4721; "a" has the top bit set, which a signed shift would read as negative.
    for (id, greedy) in [("p0", true), ("p9", true), ("p10", false), ("p11", false), ("a", false)] {
        assert_eq!(answers_greedily(seed(id), 0.25), greedy, "{id} at 0.25");
    }
    assert!(answers_greedily(seed("p10"), 0.5));
    assert!(answers_greedily(seed("a"), 1.0));
    assert!(!answers_greedily(seed("p0"), 0.0));
}

// The run, with a stand-in for the engine that the test plays by hand.

const STOP: u32 = 7;
const MAX_TOKENS: u32 = 2;
const VOCABULARY: usize = 16;

struct Started {
    request: GenerationRequest,
    sink: Box<dyn HiddenSink>,
    cancel: CancelHandle,
}

#[derive(Default)]
struct Fake {
    started: Mutex<Vec<Started>>,
    arrived: Condvar,
    flying: AtomicUsize,
    most_flying: AtomicUsize,
}

impl Answerer for Fake {
    fn answer(
        &self,
        request: GenerationRequest,
        sink: Box<dyn HiddenSink>,
        cancel: CancelHandle,
    ) -> Result<(), String> {
        let flying = self.flying.fetch_add(1, Ordering::SeqCst) + 1;
        self.most_flying.fetch_max(flying, Ordering::SeqCst);
        self.started.lock().unwrap().push(Started { request, sink, cancel });
        self.arrived.notify_all();
        Ok(())
    }
}

impl Fake {
    /// The newest answer asked for, waiting until the run asks for one: a worker of the lattice may still be starting
    /// it when the test thread gets here.
    fn await_next(&self, what: &str) -> Started {
        let mut started = self.started.lock().unwrap();
        loop {
            if let Some(next) = started.pop() {
                self.flying.fetch_sub(1, Ordering::SeqCst);
                return next;
            }
            let (guard, timeout) = self.arrived.wait_timeout(started, Duration::from_secs(60)).unwrap();
            assert!(!timeout.timed_out(), "the run never asked the engine for {what}");
            started = guard;
        }
    }

    fn take(&self) -> Option<Started> {
        let started = self.started.lock().unwrap().pop();
        if started.is_some() {
            // The test plays an answer to its end at once, so taking it is its end for the count.
            self.flying.fetch_sub(1, Ordering::SeqCst);
        }
        started
    }
}

fn rig() -> (Arc<ControlPlaneLattice>, Arc<Lake>) {
    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
    let host = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    (lattice, host)
}

fn options(batch: usize) -> Options {
    Options {
        max_tokens: MAX_TOKENS,
        greedy_fraction: 0.25,
        batch,
        sampling: SamplingSettings::default(),
        stops: vec![STOP],
        vocabulary: VOCABULARY,
    }
}

/// A prompt whose tokens say which it is: its first token is `10 + number`, and it has `4 + number` of them.
fn prompt(id: &str, number: u32) -> Prompt {
    Prompt {
        id: id.to_owned(),
        tokens: (0..4 + number).map(|index| 10 + number + 100 * index).collect(),
    }
}

/// The tokens the stand-in answers `prompt` with: one to three of them, ending in the stop token when shorter than
/// the budget of three.
fn answer(prompt: &[u32]) -> Vec<u32> {
    let length = 1 + prompt[0] as usize % 3;
    let mut tokens: Vec<u32> = (0..length as u32).map(|index| 200 + prompt[0] + index).collect();
    if length < MAX_TOKENS as usize + 1 {
        *tokens.last_mut().unwrap() = STOP;
    }
    tokens
}

fn row(prompt: &[u32], position: usize) -> Vec<u8> {
    let tag = prompt[0] as u8;
    (0..WIDTH * 2)
        .map(|index| tag ^ (position as u8).wrapping_mul(17) ^ index as u8)
        .collect()
}

fn bf16_logits(best: usize) -> Vec<u8> {
    (0..VOCABULARY)
        .flat_map(|token| if token == best { [0x80, 0x3F] } else { [0, 0] })
        .collect()
}

/// The token the greedy choice is at the row that selects answer token `index`.
fn choice(prompt: &[u32], index: usize) -> u32 {
    ((prompt.len() - 1 + index) % 13 + 1) as u32
}

/// Plays one answer the way the engine delivers it: the prefill in two chunks (the last with the logits), each token
/// as it is selected, and a decode row for every token but the last; then the end.
fn play(started: Started, reason: Option<FinishReason>) {
    let Started { request, mut sink, .. } = started;
    let mut listener = request.listener;
    let prompt = request.prompt;
    let tokens = answer(&prompt);
    let rows_of = |from: usize, to: usize| {
        (from..to)
            .flat_map(|position| row(&prompt, position))
            .collect::<Vec<u8>>()
    };
    sink.rows(0, 2, &rows_of(0, 2), None);
    let logits = |index| bf16_logits(choice(&prompt, index) as usize);
    sink.rows(2, prompt.len() - 2, &rows_of(2, prompt.len()), Some(&logits(0)));
    let now = Instant::now();
    for (index, token) in tokens.iter().enumerate() {
        listener.token(*token, now);
        if index + 1 < tokens.len() {
            let position = prompt.len() + index;
            sink.rows(
                position as u64,
                1,
                &rows_of(position, position + 1),
                Some(&logits(index + 1)),
            );
        }
    }
    let reason = reason.unwrap_or(if *tokens.last().unwrap() == STOP {
        FinishReason::Stop
    } else {
        FinishReason::Length
    });
    listener.finished(GenerationOutcome {
        reason,
        prompt_tokens: prompt.len(),
        generated: tokens.len(),
        elapsed: Duration::ZERO,
        cached_prompt_tokens: 0,
    });
}

fn expected(prompt: &Prompt, sampled: bool) -> Sequence {
    let tokens = answer(&prompt.tokens);
    let labels = if sampled {
        (0..tokens.len()).map(|index| choice(&prompt.tokens, index)).collect()
    } else {
        tokens.clone()
    };
    let rows = prompt.tokens.len() + tokens.len() - 1;
    Sequence {
        id: prompt.id.clone(),
        hidden: (0..rows).flat_map(|position| row(&prompt.tokens, position)).collect(),
        prompt: prompt.tokens.len(),
        sampled,
        tokens: [prompt.tokens.clone(), tokens.clone()].concat(),
        labels: [prompt.tokens.clone(), labels].concat(),
    }
}

fn wait<T>(done: &euhedral_inference_engine::Handoff<T>) -> T {
    done.wait(Duration::from_secs(60)).expect("the run never ended")
}

#[test]
fn every_answer_reaches_the_corpus_whole_whatever_order_the_answers_end_in() {
    let directory = tempfile::tempdir().unwrap();
    let (lattice, host) = rig();
    // p0 and p1 are answered greedily, the rest sampled.
    let prompts: Vec<Prompt> = ["p0", "a", "p10", "p1", "p11"]
        .iter()
        .enumerate()
        .map(|(number, id)| prompt(id, number as u32))
        .collect();
    let fake = Arc::new(Fake::default());
    let writer = Writer::open(directory.path(), WIDTH).unwrap();
    let done = start(
        &host,
        Arc::clone(&fake) as Arc<dyn Answerer>,
        writer,
        prompts.clone(),
        options(2),
    );
    // The newest answer ends first, so the answers end out of order; each end starts the next prompt.
    for number in 1..=5 {
        let started = fake.await_next(&format!("prompt {number} of 5"));
        let greedy = started.request.sampling.is_greedy();
        assert_eq!(started.request.max_tokens, MAX_TOKENS + 1);
        assert_eq!(started.request.stop_tokens, [STOP]);
        assert_eq!(started.request.speculation, Speculation::Off);
        assert!(!started.cancel.is_cancelled());
        let found = prompts
            .iter()
            .find(|prompt| prompt.tokens == started.request.prompt)
            .unwrap();
        assert_eq!(greedy, matches!(found.id.as_str(), "p0" | "p1"), "{}", found.id);
        play(started, None);
    }
    assert_eq!(
        wait(&done).unwrap(),
        Summary {
            answered: 5,
            tokens: prompts.iter().map(|p| answer(&p.tokens).len()).sum()
        }
    );
    assert_eq!(fake.most_flying.load(Ordering::SeqCst), 2);

    let entries = corpus::entries(directory.path()).unwrap();
    let [tokens, labels, hidden_bytes, _index] = files(directory.path()).try_into().unwrap();
    assert_eq!(entries.len(), 5);
    for prompt in &prompts {
        let sampled = !matches!(prompt.id.as_str(), "p0" | "p1");
        let want = expected(prompt, sampled);
        let entry = entries.iter().find(|entry| entry.id == prompt.id).unwrap();
        assert_eq!(
            (entry.length, entry.prompt, entry.sampled),
            (want.tokens.len() as u64, want.prompt as u64, sampled)
        );
        let (token, rows) = (4 * entry.token as usize, 2 * WIDTH * entry.row as usize);
        assert_eq!(
            tokens[token..token + 4 * want.tokens.len()],
            le(&want.tokens)[..],
            "{}",
            prompt.id
        );
        assert_eq!(
            labels[token..token + 4 * want.labels.len()],
            le(&want.labels)[..],
            "{}",
            prompt.id
        );
        assert_eq!(
            hidden_bytes[rows..rows + want.hidden.len()],
            want.hidden[..],
            "{}",
            prompt.id
        );
    }
    host.complete_gracefully();
    lattice.close();
}

#[test]
fn a_failed_answer_cancels_the_others_and_ends_the_run_with_its_reason_and_nothing_of_it_written() {
    let directory = tempfile::tempdir().unwrap();
    let (lattice, host) = rig();
    let prompts: Vec<Prompt> = ["p0", "a", "p10"]
        .iter()
        .enumerate()
        .map(|(n, id)| prompt(id, n as u32))
        .collect();
    let fake = Arc::new(Fake::default());
    let writer = Writer::open(directory.path(), WIDTH).unwrap();
    let done = start(
        &host,
        Arc::clone(&fake) as Arc<dyn Answerer>,
        writer,
        prompts,
        options(2),
    );
    let failing = fake.await_next("the first prompt");
    let other = fake.await_next("the second prompt");
    assert!(fake.take().is_none());
    let other_cancel = other.cancel.clone();
    play(failing, Some(FinishReason::ContextFull));
    // The other answer is told to stop, and the third prompt never starts.
    play(other, Some(FinishReason::Cancelled));
    let error = wait(&done).unwrap_err();
    assert!(other_cancel.is_cancelled(), "the other answer was not told to stop");
    assert!(
        matches!(error, GenerateError::Engine(_)) && error.to_string().contains("ContextFull"),
        "{error}"
    );
    assert!(corpus::entries(directory.path()).unwrap().is_empty());
    assert!(fake.take().is_none());
    host.complete_gracefully();
    lattice.close();
}

#[test]
fn rows_that_do_not_follow_on_end_the_run_as_inconsistent() {
    let directory = tempfile::tempdir().unwrap();
    let (lattice, host) = rig();
    let fake = Arc::new(Fake::default());
    let writer = Writer::open(directory.path(), WIDTH).unwrap();
    let done = start(
        &host,
        Arc::clone(&fake) as Arc<dyn Answerer>,
        writer,
        vec![prompt("p0", 0)],
        options(1),
    );
    let mut started = fake.await_next("the prompt");
    started.sink.rows(3, 1, &row(&[0], 3), None);
    let mut listener = started.request.listener;
    listener.finished(GenerationOutcome {
        reason: FinishReason::Cancelled,
        prompt_tokens: 4,
        generated: 0,
        elapsed: Duration::ZERO,
        cached_prompt_tokens: 0,
    });
    let error = wait(&done).unwrap_err();
    assert!(matches!(error, GenerateError::Inconsistent(_)), "{error}");
    host.complete_gracefully();
    lattice.close();
}

#[test]
fn a_run_with_no_prompts_ends_at_once_having_written_nothing() {
    let directory = tempfile::tempdir().unwrap();
    let (lattice, host) = rig();
    let fake = Arc::new(Fake::default());
    let writer = Writer::open(directory.path(), WIDTH).unwrap();
    let done = start(&host, fake as Arc<dyn Answerer>, writer, Vec::new(), options(4));
    assert_eq!(wait(&done).unwrap(), Summary::default());
    host.complete_gracefully();
    lattice.close();
}
