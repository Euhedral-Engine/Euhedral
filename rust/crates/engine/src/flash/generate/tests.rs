#![allow(
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "the fakes share their record with the test through a mutex: the test reads it between turns"
)]

use std::sync::{Arc, Mutex};

use euhedral_data_structures::queues::{MpscQueue, Multi, Producer};

use super::*;
use crate::generation::ConstraintError;
use crate::sampling::SamplingConfig;

const VOCABULARY: usize = 8;

/// What the fake device saw and what it will answer.
#[derive(Default)]
struct Device {
    begun: Vec<Step>,
    committed: Vec<Vec<i32>>,
    discarded: usize,
    refuse_begin: Option<String>,
    refuse_commit: bool,
    /// The token the device argmax chose; `None` when nothing could be selected.
    selected: Option<u32>,
    /// The BF16 row the host reads: the token with the largest logit.
    row: Option<Vec<u8>>,
}

struct Fake {
    device: Arc<Mutex<Device>>,
    /// The row copied back by the last quantum begun.
    row: Option<Vec<u8>>,
}

impl Stepper for Fake {
    fn begin(&mut self, step: &Step) -> Result<(), String> {
        let mut device = self.device.lock().unwrap();
        if let Some(why) = device.refuse_begin.clone() {
            return Err(why);
        }
        device.begun.push(step.clone());
        self.row = device.row.clone();
        Ok(())
    }

    fn selected(&self) -> Option<u32> {
        self.device.lock().unwrap().selected
    }

    fn row(&self) -> Option<&[u8]> {
        self.row.as_deref()
    }

    fn commit(&mut self, next_context: &[i32]) -> Result<(), String> {
        let mut device = self.device.lock().unwrap();
        if device.refuse_commit {
            return Err("full".to_owned());
        }
        device.committed.push(next_context.to_vec());
        Ok(())
    }

    fn discard(&mut self) {
        self.device.lock().unwrap().discarded += 1;
    }
}

type Log = Arc<Mutex<Vec<String>>>;

struct Recorder(Log);

impl GenerationListener for Recorder {
    fn prefill_done(&mut self, _at: Instant) {
        self.0.lock().unwrap().push("prefill".to_owned());
    }

    fn token(&mut self, id: u32, _at: Instant) {
        self.0.lock().unwrap().push(format!("token {id}"));
    }

    fn quantum(&mut self, event: QuantumEvent) {
        self.0.lock().unwrap().push(format!(
            "quantum {:?} {} at {}",
            event.kind, event.rows, event.start_position
        ));
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        self.0
            .lock()
            .unwrap()
            .push(format!("finished {:?} {}", outcome.reason, outcome.generated));
    }
}

/// A constraint that allows only the tokens below `limit` and counts its calls.
struct Below {
    limit: u32,
    /// Allows every token and refuses each one it is told to accept.
    refuses: bool,
    prepared: Arc<Mutex<usize>>,
    accepted: Arc<Mutex<Vec<u32>>>,
}

impl TokenFilter for Below {
    fn allows(&self, id: u32) -> bool {
        self.refuses || id < self.limit
    }
}

impl TokenConstraint for Below {
    fn prepare(&mut self) -> Result<(), ConstraintError> {
        *self.prepared.lock().unwrap() += 1;
        Ok(())
    }

    fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
        if self.refuses || id >= self.limit {
            return Err(ConstraintError::new("above the limit"));
        }
        self.accepted.lock().unwrap().push(id);
        Ok(())
    }
}

struct Rig {
    owner: FlashOwner<Fake>,
    producer: Producer<Message, Multi>,
    device: Arc<Mutex<Device>>,
    log: Log,
}

fn rig(chunk: u32, capacity: u32) -> Rig {
    let queue = MpscQueue::<Message>::new(4);
    let device = Arc::new(Mutex::new(Device::default()));
    let limits = Limits {
        vocabulary: VOCABULARY,
        capacity,
        chunk,
    };
    Rig {
        owner: FlashOwner::new(
            Fake {
                device: Arc::clone(&device),
                row: None,
            },
            queue.consumer(),
            limits,
        ),
        producer: queue.producer(),
        device,
        log: Log::default(),
    }
}

fn request(rig: &Rig, prompt: &[u32], wanted: u32, stops: &[u32]) -> GenerationRequest {
    GenerationRequest {
        prompt: prompt.to_vec(),
        max_tokens: wanted,
        stop_tokens: stops.to_vec(),
        speculation: Speculation::Off,
        sampling: SamplingConfig::greedy(1),
        constraint: None,
        listener: Box::new(Recorder(Arc::clone(&rig.log))),
    }
}

impl Rig {
    fn refusal(&mut self, asked: GenerationRequest) -> Option<AdmissionError> {
        self.owner
            .begin(asked, CancelHandle::default())
            .err()
            .map(|rejected| rejected.error)
    }

    fn start(&mut self, request: GenerationRequest) {
        assert!(
            self.owner.begin(request, CancelHandle::default()).is_ok(),
            "the request was refused"
        );
    }

    /// The quantum in flight retires with `context` after it, and the owner takes its turn.
    fn retire(&mut self, context: &[i32]) {
        let retired = Retired {
            result: Ok(context.into()),
            at: Instant::now(),
        };
        assert!(
            self.producer.offer(Message::Retired(retired)).is_ok(),
            "the inbox refused a message"
        );
        self.owner.turn();
    }

    fn fail(&mut self, why: &str) {
        let retired = Retired {
            result: Err(why.to_owned()),
            at: Instant::now(),
        };
        assert!(
            self.producer.offer(Message::Retired(retired)).is_ok(),
            "the inbox refused a message"
        );
        self.owner.turn();
    }

    fn select(&self, token: u32) {
        self.device.lock().unwrap().selected = Some(token);
    }

    /// A host row whose largest logit is `token`'s.
    fn row_for(&self, token: usize) {
        let mut row = vec![0u8; VOCABULARY * 2];
        // BF16 1.0 is 0x3F80, little-endian.
        row[token * 2 + 1] = 0x3F;
        row[token * 2] = 0x80;
        self.device.lock().unwrap().row = Some(row);
    }

    fn begun(&self) -> Vec<Step> {
        self.device.lock().unwrap().begun.clone()
    }

    fn told(&mut self) -> Vec<String> {
        for finished in self.owner.take_finished() {
            FlashOwner::<Fake>::deliver(finished);
        }
        self.log.lock().unwrap().clone()
    }
}

#[test]
fn prompt_runs_in_chunks_and_only_the_last_scores() {
    let mut rig = rig(2, 64);
    rig.select(7);
    rig.start(request(&rig, &[1, 2, 3, 4, 5], 1, &[]));
    assert_eq!(
        rig.begun(),
        vec![Step {
            tokens: vec![1, 2],
            score: Score::None,
            fresh: true,
            kind: QuantumKind::Prefill,
        }]
    );
    rig.retire(&[10]);
    rig.retire(&[11]);
    rig.retire(&[12]);
    let begun = rig.begun();
    assert_eq!(begun.len(), 3);
    assert_eq!(
        begun[1],
        Step {
            tokens: vec![3, 4],
            score: Score::None,
            fresh: false,
            kind: QuantumKind::Prefill,
        }
    );
    assert_eq!(
        begun[2],
        Step {
            tokens: vec![5],
            score: Score::Device,
            fresh: false,
            kind: QuantumKind::Prefill,
        }
    );
    assert_eq!(rig.device.lock().unwrap().committed, vec![vec![10], vec![11], vec![12]]);
    assert_eq!(
        rig.told(),
        vec![
            "quantum Prefill 2 at 0",
            "quantum Prefill 2 at 2",
            "quantum Prefill 1 at 4",
            "prefill",
            "token 7",
            "finished Length 1",
        ]
    );
}

#[test]
fn decode_feeds_the_previous_token_and_the_budget_ends_it() {
    let mut rig = rig(8, 64);
    rig.select(3);
    rig.start(request(&rig, &[1, 2], 3, &[]));
    rig.retire(&[]);
    rig.select(4);
    rig.retire(&[]);
    rig.select(5);
    rig.retire(&[]);
    let begun = rig.begun();
    assert_eq!(
        begun[1],
        Step {
            tokens: vec![3],
            score: Score::Device,
            fresh: false,
            kind: QuantumKind::Decode,
        }
    );
    assert_eq!(
        begun[2],
        Step {
            tokens: vec![4],
            score: Score::Device,
            fresh: false,
            kind: QuantumKind::Decode,
        }
    );
    assert_eq!(begun.len(), 3);
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished Length 3");
    assert!(told.contains(&"quantum Decode 1 at 2".to_owned()));
    assert!(told.contains(&"quantum Decode 1 at 3".to_owned()));
}

#[test]
fn a_stop_token_is_reported_and_not_fed_back() {
    let mut rig = rig(8, 64);
    rig.select(6);
    rig.start(request(&rig, &[1], 10, &[6]));
    rig.retire(&[]);
    assert_eq!(rig.begun().len(), 1);
    let told = rig.told();
    assert_eq!(told[told.len() - 2..], ["token 6", "finished Stop 1"]);
}

#[test]
fn a_sampled_request_selects_on_the_host() {
    let mut rig = rig(8, 64);
    let mut asked = request(&rig, &[1], 1, &[]);
    asked.sampling = SamplingConfig::new(0.7, 0, 1.0, 5, false).unwrap();
    rig.row_for(2);
    rig.start(asked);
    assert_eq!(rig.begun()[0].score, Score::Host);
    rig.retire(&[]);
    let told = rig.told();
    assert!(
        told.iter().any(|line| line.starts_with("token ")),
        "no token was selected: {told:?}"
    );
}

#[test]
fn a_sampled_request_with_top_k_one_takes_the_largest_logit() {
    let mut rig = rig(8, 64);
    let mut asked = request(&rig, &[1], 1, &[]);
    asked.sampling = SamplingConfig::new(0.7, 1, 1.0, 5, false).unwrap();
    rig.row_for(2);
    rig.start(asked);
    rig.retire(&[]);
    let told = rig.told();
    assert!(
        told.contains(&"token 2".to_owned()),
        "the largest logit was not taken: {told:?}"
    );
}

#[test]
fn a_sampled_request_draws_the_same_tokens_for_the_same_seed() {
    let mut drawn = Vec::new();
    for _ in 0..2 {
        let mut rig = rig(8, 64);
        let mut asked = request(&rig, &[1], 3, &[]);
        asked.sampling = SamplingConfig::new(1.0, 0, 1.0, 9, false).unwrap();
        // Every logit equal: the draw alone chooses.
        rig.device.lock().unwrap().row = Some(vec![0u8; VOCABULARY * 2]);
        rig.start(asked);
        for _ in 0..3 {
            rig.retire(&[]);
        }
        let told = rig.told();
        let tokens: Vec<String> = told.into_iter().filter(|line| line.starts_with("token ")).collect();
        assert_eq!(tokens.len(), 3, "a token was not drawn at each step: {tokens:?}");
        drawn.push(tokens);
    }
    assert_eq!(drawn[0], drawn[1], "one seed drew different tokens");
}

#[test]
fn the_first_step_of_a_later_request_returns_the_sequence_to_the_start() {
    let mut rig = rig(8, 64);
    rig.select(7);
    rig.start(request(&rig, &[1, 2], 1, &[]));
    rig.retire(&[10]);
    rig.retire(&[11]);
    rig.told();
    rig.start(request(&rig, &[3], 1, &[]));
    let begun = rig.begun();
    let fresh: Vec<bool> = begun.iter().map(|step| step.fresh).collect();
    assert_eq!(fresh.first(), Some(&true));
    assert_eq!(
        fresh.last(),
        Some(&true),
        "the second request's first step continued the old position"
    );
    assert_eq!(begun.last().map(|step| step.tokens.clone()), Some(vec![3]));
}

#[test]
fn a_constrained_greedy_request_selects_on_the_host_and_the_constraint_accepts() {
    let mut rig = rig(8, 64);
    let (prepared, accepted) = (Arc::new(Mutex::new(0)), Arc::new(Mutex::new(Vec::new())));
    let mut asked = request(&rig, &[1], 1, &[]);
    asked.constraint = Some(Box::new(Below {
        limit: 4,
        refuses: false,
        prepared: Arc::clone(&prepared),
        accepted: Arc::clone(&accepted),
    }));
    // The unconstrained best is 6, which the constraint forbids; token 3 is the best allowed.
    rig.row_for(6);
    {
        let mut device = rig.device.lock().unwrap();
        let row = device.row.as_mut().unwrap();
        row[3 * 2 + 1] = 0x3F;
        row[3 * 2] = 0x00;
    }
    rig.start(asked);
    assert_eq!(rig.begun()[0].score, Score::Host);
    assert_eq!(*prepared.lock().unwrap(), 1);
    rig.retire(&[]);
    assert_eq!(*accepted.lock().unwrap(), vec![3]);
    assert!(rig.told().contains(&"token 3".to_owned()));
}

#[test]
fn a_constraint_that_refuses_the_selected_token_fails_the_request() {
    let mut rig = rig(8, 64);
    let mut asked = request(&rig, &[1], 1, &[]);
    asked.constraint = Some(Box::new(Below {
        limit: 0,
        refuses: true,
        prepared: Arc::default(),
        accepted: Arc::default(),
    }));
    rig.row_for(2);
    rig.start(asked);
    rig.retire(&[]);
    let told = rig.told();
    assert!(
        told[told.len() - 1].starts_with("finished Failed(\"the constraint refused token"),
        "{told:?}"
    );
}

#[test]
fn a_device_selection_of_nothing_fails_the_request() {
    let mut rig = rig(8, 64);
    rig.start(request(&rig, &[1], 1, &[]));
    rig.retire(&[]);
    let told = rig.told();
    assert!(told[told.len() - 1].contains("the head selected no token"), "{told:?}");
}

#[test]
fn a_prompt_with_nothing_to_generate_scores_nothing() {
    let mut rig = rig(2, 64);
    rig.start(request(&rig, &[1, 2, 3], 0, &[]));
    rig.retire(&[]);
    rig.retire(&[]);
    let begun = rig.begun();
    assert!(begun.iter().all(|step| step.score == Score::None), "{begun:?}");
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished Length 0");
    assert!(!told.contains(&"prefill".to_owned()));
}

#[test]
fn a_cancel_ends_the_request_at_the_next_boundary() {
    let mut rig = rig(2, 64);
    let cancel = CancelHandle::default();
    assert!(
        rig.owner
            .begin(request(&rig, &[1, 2, 3, 4], 5, &[]), cancel.clone())
            .is_ok()
    );
    cancel.cancel();
    rig.retire(&[]);
    // The quantum in flight finished; no further one began.
    assert_eq!(rig.begun().len(), 1);
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished Cancelled 0");
}

#[test]
fn a_failed_quantum_is_discarded_and_ends_the_request() {
    let mut rig = rig(8, 64);
    rig.start(request(&rig, &[1], 2, &[]));
    rig.fail("the lane failed");
    assert_eq!(rig.device.lock().unwrap().discarded, 1);
    assert!(rig.device.lock().unwrap().committed.is_empty());
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished Failed(\"the lane failed\") 0");
    // The sequence is free for the next request.
    rig.select(1);
    rig.start(request(&rig, &[1], 1, &[]));
}

#[test]
fn a_quantum_that_cannot_start_fails_the_request() {
    let mut rig = rig(8, 64);
    rig.device.lock().unwrap().refuse_begin = Some("no room".to_owned());
    rig.start(request(&rig, &[1], 2, &[]));
    assert_eq!(rig.device.lock().unwrap().discarded, 1);
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished Failed(\"no room\") 0");
}

#[test]
fn a_commit_that_is_refused_fails_the_request() {
    let mut rig = rig(8, 64);
    rig.start(request(&rig, &[1], 2, &[]));
    rig.device.lock().unwrap().refuse_commit = true;
    rig.retire(&[]);
    let told = rig.told();
    assert!(told[told.len() - 1].contains("not committed"), "{told:?}");
}

#[test]
fn the_context_ends_the_request_when_the_sequence_is_full() {
    let mut rig = rig(8, 3);
    rig.select(2);
    rig.start(request(&rig, &[1, 1], 10, &[]));
    rig.retire(&[]);
    rig.retire(&[]);
    let told = rig.told();
    assert_eq!(told[told.len() - 1], "finished ContextFull 2");
    assert_eq!(rig.begun().len(), 2);
}

#[test]
fn refused_requests_name_the_reason() {
    let mut rig = rig(8, 4);
    assert_eq!(
        rig.refusal(request(&rig, &[], 1, &[])),
        Some(AdmissionError::EmptyPrompt)
    );
    assert_eq!(
        rig.refusal(request(&rig, &[1, 9], 1, &[])),
        Some(AdmissionError::TokenOutsideVocabulary {
            id: 9,
            index: 1,
            vocabulary: VOCABULARY
        })
    );
    assert_eq!(
        rig.refusal(request(&rig, &[1; 5], 1, &[])),
        Some(AdmissionError::PromptTooLong { tokens: 5, capacity: 4 })
    );
    let mut drafted = request(&rig, &[1], 1, &[]);
    drafted.speculation = Speculation::Artifact;
    assert_eq!(rig.refusal(drafted), Some(AdmissionError::SpeculationUnavailable));
    rig.start(request(&rig, &[1], 1, &[]));
    assert_eq!(rig.refusal(request(&rig, &[1], 1, &[])), Some(AdmissionError::Busy));
    assert_eq!(rig.begun().len(), 1);
}

#[test]
fn a_shutdown_ends_the_request_then_sends_the_marker_after_the_outcome() {
    let mut rig = rig(8, 64);
    rig.start(request(&rig, &[1], 5, &[]));
    let idle = Handoff::new();
    rig.owner.shut(Arc::clone(&idle));
    // The quantum in flight has not retired: the owner is not idle yet.
    assert!(!rig.owner.idle());
    assert!(rig.owner.take_finished().is_empty());
    rig.retire(&[]);
    let finished = rig.owner.take_finished();
    assert!(matches!(finished.first(), Some(Finished::Outcome(..))));
    assert!(matches!(finished.last(), Some(Finished::Idle(_))));
    assert!(rig.owner.idle());
}

#[test]
fn an_idle_owner_answers_a_shutdown_at_once() {
    let mut rig = rig(8, 64);
    rig.owner.shut(Handoff::new());
    let finished = rig.owner.take_finished();
    assert_eq!(finished.len(), 1);
    assert!(FlashOwner::<Fake>::into_barrier(finished.into_iter().next().unwrap()).is_ok());
}

#[test]
fn an_abandoned_turn_fails_the_request_and_frees_the_owner() {
    let mut rig = rig(8, 64);
    rig.start(request(&rig, &[1], 5, &[]));
    for finished in rig.owner.abandon("the turn panicked") {
        FlashOwner::<Fake>::deliver(finished);
    }
    assert_eq!(rig.device.lock().unwrap().discarded, 1);
    assert!(rig.log.lock().unwrap().last().unwrap().starts_with("finished Failed"));
    assert!(rig.owner.idle());
}
