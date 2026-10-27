//! The runner: prepares each scenario's prompts once, then runs its warmup iterations and its measured iterations on a
//! started engine, one request at a time, and turns each into a row.
//!
//! The runner is called by the binary's main thread, which starts a request and waits for its latch; nothing else
//! waits. The measurement is taken by the [`Probe`], the request's listener: it stamps the engine's events on the
//! worker that finds them and hands the whole [`Recorder`] to the main thread when the request ends, so no clock is
//! read, and nothing is allocated for the record, in the timed path of the engine beyond the events themselves.

use std::collections::HashMap;
use std::env::consts;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use euhedral_graph::{Latch, LatchSetter};
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation, Verification,
};
use euhedral_inference_engine::stages::QuantumKind;
use euhedral_inference_tokenizer::Tokenizer;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::config::{ConfigError, Options, Scenario, SpeculationMode};
use crate::metrics::{self, Ending, Phase, QuantumRecord, Recorder, Step};
use crate::prompt::Prompt;
use crate::result::{
    self, Fork, GpuMemory, IMPLEMENTATION, Provenance, Row, SCHEMA, SCHEMA_VERSION, ScenarioRecord, TOOL, iso_instant,
};

/// The prompts of every scenario, built once per distinct target length.
#[derive(Debug)]
pub struct Prepared {
    pub prompts: Vec<(Scenario, Arc<Vec<Prompt>>)>,
}

/// Builds the prompts of `options`' scenarios, one set per target (`PromptCorpus::prompts` builds the whole set, so a
/// target shared by scenarios is built once), and checks that each fits the context.
pub fn prepare(options: &Options, tokenizer: &Tokenizer) -> Result<Prepared, ConfigError> {
    let mut by_target: HashMap<usize, Arc<Vec<Prompt>>> = HashMap::new();
    let mut prompts = Vec::new();
    for scenario in &options.scenarios {
        let target = scenario.target_prompt_tokens;
        let set = match by_target.get(&target) {
            Some(set) => Arc::clone(set),
            None => {
                let built = options
                    .prompt_corpus
                    .prompts(target, options.prompt_seed as u64, tokenizer)
                    .map_err(|error| ConfigError::Invalid(format!("{}: {error}", scenario.name())))?;
                let set = Arc::new(built);
                by_target.insert(target, Arc::clone(&set));
                set
            }
        };
        for prompt in set.iter() {
            let required = prompt.ids.len() + scenario.requested_new_tokens;
            if required > options.max_context_tokens as usize {
                return Err(ConfigError::Invalid(format!(
                    "{} needs {required} positions; context is {} (maxContextTokens)",
                    scenario.name(),
                    options.max_context_tokens
                )));
            }
        }
        prompts.push((*scenario, set));
    }
    Ok(Prepared { prompts })
}

/// What every row of one run shares.
#[derive(Clone, Debug)]
pub struct Context {
    pub run_id: String,
    pub fork: Fork,
    pub provenance: Provenance,
}

/// A run id: 32 hex digits formed like a UUID, from the clock and the process id.
pub fn run_id(started: SystemTime, pid: u32) -> String {
    let nanos = started
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    let digest = Sha256::digest(format!("{nanos}:{pid}").as_bytes());
    let hex = format!("{digest:x}");
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// Hands the events of one request to a [`Recorder`] and the recorder to the waiting main thread.
struct Probe {
    recorder: Option<Recorder>,
    done: Option<LatchSetter<Recorder>>,
    drafting: Drafting,
}

impl Probe {
    fn new(recorder: Recorder, done: Option<LatchSetter<Recorder>>) -> Probe {
        Probe {
            recorder: Some(recorder),
            done,
            drafting: Drafting::default(),
        }
    }
}

/// Where the speculative step that the next draft quantum belongs to stands.
#[derive(Debug, Default)]
struct Drafting {
    /// The step's catch-up has retired, so a draft quantum that follows is a recursive draft.
    caught_up: bool,
    /// The step's recursive drafts were counted as a phase.
    recursing: bool,
    /// When the step's last draft quantum retired.
    last_retired: Option<Instant>,
}

/// The drafting phases, named as the Java decoder names them.
const PROMPT_CATCH_UP: &str = "prompt-catch-up";
const CATCH_UP: &str = "catch-up";
const RECURSION: &str = "recursion";

/// The kind of a quantum for the benchmark. Every field access of a [`QuantumEvent`] is in this function and
/// [`draft_phase_of`], and every field access of a [`Verification`] in [`step_of`], so a change of the events is a
/// change here.
fn phase_of(event: &QuantumEvent) -> Phase {
    match event.kind {
        QuantumKind::Prefill => Phase::Prefill,
        QuantumKind::Decode => Phase::Decode,
        QuantumKind::Verify | QuantumKind::Draft => Phase::Other,
    }
}

/// The drafting phase of a draft quantum and the time it adds to that phase, or `None` for a quantum that does not
/// draft. A prompt catch-up's rows end before the prompt's last token (its tokens are the prompt's next ones); the
/// first draft quantum of a step after that is the step's catch-up, the ones after it its recursive drafts. As in the
/// Java decoder, each phase counts once per step: a catch-up from its admission to its retirement, the recursive
/// drafts from the catch-up's retirement to the last draft's.
fn draft_phase_of(
    event: &QuantumEvent,
    prompt_tokens: usize,
    drafting: &mut Drafting,
) -> Option<(&'static str, u64, Duration)> {
    if event.kind != QuantumKind::Draft {
        return None;
    }
    let time = event.retired.saturating_duration_since(event.admitted);
    if event.start_position + (event.rows as u64) < prompt_tokens as u64 {
        return Some((PROMPT_CATCH_UP, 1, time));
    }
    let previous = drafting.last_retired.replace(event.retired);
    if !drafting.caught_up {
        drafting.caught_up = true;
        return Some((CATCH_UP, 1, time));
    }
    let count = u64::from(!drafting.recursing);
    drafting.recursing = true;
    let since = previous.map_or(time, |previous| event.retired.saturating_duration_since(previous));
    Some((RECURSION, count, since))
}

/// The speculative step a verification reports, whose last token was found at `executed`. The engine reports no
/// shortlist rejection, so the step carries none (-1); the verification starts `verify_nanos` before `executed`, after
/// the step's drafts.
fn step_of(verification: &Verification, executed: Instant) -> Step {
    let start = executed
        .checked_sub(Duration::from_nanos(verification.verify_nanos))
        .unwrap_or(executed);
    Step {
        start,
        executed,
        outputs: verification.committed,
        accepted: verification.accepted,
        rejection: -1,
    }
}

impl GenerationListener for Probe {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, at: Instant) {
        if let Some(recorder) = &mut self.recorder {
            recorder.tokens.push((id, at));
        }
    }

    fn quantum(&mut self, event: QuantumEvent) {
        if let Some(recorder) = &mut self.recorder {
            recorder.quanta.push(QuantumRecord {
                phase: phase_of(&event),
                rows: event.rows,
                admitted: event.admitted,
                retired: event.retired,
            });
            let prompt_tokens = recorder.encoded.map_or(0, |(_, count)| count);
            if let Some((phase, count, time)) = draft_phase_of(&event, prompt_tokens, &mut self.drafting) {
                let entry = recorder.draft_phases.entry(phase.to_string()).or_default();
                entry.0 += count;
                entry.1 += time;
            }
        }
    }

    fn verification(&mut self, event: Verification) {
        if let Some(recorder) = &mut self.recorder {
            // The engine reports the step's tokens, then the verification: its last token is the step's end.
            let executed = recorder.tokens.last().map_or_else(Instant::now, |(_, at)| *at);
            recorder.steps.push(step_of(&event, executed));
        }
        self.drafting = Drafting::default();
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        let at = Instant::now();
        let (Some(mut recorder), Some(done)) = (self.recorder.take(), self.done.take()) else {
            return;
        };
        let ending = match outcome.reason {
            FinishReason::Failed(reason) => Ending::Failed(reason),
            FinishReason::Length | FinishReason::Stop | FinishReason::ContextFull => Ending::Completed,
        };
        recorder.finished = Some((at, ending));
        done.set(recorder);
    }
}

/// The description of this engine and run a row's `engine` field carries.
pub fn snapshot(options: &Options, runtime: &EngineRuntime) -> Value {
    let config = runtime.config();
    let plan = runtime.plan();
    let artifact = options.artifact.file_name().map_or_else(
        || options.artifact.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    );
    json!({
        "schemaVersion": 1,
        "configuration": {
            "maxContextTokens": options.max_context_tokens,
            "planContextTokens": plan.max_context_tokens,
            "artifact": artifact,
            "speculation": options.speculation.label(),
            "prefixCacheBytes": 0,
            "kvDeviceTokens": plan.kv_device_tokens,
            "kvHostTokens": plan.kv_host_tokens,
        },
        "model": {
            "artifactPath": options.artifact.display().to_string(),
            "dimensions": {
                "vocabSize": config.vocab_size,
                "hiddenSize": config.hidden_size,
                "numHiddenLayers": config.num_hidden_layers,
                "numAttentionHeads": config.num_attention_heads,
                "numKeyValueHeads": config.num_key_value_heads,
                "attentionHeadDim": config.attention_head_dim,
                "intermediateSize": config.intermediate_size,
                "maxPositionEmbeddings": config.max_position_embeddings,
            },
        },
        "generation": {"mode": "greedy", "seed": options.generation_seed},
        "runtime": {
            "implementation": IMPLEMENTATION,
            "engineVersion": env!("CARGO_PKG_VERSION"),
            "osName": consts::OS,
            "osArch": consts::ARCH,
            "nativeLibraryPath": runtime.library().path().display().to_string(),
            "device": runtime.library().device().to_string(),
            "startupSeconds": runtime.report().ready.as_secs_f64(),
        },
    })
}

/// Everything one iteration needs.
struct Iteration<'a> {
    options: &'a Options,
    runtime: &'a Arc<EngineRuntime>,
    tokenizer: &'a Tokenizer,
    context: &'a Context,
    engine: &'a Value,
}

/// Runs every scenario of `options` on `runtime`: its warmup iterations, then its measured ones, each handed to
/// `sink` as it finishes. A `sink` error stops the run.
pub fn run(
    options: &Options,
    prepared: &Prepared,
    runtime: &Arc<EngineRuntime>,
    tokenizer: &Tokenizer,
    context: &Context,
    sink: &mut dyn FnMut(Row) -> Result<(), ConfigError>,
) -> Result<(), ConfigError> {
    let engine = snapshot(options, runtime);
    let iteration = Iteration {
        options,
        runtime,
        tokenizer,
        context,
        engine: &engine,
    };
    for (scenario, prompts) in &prepared.prompts {
        for index in 0..options.warmup {
            sink(iteration.run(scenario, &prompts[index % prompts.len()], true, index))?;
        }
        for index in 0..options.iterations {
            sink(iteration.run(scenario, &prompts[index % prompts.len()], false, index))?;
        }
    }
    Ok(())
}

impl Iteration<'_> {
    fn gpu_free(&self) -> Option<u64> {
        self.options
            .gpu_memory
            .then(|| self.runtime.library().free_bytes().ok())
            .flatten()
    }

    /// One generation of `prompt` in a fresh sequence, measured.
    fn run(&self, scenario: &Scenario, prompt: &Prompt, warmup: bool, index: usize) -> Row {
        let before = self.gpu_free();
        let (done, waiting) = Latch::new();
        let requested = scenario.requested_new_tokens;
        let entry = Instant::now();
        let mut recorder = Recorder::new(entry);
        recorder.tokens.reserve(requested + 1);
        recorder.quanta.reserve(requested + prompt.ids.len() / 256 + 4);

        // Tokenization is part of the timed generation, as it is in the Java session.
        let ids = match self.tokenizer.encode_with_model_special_tokens(&prompt.text) {
            Ok(ids) => ids,
            Err(error) => {
                recorder.finished = Some((Instant::now(), Ending::Failed(format!("TokenizerError: {error}"))));
                return self.row(scenario, prompt, warmup, index, &recorder, before);
            }
        };
        let encoded = (Instant::now(), ids.len());
        recorder.encoded = Some(encoded);
        let request = GenerationRequest {
            prompt: ids,
            max_tokens: requested as u32,
            stop_tokens: self.tokenizer.eos_ids().to_vec(),
            speculation: match self.options.speculation {
                SpeculationMode::Artifact => Speculation::Artifact,
                SpeculationMode::Off => Speculation::Off,
            },
            listener: Box::new(Probe::new(recorder, Some(done))),
        };
        let recorder = match self.runtime.generate(request) {
            Ok(()) => waiting.wait(),
            Err(error) => {
                // The refused request took the probe and its recorder with it.
                let mut refused = Recorder::new(entry);
                refused.encoded = Some(encoded);
                refused.finished = Some((Instant::now(), Ending::Failed(format!("AdmissionError: {error}"))));
                refused
            }
        };
        self.row(scenario, prompt, warmup, index, &recorder, before)
    }

    fn row(
        &self,
        scenario: &Scenario,
        prompt: &Prompt,
        warmup: bool,
        index: usize,
        recorder: &Recorder,
        before: Option<u64>,
    ) -> Row {
        let eos = self.tokenizer.eos_ids();
        let mut outcome = metrics::evaluate(scenario, recorder, |id| eos.contains(&id));
        if outcome.status != result::FAILED && outcome.work.prompt_tokens != Some(prompt.ids.len()) {
            outcome.status = result::FAILED;
            outcome.reason = Some("prompt_token_count_mismatch".to_string());
            // As Java keeps the timings of this failure; a failed row reports no rate.
            outcome.throughput = None;
        }
        let gpu_memory = before.map(|before| {
            let report = self.runtime.report();
            GpuMemory {
                before_free_bytes: Some(before),
                after_free_bytes: self.runtime.library().free_bytes().ok(),
                total_bytes: Some(self.runtime.library().device().total_bytes),
                allocated_before_bytes: Some(report.allocated_bytes),
                peak_allocated_bytes: None,
                allocated_after_bytes: Some(report.allocated_bytes),
                retained_workspace_bytes: None,
            }
        });
        Row {
            schema: SCHEMA.to_string(),
            schema_version: SCHEMA_VERSION,
            implementation: IMPLEMENTATION.to_string(),
            provenance: self.context.provenance.clone(),
            run_id: self.context.run_id.clone(),
            fork: self.context.fork.clone(),
            recorded_at: iso_instant(SystemTime::now()),
            scenario: ScenarioRecord {
                name: scenario.name(),
                kind: scenario.kind.label().to_string(),
                target_prompt_tokens: scenario.target_prompt_tokens,
                requested_new_tokens: scenario.requested_new_tokens,
                prompt_generator: self.options.prompt_corpus.generator().to_string(),
                prompt_seed: self.options.prompt_seed,
                prompt_sha256: prompt.sha256.clone(),
            },
            warmup,
            iteration: index,
            status: outcome.status.to_string(),
            status_reason: outcome.reason,
            work: outcome.work,
            timings: outcome.timings,
            throughput: outcome.throughput,
            engine: self.engine.clone(),
            gpu_memory,
        }
    }
}

/// The provenance of a measured row: the tool, and the command line that started it.
pub fn provenance(command_line: String) -> Provenance {
    Provenance {
        kind: "measured".to_string(),
        tool: TOOL.to_string(),
        command_line,
        source: None,
        source_sha256: None,
        note: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ScenarioKind;

    const PROMPT: usize = 10;

    fn event(kind: QuantumKind, rows: usize, start_position: u64, admitted: Instant, retired: Instant) -> QuantumEvent {
        QuantumEvent {
            kind,
            rows,
            start_position,
            admitted,
            retired,
            replayed: false,
            recorded: false,
        }
    }

    /// A probe fed the way the engine feeds a speculative request of a 10-token prompt: two prefill chunks each followed
    /// by its MTP catch-up (the last one retires after the first token), then steps of a catch-up, recursive drafts and
    /// a verification whose tokens come before its report.
    fn speculative_probe() -> (Probe, Instant) {
        let entry = Instant::now();
        let ms = |n: u64| entry + Duration::from_millis(n);
        let mut recorder = Recorder::new(entry);
        recorder.encoded = Some((entry, PROMPT));
        let mut probe = Probe::new(recorder, None);

        probe.quantum(event(QuantumKind::Prefill, 6, 0, ms(1), ms(2)));
        probe.quantum(event(QuantumKind::Draft, 5, 0, ms(1), ms(3)));
        probe.quantum(event(QuantumKind::Prefill, 4, 6, ms(2), ms(4)));
        probe.token(100, ms(4));
        probe.quantum(event(QuantumKind::Draft, 4, 5, ms(3), ms(5)));

        // Step 1 at position 10: catch-up of the first token, 3 recursive drafts, all 3 drafts accepted, 4 tokens.
        probe.quantum(event(QuantumKind::Draft, 1, 9, ms(5), ms(6)));
        probe.quantum(event(QuantumKind::Draft, 1, 10, ms(5), ms(7)));
        probe.quantum(event(QuantumKind::Draft, 1, 11, ms(5), ms(8)));
        probe.quantum(event(QuantumKind::Draft, 1, 12, ms(5), ms(9)));
        probe.quantum(event(QuantumKind::Verify, 4, 10, ms(5), ms(12)));
        for (index, id) in [101, 102, 103, 104].into_iter().enumerate() {
            probe.token(id, ms(13) + Duration::from_micros(index as u64));
        }
        probe.verification(Verification {
            accepted: 3,
            drafted: 3,
            committed: 4,
            verify_nanos: 3_000_000,
            draft_nanos: 4_000_000,
        });

        // Step 2 at position 14: catch-up of 4 rows, one recursive draft, no draft accepted, 1 token.
        probe.quantum(event(QuantumKind::Draft, 4, 10, ms(14), ms(16)));
        probe.quantum(event(QuantumKind::Draft, 1, 14, ms(14), ms(17)));
        probe.quantum(event(QuantumKind::Verify, 3, 14, ms(14), ms(20)));
        probe.token(105, ms(21));
        probe.verification(Verification {
            accepted: 0,
            drafted: 2,
            committed: 1,
            verify_nanos: 3_000_000,
            draft_nanos: 3_000_000,
        });
        (probe, entry)
    }

    #[test]
    fn verifications_become_steps_ending_at_their_last_token() {
        let (probe, entry) = speculative_probe();
        let recorder = probe.recorder.unwrap();
        let ms = |n: u64| entry + Duration::from_millis(n);
        let steps: Vec<(Instant, Instant, usize, usize, i32)> = recorder
            .steps
            .iter()
            .map(|step| (step.start, step.executed, step.outputs, step.accepted, step.rejection))
            .collect();
        let last_of_first = ms(13) + Duration::from_micros(3);
        assert_eq!(
            steps,
            vec![
                (last_of_first - Duration::from_millis(3), last_of_first, 4, 3, -1),
                (ms(18), ms(21), 1, 0, -1),
            ]
        );
    }

    #[test]
    fn draft_quanta_are_counted_by_phase_once_per_step() {
        let (probe, _) = speculative_probe();
        let recorder = probe.recorder.unwrap();
        let phases: Vec<(String, (u64, Duration))> = recorder.draft_phases.into_iter().collect();
        let ms = Duration::from_millis;
        assert_eq!(
            phases,
            vec![
                // Step 1: the catch-up 1 ms, recursion from its retirement (6 ms) to the last draft's (9 ms); step 2:
                // the catch-up 2 ms, recursion 1 ms.
                (CATCH_UP.to_string(), (2, ms(3))),
                // Both prompt chunks' catch-ups, the second one after the first token.
                (PROMPT_CATCH_UP.to_string(), (2, ms(4))),
                (RECURSION.to_string(), (2, ms(4))),
            ]
        );
    }

    #[test]
    fn a_speculative_request_fills_the_row_block_in_the_java_schema() {
        let (probe, entry) = speculative_probe();
        let mut recorder = probe.recorder.unwrap();
        recorder.finished = Some((entry + Duration::from_millis(22), Ending::Completed));
        let scenario = Scenario::new(ScenarioKind::PromptToN, PROMPT, 6).unwrap();
        let outcome = metrics::evaluate(&scenario, &recorder, |_| false);
        assert_eq!(outcome.status, result::SUCCESS);
        let speculative = outcome.work.speculative.unwrap();
        assert_eq!(speculative.verifications, 2);
        assert_eq!(speculative.accepted_drafts, vec![1, 0, 0, 1]);
        assert_eq!(speculative.committed_tokens, 5);
        assert_eq!(speculative.verify_nanos, 6_000_000);
        assert_eq!(
            (
                speculative.rejections_in_shortlist,
                speculative.rejections_outside_shortlist
            ),
            (0, 0)
        );
        assert_eq!(speculative.draft_phases["catch-up"].quanta, 2);
        assert_eq!(speculative.draft_phases["recursion"].quanta, 2);
        assert_eq!(speculative.draft_phases["prompt-catch-up"].quanta, 2);
        assert_eq!(outcome.work.generated_tokens, Some(6));
        assert_eq!(outcome.work.decode_sampled_tokens, Some(5));
        // The window runs from the first token (4 ms) to the last step's last token (21 ms).
        assert_eq!(outcome.timings.unwrap().decode, Some(17_000_000));
        assert_eq!(outcome.timings.unwrap().time_to_last_token, Some(21_000_000));
    }

    #[test]
    fn a_greedy_request_has_no_steps_or_draft_phases() {
        let entry = Instant::now();
        let mut recorder = Recorder::new(entry);
        recorder.encoded = Some((entry, PROMPT));
        let mut probe = Probe::new(recorder, None);
        let at = entry + Duration::from_millis(1);
        probe.quantum(event(QuantumKind::Prefill, PROMPT, 0, entry, at));
        probe.token(1, at);
        probe.quantum(event(
            QuantumKind::Decode,
            1,
            PROMPT as u64,
            at,
            at + Duration::from_millis(1),
        ));
        let recorder = probe.recorder.unwrap();
        assert!(recorder.steps.is_empty());
        assert!(recorder.draft_phases.is_empty());
    }
}
