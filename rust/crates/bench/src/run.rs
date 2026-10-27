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
use std::time::{Instant, SystemTime};

use euhedral_graph::{Latch, LatchSetter};
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::generation::{
    FinishReason, GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
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
}

/// The kind of a quantum for the benchmark. Every field access of a [`QuantumEvent`] is in this function and
/// [`step_of`], so a change of the event is a change here.
fn phase_of(event: &QuantumEvent) -> Phase {
    match event.kind {
        QuantumKind::Prefill => Phase::Prefill,
        QuantumKind::Decode => Phase::Decode,
        QuantumKind::Verify | QuantumKind::Draft => Phase::Other,
    }
}

/// The speculative step a quantum event reports, if it reports one. The event of a verification step carries no draft
/// statistics yet (accepted drafts, committed tokens, draft time), so no step is reported and a row's `speculative`
/// block stays null; the speculative decoder's event fills it here.
fn step_of(_event: &QuantumEvent) -> Option<Step> {
    None
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
            if let Some(step) = step_of(&event) {
                recorder.steps.push(step);
            }
        }
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
            listener: Box::new(Probe {
                recorder: Some(recorder),
                done: Some(done),
            }),
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
            outcome.timings = None;
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
