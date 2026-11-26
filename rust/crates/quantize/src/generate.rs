//! The `generate` command: the artifact answers chat prompts, and the answers are written as the drafter's training
//! corpus: tokens, the base model's greedy choice at each position, and the hidden of every row the model ran.

use std::collections::HashSet;
use std::env::consts;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_quantization::corpus::{self, Writer};
use euhedral_inference_quantization::generate::run::{Answerer, Options, start};
use euhedral_inference_quantization::generate::{PROMPT_TOKENS, Prompt, prompt};
use euhedral_inference_quantization::reference::read_tokenizer;
use euhedral_inference_settings::{Auto, ByteSize, ModelSettings, Settings, load_file, process_env};

use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// The tokens of an answer when `--max-tokens` is not given.
const DEFAULT_MAX_TOKENS: usize = 4096;
/// The prompts answered at once when `--batch` is not given.
const DEFAULT_BATCH: usize = 12;
/// The most prompts answered at once.
const MOST_BATCH: usize = 32;
/// The share of prompts answered greedily when `--greedy-fraction` is not given.
const DEFAULT_GREEDY_FRACTION: f64 = 0.25;
/// The longest a run takes: a bound on the wait, not a rate.
const LONGEST_RUN: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--artifact, --tokenizer (a checkpoint directory or an artifact), --out, --prompts (repeatable), \
--max-tokens, --batch, --greedy-fraction, --limit, --host-mib, --workers, --config, --library (or --cuda-library)";

#[derive(Debug)]
struct Flags {
    artifact: PathBuf,
    tokenizer: PathBuf,
    out: PathBuf,
    prompts: Vec<PathBuf>,
    /// The tokens of an answer the model runs; the answer holds one more.
    max_tokens: usize,
    /// Prompts answered at once.
    batch: usize,
    greedy_fraction: f64,
    /// The most prompts to answer in this run, over all the files; 0 is no limit.
    limit: usize,
    config: Option<PathBuf>,
    library: Option<PathBuf>,
    /// Host memory for host-backed weights, in MiB; unset leaves the engine to size it.
    host_mib: Option<u64>,
    /// Workers taking host work at once; unset keeps the engine's default.
    workers: Option<u32>,
}

fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut artifact, mut tokenizer, mut out, mut prompts) = (None, None, None, Vec::new());
    let (mut max_tokens, mut batch, mut greedy_fraction, mut limit) =
        (DEFAULT_MAX_TOKENS, DEFAULT_BATCH, DEFAULT_GREEDY_FRACTION, 0);
    let (mut config, mut library, mut host_mib, mut workers) = (None, None, None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("{arg} needs a whole number"))
        };
        match arg.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--prompts" => prompts.push(PathBuf::from(value)),
            "--max-tokens" => max_tokens = number()?,
            "--batch" => batch = number()?,
            "--greedy-fraction" => {
                greedy_fraction = value.parse().map_err(|_| format!("{arg} needs a number"))?;
            }
            "--limit" => limit = number()?,
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" | "--library" => library = Some(PathBuf::from(value)),
            "--host-mib" => host_mib = Some(number()?.try_into().map_err(|_| format!("{arg} is too large"))?),
            "--workers" => workers = Some(number()?.try_into().map_err(|_| format!("{arg} is too large"))?),
            other => return Err(format!("unknown generate option {other}; the options are {OPTIONS}")),
        }
    }
    let (Some(artifact), Some(out)) = (artifact, out) else {
        return Err("generate needs --artifact, --out and at least one --prompts".to_owned());
    };
    if prompts.is_empty() {
        return Err("generate needs --artifact, --out and at least one --prompts".to_owned());
    }
    if !(1..=MOST_BATCH).contains(&batch) {
        return Err(format!("--batch is 1 to {MOST_BATCH} sequences"));
    }
    if max_tokens < 1 {
        return Err("--max-tokens needs a positive number".to_owned());
    }
    if !(0.0..=1.0).contains(&greedy_fraction) {
        return Err("--greedy-fraction is a share, 0 to 1".to_owned());
    }
    if host_mib == Some(0) || workers == Some(0) {
        return Err("--host-mib and --workers need a positive number".to_owned());
    }
    Ok(Flags {
        tokenizer: tokenizer.unwrap_or_else(|| artifact.clone()),
        artifact,
        out,
        prompts,
        max_tokens,
        batch,
        greedy_fraction,
        limit,
        config,
        library,
        host_mib,
        workers,
    })
}

/// The library the build puts the CUDA layer in, where this platform has one; other platforms name it.
fn default_library() -> Option<PathBuf> {
    (consts::OS == "linux" && consts::ARCH == "x86_64")
        .then(|| PathBuf::from("build/native/linux-x64/lib/libeuhedral_cuda.so"))
}

/// The engine's settings for a run: the file `config` names, or the defaults, with the artifact, a context that holds
/// the longest prompt and answer, a sequence for each prompt in flight and no prefix cache, which prompts that start
/// fresh do not use.
fn settings(flags: &Flags) -> Result<Settings, String> {
    let model = ModelSettings {
        artifact: flags.artifact.clone(),
        name: flags
            .artifact
            .file_stem()
            .map_or_else(|| "model".to_owned(), |stem| stem.to_string_lossy().into_owned()),
    };
    let mut settings = match &flags.config {
        None => Settings::with_model(model.clone()),
        Some(path) => load_file(path, &process_env).map_err(|errors| format!("{}: {errors}", path.display()))?,
    };
    settings.model = model;
    // The prompt, the answer's run tokens and the one it chose and never ran.
    let context = PROMPT_TOKENS + flags.max_tokens + 1;
    settings.context.max_tokens = u32::try_from(context.next_multiple_of(512))
        .map_err(|_| format!("--max-tokens {} is too large", flags.max_tokens))?;
    settings.prefix_cache.size = ByteSize::mib(0);
    settings.batching.max_sequences = u32::try_from(flags.batch).map_err(|_| "--batch is too large".to_owned())?;
    if let Some(mib) = flags.host_mib {
        settings.memory.host_weights_limit = Auto::Value(ByteSize::mib(mib));
    }
    if let Some(workers) = flags.workers {
        settings.host.task_parallelism = workers;
    }
    Ok(settings)
}

/// Adds the prompts of one file to `chosen`, in order: those whose id is not in `seen` (the corpus's and the earlier
/// ones), up to `limit` in all (0: no limit). Refuses a prompt that is too long for the context. Returns whether
/// `chosen` is full.
fn choose(
    chosen: &mut Vec<Prompt>,
    seen: &mut HashSet<String>,
    prompts: Vec<Prompt>,
    limit: usize,
) -> Result<bool, String> {
    for prompt in prompts {
        if limit > 0 && chosen.len() >= limit {
            return Ok(true);
        }
        if seen.contains(&prompt.id) {
            continue;
        }
        if prompt.tokens.len() > PROMPT_TOKENS {
            return Err(format!(
                "prompt {} has {} tokens; the most is {PROMPT_TOKENS}",
                prompt.id,
                prompt.tokens.len()
            ));
        }
        seen.insert(prompt.id.clone());
        chosen.push(prompt);
    }
    Ok(limit > 0 && chosen.len() >= limit)
}

fn failed(message: String) -> Outcome {
    Outcome {
        code: EXIT_RUN_FAILED,
        message: format!("{message}\n"),
    }
}

/// Runs `generate` with the flags in `args`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads the prompts and waits for the run"
)]
pub fn run(args: &[String]) -> Outcome {
    let flags = match flags(args) {
        Ok(flags) => flags,
        Err(message) => {
            return Outcome {
                code: EXIT_USAGE,
                message: format!("{message}\n"),
            };
        }
    };
    let Some(library) = flags.library.clone().or_else(default_library) else {
        return Outcome {
            code: EXIT_USAGE,
            message: "generate needs --cuda-library on this platform\n".to_owned(),
        };
    };
    let settings = match settings(&flags) {
        Ok(settings) => settings,
        Err(message) => return failed(message),
    };
    let tokenizer = match read_tokenizer(&flags.tokenizer) {
        Ok(tokenizer) => tokenizer,
        Err(message) => return failed(format!("{}: {message}", flags.tokenizer.display())),
    };
    let mut seen: HashSet<String> = match corpus::entries(&flags.out) {
        Ok(entries) => entries.into_iter().map(|entry| entry.id).collect(),
        Err(error) => return failed(format!("{}: {error}", flags.out.display())),
    };
    let done = seen.len();
    let mut prompts = Vec::new();
    for file in &flags.prompts {
        let source = file.display().to_string();
        let read = fs::read_to_string(file)
            .map_err(|error| format!("{source}: {error}"))
            .and_then(|text| prompt::read(&text, &source, &tokenizer));
        match read.and_then(|read| choose(&mut prompts, &mut seen, read, flags.limit)) {
            Ok(false) => {}
            Ok(true) => break,
            Err(message) => return failed(message),
        }
    }
    let mut message = format!("{} prompts to answer, {done} done\n", prompts.len());
    if prompts.is_empty() {
        return Outcome { code: EXIT_OK, message };
    }
    let stops = tokenizer.eos_ids().to_vec();
    drop(tokenizer);
    let (owner, waiting) = EngineRuntime::start(&settings, &library);
    let runtime = match waiting.wait(LONGEST_RUN) {
        Ok(Ok(runtime)) => runtime,
        Ok(Err(error)) => return failed(format!("the engine did not load: {error}")),
        Err(_) => return failed("the engine did not finish loading".to_owned()),
    };
    let (hidden, vocabulary) = (runtime.config().hidden_size, runtime.config().vocab_size);
    let result = match u32::try_from(flags.max_tokens) {
        Ok(max_tokens) => match Writer::open(&flags.out, hidden) {
            Ok(writer) => {
                let answers: Arc<dyn Answerer> = Arc::clone(&runtime) as Arc<dyn Answerer>;
                let ended = start(
                    runtime.host(),
                    answers,
                    writer,
                    prompts,
                    Options {
                        max_tokens,
                        greedy_fraction: flags.greedy_fraction,
                        batch: flags.batch,
                        sampling: settings.sampling,
                        stops,
                        vocabulary,
                    },
                );
                ended
                    .wait(LONGEST_RUN)
                    .map_err(|_| "the run did not finish in time".to_owned())
                    .and_then(|ended| ended.map_err(|error| error.to_string()))
            }
            Err(error) => Err(format!("{}: {error}", flags.out.display())),
        },
        Err(_) => Err("--max-tokens does not fit".to_owned()),
    };
    let shutdown = owner
        .shutdown(runtime)
        .err()
        .map_or_else(String::new, |error| format!("shutdown: {error}\n"));
    let mut outcome = match result {
        Ok(summary) => {
            message.push_str(&format!(
                "answered {} prompts ({} answer tokens) into {}\n",
                summary.answered,
                summary.tokens,
                flags.out.display()
            ));
            Outcome { code: EXIT_OK, message }
        }
        Err(why) => failed(why),
    };
    outcome.message.push_str(&shutdown);
    outcome
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    fn prompt(id: &str, tokens: usize) -> Prompt {
        Prompt {
            id: id.to_owned(),
            tokens: vec![1; tokens],
        }
    }

    #[test]
    fn the_java_defaults_and_every_option_reach_their_setting() {
        let parsed = flags(&words("--artifact a.edrl --tokenizer /ckpt --out o --prompts p.jsonl")).unwrap();
        assert_eq!((parsed.max_tokens, parsed.batch, parsed.limit), (4096, 12, 0));
        assert_eq!(parsed.greedy_fraction, 0.25);
        let set = settings(&parsed).unwrap();
        assert_eq!(set.context.max_tokens, 8704);
        assert_eq!(set.batching.max_sequences, 12);
        assert_eq!(set.prefix_cache.size, ByteSize::mib(0));
        let parsed = flags(&words(
            "--artifact a --tokenizer /ckpt --out o --prompts p --prompts q --max-tokens 32 --batch 2 \
             --greedy-fraction 0.5 --limit 3 --host-mib 2048 --workers 6 --cuda-library libx.so",
        ))
        .unwrap();
        assert_eq!(parsed.prompts, [PathBuf::from("p"), PathBuf::from("q")]);
        assert_eq!((parsed.max_tokens, parsed.batch, parsed.limit), (32, 2, 3));
        assert_eq!(parsed.library, Some(PathBuf::from("libx.so")));
        let set = settings(&parsed).unwrap();
        assert_eq!(set.context.max_tokens, 4608);
        assert_eq!(set.batching.max_sequences, 2);
        assert_eq!(set.memory.host_weights_limit, Auto::Value(ByteSize::mib(2048)));
        assert_eq!(set.host.task_parallelism, 6);
    }

    #[test]
    fn a_batch_outside_one_to_thirty_two_is_refused_with_the_java_message() {
        for bad in ["0", "33"] {
            let message = flags(&words(&format!("--artifact a --out o --prompts p --batch {bad}")))
                .err()
                .unwrap();
            assert_eq!(message, "--batch is 1 to 32 sequences");
        }
        assert!(flags(&words("--artifact a --out o --prompts p --batch 32")).is_ok());
    }

    #[test]
    fn missing_and_malformed_flags_are_usage_errors_and_nothing_is_started() {
        for bad in [
            "--artifact a --out o",
            "--artifact a --prompts p",
            "--out o --prompts p",
            "--artifact a --out o --prompts p --max-tokens 0",
            "--artifact a --out o --prompts p --max-tokens many",
            "--artifact a --out o --prompts p --greedy-fraction 1.5",
            "--artifact a --out o --prompts p --greedy-fraction -0.1",
            "--artifact a --out o --prompts p --greedy-fraction nan",
            "--artifact a --out o --prompts p --workers 0",
            "--artifact a --out o --prompts p --limit",
            "--artifact a --out o --prompts p --frob 1",
        ] {
            assert_eq!(run(&words(bad)).code, EXIT_USAGE, "{bad}");
        }
        let message = flags(&words("--artifact a --out o --prompts p --frob 1"))
            .err()
            .unwrap();
        assert!(message.contains("--frob") && message.contains("--prompts") && message.contains("--library"));
    }

    #[test]
    fn the_limit_counts_across_files_and_skips_ids_already_answered() {
        let mut seen: HashSet<String> = ["b".to_owned()].into();
        let mut chosen = Vec::new();
        let first = vec![prompt("a", 3), prompt("b", 3), prompt("c", 3)];
        assert!(!choose(&mut chosen, &mut seen, first, 3).unwrap());
        let second = vec![prompt("c", 3), prompt("d", 3), prompt("e", 3)];
        assert!(choose(&mut chosen, &mut seen, second, 3).unwrap());
        let ids: Vec<_> = chosen.iter().map(|chosen| chosen.id.as_str()).collect();
        assert_eq!(ids, ["a", "c", "d"]);
        let mut all = Vec::new();
        let third = vec![prompt("x", 1), prompt("y", 1)];
        assert!(!choose(&mut all, &mut HashSet::new(), third, 0).unwrap());
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn a_prompt_over_the_limit_is_refused_naming_its_id() {
        let message = choose(
            &mut Vec::new(),
            &mut HashSet::new(),
            vec![prompt("long", PROMPT_TOKENS + 1)],
            0,
        )
        .err()
        .unwrap();
        assert!(message.contains("long") && message.contains("4096"), "{message}");
        assert!(
            choose(
                &mut Vec::new(),
                &mut HashSet::new(),
                vec![prompt("fits", PROMPT_TOKENS)],
                0
            )
            .is_ok()
        );
    }
}
