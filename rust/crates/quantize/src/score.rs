//! The `score` command: the logits of an artifact against the references of a model, as a summary per reference.

use std::env::consts;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_quantization::kld::{self, Reference};
use euhedral_inference_quantization::reference::{DEFAULT_BATCH, DEFAULT_ROWS, Forcer};
use euhedral_inference_quantization::score::device::Geometry;
use euhedral_inference_quantization::score::summary::{self, FIELDS, Shape};
use euhedral_inference_quantization::score::{Scorer, Source, batch_for, score};
use euhedral_inference_settings::{Auto, ByteSize, ModelSettings, Settings, SpeculationMode, load_file, process_env};
use euhedral_io::OwnedFile;

use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// The longest a score takes: a bound on the wait, not a rate.
pub(crate) const LONGEST_RUN: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// The most forced rows in one pass.
const MOST_ROWS: usize = 8;
/// The most rows in flight at once: `--rows` times `--batch`.
const MOST_IN_FLIGHT: usize = 32;
/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--artifact, --results, --reference and --name (each as often as needed, in pairs), --values, \
--rows, --batch, --host-mib, --workers, --config, --library (or --cuda-library)";

struct Flags {
    artifact: PathBuf,
    results: PathBuf,
    references: Vec<(PathBuf, String)>,
    values: Option<PathBuf>,
    /// Forced rows in one pass.
    rows: usize,
    /// Chunks forced at once.
    batch: usize,
    config: Option<PathBuf>,
    library: Option<PathBuf>,
    /// Host memory for host-backed weights, in MiB; unset leaves the engine to size it.
    host_mib: Option<u64>,
    /// Workers taking host work at once; unset keeps the engine's default.
    workers: Option<u32>,
}

fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut artifact, mut results, mut values, mut config, mut library) = (None, None, None, None, None);
    let (mut paths, mut names) = (Vec::new(), Vec::new());
    let (mut rows, mut batch, mut host_mib, mut workers) = (DEFAULT_ROWS, DEFAULT_BATCH, None, None);
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
            "--results" => results = Some(PathBuf::from(value)),
            "--reference" => paths.push(PathBuf::from(value)),
            "--name" => names.push(value.clone()),
            "--values" => values = Some(PathBuf::from(value)),
            "--rows" => rows = number()?,
            "--batch" => batch = number()?,
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" | "--library" => library = Some(PathBuf::from(value)),
            "--host-mib" => host_mib = Some(number()?.try_into().map_err(|_| format!("{arg} is too large"))?),
            "--workers" => workers = Some(number()?.try_into().map_err(|_| format!("{arg} is too large"))?),
            other => return Err(format!("unknown score option {other}; the options are {OPTIONS}")),
        }
    }
    let (Some(artifact), Some(results)) = (artifact, results) else {
        return Err("score needs --artifact, --results and at least one --reference with its --name".to_owned());
    };
    if paths.is_empty() || paths.len() != names.len() {
        return Err("score needs as many --name as --reference, and at least one".to_owned());
    }
    if batch < 1 || !(1..=MOST_ROWS).contains(&rows) || rows * batch > MOST_IN_FLIGHT {
        return Err(format!(
            "--rows (1 to {MOST_ROWS}) times --batch (at least 1) must be at most {MOST_IN_FLIGHT}"
        ));
    }
    if host_mib == Some(0) || workers == Some(0) {
        return Err("--host-mib and --workers need a positive number".to_owned());
    }
    Ok(Flags {
        artifact,
        results,
        references: paths.into_iter().zip(names).collect(),
        values,
        rows,
        batch,
        config,
        library,
        host_mib,
        workers,
    })
}

/// The library the build puts the CUDA layer in, where this platform has one; other platforms name it.
pub(crate) fn default_library() -> Option<PathBuf> {
    (consts::OS == "linux" && consts::ARCH == "x86_64")
        .then(|| PathBuf::from("build/native/linux-x64/lib/libeuhedral_cuda.so"))
}

fn failed(message: String) -> Outcome {
    Outcome {
        code: EXIT_RUN_FAILED,
        message: format!("{message}\n"),
    }
}

/// The engine's settings for chunks of up to `context` tokens: the file `config` names, or the defaults, with the
/// artifact, a context that holds a chunk and no prefix cache, which a fresh sequence per chunk does not use.
fn settings(flags: &Flags, context: usize) -> Result<Settings, String> {
    engine_settings(
        &flags.artifact,
        flags.config.as_deref(),
        context,
        Pass {
            batch: flags.batch,
            rows: flags.rows,
        },
        Memory {
            host_mib: flags.host_mib,
            workers: flags.workers,
        },
    )
}

/// How many chunks the engine holds at once and how many rows each verifies in a pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pass {
    pub batch: usize,
    pub rows: usize,
}

/// The host's share of a run: memory for host-backed weights and the workers taking host work; unset keeps the
/// engine's defaults.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Memory {
    pub host_mib: Option<u64>,
    pub workers: Option<u32>,
}

/// The engine's settings for scoring `artifact` in chunks of up to `context` tokens.
pub(crate) fn engine_settings(
    artifact: &Path,
    config: Option<&Path>,
    context: usize,
    pass: Pass,
    memory: Memory,
) -> Result<Settings, String> {
    let model = ModelSettings {
        artifact: artifact.to_path_buf(),
        name: artifact
            .file_stem()
            .map_or_else(|| "model".to_owned(), |stem| stem.to_string_lossy().into_owned()),
    };
    let mut settings = match config {
        None => Settings::with_model(model.clone()),
        Some(path) => load_file(path, &process_env).map_err(|errors| format!("{}: {errors}", path.display()))?,
    };
    settings.model = model;
    settings.context.max_tokens = u32::try_from(context.next_multiple_of(512))
        .map_err(|_| format!("a context of {context} tokens is too long"))?;
    settings.prefix_cache.size = ByteSize::mib(0);
    // Teacher-forced rows never draft: take no draft layer or draft state.
    settings.speculation.mode = SpeculationMode::None;
    // The engine holds a sequence for each chunk in flight, each sized to verify `rows` forced tokens in a pass.
    settings.batching.max_sequences = u32::try_from(pass.batch).map_err(|_| "--batch is too large".to_owned())?;
    settings.batching.forced_rows = u32::try_from(pass.rows).map_err(|_| "--rows is too large".to_owned())?;
    if let Some(mib) = memory.host_mib {
        settings.memory.host_weights_limit = Auto::Value(ByteSize::mib(mib));
    }
    if let Some(workers) = memory.workers {
        settings.host.task_parallelism = workers;
    }
    Ok(settings)
}

/// The context and the chunks in the header of a reference file, read before the engine starts so the engine's
/// context can hold a chunk.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads a header before the engine starts"
)]
pub(crate) fn dimensions(path: &Path) -> Result<(usize, usize), String> {
    let mut header = [0u8; 20];
    File::open(path)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let (context, _, chunks) =
        Reference::parse_header(&header).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok((context, chunks))
}

/// The values of one reference as `NAME.values`: little-endian int32 chunks, steps and fields, then the float64
/// values, chunk after chunk.
fn values_file(scored: &[Vec<f64>], steps: usize) -> Vec<u8> {
    let count = |value: usize| i32::try_from(value).unwrap_or(i32::MAX).to_le_bytes();
    let mut bytes = Vec::with_capacity(12 + 8 * scored.len() * steps * FIELDS);
    for header in [scored.len(), steps, FIELDS] {
        bytes.extend_from_slice(&count(header));
    }
    for value in scored.iter().flatten() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Runs `score` with the flags in `args`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads headers, waits for the runs and writes the results"
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
            message: "score needs --cuda-library on this platform\n".to_owned(),
        };
    };
    let (mut context, mut chunks) = (0, 0);
    for (path, _) in &flags.references {
        match dimensions(path) {
            Ok((wide, count)) => (context, chunks) = (context.max(wide), chunks.max(count)),
            Err(message) => return failed(message),
        }
    }
    let settings = match settings(&flags, context) {
        Ok(settings) => settings,
        Err(message) => return failed(message),
    };
    let (owner, waiting) = EngineRuntime::start(&settings, &library);
    let runtime = match waiting.wait(LONGEST_RUN) {
        Ok(Ok(runtime)) => runtime,
        Ok(Err(error)) => return failed(format!("the engine did not load: {error}")),
        Err(_) => return failed("the engine did not finish loading".to_owned()),
    };
    let result = scored(&flags, &runtime, chunks);
    let shutdown = owner
        .shutdown(runtime)
        .err()
        .map_or_else(String::new, |error| format!("shutdown: {error}\n"));
    let mut outcome = match result {
        Ok(()) => Outcome {
            code: EXIT_OK,
            message: String::new(),
        },
        Err(message) => failed(message),
    };
    outcome.message.push_str(&shutdown);
    outcome
}

/// Scores every reference on the started engine, printing and appending each summary as its reference completes.
fn scored(flags: &Flags, runtime: &Arc<EngineRuntime>, most_chunks: usize) -> Result<(), String> {
    let vocabulary = runtime.config().vocab_size;
    let batch = batch_for(Some(flags.batch), runtime.sequences(), most_chunks)?;
    let io = runtime.io();
    let mut references = Vec::new();
    for (path, name) in &flags.references {
        let reference = kld::Reference::read(io, path)
            .wait(LONGEST_RUN)
            .map_err(|_| format!("{}: reading the header timed out", path.display()))?
            .map_err(|error| format!("{}: {error}", path.display()))?;
        if reference.vocabulary != vocabulary {
            return Err(format!(
                "{}: a vocabulary of {}, not the model's {vocabulary}",
                path.display(),
                reference.vocabulary
            ));
        }
        let records = kld::open_records(io, path)
            .wait(LONGEST_RUN)
            .map_err(|_| format!("{}: opening the records timed out", path.display()))?
            .map_err(|error| format!("{}: {error}", path.display()))?;
        references.push((path, name, reference, records));
    }
    let Some(record_bytes) = references.first().map(|(_, _, reference, _)| reference.row_bytes()) else {
        return Err("no reference to score".to_owned());
    };
    let geometry = Geometry {
        vocabulary,
        record_bytes,
        group: batch * flags.rows,
    };
    let scorer = Scorer::open(runtime, geometry).map_err(|error| format!("the scorer would not open: {error}"))?;
    let mut outcome = Ok(());
    for (path, name, reference, records) in references {
        outcome = score_one(flags, runtime, &scorer, batch, path, name, reference, records);
        if outcome.is_err() {
            break;
        }
    }
    scorer.close();
    outcome
}

#[allow(
    clippy::too_many_arguments,
    reason = "one reference's run: what it reads, where it goes and how it is named"
)]
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread waits for the run and writes the results"
)]
fn score_one(
    flags: &Flags,
    runtime: &Arc<EngineRuntime>,
    scorer: &Arc<Scorer>,
    batch: usize,
    path: &Path,
    name: &str,
    reference: Reference,
    records: OwnedFile,
) -> Result<(), String> {
    let shape = Shape {
        context: reference.context,
        steps: reference.records(),
    };
    let chunks = reference.chunks();
    let started = Instant::now();
    let forcer: Arc<dyn Forcer> = Arc::clone(runtime) as Arc<dyn Forcer>;
    let source = Source {
        forcer,
        io: Arc::clone(runtime.io()),
        scorer: Arc::clone(scorer),
        reference,
        records,
        rows: flags.rows,
    };
    let values = score(source, batch)
        .wait(LONGEST_RUN)
        .map_err(|_| format!("{name}: the score did not finish in time"))?
        .map_err(|why| format!("{name}: {why}"))?;
    let seconds = started.elapsed().as_secs_f64();
    let stamp = summary::instant(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs()),
    );
    let text = summary::lines(name, &stamp, path, &flags.artifact, shape, &values)
        .ok_or_else(|| format!("{name}: nothing was scored"))?;
    io::stdout()
        .write_all(text.as_bytes())
        .map_err(|error| format!("writing the summary: {error}"))?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(&flags.results)
        .and_then(|mut file| file.write_all(text.as_bytes()))
        .map_err(|error| format!("{}: {error}", flags.results.display()))?;
    if let Some(directory) = &flags.values {
        let file = directory.join(format!("{name}.values"));
        fs::write(&file, values_file(&values, shape.steps)).map_err(|error| format!("{}: {error}", file.display()))?;
    }
    let line = format!("scored {name}: {chunks} chunks in {seconds:.1} s\n");
    io::stdout()
        .write_all(line.as_bytes())
        .map_err(|error| format!("writing the summary: {error}"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    #[test]
    fn the_engine_settings_turn_speculation_off() {
        let parsed = flags(&words(
            "--artifact a --results r --reference x --name n --library lib.so",
        ))
        .unwrap();
        assert_eq!(settings(&parsed, 512).unwrap().speculation.mode, SpeculationMode::None);
    }

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn references_pair_with_their_names_and_the_forcing_defaults_to_the_references() {
        let parsed = flags(&words(
            "--artifact a.edrl --results r.txt --reference x.kld --name x --reference y.kld --name y --batch 2",
        ))
        .unwrap();
        assert_eq!(
            parsed.references,
            [
                (PathBuf::from("x.kld"), "x".to_owned()),
                (PathBuf::from("y.kld"), "y".to_owned())
            ]
        );
        assert_eq!((parsed.rows, parsed.batch), (DEFAULT_ROWS, 2));
        assert_eq!(parsed.library, None);
        assert!(parsed.values.is_none());
    }

    #[test]
    fn rows_and_batch_keep_to_the_caps_of_the_reference_command() {
        let with = |extra: &str| {
            flags(&words(&format!(
                "--artifact a --results r --reference x --name n --library lib.so {extra}"
            )))
        };
        let ok = with("--rows 8 --batch 4 --host-mib 512 --workers 2").unwrap();
        assert_eq!((ok.rows, ok.batch, ok.host_mib, ok.workers), (8, 4, Some(512), Some(2)));
        assert_eq!(ok.library, Some(PathBuf::from("lib.so")));
        for refused in ["--rows 9", "--rows 0", "--batch 0", "--rows 8 --batch 5", "--workers 0"] {
            assert!(with(refused).is_err(), "{refused}");
        }
        assert!(with("--rows 4 --batch 8").is_ok());
    }

    #[test]
    fn missing_and_malformed_flags_are_usage_errors_and_nothing_is_started() {
        for bad in [
            "--artifact a --results r",
            "--artifact a --results r --reference x",
            "--artifact a --results r --reference x --name n --name m",
            "--artifact a --results r --reference x --name n --rows 0",
            "--artifact a --results r --reference x --name n --batch many",
            "--artifact a --results r --reference x --name n --frob 1",
            "--artifact a --reference x --name n",
        ] {
            let outcome = run(&words(bad));
            assert_eq!(outcome.code, EXIT_USAGE, "{bad}");
        }
    }

    #[test]
    fn the_values_file_is_three_counts_then_the_values_chunk_after_chunk() {
        let bytes = values_file(&[vec![1.0, 2.0], vec![3.0, 4.0]], 1);
        assert_eq!(bytes.len(), 12 + 32);
        assert_eq!(&bytes[..12], [2, 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(&bytes[12..20], 1.0f64.to_le_bytes());
        assert_eq!(&bytes[36..44], 4.0f64.to_le_bytes());
    }
}
