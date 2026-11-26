//! The `score` command: the logits of an artifact against the references of a model, as a summary per reference.

use std::env::consts;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_quantization::kld::{self, Reference};
use euhedral_inference_quantization::score::device::Geometry;
use euhedral_inference_quantization::score::summary::{self, FIELDS, Shape};
use euhedral_inference_quantization::score::{RowForcer, Scorer, Source, batch_for, score};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, load_file, process_env};

use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// The longest a score takes: a bound on the wait, not a rate.
const LONGEST_RUN: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// The rows of a verification when `--rows` is not given: one, as the reference command forces.
const DEFAULT_ROWS: usize = 1;

struct Flags {
    artifact: PathBuf,
    results: PathBuf,
    references: Vec<(PathBuf, String)>,
    values: Option<PathBuf>,
    rows: usize,
    batch: Option<usize>,
    config: Option<PathBuf>,
    library: Option<PathBuf>,
}

fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut artifact, mut results, mut values, mut config, mut library) = (None, None, None, None, None);
    let (mut paths, mut names) = (Vec::new(), Vec::new());
    let (mut rows, mut batch) = (DEFAULT_ROWS, None);
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
            "--batch" => batch = Some(number()?),
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" => library = Some(PathBuf::from(value)),
            other => return Err(format!("unknown score option {other}")),
        }
    }
    let (Some(artifact), Some(results)) = (artifact, results) else {
        return Err("score needs --artifact, --results and at least one --reference with its --name".to_owned());
    };
    if paths.is_empty() || paths.len() != names.len() {
        return Err("score needs as many --name as --reference, and at least one".to_owned());
    }
    if rows == 0 {
        return Err("--rows must be at least 1".to_owned());
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
    })
}

/// The library the build puts the CUDA layer in, where this platform has one; other platforms name it.
fn default_library() -> Option<PathBuf> {
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
    settings.context.max_tokens = u32::try_from(context.next_multiple_of(512))
        .map_err(|_| format!("a context of {context} tokens is too long"))?;
    settings.prefix_cache.size = ByteSize::mib(0);
    Ok(settings)
}

/// The context and the chunks in the header of a reference file, read before the engine starts so the engine's
/// context can hold a chunk.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads a header before the engine starts"
)]
fn dimensions(path: &Path) -> Result<(usize, usize), String> {
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
    let verified = usize::try_from(runtime.plan().verified_rows()).unwrap_or(usize::MAX);
    if flags.rows > verified {
        return Err(format!(
            "--rows {} is more than the {verified} rows the engine verifies",
            flags.rows
        ));
    }
    let batch = batch_for(flags.batch, runtime.sequences(), most_chunks)?;
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
    records: euhedral_io::OwnedFile,
) -> Result<(), String> {
    let shape = Shape {
        context: reference.context,
        steps: reference.records(),
    };
    let chunks = reference.chunks();
    let started = Instant::now();
    let forcer: Arc<dyn RowForcer> = Arc::clone(runtime) as Arc<dyn RowForcer>;
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

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn references_pair_with_their_names_and_the_forcing_defaults_to_one_row() {
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
        assert_eq!((parsed.rows, parsed.batch), (1, Some(2)));
        assert!(parsed.values.is_none());
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
