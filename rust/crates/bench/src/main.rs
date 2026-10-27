//! `euhedral-bench`: measures this engine with the Java benchmark's configuration files and writes rows in its result
//! schema.
//!
//! ```text
//! run CONFIG.json [--fork-id ID] [--validate-only]
//! dump --artifact FILE --prefix N --steps N --out FILE [--cuda-library FILE]
//! ```
//!
//! Exit codes: 0 success, 1 a measured row failed or was ineligible, 2 usage or configuration, 3 the engine did not load.
//!
//! The main thread loads the settings, starts the runtime, starts each request and waits for its latch; the work runs on
//! the engine's lattice. This file is the binary's edge: it prints, and it is the only reader of the environment.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a command-line binary: its output is the report"
)]

use std::env;
use std::path::{Path, PathBuf};
use std::process::{self, ExitCode};
use std::sync::Arc;
use std::time::SystemTime;

use euhedral_graph::Latch;
use euhedral_inference_bench::artifact::load_tokenizer;
use euhedral_inference_bench::config::{ConfigError, Options};
use euhedral_inference_bench::dump::{split_document, teacher_forced_report, write_report};
use euhedral_inference_bench::result::{Fork, ResultWriter, SUCCESS, iso_instant, summary};
use euhedral_inference_bench::run::{self, Context, prepare, provenance, run_id};
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, load_file, process_env};

const USAGE: &str = "usage:
  run CONFIG.json [--fork-id ID] [--validate-only]
  dump --artifact FILE --prefix N --steps N --out FILE [--cuda-library FILE]

CONFIG.json is a Java benchmark configuration file (see benchmark/configs); it requires \"artifact\" and \"cudaLibrary\".
--fork-id labels this process as one externally launched fork.
--validate-only checks the configuration and the prompts, prints the plan, and exits without touching the GPU.
Exit codes: 0 success, 1 a measured row failed or was ineligible, 2 usage or configuration, 3 the engine did not load.
";

/// The library the Java configuration files name, relative to the repository root.
const DEFAULT_CUDA_LIBRARY: &str = "build/native/linux-x64/lib/libeuhedral_cuda.so";

fn main() -> ExitCode {
    let started = SystemTime::now();
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|arg| arg == "--help") {
        print!("{USAGE}");
        return if args.is_empty() {
            ExitCode::from(2)
        } else {
            ExitCode::SUCCESS
        };
    }
    let result = match args[0].as_str() {
        "run" => run_command(&args[1..], &args.join(" "), started),
        "dump" => dump_command(&args[1..]),
        other => Err(Failure::Usage(format!("unknown command {other}"))),
    };
    match result {
        Ok(code) => ExitCode::from(code),
        Err(Failure::Usage(message)) => {
            eprintln!("error: {message}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::NotLoaded(message)) => {
            eprintln!("not run: the engine did not load: {message}");
            ExitCode::from(3)
        }
    }
}

enum Failure {
    /// The command line or the configuration is wrong.
    Usage(String),
    /// The engine did not start.
    NotLoaded(String),
}

impl From<ConfigError> for Failure {
    fn from(error: ConfigError) -> Self {
        let ConfigError::Invalid(message) = error;
        Failure::Usage(message)
    }
}

fn run_command(args: &[String], command_line: &str, started: SystemTime) -> Result<u8, Failure> {
    let Some(config) = args.first().filter(|arg| !arg.starts_with("--")) else {
        return Err(Failure::Usage("run needs a configuration file".to_string()));
    };
    let mut fork_id = None;
    let mut validate_only = false;
    let mut rest = args[1..].iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--validate-only" => validate_only = true,
            "--fork-id" => match rest.next().filter(|value| !value.trim().is_empty()) {
                Some(value) => fork_id = Some(value.clone()),
                None => return Err(Failure::Usage("--fork-id needs a value".to_string())),
            },
            other => return Err(Failure::Usage(format!("unknown run option {other}"))),
        }
    }

    let options = Options::load(Path::new(config), SystemTime::now())?;
    ResultWriter::check_output(&options.output, options.overwrite, options.append)?;
    let tokenizer = load_tokenizer(&options.artifact)?;
    let prepared = prepare(&options, &tokenizer)?;
    println!(
        "max context: {} tokens; prefix cache 0 bytes; warmup {}, measured {} per scenario; generation greedy; fork {}",
        options.max_context_tokens,
        options.warmup,
        options.iterations,
        fork_id.as_deref().unwrap_or("unspecified")
    );
    for (scenario, prompts) in &prepared.prompts {
        for prompt in prompts.iter() {
            println!(
                "scenario {}: {} prompt {}/{} tokens, sha256 {}",
                scenario.name(),
                options.prompt_corpus.generator(),
                prompt.ids.len(),
                scenario.target_prompt_tokens,
                prompt.sha256
            );
        }
    }
    println!("output: {}", options.output.display());
    if validate_only {
        println!("validation passed; no model was loaded");
        return Ok(0);
    }

    let settings = settings_for(&options)?;
    let (ready, waiting) = Latch::new();
    let owner = EngineRuntime::start(&settings, &options.cuda_library, ready);
    let runtime = match waiting.wait() {
        Ok(runtime) => runtime,
        Err(error) => return Err(Failure::NotLoaded(error.to_string())),
    };
    let report = runtime.report();
    println!(
        "loaded: {} device bytes in {:.1} s",
        report.allocated_bytes,
        report.ready.as_secs_f64()
    );

    let pid = process::id();
    let context = Context {
        run_id: run_id(started, pid),
        fork: Fork {
            id: fork_id.clone(),
            source: if fork_id.is_some() { "external" } else { "unspecified" }.to_string(),
            pid: Some(u64::from(pid)),
            jvm_started_at: Some(iso_instant(started)),
        },
        provenance: provenance(command_line.to_string()),
    };
    let mut writer = match ResultWriter::create(
        &options.output,
        options.json_output(),
        options.overwrite,
        options.append,
    ) {
        Ok(writer) => writer,
        Err(error) => {
            let _ = owner.shutdown(runtime);
            return Err(error.into());
        }
    };
    let outcome = run::run(&options, &prepared, &runtime, &tokenizer, &context, &mut |row| {
        println!(
            "{} {} #{}: {}{}",
            if row.warmup { "warmup" } else { "measured" },
            row.scenario.name,
            row.iteration,
            row.status,
            row.status_reason
                .as_ref()
                .map_or(String::new(), |reason| format!(" ({reason})"))
        );
        writer.write(row)
    });
    let rows = writer.rows().to_vec();
    let closed = writer.close();
    if let Err(error) = owner.shutdown(runtime) {
        eprintln!("shutdown: {error}");
    }
    outcome?;
    closed?;

    println!();
    print!("{}", summary(&rows));
    println!("results: {}", options.output.display());
    let failed = rows.iter().filter(|row| !row.warmup).any(|row| row.status != SUCCESS);
    Ok(u8::from(failed))
}

/// The engine settings of a run: the settings file the configuration names (or the defaults), with the artifact, the
/// context and the prefix cache the configuration chooses.
fn settings_for(options: &Options) -> Result<Settings, Failure> {
    let model = ModelSettings {
        artifact: options.artifact.clone(),
        name: options
            .artifact
            .file_stem()
            .map_or_else(|| "model".to_string(), |stem| stem.to_string_lossy().into_owned()),
    };
    let mut settings = match &options.engine_config {
        None => Settings::with_model(model.clone()),
        Some(path) => {
            load_file(path, &process_env).map_err(|errors| Failure::Usage(format!("{}: {errors}", path.display())))?
        }
    };
    settings.model = model;
    settings.context.max_tokens = options.max_context_tokens;
    if let Some(tokens) = options.kv_device_minimum_tokens {
        settings.context.kv_cache_on_gpu_tokens = tokens;
    }
    settings.prefix_cache.size = ByteSize::mib(0);
    Ok(settings)
}

fn dump_command(args: &[String]) -> Result<u8, Failure> {
    let (mut artifact, mut out, mut library) = (None, None, None);
    let (mut prefix, mut steps) = (None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = |name: &str| {
            rest.next()
                .cloned()
                .ok_or_else(|| Failure::Usage(format!("{name} needs a value")))
        };
        let number = |text: String, name: &str| {
            text.parse::<usize>()
                .map_err(|_| Failure::Usage(format!("{name} needs a whole number")))
        };
        match arg.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(value("--artifact")?)),
            "--out" => out = Some(PathBuf::from(value("--out")?)),
            "--cuda-library" => library = Some(PathBuf::from(value("--cuda-library")?)),
            "--prefix" => prefix = Some(number(value("--prefix")?, "--prefix")?),
            "--steps" => steps = Some(number(value("--steps")?, "--steps")?),
            other => return Err(Failure::Usage(format!("unknown dump option {other}"))),
        }
    }
    let (Some(artifact), Some(out), Some(prefix), Some(steps)) = (artifact, out, prefix, steps) else {
        return Err(Failure::Usage(
            "dump needs --artifact, --prefix, --steps and --out".to_string(),
        ));
    };
    let library = library.unwrap_or_else(|| PathBuf::from(DEFAULT_CUDA_LIBRARY));

    let tokenizer = load_tokenizer(&artifact)?;
    let (prefix_ids, forced) = split_document(&tokenizer, prefix, steps)?;
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.clone(),
        name: "dump".to_string(),
    });
    settings.context.max_tokens = ((prefix + steps) as u32).next_multiple_of(512);
    settings.prefix_cache.size = ByteSize::mib(0);

    let (ready, waiting) = Latch::new();
    let owner = EngineRuntime::start(&settings, &library, ready);
    let runtime: Arc<EngineRuntime> = match waiting.wait() {
        Ok(runtime) => runtime,
        Err(error) => return Err(Failure::NotLoaded(error.to_string())),
    };
    let report = teacher_forced_report(&runtime, prefix_ids, forced);
    if let Err(error) = owner.shutdown(runtime) {
        eprintln!("shutdown: {error}");
    }
    let bytes = report?;
    write_report(&out, &bytes)?;
    println!("wrote {} teacher-forced steps to {}", steps, out.display());
    Ok(0)
}
