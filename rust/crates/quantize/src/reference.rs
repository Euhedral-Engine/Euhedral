//! The `reference` command: the logits of a model over a text, written as a `.kld` file.

use std::env::consts;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_quantization::kld::Reference;
use euhedral_inference_quantization::reference::{Forcer, Options, tokenize, write};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, load_file, process_env};

use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// The context of a chunk when `--context` is not given.
const DEFAULT_CONTEXT: usize = 2560;
/// The longest a reference takes: a bound on the wait, not a rate.
const LONGEST_RUN: Duration = Duration::from_secs(7 * 24 * 60 * 60);

struct Flags {
    artifact: PathBuf,
    tokenizer: PathBuf,
    text: PathBuf,
    out: PathBuf,
    context: usize,
    chunks: usize,
    parse_special: bool,
    config: Option<PathBuf>,
    library: Option<PathBuf>,
}

fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut artifact, mut tokenizer, mut text, mut out) = (None, None, None, None);
    let (mut context, mut chunks) = (DEFAULT_CONTEXT, 0);
    let (mut parse_special, mut config, mut library) = (false, None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        if arg == "--parse-special" {
            parse_special = true;
            continue;
        }
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("{arg} needs a whole number"))
        };
        match arg.as_str() {
            "--artifact" => artifact = Some(PathBuf::from(value)),
            "--tokenizer" => tokenizer = Some(PathBuf::from(value)),
            "--text" => text = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--context" => context = number()?,
            "--chunks" => chunks = number()?,
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" => library = Some(PathBuf::from(value)),
            other => return Err(format!("unknown reference option {other}")),
        }
    }
    let (Some(artifact), Some(text), Some(out)) = (artifact, text, out) else {
        return Err("reference needs --artifact, --text and --out".to_owned());
    };
    Ok(Flags {
        tokenizer: tokenizer.unwrap_or_else(|| artifact.clone()),
        artifact,
        text,
        out,
        context,
        chunks,
        parse_special,
        config,
        library,
    })
}

/// The library the build puts the CUDA layer in, where this platform has one; other platforms name it.
fn default_library() -> Option<PathBuf> {
    (consts::OS == "linux" && consts::ARCH == "x86_64")
        .then(|| PathBuf::from("build/native/linux-x64/lib/libeuhedral_cuda.so"))
}

/// The engine's settings for a run of `context`-token chunks: the file `config` names, or the defaults, with the
/// artifact, a context that holds a chunk and no prefix cache, which a fresh sequence per chunk does not use.
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
    settings.context.max_tokens = u32::try_from(flags.context.next_multiple_of(512))
        .map_err(|_| format!("a context of {} tokens is too long", flags.context))?;
    settings.prefix_cache.size = ByteSize::mib(0);
    Ok(settings)
}

fn failed(message: String) -> Outcome {
    Outcome {
        code: EXIT_RUN_FAILED,
        message: format!("{message}\n"),
    }
}

/// Runs `reference` with the flags in `args`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads the text and waits for the run"
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
            message: "reference needs --cuda-library on this platform\n".to_owned(),
        };
    };
    let settings = match settings(&flags) {
        Ok(settings) => settings,
        Err(message) => return failed(message),
    };
    let text = match fs::read_to_string(&flags.text) {
        Ok(text) => text,
        Err(error) => return failed(format!("{}: {error}", flags.text.display())),
    };
    let tokens = match tokenize(
        &flags.tokenizer,
        &text,
        flags.context,
        flags.chunks,
        flags.parse_special,
    ) {
        Ok(tokens) => tokens,
        Err(error) => return failed(error.to_string()),
    };
    drop(text);
    let (owner, waiting) = EngineRuntime::start(&settings, &library);
    let runtime = match waiting.wait(LONGEST_RUN) {
        Ok(Ok(runtime)) => runtime,
        Ok(Err(error)) => return failed(format!("the engine did not load: {error}")),
        Err(_) => return failed("the engine did not finish loading".to_owned()),
    };
    let vocabulary = runtime.config().vocab_size;
    let chunks = tokens.len();
    let forcer: Arc<dyn Forcer> = Arc::clone(&runtime) as Arc<dyn Forcer>;
    let ended = write(
        runtime.io(),
        runtime.host(),
        forcer,
        &flags.out,
        Reference {
            context: flags.context,
            vocabulary,
            tokens,
        },
        Options::default(),
    );
    let result = ended.wait(LONGEST_RUN);
    let shutdown = owner
        .shutdown(runtime)
        .err()
        .map_or_else(String::new, |error| format!("shutdown: {error}\n"));
    let mut outcome = match result {
        Ok(Ok(written)) if written.finished => Outcome {
            code: EXIT_OK,
            message: summary(&flags.out, chunks),
        },
        Ok(Ok(written)) => failed(format!("stopped after {} of {chunks} chunks", written.completed)),
        Ok(Err(error)) => failed(error.to_string()),
        Err(_) => failed("the reference did not finish in time".to_owned()),
    };
    outcome.message.push_str(&shutdown);
    outcome
}

fn summary(out: &Path, chunks: usize) -> String {
    format!("wrote {chunks} chunks to {}\n", out.display())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    #[test]
    fn the_tokenizer_and_context_default_to_the_artifact_and_the_usual_context() {
        let parsed = flags(&words("--artifact a.edrl --text t.txt --out o.kld")).unwrap();
        assert_eq!(parsed.tokenizer, PathBuf::from("a.edrl"));
        assert_eq!((parsed.context, parsed.chunks, parsed.parse_special), (2560, 0, false));
        let parsed = flags(&words(
            "--artifact a --tokenizer b --text t --out o --context 64 --chunks 2 --parse-special",
        ))
        .unwrap();
        assert_eq!(parsed.tokenizer, PathBuf::from("b"));
        assert_eq!((parsed.context, parsed.chunks, parsed.parse_special), (64, 2, true));
    }

    #[test]
    fn missing_and_malformed_flags_are_usage_errors_and_nothing_is_started() {
        for bad in [
            "--artifact a --text t",
            "--artifact a --text t --out o --context many",
            "--artifact a --text t --out o --chunks",
            "--artifact a --text t --out o --frob 1",
        ] {
            let outcome = run(&words(bad));
            assert_eq!(outcome.code, EXIT_USAGE, "{bad}");
        }
    }
}
