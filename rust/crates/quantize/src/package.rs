//! The `package-tokenizer` command: parses its flags, starts the runtime and runs the library's packaging on it.
//!
//! The runtime is a lattice with the asynchronous I/O as a source and one host lake; the calling thread only starts the
//! run and waits for its one completion.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use euhedral_inference_quantization::package_tokenizer::{PackageRequest, Packaged, package_tokenizer};
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use crate::rig::Rig;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// The flags of the command.
#[derive(Debug, PartialEq, Eq)]
pub struct PackageArgs {
    pub artifact: PathBuf,
    pub checkpoint: PathBuf,
    pub out: Option<PathBuf>,
}

const USAGE: &str =
    "usage: quantization package-tokenizer --artifact ARTIFACT --tokenizer CHECKPOINT [--out ARTIFACT]\n";

/// Reads the flags of the command; `--config FILE` belongs to the settings and is skipped here.
pub fn parse(args: &[String]) -> Result<PackageArgs, String> {
    let (mut artifact, mut checkpoint, mut out) = (None, None, None);
    let mut words = args.iter();
    while let Some(flag) = words.next() {
        let (name, inline) = match flag.split_once('=') {
            Some((name, value)) => (name, Some(value.to_owned())),
            None => (flag.as_str(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| words.next().cloned())
                .ok_or_else(|| format!("{name} needs a value"))
        };
        let slot = match name {
            "--artifact" => &mut artifact,
            "--tokenizer" => &mut checkpoint,
            "--out" => &mut out,
            "--config" => {
                value()?;
                continue;
            }
            _ => return Err(format!("unknown flag '{flag}'")),
        };
        if slot.replace(PathBuf::from(value()?)).is_some() {
            return Err(format!("{name} is given twice"));
        }
    }
    Ok(PackageArgs {
        artifact: artifact.ok_or("--artifact is required")?,
        checkpoint: checkpoint.ok_or("--tokenizer is required")?,
        out,
    })
}

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n{USAGE}"),
    }
}

/// Runs the command on `args`, the words after its name.
pub fn run(args: &[String]) -> Outcome {
    let parsed = match parse(args) {
        Ok(parsed) => parsed,
        Err(message) => return usage_error(&message),
    };
    let words: Vec<OsString> = args.iter().map(OsString::from).collect();
    let settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    match package(&settings, parsed) {
        Ok(packaged) => Outcome {
            code: EXIT_OK,
            message: format!(
                "{}: {} bytes of content kept, {} bytes of tokenizer written\n",
                packaged.target.display(),
                packaged.content_bytes,
                packaged.section_bytes
            ),
        },
        Err(message) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("package-tokenizer failed: {message}\n"),
        },
    }
}

/// Starts the runtime, runs the packaging and stops the runtime.
fn package(settings: &Settings, args: PackageArgs) -> Result<Packaged, String> {
    let rig = Rig::start(settings)?;
    let mut request = match args.out {
        Some(out) => PackageRequest::to_output(args.artifact, args.checkpoint, out),
        None => PackageRequest::in_place(args.artifact, args.checkpoint),
    };
    request.copy_chunk_bytes = usize::try_from(settings.host.copy_chunk_bytes.bytes()).unwrap_or(usize::MAX);

    let (sender, receiver) = mpsc::channel();
    let started = package_tokenizer(
        Arc::clone(&rig.io),
        Arc::clone(&rig.host),
        request,
        Box::new(move |result| {
            // The receiver is gone only when the run was abandoned.
            let _ = sender.send(result);
        }),
    );
    let result = match started {
        Ok(()) => receiver
            .recv()
            .map_err(|_| "the run ended without a result".to_owned())
            .and_then(|result| result.map_err(|error| error.to_string())),
        Err(error) => Err(error.to_string()),
    };
    rig.stop();
    result
}
