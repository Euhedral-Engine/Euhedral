//! The `assemble` command: a mixed artifact from calibrated source artifacts and an allocation, written on the host
//! runtime. As in Java, a successful run prints nothing.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};

use euhedral_inference_quantization::assemble::{Assembled, Request, run as assemble};
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use crate::rig::Rig;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n"),
    }
}

/// Splits `--config FILE` (the settings') from the flags the library parses.
pub(crate) fn split_config(args: &[String]) -> Result<(Option<PathBuf>, Vec<String>), String> {
    let (mut config, mut rest) = (None, Vec::new());
    let mut words = args.iter();
    while let Some(word) = words.next() {
        if word == "--config" {
            let value = words.next().ok_or("--config needs a value")?;
            config = Some(PathBuf::from(value));
        } else {
            rest.push(word.clone());
        }
    }
    Ok((config, rest))
}

/// Runs the command on `args`, the words after its name.
pub fn run(args: &[String]) -> Outcome {
    let (config, rest) = match split_config(args) {
        Ok(split) => split,
        Err(message) => return usage_error(&message),
    };
    let request = match Request::parse(&rest) {
        Ok(request) => request,
        Err(error) => return usage_error(&error.to_string()),
    };
    let words: Vec<OsString> = config
        .into_iter()
        .flat_map(|path| [OsString::from("--config"), path.into_os_string()])
        .collect();
    let settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    match assemble_on_rig(&settings, request) {
        Ok(_) => Outcome {
            code: EXIT_OK,
            message: String::new(),
        },
        Err(message) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("assemble failed: {message}\n"),
        },
    }
}

fn assemble_on_rig(settings: &Settings, request: Request) -> Result<Assembled, String> {
    let rig = Rig::start(settings)?;
    let chunk_bytes = usize::try_from(settings.host.copy_chunk_bytes.bytes()).unwrap_or(usize::MAX);
    let (sender, receiver) = mpsc::channel();
    let started = assemble(
        &rig.io,
        &rig.host,
        request,
        chunk_bytes,
        Arc::new(move |result| {
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
