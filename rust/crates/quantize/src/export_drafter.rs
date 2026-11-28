//! The `export-drafter` command: the base artifact with its MTP layer replaced by a trained drafter's (from
//! `--parameters`, the `best.bin` of `train-drafter`) or by the checkpoint's own MTP weights. `--model`,
//! `--artifact` and `--out` are required and `--parameters` is not; `--config` names the settings file, as for
//! `assemble`.

use std::ffi::OsString;
use std::path::PathBuf;

use euhedral_inference_quantization::mtp::export::{Settings as Export, export};
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use crate::assemble::split_config;
use crate::rig::Rig;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--model, --artifact, --out, --parameters, --config";

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n"),
    }
}

/// The export the flags in `args` describe.
fn flags(args: &[String]) -> Result<Export, String> {
    let (mut parameters, mut model, mut artifact, mut out) = (None, None, None, None);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let slot = match arg.as_str() {
            "--parameters" => &mut parameters,
            "--model" => &mut model,
            "--artifact" => &mut artifact,
            "--out" => &mut out,
            other => {
                return Err(format!(
                    "unknown export-drafter option {other}; the options are {OPTIONS}"
                ));
            }
        };
        *slot = Some(PathBuf::from(value));
    }
    let (Some(model), Some(artifact), Some(out)) = (model, artifact, out) else {
        return Err("export-drafter needs --model, --artifact and --out".to_owned());
    };
    Ok(Export {
        parameters,
        model,
        artifact,
        out,
    })
}

/// Runs `export-drafter` with the flags in `args`, the words after its name.
pub fn run(args: &[String]) -> Outcome {
    let (config, rest) = match split_config(args) {
        Ok(split) => split,
        Err(message) => return usage_error(&message),
    };
    let export_settings = match flags(&rest) {
        Ok(settings) => settings,
        Err(message) => return usage_error(&message),
    };
    let words: Vec<OsString> = config
        .into_iter()
        .flat_map(|path| [OsString::from("--config"), path.into_os_string()])
        .collect();
    let settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    match export_on_rig(&settings, &export_settings) {
        Ok(()) => Outcome {
            code: EXIT_OK,
            message: format!(
                "wrote {} with the MTP layer of {}\n",
                export_settings.out.display(),
                export_settings
                    .parameters
                    .as_ref()
                    .unwrap_or(&export_settings.model)
                    .display()
            ),
        },
        Err(message) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("export-drafter failed: {message}\n"),
        },
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread runs the export and waits for each call"
)]
fn export_on_rig(settings: &Settings, export_settings: &Export) -> Result<(), String> {
    let rig = Rig::start(settings)?;
    let copy = usize::try_from(settings.host.copy_chunk_bytes.bytes()).unwrap_or(usize::MAX);
    let result = export(export_settings, &rig.io, &rig.host, copy).map_err(|error| error.to_string());
    rig.stop();
    result
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

    use std::path::Path;

    use super::*;

    fn words(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    #[test]
    fn the_parameters_are_optional_and_the_other_flags_are_not() {
        let plain = flags(&words(&["--model", "m", "--artifact", "a", "--out", "o"])).unwrap();
        assert_eq!(plain.parameters, None);
        assert_eq!(plain.out, Path::new("o"));
        let trained = flags(&words(&[
            "--out",
            "o",
            "--parameters",
            "p",
            "--artifact",
            "a",
            "--model",
            "m",
        ]))
        .unwrap();
        assert_eq!(trained.parameters.as_deref(), Some(Path::new("p")));
        for skip in ["--model", "--artifact", "--out"] {
            let mut args = words(&["--model", "m", "--artifact", "a", "--out", "o"]);
            let at = args.iter().position(|arg| arg == skip).unwrap();
            args.drain(at..at + 2);
            assert!(flags(&args).unwrap_err().contains("needs --model"), "without {skip}");
        }
    }

    #[test]
    fn a_flag_that_is_unknown_or_valueless_is_refused() {
        let base = ["--model", "m", "--artifact", "a", "--out", "o"];
        let with = |extra: &[&str]| words(&[base.as_slice(), extra].concat());
        assert!(
            flags(&with(&["--bogus", "1"]))
                .unwrap_err()
                .contains("unknown export-drafter option --bogus")
        );
        assert!(flags(&with(&["--parameters"])).unwrap_err().contains("needs a value"));
    }
}
