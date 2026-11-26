//! The `train-drafter` command: trains the MTP drafter from the corpus `generate` writes, from the base checkpoint, with
//! the draft head and the token embedding of the artifact. The output is `best.bin` (the best held-out parameters) and
//! `resume.bin` (the state a stopped run continues from) in `--out`.

use std::env::consts;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_quantization::mtp::train::settings::Settings as Training;
use euhedral_inference_quantization::mtp::train::wait::STALL;
use euhedral_inference_quantization::mtp::train::{Host, Log, Options, train};
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use crate::rig::Rig;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--corpus, --model, --artifact, --out, --cuda-libraries, --depths, --rows, --keys, --tokens, \
--epochs, --projection-rate, --norm-rate, --warmup, --holdout, --evaluate-every, --limit, --stop-after-saves, \
--library, --fresh, --poison-memory, --fail-in-save, --stall-seconds";

/// The flags of the command.
#[derive(Debug, PartialEq)]
struct Flags {
    corpus: PathBuf,
    model: PathBuf,
    artifact: PathBuf,
    out: PathBuf,
    math: PathBuf,
    library: Option<PathBuf>,
    settings: Training,
    stop_after_saves: u32,
    fresh: bool,
    poison: Option<u8>,
    fail_in_save: u32,
    stall: Duration,
}

/// What `--poison-memory` fills unwritten memory with: a NaN in every float format.
const POISON: u8 = 0xff;

fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut corpus, mut model, mut artifact, mut out, mut math, mut library) = (None, None, None, None, None, None);
    let (mut settings, mut stop_after_saves) = (Training::default(), 0);
    let (mut fresh, mut poison, mut fail_in_save, mut stall) = (false, None, 0, STALL);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        // The flags that take no value.
        match arg.as_str() {
            "--fresh" => {
                fresh = true;
                continue;
            }
            "--poison-memory" => {
                poison = Some(POISON);
                continue;
            }
            _ => {}
        }
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let whole = || {
            value
                .parse::<u32>()
                .map_err(|_| format!("{arg} needs a whole number, not {value:?}"))
        };
        let real = || {
            value
                .parse::<f32>()
                .map_err(|_| format!("{arg} needs a number, not {value:?}"))
        };
        match arg.as_str() {
            "--corpus" => corpus = Some(PathBuf::from(value)),
            "--model" => model = Some(PathBuf::from(value)),
            "--artifact" => artifact = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--cuda-libraries" => math = Some(PathBuf::from(value)),
            "--library" => library = Some(PathBuf::from(value)),
            "--depths" => settings.depths = whole()?,
            "--rows" => settings.rows = whole()?,
            "--keys" => settings.keys = whole()?,
            "--tokens" => settings.tokens = whole()?,
            "--epochs" => settings.epochs = whole()?,
            "--projection-rate" => settings.projection_rate = real()?,
            "--norm-rate" => settings.norm_rate = real()?,
            "--warmup" => settings.warmup = whole()?,
            "--holdout" => {
                settings.holdout = value
                    .parse()
                    .map_err(|_| format!("{arg} needs a whole number, not {value:?}"))?;
            }
            "--evaluate-every" => settings.evaluate_every = whole()?,
            "--limit" => {
                settings.limit = value
                    .parse()
                    .map_err(|_| format!("{arg} needs a whole number, not {value:?}"))?;
            }
            "--stop-after-saves" => stop_after_saves = whole()?,
            "--fail-in-save" => fail_in_save = whole()?,
            "--stall-seconds" => {
                stall = match whole()? {
                    0 => return Err(format!("{arg} needs at least 1 second")),
                    seconds => Duration::from_secs(u64::from(seconds)),
                };
            }
            other => {
                return Err(format!(
                    "unknown train-drafter option {other}; the options are {OPTIONS}"
                ));
            }
        }
    }
    let (Some(corpus), Some(model), Some(artifact), Some(out), Some(math)) = (corpus, model, artifact, out, math)
    else {
        return Err("train-drafter needs --corpus, --model, --artifact, --out and --cuda-libraries".to_owned());
    };
    settings.check().map_err(|error| error.to_string())?;
    Ok(Flags {
        corpus,
        model,
        artifact,
        out,
        math,
        library,
        settings,
        stop_after_saves,
        fresh,
        poison,
        fail_in_save,
        stall,
    })
}

/// The library the build puts the CUDA layer in, where this platform has one; other platforms name it.
fn default_library() -> Option<PathBuf> {
    (consts::OS == "linux" && consts::ARCH == "x86_64")
        .then(|| PathBuf::from("build/native/linux-x64/lib/libeuhedral_cuda.so"))
}

/// The line of the log: the seconds since the run began, then the message.
#[allow(
    clippy::print_stderr,
    reason = "a binary's progress log, as the Java command prints it"
)]
fn logger() -> Log {
    let started = Instant::now();
    Arc::new(move |line| eprintln!("[{:7.0} s] {line}", started.elapsed().as_secs_f64()))
}

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n"),
    }
}

/// Runs `train-drafter` with the flags in `args`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread loads the run and waits for it"
)]
pub fn run(args: &[String]) -> Outcome {
    let flags = match flags(args) {
        Ok(flags) => flags,
        Err(message) => return usage_error(&message),
    };
    let Some(library) = flags.library.clone().or_else(default_library) else {
        return usage_error("train-drafter needs --library on this platform");
    };
    let settings = match load_for_tool(&[], &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    let rig = match start(&settings) {
        Ok(rig) => rig,
        Err(message) => {
            return Outcome {
                code: EXIT_RUN_FAILED,
                message: format!("{message}\n"),
            };
        }
    };
    let options = Options {
        corpus: flags.corpus,
        model: flags.model,
        artifact: flags.artifact,
        out: flags.out,
        settings: flags.settings,
        library,
        math: flags.math,
        stop_after_saves: flags.stop_after_saves,
        fresh: flags.fresh,
        poison: flags.poison,
        fail_in_save: flags.fail_in_save,
        stall: flags.stall,
    };
    let attach = |source| rig.attach(source);
    let host = Host {
        io: &rig.io,
        lake: &rig.host,
        attach: &attach,
    };
    let result = train(&options, &host, &logger());
    rig.stop();
    match result {
        Ok(summary) => Outcome {
            code: EXIT_OK,
            message: format!(
                "{}: best held-out accepted drafts {:.4}, {} state saves\n",
                if summary.stopped { "stopped" } else { "finished" },
                summary.best,
                summary.saves
            ),
        },
        Err(error) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("train-drafter failed: {error}\n"),
        },
    }
}

fn start(settings: &Settings) -> Result<Rig, String> {
    Rig::start(settings)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

    use super::*;

    fn words(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    const REQUIRED: [&str; 10] = [
        "--corpus",
        "c",
        "--model",
        "m",
        "--artifact",
        "a",
        "--out",
        "o",
        "--cuda-libraries",
        "l",
    ];

    fn with(extra: &[&str]) -> Vec<String> {
        words(&[REQUIRED.as_slice(), extra].concat())
    }

    #[test]
    fn the_defaults_are_javas() {
        let parsed = flags(&with(&[])).unwrap();
        assert_eq!(parsed.settings, Training::default());
        assert_eq!(parsed.stop_after_saves, 0);
        assert_eq!(parsed.library, None);
        assert_eq!(parsed.out, PathBuf::from("o"));
        assert_eq!(
            (
                parsed.settings.depths,
                parsed.settings.rows,
                parsed.settings.keys,
                parsed.settings.tokens
            ),
            (4, 1024, 4096, 16384)
        );
    }

    #[test]
    fn every_value_flag_sets_its_own_setting() {
        let parsed = flags(&with(&[
            "--depths",
            "3",
            "--rows",
            "256",
            "--keys",
            "512",
            "--tokens",
            "100",
            "--epochs",
            "2",
            "--projection-rate",
            "0.5",
            "--norm-rate",
            "0.25",
            "--warmup",
            "7",
            "--holdout",
            "9",
            "--evaluate-every",
            "11",
            "--limit",
            "13",
            "--stop-after-saves",
            "1",
            "--library",
            "lib.so",
        ]))
        .unwrap();
        let settings = Training {
            depths: 3,
            rows: 256,
            keys: 512,
            tokens: 100,
            epochs: 2,
            projection_rate: 0.5,
            norm_rate: 0.25,
            warmup: 7,
            holdout: 9,
            evaluate_every: 11,
            limit: 13,
        };
        assert_eq!(parsed.settings, settings);
        assert_eq!(parsed.stop_after_saves, 1);
        assert_eq!(parsed.library, Some(PathBuf::from("lib.so")));
    }

    #[test]
    fn a_missing_required_flag_is_refused() {
        for skip in (0..REQUIRED.len()).step_by(2) {
            let mut args = words(&REQUIRED);
            args.drain(skip..skip + 2);
            assert!(
                flags(&args).unwrap_err().contains("needs --corpus"),
                "without {}",
                REQUIRED[skip]
            );
        }
    }

    #[test]
    fn a_flag_that_is_unknown_valueless_or_not_a_number_is_refused() {
        assert!(
            flags(&with(&["--bogus", "1"]))
                .unwrap_err()
                .contains("unknown train-drafter option --bogus")
        );
        assert!(flags(&with(&["--depths"])).unwrap_err().contains("needs a value"));
        assert!(
            flags(&with(&["--depths", "x"]))
                .unwrap_err()
                .contains("whole number, not \"x\"")
        );
        assert!(flags(&with(&["--depths", "-1"])).unwrap_err().contains("whole number"));
        assert!(
            flags(&with(&["--norm-rate", "fast"]))
                .unwrap_err()
                .contains("needs a number, not \"fast\"")
        );
        assert!(
            flags(&with(&["--limit", "many"]))
                .unwrap_err()
                .contains("whole number, not \"many\"")
        );
    }

    #[test]
    fn the_flags_that_take_no_value_set_only_their_own_switch() {
        let parsed = flags(&with(&["--fresh", "--poison-memory", "--fail-in-save", "2"])).unwrap();
        assert!(parsed.fresh);
        assert_eq!(parsed.poison, Some(0xff));
        assert_eq!(parsed.fail_in_save, 2);
        assert_eq!(parsed.settings, Training::default());
        let plain = flags(&with(&[])).unwrap();
        assert!((plain.fresh, plain.poison, plain.fail_in_save) == (false, None, 0));
    }

    #[test]
    fn the_stall_is_half_an_hour_unless_asked_for_and_never_nothing() {
        assert_eq!(flags(&with(&[])).unwrap().stall, STALL);
        assert_eq!(
            flags(&with(&["--stall-seconds", "90"])).unwrap().stall,
            Duration::from_secs(90)
        );
        assert!(
            flags(&with(&["--stall-seconds", "0"]))
                .unwrap_err()
                .contains("at least 1 second")
        );
    }

    #[test]
    fn settings_the_run_cannot_work_with_are_refused_before_anything_starts() {
        for bad in [
            ["--holdout", "0"],
            ["--epochs", "0"],
            ["--tokens", "0"],
            ["--depths", "17"],
            ["--projection-rate", "0"],
            ["--norm-rate", "NaN"],
            ["--keys", "100"],
        ] {
            assert!(flags(&with(&bad)).is_err(), "{bad:?}");
        }
    }
}
