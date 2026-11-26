//! The `calibrate` command for GPTQ: the BF16 checkpoint run layer by layer over calibration tokens, each layer's
//! linears rounded from the Hessians of their inputs and written to the output directory. The trellis arm is not part
//! of this build, and so are the rounding of the heads and of the MTP block: a run stops after the layers it was
//! asked for, which must be fewer than the model's, and keeps its resume file.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use euhedral_inference_quantization::calibrate::device::{Calibration, Settings as RunSettings};
use euhedral_inference_quantization::calibrate::plan::resume_key;
use euhedral_inference_quantization::calibrate::{Plan, drive};
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::gptq::format::Nvfp4Search;
use euhedral_inference_quantization::npy::Npy;
use euhedral_inference_quantization::recipe::{Quantization, Recipe};
use euhedral_inference_settings::{load_for_tool, process_env};

use crate::rig::Rig;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

/// Java's defaults and caps.
const DEFAULT_DAMP: f32 = 0.01;
const DEFAULT_BATCH: usize = 8;
const DEFAULT_LAYERS: usize = 64;
/// The most sequences run at once: each one holds a pipeline slot.
const MOST_BATCH: usize = 64;
/// Where the CUDA math libraries are when `--cuda-libraries` is not given.
const DEFAULT_MATH_DIRECTORY: &str = "/usr/local/cuda/lib64";
const DEFAULT_LIBRARY: &str = "build/native/linux-x64/lib/libeuhedral_cuda.so";
/// What the trellis arm takes: refused by name.
const TRELLIS_OPTIONS: [&str; 5] = [
    "--bits",
    "--rounding-lanes",
    "--viterbi-scratch-mib",
    "--memory-reserve-mib",
    "--write-buffer-mib",
];
/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--model, --quantization, --compressed, --output-head, --tokens, --draft-ids-from, --out, \
--act-order, --damp, --batch, --sequences, --layers, --skip-mtp, --nvfp4-above, --nvfp4-scale-max, --format, \
--config, --library (or --cuda-library), --cuda-libraries";

/// The flags of a run, checked.
#[derive(Debug)]
pub(crate) struct Flags {
    pub model: PathBuf,
    pub quantization: Quantization,
    pub compressed: bool,
    pub output_head: Option<String>,
    pub tokens: PathBuf,
    pub out: PathBuf,
    pub act_order: bool,
    pub damp: f32,
    pub batch: usize,
    /// Sequences taken from the tokens; 0 takes all.
    pub sequences: usize,
    /// Layers rounded, from the first.
    pub layers: usize,
    pub nvfp4_above: i32,
    pub nvfp4_scale_max: f32,
    pub config: Option<PathBuf>,
    pub library: Option<PathBuf>,
    pub math_directory: PathBuf,
}

fn trellis_refusal(option: &str) -> String {
    format!("calibrate: {option} is a trellis option, and the trellis arm is not part of this build")
}

pub(crate) fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut model, mut quantization, mut output_head, mut tokens, mut out) = (None, None, None, None, None);
    let (mut compressed, mut act_order) = (false, false);
    let (mut damp, mut batch, mut sequences, mut layers) = (DEFAULT_DAMP, DEFAULT_BATCH, 0, DEFAULT_LAYERS);
    let (mut above, mut scale_max) = (Nvfp4Search::DEFAULT.above(), Nvfp4Search::DEFAULT.scale_max());
    let (mut config, mut library) = (None, None);
    let mut math_directory = PathBuf::from(DEFAULT_MATH_DIRECTORY);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--compressed" => {
                compressed = true;
                continue;
            }
            "--act-order" => {
                act_order = true;
                continue;
            }
            // The MTP block is not rounded by this build either way.
            "--skip-mtp" => continue,
            _ => {}
        }
        if TRELLIS_OPTIONS.contains(&arg.as_str()) {
            return Err(trellis_refusal(arg));
        }
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("{arg} needs a whole number"))
        };
        let real = || value.parse::<f32>().map_err(|_| format!("{arg} needs a number"));
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--quantization" => {
                quantization = Some(Quantization::parse(value).map_err(|error| error.to_string())?);
            }
            "--format" => match value.as_str() {
                "gptq" => {}
                "trellis" => return Err(trellis_refusal("--format trellis")),
                other => return Err(format!("--format is gptq or trellis, not {other}")),
            },
            "--output-head" => output_head = Some(value.clone()),
            "--tokens" => tokens = Some(PathBuf::from(value)),
            // Only the heads read the draft ids, and the heads are not part of this build.
            "--draft-ids-from" => {}
            "--out" => out = Some(PathBuf::from(value)),
            "--damp" => damp = real()?,
            "--batch" => batch = number()?,
            "--sequences" => sequences = number()?,
            "--layers" => layers = number()?,
            "--nvfp4-above" => {
                above = value
                    .parse::<i32>()
                    .map_err(|_| format!("{arg} needs a whole number"))?;
            }
            "--nvfp4-scale-max" => scale_max = real()?,
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" | "--library" => library = Some(PathBuf::from(value)),
            "--cuda-libraries" => math_directory = PathBuf::from(value),
            other => return Err(format!("unknown calibrate option {other}; the options are {OPTIONS}")),
        }
    }
    let (Some(model), Some(quantization), Some(tokens), Some(out)) = (model, quantization, tokens, out) else {
        return Err("calibrate needs --model, --quantization, --tokens and --out".to_owned());
    };
    if quantization == Quantization::Trellis {
        return Err(trellis_refusal("--quantization trellis"));
    }
    if !(damp.is_finite() && damp > 0.0) {
        return Err(format!("--damp must be positive, not {damp}"));
    }
    if !(1..=MOST_BATCH).contains(&batch) {
        return Err(format!("--batch is 1 to {MOST_BATCH}, not {batch}"));
    }
    if layers == 0 {
        return Err("--layers must be at least 1".to_owned());
    }
    Nvfp4Search::new(above, scale_max).map_err(|error| error.to_string())?;
    Recipe::new(quantization, compressed, output_head.as_deref()).map_err(|error| error.to_string())?;
    Ok(Flags {
        model,
        quantization,
        compressed,
        output_head,
        tokens,
        out,
        act_order,
        damp,
        batch,
        sequences,
        layers,
        nvfp4_above: above,
        nvfp4_scale_max: scale_max,
        config,
        library,
        math_directory,
    })
}

fn failed(message: String) -> Outcome {
    Outcome {
        code: EXIT_RUN_FAILED,
        message: format!("calibrate failed: {message}\n"),
    }
}

/// The tokens of the first `limit` sequences (all when 0) and the tokens of one sequence.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads the tokens before the run starts"
)]
fn read_tokens(path: &Path, limit: usize) -> Result<(Vec<i32>, usize), String> {
    let bytes = fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let npy = Npy::parse(&bytes).map_err(|error| format!("{}: {error}", path.display()))?;
    let [sequences, length] = npy.shape[..] else {
        return Err(format!("{}: the tokens are not [sequences, length]", path.display()));
    };
    let (sequences, length) = (sequences as usize, length as usize);
    let mut tokens = npy
        .to_vec::<i32>()
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let taken = if limit == 0 { sequences } else { sequences.min(limit) };
    tokens.truncate(taken * length);
    if taken == 0 || length == 0 {
        return Err(format!("{}: no tokens", path.display()));
    }
    Ok((tokens, length))
}

/// The layers a run rounds, as Java's `--layers`: a count, which a count at or above the model's layers makes a run
/// of the whole model.
pub(crate) fn plan_for(requested: usize, model_layers: usize) -> Plan {
    Plan::new(requested.min(model_layers))
}

/// Said after a run of the whole model: the heads and the MTP block are not rounded by this build.
const HEADS_NOT_ROUNDED: &str = "calibrate: the layers are rounded; this build does not round the output head, the \
draft head or the MTP block, so the output is not complete\n";

/// Runs `calibrate` with the flags in `args`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread opens the checkpoint, waits for the run and ends it"
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
    let library = flags.library.clone().unwrap_or_else(|| PathBuf::from(DEFAULT_LIBRARY));
    let words: Vec<OsString> = flags
        .config
        .iter()
        .flat_map(|path| [OsString::from("--config"), path.clone().into_os_string()])
        .collect();
    let settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => {
            return Outcome {
                code: EXIT_USAGE,
                message: format!("{errors}\n"),
            };
        }
    };
    let (tokens, length) = match read_tokens(&flags.tokens, flags.sequences) {
        Ok(read) => read,
        Err(message) => return failed(message),
    };
    let rig = match Rig::start(&settings) {
        Ok(rig) => rig,
        Err(message) => return failed(message),
    };
    let result = calibrated(&rig, &flags, &library, tokens, length);
    rig.stop();
    match result {
        Ok(whole) => Outcome {
            code: EXIT_OK,
            message: if whole {
                HEADS_NOT_ROUNDED.to_owned()
            } else {
                String::new()
            },
        },
        Err(message) => failed(message),
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread opens the checkpoint and waits for the one completion of the run"
)]
fn calibrated(rig: &Rig, flags: &Flags, library: &Path, tokens: Vec<i32>, length: usize) -> Result<bool, String> {
    let (sender, receiver) = mpsc::channel();
    let sent = sender.clone();
    Safetensors::open(
        &rig.io,
        &flags.model,
        Arc::new(move |opened| {
            let _ = sent.send(opened);
        }),
    )
    .map_err(|error| error.to_string())?;
    let store = Arc::new(
        receiver
            .recv()
            .map_err(|_| "the checkpoint was not opened".to_owned())?
            .map_err(|error| error.to_string())?,
    );
    let (sender, receiver) = mpsc::channel();
    Qwen38Objects::open(
        store,
        &flags.model,
        Arc::new(move |opened| {
            let _ = sender.send(opened);
        }),
    )
    .map_err(|error| error.to_string())?;
    let objects = Arc::new(
        receiver
            .recv()
            .map_err(|_| "the checkpoint's objects were not opened".to_owned())?
            .map_err(|error| error.to_string())?,
    );
    let model_layers = objects.config.layers;
    let recipe = Recipe::new(flags.quantization, flags.compressed, flags.output_head.as_deref())
        .map_err(|error| error.to_string())?;
    let search = Nvfp4Search::new(flags.nvfp4_above, flags.nvfp4_scale_max).map_err(|error| error.to_string())?;
    let sequences = tokens.len() / length;
    let tokens_path = fs::canonicalize(&flags.tokens).unwrap_or_else(|_| flags.tokens.clone());
    let model_path = fs::canonicalize(&flags.model).unwrap_or_else(|_| flags.model.clone());
    let key = resume_key(&[
        &tokens_path.display().to_string(),
        &sequences.to_string(),
        &length.to_string(),
        &format!("{recipe:?}"),
        &flags.act_order.to_string(),
        &flags.damp.to_string(),
        &model_path.display().to_string(),
        &format!("{search:?}"),
    ]);
    fs::create_dir_all(&flags.out).map_err(|error| format!("{}: {error}", flags.out.display()))?;
    let mut run = RunSettings::new(flags.out.clone(), recipe, key, tokens, length);
    run.search = search;
    run.damp = flags.damp;
    run.act_order = flags.act_order;
    run.batch = flags.batch;
    let (calibration, completions) = Calibration::open(
        run,
        objects,
        library,
        &flags.math_directory,
        Arc::clone(&rig.io),
        Arc::clone(&rig.host),
    )
    .map_err(|error| error.to_string())?;
    for source in completions {
        rig.attach(source)?;
    }
    let plan = plan_for(flags.layers, model_layers);
    let whole = flags.layers >= model_layers;
    let (sender, receiver) = mpsc::channel();
    drive(
        Arc::new(calibration),
        plan,
        Box::new(move |result| {
            let _ = sender.send(result);
        }),
    );
    receiver
        .recv()
        .map_err(|_| "the run ended without a result".to_owned())?
        .map_err(|error| error.to_string())?;
    Ok(whole)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn words(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    const BASE: [&str; 8] = [
        "--model",
        "m",
        "--quantization",
        "nvfp4",
        "--tokens",
        "t.npy",
        "--out",
        "o",
    ];

    fn with(extra: &[&str]) -> Vec<String> {
        words(&BASE.iter().chain(extra).copied().collect::<Vec<_>>())
    }

    #[test]
    fn the_defaults_are_javas() {
        let flags = flags(&with(&[])).unwrap();
        assert_eq!((flags.batch, flags.layers, flags.sequences), (8, 64, 0));
        assert_eq!(flags.damp, 0.01);
        assert!(!flags.act_order && !flags.compressed);
        assert_eq!((flags.nvfp4_above, flags.nvfp4_scale_max), (2, 448.0));
    }

    #[test]
    fn the_flags_set_what_they_name() {
        let flags = flags(&with(&[
            "--layers",
            "2",
            "--batch",
            "4",
            "--sequences",
            "16",
            "--act-order",
            "--compressed",
            "--damp",
            "0.05",
            "--skip-mtp",
            "--draft-ids-from",
            "a",
        ]))
        .unwrap();
        assert_eq!((flags.layers, flags.batch, flags.sequences), (2, 4, 16));
        assert!(flags.act_order && flags.compressed);
        assert_eq!(flags.damp, 0.05);
    }

    #[test]
    fn the_default_layers_run_the_whole_64_layer_model_and_a_count_below_it_a_part() {
        let default = flags(&with(&[])).unwrap();
        assert_eq!(plan_for(default.layers, 64), Plan::new(64));
        assert_eq!(plan_for(10, 64).layers, 10);
        assert_eq!(plan_for(64, 40).layers, 40);
    }

    #[test]
    fn every_trellis_option_is_refused_by_name() {
        for option in [
            "--bits",
            "--rounding-lanes",
            "--viterbi-scratch-mib",
            "--memory-reserve-mib",
            "--write-buffer-mib",
        ] {
            let message = flags(&with(&[option, "3"])).unwrap_err();
            assert!(message.contains(option) && message.contains("trellis"), "{message}");
        }
        assert!(flags(&with(&["--format", "trellis"])).unwrap_err().contains("trellis"));
        let trellis = words(&[
            "--model",
            "m",
            "--quantization",
            "trellis",
            "--tokens",
            "t",
            "--out",
            "o",
        ]);
        assert!(flags(&trellis).unwrap_err().contains("trellis"));
    }

    #[test]
    fn a_wrong_flag_is_refused_before_anything_runs() {
        assert!(flags(&words(&["--model", "m"])).unwrap_err().contains("needs"));
        assert!(flags(&with(&["--bogus", "1"])).unwrap_err().contains("unknown"));
        assert!(flags(&with(&["--layers", "0"])).is_err());
        assert!(flags(&with(&["--layers", "x"])).is_err());
        assert!(flags(&with(&["--batch", "0"])).is_err());
        assert!(flags(&with(&["--batch", "65"])).is_err());
        assert!(flags(&with(&["--damp", "0"])).is_err());
        assert!(flags(&with(&["--nvfp4-scale-max", "449"])).is_err());
        assert!(flags(&with(&["--layers"])).is_err());
    }
}
