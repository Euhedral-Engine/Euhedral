//! The `calibrate` command for GPTQ: the BF16 checkpoint run layer by layer over calibration tokens, each layer's
//! linears rounded from the Hessians of their inputs and written to the output directory, with `calibration.json`
//! listing the files saved (the resume file among them) and how far the run got. The output directory must be empty
//! or hold the same run's manifest with every file it lists (the run resumes from what it lists); another run's
//! manifest, or this run's with listed files missing, is refused unless `--fresh`, which removes only the files the
//! manifest lists and starts over; any other content is refused, and nothing is removed or written before the
//! checkpoint and the device have opened. The trellis arm is not part of this build.
//!
//! A run of every layer then rounds, as Java does, the MTP block (unless `--skip-mtp`), `mtp/input_projection`, the
//! output head and the draft head, and lists each in the manifest as it is written; the resume file is dropped after
//! the last. The draft head keeps the rows of the output head that `text/draft_head_token_ids` of the artifact named
//! by `--draft-ids-from` select, so a run of every layer needs that flag. A run of fewer layers than the model's stops
//! after them and exits 0, keeping its resume file. A head that fails leaves the resume file, and the next run starts
//! at the heads.

use std::env::consts;
use std::ffi::OsString;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use euhedral_inference_quantization::calibrate::device::{Calibration, Settings as RunSettings};
use euhedral_inference_quantization::calibrate::draft_ids::read_draft_ids;
use euhedral_inference_quantization::calibrate::manifest::inspect;
use euhedral_inference_quantization::calibrate::plan::{resume_key, tokens_digest};
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
/// The most rounding lanes: each holds its own device and pinned buffers.
const MOST_LANES: usize = 64;
/// The most mebibytes a size flag takes: its bytes fit in 64 bits.
const MOST_MIB: u64 = u64::MAX >> 20;
/// The parts of the resume key, in order, for the refusal that names what differs.
const KEY_FIELDS: [&str; 10] = [
    "the tokens file",
    "the sequences",
    "the sequence length",
    "the recipe",
    "act-order",
    "damp",
    "the model",
    "the NVFP4 search",
    "the batch",
    "the tokens' contents",
];
/// What the trellis arm takes: refused by name.
const TRELLIS_OPTIONS: [&str; 3] = ["--bits", "--viterbi-scratch-mib", "--write-buffer-mib"];
/// What the command line accepts, for the message that refuses another option.
const OPTIONS: &str = "--model, --quantization, --compressed, --output-head, --tokens, --draft-ids-from, --out, \
--act-order, --damp, --batch, --sequences, --layers, --skip-mtp, --nvfp4-above, --nvfp4-scale-max, --format, \
--config, --library (or --cuda-library), --cuda-libraries, --rounding-lanes, --memory-reserve-mib, --staging-mib, \
--lane-pinned-mib, --upload-chunk-mib, --fresh";

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
    pub math_directory: Option<PathBuf>,
    /// Linears rounded at once (Java's `--rounding-lanes`); none takes the default.
    pub rounding_lanes: Option<usize>,
    /// Device MiB left for what the planner does not account (Java's `--memory-reserve-mib`); none sizes it from the
    /// card.
    pub memory_reserve_mib: Option<u64>,
    /// Pinned MiB the layers are read into.
    pub staging_mib: Option<u64>,
    /// Pinned MiB the GPTQ lanes stage through.
    pub lane_pinned_mib: Option<u64>,
    /// MiB of one upload copy.
    pub upload_chunk_mib: Option<u64>,
    /// Remove the files the manifest in `--out` lists and start over, instead of resuming or refusing.
    pub fresh: bool,
    /// Leave the MTP block out; the draft head is rounded under the output head's Hessian.
    pub skip_mtp: bool,
    /// The artifact whose `text/draft_head_token_ids` name the output head's rows the draft head keeps; a run of
    /// every layer needs it.
    pub draft_ids_from: Option<PathBuf>,
}

fn trellis_refusal(option: &str) -> String {
    format!("calibrate: {option} is a trellis option, and the trellis arm is not part of this build")
}

pub(crate) fn flags(args: &[String]) -> Result<Flags, String> {
    let (mut model, mut quantization, mut output_head, mut tokens, mut out) = (None, None, None, None, None);
    let (mut compressed, mut act_order, mut fresh, mut skip_mtp) = (false, false, false, false);
    let mut draft_ids_from = None;
    let (mut damp, mut batch, mut sequences, mut layers) = (DEFAULT_DAMP, DEFAULT_BATCH, 0, DEFAULT_LAYERS);
    let (mut above, mut scale_max) = (Nvfp4Search::DEFAULT.above(), Nvfp4Search::DEFAULT.scale_max());
    let (mut config, mut library) = (None, None);
    let mut math_directory = None;
    let (mut lanes, mut reserve, mut staging, mut lane_pinned, mut chunk) = (None, None, None, None, None);
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
            "--fresh" => {
                fresh = true;
                continue;
            }
            "--skip-mtp" => {
                skip_mtp = true;
                continue;
            }
            _ => {}
        }
        if TRELLIS_OPTIONS.contains(&arg.as_str()) {
            return Err(trellis_refusal(arg));
        }
        let value = rest.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let number = || {
            value
                .parse::<usize>()
                .map_err(|_| format!("{arg} needs a whole number, not {value}"))
        };
        let mebibytes = || {
            value
                .parse::<u64>()
                .ok()
                .filter(|mib| (1..=MOST_MIB).contains(mib))
                .ok_or_else(|| format!("{arg} needs a whole number of MiB from 1 to {MOST_MIB}, not {value}"))
        };
        let real = || {
            value
                .parse::<f32>()
                .map_err(|_| format!("{arg} needs a number, not {value}"))
        };
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
            "--draft-ids-from" => draft_ids_from = Some(PathBuf::from(value)),
            "--out" => out = Some(PathBuf::from(value)),
            "--damp" => damp = real()?,
            "--batch" => batch = number()?,
            "--sequences" => sequences = number()?,
            "--layers" => layers = number()?,
            "--nvfp4-above" => {
                above = value
                    .parse::<i32>()
                    .map_err(|_| format!("{arg} needs a whole number, not {value}"))?;
            }
            "--nvfp4-scale-max" => scale_max = real()?,
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" | "--library" => library = Some(PathBuf::from(value)),
            "--cuda-libraries" => math_directory = Some(PathBuf::from(value)),
            "--rounding-lanes" => {
                lanes = Some(
                    number()
                        .ok()
                        .filter(|lanes| (1..=MOST_LANES).contains(lanes))
                        .ok_or_else(|| format!("{arg} is 1 to {MOST_LANES}, not {value}"))?,
                );
            }
            "--memory-reserve-mib" => reserve = Some(mebibytes()?),
            "--staging-mib" => staging = Some(mebibytes()?),
            "--lane-pinned-mib" => lane_pinned = Some(mebibytes()?),
            "--upload-chunk-mib" => chunk = Some(mebibytes()?),
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
        rounding_lanes: lanes,
        memory_reserve_mib: reserve,
        staging_mib: staging,
        lane_pinned_mib: lane_pinned,
        upload_chunk_mib: chunk,
        fresh,
        skip_mtp,
        draft_ids_from,
    })
}

/// The library the build puts the CUDA layer in, relative to the repository root, where this platform has one; other
/// platforms name it with `--cuda-library`.
fn default_library() -> Option<PathBuf> {
    (consts::OS == "linux" && consts::ARCH == "x86_64")
        .then(|| PathBuf::from("build/native/linux-x64/lib/libeuhedral_cuda.so"))
}

/// Where the CUDA math libraries are on a platform that has them in one place; other platforms name the directory with
/// `--cuda-libraries`.
fn default_math_directory() -> Option<PathBuf> {
    (consts::OS == "linux").then(|| PathBuf::from("/usr/local/cuda/lib64"))
}

/// What the flag named, else the platform's default, else the refusal that names the flag.
fn resolved(named: Option<PathBuf>, default: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    named
        .or(default)
        .ok_or_else(|| format!("this platform has no default for {flag}: pass it"))
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

/// The draft head's token ids from the artifact at `path`, each in `[0, vocabulary)`.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread reads the ids before the run starts"
)]
pub(crate) fn draft_ids(path: &Path, vocabulary: u64) -> Result<Vec<i32>, String> {
    let shown = path.display();
    let file = fs::File::open(path).map_err(|error| format!("{shown}: {error}"))?;
    let length = file.metadata().map_err(|error| format!("{shown}: {error}"))?.len();
    let read = |range: Range<u64>| -> Result<Vec<u8>, String> {
        let size = usize::try_from(range.end - range.start).map_err(|_| "a region is too large".to_owned())?;
        let mut bytes = vec![0; size];
        let mut file = &file;
        file.seek(SeekFrom::Start(range.start))
            .and_then(|_| file.read_exact(&mut bytes))
            .map_err(|error| error.to_string())?;
        Ok(bytes)
    };
    read_draft_ids(length, read, vocabulary).map_err(|why| format!("{shown}: {why}"))
}

/// `mib` mebibytes as bytes of the flag `flag`, refusing a size that does not fit.
fn mib_bytes(flag: &str, mib: u64) -> Result<u64, String> {
    mib.checked_mul(1 << 20)
        .ok_or_else(|| format!("{flag} {mib} is too many bytes"))
}

/// The plan of a run of the model's `model_layers` layers that rounds `requested` of them, as Java's `--layers` is a
/// count: a count at or above the model's makes a run of the whole model, which finishes; a smaller one stops early.
pub(crate) fn plan_for(requested: usize, model_layers: usize) -> Plan {
    let mut plan = Plan::new(model_layers);
    plan.stop_after = requested;
    plan
}

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
    let (library, math_directory) = match (
        resolved(flags.library.clone(), default_library(), "--cuda-library"),
        resolved(
            flags.math_directory.clone(),
            default_math_directory(),
            "--cuda-libraries",
        ),
    ) {
        (Ok(library), Ok(math)) => (library, math),
        (Err(message), _) | (_, Err(message)) => {
            return Outcome {
                code: EXIT_USAGE,
                message: format!("{message}\n"),
            };
        }
    };
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
    let result = calibrated(&rig, &flags, (&library, &math_directory), tokens, length);
    rig.stop();
    match result {
        Ok(()) => Outcome {
            code: EXIT_OK,
            message: String::new(),
        },
        Err(message) => failed(message),
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "a binary: its main thread opens the checkpoint and waits for the one completion of the run"
)]
fn calibrated(
    rig: &Rig,
    flags: &Flags,
    (library, math_directory): (&Path, &Path),
    tokens: Vec<i32>,
    length: usize,
) -> Result<(), String> {
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
    let plan = plan_for(flags.layers, model_layers);
    let draft_ids = match (&flags.draft_ids_from, plan.stops_early()) {
        (Some(path), false) => draft_ids(path, objects.config.vocabulary)?,
        (None, false) => {
            return Err("a run of every layer rounds the draft head: it needs --draft-ids-from".to_owned());
        }
        (_, true) => Vec::new(),
    };
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
        &flags.batch.to_string(),
        &tokens_digest(&tokens),
    ]);
    let prepared =
        inspect(&flags.out, &key, model_layers, &KEY_FIELDS, flags.fresh).map_err(|error| error.to_string())?;
    let mut run = RunSettings::new(flags.out.clone(), recipe, prepared.manifest().clone(), tokens, length);
    run.search = search;
    run.damp = flags.damp;
    run.act_order = flags.act_order;
    run.batch = flags.batch;
    run.skip_mtp = flags.skip_mtp;
    run.draft_ids = draft_ids;
    if let Some(lanes) = flags.rounding_lanes {
        run.lanes = lanes;
    }
    if let Some(mib) = flags.memory_reserve_mib {
        run.device_reserve = Some(mib_bytes("--memory-reserve-mib", mib)?);
    }
    if let Some(mib) = flags.staging_mib {
        run.staging_bytes = mib_bytes("--staging-mib", mib)?;
    }
    if let Some(mib) = flags.lane_pinned_mib {
        run.lane_pinned_bytes = mib_bytes("--lane-pinned-mib", mib)?;
    }
    if let Some(mib) = flags.upload_chunk_mib {
        run.chunk_bytes = usize::try_from(mib_bytes("--upload-chunk-mib", mib)?)
            .map_err(|_| format!("--upload-chunk-mib {mib} is too many bytes"))?;
    }
    let (calibration, completions) = Calibration::open(
        run,
        objects,
        library,
        math_directory,
        Arc::clone(&rig.io),
        Arc::clone(&rig.host),
    )
    .map_err(|error| error.to_string())?;
    for source in completions {
        rig.attach(source)?;
    }
    // Everything the run needs is open: only now does it change the directory.
    prepared.apply().map_err(|error| error.to_string())?;
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
    Ok(())
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    reason = "a test: a failed unwrap is the failure, and it writes and reads its files on its own thread"
)]
mod tests {
    use super::*;
    use euhedral_inference_quantization::npy::encode_elements;

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
        assert!(flags.skip_mtp);
        assert_eq!(flags.draft_ids_from, Some(PathBuf::from("a")));
        let plain = super::flags(&with(&[])).unwrap();
        assert!(!plain.skip_mtp && plain.draft_ids_from.is_none());
    }

    #[test]
    fn the_default_layers_run_the_whole_64_layer_model_and_a_count_below_it_a_part() {
        let default = flags(&with(&[])).unwrap();
        let whole = plan_for(default.layers, 64);
        assert_eq!((whole.layers, whole.stop()), (64, 64));
        assert!(
            !whole.stops_early(),
            "the default finishes: it is a run of the whole model"
        );
        let part = plan_for(10, 64);
        assert_eq!((part.layers, part.stop()), (64, 10));
        assert!(part.stops_early(), "a run of 10 layers never finishes");
        assert!(
            !plan_for(100, 64).stops_early(),
            "a count past the model's is the whole model"
        );
        assert!(!plan_for(64, 64).stops_early());
    }

    #[test]
    fn the_memory_and_lane_flags_are_accepted_and_sized_by_the_platform_when_absent() {
        let defaults = flags(&with(&[])).unwrap();
        assert!(defaults.rounding_lanes.is_none() && defaults.memory_reserve_mib.is_none());
        assert!(defaults.library.is_none() && defaults.math_directory.is_none());
        let named = flags(&with(&[
            "--rounding-lanes",
            "2",
            "--memory-reserve-mib",
            "512",
            "--staging-mib",
            "256",
            "--lane-pinned-mib",
            "128",
            "--upload-chunk-mib",
            "16",
        ]))
        .unwrap();
        assert_eq!((named.rounding_lanes, named.memory_reserve_mib), (Some(2), Some(512)));
        assert_eq!(
            (named.staging_mib, named.lane_pinned_mib, named.upload_chunk_mib),
            (Some(256), Some(128), Some(16))
        );
        assert!(flags(&with(&["--rounding-lanes", "0"])).is_err());
        assert!(flags(&with(&["--memory-reserve-mib", "x"])).is_err());
    }

    #[test]
    fn a_flag_that_is_refused_says_what_it_was_given_and_sizes_have_bounds() {
        let why = flags(&with(&["--batch", "many"])).unwrap_err();
        assert!(why.contains("many"), "{why}");
        let why = flags(&with(&["--rounding-lanes", "65"])).unwrap_err();
        assert!(why.contains("65") && why.contains("64"), "{why}");
        assert!(flags(&with(&["--rounding-lanes", "64"])).is_ok());
        let huge = (MOST_MIB + 1).to_string();
        assert!(
            flags(&with(&["--staging-mib", huge.as_str()]))
                .unwrap_err()
                .contains(&huge)
        );
        assert_eq!(mib_bytes("--x", MOST_MIB).unwrap(), MOST_MIB << 20);
        assert!(mib_bytes("--x", MOST_MIB + 1).unwrap_err().contains("--x"));
    }

    #[test]
    fn fresh_is_a_switch_and_off_unless_named() {
        assert!(!flags(&with(&[])).unwrap().fresh);
        assert!(flags(&with(&["--fresh"])).unwrap().fresh);
    }

    #[test]
    fn a_library_is_what_the_flag_names_else_the_platforms_default_else_a_refusal() {
        let named = Some(PathBuf::from("a"));
        let default = Some(PathBuf::from("b"));
        assert_eq!(
            resolved(named.clone(), default.clone(), "--x").unwrap(),
            PathBuf::from("a")
        );
        assert_eq!(resolved(None, default, "--x").unwrap(), PathBuf::from("b"));
        assert!(resolved(named.clone(), None, "--x").is_ok());
        assert!(
            resolved(None, None, "--cuda-library")
                .unwrap_err()
                .contains("--cuda-library")
        );
    }

    #[test]
    fn every_trellis_option_is_refused_by_name() {
        for option in ["--bits", "--viterbi-scratch-mib", "--write-buffer-mib"] {
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

    fn tokens_file(directory: &Path, shape: &[u64], values: &[i32]) -> PathBuf {
        let path = directory.join("tokens.npy");
        fs::write(&path, encode_elements(shape, values).unwrap()).unwrap();
        path
    }

    #[test]
    fn the_tokens_are_the_first_sequences_of_the_file_or_all_of_them() {
        let directory = tempfile::tempdir().unwrap();
        let path = tokens_file(directory.path(), &[3, 2], &[1, 2, 3, 4, 5, 6]);
        assert_eq!(read_tokens(&path, 0).unwrap(), (vec![1, 2, 3, 4, 5, 6], 2));
        assert_eq!(read_tokens(&path, 2).unwrap(), (vec![1, 2, 3, 4], 2));
        assert_eq!(
            read_tokens(&path, 9).unwrap(),
            (vec![1, 2, 3, 4, 5, 6], 2),
            "a limit past the end takes all"
        );
    }

    #[test]
    fn a_tokens_file_that_is_not_sequences_of_tokens_is_refused() {
        let directory = tempfile::tempdir().unwrap();
        let flat = tokens_file(directory.path(), &[4], &[1, 2, 3, 4]);
        assert!(read_tokens(&flat, 0).unwrap_err().contains("[sequences, length]"));
        let empty = tokens_file(directory.path(), &[0, 3], &[]);
        assert!(read_tokens(&empty, 0).unwrap_err().contains("no tokens"));
        let none_long = tokens_file(directory.path(), &[2, 0], &[]);
        assert!(read_tokens(&none_long, 0).unwrap_err().contains("no tokens"));
        assert!(read_tokens(&directory.path().join("missing.npy"), 0).is_err());
        fs::write(directory.path().join("bad.npy"), b"not a npy").unwrap();
        assert!(read_tokens(&directory.path().join("bad.npy"), 0).is_err());
    }
}
