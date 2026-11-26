//! The `cost` command: times each option's linear launch per projection shape and row count on the card, through the
//! engine's own linear formats, and writes the cost table `allocate --costs` reads.
//!
//! The measurement is [`measure`]; this command parses the flags, opens the sources, checks each against its option,
//! runs the measurement on a [`DeviceBench`] and writes the table through a `.part` file renamed into place, so a run
//! that fails leaves no output (an earlier output stays as it was).

mod bench;

use std::collections::BTreeMap;
use std::fs::{File, remove_file, rename};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_quantization::assemble::SourceSet;
use euhedral_inference_quantization::mixed::{
    CostTable, DEFAULT_REPEATS, DEFAULT_ROTATE_MIB, DEFAULT_ROWS, MixedError, Plan, QuantOption, Timed, check_format,
    measure,
};
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use self::bench::DeviceBench;
use crate::rig::Rig;
use crate::score::default_library;
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

const MIB: u64 = 1 << 20;

/// What the command was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Flags {
    sources: BTreeMap<QuantOption, PathBuf>,
    base: QuantOption,
    out: PathBuf,
    rows: Vec<u32>,
    repeats: u32,
    rotate_mib: u64,
    config: Option<PathBuf>,
    library: Option<PathBuf>,
}

/// The flags in `args`: `--source OPTION=ARTIFACT` (repeated), `--base OPTION`, `--out FILE`, and optionally
/// `--rows A,B,...`, `--repeats N`, `--rotate-mib N`, `--config YAML` and `--cuda-library FILE`.
fn parse(args: &[String]) -> Result<Flags, String> {
    let (mut sources, mut base, mut out) = (BTreeMap::new(), None, None);
    let (mut rows, mut repeats, mut rotate_mib) = (DEFAULT_ROWS.to_vec(), DEFAULT_REPEATS, DEFAULT_ROTATE_MIB);
    let (mut config, mut library) = (None, None);
    let mut words = args.iter();
    while let Some(flag) = words.next() {
        let value = words.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--source" => {
                let (key, path) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--source {value} is not OPTION=ARTIFACT"))?;
                let option = QuantOption::of(key).map_err(|error| error.to_string())?;
                if sources.insert(option, PathBuf::from(path)).is_some() {
                    return Err(format!("--source {key} is given twice"));
                }
            }
            "--base" => base = Some(QuantOption::of(value).map_err(|error| error.to_string())?),
            "--out" => out = Some(PathBuf::from(value)),
            "--rows" => {
                rows = value
                    .split(',')
                    .map(|count| count.trim().parse::<u32>().ok().filter(|rows| *rows >= 1))
                    .collect::<Option<Vec<u32>>>()
                    .ok_or_else(|| format!("--rows {value} is not a comma-separated list of counts of at least 1"))?;
            }
            "--repeats" => {
                repeats = value
                    .parse::<u32>()
                    .ok()
                    .filter(|repeats| *repeats >= 1)
                    .ok_or_else(|| format!("--repeats {value} is not a count of at least 1"))?;
            }
            "--rotate-mib" => {
                rotate_mib = value
                    .parse::<u64>()
                    .map_err(|_| format!("--rotate-mib {value} is not a count"))?;
            }
            "--config" => config = Some(PathBuf::from(value)),
            "--cuda-library" | "--library" => library = Some(PathBuf::from(value)),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    let base = base.ok_or("--base is required")?;
    if !sources.contains_key(&base) {
        return Err(format!("no --source for the base {base}"));
    }
    Ok(Flags {
        sources,
        base,
        out: out.ok_or("--out is required")?,
        rows,
        repeats,
        rotate_mib,
        config,
        library,
    })
}

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n"),
    }
}

/// Runs the command on `args`, the words after its name.
pub fn run(args: &[String]) -> Outcome {
    let flags = match parse(args) {
        Ok(flags) => flags,
        Err(message) => return usage_error(&message),
    };
    let Some(library) = flags.library.clone().or_else(default_library) else {
        return usage_error("cost needs --cuda-library on this platform");
    };
    let words: Vec<_> = flags
        .config
        .iter()
        .flat_map(|path| ["--config".into(), path.clone().into_os_string()])
        .collect();
    let settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    match measured(&settings, &flags, &library).and_then(|table| write(&flags.out, &table)) {
        Ok(()) => Outcome {
            code: EXIT_OK,
            message: String::new(),
        },
        Err(message) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("cost failed: {message}\n"),
        },
    }
}

/// Opens the sources on the host runtime, then measures on the card on the calling thread.
fn measured(settings: &Settings, flags: &Flags, library: &Path) -> Result<CostTable, String> {
    let rig = Rig::start(settings)?;
    let chunk_bytes = usize::try_from(settings.host.copy_chunk_bytes.bytes()).unwrap_or(usize::MAX);
    let (sender, receiver) = mpsc::channel();
    let started = SourceSet::open(
        &rig.io,
        &flags.sources,
        flags.base,
        chunk_bytes,
        Arc::new(move |opened| {
            // The receiver is gone only when the run was abandoned.
            let _ = sender.send(opened);
        }),
    );
    let opened = match started {
        Ok(()) => receiver
            .recv()
            .map_err(|_| "the sources were not opened".to_owned())
            .and_then(|opened| opened.map_err(|error| error.to_string())),
        Err(error) => Err(error.to_string()),
    };
    rig.stop();
    let sources = opened?;
    let plan = Plan::new(
        sources.options(),
        flags.rows.clone(),
        flags.repeats,
        flags.rotate_mib.saturating_mul(MIB),
    )
    .map_err(|error| error.to_string())?;
    check_sources(&sources).map_err(|error| error.to_string())?;
    let library = CudaLibrary::load(library).map_err(|error| format!("the CUDA library: {error}"))?;
    let mut bench = DeviceBench::open(&library, settings, &sources).map_err(|error| error.to_string())?;
    measure(&sources, &plan, &mut bench, &mut report).map_err(|error| error.to_string())
}

/// Each source stores every projection of the base in the format its option names.
fn check_sources(sources: &SourceSet) -> Result<(), MixedError> {
    if sources.projections().is_empty() {
        return Err(MixedError::Invalid("the base has no projection".to_owned()));
    }
    for option in sources.options() {
        for name in sources.projections() {
            let entry = sources
                .descriptor(option, name)
                .map_err(|error| MixedError::Invalid(error.to_string()))?;
            check_format(option, entry.format())?;
        }
    }
    Ok(())
}

#[allow(clippy::print_stderr, reason = "the command's progress, one line a measurement")]
fn report(timed: &Timed) {
    let key = timed.key;
    eprintln!(
        "{} {}x{} rows {}: {:.3} us",
        key.option, key.shape.input, key.shape.output, key.rows, timed.micros
    );
}

/// Writes `table` to `out` through `out.part`, which a failure removes: `out` is complete or as it was.
fn write(out: &Path, table: &CostTable) -> Result<(), String> {
    let mut part = out.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    let written = File::create(&part)
        .and_then(|mut file| {
            file.write_all(table.render().as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| rename(&part, out));
    written.map_err(|error| {
        // The failure being reported is the write's; a part that cannot be removed is left for the next run.
        let _ = remove_file(&part);
        format!("{}: {error}", out.display())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use std::collections::BTreeMap;
    use std::fs::{create_dir, read_dir, read_to_string, write as write_file};

    use euhedral_inference_quantization::mixed::{Key, Shape};

    use super::*;

    fn words(words: &[&str]) -> Vec<String> {
        words.iter().map(|word| word.to_string()).collect()
    }

    const MINIMAL: [&str; 6] = ["--source", "nvfp4=a.edrl", "--base", "nvfp4", "--out", "COSTS.json"];

    #[test]
    fn the_flags_default_to_the_rows_repeats_and_rotation_of_the_command() {
        let flags = parse(&words(&MINIMAL)).unwrap();
        assert_eq!(flags.rows, DEFAULT_ROWS.to_vec());
        assert_eq!((flags.repeats, flags.rotate_mib), (DEFAULT_REPEATS, DEFAULT_ROTATE_MIB));
        assert_eq!(flags.sources.get(&QuantOption::Nvfp4), Some(&PathBuf::from("a.edrl")));
    }

    #[test]
    fn the_flags_set_the_rows_repeats_and_rotation() {
        let mut args = words(&MINIMAL);
        args.extend(words(&["--rows", "1, 4,512", "--repeats", "5", "--rotate-mib", "0"]));
        let flags = parse(&args).unwrap();
        assert_eq!((flags.rows, flags.repeats, flags.rotate_mib), (vec![1, 4, 512], 5, 0));
    }

    #[test]
    fn an_unknown_option_and_malformed_flags_are_refused() {
        let with = |extra: &[&str]| {
            let mut args = words(&MINIMAL);
            args.extend(words(extra));
            parse(&args)
        };
        assert!(parse(&words(&["--source", "nvfp4=a", "--base", "trellis-9.0", "--out", "o"])).is_err());
        assert!(parse(&words(&["--source", "int8=a", "--base", "nvfp4", "--out", "o"])).is_err());
        assert!(parse(&words(&["--source", "nvfp4", "--base", "nvfp4", "--out", "o"])).is_err());
        assert!(with(&["--source", "nvfp4=b"]).is_err());
        assert!(with(&["--rows", "1,0"]).is_err());
        assert!(with(&["--rows", ""]).is_err());
        assert!(with(&["--repeats", "0"]).is_err());
        assert!(with(&["--bogus", "1"]).is_err());
        assert!(with(&["--repeats"]).is_err());
        assert!(parse(&words(&["--source", "nvfp4=a", "--base", "trellis-3.0", "--out", "o"])).is_err());
        assert!(parse(&words(&["--source", "nvfp4=a", "--base", "nvfp4"])).is_err());
    }

    fn table() -> CostTable {
        let key = Key {
            shape: Shape { input: 64, output: 128 },
            option: QuantOption::Nvfp4,
            rows: 4,
        };
        CostTable::new(BTreeMap::from([(key, 12.5)]), 0.0, BTreeMap::new())
    }

    #[test]
    fn a_written_table_is_whole_under_its_name_and_leaves_no_part() {
        let directory = tempfile::tempdir().unwrap();
        let out = directory.path().join("COSTS.json");
        write(&out, &table()).unwrap();
        let read = CostTable::parse(&read_to_string(&out).unwrap()).unwrap();
        assert_eq!(read, table());
        let names: Vec<_> = read_dir(directory.path()).unwrap().collect();
        assert_eq!(names.len(), 1);
    }

    #[test]
    fn a_failed_write_leaves_no_output_and_no_part_and_keeps_an_earlier_output() {
        let directory = tempfile::tempdir().unwrap();
        // The destination is a directory: the rename onto it fails after the part was written.
        let out = directory.path().join("COSTS.json");
        create_dir(&out).unwrap();
        write_file(out.join("keep"), b"earlier").unwrap();
        assert!(write(&out, &table()).is_err());
        assert!(out.join("keep").is_file());
        let mut part = out.as_os_str().to_owned();
        part.push(".part");
        assert!(!Path::new(&part).exists());
    }
}
