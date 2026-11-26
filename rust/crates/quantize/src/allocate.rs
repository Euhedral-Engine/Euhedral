//! The `allocate` command: the mixed search over calibrated sources. It proposes allocations, assembles and scores
//! each against the references on the card, and writes the quality and speed front to `--out`, resumable from the one
//! `--state` file.
//!
//! The search is [`Search`]; this command parses and checks the flags, reads the inputs, opens the sources, the state
//! and the engine's settings, and hands the search what it runs on: a [`FileLedger`] for the files, a [`LakeFan`] for
//! the host tasks, [`Device`] for scoring and [`Resident`] for the bytes of an assembled candidate. Everything it can
//! refuse is refused before the card is opened.

mod device;
mod fan;
mod flags;
mod ledger;

use std::fs::{create_dir_all, read_to_string};
use std::path::Path;
use std::sync::{Arc, mpsc};

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_quantization::assemble::{SourceSet, view};
use euhedral_inference_quantization::mixed::search::{
    Assembly, Outcome as Found, Search, SearchFailure, Settings as SearchSettings, World,
};
use euhedral_inference_quantization::mixed::{
    Allocation, CostTable, MixedError, PlannerFeasibility, Priors, read_seeds,
};
use euhedral_inference_quantization::reference::DEFAULT_BATCH;
use euhedral_inference_settings::{Settings, load_for_tool, process_env};

use self::device::Device;
use self::fan::LakeFan;
use self::flags::{Flags, parse};
use self::ledger::{FileLedger, read_measured};
use crate::rig::Rig;
use crate::score::{Memory, Pass, default_library, dimensions, engine_settings};
use crate::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, Outcome};

const MIB: u64 = 1 << 20;

/// What the files the flags name hold.
struct Inputs {
    priors: Priors,
    costs: CostTable,
    seeds: Vec<Allocation>,
}

fn usage_error(message: &str) -> Outcome {
    Outcome {
        code: EXIT_USAGE,
        message: format!("{message}\n"),
    }
}

fn read(path: &Path, flag: &str) -> Result<String, String> {
    read_to_string(path).map_err(|error| format!("{flag} {}: {error}", path.display()))
}

/// The priors, the cost table and the seeds the flags name, read and parsed.
fn inputs(flags: &Flags) -> Result<Inputs, String> {
    let priors = Priors::parse(&read(&flags.priors, "--priors")?).map_err(|error| format!("--priors: {error}"))?;
    let costs = CostTable::parse(&read(&flags.costs, "--costs")?).map_err(|error| format!("--costs: {error}"))?;
    let seeds = match &flags.seed_file {
        Some(path) => read_seeds(&read(path, "--seed-file")?).map_err(|error| format!("--seed-file: {error}"))?,
        None => Vec::new(),
    };
    Ok(Inputs { priors, costs, seeds })
}

/// What tells this search from another for the state file: the measurements depend on the references, the cost
/// table's predictions and the sources. Priors, the batch and the other tuning only steer proposals, so a search may
/// change them and resume.
fn identity(flags: &Flags) -> String {
    let mut lines: Vec<String> = flags.references.iter().map(|path| path.display().to_string()).collect();
    lines.push(flags.costs.display().to_string());
    lines.extend(
        flags
            .sources
            .iter()
            .map(|(option, path)| format!("{}={}", option.key(), path.display())),
    );
    lines.join("\n")
}

/// The search's settings from the flags.
fn search_settings(flags: &Flags, seeds: Vec<Allocation>) -> SearchSettings {
    SearchSettings {
        context: flags.context,
        seeds,
        batch: flags.batch,
        explore: flags.explore,
        patience: flags.patience,
        random: flags.random,
        bands: flags.bands,
        verify_rows: flags.verify_rows,
        members: flags.members,
        ridge: flags.ridge,
        lambdas: flags.lambdas.clone(),
        budget_fractions: flags.budget_fractions.clone(),
        rco: flags.rco,
        knapsack_steps: flags.knapsack_steps,
        max_rounds: flags.max_rounds,
        seed: flags.seed,
        identity: identity(flags),
    }
}

/// The bytes of an assembled candidate's objects.
struct Resident(Arc<SourceSet>);

impl Assembly for Resident {
    fn resident_bytes(&self, allocation: &Allocation) -> Result<i64, SearchFailure> {
        let assembled = view(&self.0, allocation, None).map_err(|error| MixedError::Invalid(error.to_string()))?;
        Ok(i64::try_from(assembled.resident_bytes())
            .map_err(|_| MixedError::Invalid("the bytes of a candidate do not fit a 64-bit count".to_owned()))?)
    }
}

/// Runs the command on `args`, the words after its name.
pub fn run(args: &[String]) -> Outcome {
    let flags = match parse(args) {
        Ok(flags) => flags,
        Err(message) => return usage_error(&message),
    };
    let Some(library) = flags.library.clone().or_else(default_library) else {
        return usage_error("allocate needs --cuda-library on this platform");
    };
    let inputs = match inputs(&flags) {
        Ok(inputs) => inputs,
        Err(message) => return usage_error(&message),
    };
    if let Err(error) = create_dir_all(&flags.out) {
        return usage_error(&format!("--out {} cannot be made: {error}", flags.out.display()));
    }
    let words: Vec<_> = flags
        .config
        .iter()
        .flat_map(|path| ["--config".into(), path.clone().into_os_string()])
        .collect();
    let mut settings = match load_for_tool(&words, &process_env) {
        Ok(settings) => settings,
        Err(errors) => return usage_error(&errors.to_string()),
    };
    if let Some(workers) = flags.workers {
        settings.host.task_parallelism = workers;
    }
    match searched(&settings, &flags, &library, inputs) {
        Ok(()) => Outcome {
            code: EXIT_OK,
            message: String::new(),
        },
        Err(message) => Outcome {
            code: EXIT_RUN_FAILED,
            message: format!("allocate failed: {message}\n"),
        },
    }
}

/// Searches with the host runtime up, and stops it after.
fn searched(settings: &Settings, flags: &Flags, library: &Path, inputs: Inputs) -> Result<(), String> {
    let rig = Rig::start(settings)?;
    let result = searched_on(&rig, settings, flags, library, inputs);
    rig.stop();
    result
}

fn open_sources(rig: &Rig, settings: &Settings, flags: &Flags) -> Result<SourceSet, String> {
    let chunk_bytes = usize::try_from(settings.host.copy_chunk_bytes.bytes()).unwrap_or(usize::MAX);
    let (sender, receiver) = mpsc::channel();
    SourceSet::open(
        &rig.io,
        &flags.sources,
        flags.base,
        chunk_bytes,
        Arc::new(move |opened| {
            // The receiver is gone only when the run was abandoned.
            let _ = sender.send(opened);
        }),
    )
    .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|_| "the sources were not opened".to_owned())?
        .map_err(|error| error.to_string())
}

/// The free device bytes the search plans against: `--free-mib`, or what the card has now.
fn free_bytes(flags: &Flags, library: &Path) -> Result<u64, String> {
    if let Some(mib) = flags.free_mib {
        return Ok(mib.saturating_mul(MIB));
    }
    CudaLibrary::load(library)
        .map_err(|error| format!("the CUDA library: {error}"))?
        .free_bytes()
        .map_err(|error| format!("the free device memory: {error}"))
}

/// The engine's side of scoring: its settings and what each candidate's engine is started with.
fn device(settings_for: &Flags, library: &Path, mtp: bool) -> Result<Device, String> {
    let (mut context, mut chunks) = (0, 0);
    for path in &settings_for.references {
        let (wide, count) = dimensions(path)?;
        (context, chunks) = (context.max(wide), chunks.max(count));
    }
    let batch = match settings_for.score_batch {
        0 => DEFAULT_BATCH.min(chunks).max(1),
        asked => asked,
    };
    let base = settings_for
        .sources
        .get(&settings_for.base)
        .ok_or("no --source for the base")?;
    let settings = engine_settings(
        base,
        settings_for.config.as_deref(),
        context,
        Pass {
            batch,
            rows: settings_for.score_rows,
        },
        Memory {
            host_mib: settings_for.host_mib,
            workers: settings_for.workers,
        },
    )?;
    let depth = u8::try_from(settings_for.verify_rows - 1).map_err(|_| "--verify-rows is too large".to_owned())?;
    Ok(Device {
        settings,
        library: library.to_path_buf(),
        base: settings_for.base,
        sources: settings_for.sources.clone(),
        references: settings_for.references.clone(),
        batch,
        rows: settings_for.score_rows,
        draft_depth: mtp.then_some(depth),
        chunks,
    })
}

fn searched_on(rig: &Rig, settings: &Settings, flags: &Flags, library: &Path, inputs: Inputs) -> Result<(), String> {
    let Inputs { priors, costs, seeds } = inputs;
    let sources = Arc::new(open_sources(rig, settings, flags)?);
    let options = sources.options();
    let search = Search::new(search_settings(flags, seeds), &*sources, &options, &priors, &costs)
        .map_err(|error| error.to_string())?;
    let metadata = sources
        .artifact(flags.base)
        .map_err(|error| error.to_string())?
        .metadata();
    let config = ModelConfig::from(metadata);
    let mut device = device(flags, library, metadata.mtp_layer_count > 0)?;
    let imported = match &flags.import_state {
        Some(path) => read_measured(&rig.io, &rig.host, path).map_err(|error| format!("--import-state: {error}"))?,
        None => Vec::new(),
    };
    let measured_free = free_bytes(flags, library)?;
    let mut ledger = FileLedger::open(
        &rig.io,
        &rig.host,
        &flags.state,
        &search.fingerprint(),
        flags.seed,
        measured_free,
        &flags.out,
    )
    .map_err(|error| error.to_string())?;
    // A resumed search plans against what its first run measured, as its measurements were made against it.
    let planned = i64::try_from(ledger.free_bytes().unwrap_or(measured_free)).unwrap_or(i64::MAX);
    let depth = u8::try_from(flags.verify_rows - 1).map_err(|_| "--verify-rows is too large".to_owned())?;
    let feasibility = PlannerFeasibility::new(Arc::clone(&sources), &config, settings, planned, flags.context, depth);
    let sinks = flags
        .host_sinks
        .unwrap_or_else(|| usize::try_from(settings.host.task_parallelism).unwrap_or(1));
    let fan = LakeFan::new(Arc::clone(&rig.host), sinks);
    let assembly = Resident(Arc::clone(&sources));
    let outcome = search
        .run(World {
            ledger: &mut ledger,
            assembly: &assembly,
            scoring: &mut device,
            feasibility: Arc::new(feasibility),
            fan: &fan,
            imported,
        })
        .map_err(|error| error.to_string())?;
    log_summary(&outcome, flags);
    Ok(())
}

#[allow(clippy::print_stderr, reason = "the command's result, one line")]
fn log_summary(outcome: &Found, flags: &Flags) {
    eprintln!(
        "allocate: {} measured, {} on the front after {} rounds; the front is in {}",
        outcome.measured.len(),
        outcome.front.len(),
        outcome.rounds,
        flags.out.display()
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn flags(extra: &str) -> Flags {
        let text = format!(
            "--source nvfp4=a.edrl --source trellis-3.0=b.edrl --base trellis-3.0 --costs c.json --priors p.json \
             --reference r.kld --state s.resume --out dir {extra}"
        );
        parse(&text.split_whitespace().map(str::to_owned).collect::<Vec<_>>()).unwrap()
    }

    #[test]
    fn the_state_key_follows_the_references_costs_and_sources() {
        let base = identity(&flags(""));
        assert_ne!(
            identity(&flags("--reference other.kld")),
            base,
            "another reference is another search"
        );
        let other_costs = base.replace("c.json", "other.json");
        assert_ne!(other_costs, base);
        assert_ne!(
            identity(&flags("--costs d.json")),
            base,
            "another cost file is another search"
        );
        assert_eq!(
            identity(&flags("--priors q.json --batch 4 --patience 9 --seed 5 --context 2048")),
            base,
            "the priors and the tuning may change and resume"
        );
    }

    #[test]
    fn the_search_settings_carry_every_flag() {
        let settings = search_settings(&flags("--batch 5 --members 3"), Vec::new());
        assert_eq!((settings.batch, settings.members, settings.context), (5, 3, 32_768));
        assert!(settings.validate().is_ok());
    }
}
