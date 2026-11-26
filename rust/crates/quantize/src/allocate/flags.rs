//! The flags of `allocate`: what each takes, its default and its limit. A value outside its limit is refused here,
//! naming the flag, the value and the limit, before a file is opened or the card is touched.

use std::collections::BTreeMap;
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;

use euhedral_inference_quantization::mixed::rco::{self, DEFAULT_STEPS};
use euhedral_inference_quantization::mixed::search::{
    MOST_BANDS, MOST_CANDIDATES, MOST_KNAPSACK_STEPS, MOST_MEMBERS, MOST_WEIGHTS,
};
use euhedral_inference_quantization::mixed::{DEFAULT_BANDS, QuantOption, VERIFY_MAX_ROWS};
use euhedral_inference_quantization::reference::DEFAULT_ROWS;

/// The most rows in flight at once while scoring: `--score-rows` times `--score-batch`.
const MOST_IN_FLIGHT: usize = 32;
/// The most rows of a scored verification.
const MOST_SCORE_ROWS: usize = 8;
/// The default bound on one wave of host tasks, in seconds: a day.
const DEFAULT_WAVE_SECS: u64 = 24 * 60 * 60;
/// The most a wave may be given, in seconds: a year, so that the deadline stays a time the clock can hold.
const MOST_WAVE_SECS: u64 = 366 * 24 * 60 * 60;
/// The default target context in tokens.
const DEFAULT_CONTEXT: u32 = 32_768;

/// What the command was asked for.
#[derive(Debug, Clone)]
pub(super) struct Flags {
    pub sources: BTreeMap<QuantOption, PathBuf>,
    pub base: QuantOption,
    pub costs: PathBuf,
    pub priors: PathBuf,
    pub references: Vec<PathBuf>,
    pub context: u32,
    pub state: PathBuf,
    pub out: PathBuf,
    pub seed_file: Option<PathBuf>,
    pub import_state: Option<PathBuf>,
    /// Free device bytes to plan against, in MiB; unset reads them from the card.
    pub free_mib: Option<u64>,
    pub workers: Option<u32>,
    pub host_mib: Option<u64>,
    /// Rows of a scored verification.
    pub score_rows: usize,
    /// Chunks scored at once; 0 lets the engine's sequences and the references decide.
    pub score_batch: usize,
    pub batch: usize,
    pub explore: f64,
    pub patience: u32,
    pub random: usize,
    pub bands: usize,
    pub verify_rows: u32,
    /// The most host tasks of the search the lake holds at once; unset takes the lake's workers.
    pub host_sinks: Option<usize>,
    /// The longest, in seconds, the search waits for one wave of host tasks before it gives the run up.
    pub wave_secs: u64,
    pub members: usize,
    pub ridge: f64,
    pub lambdas: Vec<f64>,
    pub budget_fractions: Vec<f64>,
    pub rco: rco::Settings,
    pub knapsack_steps: u32,
    pub max_rounds: u32,
    pub seed: i64,
    pub config: Option<PathBuf>,
    pub library: Option<PathBuf>,
}

fn whole<T: FromStr + PartialOrd + Display + Copy>(flag: &str, value: &str, least: T, most: T) -> Result<T, String> {
    value
        .parse::<T>()
        .ok()
        .filter(|number| (least..=most).contains(number))
        .ok_or_else(|| format!("{flag} {value} is not a whole number from {least} to {most}"))
}

fn real(flag: &str, value: &str, low: f64, high: f64) -> Result<f64, String> {
    value
        .parse::<f64>()
        .ok()
        .filter(|number| number.is_finite() && (low..=high).contains(number))
        .ok_or_else(|| format!("{flag} {value} is not a number from {low} to {high}"))
}

fn reals(flag: &str, value: &str, low: f64, high: f64) -> Result<Vec<f64>, String> {
    let list: Vec<f64> = value
        .split(',')
        .map(|each| {
            each.trim()
                .parse::<f64>()
                .ok()
                .filter(|number| number.is_finite() && (low..=high).contains(number))
        })
        .collect::<Option<_>>()
        .ok_or_else(|| format!("{flag} {value} is not a comma-separated list of numbers from {low} to {high}"))?;
    if list.len() > MOST_WEIGHTS {
        return Err(format!(
            "{flag} {value} has {} values, not from 1 to {MOST_WEIGHTS}",
            list.len()
        ));
    }
    Ok(list)
}

fn concurrent(flag: &str, value: &str) -> Result<(), String> {
    Err(format!(
        "{flag} {value}: concurrent scoring is not yet ported; the search scores one candidate at a time"
    ))
}

/// The flags in `args`. Every value is checked against its limit; a flag the command does not know, a flag with no
/// value, a base without a source and fewer than two sources are refused.
pub(super) fn parse(args: &[String]) -> Result<Flags, String> {
    let mut flags = Flags {
        sources: BTreeMap::new(),
        base: QuantOption::Nvfp4,
        costs: PathBuf::new(),
        priors: PathBuf::new(),
        references: Vec::new(),
        context: DEFAULT_CONTEXT,
        state: PathBuf::new(),
        out: PathBuf::new(),
        seed_file: None,
        import_state: None,
        free_mib: None,
        workers: None,
        host_mib: None,
        score_rows: DEFAULT_ROWS,
        score_batch: 0,
        batch: 8,
        explore: 0.3,
        patience: 3,
        random: 16,
        bands: DEFAULT_BANDS,
        verify_rows: 5,
        host_sinks: None,
        wave_secs: DEFAULT_WAVE_SECS,
        members: 8,
        ridge: 1e-3,
        lambdas: vec![0.0, 0.002, 0.005, 0.01, 0.02],
        budget_fractions: vec![1.0, 0.8],
        rco: rco::Settings::new(DEFAULT_STEPS, 0.1, 8, 4).map_err(|error| error.to_string())?,
        knapsack_steps: 512,
        max_rounds: i32::MAX.unsigned_abs(),
        seed: 1,
        config: None,
        library: None,
    };
    let (mut base, mut rco_steps, mut rco_lr, mut rco_samples, mut rco_restarts) = (None, DEFAULT_STEPS, 0.1, 8, 4);
    let (mut costs, mut priors, mut state, mut out) = (None, None, None, None);
    let mut words = args.iter();
    while let Some(flag) = words.next() {
        let value = words.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let path = || PathBuf::from(value);
        match flag.as_str() {
            "--source" => {
                let (key, artifact) = value
                    .split_once('=')
                    .ok_or_else(|| format!("--source {value} is not OPTION=ARTIFACT"))?;
                let option = QuantOption::of(key).map_err(|error| error.to_string())?;
                if flags.sources.insert(option, PathBuf::from(artifact)).is_some() {
                    return Err(format!("--source {key} is given twice"));
                }
            }
            "--base" => base = Some(QuantOption::of(value).map_err(|error| error.to_string())?),
            "--costs" => costs = Some(path()),
            "--priors" => priors = Some(path()),
            "--reference" => flags.references.push(path()),
            "--context" => flags.context = whole(flag, value, 1, u32::MAX)?,
            "--state" => state = Some(path()),
            "--out" => out = Some(path()),
            "--seed-file" => flags.seed_file = Some(path()),
            "--import-state" => flags.import_state = Some(path()),
            "--free-mib" => flags.free_mib = Some(whole(flag, value, 1, i64::MAX as u64 >> 20)?),
            "--workers" => flags.workers = Some(whole(flag, value, 1, u32::MAX)?),
            "--host-mib" => flags.host_mib = Some(whole(flag, value, 1, i64::MAX as u64 >> 20)?),
            "--score-rows" => flags.score_rows = whole(flag, value, 1, MOST_SCORE_ROWS)?,
            "--score-batch" => flags.score_batch = whole(flag, value, 0, MOST_IN_FLIGHT)?,
            "--batch" => flags.batch = whole(flag, value, 1, MOST_CANDIDATES)?,
            "--explore" => flags.explore = real(flag, value, 0.0, 1.0)?,
            "--patience" => flags.patience = whole(flag, value, 1, u32::MAX)?,
            "--random" => flags.random = whole(flag, value, 0, MOST_CANDIDATES)?,
            "--bands" => flags.bands = whole(flag, value, 1, MOST_BANDS)?,
            "--verify-rows" => flags.verify_rows = whole(flag, value, 2, VERIFY_MAX_ROWS)?,
            "--host-sinks" => flags.host_sinks = Some(whole(flag, value, 1, MOST_CANDIDATES)?),
            "--wave-secs" => flags.wave_secs = whole(flag, value, 1, MOST_WAVE_SECS)?,
            "--members" => flags.members = whole(flag, value, 1, MOST_MEMBERS)?,
            "--ridge" => flags.ridge = real(flag, value, 0.0, f64::MAX)?,
            "--lambdas" => flags.lambdas = reals(flag, value, 0.0, f64::MAX)?,
            "--budget-fractions" => flags.budget_fractions = reals(flag, value, 0.0, 1.0)?,
            "--rco-steps" => rco_steps = whole(flag, value, 1, usize::MAX)?,
            "--rco-lr" => rco_lr = real(flag, value, f64::MIN_POSITIVE, f64::MAX)?,
            "--rco-samples" => rco_samples = whole(flag, value, 1, usize::MAX)?,
            "--rco-restarts" => rco_restarts = whole(flag, value, 1, usize::MAX)?,
            "--knapsack-steps" => flags.knapsack_steps = whole(flag, value, 1, MOST_KNAPSACK_STEPS)?,
            "--max-rounds" => flags.max_rounds = whole(flag, value, 1, i32::MAX.unsigned_abs())?,
            "--seed" => flags.seed = whole(flag, value, i64::MIN, i64::MAX)?,
            "--max-concurrent" if value != "1" => return concurrent(flag, value).map(|()| flags),
            "--max-concurrent" => {}
            "--concurrent-margin-mib" => return concurrent(flag, value).map(|()| flags),
            "--config" => flags.config = Some(path()),
            "--cuda-library" | "--library" => flags.library = Some(path()),
            other => return Err(format!("unknown flag {other}")),
        }
    }
    flags.rco = rco::Settings::new(rco_steps, rco_lr, rco_samples, rco_restarts).map_err(|error| error.to_string())?;
    if flags.budget_fractions.is_empty() {
        return Err("--budget-fractions has no value".to_owned());
    }
    if flags.lambdas.is_empty() {
        return Err("--lambdas has no value".to_owned());
    }
    flags.base = base.ok_or("--base is required")?;
    if !flags.sources.contains_key(&flags.base) {
        return Err(format!("no --source for the base {}", flags.base));
    }
    if flags.sources.len() < 2 {
        return Err("allocate needs at least two --source options to choose between".to_owned());
    }
    if flags.references.is_empty() {
        return Err("--reference is required, once or more".to_owned());
    }
    if flags.score_batch > 0 && flags.score_rows * flags.score_batch > MOST_IN_FLIGHT {
        return Err(format!(
            "--score-rows {} times --score-batch {} is more than the {MOST_IN_FLIGHT} rows that can be in flight",
            flags.score_rows, flags.score_batch
        ));
    }
    flags.costs = costs.ok_or("--costs is required")?;
    flags.priors = priors.ok_or("--priors is required")?;
    flags.state = state.ok_or("--state is required")?;
    flags.out = out.ok_or("--out is required")?;
    Ok(flags)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    fn words(text: &str) -> Vec<String> {
        text.split_whitespace().map(str::to_owned).collect()
    }

    const MINIMAL: &str = "--source nvfp4=a.edrl --source trellis-3.0=b.edrl --base trellis-3.0 --costs c.json \
                           --priors p.json --reference r.kld --state s.resume --out dir";

    fn with(extra: &str) -> Result<Flags, String> {
        parse(&words(&format!("{MINIMAL} {extra}")))
    }

    #[test]
    fn the_defaults_are_those_of_the_search() {
        let flags = parse(&words(MINIMAL)).unwrap();
        assert_eq!(
            (flags.context, flags.batch, flags.patience, flags.random),
            (32_768, 8, 3, 16)
        );
        assert_eq!(
            (flags.verify_rows, flags.members, flags.knapsack_steps, flags.seed),
            (5, 8, 512, 1)
        );
        assert_eq!(
            (flags.score_rows, flags.score_batch, flags.max_rounds),
            (8, 0, i32::MAX.unsigned_abs())
        );
        assert_eq!(flags.lambdas, [0.0, 0.002, 0.005, 0.01, 0.02]);
        assert_eq!(flags.budget_fractions, [1.0, 0.8]);
        assert_eq!((flags.explore, flags.ridge, flags.bands), (0.3, 1e-3, DEFAULT_BANDS));
        assert_eq!(flags.base, QuantOption::Trellis3_0);
        assert_eq!(flags.references, [PathBuf::from("r.kld")]);
    }

    #[test]
    fn every_flag_sets_its_value() {
        let flags = with(
            "--context 4096 --seed-file s.json --import-state o.resume --free-mib 20000 --workers 3 --host-mib 512 \
             --score-rows 4 --score-batch 2 --batch 5 --explore 0.5 --patience 2 --random 0 --bands 4 \
             --verify-rows 3 --host-sinks 2 --wave-secs 90 --members 4 --ridge 0.01 --lambdas 0,0.5 --budget-fractions 1 \
             --rco-steps 10 --rco-lr 0.2 --rco-samples 3 --rco-restarts 2 --knapsack-steps 64 --max-rounds 9 \
             --seed -7 --max-concurrent 1 --reference r2.kld",
        )
        .unwrap();
        assert_eq!(
            (flags.context, flags.free_mib, flags.workers, flags.host_mib),
            (4096, Some(20000), Some(3), Some(512))
        );
        assert_eq!(
            (flags.score_rows, flags.score_batch, flags.batch, flags.random),
            (4, 2, 5, 0)
        );
        assert_eq!(
            (
                flags.verify_rows,
                flags.host_sinks,
                flags.wave_secs,
                flags.members,
                flags.seed
            ),
            (3, Some(2), 90, 4, -7)
        );
        assert_eq!(flags.rco, rco::Settings::new(10, 0.2, 3, 2).unwrap());
        assert_eq!(flags.lambdas, [0.0, 0.5]);
        assert_eq!(flags.references.len(), 2);
        assert_eq!(flags.import_state, Some(PathBuf::from("o.resume")));
    }

    #[test]
    fn a_value_outside_its_limit_is_refused_naming_the_flag_and_the_limit() {
        for (extra, named) in [
            ("--batch 0", "--batch 0 is not a whole number from 1 to 4096"),
            ("--batch 4097", "from 1 to 4096"),
            ("--random 4097", "--random 4097"),
            ("--members 0", "--members 0"),
            ("--members 257", "from 1 to 256"),
            ("--bands 0", "--bands 0"),
            ("--patience 0", "--patience 0"),
            ("--max-rounds 0", "--max-rounds 0"),
            ("--verify-rows 1", "--verify-rows 1 is not a whole number from 2 to 8"),
            ("--verify-rows 9", "--verify-rows 9"),
            ("--explore 1.5", "--explore 1.5 is not a number from 0 to 1"),
            ("--explore nan", "--explore nan"),
            ("--ridge -1", "--ridge -1"),
            ("--lambdas 0,-1", "--lambdas 0,-1"),
            ("--lambdas 0,x", "--lambdas 0,x"),
            (
                "--budget-fractions 1.2",
                "--budget-fractions 1.2 is not a comma-separated list of numbers from 0 to 1",
            ),
            ("--rco-lr 0", "--rco-lr 0"),
            ("--rco-steps 0", "--rco-steps 0"),
            ("--rco-samples 0", "--rco-samples 0"),
            ("--rco-restarts 0", "--rco-restarts 0"),
            ("--knapsack-steps 0", "--knapsack-steps 0"),
            ("--context 0", "--context 0"),
            ("--free-mib 0", "--free-mib 0"),
            ("--workers 0", "--workers 0"),
            ("--host-mib 0", "--host-mib 0"),
            ("--host-sinks 0", "--host-sinks 0"),
            ("--wave-secs 0", "--wave-secs 0"),
            ("--score-rows 9", "--score-rows 9 is not a whole number from 1 to 8"),
            ("--score-rows 0", "--score-rows 0"),
            ("--score-batch 33", "--score-batch 33"),
            ("--score-rows 8 --score-batch 5", "more than the 32 rows"),
            ("--seed many", "--seed many"),
        ] {
            let refusal = with(extra).unwrap_err();
            assert!(refusal.contains(named), "{extra}: {refusal}");
        }
    }

    #[test]
    fn concurrent_scoring_is_refused_as_not_yet_ported() {
        for extra in [
            "--max-concurrent 0",
            "--max-concurrent 2",
            "--concurrent-margin-mib 512",
        ] {
            let refusal = with(extra).unwrap_err();
            assert!(
                refusal.contains("concurrent scoring is not yet ported"),
                "{extra}: {refusal}"
            );
        }
        assert!(with("--max-concurrent 1").is_ok());
    }

    #[test]
    fn missing_and_malformed_flags_are_refused() {
        let without = |flag: &str| {
            let mut kept = Vec::new();
            let mut words = MINIMAL.split_whitespace();
            while let Some(word) = words.next() {
                if word == flag {
                    words.next();
                } else {
                    kept.push(word.to_owned());
                }
            }
            parse(&kept).unwrap_err()
        };
        for flag in ["--base", "--costs", "--priors", "--reference", "--state", "--out"] {
            assert!(without(flag).contains(flag), "{flag}");
        }
        assert!(with("--bogus 1").unwrap_err().contains("unknown flag --bogus"));
        assert!(with("--batch").unwrap_err().contains("--batch needs a value"));
        assert!(with("--source nvfp4=c").unwrap_err().contains("given twice"));
        assert!(with("--source nvfp4").unwrap_err().contains("OPTION=ARTIFACT"));
        assert!(with("--source int8=x").is_err());
        assert!(
            with("--base trellis-4.0")
                .unwrap_err()
                .contains("no --source for the base")
        );
        let one = "--source nvfp4=a --base nvfp4 --costs c --priors p --reference r --state s --out o";
        assert!(parse(&words(one)).unwrap_err().contains("at least two --source"));
    }
}
