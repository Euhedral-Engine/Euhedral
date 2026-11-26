use std::collections::BTreeMap;
use std::sync::Arc;

use super::super::allocation::Allocation;
use super::super::cost_table::{CostTable, Key, PREFILL_ROWS, Shape};
use super::super::error::MixedError;
use super::super::feasibility::{Check, Feasibility};
use super::super::fidelity::DEFAULT_BANDS;
use super::super::option::QuantOption;
use super::super::priors::Priors;
use super::super::rco;
use super::super::search_state::{Measured, Progress};
use super::super::space::Sources;
use super::super::space::tests::TinySources;
use super::{Assembly, Inline, Ledger, Score, Scoring, Search, SearchFailure, Settings, World};

const VERIFY_ROWS: u32 = 4;
const FREE: i64 = 1 << 30;

fn costs() -> CostTable {
    let mut micros = BTreeMap::new();
    // The more accurate option is the slower one, so the front is a real trade.
    for (option, prefill, decode) in [(QuantOption::Nvfp4, 102.4, 6.0), (QuantOption::Trellis3_0, 51.2, 5.0)] {
        let shape = Shape { input: 64, output: 64 };
        micros.insert(
            Key {
                shape,
                option,
                rows: PREFILL_ROWS,
            },
            prefill,
        );
        micros.insert(
            Key {
                shape,
                option,
                rows: VERIFY_ROWS,
            },
            decode,
        );
    }
    CostTable::new(micros, 0.0, BTreeMap::new())
}

fn priors(error: f64) -> Priors {
    let sources = TinySources::new();
    Priors::new(
        sources
            .projections()
            .iter()
            .map(|object| {
                let by_option = BTreeMap::from([(QuantOption::Nvfp4, 0.01), (QuantOption::Trellis3_0, error)]);
                (object.clone(), by_option)
            })
            .collect(),
    )
}

fn settings() -> Settings {
    Settings {
        context: 1024,
        seeds: Vec::new(),
        batch: 3,
        explore: 0.3,
        patience: 2,
        random: 4,
        bands: DEFAULT_BANDS,
        verify_rows: VERIFY_ROWS,
        members: 3,
        ridge: 1e-3,
        lambdas: vec![0.0, 0.01],
        budget_fractions: vec![1.0, 0.8],
        rco: rco::Settings::new(20, 0.1, 2, 1).unwrap(),
        knapsack_steps: 64,
        max_rounds: 50,
        seed: 1,
        identity: "references\ncosts".to_owned(),
    }
}

fn options() -> [QuantOption; 2] {
    [QuantOption::Nvfp4, QuantOption::Trellis3_0]
}

/// Fits while at most `most_nvfp4` objects are on the accurate option.
struct Cap {
    most_nvfp4: usize,
}

impl Feasibility for Cap {
    fn free_bytes(&self) -> i64 {
        FREE
    }

    fn check(&self, allocation: &Allocation) -> Result<Check, MixedError> {
        let count = allocation
            .choices()
            .values()
            .filter(|option| **option == QuantOption::Nvfp4)
            .count();
        Ok(Check {
            fits: count <= self.most_nvfp4,
            device_bytes: 1000 * i64::try_from(count).unwrap(),
        })
    }
}

struct Resident<'s>(&'s TinySources);

impl Assembly for Resident<'_> {
    fn resident_bytes(&self, allocation: &Allocation) -> Result<i64, SearchFailure> {
        let mut total = 0;
        for (object, option) in allocation.choices() {
            total += self.0.bytes(*option, object)?;
        }
        Ok(total)
    }
}

/// The ledger of one search, kept in memory; reusing it for a second run is a resume.
#[derive(Default)]
struct Memory {
    measured: Vec<Measured>,
    progress: Progress,
    files: BTreeMap<String, String>,
    notes: Vec<String>,
}

impl Ledger for Memory {
    fn seed(&self) -> i64 {
        1
    }

    fn measured(&self) -> &[Measured] {
        &self.measured
    }

    fn contains(&self, allocation: &Allocation) -> bool {
        self.measured.iter().any(|each| &each.allocation == allocation)
    }

    fn progress(&self) -> Progress {
        self.progress
    }

    fn add(&mut self, candidate: Measured) -> Result<bool, SearchFailure> {
        if self.contains(&candidate.allocation) {
            return Ok(false);
        }
        self.measured.push(candidate);
        Ok(true)
    }

    fn set_progress(&mut self, progress: Progress) -> Result<(), SearchFailure> {
        self.progress = progress;
        Ok(())
    }

    fn publish(&mut self, files: &[(String, String)]) -> Result<(), SearchFailure> {
        self.files = files.iter().cloned().collect();
        Ok(())
    }

    fn note(&mut self, line: &str) {
        self.notes.push(line.to_owned());
    }
}

/// Reads a candidate's KL off its allocation: each object on the cheaper option costs its own weight. Fails the call
/// after `allow` scored candidates, to stand for a search that is stopped.
struct Fake {
    scored: Vec<Allocation>,
    allow: usize,
}

impl Fake {
    fn new() -> Fake {
        Fake {
            scored: Vec::new(),
            allow: usize::MAX,
        }
    }

    fn kl(allocation: &Allocation) -> f64 {
        allocation
            .choices()
            .iter()
            .enumerate()
            .filter(|(_, (_, option))| **option == QuantOption::Trellis3_0)
            .map(|(i, _)| 0.01 * (i + 1) as f64)
            .sum()
    }
}

impl Scoring for Fake {
    fn score(&mut self, allocation: &Allocation) -> Result<Score, SearchFailure> {
        if self.scored.len() >= self.allow {
            return Err(SearchFailure::Scoring("stopped".to_owned()));
        }
        self.scored.push(allocation.clone());
        Ok(Score {
            mean_kl: Fake::kl(allocation),
            kl_se: 0.001,
        })
    }
}

fn run(
    search: &Search<'_>,
    ledger: &mut Memory,
    scoring: &mut Fake,
    most_nvfp4: usize,
    imported: Vec<Measured>,
) -> Result<super::Outcome, SearchFailure> {
    let sources = TinySources::new();
    let assembly = Resident(&sources);
    search.run(World {
        ledger,
        assembly: &assembly,
        scoring,
        feasibility: Arc::new(Cap { most_nvfp4 }),
        fan: &Inline,
        imported,
    })
}

fn front_json(ledger: &Memory) -> &str {
    &ledger.files["front.json"]
}

#[test]
fn stops_after_patience_rounds_without_a_new_front_point() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    let outcome = run(&search, &mut ledger, &mut scoring, 4, Vec::new()).unwrap();
    let last_growth = outcome.front.iter().map(|each| each.round).max().unwrap();
    let patience = settings().patience;
    assert_eq!(ledger.progress.stale, patience);
    assert_eq!(ledger.progress.round, last_growth + patience + 1, "the last round run");
    assert_eq!(outcome.rounds, last_growth + patience);
}

#[test]
fn a_resumed_search_scores_nothing_twice_and_reaches_the_front_of_an_unbroken_one() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    let (mut whole, mut whole_scoring) = (Memory::default(), Fake::new());
    run(&search, &mut whole, &mut whole_scoring, 4, Vec::new()).unwrap();
    assert!(whole.measured.iter().any(|each| each.round >= 1), "proposed rounds ran");

    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    // Stopped inside the first proposed round: more than the seeds, fewer than all.
    scoring.allow = 7;
    let stopped = run(&search, &mut ledger, &mut scoring, 4, Vec::new());
    assert!(matches!(stopped, Err(SearchFailure::Scoring(_))));
    assert_eq!(
        ledger.measured.len(),
        7,
        "each scored candidate was recorded before the stop"
    );
    scoring.allow = usize::MAX;
    run(&search, &mut ledger, &mut scoring, 4, Vec::new()).unwrap();

    let mut seen = scoring.scored.clone();
    seen.sort();
    seen.dedup();
    assert_eq!(seen.len(), scoring.scored.len(), "no candidate was scored twice");
    assert_eq!(ledger.measured, whole.measured);
    assert_eq!(front_json(&ledger), front_json(&whole));
}

#[test]
fn an_infeasible_candidate_is_not_scored() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    run(&search, &mut ledger, &mut scoring, 2, Vec::new()).unwrap();
    assert!(!scoring.scored.is_empty());
    for allocation in &scoring.scored {
        assert!(
            Cap { most_nvfp4: 2 }.check(allocation).unwrap().fits,
            "scored {}",
            allocation.render()
        );
    }
    assert!(ledger.measured.iter().all(|each| each.fits));
    assert!(
        ledger
            .notes
            .iter()
            .any(|line| line.contains("does not fit the device, not scored"))
    );
}

#[test]
fn nothing_is_searched_when_not_even_the_smallest_allocation_fits() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    struct Never;
    impl Feasibility for Never {
        fn free_bytes(&self) -> i64 {
            FREE
        }
        fn check(&self, _: &Allocation) -> Result<Check, MixedError> {
            Ok(Check {
                fits: false,
                device_bytes: 0,
            })
        }
    }
    let sources = TinySources::new();
    let assembly = Resident(&sources);
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    let failure = search.run(World {
        ledger: &mut ledger,
        assembly: &assembly,
        scoring: &mut scoring,
        feasibility: Arc::new(Never),
        fan: &Inline,
        imported: Vec::new(),
    });
    assert!(matches!(failure, Err(SearchFailure::NothingFits { context: 1024, .. })));
    assert!(scoring.scored.is_empty());
}

#[test]
fn an_unknown_seed_object_is_refused_by_name() {
    let (sources, costs) = (TinySources::new(), costs());
    let mut with_seed = settings();
    with_seed.seeds = vec![Allocation::new(BTreeMap::from([(
        "text/layers/9/mlp/down".to_owned(),
        QuantOption::Nvfp4,
    )]))];
    let search = Search::new(with_seed, &sources, &options(), &priors(0.5), &costs).unwrap();
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    let failure = run(&search, &mut ledger, &mut scoring, 4, Vec::new()).unwrap_err();
    assert!(failure.to_string().contains("text/layers/9/mlp/down"), "{failure}");
    assert!(scoring.scored.is_empty());
}

#[test]
fn imported_candidates_are_recorded_unscored_before_the_seeds() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    let allocation = Allocation::new(BTreeMap::from([(
        "text/layers/0/mlp/down".to_owned(),
        QuantOption::Nvfp4,
    )]));
    let imported = Measured {
        allocation,
        kl: 0.123,
        kl_se: 0.004,
        bytes: 1,
        prefill: 1.0,
        decode: 1.0,
        fits: false,
        round: 7,
        origin: "elsewhere".to_owned(),
    };
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    run(&search, &mut ledger, &mut scoring, 4, vec![imported]).unwrap();
    let first = &ledger.measured[0];
    assert_eq!(
        (first.kl, first.round, first.origin.as_str()),
        (0.123, 0, "import:elsewhere")
    );
    assert!(first.fits, "fit is judged again for this search");
    assert!(!scoring.scored.contains(&first.allocation));
}

#[test]
fn the_front_is_written_with_every_allocation_it_names() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = Search::new(settings(), &sources, &options(), &priors(0.5), &costs).unwrap();
    let (mut ledger, mut scoring) = (Memory::default(), Fake::new());
    let outcome = run(&search, &mut ledger, &mut scoring, 4, Vec::new()).unwrap();
    assert!(ledger.files.contains_key("report.txt"));
    for (i, point) in outcome.front.iter().enumerate() {
        assert_eq!(
            ledger.files[&format!("allocations/front-{i:02}.json")],
            point.allocation.render()
        );
    }
    assert_eq!(
        ledger
            .files
            .keys()
            .filter(|path| path.starts_with("allocations/"))
            .count(),
        outcome.front.len()
    );
}

#[test]
fn the_fingerprint_follows_the_identity_but_not_the_priors_or_the_batch() {
    let (sources, costs) = (TinySources::new(), costs());
    let search = |settings: Settings, priors: &Priors| {
        Search::new(settings, &sources, &options(), priors, &costs)
            .unwrap()
            .fingerprint()
    };
    let base = search(settings(), &priors(0.5));
    let mut batch = settings();
    batch.batch = 5;
    assert_eq!(search(batch, &priors(0.5)), base, "a changed batch resumes");
    assert_eq!(search(settings(), &priors(0.9)), base, "changed priors resume");
    let mut other = settings();
    other.identity = "another reference\ncosts".to_owned();
    assert_ne!(
        search(other, &priors(0.5)),
        base,
        "a different reference is a different search"
    );
    let mut other = settings();
    other.identity = "references\nother costs".to_owned();
    assert_ne!(
        search(other, &priors(0.5)),
        base,
        "a different cost table is a different search"
    );
}

#[test]
fn settings_outside_their_limits_are_refused_with_the_value() {
    let (sources, costs) = (TinySources::new(), costs());
    let refused = |change: fn(&mut Settings)| {
        let mut bad = settings();
        change(&mut bad);
        Search::new(bad, &sources, &options(), &priors(0.5), &costs)
            .unwrap_err()
            .to_string()
    };
    assert!(refused(|s| s.batch = 0).contains("batch is 0"));
    assert!(refused(|s| s.explore = 1.5).contains("explore is 1.5"));
    assert!(refused(|s| s.verify_rows = 1).contains("verify rows is 1"));
    assert!(refused(|s| s.patience = 0).contains("patience is 0"));
}

#[test]
fn an_option_without_a_prior_is_left_out_and_a_base_without_one_is_refused() {
    let (sources, costs) = (TinySources::new(), costs());
    let only_base = Priors::new(
        sources
            .projections()
            .iter()
            .map(|object| (object.clone(), BTreeMap::from([(QuantOption::Trellis3_0, 0.5)])))
            .collect(),
    );
    let failure = Search::new(settings(), &sources, &options(), &only_base, &costs).unwrap_err();
    assert!(failure.to_string().contains("two options"), "{failure}");
    let only_other = Priors::new(
        sources
            .projections()
            .iter()
            .map(|object| (object.clone(), BTreeMap::from([(QuantOption::Nvfp4, 0.5)])))
            .collect(),
    );
    let failure = Search::new(settings(), &sources, &options(), &only_other, &costs).unwrap_err();
    assert!(failure.to_string().contains("base trellis"), "{failure}");
}

#[test]
fn scores_pool_over_chunks_and_a_malformed_chunk_has_none() {
    let chunk = |kl: [f64; 2]| -> Vec<f64> { kl.iter().flat_map(|kl| [0.0, 0.0, *kl, 0.0, 0.0]).collect() };
    let score = Score::of(&[chunk([1.0, 3.0]), chunk([5.0, 7.0])]).unwrap();
    assert_eq!(score.mean_kl, 4.0);
    assert!(score.kl_se > 0.0);
    assert!(Score::of(&[]).is_none());
    assert!(Score::of(&[vec![1.0; 4]]).is_none());
}
