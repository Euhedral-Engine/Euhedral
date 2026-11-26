//! Scoring a candidate on the card: an engine started on the composite the allocation makes, the references forced
//! through it, the engine shut down.
//!
//! The engine loads its weights once per start, so each candidate starts its own engine: its composite replaces the
//! objects the allocation takes from the sources of other options (each payload is read from the file it belongs
//! to). The runtime of the sources and the state is another lattice and stays up between candidates.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_inference_engine::{Composite, EngineRuntime, Replacement};
use euhedral_inference_quantization::kld::{self, Reference};
use euhedral_inference_quantization::mixed::search::{Score, Scoring, SearchFailure};
use euhedral_inference_quantization::mixed::{Allocation, QuantOption};
use euhedral_inference_quantization::reference::Forcer;
use euhedral_inference_quantization::score::device::Geometry;
use euhedral_inference_quantization::score::{Scorer, Source, batch_for, score};
use euhedral_inference_settings::Settings;

use crate::score::LONGEST_RUN;

/// What scoring a candidate needs besides the candidate.
pub(super) struct Device {
    /// The engine's settings: the base artifact, a context that holds a reference's chunk, no speculation.
    pub settings: Settings,
    pub library: PathBuf,
    pub base: QuantOption,
    pub sources: BTreeMap<QuantOption, PathBuf>,
    pub references: Vec<PathBuf>,
    /// Chunks forced together, as the engine's settings hold sequences for them.
    pub batch: usize,
    pub rows: usize,
    /// The draft depth a composite stores, when the base has an MTP layer to draft with.
    pub draft_depth: Option<u8>,
    /// The most chunks of any reference.
    pub chunks: usize,
}

impl Device {
    /// The objects the allocation takes from each source other than the base's.
    fn composite(&self, allocation: &Allocation) -> Composite {
        let mut taken: BTreeMap<QuantOption, Vec<String>> = BTreeMap::new();
        for (object, option) in allocation.choices() {
            if *option != self.base {
                taken.entry(*option).or_default().push(object.clone());
            }
        }
        Composite {
            replacements: taken
                .into_iter()
                .filter_map(|(option, objects)| {
                    self.sources.get(&option).map(|source| Replacement {
                        source: source.clone(),
                        objects,
                    })
                })
                .collect(),
            draft_depth: self.draft_depth,
        }
    }

    /// Every chunk of every reference scored on the started engine, pooled.
    fn scored(&self, runtime: &Arc<EngineRuntime>) -> Result<Score, String> {
        let vocabulary = runtime.config().vocab_size;
        let batch = batch_for(Some(self.batch), runtime.sequences(), self.chunks)?;
        let io = runtime.io();
        let mut references = Vec::new();
        for path in &self.references {
            let reference = kld::Reference::read(io, path)
                .wait(LONGEST_RUN)
                .map_err(|_| format!("{}: reading the header timed out", path.display()))?
                .map_err(|error| format!("{}: {error}", path.display()))?;
            if reference.vocabulary != vocabulary {
                return Err(format!(
                    "{}: a vocabulary of {}, not the model's {vocabulary}",
                    path.display(),
                    reference.vocabulary
                ));
            }
            let records = kld::open_records(io, path)
                .wait(LONGEST_RUN)
                .map_err(|_| format!("{}: opening the records timed out", path.display()))?
                .map_err(|error| format!("{}: {error}", path.display()))?;
            references.push((reference, records));
        }
        let record_bytes = references
            .first()
            .map(|(reference, _): &(Reference, _)| reference.row_bytes())
            .ok_or("no reference to score")?;
        let geometry = Geometry {
            vocabulary,
            record_bytes,
            group: batch * self.rows,
        };
        let scorer = Scorer::open(runtime, geometry).map_err(|error| format!("the scorer would not open: {error}"))?;
        let mut pooled: Vec<Vec<f64>> = Vec::new();
        let mut outcome = Ok(());
        for (reference, records) in references {
            let forcer: Arc<dyn Forcer> = Arc::clone(runtime) as Arc<dyn Forcer>;
            let source = Source {
                forcer,
                io: Arc::clone(io),
                scorer: Arc::clone(&scorer),
                reference,
                records,
                rows: self.rows,
            };
            match score(source, batch).wait(LONGEST_RUN) {
                Ok(Ok(chunks)) => pooled.extend(chunks),
                Ok(Err(why)) => outcome = Err(why),
                Err(_) => outcome = Err("the score did not finish in time".to_owned()),
            }
            if outcome.is_err() {
                break;
            }
        }
        scorer.close();
        outcome?;
        Score::of(&pooled).ok_or_else(|| "nothing was scored".to_owned())
    }
}

impl Scoring for Device {
    fn score(&mut self, allocation: &Allocation) -> Result<Score, SearchFailure> {
        let failed = |why: String| SearchFailure::Scoring(why);
        let composite = self.composite(allocation);
        let (owner, waiting) = EngineRuntime::start_composite(&self.settings, &self.library, None, composite);
        let runtime = match waiting.wait(LONGEST_RUN) {
            Ok(Ok(runtime)) => runtime,
            Ok(Err(error)) => return Err(failed(format!("the engine did not load: {error}"))),
            Err(_) => return Err(failed("the engine did not finish loading".to_owned())),
        };
        let result = self.scored(&runtime);
        let shutdown = owner.shutdown(runtime);
        let score = result.map_err(failed)?;
        shutdown.map_err(|error| failed(format!("the engine's shutdown: {error}")))?;
        Ok(score)
    }
}
