//! Which object of the assembled artifact comes from which source.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_inference_edrl::{ModelMetadata, TensorEntry, TokenizerSection};
use euhedral_io::OwnedFile;

use super::AssembleError;
use super::sources::{COMPANION_SUFFIXES, SourceSet};
use crate::mixed::{Allocation, QuantOption};

/// One object of the assembled artifact: its descriptor in the source it is read from, and that source.
#[derive(Debug, Clone)]
pub struct Piece {
    /// The descriptor in the source file: where the payload is read from.
    pub entry: TensorEntry,
    /// The option whose source holds the payload.
    pub option: QuantOption,
    /// The source file, held open.
    pub file: Arc<OwnedFile>,
    /// The source file's path.
    pub path: PathBuf,
}

/// The assembled artifact before it is written: the metadata, the objects in the base's order, and the base's
/// tokenizer.
#[derive(Debug, Clone)]
pub struct View {
    pub metadata: ModelMetadata,
    pub pieces: Vec<Piece>,
    pub tokenizer: TokenizerSection,
}

impl View {
    /// The payload bytes the assembled artifact holds.
    pub fn resident_bytes(&self) -> u64 {
        self.pieces.iter().map(|piece| piece.entry.bytes()).sum()
    }
}

/// The assembled artifact: the base's objects in its order, the chosen projections and their companions from the
/// sources of their options. `draft_depth`, when given, is the stored MTP draft depth; otherwise the base's stays.
///
/// Refused, naming the object or the option: an object that is unknown, is not a projection or is one of the MTP
/// layer's, an option without a source, a base that carries no tokenizer.
pub fn view(sources: &SourceSet, allocation: &Allocation, draft_depth: Option<u8>) -> Result<View, AssembleError> {
    let base = sources.base();
    let choosable: HashSet<&str> = sources.projections().iter().map(String::as_str).collect();
    let options = sources.options();
    let base_file = sources.artifact(base)?;
    let mut chosen: BTreeMap<&str, QuantOption> = BTreeMap::new();
    for (name, option) in allocation.choices() {
        if !choosable.contains(name.as_str()) {
            if sources.drafter_projections().contains(name) {
                return Err(AssembleError::invalid(format!(
                    "{name} is a projection of the MTP layer, which is not chosen"
                )));
            }
            return Err(AssembleError::invalid(if base_file.table().get(name).is_some() {
                format!("{name} is not a projection")
            } else {
                format!("unknown object {name}")
            }));
        }
        if !options.contains(option) {
            return Err(AssembleError::invalid(format!(
                "no source for {} (chosen for {name})",
                option.key()
            )));
        }
        chosen.insert(name, *option);
    }
    let tokenizer = base_file
        .tokenizer()
        .cloned()
        .ok_or_else(|| AssembleError::invalid(format!("{} carries no tokenizer", base_file.path().display())))?;
    let mut pieces = Vec::new();
    for entry in base_file.table().entries() {
        let name = entry.name();
        match chosen.get(name) {
            Some(option) if *option != base => {
                for taken in sources.object_with_companions(*option, name)? {
                    pieces.push(piece(sources, *option, taken)?);
                }
            }
            _ if !companion_of_replaced(name, &chosen, base) => pieces.push(piece(sources, base, entry)?),
            _ => {}
        }
    }
    let mut metadata = base_file.metadata().clone();
    if draft_depth.is_some() {
        metadata.draft_depth = draft_depth;
    }
    Ok(View {
        metadata,
        pieces,
        tokenizer,
    })
}

fn piece(sources: &SourceSet, option: QuantOption, entry: &TensorEntry) -> Result<Piece, AssembleError> {
    let source = sources.artifact(option)?;
    Ok(Piece {
        entry: entry.clone(),
        option,
        file: Arc::clone(source.file()),
        path: source.path().to_path_buf(),
    })
}

/// Whether `name` is a companion of an object that comes from a source other than the base.
fn companion_of_replaced(name: &str, chosen: &BTreeMap<&str, QuantOption>, base: QuantOption) -> bool {
    COMPANION_SUFFIXES.iter().any(|suffix| {
        name.strip_suffix(suffix)
            .and_then(|stem| chosen.get(stem))
            .is_some_and(|option| *option != base)
    })
}
