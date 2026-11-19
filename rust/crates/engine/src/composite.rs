//! Weights whose objects come from several artifacts: a base artifact, and for some of its quantized projections an
//! artifact each, taken whole with their companions. The model is loaded from the composite table, each payload read
//! from the file its entry belongs to, so a candidate mix of quantizations runs without a file written for it.

use std::collections::{HashMap, HashSet};
use std::iter;
use std::path::PathBuf;

use euhedral_inference_edrl::{TensorEntry, TensorTable, WeightFormat};

use crate::error::{EngineError, PlanError};
use crate::model::{companion_name, is_companion, trellis_logical};

/// The companions that travel with an object.
const COMPANIONS: [&str; 3] = [".suh", ".svh", ".segments"];

/// The objects the choosable projections are named under.
const CHOOSABLE_PREFIX: &str = "text/layers/";

/// Objects taken from one artifact in place of the base's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Replacement {
    /// The artifact file the objects (and their companions) are read from.
    pub source: PathBuf,
    /// The quantized linear objects of `text/layers/*` to take from it.
    pub objects: Vec<String>,
}

/// The replacements of a base artifact (`model.artifact`). The default has none, and the model loads from the base
/// alone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Composite {
    pub replacements: Vec<Replacement>,
}

impl Composite {
    /// Whether the model loads from the base alone.
    pub fn is_empty(&self) -> bool {
        self.replacements
            .iter()
            .all(|replacement| replacement.objects.is_empty())
    }
}

/// The table of a composite and where each entry's payload is.
#[derive(Debug)]
pub(crate) struct Composed {
    pub table: TensorTable,
    /// For an entry taken from a replacement's source, the index of that replacement plus one. An entry from the base
    /// has no mark.
    pub source_of: HashMap<String, usize>,
}

fn invalid(why: String) -> EngineError {
    EngineError::Plan(PlanError::Inventory(why))
}

/// The input and output features of a quantized projection, whatever the format stores them as.
fn features(entry: &TensorEntry) -> Option<(u64, u64)> {
    match (entry.format(), entry.shape()) {
        (WeightFormat::Trellis, shape) => trellis_logical(shape),
        (WeightFormat::Nvfp4, [rows, k]) => Some((*rows, *k)),
        _ => None,
    }
}

/// The composite table: the base's entries in its order, each chosen object (with its companions) replaced by the
/// source's, whose tables are `sources`, one per replacement.
pub(crate) fn compose(
    base: &TensorTable,
    composite: &Composite,
    sources: &[TensorTable],
) -> Result<Composed, EngineError> {
    let mut chosen: HashMap<&str, usize> = HashMap::new();
    for (index, replacement) in composite.replacements.iter().enumerate() {
        for name in &replacement.objects {
            let name = name.as_str();
            if !name.starts_with(CHOOSABLE_PREFIX) || is_companion(name) {
                return Err(invalid(format!(
                    "'{name}' is not a projection a composite can take from a source"
                )));
            }
            let Some(mine) = base.get(name) else {
                return Err(invalid(format!("unknown object '{name}'")));
            };
            let Some(theirs) = sources[index].get(name) else {
                return Err(invalid(format!(
                    "{} has no object '{name}'",
                    replacement.source.display()
                )));
            };
            let (Some(want), Some(have)) = (features(mine), features(theirs)) else {
                return Err(invalid(format!("'{name}' is not a quantized linear object")));
            };
            if want != have {
                return Err(invalid(format!(
                    "'{name}' has {have:?} features in {} and {want:?} in the base",
                    replacement.source.display()
                )));
            }
            if chosen.insert(name, index).is_some() {
                return Err(invalid(format!("'{name}' is taken from more than one source")));
            }
        }
    }
    let replaced: HashSet<String> = chosen
        .keys()
        .flat_map(|name| COMPANIONS.map(|suffix| companion_name(name, suffix)))
        .collect();
    let mut entries = Vec::with_capacity(base.len());
    let mut source_of = HashMap::new();
    for entry in base.entries() {
        let name = entry.name();
        if let Some(&index) = chosen.get(name) {
            let table = &sources[index];
            let taken = iter::once(name.to_string())
                .chain(COMPANIONS.map(|suffix| companion_name(name, suffix)))
                .filter_map(|wanted| table.get(&wanted));
            for taken in taken {
                source_of.insert(taken.name().to_string(), index + 1);
                entries.push(taken.clone());
            }
        } else if !replaced.contains(name) {
            entries.push(entry.clone());
        }
    }
    let table = TensorTable::from_entries(entries).map_err(|error| invalid(error.to_string()))?;
    Ok(Composed { table, source_of })
}

#[cfg(test)]
mod tests {
    use euhedral_inference_edrl::{TensorDataType, WeightLayout};

    use super::*;

    const OBJECT: &str = "text/layers/0/mlp/down";
    const OTHER: &str = "text/layers/1/mlp/down";

    fn nvfp4(name: &str, offset: u64) -> TensorEntry {
        TensorEntry::new(
            name,
            vec![64, 128],
            TensorDataType::Uint8,
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128V1,
            offset,
            100,
        )
    }

    fn trellis(name: &str, offset: u64) -> [TensorEntry; 4] {
        let companion = |suffix: &str, dtype, format| {
            TensorEntry::new(
                companion_name(name, suffix),
                vec![1, 64],
                dtype,
                format,
                WeightLayout::ContiguousLeV1,
                offset + 10,
                8,
            )
        };
        [
            // 128 input features, 64 output features.
            TensorEntry::new(
                name,
                vec![8, 4, 48],
                TensorDataType::Uint8,
                WeightFormat::Trellis,
                WeightLayout::TrellisTile16V1,
                offset,
                10,
            ),
            companion(".suh", TensorDataType::Fp16, WeightFormat::Fp16),
            companion(".svh", TensorDataType::Fp16, WeightFormat::Fp16),
            companion(".segments", TensorDataType::Int32, WeightFormat::I32),
        ]
    }

    fn norm(offset: u64) -> TensorEntry {
        TensorEntry::new(
            "text/final_norm",
            vec![64],
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            WeightLayout::ContiguousLeV1,
            offset,
            128,
        )
    }

    fn table(entries: impl IntoIterator<Item = TensorEntry>) -> TensorTable {
        TensorTable::from_entries(entries.into_iter().collect()).unwrap()
    }

    fn take(objects: &[&str]) -> Composite {
        Composite {
            replacements: vec![Replacement {
                source: PathBuf::from("source.edrl"),
                objects: objects.iter().map(ToString::to_string).collect(),
            }],
        }
    }

    fn names(composed: &Composed) -> Vec<&str> {
        composed.table.entries().iter().map(TensorEntry::name).collect()
    }

    fn refusal(base: &TensorTable, composite: &Composite, source: TensorTable) -> String {
        compose(base, composite, &[source]).unwrap_err().to_string()
    }

    #[test]
    fn a_chosen_object_and_its_companions_come_from_the_source_in_the_base_order() {
        let base = table([nvfp4(OBJECT, 4096), norm(8192), nvfp4(OTHER, 12288)]);
        let source = table(trellis(OBJECT, 0).into_iter().chain([norm(900)]));
        let composed = compose(&base, &take(&[OBJECT]), &[source]).unwrap();
        assert_eq!(
            names(&composed),
            [
                OBJECT,
                "text/layers/0/mlp/down.suh",
                "text/layers/0/mlp/down.svh",
                "text/layers/0/mlp/down.segments",
                "text/final_norm",
                OTHER,
            ]
        );
        let taken = composed.table.get(OBJECT).unwrap();
        assert_eq!((taken.format(), taken.offset()), (WeightFormat::Trellis, 0));
        assert_eq!(
            composed.table.get("text/final_norm").unwrap().offset(),
            8192,
            "the base's norm stays"
        );
        assert_eq!(composed.table.get(OTHER).unwrap().offset(), 12288);
        let mut marked: Vec<(&str, usize)> = composed
            .source_of
            .iter()
            .map(|(name, index)| (name.as_str(), *index))
            .collect();
        marked.sort_unstable();
        assert_eq!(
            marked,
            [
                (OBJECT, 1),
                ("text/layers/0/mlp/down.segments", 1),
                ("text/layers/0/mlp/down.suh", 1),
                ("text/layers/0/mlp/down.svh", 1),
            ]
        );
    }

    #[test]
    fn the_companions_of_a_replaced_object_are_dropped_when_the_source_has_none() {
        let base = table(trellis(OBJECT, 0).into_iter().chain([norm(8192)]));
        let source = table([nvfp4(OBJECT, 4096)]);
        let composed = compose(&base, &take(&[OBJECT]), &[source]).unwrap();
        assert_eq!(names(&composed), [OBJECT, "text/final_norm"]);
        assert_eq!(composed.table.get(OBJECT).unwrap().format(), WeightFormat::Nvfp4);
    }

    #[test]
    fn each_source_marks_its_own_objects() {
        let base = table([nvfp4(OBJECT, 4096), nvfp4(OTHER, 8192)]);
        let composite = Composite {
            replacements: vec![
                Replacement {
                    source: PathBuf::from("a.edrl"),
                    objects: vec![OBJECT.to_string()],
                },
                Replacement {
                    source: PathBuf::from("b.edrl"),
                    objects: vec![OTHER.to_string()],
                },
            ],
        };
        let composed = compose(
            &base,
            &composite,
            &[table([nvfp4(OBJECT, 1)]), table([nvfp4(OTHER, 2)])],
        )
        .unwrap();
        assert_eq!(composed.source_of.get(OBJECT), Some(&1));
        assert_eq!(composed.source_of.get(OTHER), Some(&2));
        assert_eq!(composed.table.get(OBJECT).unwrap().offset(), 1);
        assert_eq!(composed.table.get(OTHER).unwrap().offset(), 2);
    }

    #[test]
    fn an_object_the_base_keeps_is_left_unmarked() {
        let base = table([nvfp4(OBJECT, 4096), nvfp4(OTHER, 8192)]);
        let composed = compose(&base, &take(&[OBJECT]), &[table([nvfp4(OBJECT, 1)])]).unwrap();
        assert!(!composed.source_of.contains_key(OTHER));
    }

    #[test]
    fn a_source_with_other_features_is_refused_naming_the_object() {
        let base = table([nvfp4(OBJECT, 4096)]);
        let wide = TensorEntry::new(
            OBJECT,
            vec![64, 256],
            TensorDataType::Uint8,
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128V1,
            0,
            100,
        );
        assert!(refusal(&base, &take(&[OBJECT]), table([wide])).contains(OBJECT));
    }

    #[test]
    fn only_quantized_projections_of_the_layers_can_be_taken() {
        let base = table([nvfp4(OBJECT, 4096), norm(8192), nvfp4("mtp/layer/mlp/down", 12288)]);
        let source = || table([nvfp4(OBJECT, 0), norm(0), nvfp4("mtp/layer/mlp/down", 0)]);
        for name in ["text/final_norm", "mtp/layer/mlp/down", "text/layers/0/mlp/down.suh"] {
            assert!(refusal(&base, &take(&[name]), source()).contains(name), "{name}");
        }
        let layer_norm = TensorEntry::new(
            "text/layers/0/input_norm",
            vec![64],
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            WeightLayout::ContiguousLeV1,
            0,
            128,
        );
        let base = table([layer_norm.clone()]);
        assert!(refusal(&base, &take(&["text/layers/0/input_norm"]), table([layer_norm])).contains("not a quantized"));
    }

    #[test]
    fn an_unknown_object_and_one_the_source_lacks_are_refused() {
        let base = table([nvfp4(OBJECT, 4096)]);
        assert!(refusal(&base, &take(&[OTHER]), table([nvfp4(OTHER, 0)])).contains("unknown object"));
        assert!(refusal(&base, &take(&[OBJECT]), table([nvfp4(OTHER, 0)])).contains("source.edrl has no object"));
    }

    #[test]
    fn an_object_named_twice_is_refused() {
        let base = table([nvfp4(OBJECT, 4096)]);
        assert!(refusal(&base, &take(&[OBJECT, OBJECT]), table([nvfp4(OBJECT, 0)])).contains("more than one"));
    }

    #[test]
    fn a_composite_without_objects_is_empty() {
        assert!(Composite::default().is_empty());
        assert!(take(&[]).is_empty());
        assert!(!take(&[OBJECT]).is_empty());
    }
}
