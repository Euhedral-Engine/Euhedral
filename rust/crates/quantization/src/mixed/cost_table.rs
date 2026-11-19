use std::collections::BTreeMap;

use euhedral_inference_edrl::{TensorEntry, WeightFormat};
use euhedral_inference_engine::model::trellis_logical;
use serde_json::Value;

use super::allocation::{Allocation, parse_json};
use super::error::MixedError;
use super::option::QuantOption;
use super::pretty::{Node, render};
use super::space::Sources;

/// The rows of the launch a prefill token is charged at.
pub const PREFILL_ROWS: u32 = 512;

/// The input and output features of a linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Shape {
    pub input: u32,
    pub output: u32,
}

/// One measurement: a launch of `option`'s linear of `shape` over `rows` rows. Ordered by input features, output
/// features, option, then rows, which is the order the table's file lists them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key {
    pub shape: Shape,
    pub option: QuantOption,
    pub rows: u32,
}

/// The features of the linear stored as `entry`: `[n, k]` for the row-split formats and, for trellis tiles,
/// `[k/16, n/16, w]`.
pub fn shape_of(entry: &TensorEntry) -> Result<Shape, MixedError> {
    let shape = entry.shape();
    let (input, output) = if entry.format() == WeightFormat::Trellis && shape.len() == 3 {
        let (n, k) = trellis_logical(shape)
            .ok_or_else(|| MixedError::invalid(format!("{} is not a trellis linear: shape {shape:?}", entry.name())))?;
        (k, n)
    } else if let [n, k] = shape {
        (*k, *n)
    } else {
        return Err(MixedError::invalid(format!(
            "{} is not a linear: shape rank {}",
            entry.name(),
            shape.len()
        )));
    };
    let narrow = |value: u64| {
        u32::try_from(value)
            .map_err(|_| MixedError::invalid(format!("{} has a feature count of {value}, too large", entry.name())))
    };
    Ok(Shape {
        input: narrow(input)?,
        output: narrow(output)?,
    })
}

/// The microseconds of one launch of a linear for each `(shape, option, rows)`, and the terms of a step that do not
/// depend on the allocation. The cost of an allocation is additive: the sum of its projections' launches plus the
/// fixed term.
///
/// The file is `{"fixed": {"prefillMicrosPerToken": x, "stepMicros": {"4": y}}, "entries": [{"in": k, "out": n,
/// "option": "trellis-3.0", "rows": 4, "micros": z}, ...]}`.
#[derive(Debug, Clone, PartialEq)]
pub struct CostTable {
    micros: BTreeMap<Key, f64>,
    fixed_prefill_micros_per_token: f64,
    fixed_step_micros: BTreeMap<u32, f64>,
}

impl CostTable {
    /// A table of `micros`; `fixed_prefill_micros_per_token` is added to a prefill token's cost and
    /// `fixed_step_micros` (by rows, absent meaning 0) to a verification step's.
    pub fn new(
        micros: BTreeMap<Key, f64>,
        fixed_prefill_micros_per_token: f64,
        fixed_step_micros: BTreeMap<u32, f64>,
    ) -> CostTable {
        CostTable {
            micros,
            fixed_prefill_micros_per_token,
            fixed_step_micros,
        }
    }

    /// This table with the fixed terms replaced, as measured end to end.
    pub fn with_fixed(&self, prefill_per_token: f64, step_by_rows: BTreeMap<u32, f64>) -> CostTable {
        CostTable::new(self.micros.clone(), prefill_per_token, step_by_rows)
    }

    /// The microseconds of one launch of `key`; the refusal names it when it was not measured.
    pub fn micros(&self, key: Key) -> Result<f64, MixedError> {
        self.micros.get(&key).copied().ok_or_else(|| {
            MixedError::invalid(format!(
                "no measured cost for {} at {} x {} with {} rows",
                key.option.key(),
                key.shape.input,
                key.shape.output,
                key.rows
            ))
        })
    }

    /// The prefill cost of a token under `allocation`: each projection's launch at [`PREFILL_ROWS`] rows over the
    /// rows, plus the fixed term.
    pub fn prefill_micros_per_token(&self, sources: &dyn Sources, allocation: &Allocation) -> Result<f64, MixedError> {
        let mut sum = 0.0;
        for name in sources.projections() {
            sum += self.micros(key(sources, allocation, name, PREFILL_ROWS)?)?;
        }
        Ok(sum / f64::from(PREFILL_ROWS) + self.fixed_prefill_micros_per_token)
    }

    /// The cost of a `rows`-row step under `allocation`: each projection's launch at `rows`, plus the fixed term of
    /// that row count.
    pub fn verify_micros(&self, sources: &dyn Sources, allocation: &Allocation, rows: u32) -> Result<f64, MixedError> {
        let mut sum = 0.0;
        for name in sources.projections() {
            sum += self.micros(key(sources, allocation, name, rows)?)?;
        }
        Ok(sum + self.fixed_step_micros.get(&rows).copied().unwrap_or(0.0))
    }

    /// The table in the text of a file.
    pub fn parse(text: &str) -> Result<CostTable, MixedError> {
        let root = parse_json(text)?;
        let entries = root
            .get("entries")
            .and_then(Value::as_array)
            .ok_or_else(|| MixedError::invalid("the cost table has no \"entries\""))?;
        let mut micros = BTreeMap::new();
        for entry in entries {
            let option = entry
                .get("option")
                .and_then(Value::as_str)
                .ok_or_else(|| MixedError::invalid("a cost entry has no option"))?;
            let key = Key {
                shape: Shape {
                    input: count(entry, "in")?,
                    output: count(entry, "out")?,
                },
                option: QuantOption::of(option)?,
                rows: count(entry, "rows")?,
            };
            let value = entry
                .get("micros")
                .and_then(Value::as_f64)
                .ok_or_else(|| MixedError::invalid("a cost entry has no micros"))?;
            micros.insert(key, value);
        }
        let mut prefill = 0.0;
        let mut step = BTreeMap::new();
        if let Some(fixed) = root.get("fixed") {
            prefill = fixed
                .get("prefillMicrosPerToken")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            if let Some(steps) = fixed.get("stepMicros").and_then(Value::as_object) {
                for (rows, value) in steps {
                    let rows = rows
                        .parse::<u32>()
                        .map_err(|_| MixedError::invalid(format!("the step cost has the row count '{rows}'")))?;
                    step.insert(rows, value.as_f64().unwrap_or(0.0));
                }
            }
        }
        Ok(CostTable::new(micros, prefill, step))
    }

    /// The text of this table's file, entries in a stable order.
    pub fn render(&self) -> String {
        let steps = self
            .fixed_step_micros
            .iter()
            .map(|(rows, value)| (rows.to_string(), Node::Float(*value)))
            .collect();
        let fixed = Node::Object(vec![
            (
                "prefillMicrosPerToken".to_string(),
                Node::Float(self.fixed_prefill_micros_per_token),
            ),
            ("stepMicros".to_string(), Node::Object(steps)),
        ]);
        let entries = self
            .micros
            .iter()
            .map(|(key, value)| {
                Node::Object(vec![
                    ("in".to_string(), Node::Int(i64::from(key.shape.input))),
                    ("out".to_string(), Node::Int(i64::from(key.shape.output))),
                    ("option".to_string(), Node::Text(key.option.key().to_string())),
                    ("rows".to_string(), Node::Int(i64::from(key.rows))),
                    ("micros".to_string(), Node::Float(*value)),
                ])
            })
            .collect();
        render(&Node::Object(vec![
            ("fixed".to_string(), fixed),
            ("entries".to_string(), Node::Array(entries)),
        ]))
    }
}

fn count(entry: &Value, field: &str) -> Result<u32, MixedError> {
    entry
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| MixedError::invalid(format!("a cost entry has no valid \"{field}\"")))
}

fn key(sources: &dyn Sources, allocation: &Allocation, name: &str, rows: u32) -> Result<Key, MixedError> {
    let option = allocation
        .choices()
        .get(name)
        .copied()
        .unwrap_or_else(|| sources.base());
    Ok(Key {
        shape: sources.shape(option, name)?,
        option,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use euhedral_inference_edrl::{TensorDataType, WeightLayout};

    use super::*;
    use crate::mixed::space::tests::TinySources;

    const SHAPE: Shape = Shape { input: 64, output: 64 };

    fn key_of(option: QuantOption, rows: u32) -> Key {
        Key {
            shape: SHAPE,
            option,
            rows,
        }
    }

    fn table() -> CostTable {
        CostTable::new(
            BTreeMap::from([
                (key_of(QuantOption::Nvfp4, 512), 51.2),
                (key_of(QuantOption::Trellis3_0, 512), 102.4),
                (key_of(QuantOption::Nvfp4, 4), 3.0),
                (key_of(QuantOption::Trellis3_0, 4), 5.0),
            ]),
            0.0,
            BTreeMap::new(),
        )
    }

    fn two_on_nvfp4() -> Allocation {
        Allocation::new(BTreeMap::from([
            ("text/layers/0/mlp/down".to_string(), QuantOption::Nvfp4),
            ("text/layers/1/mlp/down".to_string(), QuantOption::Nvfp4),
        ]))
    }

    #[test]
    fn a_known_table_and_allocation_give_exact_sums() {
        let sources = TinySources::new();
        let allocation = two_on_nvfp4();
        let prefill = table().prefill_micros_per_token(&sources, &allocation).unwrap();
        assert!((prefill - (2.0 * 51.2 / 512.0 + 2.0 * 102.4 / 512.0)).abs() < 1e-12);
        let verify = table().verify_micros(&sources, &allocation, 4).unwrap();
        assert!((verify - (2.0 * 3.0 + 2.0 * 5.0)).abs() < 1e-12);
    }

    #[test]
    fn the_fixed_terms_are_added_once_and_leave_the_original_unchanged() {
        let sources = TinySources::new();
        let allocation = Allocation::default();
        let fixed = table().with_fixed(7.5, BTreeMap::from([(4, 11.0)]));
        let prefill = fixed.prefill_micros_per_token(&sources, &allocation).unwrap();
        assert!((prefill - (4.0 * 102.4 / 512.0 + 7.5)).abs() < 1e-12);
        assert!((fixed.verify_micros(&sources, &allocation, 4).unwrap() - (4.0 * 5.0 + 11.0)).abs() < 1e-12);
        assert!((table().verify_micros(&sources, &allocation, 4).unwrap() - 20.0).abs() < 1e-12);
    }

    #[test]
    fn a_missing_entry_is_refused_naming_it() {
        let sources = TinySources::new();
        let failure = table()
            .verify_micros(&sources, &Allocation::default(), 3)
            .unwrap_err()
            .to_string();
        assert!(
            failure.contains("trellis-3.0") && failure.contains("64") && failure.contains('3'),
            "{failure}"
        );
    }

    #[test]
    fn the_table_round_trips_through_its_file() {
        let fixed = table().with_fixed(7.5, BTreeMap::from([(4, 11.0), (1, 2.5)]));
        let read = CostTable::parse(&fixed.render()).unwrap();
        assert_eq!(read, fixed);
        assert_eq!(read.micros(key_of(QuantOption::Nvfp4, 512)).unwrap(), 51.2);
        assert_eq!(read.render(), fixed.render());
    }

    #[test]
    fn the_file_lists_entries_by_features_option_and_rows() {
        let small = Shape { input: 8, output: 128 };
        let table = CostTable::new(
            BTreeMap::from([
                (key_of(QuantOption::Trellis3_0, 4), 5.0),
                (key_of(QuantOption::Nvfp4, 512), 0.0001),
                (
                    Key {
                        shape: small,
                        option: QuantOption::Trellis4_0,
                        rows: 1,
                    },
                    1.0,
                ),
                (key_of(QuantOption::Nvfp4, 4), 3.0),
            ]),
            0.0,
            BTreeMap::new(),
        );
        assert_eq!(
            table.render(),
            "{\n  \"fixed\" : {\n    \"prefillMicrosPerToken\" : 0.0,\n    \"stepMicros\" : { }\n  },\n  \"entries\" : [ {\n    \"in\" : 8,\n    \"out\" : 128,\n    \"option\" : \"trellis-4.0\",\n    \"rows\" : 1,\n    \"micros\" : 1.0\n  }, {\n    \"in\" : 64,\n    \"out\" : 64,\n    \"option\" : \"nvfp4\",\n    \"rows\" : 4,\n    \"micros\" : 3.0\n  }, {\n    \"in\" : 64,\n    \"out\" : 64,\n    \"option\" : \"nvfp4\",\n    \"rows\" : 512,\n    \"micros\" : 1.0E-4\n  }, {\n    \"in\" : 64,\n    \"out\" : 64,\n    \"option\" : \"trellis-3.0\",\n    \"rows\" : 4,\n    \"micros\" : 5.0\n  } ]\n}\n"
        );
    }

    #[test]
    fn a_file_without_entries_or_with_a_bad_entry_is_refused() {
        assert!(CostTable::parse("{}").is_err());
        assert!(
            CostTable::parse(
                "{\"entries\": [{\"in\": 1, \"out\": 1, \"option\": \"fp8\", \"rows\": 1, \"micros\": 1}]}"
            )
            .is_err()
        );
        assert!(
            CostTable::parse("{\"entries\": [{\"in\": 1, \"out\": 1, \"option\": \"nvfp4\", \"micros\": 1}]}").is_err()
        );
    }

    fn entry(shape: &[u64], format: WeightFormat, layout: WeightLayout) -> TensorEntry {
        TensorEntry::new(
            "t".to_string(),
            shape.to_vec(),
            TensorDataType::Bf16,
            format,
            layout,
            0,
            0,
        )
    }

    #[test]
    fn features_come_from_the_stored_shape_of_each_format() {
        let trellis = entry(&[320, 1120, 56], WeightFormat::Trellis, WeightLayout::TrellisTile16V1);
        assert_eq!(
            shape_of(&trellis).unwrap(),
            Shape {
                input: 5120,
                output: 17920
            }
        );
        let rows = entry(&[17920, 5120], WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1);
        assert_eq!(
            shape_of(&rows).unwrap(),
            Shape {
                input: 5120,
                output: 17920
            }
        );
        let flat = entry(&[5120], WeightFormat::Bf16, WeightLayout::ContiguousLeV1);
        assert!(shape_of(&flat).is_err());
    }
}
