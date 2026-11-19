use std::collections::{BTreeMap, HashMap};

use super::allocation::Allocation;
use super::cost_table::{CostTable, Key, PREFILL_ROWS, Shape};
use super::error::MixedError;
use super::option::QuantOption;

/// What the search needs of the source artifacts: which objects can be chosen, which option the others keep, and,
/// for each option of an object, its linear's features and the bytes of the object with its companions.
pub trait Sources {
    /// The option every object an allocation does not name keeps.
    fn base(&self) -> QuantOption;
    /// The names of the choosable projection objects.
    fn projections(&self) -> &[String];
    /// The features of `object`'s linear as `option` stores it.
    fn shape(&self, option: QuantOption, object: &str) -> Result<Shape, MixedError>;
    /// The bytes of `object` as `option` stores it, with the companions that travel with it.
    fn bytes(&self, option: QuantOption, object: &str) -> Result<i64, MixedError>;
}

/// `names` ordered by layer number, then by name: the order the fidelity model numbers its objects in. The names are
/// `text/layers/<n>/...`.
fn total(mut counts: impl Iterator<Item = i64>) -> Result<i64, MixedError> {
    counts
        .try_fold(0i64, i64::checked_add)
        .ok_or_else(|| MixedError::invalid("the bytes of the objects do not fit a 64-bit count"))
}

pub fn sorted_by_layer(names: &[String]) -> Result<Vec<String>, MixedError> {
    let mut keyed = Vec::with_capacity(names.len());
    for name in names {
        let layer = name
            .split('/')
            .nth(2)
            .and_then(|part| part.parse::<i64>().ok())
            .ok_or_else(|| MixedError::invalid(format!("{name} is not a layer object (text/layers/<n>/...)")))?;
        keyed.push((layer, name.clone()));
    }
    keyed.sort();
    Ok(keyed.into_iter().map(|(_, name)| name).collect())
}

/// The choosable objects of a search in the order the fidelity model numbers them, the options that can be chosen,
/// and what each choice costs: the bytes of the object with its companions, and the prefill and verification
/// microseconds the cost table gives its launch.
#[derive(Debug, Clone)]
pub struct Space {
    base: QuantOption,
    objects: Vec<String>,
    options: Vec<QuantOption>,
    bytes: Vec<Vec<i64>>,
    prefill: Vec<Vec<f64>>,
    decode: Vec<Vec<f64>>,
    index: HashMap<String, usize>,
    option_index: BTreeMap<QuantOption, usize>,
}

impl Space {
    /// The space of `sources`' projections over `options`, costed by `costs`, verifying `verify_rows` rows per step.
    /// The refusal names the cost-table entry that is missing.
    pub fn new(
        sources: &dyn Sources,
        options: &[QuantOption],
        costs: &CostTable,
        verify_rows: u32,
    ) -> Result<Space, MixedError> {
        let objects = sorted_by_layer(sources.projections())?;
        let (n, k) = (objects.len(), options.len());
        let mut bytes = vec![vec![0i64; k]; n];
        let mut prefill = vec![vec![0.0; k]; n];
        let mut decode = vec![vec![0.0; k]; n];
        let mut index = HashMap::new();
        for (i, object) in objects.iter().enumerate() {
            index.insert(object.clone(), i);
            for (j, option) in options.iter().enumerate() {
                let stored = sources.bytes(*option, object)?;
                if stored < 0 {
                    return Err(MixedError::invalid(format!(
                        "{object} as {} has {stored} bytes",
                        option.key()
                    )));
                }
                bytes[i][j] = stored;
                let shape = sources.shape(*option, object)?;
                prefill[i][j] = costs.micros(Key {
                    shape,
                    option: *option,
                    rows: PREFILL_ROWS,
                })? / f64::from(PREFILL_ROWS);
                decode[i][j] = costs.micros(Key {
                    shape,
                    option: *option,
                    rows: verify_rows,
                })?;
            }
        }
        let option_index = options.iter().enumerate().map(|(j, option)| (*option, j)).collect();
        Ok(Space {
            base: sources.base(),
            objects,
            options: options.to_vec(),
            bytes,
            prefill,
            decode,
            index,
            option_index,
        })
    }

    /// The objects, in the order the fidelity model numbers them.
    pub fn objects(&self) -> &[String] {
        &self.objects
    }

    /// The options that can be chosen; a choice is an index into this list.
    pub fn options(&self) -> &[QuantOption] {
        &self.options
    }

    /// The bytes of each object (rows) as each option (columns) stores it.
    pub fn bytes(&self) -> &[Vec<i64>] {
        &self.bytes
    }

    /// The microseconds of one prefill token's share of each object's launch, by object and option.
    pub fn prefill(&self) -> &[Vec<f64>] {
        &self.prefill
    }

    /// The microseconds of one verification step's launch of each object, by object and option.
    pub fn decode(&self) -> &[Vec<f64>] {
        &self.decode
    }

    /// The allocation that gives each object the option its index in `choice` names.
    pub fn allocation(&self, choice: &[usize]) -> Allocation {
        Allocation::new(
            choice
                .iter()
                .enumerate()
                .map(|(i, option)| (self.objects[i].clone(), self.options[*option]))
                .collect(),
        )
    }

    /// The option indices of a full allocation; the refusal names an object it does not choose for, or an option this
    /// space does not use.
    pub fn choice(&self, allocation: &Allocation) -> Result<Vec<usize>, MixedError> {
        self.objects
            .iter()
            .map(|object| {
                let option = allocation
                    .choices()
                    .get(object)
                    .ok_or_else(|| MixedError::invalid(format!("the allocation has no option for {object}")))?;
                self.option_index.get(option).copied().ok_or_else(|| {
                    MixedError::invalid(format!(
                        "this search does not use {} (chosen for {object})",
                        option.key()
                    ))
                })
            })
            .collect()
    }

    /// `allocation` with every object it does not name on the base's option. The refusal names an object that is not a
    /// choosable projection or an option this search does not use.
    pub fn normalize(&self, allocation: &Allocation) -> Result<Allocation, MixedError> {
        for (object, option) in allocation.choices() {
            if !self.index.contains_key(object) {
                return Err(MixedError::invalid(format!(
                    "{object} is not a projection that can be chosen"
                )));
            }
            if !self.option_index.contains_key(option) {
                return Err(MixedError::invalid(format!(
                    "no source and prior for {} (chosen for {object})",
                    option.key()
                )));
            }
        }
        Ok(Allocation::new(
            self.objects
                .iter()
                .map(|object| {
                    (
                        object.clone(),
                        allocation.choices().get(object).copied().unwrap_or(self.base),
                    )
                })
                .collect(),
        ))
    }

    /// The bytes of the objects under `choice`.
    /// The refusal when the sum does not fit a signed 64-bit count.
    pub fn bytes_of(&self, choice: &[usize]) -> Result<i64, MixedError> {
        total(choice.iter().enumerate().map(|(i, option)| self.bytes[i][*option]))
    }

    /// The option that stores `object` in the fewest bytes; the first of equals.
    pub fn cheapest(&self, object: usize) -> usize {
        let row = &self.bytes[object];
        (1..row.len()).fold(0, |best, j| if row[j] < row[best] { j } else { best })
    }

    /// The option that stores `object` in the most bytes; the first of equals.
    pub fn dearest(&self, object: usize) -> usize {
        let row = &self.bytes[object];
        (1..row.len()).fold(0, |best, j| if row[j] > row[best] { j } else { best })
    }

    /// The bytes when every object takes its cheapest option.
    pub fn minimum_bytes(&self) -> Result<i64, MixedError> {
        total((0..self.objects.len()).map(|i| self.bytes[i][self.cheapest(i)]))
    }

    /// The bytes when every object takes its dearest option.
    pub fn maximum_bytes(&self) -> Result<i64, MixedError> {
        total((0..self.objects.len()).map(|i| self.bytes[i][self.dearest(i)]))
    }

    /// The choice that gives every object `option`; the refusal when this space does not use it.
    pub fn uniform(&self, option: QuantOption) -> Result<Vec<usize>, MixedError> {
        let at = self
            .option_index
            .get(&option)
            .ok_or_else(|| MixedError::invalid(format!("this search does not use {}", option.key())))?;
        Ok(vec![*at; self.objects.len()])
    }

    /// One allocation per option, every object on it.
    pub fn pure(&self) -> Vec<Allocation> {
        (0..self.options.len())
            .map(|option| self.allocation(&vec![option; self.objects.len()]))
            .collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Four projections of 64 x 64 over two layers, bases on trellis 3.0, with bytes that differ by option and object.
    pub(crate) struct TinySources {
        projections: Vec<String>,
    }

    impl TinySources {
        pub(crate) fn new() -> TinySources {
            let projections = [
                "text/layers/1/mlp/gate_up",
                "text/layers/1/mlp/down",
                "text/layers/0/mlp/gate_up",
                "text/layers/0/mlp/down",
            ]
            .iter()
            .map(|name| name.to_string())
            .collect();
            TinySources { projections }
        }
    }

    impl Sources for TinySources {
        fn base(&self) -> QuantOption {
            QuantOption::Trellis3_0
        }

        fn projections(&self) -> &[String] {
            &self.projections
        }

        fn shape(&self, _option: QuantOption, _object: &str) -> Result<Shape, MixedError> {
            Ok(Shape { input: 64, output: 64 })
        }

        fn bytes(&self, option: QuantOption, object: &str) -> Result<i64, MixedError> {
            let object_bytes = if object.ends_with("down") { 10 } else { 20 };
            Ok(match option {
                QuantOption::Nvfp4 => 100 + object_bytes,
                QuantOption::Trellis3_0 => 70 + object_bytes,
                other => 50 + other.key().len() as i64 + object_bytes,
            })
        }
    }

    fn costs() -> CostTable {
        let mut micros = BTreeMap::new();
        for (option, prefill, decode) in [(QuantOption::Nvfp4, 51.2, 3.0), (QuantOption::Trellis3_0, 102.4, 5.0)] {
            let shape = Shape { input: 64, output: 64 };
            micros.insert(
                Key {
                    shape,
                    option,
                    rows: PREFILL_ROWS,
                },
                prefill,
            );
            micros.insert(Key { shape, option, rows: 4 }, decode);
        }
        CostTable::new(micros, 0.0, BTreeMap::new())
    }

    fn space() -> Space {
        Space::new(
            &TinySources::new(),
            &[QuantOption::Nvfp4, QuantOption::Trellis3_0],
            &costs(),
            4,
        )
        .unwrap()
    }

    #[test]
    fn objects_are_numbered_by_layer_then_name() {
        let names: Vec<String> = [
            "text/layers/10/a",
            "text/layers/2/b",
            "text/layers/2/a",
            "text/layers/0/z",
        ]
        .iter()
        .map(|name| name.to_string())
        .collect();
        assert_eq!(
            sorted_by_layer(&names).unwrap(),
            [
                "text/layers/0/z",
                "text/layers/2/a",
                "text/layers/2/b",
                "text/layers/10/a"
            ]
        );
        assert!(sorted_by_layer(&["mtp/layer/mlp/down".to_string()]).is_err());
    }

    #[test]
    fn the_space_costs_each_choice_from_the_table_and_the_sources() {
        let space = space();
        assert_eq!(
            space.objects(),
            [
                "text/layers/0/mlp/down",
                "text/layers/0/mlp/gate_up",
                "text/layers/1/mlp/down",
                "text/layers/1/mlp/gate_up"
            ]
        );
        assert_eq!(space.bytes()[0], [110, 80]);
        assert_eq!(space.bytes()[1], [120, 90]);
        assert_eq!(space.prefill()[0], [51.2 / 512.0, 102.4 / 512.0]);
        assert_eq!(space.decode()[3], [3.0, 5.0]);
        assert_eq!(space.minimum_bytes().unwrap(), 80 + 90 + 80 + 90);
        assert_eq!(space.maximum_bytes().unwrap(), 110 + 120 + 110 + 120);
        assert_eq!((space.cheapest(0), space.dearest(0)), (1, 0));
        assert_eq!(space.bytes_of(&[0, 1, 1, 0]).unwrap(), 110 + 90 + 80 + 120);
    }

    #[test]
    fn a_missing_cost_entry_is_refused_naming_it() {
        let failure = Space::new(
            &TinySources::new(),
            &[QuantOption::Nvfp4, QuantOption::Trellis3_0],
            &costs(),
            3,
        )
        .unwrap_err()
        .to_string();
        assert!(failure.contains("3 rows"), "{failure}");
    }

    struct Wrapped(TinySources, i64);

    impl Sources for Wrapped {
        fn base(&self) -> QuantOption {
            self.0.base()
        }
        fn projections(&self) -> &[String] {
            self.0.projections()
        }
        fn shape(&self, option: QuantOption, object: &str) -> Result<Shape, MixedError> {
            self.0.shape(option, object)
        }
        fn bytes(&self, option: QuantOption, object: &str) -> Result<i64, MixedError> {
            Ok(self.0.bytes(option, object)? * self.1)
        }
    }

    #[test]
    fn negative_bytes_are_refused() {
        let failure = Space::new(
            &Wrapped(TinySources::new(), -1),
            &[QuantOption::Nvfp4, QuantOption::Trellis3_0],
            &costs(),
            4,
        )
        .unwrap_err()
        .to_string();
        assert!(failure.contains("-"), "{failure}");
    }

    #[test]
    fn sums_that_overflow_are_refused_not_panicked() {
        let huge = Space::new(
            &Wrapped(TinySources::new(), i64::MAX / 150),
            &[QuantOption::Nvfp4, QuantOption::Trellis3_0],
            &costs(),
            4,
        )
        .unwrap();
        assert!(huge.maximum_bytes().is_err());
        assert!(huge.bytes_of(&[0, 0, 0, 0]).is_err());
        assert!(huge.minimum_bytes().is_err());
    }

    #[test]
    fn choices_and_allocations_convert_both_ways() {
        let space = space();
        let allocation = space.allocation(&[0, 1, 1, 0]);
        assert_eq!(space.choice(&allocation).unwrap(), [0, 1, 1, 0]);
        assert_eq!(space.pure().len(), 2);
        assert_eq!(
            space.choice(&space.pure()[1]).unwrap(),
            space.uniform(QuantOption::Trellis3_0).unwrap()
        );
        assert!(space.uniform(QuantOption::Trellis4_0).is_err());
    }

    #[test]
    fn a_partial_allocation_takes_the_base_for_what_it_leaves_out() {
        let space = space();
        let partial = Allocation::new(BTreeMap::from([(
            "text/layers/1/mlp/down".to_string(),
            QuantOption::Nvfp4,
        )]));
        let full = space.normalize(&partial).unwrap();
        assert_eq!(space.choice(&full).unwrap(), [1, 1, 0, 1]);
        assert!(space.choice(&partial).is_err());
        let stranger = Allocation::new(BTreeMap::from([(
            "text/layers/7/mlp/down".to_string(),
            QuantOption::Nvfp4,
        )]));
        assert!(
            space
                .normalize(&stranger)
                .unwrap_err()
                .to_string()
                .contains("text/layers/7/mlp/down")
        );
        let unused = Allocation::new(BTreeMap::from([(
            "text/layers/1/mlp/down".to_string(),
            QuantOption::Trellis4_0,
        )]));
        assert!(
            space
                .normalize(&unused)
                .unwrap_err()
                .to_string()
                .contains("trellis-4.0")
        );
    }
}
