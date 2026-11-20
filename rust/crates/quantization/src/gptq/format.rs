//! The storage formats GPTQ rounds into, the NVFP4 scale search's reach, and the host side of a Hessian's
//! preparation: the column order and the damped diagonal.

use thiserror::Error;

/// Columns rounded between two error-feedback products.
pub const BLOCK: usize = 128;
/// How many E4M3 codes under the nearest one a block scale may take.
pub const SEARCH_BELOW: i32 = 6;
/// The largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;
/// The largest finite E4M3 magnitude.
const E4M3_MAX: f32 = 448.0;

/// A request GPTQ cannot serve.
#[derive(Debug, Error, PartialEq)]
pub enum FormatError {
    #[error("unknown format {0}")]
    UnknownFormat(String),
    #[error("invalid NVFP4 search: above {above}, scale max {scale_max}")]
    Search { above: i32, scale_max: f32 },
}

/// The NVFP4 block-scale search: a block may take a scale up to `above` E4M3 codes over the nearest one (always down
/// to [`SEARCH_BELOW`] under it), and the tensor's global scale maps its largest magnitude to E2M1's 6 times
/// `scale_max` (E4M3's 448 by default; a smaller one leaves the blocks with the largest values codes to search above).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nvfp4Search {
    above: i32,
    scale_max: f32,
}

impl Nvfp4Search {
    pub const DEFAULT: Nvfp4Search = Nvfp4Search {
        above: 2,
        scale_max: E4M3_MAX,
    };

    pub fn new(above: i32, scale_max: f32) -> Result<Nvfp4Search, FormatError> {
        // The negated comparisons also refuse NaN.
        if above < 0 || !(scale_max > 0.0 && scale_max <= E4M3_MAX) {
            return Err(FormatError::Search { above, scale_max });
        }
        Ok(Nvfp4Search { above, scale_max })
    }

    pub fn above(&self) -> i32 {
        self.above
    }

    pub fn scale_max(&self) -> f32 {
        self.scale_max
    }
}

impl Default for Nvfp4Search {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A storage format: grouped (Q3, Q4 or Q5 over groups of 64, codes `qmin..=qmax`) or NVFP4 (blocks of 16,
/// optionally with an SD4 table).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Format {
    pub name: String,
    pub layout: String,
    pub group: usize,
    pub qmin: i32,
    pub qmax: i32,
    pub nvfp4: bool,
    pub sd4: bool,
}

impl Format {
    /// The format `name` stored in `layout`; a grouped format's lowest code may be raised to `lowest_code`.
    pub fn of(name: &str, layout: &str, lowest_code: Option<i32>) -> Result<Format, FormatError> {
        let sd4 = layout.ends_with("sd4-v1");
        let grouped = |qmin: i32, qmax: i32| Format {
            name: name.to_string(),
            layout: layout.to_string(),
            group: 64,
            qmin: lowest_code.map_or(qmin, |lowest| qmin.max(lowest)),
            qmax,
            nvfp4: false,
            sd4: false,
        };
        match name {
            "NVFP4" => Ok(Format {
                name: name.to_string(),
                layout: layout.to_string(),
                group: 16,
                qmin: 0,
                qmax: 0,
                nvfp4: true,
                sd4,
            }),
            "Q3G64_F16S" => Ok(grouped(-4, 3)),
            "Q4G64_F16S" => Ok(grouped(-8, 7)),
            "Q5G64_F16S" => Ok(grouped(-16, 15)),
            other => Err(FormatError::UnknownFormat(other.to_string())),
        }
    }

    /// The layout the state is stored in: NVFP4 keeps its own, the grouped formats are split by rows in blocks of 128.
    pub fn state_layout(&self) -> &str {
        if self.nvfp4 { &self.layout } else { "row-split-k128-v1" }
    }
}

/// The column order: natural, or by decreasing importance (stable, so equal columns keep their place).
pub fn column_order(importance: &[f32], act_order: bool) -> Vec<u32> {
    let mut order: Vec<u32> = (0..importance.len() as u32).collect();
    if act_order {
        order.sort_by(|&a, &b| {
            let (a, b) = (-f64::from(importance[a as usize]), -f64::from(importance[b as usize]));
            a.total_cmp(&b)
        });
    }
    order
}

/// Where each column sits in `order`.
pub fn positions(order: &[u32]) -> Vec<u32> {
    let mut position = vec![0u32; order.len()];
    for (place, &column) in order.iter().enumerate() {
        position[column as usize] = place as u32;
    }
    position
}

/// The damped diagonal of the permuted Hessian: a dead column (importance 0) gets 1, and every entry gains `damp`
/// times the mean of the diagonal. The mean is summed in float64 and rounded once.
pub fn damped_diagonal(importance: &[f32], order: &[u32], damp: f32) -> Vec<f32> {
    let mut diagonal: Vec<f32> = order
        .iter()
        .map(|&column| match importance[column as usize] {
            0.0 => 1.0,
            value => value,
        })
        .collect();
    let sum: f64 = diagonal.iter().map(|&value| f64::from(value)).sum();
    let mean = (sum / diagonal.len() as f64) as f32;
    let add = damp * mean;
    for value in &mut diagonal {
        *value += add;
    }
    diagonal
}

/// The columns, in rounding order, whose importance is zero.
pub fn dead_columns(importance: &[f32], order: &[u32]) -> Vec<u32> {
    (0..order.len() as u32)
        .filter(|&place| importance[order[place as usize] as usize] == 0.0)
        .collect()
}

/// The group of the column at each place of the rounding order.
pub fn group_of_places(order: &[u32], group: usize) -> Vec<u32> {
    order.iter().map(|&column| column / group as u32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_natural_order_is_the_identity_and_act_order_is_stable_by_decreasing_importance() {
        let importance = [1.0, 3.0, 3.0, 0.0, 2.0];
        assert_eq!(column_order(&importance, false), [0, 1, 2, 3, 4]);
        let order = column_order(&importance, true);
        assert_eq!(order, [1, 2, 4, 0, 3]);
        let position = positions(&order);
        for (place, &column) in order.iter().enumerate() {
            assert_eq!(position[column as usize] as usize, place);
        }
    }

    #[test]
    fn a_dead_column_gets_a_unit_diagonal_and_the_damping_is_a_fraction_of_the_mean() {
        let importance = [4.0, 0.0, 2.0, 2.0];
        let order = column_order(&importance, false);
        // The diagonal is 4, 1, 2, 2: mean 2.25, so a damping of 0.5 adds 1.125.
        assert_eq!(damped_diagonal(&importance, &order, 0.5), [5.125, 2.125, 3.125, 3.125]);
        assert_eq!(dead_columns(&importance, &order), [1]);
        let order = column_order(&importance, true);
        assert_eq!(dead_columns(&importance, &order), [3]);
    }

    #[test]
    fn the_search_refuses_what_the_format_cannot_hold() {
        assert!(Nvfp4Search::new(0, 448.0).is_ok());
        assert!(Nvfp4Search::new(-1, 448.0).is_err());
        assert!(Nvfp4Search::new(2, 0.0).is_err());
        assert!(Nvfp4Search::new(2, 449.0).is_err());
        assert!(Nvfp4Search::new(2, f32::NAN).is_err());
        assert_eq!(Nvfp4Search::default(), Nvfp4Search::new(2, 448.0).unwrap());
    }

    #[test]
    fn formats_carry_their_code_range_and_group() {
        let q3 = Format::of("Q3G64_F16S", "row-split-k128-v1", None).unwrap();
        assert_eq!((q3.group, q3.qmin, q3.qmax, q3.nvfp4), (64, -4, 3, false));
        let raised = Format::of("Q5G64_F16S", "row-split-k128-v1", Some(-3)).unwrap();
        assert_eq!((raised.qmin, raised.qmax), (-3, 15));
        let lowered = Format::of("Q4G64_F16S", "row-split-k128-v1", Some(-20)).unwrap();
        assert_eq!(lowered.qmin, -8);
        let nvfp4 = Format::of("NVFP4", "row-split-k128-sd4-v1", None).unwrap();
        assert_eq!((nvfp4.group, nvfp4.nvfp4, nvfp4.sd4), (16, true, true));
        assert_eq!(nvfp4.state_layout(), "row-split-k128-sd4-v1");
        assert_eq!(q3.state_layout(), "row-split-k128-v1");
        assert_eq!(
            Format::of("Q9", "x", None),
            Err(FormatError::UnknownFormat("Q9".to_string()))
        );
    }
}
