use super::super::allocation::Allocation;
use super::super::pareto;
use super::super::search_state::Measured;

/// A candidate proposed for scoring: where it came from, what the ensemble predicted of it and how unsure it was, the
/// RCO objective at it, and the cell of the lambda sweep that proposed it (none for a seed or a move).
#[derive(Debug, Clone)]
pub(super) struct Candidate {
    pub allocation: Allocation,
    pub origin: String,
    pub predicted: f64,
    pub spread: f64,
    pub value: f64,
    pub cell: Option<usize>,
}

impl Candidate {
    /// A candidate of a seed or an import: nothing is predicted of it.
    pub fn given(allocation: Allocation, origin: impl Into<String>) -> Candidate {
        Candidate {
            allocation,
            origin: origin.into(),
            predicted: f64::NAN,
            spread: 0.0,
            value: 0.0,
            cell: None,
        }
    }

    /// The same candidate labelled as picked for `kind`.
    pub fn relabel(&self, kind: &str) -> Candidate {
        Candidate {
            origin: format!("{kind}:{}", self.origin),
            ..self.clone()
        }
    }
}

/// A candidate assembled and checked: its resident bytes, whether it fits the device, and the cost table's two
/// predictions.
#[derive(Debug, Clone)]
pub(super) struct Prepared {
    pub candidate: Candidate,
    pub bytes: i64,
    pub prefill: f64,
    pub decode: f64,
    pub fits: bool,
}

impl pareto::Candidate for Measured {
    fn kl(&self) -> f64 {
        self.kl
    }

    fn prefill(&self) -> f64 {
        self.prefill
    }

    fn decode(&self) -> f64 {
        self.decode
    }

    fn fits(&self) -> bool {
        self.fits
    }
}

/// The front of `data`, as the measured candidates themselves.
pub(super) fn front_of(data: &[Measured]) -> Vec<Measured> {
    pareto::front(data).into_iter().map(|at| data[at].clone()).collect()
}
