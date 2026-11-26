//! The Hessians of a layer's linears: one matrix per distinct input, summed over the batches of a pass.

use std::collections::HashMap;

use super::CalibrateError;

/// Where a linear's Hessian is and whether this linear adds to it.
#[derive(Debug, Clone, Copy)]
struct Place {
    matrix: usize,
    /// A linear that reads the rows the one before it read shares that one's matrix and adds nothing: the sum of
    /// the rows' products is the same.
    owns: bool,
}

/// The rows a linear read: the device address of its input and their number and width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Input {
    address: u64,
    rows: u32,
    columns: u32,
}

/// The Hessians of one layer's linears, `M` being a matrix of the device (or of a host fake).
///
/// The observer calls [`observe`](HessianSet::observe) with each linear's input, in the order the layer applies them,
/// once per batch of the pass. The first batch makes each matrix, zeroed, and finds which linears share one.
#[derive(Debug)]
pub struct HessianSet<M> {
    matrices: Vec<M>,
    places: HashMap<String, Place>,
    names: Vec<String>,
    previous: Option<(Input, usize)>,
}

impl<M> Default for HessianSet<M> {
    fn default() -> Self {
        HessianSet {
            matrices: Vec::new(),
            places: HashMap::new(),
            names: Vec::new(),
            previous: None,
        }
    }
}

impl<M> HessianSet<M> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Takes one linear's input: makes the linear's matrix on its first sight (`make(columns)`, zeroed), and has
    /// `add(matrix)` add the input's products unless the matrix is another linear's.
    pub fn observe(
        &mut self,
        object: &str,
        input: (u64, u32, u32),
        make: impl FnOnce(u32) -> Result<M, CalibrateError>,
        add: impl FnOnce(&mut M) -> Result<(), CalibrateError>,
    ) -> Result<(), CalibrateError> {
        let seen = Input {
            address: input.0,
            rows: input.1,
            columns: input.2,
        };
        let place = match self.places.get(object) {
            Some(place) => *place,
            None => {
                let sharing = self
                    .previous
                    .filter(|(previous, _)| *previous == seen)
                    .map(|(_, matrix)| matrix);
                let place = match sharing {
                    Some(matrix) => Place { matrix, owns: false },
                    None => {
                        self.matrices.push(make(seen.columns)?);
                        Place {
                            matrix: self.matrices.len() - 1,
                            owns: true,
                        }
                    }
                };
                self.places.insert(object.to_owned(), place);
                self.names.push(object.to_owned());
                place
            }
        };
        if place.owns {
            add(&mut self.matrices[place.matrix])?;
            self.previous = Some((seen, place.matrix));
        }
        Ok(())
    }

    /// The matrix of `object`, once it was observed.
    pub fn matrix(&self, object: &str) -> Option<&M> {
        self.places.get(object).map(|place| &self.matrices[place.matrix])
    }

    /// The distinct matrices, in the order they were made.
    pub fn matrices(&self) -> &[M] {
        &self.matrices
    }

    /// The distinct matrices, to be completed and freed.
    pub fn into_matrices(self) -> Vec<M> {
        self.matrices
    }

    /// The linears seen, in the order they were first observed.
    pub fn objects(&self) -> &[String] {
        &self.names
    }

    /// Whether `object` and `other` are summed in one matrix.
    pub fn shares(&self, object: &str, other: &str) -> bool {
        matches!((self.places.get(object), self.places.get(other)), (Some(a), Some(b)) if a.matrix == b.matrix)
    }
}
