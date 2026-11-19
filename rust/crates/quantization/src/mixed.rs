//! The data of the mixed allocation search: what each projection object can be stored as ([`QuantOption`]), the choice made
//! for each ([`Allocation`]), the local error of each option ([`Priors`]), the measured cost of a launch
//! ([`CostTable`]), the combinatorial solvers ([`knapsack`], [`pareto`]) and the residency check ([`feasibility`]).
//!
//! Nothing here opens a file: every reader takes the text of a file and every writer returns the text to write, so
//! the caller reads and writes through the runtime's asynchronous I/O.
//!
//! The objects a search chooses over come from the source artifacts, which [`assemble`](crate::assemble) opens. What the search needs
//! of them is the narrow [`Sources`] trait.

mod allocation;
mod cost_table;
mod error;
pub mod feasibility;
pub mod knapsack;
mod option;
pub mod pareto;
mod pretty;
mod priors;
mod seeds;
mod space;
pub mod synthetic;

pub use allocation::Allocation;
pub use cost_table::{CostTable, Key, PREFILL_ROWS, Shape, shape_of};
pub use error::MixedError;
pub use feasibility::{Check, Composites, Feasibility, PlannerFeasibility};
pub use option::QuantOption;
pub use priors::Priors;
pub use seeds::read_seeds;
pub use space::{Sources, Space, sorted_by_layer};
