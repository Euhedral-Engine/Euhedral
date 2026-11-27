//! The order a request's quanta conclude in: admission order, whatever order they retire in.
//!
//! A quantum is [admitted](Conclusion::admit) when the owner publishes it, which offers a place in the order to a
//! RustEE [`Sequencer`]. The owner completes the quantum's [`Slot`] when it retires, or lets the slot drop when the
//! quantum never ran (its start failed): a dropped slot is passed, so the quanta behind it still conclude. The
//! sequencer concludes the ready prefix in admission order, and what it concludes lands in a queue that only the owner
//! reads ([`next`](Conclusion::next)), so the conclusion itself (commits, the listener, the failure of a request) runs
//! in the owner's turn on state the owner alone touches, wherever the draining thread is.
//!
//! A value concludes at most once: the sequencer hands it over when its position is reached and never again.

use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};
use euhedral_graph::{Sequencer, Slot, Ticket};
use std::fmt;
use std::sync::Arc;

/// The chunk size of the queue of concluded values, and the chunks it keeps for reuse.
const CONCLUDED_CHUNK: usize = 16;
const CONCLUDED_POOLED: usize = 2;

/// The positions of a request's quanta and the values concluded, in order, for its owner.
pub(super) struct Conclusion<T: Send + 'static> {
    order: Arc<Sequencer<Arc<Ticket<T>>>>,
    /// The concluded values, which the owner takes in conclusion order.
    concluded: Consumer<T, Single>,
}

impl<T: Send + 'static> fmt::Debug for Conclusion<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Conclusion").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Conclusion<T> {
    /// An empty order.
    pub(super) fn new() -> Conclusion<T> {
        let queue = MpscQueue::<T>::with_pool(CONCLUDED_CHUNK, CONCLUDED_POOLED);
        let producer = queue.producer();
        let order = Sequencer::ordered(move |value| {
            if producer.offer(value).is_err() {
                // The queue is unbounded: this does not happen.
                log::error!("the queue of concluded quanta refused a value");
            }
        });
        Conclusion {
            order,
            concluded: queue.consumer(),
        }
    }

    /// The next position in the order. Complete the slot with the value when it is ready; dropping it passes the position.
    pub(super) fn admit(&self) -> Slot<T> {
        self.order.register()
    }

    /// The next concluded value: the earliest admitted one whose turn has come and which the owner has not taken.
    pub(super) fn next(&mut self) -> Option<T> {
        self.concluded.poll()
    }

    /// The positions admitted whose value the owner has not taken.
    pub(super) fn pending(&self) -> usize {
        self.order.len() + usize::try_from(self.concluded.size_long()).unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::from_fn;

    fn drain(conclusion: &mut Conclusion<u32>) -> Vec<u32> {
        from_fn(|| conclusion.next()).collect()
    }

    #[test]
    fn values_conclude_in_admission_order_whatever_order_they_are_completed_in() {
        let mut conclusion = Conclusion::new();
        let (first, second, third) = (conclusion.admit(), conclusion.admit(), conclusion.admit());
        third.complete(3);
        second.complete(2);
        assert_eq!(drain(&mut conclusion), Vec::<u32>::new(), "the head has not completed");
        assert_eq!(conclusion.pending(), 3);
        first.complete(1);
        assert_eq!(drain(&mut conclusion), [1, 2, 3]);
        assert_eq!(conclusion.pending(), 0);
    }

    #[test]
    fn a_dropped_position_is_passed_and_the_ones_behind_it_conclude() {
        let mut conclusion = Conclusion::new();
        let (first, second, third) = (conclusion.admit(), conclusion.admit(), conclusion.admit());
        third.complete(3);
        first.complete(1);
        assert_eq!(drain(&mut conclusion), [1], "the second position is still pending");
        drop(second);
        assert_eq!(drain(&mut conclusion), [3]);
        assert_eq!(conclusion.pending(), 0);
    }

    #[test]
    fn a_value_is_taken_once() {
        let mut conclusion = Conclusion::new();
        conclusion.admit().complete(7);
        assert_eq!(conclusion.pending(), 1, "concluded and not yet taken");
        assert_eq!(conclusion.next(), Some(7));
        assert_eq!(conclusion.next(), None);
        assert_eq!(conclusion.pending(), 0);
    }

    #[test]
    fn positions_admitted_after_a_conclusion_conclude_after_it() {
        let mut conclusion = Conclusion::new();
        conclusion.admit().complete(1);
        let later = conclusion.admit();
        assert_eq!(drain(&mut conclusion), [1]);
        later.complete(2);
        assert_eq!(drain(&mut conclusion), [2]);
    }
}
