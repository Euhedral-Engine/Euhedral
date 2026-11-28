//! The order a sequence's work concludes in: admission order, whatever order it ends in.
//!
//! Work is [admitted](Conclusion::admit) when the owner publishes it, which offers a place in the order to a RustEE
//! [`Sequencer`] and hands the place to the work. The frame that ends the work completes the place with what it
//! ended with; work that is dropped without having ended passes its place, so the work behind it still concludes.
//! Whichever thread completes a place that heads the order concludes the ready prefix, in admission order, by calling
//! the function the order was built with ([`Conclusion::new`]) on that thread. That function must not wait.
//!
//! A value concludes at most once, when its position is reached.

use euhedral_graph::{Sequencer, Slot, Ticket};
use std::fmt;
use std::sync::Arc;

/// The places of a sequence's work, concluded in the order they were admitted.
pub(super) struct Conclusion<T: Send + 'static> {
    order: Arc<Sequencer<Arc<Ticket<T>>>>,
}

impl<T: Send + 'static> fmt::Debug for Conclusion<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Conclusion").finish_non_exhaustive()
    }
}

impl<T: Send + 'static> Conclusion<T> {
    /// An empty order that concludes each completed value through `concluded`, in admission order, on the thread that
    /// completes the place that makes it ready.
    pub(super) fn new(concluded: impl Fn(T) + Send + Sync + 'static) -> Conclusion<T> {
        Conclusion {
            order: Sequencer::ordered(concluded),
        }
    }

    /// The next position in the order. Complete the slot with the value when the work ends; dropping it passes the
    /// position.
    pub(super) fn admit(&self) -> Slot<T> {
        self.order.register()
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the recording fake is shared between the test and the function the order calls"
)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    /// A conclusion whose values land in a list, in the order they concluded.
    fn recording() -> (Conclusion<u32>, Arc<Mutex<Vec<u32>>>) {
        let concluded = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&concluded);
        let conclusion =
            Conclusion::new(move |value| sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(value));
        (conclusion, concluded)
    }

    fn concluded(list: &Arc<Mutex<Vec<u32>>>) -> Vec<u32> {
        list.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    #[test]
    fn values_conclude_in_admission_order_whatever_order_they_are_completed_in() {
        let (conclusion, list) = recording();
        let (first, second, third) = (conclusion.admit(), conclusion.admit(), conclusion.admit());
        third.complete(3);
        second.complete(2);
        assert_eq!(concluded(&list), Vec::<u32>::new(), "the head has not completed");
        first.complete(1);
        assert_eq!(concluded(&list), [1, 2, 3]);
    }

    #[test]
    fn a_dropped_position_is_passed_and_the_ones_behind_it_conclude() {
        let (conclusion, list) = recording();
        let (first, second, third) = (conclusion.admit(), conclusion.admit(), conclusion.admit());
        third.complete(3);
        first.complete(1);
        assert_eq!(concluded(&list), [1], "the second position is still pending");
        drop(second);
        assert_eq!(concluded(&list), [1, 3]);
    }

    #[test]
    fn a_head_that_is_dropped_before_the_ones_behind_it_complete_concludes_them_as_they_complete() {
        let (conclusion, list) = recording();
        let (first, second) = (conclusion.admit(), conclusion.admit());
        drop(first);
        assert_eq!(concluded(&list), Vec::<u32>::new());
        second.complete(2);
        assert_eq!(concluded(&list), [2]);
    }

    #[test]
    fn a_value_concludes_once() {
        let (conclusion, list) = recording();
        let (first, second) = (conclusion.admit(), conclusion.admit());
        first.complete(1);
        second.complete(2);
        drop(conclusion);
        assert_eq!(concluded(&list), [1, 2]);
    }

    #[test]
    fn positions_admitted_after_a_conclusion_conclude_after_it() {
        let (conclusion, list) = recording();
        conclusion.admit().complete(1);
        let later = conclusion.admit();
        assert_eq!(concluded(&list), [1]);
        later.complete(2);
        assert_eq!(concluded(&list), [1, 2]);
    }

    #[test]
    fn a_completion_during_a_conclusion_is_left_to_the_running_drain() {
        // The conclusion of the first value completes the third place, which is behind the second: that completion
        // finds the drain busy and leaves its value to the drain that is running.
        let list = Arc::new(Mutex::new(Vec::new()));
        let third = Arc::new(Mutex::new(None::<Slot<u32>>));
        let (sink, held) = (Arc::clone(&list), Arc::clone(&third));
        let conclusion = Conclusion::new(move |value: u32| {
            sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(value);
            if value == 1
                && let Some(slot) = held.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take()
            {
                slot.complete(3);
            }
        });
        let (first, second) = (conclusion.admit(), conclusion.admit());
        *third.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(conclusion.admit());
        second.complete(2);
        first.complete(1);
        assert_eq!(concluded(&list), [1, 2, 3]);
    }

    #[test]
    fn a_place_dropped_during_a_conclusion_is_passed_and_the_ones_behind_it_conclude() {
        // The conclusion of the first value drops the second place (a start that fails inside the owner's turn): the
        // third, completed before, concludes behind the first.
        let list = Arc::new(Mutex::new(Vec::new()));
        let second = Arc::new(Mutex::new(None::<Slot<u32>>));
        let (sink, held) = (Arc::clone(&list), Arc::clone(&second));
        let conclusion = Conclusion::new(move |value: u32| {
            sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(value);
            drop(held.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take());
        });
        let first = conclusion.admit();
        *second.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(conclusion.admit());
        let third = conclusion.admit();
        third.complete(3);
        assert!(concluded(&list).is_empty());
        first.complete(1);
        assert_eq!(concluded(&list), [1, 3]);
    }

    #[test]
    fn a_place_admitted_during_a_conclusion_concludes_after_it_when_completed() {
        // The conclusion of the first value admits the next place, as the owner's turn does for the next quantum.
        let list = Arc::new(Mutex::new(Vec::new()));
        let order: Arc<OnceLock<Conclusion<u32>>> = Arc::new(OnceLock::new());
        let next = Arc::new(Mutex::new(None::<Slot<u32>>));
        let (sink, own, held) = (Arc::clone(&list), Arc::clone(&order), Arc::clone(&next));
        let conclusion = Conclusion::new(move |value: u32| {
            sink.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).push(value);
            if value == 1
                && let Some(conclusion) = own.get()
            {
                *held.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(conclusion.admit());
            }
        });
        let first = conclusion.admit();
        let shared = Conclusion {
            order: Arc::clone(&conclusion.order),
        };
        assert!(order.set(shared).is_ok());
        first.complete(1);
        let admitted = next.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).take();
        assert_eq!(concluded(&list), [1]);
        admitted.expect("the conclusion admitted a place").complete(2);
        assert_eq!(concluded(&list), [1, 2]);
    }

    #[test]
    fn a_conclusion_dropped_with_places_outstanding_still_concludes_and_passes_them() {
        let (conclusion, list) = recording();
        let (first, second, third) = (conclusion.admit(), conclusion.admit(), conclusion.admit());
        third.complete(3);
        drop(conclusion);
        drop(second);
        first.complete(1);
        assert_eq!(concluded(&list), [1, 3]);
    }
}
