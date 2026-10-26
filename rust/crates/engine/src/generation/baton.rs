//! A baton: one value that one holder at a time uses, passed by taking it and putting it back.

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use std::fmt::{self, Formatter};

/// A slot for one value. [`Baton::take`] gives it to the caller, who is then its only user, until they
/// [`Baton::put`] it back (or let it go). It takes no lock and waits for nothing: a taker that finds the slot empty,
/// because another holds the value, goes on without it and tries again at its next poll.
pub(crate) struct Baton<T> {
    producer: Producer<T, Multi>,
    consumer: Consumer<T, Multi>,
}

impl<T> fmt::Debug for Baton<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Baton").finish_non_exhaustive()
    }
}

impl<T> Baton<T> {
    /// An empty baton.
    pub(crate) fn new() -> Baton<T> {
        let queue = MpmcQueue::bounded(1);
        Baton {
            producer: queue.producer(),
            consumer: queue.consumer(),
        }
    }

    /// The value, if no one holds it.
    pub(crate) fn take(&self) -> Option<T> {
        self.consumer.poll()
    }

    /// Hands the value back. There is one value: a second put while the first is in the slot drops the second and
    /// logs it.
    pub(crate) fn put(&self, value: T) {
        if self.producer.offer(value).is_err() {
            log::error!("a baton was handed back twice");
        }
    }
}
