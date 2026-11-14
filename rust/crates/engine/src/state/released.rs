//! Things a cache has given up, waiting for the owner of the generation to free them.

use std::fmt;

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_inference_cuda::DeviceBuffer;

/// Slots in one chunk of the queue. The queue is unbounded and grows by chunks, so this is the size of an allocation
/// step and never a limit: a buffer handed over is always taken.
const CHUNK: usize = 64;

/// The buffers a cache has given up, handed from whichever thread dropped the last holder to the owner of the
/// generation, which frees them in stream order on its own lane.
pub type ReleasedBuffers = Released<DeviceBuffer>;

/// A queue of items given up by any thread and taken by the one that frees them. Handing an item over never fails and
/// never waits; taking is by [`Released::drain`].
pub struct Released<T> {
    producer: Producer<T, Multi>,
    consumer: Consumer<T, Multi>,
}

impl<T> fmt::Debug for Released<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Released")
            .field("waiting", &self.consumer.size_long())
            .finish()
    }
}

impl<T> Default for Released<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Released<T> {
    pub fn new() -> Released<T> {
        let queue = MpmcQueue::<T>::new(CHUNK);
        Released {
            producer: queue.producer(),
            consumer: queue.consumer(),
        }
    }

    /// Hands `item` over. Any thread may call it. The queue is unbounded, so the offer is always taken; a refusal
    /// would be a defect in the queue, and is caught in debug builds (the item is dropped in release builds).
    pub fn give(&self, item: T) {
        let offered = self.producer.offer(item);
        debug_assert!(offered.is_ok(), "an unbounded queue refused an item");
    }

    /// Items handed over and not yet taken.
    pub fn waiting(&self) -> u64 {
        self.consumer.size_long()
    }

    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.consumer.is_empty()
    }

    /// Passes every waiting item to `release`, oldest first, and returns how many. If another thread is taking at the
    /// same moment it stops early: what is left is taken by the next call.
    pub fn drain(&self, release: impl FnMut(T)) -> u64 {
        self.consumer.drain_all(release)
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "the test starts threads to hand items over from several producers at once, and joins them"
)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    #[test]
    fn items_from_several_producers_are_each_taken_once() {
        const PRODUCERS: usize = 4;
        const EACH: usize = 500;
        let released = Arc::new(Released::<usize>::new());
        let workers: Vec<_> = (0..PRODUCERS)
            .map(|producer| {
                let released = Arc::clone(&released);
                thread::spawn(move || {
                    for index in 0..EACH {
                        released.give(producer * EACH + index);
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(released.waiting(), (PRODUCERS * EACH) as u64);
        let mut seen = vec![0u8; PRODUCERS * EACH];
        let taken = released.drain(|item| seen[item] += 1);
        assert_eq!(taken, (PRODUCERS * EACH) as u64);
        assert!(seen.iter().all(|&count| count == 1));
        assert!(released.is_empty());
        assert_eq!(released.drain(|_| panic!("nothing is left")), 0);
    }

    #[test]
    fn items_given_between_two_drains_are_taken_by_the_second() {
        let released = Released::<u32>::new();
        released.give(1);
        released.give(2);
        let mut first = Vec::new();
        released.drain(|item| first.push(item));
        released.give(3);
        let mut second = Vec::new();
        released.drain(|item| second.push(item));
        assert_eq!((first, second), (vec![1, 2], vec![3]));
    }

    #[test]
    fn dropping_the_queue_drops_what_waits() {
        let witness = Arc::new(());
        let released = Released::<Arc<()>>::new();
        released.give(Arc::clone(&witness));
        drop(released);
        assert_eq!(Arc::strong_count(&witness), 1);
    }
}
