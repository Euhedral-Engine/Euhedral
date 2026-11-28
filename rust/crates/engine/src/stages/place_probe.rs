//! A sequence's order of conclusion over bare places, counting what concludes, for a test that holds a batch's places
//! to the sequences they belong to.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_graph::{Sequencer, Slot, Ticket};

use super::Retirement;

pub struct PlaceProbe {
    order: Arc<Sequencer<Arc<Ticket<Retirement>>>>,
    concluded: Arc<AtomicUsize>,
}

impl fmt::Debug for PlaceProbe {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("PlaceProbe").finish_non_exhaustive()
    }
}

impl PlaceProbe {
    pub fn new() -> PlaceProbe {
        let concluded = Arc::new(AtomicUsize::new(0));
        let sink = Arc::clone(&concluded);
        let order = Sequencer::ordered(move |_: Retirement| {
            sink.fetch_add(1, Ordering::AcqRel);
        });
        PlaceProbe { order, concluded }
    }

    /// How many places of this order concluded.
    pub fn concluded(&self) -> usize {
        self.concluded.load(Ordering::Acquire)
    }

    pub(crate) fn place(&self) -> Slot<Retirement> {
        self.order.register()
    }
}

impl Default for PlaceProbe {
    fn default() -> PlaceProbe {
        PlaceProbe::new()
    }
}
