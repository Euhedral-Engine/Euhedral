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

/// A place a quantum's end took out, for the test to complete.
pub struct ProbePlace(Option<Slot<Retirement>>);

impl fmt::Debug for ProbePlace {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ProbePlace").finish_non_exhaustive()
    }
}

impl ProbePlace {
    pub(crate) fn new(place: Option<Slot<Retirement>>) -> ProbePlace {
        ProbePlace(place)
    }

    /// Completes the place, as the frame that ends the quantum does. Whether there was one.
    pub fn complete(self) -> bool {
        self.0.map(|place| place.complete(Retirement::Copy)).is_some()
    }
}
