//! Work a failed submission queued behind no marker of its quantum.
//!
//! RustEE retires a quantum after the latest marker each of its lanes recorded, and records a stage's marker only when
//! the stage succeeded. A submission that fails after it queued kernels (a fused unit whose second stage refused, or a
//! launch that panicked) leaves those kernels on its lane, and when that lane is not the quantum's home lane nothing
//! orders the retirement behind them: the quantum could retire, and give its pages back, while a kernel of its own
//! still runs.
//!
//! A marked submission therefore always publishes its marker ([`submit_on`](super::submit_on) hands the marked
//! submission an `Ok` that carries the launch's own result), and when the launch failed the marker is kept here, the
//! latest per lane. [`Stage::abandoned`](euhedral_graph::Stage::abandoned) orders the home lane behind the kept markers
//! ([`Unmarked::await_on`]), and RustEE arms the quantum's retirement on the home lane after that hook.

use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_graph::{DeviceQueue, LaneId, Marker, QueueId};

/// The latest marker of a failed submission of a quantum, for each lane. Marker values start at 1, so 0 is none.
#[derive(Debug)]
pub struct Unmarked {
    lanes: Box<[AtomicU64]>,
}

impl Unmarked {
    /// The record of a quantum over `lanes` lanes.
    pub fn new(lanes: usize) -> Self {
        Self {
            lanes: (0..lanes).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// A submission on `marker`'s lane failed after it may have queued work: the marker covers that work.
    pub fn record(&self, marker: Marker) {
        if let Some(slot) = self.lanes.get(usize::from(marker.lane.0)) {
            slot.fetch_max(marker.value, Ordering::AcqRel);
        }
    }

    /// The kept marker of `lane`, if any.
    pub fn latest(&self, lane: LaneId) -> Option<u64> {
        self.lanes
            .get(usize::from(lane.0))
            .map(|slot| slot.load(Ordering::Acquire))
            .filter(|&value| value != 0)
    }

    /// Orders `home` behind the kept markers of the other lanes, on the device: nothing waits. `queue_of` names the
    /// queue of a lane. A marker whose await is refused (only a poisoned device or an invalid marker does that) is
    /// logged and stays kept, and retirement proceeds without it; a home lane that cannot order anything (a query-only
    /// lane) leaves them all.
    pub fn await_on(&self, home: &dyn DeviceQueue, queue_of: impl Fn(LaneId) -> Option<QueueId>) {
        let Some(control) = home.control() else {
            return;
        };
        for (index, slot) in self.lanes.iter().enumerate() {
            let value = slot.load(Ordering::Acquire);
            let (Some(lane), true) = (u16::try_from(index).ok().map(LaneId), value != 0) else {
                continue;
            };
            let Some(queue) = queue_of(lane) else {
                log::error!("lane {} of a failed quantum is not in its lane table", lane.0);
                continue;
            };
            if queue == home.id() {
                // The home lane is ordered by itself: its own work is behind the retirement already.
                let _ = slot.compare_exchange(value, 0, Ordering::AcqRel, Ordering::Acquire);
                continue;
            }
            match control.await_marker(Marker::new(queue, lane, value)) {
                Ok(()) => {
                    let _ = slot.compare_exchange(value, 0, Ordering::AcqRel, Ordering::Acquire);
                }
                Err(error) => log::error!(
                    "the home lane could not be ordered behind the work a failed submission left on lane {}: {error}",
                    lane.0
                ),
            }
        }
    }
}
