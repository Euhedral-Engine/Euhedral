//! Hidden rows: a generation whose every row's post-final-norm hidden state is handed to a sink with the row's logits.
//!
//! The rows are the ones the draft layer reads to seed its catch-up: the engine's sequence already writes them for a
//! request that drafts, and a hidden request asks for the same write on every prompt chunk and decode row. After a
//! quantum retires its rows (and its logits row, when it has one) are copied to pinned memory on the owner's lane; once
//! the lane has reached the copy a frame of the lake calls the sink, and the next quantum, which would overwrite the
//! rows, is admitted after the copy. The tokens are selected by the host as for a sampled request and reported to the
//! request's listener as usual.

use std::collections::VecDeque;
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_inference_cuda::PooledPinned;

use super::mailbox::Address;
use super::source::{Message, Owner};
use crate::frames::{Kind, step};

/// The most quanta whose rows wait in pinned memory for the sink: past it the request admits no quantum until the sink
/// has taken some, so a slow sink holds back the request instead of piling up pinned memory.
const MAX_QUEUED: usize = 2;

/// Receives the hidden rows of a generation. Every call runs as a frame on a worker of the engine's lattice, one at a
/// time and in order, before the generation's end is reported, and must not wait.
pub trait HiddenSink: Send {
    /// The rows of one quantum: `rows` rows from `first_position` on, each `hidden_size` BF16 values in little-endian
    /// bytes (the output of the final norm), back to back in `hidden`, and, when the quantum selected a token, the BF16
    /// logits row of its last row (`vocabulary` values). The bytes are valid for the call only.
    fn rows(&mut self, first_position: u64, rows: usize, hidden: &[u8], logits: Option<&[u8]>);
}

/// The rows of one quantum in pinned memory: the hidden rows, then the logits row when there is one.
pub(super) struct HiddenRows {
    pub(super) first_position: u64,
    pub(super) rows: usize,
    pub(super) hidden_bytes: usize,
    pub(super) logits_bytes: usize,
    pub(super) loan: PooledPinned,
}

/// What a hidden request keeps: the sink, which is out while a frame delivers to it, the rows waiting for it, and
/// whether the rows of a quantum are being copied.
pub(super) struct Hidden {
    sink: Option<Box<dyn HiddenSink>>,
    queue: VecDeque<HiddenRows>,
    pub(super) copying: bool,
}

impl Hidden {
    pub(super) fn new(sink: Box<dyn HiddenSink>) -> Hidden {
        Hidden {
            sink: Some(sink),
            queue: VecDeque::new(),
            copying: false,
        }
    }

    /// Whether rows are still on their way to the sink: being copied, waiting, or being delivered.
    pub(super) fn busy(&self) -> bool {
        self.copying || !self.queue.is_empty() || self.sink.is_none()
    }

    /// Whether the sink is so far behind that no further rows are copied: the next quantum waits.
    pub(super) fn full(&self) -> bool {
        self.copying || self.queue.len() >= MAX_QUEUED
    }

    /// Takes in rows whose copy reached pinned memory.
    pub(super) fn arrived(&mut self, rows: HiddenRows) {
        self.copying = false;
        self.queue.push_back(rows);
        debug_assert!(
            self.queue.len() <= MAX_QUEUED,
            "{} rows wait for the sink, past the {MAX_QUEUED} the owner allows",
            self.queue.len()
        );
    }

    /// The sink came back from a delivery.
    pub(super) fn returned(&mut self, sink: Box<dyn HiddenSink>) {
        self.sink = Some(sink);
    }

    /// The sink and the rows of the next delivery, when the sink is in and rows wait.
    pub(super) fn next_delivery(&mut self) -> Option<(Box<dyn HiddenSink>, HiddenRows)> {
        if self.queue.is_empty() {
            return None;
        }
        let sink = self.sink.take()?;
        let rows = self.queue.pop_front()?;
        Some((sink, rows))
    }
}

/// The sink on its way through a delivery frame. It goes back to the owner when the frame has run or is dropped unrun,
/// so that the request never waits for a sink that is gone.
struct Delivery {
    sink: Option<Box<dyn HiddenSink>>,
    address: Arc<Address<Owner>>,
}

impl Drop for Delivery {
    fn drop(&mut self) {
        if let Some(sink) = self.sink.take() {
            self.address.post(Message::HiddenDelivered(sink));
        }
    }
}

/// The frame that calls `sink` with `rows`, then tells the owner `sink` is free.
pub(super) fn delivery_frame(
    chain: usize,
    address: Arc<Address<Owner>>,
    sink: Box<dyn HiddenSink>,
    rows: HiddenRows,
) -> Box<dyn Frame> {
    let mut delivery = Delivery {
        sink: Some(sink),
        address,
    };
    step(Kind::Hidden, chain, move || {
        if let Some(sink) = delivery.sink.as_mut() {
            let all = rows.loan.buffer().as_slice();
            let hidden = &all[..rows.hidden_bytes];
            let logits =
                (rows.logits_bytes > 0).then(|| &all[rows.hidden_bytes..rows.hidden_bytes + rows.logits_bytes]);
            sink.rows(rows.first_position, rows.rows, hidden, logits);
        }
    })
}
