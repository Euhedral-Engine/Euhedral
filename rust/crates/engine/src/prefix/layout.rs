//! Where each piece of a sequence's state sits in a checkpoint's slots, and the copies that move it.
//!
//! A slot holds one 256-token KV page group: the page of every full-attention layer, in layer order. A node's slots
//! are, in this order, each section starting on a slot boundary:
//! - the GDN state: for every GDN layer its convolution then recurrent buffer, laid end to end over
//!   `ceil(bytes / slot)` slots (a buffer may straddle two slots);
//! - the KV pages of the span, one slot per page, from `start / 256` to `ceil(position / 256)`;
//! - for a node that holds speculative state, that state, laid end to end the same way as the GDN state.
//!
//! The MTP state is the base hidden row of the node's last position, then the MTP cache's pages. The MTP cache of a
//! checkpoint at `position` holds rows `[0, position - 1)`: row `position - 1` pairs the hidden row of that position
//! with the token at `position`, which belongs to whichever prompt resumes here, so a restore recomputes it from the
//! stored hidden row. Its pages run from the page of row `start - 1` (which a restore overwrites from the later node,
//! as that row is the parent's last) to `ceil((position - 1) / 256)`.

use std::slice;

use euhedral_inference_settings::KvFormat;
use euhedral_memory::SlotLease;

use crate::model::{LayerKind, ModelConfig};
use crate::prefix::tree::{POSITION_GRANULE, PrefixNode, Speculation};
use crate::residency::footprint::page_bytes;

const PAGE: u32 = POSITION_GRANULE;

/// A device buffer, or the part of a sequence's state a copy moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Buffer {
    /// A GDN layer's convolution state.
    Convolution { layer: usize },
    /// A GDN layer's recurrent state.
    Recurrent { layer: usize },
    /// Page `page` of a full-attention layer's cache.
    KvPage { layer: usize, page: usize },
    /// Page `page` of the MTP layer's cache.
    MtpPage { page: usize },
    /// The sequence's draft seed row: the base hidden row of the checkpoint's position.
    SeedRow,
}

/// One copy between a slot of the host arena (at `host`) and `bytes` of `buffer` from `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotCopy {
    pub host: u64,
    pub buffer: Buffer,
    pub offset: u64,
    pub bytes: u64,
}

/// The geometry of a model's checkpoints.
#[derive(Clone, Debug)]
pub struct CheckpointLayout {
    gdn_layers: Vec<usize>,
    kv_layers: Vec<usize>,
    mtp_pages: bool,
    convolution_bytes: u64,
    recurrent_bytes: u64,
    page_bytes: u64,
    hidden_row_bytes: u64,
    slot_bytes: u64,
}

fn pages(position: u32) -> usize {
    position.div_ceil(PAGE) as usize
}

impl CheckpointLayout {
    /// The layout of sequences whose attention caches are in `format`; `None` for a model without a full-attention
    /// layer, which has no page to size a slot by.
    pub fn new(config: &ModelConfig, format: KvFormat) -> Option<Self> {
        let layers_of = |wanted: LayerKind| -> Vec<usize> {
            config
                .layer_kinds
                .iter()
                .enumerate()
                .filter(|(_, kind)| **kind == wanted)
                .map(|(layer, _)| layer)
                .collect()
        };
        let kv_layers = layers_of(LayerKind::FullAttention);
        if kv_layers.is_empty() {
            return None;
        }
        let page = page_bytes(config, format);
        let convolution = config.gdn_convolution_channels() as u64 * (config.linear_conv_kernel_dim as u64 - 1) * 2;
        let recurrent = config.linear_num_value_heads as u64
            * config.linear_key_head_dim as u64
            * config.linear_value_head_dim as u64
            * 4;
        Some(CheckpointLayout {
            gdn_layers: layers_of(LayerKind::GatedDeltaNet),
            slot_bytes: kv_layers.len() as u64 * page,
            kv_layers,
            mtp_pages: config.mtp_layer_count > 0,
            convolution_bytes: convolution,
            recurrent_bytes: recurrent,
            page_bytes: page,
            hidden_row_bytes: config.hidden_size as u64 * 2,
        })
    }

    /// Bytes of one slot: one KV page of every full-attention layer.
    pub fn slot_bytes(&self) -> u64 {
        self.slot_bytes
    }

    /// Whether the model has an MTP layer whose state a checkpoint can hold.
    pub fn has_speculation(&self, kind: Speculation) -> bool {
        match kind {
            Speculation::Mtp => self.mtp_pages,
        }
    }

    fn gdn_bytes(&self) -> u64 {
        self.gdn_layers.len() as u64 * (self.convolution_bytes + self.recurrent_bytes)
    }

    /// Slots of a node's GDN state.
    pub fn gdn_slots(&self) -> usize {
        self.gdn_bytes().div_ceil(self.slot_bytes) as usize
    }

    /// Slots of the KV pages of the span `[start, position)`: one per page.
    pub fn page_slots(&self, start: u32, position: u32) -> usize {
        pages(position) - (start / PAGE) as usize
    }

    fn mtp_first_page(start: u32) -> usize {
        if start == 0 { 0 } else { (start / PAGE) as usize - 1 }
    }

    fn mtp_end_page(position: u32) -> usize {
        pages(position - 1)
    }

    fn speculative_bytes(&self, kind: Speculation, start: u32, position: u32) -> u64 {
        match kind {
            Speculation::Mtp => {
                let count = Self::mtp_end_page(position).saturating_sub(Self::mtp_first_page(start));
                self.hidden_row_bytes + count as u64 * self.page_bytes
            }
        }
    }

    /// Slots of the speculative state of the span `[start, position)`.
    pub fn speculative_slots(&self, kind: Speculation, start: u32, position: u32) -> usize {
        self.speculative_bytes(kind, start, position).div_ceil(self.slot_bytes) as usize
    }

    /// Slots of a node over `[start, position)` that holds `speculation`.
    pub fn slots(&self, start: u32, position: u32, speculation: Option<Speculation>) -> usize {
        self.gdn_slots()
            + self.page_slots(start, position)
            + speculation.map_or(0, |kind| self.speculative_slots(kind, start, position))
    }

    /// The copies that save `node`'s state from the sequence's buffers into its slots (or, read the other way, load
    /// it): everything the node holds.
    pub fn copies(&self, node: &PrefixNode) -> Vec<SlotCopy> {
        let mut copies = self.gdn_copies(node);
        self.page_copies(node, &mut copies);
        if let Some(kind) = node.speculation() {
            self.speculative_copies(kind, node, |_| true, &mut copies);
        }
        copies
    }

    /// The copies that load a matched chain into a sequence: the KV pages of every node and the GDN state of the last
    /// (the state at the matched position), then the speculative state `speculation` names: the pages of every node
    /// in chain order, so a later node overwrites the page it shares with its parent, and the seed row of the last.
    pub fn restore_copies(&self, chain: &[&PrefixNode], speculation: Option<Speculation>) -> Vec<SlotCopy> {
        let Some(leaf) = chain.last() else {
            return Vec::new();
        };
        debug_assert!(
            speculation.is_none() || chain.iter().all(|node| node.speculation() == speculation),
            "every node of the chain must hold the speculative state asked for"
        );
        let mut copies = self.gdn_copies(leaf);
        for node in chain {
            self.page_copies(node, &mut copies);
        }
        if let Some(kind) = speculation {
            for node in chain {
                self.speculative_copies(
                    kind,
                    node,
                    |buffer| matches!(buffer, Buffer::MtpPage { .. }),
                    &mut copies,
                );
            }
            self.speculative_copies(kind, leaf, |buffer| buffer == Buffer::SeedRow, &mut copies);
        }
        copies
    }

    fn section<'a>(&self, node: &'a PrefixNode, from: usize, count: usize) -> &'a [SlotLease] {
        let slots = node.slots();
        assert!(
            from + count <= slots.len(),
            "the node holds {} slots, its layout needs {}",
            slots.len(),
            from + count
        );
        &slots[from..from + count]
    }

    fn gdn_copies(&self, node: &PrefixNode) -> Vec<SlotCopy> {
        let mut pieces = Vec::new();
        for &layer in &self.gdn_layers {
            pieces.push((Buffer::Convolution { layer }, self.convolution_bytes));
            pieces.push((Buffer::Recurrent { layer }, self.recurrent_bytes));
        }
        let mut copies = Vec::new();
        lay(
            &pieces,
            self.section(node, 0, self.gdn_slots()),
            self.slot_bytes,
            |_| true,
            &mut copies,
        );
        copies
    }

    fn page_copies(&self, node: &PrefixNode, copies: &mut Vec<SlotCopy>) {
        let first = (node.start() / PAGE) as usize;
        let count = self.page_slots(node.start(), node.position());
        let slots = self.section(node, self.gdn_slots(), count);
        for (page, slot) in (first..first + count).zip(slots) {
            let pieces: Vec<_> = self
                .kv_layers
                .iter()
                .map(|&layer| (Buffer::KvPage { layer, page }, self.page_bytes))
                .collect();
            lay(&pieces, slice::from_ref(slot), self.slot_bytes, |_| true, copies);
        }
    }

    fn speculative_copies(
        &self,
        kind: Speculation,
        node: &PrefixNode,
        keep: impl Fn(Buffer) -> bool,
        copies: &mut Vec<SlotCopy>,
    ) {
        let from = self.gdn_slots() + self.page_slots(node.start(), node.position());
        let count = self.speculative_slots(kind, node.start(), node.position());
        let slots = self.section(node, from, count);
        let Speculation::Mtp = kind;
        let mut pieces = vec![(Buffer::SeedRow, self.hidden_row_bytes)];
        for page in Self::mtp_first_page(node.start())..Self::mtp_end_page(node.position()) {
            pieces.push((Buffer::MtpPage { page }, self.page_bytes));
        }
        lay(&pieces, slots, self.slot_bytes, keep, copies);
    }
}

/// Lays `pieces` end to end over `slots`, splitting a piece where it crosses a slot boundary, and appends the copies
/// whose buffer `keep` accepts.
fn lay(
    pieces: &[(Buffer, u64)],
    slots: &[SlotLease],
    slot_bytes: u64,
    keep: impl Fn(Buffer) -> bool,
    copies: &mut Vec<SlotCopy>,
) {
    let mut at = 0;
    for &(buffer, bytes) in pieces {
        let mut done = 0;
        while done < bytes {
            let slot = (at / slot_bytes) as usize;
            let inside = at % slot_bytes;
            let take = (bytes - done).min(slot_bytes - inside);
            if keep(buffer) {
                copies.push(SlotCopy {
                    host: slots[slot].address() + inside,
                    buffer,
                    offset: done,
                    bytes: take,
                });
            }
            done += take;
            at += take;
        }
    }
}
