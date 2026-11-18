use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use euhedral_inference_settings::KvFormat;

use super::fixtures::{arena, tokens};
use super::{Buffer, CheckpointLayout, Copy, NodeId, PrefixTree, Speculation};
use crate::model::{LayerKind, ModelConfig};
use crate::residency::footprint::page_bytes;

const FORMAT: KvFormat = KvFormat::Nvfp4;

fn config(kinds: Vec<LayerKind>, key_heads: usize, value_heads: usize, key_dim: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: 1,
        hidden_size: 64,
        num_hidden_layers: kinds.len(),
        num_attention_heads: 1,
        num_key_value_heads: 1,
        attention_head_dim: 256,
        intermediate_size: 1,
        linear_num_key_heads: key_heads,
        linear_num_value_heads: value_heads,
        linear_key_head_dim: key_dim,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1.0,
        rope_theta: 1.0,
        partial_rotary_factor: 1.0,
        max_position_embeddings: 1,
        layer_kinds: kinds,
        num_experts: 0,
        tie_word_embeddings: false,
        attention_output_gate: true,
        mtp_layer_count: 1,
        draft_depth: None,
    }
}

/// Eight layers, GDN on the even ones and full attention on the odd ones; small GDN state.
fn compact() -> ModelConfig {
    let kinds = (0..8)
        .map(|layer| {
            if layer % 2 == 1 {
                LayerKind::FullAttention
            } else {
                LayerKind::GatedDeltaNet
            }
        })
        .collect();
    config(kinds, 2, 4, 8)
}

/// One attention layer and a GDN state of several slots, so its buffers straddle slot boundaries.
fn wide_gdn() -> ModelConfig {
    config(
        vec![
            LayerKind::GatedDeltaNet,
            LayerKind::GatedDeltaNet,
            LayerKind::GatedDeltaNet,
            LayerKind::FullAttention,
        ],
        4,
        8,
        64,
    )
}

/// The device's buffers of one sequence, as host bytes, and the host arena the copies move them through.
struct Device {
    buffers: HashMap<Buffer, Vec<u8>>,
}

impl Device {
    fn new(config: &ModelConfig, kv_pages: usize, mtp_pages: usize) -> Self {
        let convolution = config.gdn_convolution_channels() * (config.linear_conv_kernel_dim - 1) * 2;
        let recurrent = config.linear_num_value_heads * config.linear_key_head_dim * config.linear_value_head_dim * 4;
        let page = page_bytes(config, FORMAT) as usize;
        let mut buffers = HashMap::new();
        for (layer, kind) in config.layer_kinds.iter().enumerate() {
            match kind {
                LayerKind::GatedDeltaNet => {
                    buffers.insert(Buffer::Convolution { layer }, vec![0; convolution]);
                    buffers.insert(Buffer::Recurrent { layer }, vec![0; recurrent]);
                }
                LayerKind::FullAttention => {
                    for page_index in 0..kv_pages {
                        buffers.insert(
                            Buffer::KvPage {
                                layer,
                                page: page_index,
                            },
                            vec![0; page],
                        );
                    }
                }
            }
        }
        for page_index in 0..mtp_pages {
            buffers.insert(Buffer::MtpPage { page: page_index }, vec![0; page]);
        }
        buffers.insert(Buffer::SeedRow, vec![0; config.hidden_size * 2]);
        Device { buffers }
    }

    /// Fills the buffers `which` accepts with bytes that differ per buffer and per `seed`.
    fn fill(&mut self, seed: u8, which: impl Fn(Buffer) -> bool) {
        for (buffer, bytes) in &mut self.buffers {
            if !which(*buffer) {
                continue;
            }
            let mut hasher = DefaultHasher::new();
            buffer.hash(&mut hasher);
            let key = hasher.finish();
            for (i, byte) in bytes.iter_mut().enumerate() {
                *byte = (key.rotate_left(i as u32 % 64) as u8)
                    .wrapping_add((i / 7) as u8)
                    .wrapping_mul(seed | 1)
                    .wrapping_add(seed);
            }
        }
    }

    fn capture(&self, host: &mut [u8], copies: &[Copy]) {
        for copy in copies {
            let from = &self.buffers[&copy.buffer][copy.offset as usize..][..copy.bytes as usize];
            host[copy.host as usize..][..copy.bytes as usize].copy_from_slice(from);
        }
    }

    fn restore(&mut self, host: &[u8], copies: &[Copy]) {
        for copy in copies {
            let into = self
                .buffers
                .get_mut(&copy.buffer)
                .unwrap_or_else(|| panic!("no buffer {:?}", copy.buffer));
            into[copy.offset as usize..][..copy.bytes as usize]
                .copy_from_slice(&host[copy.host as usize..][..copy.bytes as usize]);
        }
    }

    fn same(&self, other: &Device, buffer: Buffer) -> bool {
        self.buffers[&buffer] == other.buffers[&buffer]
    }
}

struct Rig {
    config: ModelConfig,
    layout: CheckpointLayout,
    tree: PrefixTree,
    host: Vec<u8>,
}

impl Rig {
    fn new(config: ModelConfig, slots: u32) -> Self {
        let layout = CheckpointLayout::new(&config, FORMAT).unwrap();
        let tree = PrefixTree::new(arena(slots, layout.slot_bytes()));
        let host = vec![0; slots as usize * layout.slot_bytes() as usize];
        Rig {
            config,
            layout,
            tree,
            host,
        }
    }

    fn device(&self, kv_pages: usize, mtp_pages: usize) -> Device {
        Device::new(&self.config, kv_pages, mtp_pages)
    }

    /// Reserves, captures and publishes the node `[parent.position, position)` of `tokens` from `device`.
    fn capture(
        &mut self,
        device: &Device,
        parent: NodeId,
        tokens: &[u32],
        position: u32,
        speculation: Option<Speculation>,
    ) -> NodeId {
        let start = self.tree.node(parent).position();
        let slots = self.layout.slots(start, position, speculation);
        let id = self.tree.reserve(parent, tokens, position, speculation, slots).unwrap();
        let copies = self.layout.copies(self.tree.node(id));
        device.capture(&mut self.host, &copies);
        self.tree.publish(id);
        id
    }

    fn restore(&self, device: &mut Device, chain: &[NodeId], speculation: Option<Speculation>) {
        let nodes: Vec<_> = chain.iter().map(|&id| self.tree.node(id)).collect();
        let copies = self.layout.restore_copies(&nodes, speculation);
        device.restore(&self.host, &copies);
    }
}

fn is_gdn(buffer: Buffer) -> bool {
    matches!(buffer, Buffer::Convolution { .. } | Buffer::Recurrent { .. })
}

fn is_kv(buffer: Buffer) -> bool {
    matches!(buffer, Buffer::KvPage { .. })
}

fn is_mtp(buffer: Buffer) -> bool {
    matches!(buffer, Buffer::MtpPage { .. } | Buffer::SeedRow)
}

fn page_of(buffer: Buffer) -> usize {
    match buffer {
        Buffer::KvPage { page, .. } | Buffer::MtpPage { page } => page,
        other => panic!("{other:?} has no page"),
    }
}

#[test]
fn sizes_a_matching_geometry_to_the_state_classes() {
    let config = compact();
    let layout = CheckpointLayout::new(&config, FORMAT).unwrap();
    assert_eq!(layout.slot_bytes(), 4 * page_bytes(&config, FORMAT));
    // Four GDN layers of 576 + 2048 bytes fit in one slot.
    assert_eq!(layout.gdn_slots(), 1);
    assert_eq!(layout.page_slots(0, 768), 3);
    assert_eq!(
        layout.page_slots(512, 1024),
        2,
        "pages 2 and 3 of a span that starts at 512"
    );
    assert_eq!(layout.slots(0, 768, None), 4);
    assert_eq!(layout.slots(512, 1024, None), 3);
}

#[test]
fn a_model_without_attention_has_no_layout() {
    let config = config(vec![LayerKind::GatedDeltaNet; 2], 2, 4, 8);
    assert!(CheckpointLayout::new(&config, FORMAT).is_none());
}

#[test]
fn capturing_then_restoring_reproduces_the_state_byte_for_byte() {
    let mut rig = Rig::new(compact(), 8);
    let stored = tokens(0, 768);
    let mut device = rig.device(3, 0);
    device.fill(3, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 768, None);

    let mut restored = rig.device(3, 0);
    rig.restore(&mut restored, &[node], None);
    for &buffer in device.buffers.keys().filter(|&&buffer| !is_mtp(buffer)) {
        assert!(device.same(&restored, buffer), "{buffer:?}");
    }
}

#[test]
fn restoring_touches_nothing_beyond_the_chain() {
    let mut rig = Rig::new(compact(), 8);
    let stored = tokens(0, 768);
    let mut device = rig.device(5, 0);
    device.fill(3, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 768, None);

    let mut restored = rig.device(5, 0);
    rig.restore(&mut restored, &[node], None);
    for (&buffer, bytes) in &restored.buffers {
        if matches!(buffer, Buffer::KvPage { page, .. } if page >= 3) {
            assert!(bytes.iter().all(|&b| b == 0), "{buffer:?} was written");
        }
    }
}

#[test]
fn a_chain_restores_each_nodes_pages_and_the_leafs_gdn_state() {
    let mut rig = Rig::new(compact(), 16);
    let stored = tokens(0, 1280);
    let mut device = rig.device(5, 0);
    device.fill(5, |_| true);
    let root = rig.tree.root();
    let first = rig.capture(&device, root, &stored, 512, None);
    // Pages 0 and 1 (rows 0-511) are the first node's, as filled now.
    let mut snapshot = rig.device(5, 0);
    snapshot.buffers.clone_from(&device.buffers);
    // The sequence moves on: every buffer is rewritten, then the second node (pages 2-4) is captured.
    device.fill(9, |_| true);
    let second = rig.capture(&device, first, &stored, 1280, None);

    let mut restored = rig.device(5, 0);
    rig.restore(&mut restored, &[first, second], None);
    for &buffer in device.buffers.keys() {
        if is_kv(buffer) {
            let source = if page_of(buffer) < 2 { &snapshot } else { &device };
            assert!(source.same(&restored, buffer), "{buffer:?}");
        } else if is_gdn(buffer) {
            assert!(
                device.same(&restored, buffer),
                "the leaf's GDN state, not the first node's: {buffer:?}"
            );
            assert!(!snapshot.same(&restored, buffer));
        }
    }
}

#[test]
fn gdn_state_that_spans_several_slots_round_trips_across_slot_boundaries() {
    let config = wide_gdn();
    let mut rig = Rig::new(config, 16);
    assert!(rig.layout.gdn_slots() > 1, "{} gdn slots", rig.layout.gdn_slots());
    let stored = tokens(0, 512);
    let mut device = rig.device(2, 0);
    device.fill(11, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 512, None);

    let copies = rig.layout.copies(rig.tree.node(node));
    let pieces = |buffer: Buffer| copies.iter().filter(|copy| copy.buffer == buffer).count();
    assert!(
        (0..3).any(|layer| pieces(Buffer::Recurrent { layer }) > 1 || pieces(Buffer::Convolution { layer }) > 1),
        "a buffer straddles a slot boundary and is split in two"
    );
    let slot_bytes = rig.layout.slot_bytes();
    for copy in &copies {
        assert_eq!(
            copy.host / slot_bytes,
            (copy.host + copy.bytes - 1) / slot_bytes,
            "{copy:?} crosses a slot boundary"
        );
    }
    let mut restored = rig.device(2, 0);
    rig.restore(&mut restored, &[node], None);
    for &buffer in device.buffers.keys().filter(|&&buffer| !is_mtp(buffer)) {
        assert!(device.same(&restored, buffer), "{buffer:?}");
    }
}

#[test]
fn a_nodes_copies_cover_each_byte_of_its_slots_once_without_overlap() {
    let mut rig = Rig::new(wide_gdn(), 16);
    let stored = tokens(0, 1024);
    let mut device = rig.device(4, 4);
    device.fill(2, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 1024, Some(Speculation::Mtp));
    let mut ranges: Vec<_> = rig
        .layout
        .copies(rig.tree.node(node))
        .iter()
        .map(|copy| (copy.host, copy.host + copy.bytes))
        .collect();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        assert!(pair[0].1 <= pair[1].0, "{pair:?} overlap");
    }
    let slots = rig.tree.node(node).slots();
    let slot_bytes = rig.layout.slot_bytes();
    let covered: Vec<u32> = ranges.iter().map(|range| (range.0 / slot_bytes) as u32).collect();
    for slot in slots {
        assert!(covered.contains(&slot.index()), "slot {} holds nothing", slot.index());
    }
    for range in &ranges {
        assert!(slots.iter().any(|slot| slot.index() as u64 == range.0 / slot_bytes));
    }
}

// --- MTP state: the MTP cache's rows [0, position - 1) and the base hidden row of position - 1.

#[test]
fn mtp_state_adds_the_seed_row_and_only_the_mtp_pages_below_position_minus_one() {
    let config = compact();
    let layout = CheckpointLayout::new(&config, FORMAT).unwrap();
    let mtp = Speculation::Mtp;
    // Rows [0, 767) fill 3 pages and the seed row: both in one slot of four pages.
    assert_eq!(layout.speculative_slots(mtp, 0, 768), 1);
    // A span from 512 starts one MTP page early (the page of row 511): pages 1, 2 and 3 for rows below 1023.
    assert_eq!(layout.speculative_slots(mtp, 512, 1024), 1);
    // Rows [0, 2047) fill 8 pages, and the seed row displaces the last page into a third slot.
    assert_eq!(layout.speculative_slots(mtp, 0, 2048), 3);
    // Rows [0, 1023) fill four pages: the seed row takes room from the fourth.
    assert_eq!(layout.speculative_slots(mtp, 0, 1024), 2);
    assert_eq!(layout.slots(0, 768, Some(mtp)), 4 + 1);
}

#[test]
fn capturing_then_restoring_reproduces_the_mtp_cache_and_the_seed_row() {
    let mut rig = Rig::new(compact(), 16);
    let stored = tokens(0, 2048);
    let mut device = rig.device(8, 8);
    device.fill(3, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 2048, Some(Speculation::Mtp));

    let mut restored = rig.device(8, 8);
    rig.restore(&mut restored, &[node], Some(Speculation::Mtp));
    for &buffer in device.buffers.keys() {
        if is_mtp(buffer) && matches!(buffer, Buffer::MtpPage { page } if page >= 8) {
            continue;
        }
        assert!(device.same(&restored, buffer), "{buffer:?}");
    }
}

#[test]
fn a_restore_without_speculation_leaves_the_mtp_state_alone() {
    let mut rig = Rig::new(compact(), 16);
    let stored = tokens(0, 768);
    let mut device = rig.device(3, 3);
    device.fill(3, |_| true);
    let root = rig.tree.root();
    let node = rig.capture(&device, root, &stored, 768, Some(Speculation::Mtp));

    let mut restored = rig.device(3, 3);
    rig.restore(&mut restored, &[node], None);
    for (&buffer, bytes) in &restored.buffers {
        if is_mtp(buffer) {
            assert!(bytes.iter().all(|&b| b == 0), "{buffer:?}");
        }
    }
}

#[test]
fn a_chain_restores_the_overlapping_mtp_page_from_the_later_node() {
    let mut rig = Rig::new(compact(), 24);
    let stored = tokens(0, 1280);
    let mut device = rig.device(5, 5);
    device.fill(5, |_| true);
    let root = rig.tree.root();
    let mtp = Some(Speculation::Mtp);
    let first = rig.capture(&device, root, &stored, 512, mtp);
    let mut snapshot = rig.device(5, 5);
    snapshot.buffers.clone_from(&device.buffers);
    // The sequence moves on: the MTP page that holds row 511 changes (row 511 is re-paired), and the seed row.
    device.fill(200, is_mtp);
    let second = rig.capture(&device, first, &stored, 1280, mtp);

    let mut restored = rig.device(5, 5);
    rig.restore(&mut restored, &[first, second], mtp);
    for &buffer in device.buffers.keys() {
        match buffer {
            Buffer::MtpPage { page } => {
                let source = if page == 0 { &snapshot } else { &device };
                assert!(
                    source.same(&restored, buffer),
                    "MTP page {page}: page 0 is the first node's, page 1 overlaps and comes from the second"
                );
            }
            Buffer::SeedRow => assert!(device.same(&restored, buffer), "the last node's seed row"),
            _ => {}
        }
    }
}

#[test]
fn the_first_nodes_unshared_mtp_pages_survive_a_chain_restore() {
    let mut rig = Rig::new(compact(), 24);
    let stored = tokens(0, 2048);
    let mut device = rig.device(8, 8);
    device.fill(5, |_| true);
    let root = rig.tree.root();
    let mtp = Some(Speculation::Mtp);
    let first = rig.capture(&device, root, &stored, 1024, mtp);
    let mut snapshot = rig.device(8, 8);
    snapshot.buffers.clone_from(&device.buffers);
    // Only the page that holds row 1023 (index 3) is re-paired by the second node.
    device.fill(77, |buffer| {
        matches!(buffer, Buffer::MtpPage { page } if page >= 3) || buffer == Buffer::SeedRow
    });
    let second = rig.capture(&device, first, &stored, 2048, mtp);

    let mut restored = rig.device(8, 8);
    rig.restore(&mut restored, &[first, second], mtp);
    for &buffer in device.buffers.keys() {
        if let Buffer::MtpPage { page } = buffer {
            if page >= 8 {
                continue;
            }
            let source = if page < 3 { &snapshot } else { &device };
            assert!(source.same(&restored, buffer), "MTP page {page}");
        }
    }
    assert!(device.same(&restored, Buffer::SeedRow));
}

#[test]
fn a_node_that_lost_its_speculative_state_has_no_speculative_copies() {
    let mut rig = Rig::new(compact(), 24);
    let stored = tokens(0, 2048);
    let device = rig.device(8, 8);
    let root = rig.tree.root();
    let plain = rig.capture(&device, root, &stored, 1024, None);
    // The child asks for MTP state under a plain parent: the tree drops the kind, and the layout follows the node.
    let child = rig
        .tree
        .reserve(
            plain,
            &stored,
            2048,
            Some(Speculation::Mtp),
            rig.layout.slots(1024, 2048, None),
        )
        .unwrap();
    assert_eq!(rig.tree.node(child).speculation(), None);
    assert!(
        rig.layout
            .copies(rig.tree.node(child))
            .iter()
            .all(|copy| !is_mtp(copy.buffer))
    );
    rig.tree.abort(child);
}

#[test]
fn an_empty_chain_restores_nothing() {
    let rig = Rig::new(compact(), 4);
    assert!(rig.layout.restore_copies(&[], None).is_empty());
}
