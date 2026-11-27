use std::sync::Arc;

use crate::model::dflash2::Dflash2Config;
use crate::model::{LayerKind, ModelConfig};
use euhedral_memory::{LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope, SlotArena};

/// An arena of `count` slots of `slot_bytes`, addressed as offsets from 0, accounted on a host location.
pub fn arena(count: u32, slot_bytes: u64) -> Arc<SlotArena> {
    let planner = Planner::new([LocationSpec {
        kind: LocationKind::HostPinned,
        capacity: u64::from(count.max(1)) * slot_bytes,
        reserve: 0,
    }]);
    let location = planner.location(LocationKind::HostPinned).expect("the host location");
    let owner = planner.open_owner(OwnerSpec::new("prefix", Priority(10), Scope::Epoch(1)));
    let claim = planner
        .claim(owner, location, u64::from(count) * slot_bytes, "prefix slots")
        .expect("the claim fits");
    SlotArena::new(0, slot_bytes, count, claim)
}

pub fn tokens(from: u32, count: u32) -> Vec<u32> {
    (from..from + count).collect()
}

pub fn config(kinds: Vec<LayerKind>, key_heads: usize, value_heads: usize, key_dim: usize) -> ModelConfig {
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
        dflash2: None,
    }
}

/// Eight layers, GDN on the even ones and full attention on the odd ones; small GDN state.
pub fn compact() -> ModelConfig {
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

/// A drafter of `layers` layers whose rings hold `window` rows of 64 values (BF16): a ring is `128 * window` bytes.
pub fn drafter(layers: usize, window: usize) -> Dflash2Config {
    Dflash2Config {
        layers,
        hidden: 64,
        intermediate: 64,
        heads: 1,
        kv_heads: 4,
        head_dim: 16,
        block_size: 8,
        mask_token: 0,
        conv_kernel: 2,
        conv_group: 16,
        selector_top_k: 16,
        selector_rank: 16,
        sliding_window: window,
        vocab: 1024,
        eps: 1e-6,
        theta: 1e7,
        taps: vec![1, 3],
    }
}
