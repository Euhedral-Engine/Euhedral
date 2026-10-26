//! What the sequence state costs on the device: the KV pages, the GDN state and the rest of a sequence's buffers
//! (`ResidencyPlanner.kvBytes`, `SequenceFootprint`, `AttentionKvState.planePageBytes`).

use euhedral_inference_settings::KvFormat;

use crate::model::ModelConfig;

/// Tokens in one KV page (`AttentionKvState.PAGE_TOKENS`).
pub const PAGE_TOKENS: u64 = 256;

/// Rows `GdnState.speculative` allocates at least, however few are verified.
pub const MINIMUM_VERIFY_ROWS: u64 = 8;

/// A scored selection on the device: the 8-byte key then the log-probability, padded (`SequenceFootprint`).
const SELECTION_BYTES: u64 = 2 * 8;

/// Most rows of a draft quantum (MTP catch-up) that run on the decode row twins (`AttentionStates.DRAFT_TWIN_ROWS`).
pub const DRAFT_TWIN_ROWS: u64 = 8;

/// Split areas of one decode row's attention scratch, and the floats of each (`AttentionStates`).
const DECODE_SPLITS: u64 = 64;
const DECODE_SPLIT_FLOATS: u64 = 258;

/// Bytes of one 256-value head row (`KvFormat.headRowBytes`).
pub fn head_row_bytes(format: KvFormat) -> u64 {
    match format {
        KvFormat::Nvfp4 => 144,
        KvFormat::Bf16 => 512,
    }
}

/// Bytes of one plane (K or V) of one page of `kv_width`-wide rows.
pub fn plane_page_bytes(kv_width: u64, format: KvFormat) -> u64 {
    PAGE_TOKENS * head_row_bytes(format) * (kv_width / 256)
}

/// Bytes of one cache page of a full-attention layer: its K plane then its V plane.
pub fn page_bytes(config: &ModelConfig, format: KvFormat) -> u64 {
    2 * plane_page_bytes(config.kv_width() as u64, format)
}

/// Pages that hold `tokens` positions.
pub fn pages_for(tokens: u64) -> u64 {
    tokens.div_ceil(PAGE_TOKENS)
}

/// KV bytes of one sequence of `tokens` positions in `format`, in whole pages.
pub fn kv_bytes(config: &ModelConfig, tokens: u64, format: KvFormat) -> u64 {
    config.attention_layers() as u64 * pages_for(tokens) * page_bytes(config, format)
}

/// GDN state of every GDN layer (recurrent and convolution), its verification checkpoint and its row buffers at
/// `verified_rows` rows, at least [`MINIMUM_VERIFY_ROWS`] (`SequenceFootprint.gdnBytes`).
pub fn gdn_bytes(config: &ModelConfig, verified_rows: u64) -> u64 {
    let layers = config.gdn_layers() as u64;
    let query_key = config.gdn_query_key_width() as u64;
    let value = config.gdn_value_width() as u64;
    let convolved = query_key + value;
    let convolution = convolved * (config.linear_conv_kernel_dim as u64 - 1) * 2;
    let recurrent = config.linear_num_value_heads as u64
        * config.linear_key_head_dim as u64
        * config.linear_value_head_dim as u64
        * 4;
    let rows = verified_rows.max(MINIMUM_VERIFY_ROWS);
    let heads = config.linear_num_value_heads as u64;
    let verification = recurrent
        + convolution
        + rows * query_key * 2
        + rows * 2 * value * 2
        + 2 * rows * heads * 4
        + rows * convolved * 2
        + rows * value * 2;
    layers * (recurrent + convolution + verification)
}

/// Bytes of the split-KV decode scratch for `rows` rows of `query_heads` heads (`AttentionStates.decodeScratchBytes`).
pub fn decode_scratch_bytes(query_heads: u64, rows: u64) -> u64 {
    query_heads * rows * DECODE_SPLITS * DECODE_SPLIT_FLOATS * 4
}

/// What one sequence needs on the device beyond its attention KV (`SequenceFootprint.extraSequenceBytes`): the GDN
/// state, the decode scratch for the most rows it verifies, its device selection words and, when it drafts, the
/// draft seed rows of a prefill chunk (`seed_rows`), the MTP layer's cache for `context_tokens` and the recursion
/// hidden.
pub fn sequence_bytes(
    config: &ModelConfig,
    format: KvFormat,
    context_tokens: u64,
    verified_rows: u64,
    drafts: bool,
    seed_rows: u64,
) -> u64 {
    let scratch_rows = if drafts {
        verified_rows.max(DRAFT_TWIN_ROWS)
    } else {
        verified_rows
    };
    let mut total = gdn_bytes(config, verified_rows)
        + decode_scratch_bytes(config.num_attention_heads as u64, scratch_rows)
        + SELECTION_BYTES
        + verified_rows * 8;
    if !drafts {
        return total;
    }
    let seed_rows = seed_rows.max(MINIMUM_VERIFY_ROWS);
    let hidden = config.hidden_size as u64;
    total += seed_rows * hidden * 2 + hidden * 2;
    // The MTP layer is one full-attention layer's cache.
    total += pages_for(context_tokens) * page_bytes(config, format);
    total
}
