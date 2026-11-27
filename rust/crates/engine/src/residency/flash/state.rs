//! The device bytes a running sequence and the execution workspaces need, from the configuration alone
//! (`SequenceState`): what the residency plan sets aside beside the weights.
//!
//! - Attention keys and values live in NVFP4 pages of [`PAGE_TOKENS`] tokens: a row of 256 values takes 144 bytes, and a
//!   token stores one key row and one value row per 256 values of each KV head, in every sparse-attention layer.
//! - The indexer keeps one BF16 key of `qsa.head_dim` per `qsa.kv_heads` for every complete block of
//!   `qsa.compress_ratio` tokens of each sparse-attention layer, and two raw tails of the incomplete block's keys.
//! - Gated DeltaNet keeps an FP32 `key_head_dim x value_head_dim` state per value head and the last `conv_kernel_dim - 1`
//!   inputs of its convolution in BF16, for each GDN layer, whatever the length.
//!
//! Every size is `None` when it overflows 64 bits; [`FlashModelConfig::check_supported`] proves the sizes a plan asks
//! for do not.

use crate::flash::qsa::state::indexer_layer_bytes;
use crate::model::{FlashModelConfig, KV_ROW_VALUES};

/// Tokens per KV page.
pub const PAGE_TOKENS: u64 = 256;

/// The bytes of one NVFP4 row of [`KV_ROW_VALUES`] values: codes, scales and the global.
const KV_ROW_BYTES: u64 = 144;

/// Prefill tokens an execution step holds at once unless the plan finds room for more.
pub const PREFILL_CHUNK_TOKENS: u64 = 512;

/// Rows of the workspaces that serve a decode token and a short chunk, beside the full chunk's.
const SMALL_WORKSPACE_ROWS: u64 = 1 + 16;

/// Rows of one decode or verification step's logits.
const LOGIT_ROWS: u64 = 8;

/// The accounting of one configuration.
#[derive(Clone, Copy, Debug)]
pub struct SequenceState<'a> {
    config: &'a FlashModelConfig,
}

impl<'a> SequenceState<'a> {
    pub fn new(config: &'a FlashModelConfig) -> Self {
        Self { config }
    }

    /// Key and value bytes of one sequence of `tokens` positions, in whole pages.
    pub fn kv_bytes(&self, tokens: usize) -> Option<u64> {
        let config = self.config;
        let pages = (tokens as u64).div_ceil(PAGE_TOKENS);
        let rows_per_token = config.kv_heads.checked_mul(config.head_dim)? / KV_ROW_VALUES;
        config
            .sparse_attention_layers
            .checked_mul(pages)?
            .checked_mul(PAGE_TOKENS * 2 * KV_ROW_BYTES)?
            .checked_mul(rows_per_token)
    }

    /// Indexer bytes of one sequence: per sparse-attention layer, one key per complete block and the two raw tails of
    /// the incomplete block (`QsaState.indexerBytes`).
    pub fn indexer_bytes(&self, tokens: usize) -> Option<u64> {
        let config = self.config;
        let key_bytes = config.qsa_kv_heads.checked_mul(config.qsa_head_dim)?.checked_mul(2)?;
        let layer = indexer_layer_bytes(tokens as u64, config.qsa_compress_ratio, key_bytes)?;
        config.sparse_attention_layers.checked_mul(layer)
    }

    /// Recurrent state and convolution history of one sequence.
    pub fn gdn_state_bytes(&self) -> Option<u64> {
        let config = self.config;
        let state = config
            .gdn_value_heads
            .checked_mul(config.gdn_key_head_dim)?
            .checked_mul(config.gdn_value_head_dim)?
            .checked_mul(4)?;
        let convolution = config
            .gdn_convolution_channels()?
            .checked_mul(config.gdn_conv_kernel_dim - 1)?
            .checked_mul(2)?;
        config.gdn_layers().checked_mul(state.checked_add(convolution)?)
    }

    /// Everything that grows with, or is held for, one sequence of `tokens` positions.
    pub fn context_bytes(&self, tokens: usize) -> Option<u64> {
        self.kv_bytes(tokens)?
            .checked_add(self.indexer_bytes(tokens)?)?
            .checked_add(self.gdn_state_bytes()?)
    }

    /// Activation storage of one execution step for chunks of `chunk_tokens` tokens: for each token of a prefill chunk
    /// the hyper-connection streams (the layer's input, its mixed input and its output), the widest token mixer's
    /// intermediates (double buffered, BF16), the router logits and the selected experts' gate/up and down activations;
    /// plus the logits of a decode step. It counts every workspace the execution plan allocates: the smaller ones that
    /// serve a decode token and a short chunk beside the full chunk's. A bound sized from the topology, not a
    /// measurement.
    pub fn workspace_bytes(&self, chunk_tokens: u64) -> Option<u64> {
        let config = self.config;
        let hidden = config.hidden_size;
        let stream = config.hyper_connections.checked_mul(hidden)?.checked_mul(2)?;
        let gdn_width = config.gdn_convolution_channels()?.checked_add(
            2u64.checked_mul(config.gdn_value_heads)?
                .checked_mul(config.gdn_value_head_dim)?,
        )?;
        let attention_width = 2u64
            .checked_mul(config.attention_heads)?
            .checked_mul(config.head_dim)?
            .checked_add(2u64.checked_mul(config.kv_heads)?.checked_mul(config.head_dim)?)?;
        let mixer = 2u64.checked_mul(gdn_width.max(attention_width))?.checked_mul(2)?;
        let selected = config
            .experts_per_token
            .checked_mul(config.expert_intermediate_size.checked_mul(2)?.checked_add(hidden)?)?
            .checked_mul(2)?;
        let experts = selected
            .checked_add(config.num_experts.checked_mul(4)?)?
            .checked_add(config.experts_per_token.checked_mul(8)?)?;
        let per_token = stream
            .checked_mul(3)?
            .checked_add(mixer)?
            .checked_add(experts)?
            .checked_add(hidden.checked_mul(8)?)?;
        let logits = LOGIT_ROWS.checked_mul(config.vocab_size)?.checked_mul(4)?;
        chunk_tokens
            .checked_add(SMALL_WORKSPACE_ROWS)?
            .checked_mul(per_token)?
            .checked_add(logits)
    }

    /// The shared expert's width as a prefill's down projection reads it: a multiple of 256, as the native NVFP4 kernels
    /// need for more than 64 rows (they fetch the weight scales 256 values at a time).
    pub fn padded_shared_width(&self) -> u64 {
        self.config.shared_expert_intermediate_size.next_multiple_of(256)
    }

    /// Device bytes of the shared expert's down projections padded to [`padded_shared_width`] (one per layer, plain
    /// NVFP4: codes, 256-aligned scales, the 256-aligned global) and of the padding in the activation of the longest
    /// prefill chunk; 0 when the width needs none.
    ///
    /// [`padded_shared_width`]: SequenceState::padded_shared_width
    pub fn shared_down_padding_bytes(&self) -> Option<u64> {
        let config = self.config;
        let width = config.shared_expert_intermediate_size;
        let padded = self.padded_shared_width();
        if padded == width {
            return Some(0);
        }
        let hidden = config.hidden_size;
        let scales = (hidden.checked_mul(padded)? / 2).next_multiple_of(256);
        let tensor = scales
            .checked_add(hidden.checked_mul(padded)? / 16)?
            .next_multiple_of(256)
            .checked_add(4)?;
        let activation = super::LARGEST_PREFILL_CHUNK_TOKENS
            .checked_mul(padded - width)?
            .checked_mul(2)?;
        config
            .num_layers
            .checked_mul(tensor.next_multiple_of(256))?
            .checked_add(activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::flash_config::fixtures::flash_config;

    /// Two layers (one GDN, one sparse), hidden 128, 2 streams, KV 1 x 256, indexer 1 x 16 per 4 tokens, GDN 2 value
    /// heads of 16 x 64 with a 4-tap convolution, 3 experts of which 2 serve a token, vocabulary 32.
    fn config() -> FlashModelConfig {
        FlashModelConfig::from_flash(&flash_config()).unwrap()
    }

    #[test]
    fn the_kv_cache_is_whole_pages_of_two_rows_per_kv_head_in_every_sparse_layer() {
        let config = config();
        let state = SequenceState::new(&config);
        // 1 sparse layer x 1 page x 256 tokens x (K + V) x 144 bytes x 1 row.
        assert_eq!(state.kv_bytes(1), Some(256 * 2 * 144));
        assert_eq!(state.kv_bytes(256), Some(256 * 2 * 144));
        assert_eq!(state.kv_bytes(257), Some(2 * 256 * 2 * 144));
        assert_eq!(state.kv_bytes(0), Some(0));
    }

    #[test]
    fn the_indexer_keeps_one_bf16_key_per_complete_block_and_two_raw_tails() {
        let config = config();
        let state = SequenceState::new(&config);
        // 10 / 4 = 2 keys and two tails of 3 keys, of 1 head x 16 values x 2 bytes.
        assert_eq!(state.indexer_bytes(10), Some((2 + 6) * 16 * 2));
        // Fewer tokens than a block still keep one key.
        assert_eq!(state.indexer_bytes(1), Some((1 + 6) * 16 * 2));
    }

    #[test]
    fn the_gdn_state_does_not_grow_with_the_context() {
        let config = config();
        let state = SequenceState::new(&config);
        // 1 layer: 2 value heads x 16 x 64 x 4 bytes, plus (2 x 16 + 2 x 64) channels x 3 inputs x 2 bytes.
        assert_eq!(state.gdn_state_bytes(), Some(2 * 16 * 64 * 4 + 160 * 3 * 2));
        assert_eq!(
            state.context_bytes(256),
            Some(state.kv_bytes(256).unwrap() + state.indexer_bytes(256).unwrap() + state.gdn_state_bytes().unwrap())
        );
    }

    #[test]
    fn the_workspace_counts_every_row_of_the_chunk_and_the_small_workspaces_and_the_logits() {
        let config = config();
        let state = SequenceState::new(&config);
        let stream = 2 * 128 * 2;
        // GDN: 160 channels + 2 x 2 x 64 = 416; attention: 2 x 2 x 256 + 2 x 1 x 256 = 1536; the wider, doubled twice.
        let mixer = 2 * 1536 * 2;
        let experts = 2 * (2 * 128 + 128) * 2 + 3 * 4 + 2 * 8;
        let per_token = 3 * stream + mixer + experts + 4 * 128 * 2;
        let logits = 8 * 32 * 4;
        assert_eq!(state.workspace_bytes(512), Some((512 + 17) * per_token + logits));
        assert_eq!(state.workspace_bytes(0), Some(17 * per_token + logits));
    }

    #[test]
    fn a_shared_expert_whose_width_is_a_multiple_of_256_needs_no_padding_and_another_does() {
        let mut config = config();
        config.shared_expert_intermediate_size = 256;
        assert_eq!(SequenceState::new(&config).shared_down_padding_bytes(), Some(0));
        config.shared_expert_intermediate_size = 128;
        let state = SequenceState::new(&config);
        assert_eq!(state.padded_shared_width(), 256);
        // Per layer: scales 128 x 256 / 2 = 16384; plus 128 x 256 / 16 = 2048 aligned to 256 is 18432; plus the global
        // of 4 bytes, aligned again, is 18688. Two layers, and 4096 rows of 128 padded values in BF16.
        assert_eq!(state.shared_down_padding_bytes(), Some(2 * 18688 + 4096 * 128 * 2));
    }

    #[test]
    fn a_size_that_overflows_is_none_not_a_wrapped_number() {
        let mut config = config();
        config.sparse_attention_layers = u64::MAX;
        assert_eq!(SequenceState::new(&config).kv_bytes(1 << 20), None);
    }
}
