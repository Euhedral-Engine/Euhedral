//! The DFlash2 drafter's configuration: the words of the artifact's `dflash2/config` object, read on the host, and the
//! limits the drafter's kernels put on them (`DFlash2Config`).

use crate::error::PlanError;
use crate::model::config::ModelConfig;

/// The artifact's name for the drafter's configuration object.
pub const CONFIG_OBJECT: &str = "dflash2/config";

/// The version of the configuration words the engine reads.
const VERSION: i32 = 1;

/// Words before the tap layers: the version, 16 dimensions and the tap count.
const FIXED_WORDS: usize = 18;

/// The head dimension the drafter's attention kernels run.
pub const HEAD_DIM: usize = 128;

/// The sliding window the drafter's ring holds at most.
pub const MAX_WINDOW: usize = 2048;

/// The selector's candidate count per row (the top-k kernel's width).
pub const SELECTOR_TOP_K: usize = 16;

/// The selector's rank, at most (the select kernel's block width).
pub const MAX_SELECTOR_RANK: usize = 1024;

/// Query rows times query heads per KV head the attention kernel holds in one tile.
pub const MAX_ATTENTION_TILE_ROWS: usize = 32;

/// The drafter's dimensions and taps.
#[derive(Clone, Debug, PartialEq)]
pub struct Dflash2Config {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rows of a block: the anchor and the mask rows that draft after it.
    pub block_size: usize,
    pub mask_token: u32,
    pub conv_kernel: usize,
    pub conv_group: usize,
    pub selector_top_k: usize,
    pub selector_rank: usize,
    pub sliding_window: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
    /// The target layers whose output rows the drafter reads, increasing.
    pub taps: Vec<usize>,
}

impl Dflash2Config {
    /// The configuration of the `bytes` of the `dflash2/config` object: little-endian 32-bit words.
    pub fn parse(bytes: &[u8]) -> Result<Self, PlanError> {
        let invalid = |why: String| Err(PlanError::Inventory(why));
        if !bytes.len().is_multiple_of(4) {
            return invalid(format!(
                "'{CONFIG_OBJECT}' holds {} bytes, which is not a whole number of 32-bit words",
                bytes.len()
            ));
        }
        let words: Vec<i32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect();
        if words.len() < FIXED_WORDS {
            return invalid(format!(
                "'{CONFIG_OBJECT}' holds {} words, at least {FIXED_WORDS} are needed",
                words.len()
            ));
        }
        if words[0] != VERSION {
            return invalid(format!(
                "'{CONFIG_OBJECT}' is version {}, the engine reads {VERSION}",
                words[0]
            ));
        }
        let tap_count = usize::try_from(words[17]).unwrap_or(usize::MAX);
        if words.len() != FIXED_WORDS.saturating_add(tap_count) {
            return invalid(format!(
                "'{CONFIG_OBJECT}' holds {} words for {} taps, expected {}",
                words.len(),
                words[17],
                FIXED_WORDS.saturating_add(tap_count)
            ));
        }
        let size = |index: usize| usize::try_from(words[index]).map_err(|_| negative(index, words[index]));
        let mask_token = u32::try_from(words[8]).map_err(|_| negative(8, words[8]))?;
        let taps = words[FIXED_WORDS..]
            .iter()
            .map(|tap| usize::try_from(*tap).map_err(|_| negative(FIXED_WORDS, *tap)))
            .collect::<Result<Vec<_>, _>>()?;
        let config = Self {
            layers: size(1)?,
            hidden: size(2)?,
            intermediate: size(3)?,
            heads: size(4)?,
            kv_heads: size(5)?,
            head_dim: size(6)?,
            block_size: size(7)?,
            mask_token,
            conv_kernel: size(9)?,
            conv_group: size(10)?,
            selector_top_k: size(11)?,
            selector_rank: size(12)?,
            sliding_window: size(13)?,
            vocab: size(14)?,
            eps: f32::from_bits(words[15] as u32),
            theta: f32::from_bits(words[16] as u32),
            taps,
        };
        config.check_shape()?;
        Ok(config)
    }

    /// Refuses a configuration the drafter's layers cannot be built from.
    fn check_shape(&self) -> Result<(), PlanError> {
        let bad = self.layers == 0
            || self.hidden == 0
            || self.intermediate == 0
            || self.block_size < 2
            || self.taps.is_empty()
            || self.heads == 0
            || self.kv_heads == 0
            || !self.head_dim.is_multiple_of(2)
            || self.head_dim == 0
            || !self.heads.is_multiple_of(self.kv_heads)
            || self.conv_kernel == 0
            || self.conv_group == 0
            || !self.hidden.is_multiple_of(self.conv_group);
        if bad {
            return Err(PlanError::Inventory(format!(
                "'{CONFIG_OBJECT}' is not a drafter: {} layers, hidden {}, intermediate {}, {} heads over {} KV heads \
                 of {}, block size {} (at least 2), {} taps, convolution kernel {} in groups of {}",
                self.layers,
                self.hidden,
                self.intermediate,
                self.heads,
                self.kv_heads,
                self.head_dim,
                self.block_size,
                self.taps.len(),
                self.conv_kernel,
                self.conv_group,
            )));
        }
        Ok(())
    }

    /// Refuses a drafter the target or the kernels cannot run: the taps must be increasing layers of the target, the
    /// hidden and vocabulary widths the target's, the mask token inside the vocabulary, the epsilon and the rotary base
    /// finite and positive, and the sizes within what the kernels hold.
    pub fn check_against(&self, target: &ModelConfig) -> Result<(), PlanError> {
        let unsupported = |why: String| Err(PlanError::Unsupported(why));
        if let Some(tap) = self.taps.iter().find(|tap| **tap >= target.num_hidden_layers) {
            return unsupported(format!(
                "a drafter tap reads layer {tap}, the target has {} layers",
                target.num_hidden_layers
            ));
        }
        if self.taps.windows(2).any(|pair| pair[0] >= pair[1]) {
            return unsupported(format!(
                "the drafter's taps {:?} are not strictly increasing",
                self.taps
            ));
        }
        if self.hidden != target.hidden_size || self.vocab != target.vocab_size {
            return unsupported(format!(
                "the drafter's hidden {} and vocabulary {} must be the target's ({} and {})",
                self.hidden, self.vocab, target.hidden_size, target.vocab_size
            ));
        }
        if self.mask_token as usize >= target.vocab_size {
            return unsupported(format!(
                "the drafter's mask token {} is outside the target's vocabulary of {}",
                self.mask_token, target.vocab_size
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return unsupported(format!(
                "the drafter's norm epsilon is {}, it must be finite and above 0",
                self.eps
            ));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return unsupported(format!(
                "the drafter's rotary base is {}, it must be finite and above 0",
                self.theta
            ));
        }
        if self.head_dim != HEAD_DIM {
            return unsupported(format!(
                "the drafter's head dimension is {}, the kernels run {HEAD_DIM}",
                self.head_dim
            ));
        }
        if self.sliding_window == 0 || self.sliding_window > MAX_WINDOW {
            return unsupported(format!(
                "the drafter's window is {}, the ring holds 1 to {MAX_WINDOW}",
                self.sliding_window
            ));
        }
        let group = self.heads / self.kv_heads;
        if group > 4 || self.block_size * group > MAX_ATTENTION_TILE_ROWS {
            return unsupported(format!(
                "the drafter's {} heads over {} KV heads with {} rows exceed the attention tile of {MAX_ATTENTION_TILE_ROWS} \
                 rows at 4 heads per KV head",
                self.heads, self.kv_heads, self.block_size
            ));
        }
        if self.selector_top_k != SELECTOR_TOP_K || self.selector_rank == 0 || self.selector_rank > MAX_SELECTOR_RANK {
            return unsupported(format!(
                "the drafter's selector takes the top {} at rank {}, the kernels run the top {SELECTOR_TOP_K} at rank \
                 1 to {MAX_SELECTOR_RANK}",
                self.selector_top_k, self.selector_rank
            ));
        }
        if self.vocab < 1024 {
            return unsupported(format!(
                "the drafter's top-k needs a vocabulary of 1024 or more, got {}",
                self.vocab
            ));
        }
        Ok(())
    }

    /// Width of the concatenated tap rows the drafter's input projection reads.
    pub fn tap_width(&self) -> usize {
        self.taps.len() * self.hidden
    }

    /// Width of the attention queries.
    pub fn query_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Width of one of the keys or values.
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Output width of a convolution's projection: a dynamic filter per group, per tap, per half.
    pub fn conv_projection_width(&self) -> usize {
        2 * self.conv_kernel * (self.hidden / self.conv_group)
    }

    /// Rows a block drafts after its anchor.
    pub fn drafts_per_block(&self) -> usize {
        self.block_size - 1
    }
}

fn negative(index: usize, value: i32) -> PlanError {
    PlanError::Inventory(format!(
        "'{CONFIG_OBJECT}' word {index} is {value}, which cannot be negative"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn words() -> Vec<i32> {
        vec![
            1,
            5,
            5120,
            17408,
            32,
            8,
            128,
            8,
            248070,
            2,
            16,
            16,
            256,
            2048,
            248320,
            1e-6f32.to_bits() as i32,
            1e7f32.to_bits() as i32,
            5,
            5,
            19,
            33,
            47,
            61,
        ]
    }

    pub(crate) fn bytes(words: &[i32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    fn target() -> ModelConfig {
        ModelConfig {
            vocab_size: 248320,
            hidden_size: 5120,
            num_hidden_layers: 64,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            attention_head_dim: 1,
            intermediate_size: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            rms_norm_epsilon: 1.0,
            rope_theta: 1.0,
            partial_rotary_factor: 1.0,
            max_position_embeddings: 1,
            layer_kinds: Vec::new(),
            num_experts: 0,
            tie_word_embeddings: false,
            attention_output_gate: true,
            mtp_layer_count: 0,
            draft_depth: None,
            dflash2: None,
        }
    }

    #[test]
    fn the_artifacts_words_give_the_drafter() {
        let config = Dflash2Config::parse(&bytes(&words())).unwrap();
        assert_eq!(config.taps, [5, 19, 33, 47, 61]);
        assert_eq!(config.tap_width(), 5 * 5120);
        assert_eq!(config.conv_projection_width(), 2 * 2 * 320);
        assert_eq!(config.drafts_per_block(), 7);
        assert_eq!(config.mask_token, 248070);
    }

    #[test]
    fn a_malformed_object_is_refused_with_what_it_holds() {
        let refused = |words: Vec<i32>| Dflash2Config::parse(&bytes(&words)).unwrap_err().to_string();
        assert!(refused(words()[..10].to_vec()).contains("10 words"));
        let mut bad_version = words();
        bad_version[0] = 2;
        assert!(refused(bad_version).contains("version 2"));
        let mut short_taps = words();
        short_taps.pop();
        assert!(refused(short_taps).contains("4 taps") || refused(words()[..22].to_vec()).contains("5 taps"));
        let mut small_block = words();
        small_block[7] = 1;
        assert!(refused(small_block).contains("block size 1"));
        let mut no_taps = words()[..18].to_vec();
        no_taps[17] = 0;
        assert!(refused(no_taps).contains("0 taps"));
        let mut odd_group = words();
        odd_group[10] = 7;
        assert!(refused(odd_group).contains("groups of 7"));
        assert!(
            Dflash2Config::parse(&[1, 2, 3])
                .unwrap_err()
                .to_string()
                .contains("3 bytes")
        );
    }

    #[test]
    fn a_drafter_the_target_or_kernels_cannot_run_is_refused() {
        let target = target();
        let refused = |edit: &dyn Fn(&mut Dflash2Config)| {
            let mut config = Dflash2Config::parse(&bytes(&words())).unwrap();
            edit(&mut config);
            config.check_against(&target).unwrap_err().to_string()
        };
        assert!(
            Dflash2Config::parse(&bytes(&words()))
                .unwrap()
                .check_against(&target)
                .is_ok()
        );
        assert!(refused(&|c| c.taps[4] = 64).contains("layer 64"));
        assert!(refused(&|c| c.taps = vec![19, 5]).contains("increasing"));
        assert!(refused(&|c| c.hidden = 4096).contains("4096"));
        assert!(refused(&|c| c.head_dim = 64).contains("head dimension is 64"));
        assert!(refused(&|c| c.sliding_window = 4096).contains("window is 4096"));
        assert!(refused(&|c| c.block_size = 17).contains("attention tile"));
        assert!(refused(&|c| c.selector_rank = 2048).contains("rank"));
    }
}
