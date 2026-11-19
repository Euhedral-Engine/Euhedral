//! Tiny artifacts for tests, shaped like calibrated ones: raw objects shared by every source, projections and heads
//! that each source fills with its own bytes, and, for a source with companions, `.suh`, `.svh` and `.segments`
//! after each projection. Projections are stored as 64 x 64 Q3 objects, a quantized format.
//!
//! Every byte is a function of the seeds: the payloads come from a 48-bit linear congruential generator
//! (multiplier 0x5DEECE66D, addend 0xB), so the same seed gives the same artifact on every platform.

use crate::error::EdrlError;
use crate::header::HEADER_BYTES;
use crate::format::{TensorDataType, WeightFormat, WeightLayout, direct_size, grouped_size};
use crate::metadata::ModelMetadata;
use crate::writer::{ArtifactPlan, ObjectSpec, TokenizerTexts, encode_metadata, encode_tokenizer_section};

/// The alignment of each payload in a tiny artifact.
pub const TINY_ALIGNMENT: u64 = 256;

/// The tokenizer the tiny artifacts carry: assembly copies it, nothing tokenizes with it.
pub const TOKENIZER: TokenizerTexts<'static> = TokenizerTexts {
    tokenizer: "{}",
    tokenizer_config: "{}",
    chat_template: "template",
    special_tokens: "{\"eos_token_id\":[2]}",
};

/// The projections every tiny artifact has, in file order.
pub const PROJECTIONS: [&str; 4] = [
    "text/layers/0/mlp/gate_up",
    "text/layers/0/mlp/down",
    "text/layers/1/attention/output",
    "text/layers/1/mlp/down",
];

/// The projection of the drafter.
pub const DRAFTER_PROJECTION: &str = "mtp/layer/mlp/down";

/// The seed the shared raw objects come from unless a test says otherwise.
pub const SHARED_SEED: i64 = 99;

/// What a tiny artifact holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TinyArtifact<'a> {
    /// Fills everything but the shared raw objects.
    pub seed: i64,
    /// Adds the trellis companions to the layers' and the drafter's projections.
    pub companions: bool,
    /// Fills the shared raw objects.
    pub shared_seed: i64,
    /// A projection left out (the empty string leaves none out).
    pub omitted: &'a str,
}

impl TinyArtifact<'_> {
    /// An artifact of `seed`, with the shared objects from [`SHARED_SEED`] and no projection left out.
    pub fn new(seed: i64, companions: bool) -> TinyArtifact<'static> {
        TinyArtifact {
            seed,
            companions,
            shared_seed: SHARED_SEED,
            omitted: "",
        }
    }

    /// The whole file: header, metadata, table, payloads at their offsets (zero between them), then the tokenizer
    /// section.
    pub fn bytes(&self) -> Result<Vec<u8>, EdrlError> {
        let (specs, payloads) = self.objects()?;
        let metadata = encode_metadata(&metadata())?;
        let plan = ArtifactPlan::new(metadata.len() as u64, &specs, TINY_ALIGNMENT)?;
        let table = plan.table()?;
        let content_end = to_len(plan.content_end())?;
        let mut file = vec![0; content_end];
        let header = plan.header();
        file[..HEADER_BYTES].copy_from_slice(&header.encode());
        let metadata_at = to_len(header.metadata_offset())?;
        file[metadata_at..metadata_at + metadata.len()].copy_from_slice(&metadata);
        let table_at = to_len(header.tensor_table_offset())?;
        file[table_at..table_at + table.len()].copy_from_slice(&table);
        for (entry, payload) in plan.entries().iter().zip(&payloads) {
            let at = to_len(entry.offset())?;
            file[at..at + payload.len()].copy_from_slice(payload);
        }
        file.extend_from_slice(&encode_tokenizer_section(&TOKENIZER)?);
        Ok(file)
    }

    /// The objects and their payloads, in file order.
    pub fn objects(&self) -> Result<(Vec<ObjectSpec>, Vec<Vec<u8>>), EdrlError> {
        let mut objects = Objects::default();
        let shared = self.shared_seed;
        objects.raw(
            "text/layers/0/input_norm",
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            32,
            shared,
        )?;
        objects.raw(
            "text/layers/0/gdn/a_log",
            TensorDataType::Fp32,
            WeightFormat::Fp32,
            8,
            shared.wrapping_add(1),
        )?;
        objects.quantized("text/token_embedding", self.seed, false)?;
        for name in PROJECTIONS {
            if name != self.omitted {
                objects.quantized(name, self.seed, self.companions)?;
            }
        }
        objects.raw(
            "text/final_norm",
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            32,
            shared.wrapping_add(2),
        )?;
        objects.quantized("text/output_head", self.seed, false)?;
        objects.raw(
            "text/draft_head_token_ids",
            TensorDataType::Int32,
            WeightFormat::I32,
            16,
            shared.wrapping_add(3),
        )?;
        objects.raw(
            "mtp/layer/input_norm",
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            32,
            self.seed,
        )?;
        objects.quantized(DRAFTER_PROJECTION, self.seed, self.companions)?;
        Ok((objects.specs, objects.payloads))
    }
}

/// The model configuration every tiny artifact stores.
pub fn metadata() -> ModelMetadata {
    ModelMetadata {
        vocab_size: 1,
        hidden_size: 1,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        attention_head_dim: 1,
        intermediate_size: 1,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 1,
        linear_value_head_dim: 1,
        linear_conv_kernel_dim: 1,
        rms_norm_epsilon: 1.0e-6,
        rope_theta: 1.0,
        partial_rotary_factor: 1.0,
        max_position_embeddings: 1,
        hidden_activation: "silu".into(),
        layer_types: Vec::new(),
        num_experts: 0,
        num_experts_per_token: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        tie_word_embeddings: false,
        attention_output_gate: false,
        mtp_layer_count: 0,
        draft_depth: None,
    }
}

#[derive(Default)]
struct Objects {
    specs: Vec<ObjectSpec>,
    payloads: Vec<Vec<u8>>,
}

impl Objects {
    fn quantized(&mut self, name: &str, seed: i64, companions: bool) -> Result<(), EdrlError> {
        let shape = vec![64, 64];
        let bytes = grouped_size(&shape, TensorDataType::Bf16, WeightFormat::Q3G64Fp16).map_err(refused(name))?;
        let seed = seed.wrapping_add(i64::from(text_hash(name)));
        self.add(
            name,
            shape,
            (
                TensorDataType::Bf16,
                WeightFormat::Q3G64Fp16,
                WeightLayout::RowSplitK128V1,
            ),
            bytes,
            seed,
        )?;
        if !companions || name == "text/token_embedding" || name == "text/output_head" {
            return Ok(());
        }
        self.raw(
            &format!("{name}.suh"),
            TensorDataType::Fp16,
            WeightFormat::Fp16,
            64,
            seed.wrapping_add(1),
        )?;
        self.raw(
            &format!("{name}.svh"),
            TensorDataType::Fp16,
            WeightFormat::Fp16,
            64,
            seed.wrapping_add(2),
        )?;
        self.raw(
            &format!("{name}.segments"),
            TensorDataType::Int32,
            WeightFormat::I32,
            4,
            seed.wrapping_add(3),
        )
    }

    fn raw(
        &mut self,
        name: &str,
        dtype: TensorDataType,
        format: WeightFormat,
        elements: u64,
        seed: i64,
    ) -> Result<(), EdrlError> {
        let shape = vec![elements];
        let bytes = direct_size(&shape, dtype, format).map_err(refused(name))?;
        self.add(name, shape, (dtype, format, WeightLayout::ContiguousLeV1), bytes, seed)
    }

    fn add(
        &mut self,
        name: &str,
        shape: Vec<u64>,
        (dtype, format, layout): (TensorDataType, WeightFormat, WeightLayout),
        bytes: u64,
        seed: i64,
    ) -> Result<(), EdrlError> {
        let mut payload = vec![0; to_len(bytes)?];
        Lcg::new(seed).fill(&mut payload);
        self.specs.push(ObjectSpec {
            name: name.to_owned(),
            shape,
            dtype,
            format,
            layout,
            bytes,
        });
        self.payloads.push(payload);
        Ok(())
    }
}

fn refused(name: &str) -> impl Fn(String) -> EdrlError + '_ {
    move |reason| EdrlError::UnsupportedTensor {
        name: name.to_owned(),
        reason,
    }
}

fn to_len(value: u64) -> Result<usize, EdrlError> {
    usize::try_from(value).map_err(|_| EdrlError::invalid("a tiny artifact does not fit memory"))
}

/// A 31-based hash over the text's UTF-16 code units, wrapping in 32 bits.
fn text_hash(text: &str) -> i32 {
    text.encode_utf16()
        .fold(0i32, |hash, unit| hash.wrapping_mul(31).wrapping_add(i32::from(unit)))
}

/// A 48-bit linear congruential generator; its draws and the byte filling are fixed, because the golden files
/// pin them.
struct Lcg {
    state: u64,
}

impl Lcg {
    const MULTIPLIER: u64 = 0x5_DEEC_E66D;
    const ADDEND: u64 = 0xB;
    const MASK: u64 = (1 << 48) - 1;

    fn new(seed: i64) -> Self {
        Self {
            state: (seed as u64 ^ Self::MULTIPLIER) & Self::MASK,
        }
    }

    /// The next 32 bits.
    fn next_int(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(Self::MULTIPLIER).wrapping_add(Self::ADDEND) & Self::MASK;
        (self.state >> 16) as u32 as i32
    }

    /// Each draw fills up to four bytes, low byte first.
    fn fill(&mut self, bytes: &mut [u8]) {
        for chunk in bytes.chunks_mut(4) {
            let mut word = self.next_int();
            for byte in chunk {
                *byte = word as u8;
                word >>= 8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Lcg, text_hash};

    #[test]
    fn lcg_matches_the_documented_sequence() {
        // `new Random(42).nextInt()` is -1170105035 and the next is 234785527.
        let mut random = Lcg::new(42);
        assert_eq!(random.next_int(), -1_170_105_035);
        assert_eq!(random.next_int(), 234_785_527);
    }

    #[test]
    fn string_hash_is_the_expected_one() {
        assert_eq!(text_hash(""), 0);
        assert_eq!(text_hash("a"), 97);
        assert_eq!(text_hash("hello"), 99_162_322);
    }
}
