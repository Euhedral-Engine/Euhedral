//! Tiny Flash-Next artifacts for tests: the objects a small configuration implies, laid out as a writer would, with
//! zeroed payloads (the reader never reads them). A test edits the public fields (drops or adds an object, changes a
//! shape or an offset, damages the metadata) and builds the bytes.

#![allow(
    clippy::expect_used,
    clippy::panic,
    reason = "a test builder: an artifact that cannot be built is a broken test"
)]

use crate::error::EdrlError;
use crate::flash::config::{
    AttentionConfig, FlashConfig, FlashLayerType, GdnConfig, HyperConnectionConfig, MoeConfig, MtpConfig, NgramConfig,
    PleConfig, QsaConfig, TextConfig, VisionConfig,
};
use crate::flash::header::{DATA_ALIGNMENT, FLASH_HEADER_BYTES, FlashHeader, OBJECT_ALIGNMENT};
use crate::flash::inventory::ExpectedInventory;
use crate::flash::metadata::FlashMetadata;
use crate::flash::ngram::NgramLayout;
use crate::flash::tables::{ExpertBank, ExpertProjection, ExpertRecord, FlashTensor, encode_flash_tables};
use crate::format::{TensorDataType, WeightFormat, WeightLayout, payload_size};
use crate::writer::{TokenizerTexts, encode_tokenizer_section};

/// The tokenizer the tiny artifacts carry.
pub const TOKENIZER: TokenizerTexts<'static> = TokenizerTexts {
    tokenizer: "{}",
    tokenizer_config: "{}",
    chat_template: "template",
    special_tokens: "{\"eos_token_id\":[2]}",
};

/// A small configuration that exercises every object kind: a GDN and a sparse-attention layer, an n-gram table in the
/// second layer, one MTP layer and one vision block.
pub fn tiny_config() -> FlashConfig {
    FlashConfig {
        text: TextConfig {
            vocab_size: 32,
            hidden_size: 128,
            num_layers: 2,
            max_position_embeddings: 256,
            rms_norm_epsilon: 1e-6,
            hidden_activation: "silu".into(),
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            layer_types: vec![FlashLayerType::GatedDeltaNet, FlashLayerType::SparseAttention],
        },
        attention: AttentionConfig {
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            partial_rotary_factor: 0.25,
            rope_theta: 10000.0,
            mrope_section: vec![1, 1, 1],
            mrope_interleaved: true,
            output_gate: "sigmoid".into(),
            full_attention_interval: 2,
        },
        gdn: GdnConfig {
            num_key_heads: 1,
            num_value_heads: 2,
            key_head_dim: 16,
            value_head_dim: 64,
            conv_kernel_dim: 4,
            ssm_dtype: "float32".into(),
        },
        qsa: QsaConfig {
            indexer_budget: 8,
            compress_ratio: 2,
            head_dim: 16,
            kv_heads: 1,
            num_heads: 2,
        },
        moe: MoeConfig {
            num_experts: 3,
            experts_per_token: 2,
            intermediate_size: 128,
            shared_expert_intermediate_size: 128,
        },
        hyper_connection: HyperConnectionConfig { count: 2, lowrank: 4 },
        ngram: NgramConfig {
            size: 2,
            vocab_size_base: 100,
            heads_per_ngram: 2,
            vocab_divisible_by: 1,
            split_parts: 2,
            shard_rows: 16,
            layer_multipliers: vec![3, 5],
            heads_offsets: vec![0, 10],
            heads_vocab_sizes: vec![10, 12],
        },
        ple: PleConfig {
            layers: vec![1],
            embed_dim: 64,
            conv_kernel_size: 4,
        },
        mtp: MtpConfig {
            num_layers: 1,
            hybrid: true,
            layer_types: vec![FlashLayerType::GatedDeltaNet],
            rope_theta: 10000.0,
            use_dedicated_embeddings: false,
            use_hidden_state_from_layer: -1,
        },
        vision: VisionConfig {
            depth: 1,
            hidden_size: 16,
            intermediate_size: 32,
            num_heads: 2,
            in_channels: 3,
            patch_size: 2,
            spatial_merge_size: 2,
            temporal_patch_size: 2,
            out_hidden_size: 24,
            num_position_embeddings: 8,
            hidden_activation: "gelu".into(),
            deepstack_visual_indexes: vec![0],
            image_token_id: 3,
            video_token_id: 4,
            start_token_id: 5,
            end_token_id: 6,
        },
    }
}

/// A Flash-Next artifact under construction. [`TinyFlash::new`] holds the inventory `config` implies, laid out;
/// [`TinyFlash::layout`] lays out again after an edit (sizes follow shapes, offsets follow order); [`TinyFlash::image`]
/// encodes the fields as they are.
#[derive(Debug, Clone)]
pub struct TinyFlash {
    pub config: FlashConfig,
    pub metadata: FlashMetadata,
    pub tensors: Vec<FlashTensor>,
    pub banks: Vec<ExpertBank>,
    /// Whether the file carries a tokenizer section.
    pub tokenizer: bool,
    /// Where the payloads start; set by [`TinyFlash::layout`].
    pub data_offset: u64,
    /// Where the content ends; set by [`TinyFlash::layout`].
    pub content_end: u64,
}

/// The bytes of a tiny artifact: a whole file.
#[derive(Debug, Clone)]
pub struct FlashImage {
    pub bytes: Vec<u8>,
}

impl FlashImage {
    pub fn file_len(&self) -> u64 {
        self.bytes.len() as u64
    }

    /// Opens the image through [`crate::FlashArtifact::open`].
    pub fn open(&self) -> Result<crate::FlashArtifact, EdrlError> {
        crate::FlashArtifact::open(self.file_len(), |range| {
            let end = usize::try_from(range.end).map_err(|_| EdrlError::Truncated { what: "image" })?;
            let start = usize::try_from(range.start).map_err(|_| EdrlError::Truncated { what: "image" })?;
            self.bytes
                .get(start..end)
                .map(<[u8]>::to_vec)
                .ok_or(EdrlError::Truncated { what: "image" })
        })
    }
}

impl TinyFlash {
    /// The artifact `config` implies, with every object laid out and a tokenizer section.
    pub fn new(config: FlashConfig) -> TinyFlash {
        let inventory = ExpectedInventory::from_config(&config).expect("the tiny configuration is derivable");
        let mut names: Vec<_> = inventory.tensors().keys().collect();
        names.sort();
        let tensors = names
            .into_iter()
            .map(|name| {
                let want = &inventory.tensors()[name];
                FlashTensor {
                    name: name.clone(),
                    shape: want.shape.clone(),
                    dtype: TensorDataType::Bf16,
                    format: want.format,
                    layout: want.layout,
                    group: want.group,
                    offset: 0,
                    bytes: 0,
                    crc32: 0,
                }
            })
            .collect();
        let banks = inventory
            .banks()
            .iter()
            .map(|want| {
                let projections = want
                    .projections
                    .iter()
                    .map(|(name, shape)| ExpertProjection {
                        name: name.clone(),
                        shape: shape.clone(),
                        dtype: TensorDataType::Bf16,
                        format: WeightFormat::Nvfp4,
                        layout: WeightLayout::RowSplitK128V1,
                        record_offset: 0,
                        bytes: 0,
                    })
                    .collect();
                let records = vec![
                    ExpertRecord {
                        offset: 0,
                        bytes: 0,
                        crc32: 0
                    };
                    want.experts
                ];
                ExpertBank::new(want.name.clone(), want.group, want.layer, projections, records)
                    .expect("the tiny bank is non-empty")
            })
            .collect();
        let metadata = config.to_metadata();
        let mut tiny = TinyFlash {
            config,
            metadata,
            tensors,
            banks,
            tokenizer: true,
            data_offset: 0,
            content_end: 0,
        };
        tiny.layout();
        tiny
    }

    /// Sizes every object from its shape and places it: fixed objects in order, then each bank's records, every
    /// object on a 256-byte boundary, the first at the data section.
    pub fn layout(&mut self) {
        let metadata = self.metadata.encode().expect("tiny metadata encodes");
        for tensor in &mut self.tensors {
            tensor.bytes = size_of(&tensor.shape, tensor.dtype, tensor.format, tensor.layout);
        }
        let mut banks = Vec::new();
        let mut sized = Vec::new();
        for bank in &self.banks {
            let mut projections = bank.projections().to_vec();
            let mut at = 0;
            for projection in &mut projections {
                projection.bytes = size_of(
                    &projection.shape,
                    projection.dtype,
                    projection.format,
                    projection.layout,
                );
                projection.record_offset = at;
                at = align(at + projection.bytes);
            }
            sized.push((at, bank.records().len()));
            banks.push(projections);
        }
        // The tables' size depends on names only, so lay out once with placeholder offsets to measure them.
        let draft = self.encode_tables(&banks, &sized, 0);
        let tables_end = FLASH_HEADER_BYTES as u64 + metadata.len() as u64 + draft.len() as u64;
        self.data_offset = tables_end.div_ceil(DATA_ALIGNMENT) * DATA_ALIGNMENT;
        let mut cursor = self.data_offset;
        for tensor in &mut self.tensors {
            tensor.offset = cursor;
            cursor = align(cursor + tensor.bytes);
        }
        let mut rebuilt = Vec::new();
        for (bank, (projections, (record_bytes, experts))) in self.banks.iter().zip(banks.into_iter().zip(sized)) {
            let records = (0..experts)
                .map(|_| {
                    let record = ExpertRecord {
                        offset: cursor,
                        bytes: record_bytes,
                        crc32: 0,
                    };
                    cursor = align(cursor + record_bytes);
                    record
                })
                .collect();
            rebuilt.push(
                ExpertBank::new(bank.name().to_owned(), bank.group(), bank.layer(), projections, records)
                    .expect("a laid-out bank is valid"),
            );
        }
        self.banks = rebuilt;
        self.content_end = self
            .tensors
            .iter()
            .map(|t| t.offset + t.bytes)
            .chain(
                self.banks
                    .iter()
                    .flat_map(|bank| bank.records().iter().map(|r| r.offset + r.bytes)),
            )
            .max()
            .unwrap_or(self.data_offset);
    }

    fn encode_tables(&self, projections: &[Vec<ExpertProjection>], sized: &[(u64, usize)], offset: u64) -> Vec<u8> {
        let banks: Vec<ExpertBank> = self
            .banks
            .iter()
            .zip(projections.iter().zip(sized))
            .map(|(bank, (projections, (record_bytes, experts)))| {
                let records = vec![
                    ExpertRecord {
                        offset,
                        bytes: *record_bytes,
                        crc32: 0
                    };
                    *experts
                ];
                ExpertBank::new(
                    bank.name().to_owned(),
                    bank.group(),
                    bank.layer(),
                    projections.clone(),
                    records,
                )
                .expect("a bank is valid")
            })
            .collect();
        encode_flash_tables(&self.tensors, &banks).expect("tiny tables encode")
    }

    /// The file for the fields as they are: header, metadata, tables, zeroed payloads, tokenizer section.
    pub fn image(&self) -> FlashImage {
        let metadata = self.metadata.encode().expect("tiny metadata encodes");
        let tables = encode_flash_tables(&self.tensors, &self.banks).expect("tiny tables encode");
        let header = FlashHeader::new(
            metadata.len() as u64,
            tables.len() as u64,
            self.data_offset,
            self.content_end,
        )
        .expect("tiny header is valid");
        let mut bytes = header.encode().to_vec();
        bytes.extend_from_slice(&metadata);
        bytes.extend_from_slice(&tables);
        bytes.resize(usize::try_from(self.content_end).expect("tiny file fits"), 0);
        if self.tokenizer {
            bytes.extend_from_slice(&encode_tokenizer_section(&TOKENIZER).expect("tiny tokenizer encodes"));
        }
        FlashImage { bytes }
    }
}

fn align(value: u64) -> u64 {
    value.div_ceil(OBJECT_ALIGNMENT) * OBJECT_ALIGNMENT
}

fn size_of(shape: &[u64], dtype: TensorDataType, format: WeightFormat, layout: WeightLayout) -> u64 {
    if layout == WeightLayout::RowInterleavedNvfp4V1 {
        let &[rows, k] = shape else {
            panic!("an n-gram shard is rank 2")
        };
        return NgramLayout::new(rows, k).expect("tiny shard is valid").byte_size();
    }
    payload_size(shape, dtype, format, layout).expect("tiny object has a size")
}
