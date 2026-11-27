//! One test per refusal of the Flash-Next reader, each through the interface of the unit that refuses (header, metadata,
//! configuration, tables, n-gram layout, inventory) and each asserting the error it expects and that the message names
//! the value and the limit. Bad input is built with the tiny artifact builder or as raw bytes.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_edrl::tiny_flash::{TinyFlash, tiny_config};
use euhedral_inference_edrl::{
    ComponentGroup, EdrlError, ExpectedInventory, ExpertBank, ExpertProjection, FLASH_HEADER_BYTES, FlashConfig,
    FlashHeader, FlashLayerType, FlashMetadata, FlashTables, FlashTensor, InventoryProblem, MetadataValue, NgramLayout,
    WeightFormat, WeightLayout,
};

fn tiny() -> TinyFlash {
    TinyFlash::new(tiny_config())
}

/// The message of an `Invalid` refusal; any other error fails the test.
fn invalid(error: EdrlError) -> String {
    match error {
        EdrlError::Invalid { what } => what,
        other => panic!("expected an invalid region, got {other:?}"),
    }
}

fn problems(error: EdrlError) -> Vec<InventoryProblem> {
    match error {
        EdrlError::Inventory { problems } => problems,
        other => panic!("expected an inventory refusal, got {other:?}"),
    }
}

/// Fails unless `message` holds every needle: the rule that was broken, the value found and the limit.
fn names(message: &str, needles: &[&str]) {
    for needle in needles {
        assert!(message.contains(needle), "`{message}` does not name `{needle}`");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Header

const FILE_SIZE: u64 = 8192;

fn header_bytes() -> [u8; FLASH_HEADER_BYTES] {
    FlashHeader::new(100, 200, 4096, FILE_SIZE).unwrap().encode()
}

fn header_with(patches: &[(usize, i64)]) -> Result<FlashHeader, EdrlError> {
    let mut bytes = header_bytes();
    for (at, value) in patches {
        let width = if *at == 12 { 4 } else { 8 };
        let encoded = value.to_be_bytes();
        bytes[*at..*at + width].copy_from_slice(&encoded[8 - width..]);
    }
    FlashHeader::parse(&bytes)
}

const RESERVED_AT: usize = 12;
const METADATA_OFFSET_AT: usize = 16;
const METADATA_SIZE_AT: usize = 24;
const TABLES_OFFSET_AT: usize = 32;
const TABLES_SIZE_AT: usize = 40;
const DATA_OFFSET_AT: usize = 48;
const FILE_SIZE_AT: usize = 56;

#[test]
fn a_valid_header_is_read_back() {
    let header = header_with(&[]).unwrap();
    assert_eq!(header.tables_offset(), 164);
    assert_eq!(header.file_size(), FILE_SIZE);
}

#[test]
fn a_header_with_a_reserved_field_that_is_not_zero_is_refused() {
    names(
        &invalid(header_with(&[(RESERVED_AT, 7)]).unwrap_err()),
        &["reserved field is 7", "must be zero"],
    );
}

#[test]
fn a_metadata_offset_that_does_not_follow_the_header_is_refused() {
    names(
        &invalid(header_with(&[(METADATA_OFFSET_AT, 65)]).unwrap_err()),
        &["metadata offset is 65", "must be 64"],
    );
}

#[test]
fn a_tables_offset_that_does_not_follow_the_metadata_is_refused() {
    names(
        &invalid(header_with(&[(TABLES_OFFSET_AT, 200)]).unwrap_err()),
        &["tables offset is 200", "must be 164"],
    );
}

#[test]
fn a_negative_metadata_offset_is_refused() {
    names(
        &invalid(header_with(&[(METADATA_OFFSET_AT, -1)]).unwrap_err()),
        &["metadata offset", "negative: -1"],
    );
}

#[test]
fn a_negative_metadata_size_is_refused() {
    names(
        &invalid(header_with(&[(METADATA_SIZE_AT, -2)]).unwrap_err()),
        &["metadata size", "negative: -2"],
    );
}

#[test]
fn a_negative_tables_offset_is_refused() {
    names(
        &invalid(header_with(&[(TABLES_OFFSET_AT, -3)]).unwrap_err()),
        &["tables offset", "negative: -3"],
    );
}

#[test]
fn a_negative_tables_size_is_refused() {
    names(
        &invalid(header_with(&[(TABLES_SIZE_AT, -4)]).unwrap_err()),
        &["tables size", "negative: -4"],
    );
}

#[test]
fn a_negative_data_offset_is_refused() {
    names(
        &invalid(header_with(&[(DATA_OFFSET_AT, -5)]).unwrap_err()),
        &["data offset", "negative: -5"],
    );
}

#[test]
fn a_negative_file_size_is_refused() {
    names(
        &invalid(header_with(&[(FILE_SIZE_AT, -6)]).unwrap_err()),
        &["file size", "negative: -6"],
    );
}

#[test]
fn a_metadata_size_over_the_limit_is_refused() {
    let size = (8 << 20) + 1;
    let message = invalid(header_with(&[(METADATA_SIZE_AT, size), (TABLES_OFFSET_AT, 64 + size)]).unwrap_err());
    names(&message, &["metadata size 8388609", "over the limit 8388608"]);
}

#[test]
fn a_metadata_size_over_the_limit_is_refused_when_building_a_header() {
    let message = invalid(FlashHeader::new(1 << 40, 0, 4096, FILE_SIZE).unwrap_err());
    names(&message, &["metadata size 1099511627776", "over the limit 8388608"]);
}

#[test]
fn a_tables_size_over_the_limit_is_refused() {
    let message = invalid(FlashHeader::new(0, (1 << 30) + 1, 4096, FILE_SIZE).unwrap_err());
    names(&message, &["tables size 1073741825", "over the limit 1073741824"]);
}

#[test]
fn tables_that_end_beyond_the_file_are_refused() {
    let message = invalid(FlashHeader::new(100, 200, 4096, 200).unwrap_err());
    names(&message, &["tables end at 364", "beyond the file size 200"]);
}

#[test]
fn a_data_offset_that_is_not_aligned_is_refused() {
    let message = invalid(FlashHeader::new(100, 200, 4097, FILE_SIZE).unwrap_err());
    names(&message, &["data section offset 4097", "multiple of 4096"]);
}

#[test]
fn a_data_offset_before_the_end_of_the_tables_is_refused() {
    let message = invalid(FlashHeader::new(5000, 100, 4096, FILE_SIZE * 4).unwrap_err());
    names(&message, &["data section offset 4096", "tables end 5164"]);
}

#[test]
fn a_data_offset_beyond_the_file_is_refused() {
    let message = invalid(FlashHeader::new(100, 200, 12288, FILE_SIZE).unwrap_err());
    names(&message, &["data section offset 12288", "file size 8192"]);
}

#[test]
fn a_file_size_over_the_signed_range_is_refused() {
    let message = invalid(FlashHeader::new(0, 0, 4096, u64::MAX).unwrap_err());
    names(
        &message,
        &["file size 18446744073709551615", "over the limit 9223372036854775807"],
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Metadata region

fn entry(key: &str, kind: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = (key.len() as i32).to_be_bytes().to_vec();
    out.extend_from_slice(key.as_bytes());
    out.push(kind);
    out.extend_from_slice(payload);
    out
}

fn region(count: i32, entries: &[Vec<u8>]) -> Vec<u8> {
    let mut out = count.to_be_bytes().to_vec();
    for entry in entries {
        out.extend_from_slice(entry);
    }
    out
}

fn decode(bytes: &[u8]) -> String {
    invalid(FlashMetadata::decode(bytes).unwrap_err())
}

#[test]
fn a_well_formed_metadata_region_is_decoded() {
    let bytes = region(1, &[entry("a", 0, &7_i64.to_be_bytes())]);
    assert_eq!(
        FlashMetadata::decode(&bytes).unwrap().get("a"),
        Some(&MetadataValue::Int(7))
    );
}

#[test]
fn a_negative_metadata_entry_count_is_refused() {
    names(
        &decode(&region(-1, &[])),
        &["entry count is out of range: -1", "0 to 4096"],
    );
}

#[test]
fn a_metadata_entry_count_over_the_limit_is_refused() {
    names(
        &decode(&region(4097, &[])),
        &["entry count is out of range: 4097", "0 to 4096"],
    );
}

#[test]
fn a_negative_metadata_key_length_is_refused() {
    let mut bytes = region(1, &[]);
    bytes.extend_from_slice(&(-1_i32).to_be_bytes());
    names(&decode(&bytes), &["string length is out of range: -1", "0 to 1048576"]);
}

#[test]
fn a_metadata_key_that_is_not_utf8_is_refused() {
    let mut bytes = region(1, &[]);
    bytes.extend_from_slice(&1_i32.to_be_bytes());
    bytes.push(0xFF);
    names(&decode(&bytes), &["not valid UTF-8"]);
}

#[test]
fn a_repeated_metadata_key_is_refused() {
    let one = entry("a", 0, &1_i64.to_be_bytes());
    names(
        &decode(&region(2, &[one.clone(), one])),
        &["strictly increasing", "'a'"],
    );
}

#[test]
fn a_non_finite_metadata_double_is_refused() {
    let bytes = region(1, &[entry("a", 1, &f64::NAN.to_be_bytes())]);
    names(&decode(&bytes), &["metadata a is not finite"]);
}

#[test]
fn a_metadata_boolean_above_one_is_refused() {
    let bytes = region(1, &[entry("a", 2, &[2])]);
    names(&decode(&bytes), &["metadata a boolean byte is 2", "expected 0 or 1"]);
}

#[test]
fn an_unknown_metadata_type_is_refused() {
    let bytes = region(1, &[entry("a", 9, &[])]);
    names(&decode(&bytes), &["metadata a has unknown type 9"]);
}

#[test]
fn a_metadata_string_over_the_limit_is_refused() {
    let bytes = region(1, &[entry("a", 3, &((1_i32 << 20) + 1).to_be_bytes())]);
    names(
        &decode(&bytes),
        &["'a' string length is out of range: 1048577", "0 to 1048576"],
    );
}

#[test]
fn a_metadata_string_that_is_not_utf8_is_refused() {
    let mut payload = 1_i32.to_be_bytes().to_vec();
    payload.push(0xFF);
    names(
        &decode(&region(1, &[entry("a", 3, &payload)])),
        &["'a' is not valid UTF-8"],
    );
}

#[test]
fn a_negative_metadata_array_length_is_refused() {
    let bytes = region(1, &[entry("a", 4, &(-1_i32).to_be_bytes())]);
    names(&decode(&bytes), &["'a' length is out of range: -1", "0 to 1048576"]);
}

#[test]
fn a_metadata_array_that_ends_early_is_refused() {
    let mut payload = 3_i32.to_be_bytes().to_vec();
    payload.extend_from_slice(&1_i64.to_be_bytes());
    names(
        &decode(&region(1, &[entry("a", 4, &payload)])),
        &["truncated while reading 'a'"],
    );
}

#[test]
fn bytes_after_the_last_metadata_entry_are_refused() {
    let mut bytes = region(1, &[entry("a", 0, &1_i64.to_be_bytes())]);
    bytes.extend_from_slice(&[0, 0, 0]);
    names(&decode(&bytes), &["3 trailing bytes", "1 entries"]);
}

#[test]
fn a_non_finite_metadata_double_is_not_encoded() {
    let mut metadata = FlashMetadata::new();
    metadata.insert("a", MetadataValue::Real(f64::INFINITY));
    names(&invalid(metadata.encode().unwrap_err()), &["metadata a is not finite"]);
}

#[test]
fn too_many_metadata_entries_are_not_encoded() {
    let mut metadata = FlashMetadata::new();
    for index in 0..4097 {
        metadata.insert(format!("k{index:05}"), MetadataValue::Bool(true));
    }
    names(
        &invalid(metadata.encode().unwrap_err()),
        &["4097 entries", "limit 4096"],
    );
}

#[test]
fn a_metadata_string_over_the_limit_is_not_encoded() {
    let mut metadata = FlashMetadata::new();
    metadata.insert("a", MetadataValue::Text("x".repeat((1 << 20) + 1)));
    names(
        &invalid(metadata.encode().unwrap_err()),
        &["'a' string has 1048577 bytes", "limit 1048576"],
    );
}

#[test]
fn a_metadata_array_over_the_limit_is_not_encoded() {
    let mut metadata = FlashMetadata::new();
    metadata.insert("a", MetadataValue::Ints(vec![0; (1 << 20) + 1]));
    names(
        &invalid(metadata.encode().unwrap_err()),
        &["'a' array has 1048577 elements", "limit 1048576"],
    );
}

#[test]
fn a_metadata_region_over_the_size_limit_is_not_encoded() {
    let mut metadata = FlashMetadata::new();
    metadata.insert("a", MetadataValue::Ints(vec![0; 1 << 20]));
    names(
        &invalid(metadata.encode().unwrap_err()),
        &["metadata of 8388622 bytes", "limit 8388608"],
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Configuration, read from metadata

fn refused_metadata(edit: impl FnOnce(&mut FlashMetadata)) -> String {
    let mut metadata = tiny_config().to_metadata();
    edit(&mut metadata);
    invalid(FlashConfig::from_metadata(&metadata).unwrap_err())
}

fn refused_config(edit: impl FnOnce(&mut FlashConfig)) -> String {
    let mut config = tiny_config();
    edit(&mut config);
    invalid(FlashConfig::from_metadata(&config.to_metadata()).unwrap_err())
}

macro_rules! metadata_refusal {
    ($name:ident, $key:expr, $value:expr, $needles:expr) => {
        #[test]
        fn $name() {
            names(
                &refused_metadata(|metadata| metadata.insert($key, $value)),
                &$needles,
            );
        }
    };
}

macro_rules! config_refusal {
    ($name:ident, $edit:expr, $needles:expr) => {
        #[test]
        fn $name() {
            let edit: fn(&mut FlashConfig) = $edit;
            names(&refused_config(edit), &$needles);
        }
    };
}

#[test]
fn the_tiny_configuration_is_read_back() {
    assert_eq!(
        FlashConfig::from_metadata(&tiny_config().to_metadata()).unwrap(),
        tiny_config()
    );
}

metadata_refusal!(
    a_foreign_architecture_is_refused,
    "architecture",
    MetadataValue::Text("qwen3".into()),
    ["architecture is 'qwen3'", "not qwen4_exp"]
);
metadata_refusal!(
    a_negative_count_is_refused,
    "text.vocab_size",
    MetadataValue::Int(-1),
    ["'text.vocab_size' is out of range: -1"]
);
metadata_refusal!(
    a_count_over_the_signed_range_is_refused,
    "text.vocab_size",
    MetadataValue::Int(1 << 40),
    ["'text.vocab_size' is out of range: 1099511627776"]
);
metadata_refusal!(
    a_signed_value_over_the_range_is_refused,
    "text.bos_token_id",
    MetadataValue::Int(1 << 40),
    ["'text.bos_token_id' is out of range: 1099511627776"]
);
metadata_refusal!(
    an_array_element_over_the_signed_range_is_refused,
    "ple.layers",
    MetadataValue::Ints(vec![1 << 40]),
    ["'ple.layers' has an out-of-range element 1099511627776"]
);
metadata_refusal!(
    an_unknown_layer_type_is_refused,
    "text.layer_types",
    MetadataValue::Texts(vec!["bogus".into(), "full_attention".into()]),
    ["unknown layer type 'bogus'"]
);
metadata_refusal!(
    an_unsupported_quantization_format_is_refused,
    "quant.format",
    MetadataValue::Text("int8".into()),
    ["unsupported quantization int8", "block 16", "scales e4m3"]
);
metadata_refusal!(
    an_unsupported_quantization_block_is_refused,
    "quant.block",
    MetadataValue::Int(32),
    ["unsupported quantization nvfp4 block 32"]
);
metadata_refusal!(
    an_unsupported_scale_type_is_refused,
    "quant.scale_dtype",
    MetadataValue::Text("fp32".into()),
    ["scales fp32"]
);

config_refusal!(
    a_configuration_without_layers_is_refused,
    |c| c.text.num_layers = 0,
    ["2 layer types for 0 layers", "1 to 4096"]
);
config_refusal!(
    more_layers_than_the_limit_are_refused,
    |c| {
        c.text.num_layers = 4097;
        c.text.layer_types = vec![FlashLayerType::GatedDeltaNet; 4097];
    },
    ["4097 layer types for 4097 layers", "1 to 4096"]
);
config_refusal!(
    layer_types_that_do_not_match_the_layer_count_are_refused,
    |c| {
        c.text.layer_types.pop();
    },
    ["1 layer types for 2 layers"]
);
config_refusal!(
    a_hidden_size_that_is_not_a_multiple_of_128_is_refused,
    |c| c.text.hidden_size = 130,
    ["hidden size 130", "multiple of 128"]
);
config_refusal!(
    a_zero_hidden_size_is_refused,
    |c| c.text.hidden_size = 0,
    ["hidden size 0", "positive multiple of 128"]
);
config_refusal!(
    a_zero_vocabulary_is_refused,
    |c| c.text.vocab_size = 0,
    ["vocabulary size 0", "positive"]
);
config_refusal!(
    a_zero_context_length_is_refused,
    |c| c.text.max_position_embeddings = 0,
    ["maximum position embeddings 0", "positive"]
);
config_refusal!(
    zero_kv_heads_are_refused_not_divided_by,
    |c| {
        c.attention.num_heads = 0;
        c.attention.num_kv_heads = 0;
    },
    ["attention KV heads 0 must be positive"]
);
config_refusal!(
    attention_heads_that_are_not_a_multiple_of_the_kv_heads_are_refused,
    |c| {
        c.attention.num_heads = 3;
        c.attention.num_kv_heads = 2;
    },
    ["attention heads 3", "multiple of the KV heads 2"]
);
config_refusal!(
    an_unsupported_attention_output_gate_is_refused,
    |c| c.attention.output_gate = "softmax".into(),
    ["output gate softmax", "(sigmoid)"]
);
config_refusal!(
    zero_gdn_key_heads_are_refused_not_divided_by,
    |c| {
        c.gdn.num_key_heads = 0;
        c.gdn.num_value_heads = 0;
    },
    ["GDN key heads 0 must be positive"]
);
config_refusal!(
    gdn_value_heads_that_are_not_a_multiple_of_the_key_heads_are_refused,
    |c| {
        c.gdn.num_key_heads = 2;
        c.gdn.num_value_heads = 3;
    },
    ["GDN value heads 3", "multiple of the key heads 2"]
);
config_refusal!(
    too_many_gdn_key_heads_are_refused,
    |c| {
        c.gdn.num_key_heads = (1 << 20) + 1;
        c.gdn.num_value_heads = (1 << 20) + 1;
    },
    ["GDN key heads 1048577", "limit 1048576"]
);
config_refusal!(
    too_many_gdn_value_heads_are_refused,
    |c| c.gdn.num_value_heads = 1 << 21,
    ["GDN value heads 2097152", "limit 1048576"]
);
config_refusal!(
    a_gdn_key_head_dimension_over_the_limit_is_refused,
    |c| c.gdn.key_head_dim = (1 << 20) + 1,
    ["GDN key head dimension 1048577", "limit 1048576"]
);
config_refusal!(
    a_gdn_value_head_dimension_over_the_limit_is_refused,
    |c| c.gdn.value_head_dim = (1 << 20) + 1,
    ["GDN value head dimension 1048577", "limit 1048576"]
);
config_refusal!(
    an_unsupported_gdn_state_type_is_refused,
    |c| c.gdn.ssm_dtype = "bfloat16".into(),
    ["GDN state type bfloat16", "(float32)"]
);
config_refusal!(
    more_experts_than_the_limit_are_refused,
    |c| c.moe.num_experts = 65537,
    ["expert count 65537", "limit 65536"]
);
config_refusal!(
    zero_experts_per_token_are_refused,
    |c| c.moe.experts_per_token = 0,
    ["experts per token 0", "1 to the expert count 3"]
);
config_refusal!(
    more_experts_per_token_than_experts_are_refused,
    |c| c.moe.experts_per_token = 4,
    ["experts per token 4", "expert count 3"]
);
config_refusal!(
    a_zero_hyper_connection_count_is_refused,
    |c| c.hyper_connection.count = 0,
    ["hyper-connection count 0", "positive"]
);
config_refusal!(
    a_zero_hyper_connection_rank_is_refused,
    |c| c.hyper_connection.lowrank = 0,
    ["hyper-connection low rank 0", "positive"]
);
config_refusal!(
    an_ngram_size_below_two_is_refused,
    |c| c.ngram.size = 1,
    ["n-gram size 1", "at least 2"]
);
config_refusal!(
    zero_ngram_heads_per_ngram_are_refused_not_divided_by,
    |c| c.ngram.heads_per_ngram = 0,
    ["n-gram heads per n-gram 0 must be positive"]
);
config_refusal!(
    zero_ngram_split_parts_are_refused,
    |c| c.ngram.split_parts = 0,
    ["n-gram split parts 0", "1 to 4096"]
);
config_refusal!(
    more_ngram_split_parts_than_the_limit_are_refused,
    |c| c.ngram.split_parts = 4097,
    ["n-gram split parts 4097", "1 to 4096"]
);
config_refusal!(
    zero_ngram_shard_rows_are_refused,
    |c| c.ngram.shard_rows = 0,
    ["n-gram shard rows 0", "positive"]
);
config_refusal!(
    negative_ngram_shard_rows_are_refused,
    |c| c.ngram.shard_rows = -16,
    ["n-gram shard rows -16", "positive"]
);
config_refusal!(
    ngram_head_tables_of_the_wrong_length_are_refused,
    |c| {
        c.ngram.heads_offsets.pop();
    },
    ["1 offsets and 2 vocabulary sizes", "expected 2 each"]
);
config_refusal!(
    ngram_multipliers_of_the_wrong_length_are_refused,
    |c| c.ngram.layer_multipliers.push(7),
    ["3 entries", "n-gram size 2"]
);
config_refusal!(
    a_hidden_size_that_the_ngram_heads_do_not_divide_is_refused,
    |c| {
        c.ngram.heads_per_ngram = 3;
        c.ngram.heads_offsets = vec![0, 10, 22];
        c.ngram.heads_vocab_sizes = vec![10, 12, 5];
    },
    ["hidden size 128", "3 n-gram heads"]
);
config_refusal!(
    no_layers_holding_ngram_tables_are_refused,
    |c| c.ple.layers.clear(),
    ["0 layers hold the n-gram tables", "1 to 64"]
);
config_refusal!(
    too_many_layers_holding_ngram_tables_are_refused,
    |c| c.ple.layers = vec![1; 65],
    ["65 layers hold the n-gram tables", "1 to 64"]
);
config_refusal!(
    an_ngram_layer_past_the_last_layer_is_refused,
    |c| c.ple.layers = vec![2],
    ["n-gram layer 2", "2 text layers"]
);
config_refusal!(
    a_negative_ngram_layer_is_refused,
    |c| c.ple.layers = vec![-1],
    ["n-gram layer -1", "2 text layers"]
);
config_refusal!(
    ngram_heads_that_do_not_start_where_the_previous_ends_are_refused,
    |c| c.ngram.heads_offsets = vec![0, 11],
    ["n-gram head 1 starts at 11", "previous head 10"]
);
config_refusal!(
    an_ngram_head_without_vocabulary_is_refused,
    |c| c.ngram.heads_vocab_sizes = vec![10, 0],
    ["n-gram head 1 vocabulary size 0", "positive"]
);
config_refusal!(
    ngram_heads_whose_sizes_overflow_are_refused,
    |c| {
        c.ngram.heads_offsets = vec![0, i64::MAX];
        c.ngram.heads_vocab_sizes = vec![i64::MAX, 1];
    },
    [
        "n-gram head 1 vocabulary size 1 overflows",
        "running total 9223372036854775807"
    ]
);
config_refusal!(
    ngram_heads_that_address_more_rows_than_the_shards_hold_are_refused,
    |c| c.ngram.heads_vocab_sizes = vec![10, 30],
    ["address 40 rows", "2 shards hold 32"]
);
config_refusal!(
    more_mtp_layers_than_the_limit_are_refused,
    |c| {
        c.mtp.num_layers = 65;
        c.mtp.layer_types = vec![FlashLayerType::GatedDeltaNet; 65];
    },
    ["MTP layer count 65", "limit 64"]
);
config_refusal!(
    mtp_layer_types_that_do_not_match_the_mtp_layer_count_are_refused,
    |c| c.mtp.num_layers = 2,
    ["1 MTP layer types for 2 MTP layers"]
);
config_refusal!(
    a_vision_depth_over_the_limit_is_refused,
    |c| c.vision.depth = 4097,
    ["vision depth 4097", "limit 4096"]
);

#[test]
fn an_inventory_is_not_derived_from_a_configuration_that_does_not_validate() {
    let mut config = tiny_config();
    config.text.layer_types.pop();
    names(
        &invalid(ExpectedInventory::from_config(&config).unwrap_err()),
        &["1 layer types for 2 layers"],
    );
}

#[test]
fn the_helpers_of_a_configuration_that_was_not_validated_do_not_panic() {
    let mut config = tiny_config();
    config.ngram.size = 0;
    config.gdn.num_key_heads = u32::MAX;
    config.gdn.num_value_heads = u32::MAX;
    config.gdn.key_head_dim = u32::MAX;
    config.gdn.value_head_dim = u32::MAX;
    assert_eq!(config.ngram.heads(), 0);
    assert_eq!(config.gdn.conv_channels(), u64::MAX);
    assert_eq!(config.layer_type(config.text.layer_types.len()), None);
    assert_eq!(config.layer_type(0), Some(FlashLayerType::GatedDeltaNet));
}

// ---------------------------------------------------------------------------------------------------------------
// N-gram layout

fn ngram_refusal(rows: u64, k: u64) -> String {
    invalid(NgramLayout::new(rows, k).unwrap_err())
}

#[test]
fn an_ngram_table_without_rows_is_refused() {
    names(&ngram_refusal(0, 16), &["positive multiple of 16", "0 x 16"]);
}

#[test]
fn an_ngram_table_of_zero_width_is_refused() {
    names(&ngram_refusal(4, 0), &["positive multiple of 16", "4 x 0"]);
}

#[test]
fn an_ngram_table_whose_width_is_not_a_multiple_of_16_is_refused() {
    names(&ngram_refusal(4, 24), &["positive multiple of 16", "4 x 24"]);
}

#[test]
fn an_ngram_table_whose_size_overflows_is_refused() {
    names(
        &ngram_refusal(u64::MAX, 16),
        &["size overflows", "18446744073709551615 x 16"],
    );
}

// ---------------------------------------------------------------------------------------------------------------
// Tables, as raw bytes

const BF16: i32 = 0;
const NVFP4: i32 = 7;
const CONTIGUOUS: i32 = 0;
const ROW_SPLIT: i32 = 1;
const COMPRESSED: i32 = 2;
const INTERLEAVED: i32 = 4;
const DATA_START: i64 = 4096;

fn put_name(out: &mut Vec<u8>, length: i32, bytes: &[u8]) {
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(bytes);
}

fn put_shape(out: &mut Vec<u8>, rank: i32, dims: &[i64]) {
    out.extend_from_slice(&rank.to_be_bytes());
    for dim in dims {
        out.extend_from_slice(&dim.to_be_bytes());
    }
}

/// A fixed object as stored, every field open to damage. The default is a valid 4-element BF16 vector.
#[derive(Clone)]
struct RawTensor {
    name: Vec<u8>,
    name_length: i32,
    rank: i32,
    dims: Vec<i64>,
    dtype: i32,
    format: i32,
    layout: i32,
    group: i32,
    offset: i64,
    size: i64,
}

impl RawTensor {
    fn new() -> RawTensor {
        RawTensor {
            name: b"t".to_vec(),
            name_length: 1,
            rank: 1,
            dims: vec![4],
            dtype: BF16,
            format: BF16,
            layout: CONTIGUOUS,
            group: 0,
            offset: DATA_START,
            size: 8,
        }
    }

    fn named(name: &str, offset: i64) -> RawTensor {
        RawTensor {
            name: name.as_bytes().to_vec(),
            name_length: name.len() as i32,
            offset,
            ..RawTensor::new()
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_name(&mut out, self.name_length, &self.name);
        put_shape(&mut out, self.rank, &self.dims);
        for field in [self.dtype, self.format, self.layout, self.group] {
            out.extend_from_slice(&field.to_be_bytes());
        }
        out.extend_from_slice(&self.offset.to_be_bytes());
        out.extend_from_slice(&self.size.to_be_bytes());
        out.extend_from_slice(&0_u32.to_be_bytes());
        out
    }
}

/// One projection of an expert record as stored.
#[derive(Clone)]
struct RawProjection {
    name: Vec<u8>,
    dims: Vec<i64>,
    dtype: i32,
    format: i32,
    layout: i32,
    record_offset: i64,
    size: i64,
}

impl RawProjection {
    /// A valid projection: the first one of the tiny artifact's first bank.
    fn new(name: &str) -> RawProjection {
        let tiny = tiny();
        let projection = &tiny.banks[0].projections()[0];
        RawProjection {
            name: name.as_bytes().to_vec(),
            dims: projection.shape.iter().map(|dim| *dim as i64).collect(),
            dtype: projection.dtype.ordinal(),
            format: projection.format.ordinal(),
            layout: projection.layout.ordinal(),
            record_offset: 0,
            size: projection.bytes as i64,
        }
    }
}

/// An expert bank as stored.
#[derive(Clone)]
struct RawBank {
    name: Vec<u8>,
    group: i32,
    layer: i32,
    experts: i32,
    projection_count: i32,
    projections: Vec<RawProjection>,
    records: Vec<(i64, i64)>,
}

impl RawBank {
    /// A valid bank of two experts holding one projection, its records in the data section.
    fn new() -> RawBank {
        let projection = RawProjection::new("p");
        let size = projection.size;
        RawBank {
            name: b"b".to_vec(),
            group: 1,
            layer: 0,
            experts: 2,
            projection_count: 1,
            projections: vec![projection],
            records: vec![(8192, size), (8192 + (1 << 16), size)],
        }
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        put_name(&mut out, self.name.len() as i32, &self.name);
        out.extend_from_slice(&self.group.to_be_bytes());
        out.extend_from_slice(&self.layer.to_be_bytes());
        out.extend_from_slice(&self.experts.to_be_bytes());
        out.extend_from_slice(&self.projection_count.to_be_bytes());
        for projection in &self.projections {
            put_name(&mut out, projection.name.len() as i32, &projection.name);
            put_shape(&mut out, projection.dims.len() as i32, &projection.dims);
            for field in [projection.dtype, projection.format, projection.layout] {
                out.extend_from_slice(&field.to_be_bytes());
            }
            out.extend_from_slice(&projection.record_offset.to_be_bytes());
            out.extend_from_slice(&projection.size.to_be_bytes());
        }
        for (offset, size) in &self.records {
            out.extend_from_slice(&offset.to_be_bytes());
            out.extend_from_slice(&size.to_be_bytes());
            out.extend_from_slice(&0_u32.to_be_bytes());
        }
        out
    }
}

fn table(tensors: &[Vec<u8>], banks: &[Vec<u8>]) -> Vec<u8> {
    let mut out = (tensors.len() as i32).to_be_bytes().to_vec();
    for tensor in tensors {
        out.extend_from_slice(tensor);
    }
    out.extend_from_slice(&(banks.len() as i32).to_be_bytes());
    for bank in banks {
        out.extend_from_slice(bank);
    }
    out
}

fn parse_table(bytes: &[u8]) -> Result<FlashTables, EdrlError> {
    let header = FlashHeader::new(0, bytes.len() as u64, 4096, 1 << 20).unwrap();
    FlashTables::parse(bytes, &header)
}

fn tensor_refusal(tensor: &RawTensor) -> EdrlError {
    parse_table(&table(&[tensor.encode()], &[])).unwrap_err()
}

fn bank_refusal(bank: &RawBank) -> EdrlError {
    parse_table(&table(&[], &[bank.encode()])).unwrap_err()
}

fn unsupported(error: EdrlError) -> (String, String) {
    match error {
        EdrlError::UnsupportedTensor { name, reason } => (name, reason),
        other => panic!("expected an unsupported tensor, got {other:?}"),
    }
}

#[test]
fn a_valid_raw_table_is_parsed() {
    let tables = parse_table(&table(&[RawTensor::new().encode()], &[RawBank::new().encode()])).unwrap();
    assert_eq!(tables.tensors().len(), 1);
    assert_eq!(tables.bank("b").unwrap().expert_count(), 2);
}

#[test]
fn a_tables_region_of_another_size_than_the_header_declares_is_refused() {
    let bytes = table(&[RawTensor::new().encode()], &[]);
    let header = FlashHeader::new(0, bytes.len() as u64 + 1, 4096, 1 << 20).unwrap();
    assert_eq!(
        FlashTables::parse(&bytes, &header).unwrap_err(),
        EdrlError::Truncated { what: "tables" }
    );
}

#[test]
fn a_negative_tensor_count_is_refused() {
    let mut bytes = table(&[], &[]);
    bytes[0..4].copy_from_slice(&(-1_i32).to_be_bytes());
    names(
        &invalid(parse_table(&bytes).unwrap_err()),
        &["tensor count is out of range: -1", "0 to 1048576"],
    );
}

#[test]
fn a_tensor_count_over_the_limit_is_refused() {
    let mut bytes = table(&[], &[]);
    bytes[0..4].copy_from_slice(&((1_i32 << 20) + 1).to_be_bytes());
    names(
        &invalid(parse_table(&bytes).unwrap_err()),
        &["tensor count is out of range: 1048577", "0 to 1048576"],
    );
}

#[test]
fn an_expert_bank_count_over_the_limit_is_refused() {
    let mut bytes = table(&[], &[]);
    bytes[4..8].copy_from_slice(&((1_i32 << 20) + 1).to_be_bytes());
    names(
        &invalid(parse_table(&bytes).unwrap_err()),
        &["expert bank count is out of range: 1048577", "0 to 1048576"],
    );
}

#[test]
fn two_tensors_with_one_name_are_refused() {
    let bytes = table(
        &[
            RawTensor::named("t", 4096).encode(),
            RawTensor::named("t", 4352).encode(),
        ],
        &[],
    );
    assert_eq!(
        parse_table(&bytes).unwrap_err(),
        EdrlError::DuplicateName { name: "t".into() }
    );
}

#[test]
fn a_bank_named_like_a_tensor_is_refused() {
    let mut bank = RawBank::new();
    bank.name = b"t".to_vec();
    let bytes = table(&[RawTensor::new().encode()], &[bank.encode()]);
    assert_eq!(
        parse_table(&bytes).unwrap_err(),
        EdrlError::DuplicateName { name: "t".into() }
    );
}

#[test]
fn an_empty_tensor_name_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.name_length = 0;
    tensor.name.clear();
    names(
        &invalid(tensor_refusal(&tensor)),
        &["tensor name length is out of range: 0", "1 to 4096"],
    );
}

#[test]
fn a_negative_tensor_name_length_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.name_length = -3;
    names(
        &invalid(tensor_refusal(&tensor)),
        &["tensor name length is out of range: -3", "1 to 4096"],
    );
}

#[test]
fn a_tensor_name_over_the_limit_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.name_length = 4097;
    names(
        &invalid(tensor_refusal(&tensor)),
        &["length is out of range: 4097", "1 to 4096"],
    );
}

#[test]
fn a_tensor_name_that_is_not_utf8_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.name = vec![0xFF];
    names(&invalid(tensor_refusal(&tensor)), &["tensor name is not valid UTF-8"]);
}

#[test]
fn a_negative_tensor_rank_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.rank = -1;
    names(
        &invalid(tensor_refusal(&tensor)),
        &["rank of t is out of range: -1", "0 to 16"],
    );
}

#[test]
fn a_tensor_rank_over_the_limit_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.rank = 17;
    names(
        &invalid(tensor_refusal(&tensor)),
        &["rank of t is out of range: 17", "0 to 16"],
    );
}

#[test]
fn a_zero_dimension_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.dims = vec![0];
    names(&invalid(tensor_refusal(&tensor)), &["t has an empty dimension: 0"]);
}

#[test]
fn a_negative_dimension_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.dims = vec![-4];
    names(&invalid(tensor_refusal(&tensor)), &["t has an empty dimension: -4"]);
}

#[test]
fn a_negative_tensor_offset_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.offset = -256;
    names(&invalid(tensor_refusal(&tensor)), &["t has a negative offset: -256"]);
}

#[test]
fn a_tensor_group_past_the_last_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.group = 14;
    names(
        &invalid(tensor_refusal(&tensor)),
        &["component group", "outside the supported range: 14"],
    );
}

#[test]
fn a_negative_tensor_group_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.group = -1;
    names(&invalid(tensor_refusal(&tensor)), &["outside the supported range: -1"]);
}

#[test]
fn a_tensor_without_payload_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.size = 0;
    names(&invalid(tensor_refusal(&tensor)), &["t has no payload: tensor size 0"]);
}

#[test]
fn a_tensor_of_the_wrong_size_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.size = 9;
    assert_eq!(
        tensor_refusal(&tensor),
        EdrlError::SizeMismatch {
            name: "t".into(),
            bytes: 9
        }
    );
}

#[test]
fn a_compressed_layout_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.format = NVFP4;
    tensor.layout = COMPRESSED;
    let (name, reason) = unsupported(tensor_refusal(&tensor));
    assert_eq!(name, "t");
    names(&reason, &["compressed layout"]);
}

#[test]
fn an_nvfp4_tensor_without_the_row_split_layout_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.format = NVFP4;
    tensor.layout = CONTIGUOUS;
    let (name, reason) = unsupported(tensor_refusal(&tensor));
    assert_eq!(name, "t");
    names(&reason, &["NVFP4 needs the row-split layout"]);
}

#[test]
fn a_plain_tensor_without_the_contiguous_layout_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.layout = ROW_SPLIT;
    let (name, reason) = unsupported(tensor_refusal(&tensor));
    assert_eq!(name, "t");
    names(&reason, &["Bf16 needs the contiguous layout"]);
}

#[test]
fn an_interleaved_table_that_is_not_rank_two_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.format = NVFP4;
    tensor.layout = INTERLEAVED;
    let (_, reason) = unsupported(tensor_refusal(&tensor));
    names(&reason, &["interleaved NVFP4 table is rank 2"]);
}

#[test]
fn an_interleaved_table_that_is_not_nvfp4_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.layout = INTERLEAVED;
    tensor.rank = 2;
    tensor.dims = vec![4, 16];
    let (_, reason) = unsupported(tensor_refusal(&tensor));
    names(&reason, &["interleaved rows hold NVFP4"]);
}

#[test]
fn an_interleaved_table_with_an_unsupported_width_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.format = NVFP4;
    tensor.layout = INTERLEAVED;
    tensor.rank = 2;
    tensor.dims = vec![4, 24];
    let (_, reason) = unsupported(tensor_refusal(&tensor));
    names(&reason, &["4 x 24"]);
}

#[test]
fn an_interleaved_table_of_the_wrong_size_is_refused() {
    let mut tensor = RawTensor::new();
    tensor.format = NVFP4;
    tensor.layout = INTERLEAVED;
    tensor.rank = 2;
    tensor.dims = vec![4, 16];
    tensor.size = NgramLayout::new(4, 16).unwrap().byte_size() as i64 + 1;
    assert!(matches!(tensor_refusal(&tensor), EdrlError::SizeMismatch { bytes, .. } if bytes == tensor.size as u64));
}

#[test]
fn a_tensor_that_is_not_256_byte_aligned_is_refused() {
    let tensor = RawTensor::named("t", DATA_START + 8);
    names(
        &invalid(tensor_refusal(&tensor)),
        &["t at 4104 is not 256-byte aligned"],
    );
}

#[test]
fn a_tensor_that_extends_beyond_the_file_is_refused() {
    let tensor = RawTensor::named("t", (1 << 20) - 256);
    let tensor = RawTensor {
        size: 512,
        dims: vec![256],
        ..tensor
    };
    names(
        &invalid(tensor_refusal(&tensor)),
        &["t at 1048320 extends beyond the file"],
    );
}

#[test]
fn bytes_after_the_last_bank_are_refused() {
    let mut bytes = table(&[RawTensor::new().encode()], &[]);
    bytes.extend_from_slice(&[0; 5]);
    names(&invalid(parse_table(&bytes).unwrap_err()), &["5 trailing bytes"]);
}

#[test]
fn a_bank_without_experts_is_refused() {
    let mut bank = RawBank::new();
    bank.experts = 0;
    bank.records.clear();
    names(
        &invalid(bank_refusal(&bank)),
        &["expert bank b is empty", "0 experts", "1 projections"],
    );
}

#[test]
fn a_bank_without_projections_is_refused() {
    let mut bank = RawBank::new();
    bank.projection_count = 0;
    bank.projections.clear();
    names(
        &invalid(bank_refusal(&bank)),
        &["expert bank b is empty", "2 experts", "0 projections"],
    );
}

#[test]
fn a_bank_with_too_many_experts_is_refused() {
    let mut bank = RawBank::new();
    bank.experts = 65537;
    names(
        &invalid(bank_refusal(&bank)),
        &["expert count is out of range: 65537", "0 to 65536"],
    );
}

#[test]
fn a_bank_with_too_many_projections_is_refused() {
    let mut bank = RawBank::new();
    bank.projection_count = 65;
    names(
        &invalid(bank_refusal(&bank)),
        &["projection count is out of range: 65", "0 to 64"],
    );
}

#[test]
fn a_bank_repeating_a_projection_is_refused() {
    let mut bank = RawBank::new();
    let mut again = RawProjection::new("p");
    again.record_offset = 4096;
    bank.projections.push(again);
    bank.projection_count = 2;
    names(&invalid(bank_refusal(&bank)), &["b repeats projection p"]);
}

#[test]
fn a_projection_of_the_interleaved_layout_is_refused() {
    let mut bank = RawBank::new();
    bank.projections[0].dims = vec![4, 16];
    bank.projections[0].format = NVFP4;
    bank.projections[0].layout = INTERLEAVED;
    bank.projections[0].size = NgramLayout::new(4, 16).unwrap().byte_size() as i64;
    let (name, reason) = unsupported(bank_refusal(&bank));
    assert_eq!(name, "b/p");
    names(&reason, &["row-split layout"]);
}

#[test]
fn a_projection_of_the_wrong_size_is_refused() {
    let mut bank = RawBank::new();
    bank.projections[0].size += 1;
    let size = bank.projections[0].size as u64;
    assert_eq!(
        bank_refusal(&bank),
        EdrlError::SizeMismatch {
            name: "b/p".into(),
            bytes: size
        }
    );
}

#[test]
fn a_projection_not_aligned_in_its_record_is_refused() {
    let mut bank = RawBank::new();
    bank.projections[0].record_offset = 8;
    names(
        &invalid(bank_refusal(&bank)),
        &["b/p is not 256-byte aligned in its record"],
    );
}

#[test]
fn a_projection_at_a_negative_record_offset_is_refused() {
    let mut bank = RawBank::new();
    bank.projections[0].record_offset = -256;
    names(
        &invalid(bank_refusal(&bank)),
        &["b/p is not 256-byte aligned in its record"],
    );
}

#[test]
fn an_expert_record_with_a_negative_offset_is_refused() {
    let mut bank = RawBank::new();
    bank.records[1].0 = -256;
    names(
        &invalid(bank_refusal(&bank)),
        &["expert record of b has a negative offset -256"],
    );
}

#[test]
fn an_expert_record_smaller_than_its_projections_is_refused() {
    let mut bank = RawBank::new();
    bank.records[1].1 -= 1;
    names(
        &invalid(bank_refusal(&bank)),
        &["expert 1 of b", "record of", "smaller than its projections"],
    );
}

#[test]
fn an_expert_record_that_is_not_256_byte_aligned_is_refused() {
    let mut bank = RawBank::new();
    bank.records[0].0 = 8192 + 8;
    names(&invalid(bank_refusal(&bank)), &["b#0 at 8200 is not 256-byte aligned"]);
}

// ---------------------------------------------------------------------------------------------------------------
// Inventory, through opening a tiny artifact

fn tensor_mut<'a>(tiny: &'a mut TinyFlash, name: &str) -> &'a mut FlashTensor {
    tiny.tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("the tiny artifact has no {name}"))
}

/// The parts of a bank, open to edits; [`build`] makes the bank again.
struct Parts {
    name: String,
    group: ComponentGroup,
    layer: i32,
    projections: Vec<ExpertProjection>,
    records: Vec<euhedral_inference_edrl::ExpertRecord>,
}

fn parts(bank: &ExpertBank) -> Parts {
    Parts {
        name: bank.name().to_owned(),
        group: bank.group(),
        layer: bank.layer(),
        projections: bank.projections().to_vec(),
        records: bank.records().to_vec(),
    }
}

fn build(parts: Parts) -> ExpertBank {
    ExpertBank::new(parts.name, parts.group, parts.layer, parts.projections, parts.records).unwrap()
}

const BANK: &str = "text/layers/0/moe/experts";

fn edit_bank(tiny: &mut TinyFlash, name: &str, edit: impl FnOnce(&mut Parts)) {
    let at = tiny.banks.iter().position(|bank| bank.name() == name).unwrap();
    let mut changed = parts(&tiny.banks[at]);
    edit(&mut changed);
    tiny.banks[at] = build(changed);
}

fn bank_problems(tiny: &TinyFlash, name: &str) -> Vec<String> {
    problems(tiny.image().open().unwrap_err())
        .into_iter()
        .map(|problem| match problem {
            InventoryProblem::Bank { name: bank, what } if bank.starts_with(name) => format!("{bank} {what}"),
            other => panic!("expected only problems of {name}, got {other:?}"),
        })
        .collect()
}

#[test]
fn a_tensor_stored_in_another_format_is_refused_by_name() {
    let mut tiny = tiny();
    let router = tensor_mut(&mut tiny, "text/layers/0/moe/router");
    router.format = WeightFormat::Nvfp4;
    router.layout = WeightLayout::RowSplitK128V1;
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert_eq!(
        found,
        vec![InventoryProblem::Storage {
            name: "text/layers/0/moe/router".into(),
            found: (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1),
            expected: (WeightFormat::Bf16, WeightLayout::ContiguousLeV1),
        }]
    );
    names(&found[0].to_string(), &["text/layers/0/moe/router", "Nvfp4", "Bf16"]);
}

#[test]
fn a_tensor_in_another_group_is_refused_by_name() {
    let mut tiny = tiny();
    tensor_mut(&mut tiny, "text/output_head").group = ComponentGroup::Other;
    let found = problems(tiny.image().open().unwrap_err());
    assert_eq!(
        found,
        vec![InventoryProblem::Group {
            name: "text/output_head".into(),
            found: ComponentGroup::Other,
            expected: ComponentGroup::OutputHead,
        }]
    );
    names(&found[0].to_string(), &["text/output_head", "Other", "OutputHead"]);
}

#[test]
fn an_expert_bank_the_configuration_does_not_imply_is_refused_by_name() {
    let mut tiny = tiny();
    let extra = parts(&tiny.banks[0]);
    tiny.banks.push(build(Parts {
        name: "text/surplus/experts".into(),
        ..extra
    }));
    tiny.layout();
    assert_eq!(
        problems(tiny.image().open().unwrap_err()),
        vec![InventoryProblem::ExtraBank {
            name: "text/surplus/experts".into()
        }]
    );
}

#[test]
fn an_expert_bank_in_another_group_is_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.group = ComponentGroup::Mtp);
    names(
        &bank_problems(&tiny, BANK)[0],
        &[BANK, "is in group Mtp", "expected RoutedExpert"],
    );
}

#[test]
fn an_expert_bank_for_another_layer_is_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.layer = 5);
    names(&bank_problems(&tiny, BANK)[0], &[BANK, "is for layer 5", "expected 0"]);
}

#[test]
fn an_expert_bank_with_a_missing_projection_is_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| {
        bank.projections.pop();
    });
    tiny.layout();
    let found = bank_problems(&tiny, BANK);
    assert_eq!(found.len(), 2, "{found:?}");
    names(&found[0], &["has 1 projections", "expected 2"]);
    names(&found[1], &["lacks projection down"]);
}

#[test]
fn an_expert_bank_projection_of_the_wrong_shape_is_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.projections[1].shape = vec![64, 128]);
    tiny.layout();
    names(
        &bank_problems(&tiny, BANK)[0],
        &[&format!("{BANK}/down"), "has shape [64, 128]", "expected [128, 128]"],
    );
}

#[test]
fn an_expert_bank_projection_stored_as_another_format_is_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| {
        bank.projections[1].format = WeightFormat::Bf16;
        bank.projections[1].layout = WeightLayout::ContiguousLeV1;
    });
    tiny.layout();
    names(
        &bank_problems(&tiny, BANK)[0],
        &[&format!("{BANK}/down"), "Bf16/ContiguousLeV1", "Nvfp4/RowSplitK128V1"],
    );
}

#[test]
fn overlapping_projections_in_a_record_are_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.projections[1].record_offset = 0);
    let found = bank_problems(&tiny, BANK);
    assert_eq!(found.len(), 1, "{found:?}");
    names(
        &found[0],
        &[&format!("{BANK}/"), "starts at 0 in the record", "ends at"],
    );
}

#[test]
fn projections_stored_in_either_order_are_accepted() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.projections.reverse());
    tiny.layout();
    let artifact = tiny.image().open().unwrap();
    let names: Vec<&str> = artifact
        .tables()
        .bank(BANK)
        .unwrap()
        .projections()
        .iter()
        .map(|p| p.name.as_str())
        .collect();
    assert_eq!(names, ["down", "gate_up"]);
}

#[test]
fn expert_records_out_of_expert_order_are_refused_by_name() {
    let mut tiny = tiny();
    edit_bank(&mut tiny, BANK, |bank| bank.records.swap(0, 1));
    let found = bank_problems(&tiny, BANK);
    assert_eq!(found.len(), 1, "{found:?}");
    names(&found[0], &[BANK, "expert 1 at", "does not follow expert 0"]);
}

#[test]
fn an_inventory_problem_names_the_object_in_the_error_message() {
    let mut tiny = tiny();
    tiny.tensors.retain(|tensor| tensor.name != "text/output_head");
    tiny.layout();
    let message = tiny.image().open().unwrap_err().to_string();
    names(&message, &["1 problems", "missing object text/output_head"]);
}
