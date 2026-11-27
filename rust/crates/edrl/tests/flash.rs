//! Reads tiny Flash-Next artifacts built in memory: a round trip, then one refusal per check, each naming the object
//! or key at fault. No artifact or GPU is needed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_edrl::tiny_flash::{TinyFlash, tiny_config};
use euhedral_inference_edrl::{
    ComponentGroup, DETECT_BYTES, EdrlError, ExpectedInventory, FLASH_HEADER_BYTES, FlashMetadata, FlashTensor,
    InventoryProblem, MetadataValue, ModelFamily, NgramLayout, TensorDataType, WeightFormat,
};

const MAGIC: u32 = 0x5157_454E;

fn tiny() -> TinyFlash {
    TinyFlash::new(tiny_config())
}

fn problems(error: EdrlError) -> Vec<InventoryProblem> {
    match error {
        EdrlError::Inventory { problems } => problems,
        other => panic!("expected an inventory refusal, got {other:?}"),
    }
}

fn invalid(error: EdrlError) -> String {
    match error {
        EdrlError::Invalid { what } => what,
        other => panic!("expected an invalid region, got {other:?}"),
    }
}

fn tensor_mut<'a>(tiny: &'a mut TinyFlash, name: &str) -> &'a mut FlashTensor {
    tiny.tensors
        .iter_mut()
        .find(|tensor| tensor.name == name)
        .unwrap_or_else(|| panic!("the tiny artifact has no {name}"))
}

#[test]
fn a_tiny_artifact_round_trips_through_open() {
    let tiny = tiny();
    let artifact = tiny.image().open().unwrap();
    assert_eq!(artifact.config(), &tiny.config);
    assert_eq!(artifact.tables().tensors(), tiny.tensors.as_slice());
    assert_eq!(artifact.tables().banks(), tiny.banks.as_slice());
    let report = artifact.report();
    assert_eq!(report.tensors, tiny.tensors.len());
    assert_eq!(report.banks, 3, "two text layers and one MTP layer");
    assert_eq!(report.expert_records, 3 * tiny.config.moe.num_experts as usize);
    let tokenizer = artifact.tokenizer().expect("the tiny file carries a tokenizer");
    assert_eq!(tokenizer.file("chat_template.jinja"), Some(b"template".as_slice()));
    assert_eq!(artifact.header().file_size(), tiny.content_end);
}

#[test]
fn an_artifact_without_a_tokenizer_section_opens_without_one() {
    let mut tiny = tiny();
    tiny.tokenizer = false;
    assert!(tiny.image().open().unwrap().tokenizer().is_none());
}

#[test]
fn the_tiny_inventory_holds_every_object_kind() {
    let tiny = tiny();
    let inventory = ExpectedInventory::from_config(&tiny.config).unwrap();
    for name in [
        "text/token_embedding",
        "text/layers/0/gdn/in_proj_qkv",
        "text/layers/1/attention/q_proj",
        "text/layers/1/ple/ngram/shard_001",
        "mtp/fc_hidden",
        "vision/blocks/0/attn/qkv/weight",
    ] {
        assert!(inventory.tensors().contains_key(name), "the inventory lacks {name}");
    }
    let shard = &inventory.tensors()["text/layers/1/ple/ngram/shard_000"];
    assert_eq!(shard.group, ComponentGroup::Ngram);
    let layout = NgramLayout::new(shard.shape[0], shard.shape[1]).unwrap();
    let stored = tiny
        .tensors
        .iter()
        .find(|t| t.name == "text/layers/1/ple/ngram/shard_000")
        .unwrap();
    assert_eq!(stored.bytes, layout.byte_size());
}

#[test]
fn the_family_is_read_from_the_first_twelve_bytes() {
    let head = |version: u32, architecture: u32| -> [u8; DETECT_BYTES] {
        let mut out = [0; DETECT_BYTES];
        out[0..4].copy_from_slice(&MAGIC.to_be_bytes());
        out[4..8].copy_from_slice(&version.to_be_bytes());
        out[8..12].copy_from_slice(&architecture.to_be_bytes());
        out
    };
    assert_eq!(ModelFamily::detect(&head(2, 0)), Ok(ModelFamily::DenseV2));
    assert_eq!(ModelFamily::detect(&head(3, 1)), Ok(ModelFamily::FlashNextV3));
    assert_eq!(
        ModelFamily::detect(&head(3, 7)),
        Err(EdrlError::UnsupportedArchitecture { found: 7 })
    );
    assert_eq!(
        ModelFamily::detect(&head(4, 1)),
        Err(EdrlError::UnsupportedVersion { found: 4 })
    );
    let mut wrong = head(3, 1);
    wrong[0] = 0;
    assert!(matches!(ModelFamily::detect(&wrong), Err(EdrlError::BadMagic { .. })));
    let image = tiny().image();
    let first: [u8; DETECT_BYTES] = image.bytes[..DETECT_BYTES].try_into().unwrap();
    assert_eq!(ModelFamily::detect(&first), Ok(ModelFamily::FlashNextV3));
}

#[test]
fn a_bad_magic_is_refused() {
    let mut image = tiny().image();
    image.bytes[0] ^= 0xFF;
    assert!(matches!(image.open().unwrap_err(), EdrlError::BadMagic { .. }));
}

#[test]
fn a_dense_version_is_refused_by_the_flash_reader() {
    let mut image = tiny().image();
    image.bytes[7] = 2;
    assert_eq!(image.open().unwrap_err(), EdrlError::UnsupportedVersion { found: 2 });
}

#[test]
fn an_unknown_architecture_is_refused() {
    let mut image = tiny().image();
    image.bytes[11] = 9;
    assert_eq!(
        image.open().unwrap_err(),
        EdrlError::UnsupportedArchitecture { found: 9 }
    );
}

#[test]
fn a_file_shorter_than_the_header_is_refused() {
    let mut image = tiny().image();
    image.bytes.truncate(FLASH_HEADER_BYTES - 1);
    assert!(invalid(image.open().unwrap_err()).contains("header"));
}

#[test]
fn an_object_beyond_the_content_end_is_refused_by_name() {
    let mut tiny = tiny();
    tiny.layout();
    let end = tiny.content_end;
    tensor_mut(&mut tiny, "text/output_head").offset = end;
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(
        message.contains("text/output_head") && message.contains("beyond the file"),
        "{message}"
    );
}

#[test]
fn overlapping_objects_are_refused_by_name() {
    let mut tiny = tiny();
    let first = tiny.tensors[0].clone();
    tiny.tensors[1].offset = first.offset;
    match tiny.image().open().unwrap_err() {
        EdrlError::Overlap { first: a, second: b } => {
            let names = [tiny.tensors[0].name.clone(), tiny.tensors[1].name.clone()];
            assert!(names.contains(&a) && names.contains(&b), "{a} and {b}");
        }
        other => panic!("expected an overlap, got {other:?}"),
    }
}

#[test]
fn an_object_before_the_data_section_is_refused_by_name() {
    let mut tiny = tiny();
    tiny.layout();
    tensor_mut(&mut tiny, "text/output_head").offset = 256;
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(
        message.contains("text/output_head") && message.contains("before the data section"),
        "{message}"
    );
}

#[test]
fn an_expert_record_beyond_the_content_end_is_refused_by_name() {
    let mut tiny = tiny();
    let mut bank = tiny.banks.remove(0);
    let name = bank.name().to_owned();
    let mut records = bank.records().to_vec();
    records[1].offset = tiny.content_end + 4096;
    let projections = bank.projections().to_vec();
    bank = euhedral_inference_edrl::ExpertBank::new(name.clone(), bank.group(), bank.layer(), projections, records)
        .unwrap();
    tiny.banks.insert(0, bank);
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(message.contains(&format!("{name}#1")), "{message}");
}

#[test]
fn a_wrong_shape_is_refused_by_name() {
    let mut tiny = tiny();
    tensor_mut(&mut tiny, "text/token_embedding").shape = vec![31, 128];
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert_eq!(
        found,
        vec![InventoryProblem::Shape {
            name: "text/token_embedding".into(),
            found: vec![31, 128],
            expected: vec![32, 128],
        }]
    );
}

#[test]
fn a_wrong_data_type_is_refused_by_name() {
    let mut tiny = tiny();
    let tensor = tensor_mut(&mut tiny, "text/output_head");
    tensor.dtype = TensorDataType::Fp16;
    tensor.format = WeightFormat::Fp16;
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert!(
        found.iter().any(|problem| matches!(
            problem,
            InventoryProblem::DataType { name, found: TensorDataType::Fp16, .. } if name == "text/output_head"
        )),
        "{found:?}"
    );
}

#[test]
fn a_missing_object_is_refused_by_name() {
    let mut tiny = tiny();
    tiny.tensors.retain(|tensor| tensor.name != "text/layers/0/moe/router");
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert_eq!(
        found,
        vec![InventoryProblem::Missing {
            name: "text/layers/0/moe/router".into()
        }]
    );
}

#[test]
fn an_extra_object_is_refused_by_name() {
    let mut tiny = tiny();
    let mut extra = tiny.tensors[0].clone();
    extra.name = "text/surplus".into();
    tiny.tensors.push(extra);
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert_eq!(
        found,
        vec![InventoryProblem::Extra {
            name: "text/surplus".into()
        }]
    );
}

#[test]
fn a_missing_expert_bank_is_refused_by_name() {
    let mut tiny = tiny();
    let gone = tiny.banks.remove(1).name().to_owned();
    tiny.layout();
    assert_eq!(
        problems(tiny.image().open().unwrap_err()),
        vec![InventoryProblem::MissingBank { name: gone }]
    );
}

#[test]
fn an_expert_bank_with_the_wrong_expert_count_is_refused_by_name() {
    let mut tiny = tiny();
    let bank = tiny.banks.remove(0);
    let records = bank.records()[..2].to_vec();
    let name = bank.name().to_owned();
    let short = euhedral_inference_edrl::ExpertBank::new(
        name.clone(),
        bank.group(),
        bank.layer(),
        bank.projections().to_vec(),
        records,
    )
    .unwrap();
    tiny.banks.insert(0, short);
    tiny.layout();
    let found = problems(tiny.image().open().unwrap_err());
    assert!(
        found
            .iter()
            .any(|problem| matches!(problem, InventoryProblem::Bank { name: n, .. } if *n == name)),
        "{found:?}"
    );
}

#[test]
fn a_damaged_metadata_type_is_refused_by_key() {
    let mut tiny = tiny();
    let key = "text.hidden_size";
    assert!(matches!(tiny.metadata.get(key), Some(MetadataValue::Int(_))));
    tiny.metadata.insert(key, MetadataValue::Text("128".into()));
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(message.contains(key), "{message}");
}

#[test]
fn an_unknown_metadata_key_is_refused_by_name() {
    let mut tiny = tiny();
    tiny.metadata.insert("zzz.unknown", MetadataValue::Int(1));
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(message.contains("zzz.unknown"), "{message}");
}

#[test]
fn a_missing_metadata_key_is_refused_by_name() {
    let mut tiny = tiny();
    tiny.metadata.remove("moe.num_experts");
    let message = invalid(tiny.image().open().unwrap_err());
    assert!(message.contains("moe.num_experts"), "{message}");
}

#[test]
fn metadata_keys_out_of_order_are_refused() {
    let mut metadata = FlashMetadata::new();
    metadata.insert("a", MetadataValue::Int(1));
    metadata.insert("b", MetadataValue::Int(2));
    let mut bytes = metadata.encode().unwrap();
    // Entry layout: count, then (key length, key, type, value) per entry. Turn key "b" into "A" (before "a").
    let at = bytes.iter().rposition(|byte| *byte == b'b').unwrap();
    bytes[at] = b'A';
    assert!(invalid(FlashMetadata::decode(&bytes).unwrap_err()).contains("increasing"));
}

#[test]
fn an_unknown_version_is_refused_with_a_message_naming_both_readers() {
    let mut image = tiny().image();
    image.bytes[7] = 4;
    let message = image.open().unwrap_err().to_string();
    assert!(
        message.contains("version 4") && message.contains("version 3"),
        "{message}"
    );
}

#[test]
fn a_tokenizer_section_with_a_damaged_body_is_refused() {
    let mut image = tiny().image();
    let flip = image.bytes.len() - 24 - 3;
    image.bytes[flip] ^= 0xFF;
    assert_eq!(image.open().unwrap_err(), EdrlError::TokenizerCrc);
}

#[test]
fn a_truncated_tokenizer_section_is_refused() {
    let mut image = tiny().image();
    let cut = image.bytes.len() - 24 - 5;
    image.bytes.drain(cut..cut + 5);
    let message = invalid(image.open().unwrap_err());
    assert!(
        message.contains("header declares") && message.contains("content holds"),
        "{message}"
    );
}

#[test]
fn a_file_longer_than_its_content_without_a_section_is_refused() {
    let mut tiny = tiny();
    tiny.tokenizer = false;
    let mut image = tiny.image();
    image.bytes.extend_from_slice(&[0; 512]);
    let message = invalid(image.open().unwrap_err());
    assert!(
        message.contains("header declares") && message.contains("content holds"),
        "{message}"
    );
}
