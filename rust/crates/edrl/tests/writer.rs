//! The encoders against the parsers: what is encoded reads back as what went in, and the bytes of the Python-written
//! fixture and of the tiny artifacts are reproduced exactly. The tiny artifacts are also compared with files
//! written by the reference implementation, kept in `fixtures/tiny-artifacts`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::ops::Range;

use euhedral_inference_edrl::tiny::{PROJECTIONS, TinyArtifact};
use euhedral_inference_edrl::{
    ArtifactPlan, EdrlError, FOOTER_BYTES, HEADER_BYTES, Header, Metadata, ModelMetadata, ObjectSpec,
    PAYLOAD_ALIGNMENT, ReadPlan, Region, TOKENIZER_ENTRIES, TensorDataType, TensorEntry, TensorTable, TokenizerFooter,
    TokenizerSection, TokenizerTexts, WeightFormat, WeightLayout, content_end, encode_metadata, encode_table,
    encode_tokenizer_section,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/tiny.edrl");

struct Parsed {
    header: Header,
    metadata: ModelMetadata,
    table: TensorTable,
    section: TokenizerSection,
    content_end: u64,
}

fn range(bytes: &[u8], range: Range<u64>) -> &[u8] {
    &bytes[range.start as usize..range.end as usize]
}

fn parse(file: &[u8]) -> Parsed {
    let file_len = file.len() as u64;
    let probe = ReadPlan::probe(file_len).unwrap();
    let header = Header::parse(range(file, probe.range(Region::Header).unwrap()).try_into().unwrap()).unwrap();
    let footer = TokenizerFooter::parse(
        range(file, probe.range(Region::TokenizerFooter).unwrap())
            .try_into()
            .unwrap(),
    )
    .unwrap()
    .expect("the file has a tokenizer section");
    let content_end = content_end(file_len, Some(&footer)).unwrap();
    let plan = ReadPlan::new(&header, file_len, Some(&footer)).unwrap();
    let metadata = Metadata::parse(range(file, plan.range(Region::Metadata).unwrap())).unwrap();
    let table = TensorTable::parse(range(file, plan.range(Region::Table).unwrap()), &header, content_end).unwrap();
    let section = TokenizerSection::parse(
        range(file, plan.range(Region::TokenizerBody).unwrap()).to_vec(),
        &footer,
    )
    .unwrap();
    Parsed {
        header,
        metadata,
        table,
        section,
        content_end,
    }
}

fn texts(section: &TokenizerSection) -> TokenizerTexts<'_> {
    TokenizerTexts {
        tokenizer: section.text(TOKENIZER_ENTRIES[0]).unwrap(),
        tokenizer_config: section.text(TOKENIZER_ENTRIES[1]).unwrap(),
        chat_template: section.text(TOKENIZER_ENTRIES[2]).unwrap(),
        special_tokens: section.text(TOKENIZER_ENTRIES[3]).unwrap(),
    }
}

fn specs(table: &TensorTable) -> Vec<ObjectSpec> {
    table
        .entries()
        .iter()
        .map(|entry| ObjectSpec {
            name: entry.name().to_owned(),
            shape: entry.shape().to_vec(),
            dtype: entry.dtype(),
            format: entry.format(),
            layout: entry.layout(),
            bytes: entry.bytes(),
        })
        .collect()
}

#[test]
fn every_region_of_the_fixture_encodes_to_its_own_bytes() {
    let parsed = parse(FIXTURE);
    assert_eq!(parsed.header.encode().as_slice(), &FIXTURE[..HEADER_BYTES]);
    assert_eq!(
        encode_metadata(&parsed.metadata).unwrap(),
        range(FIXTURE, parsed.header.metadata_range())
    );
    assert_eq!(
        encode_table(parsed.table.entries()).unwrap(),
        range(FIXTURE, parsed.header.table_range())
    );
    assert_eq!(
        encode_tokenizer_section(&texts(&parsed.section)).unwrap(),
        &FIXTURE[parsed.content_end as usize..]
    );
}

#[test]
fn the_plan_places_the_fixture_as_its_writer_did() {
    let parsed = parse(FIXTURE);
    let metadata_size = parsed.header.metadata_size();
    let plan = ArtifactPlan::new(metadata_size, &specs(&parsed.table), PAYLOAD_ALIGNMENT).unwrap();
    assert_eq!(plan.header(), &parsed.header);
    assert_eq!(plan.entries(), parsed.table.entries());
    assert_eq!(plan.content_end(), parsed.content_end);
    assert_eq!(plan.table().unwrap(), range(FIXTURE, parsed.header.table_range()));
}

#[test]
fn tiny_artifacts_read_back_as_written() {
    for (seed, companions, objects) in [(1, true, 27), (2, false, 12)] {
        let file = TinyArtifact::new(seed, companions).bytes().unwrap();
        let parsed = parse(&file);
        assert_eq!(parsed.table.len(), objects);
        assert_eq!(parsed.header.tensor_count() as usize, parsed.table.len());
        assert_eq!(parsed.header.version(), 2);
        assert_eq!(
            parsed.section.text("special_tokens.json"),
            Some("{\"eos_token_id\":[2]}")
        );
        for entry in parsed.table.entries() {
            assert_eq!(entry.offset() % 256, 0, "{} is aligned", entry.name());
        }
        // Everything between the table and the first payload, and between payloads, is zero.
        // Every byte between the table and the content's end that no payload covers is zero.
        let mut covered = vec![false; parsed.content_end as usize];
        for entry in parsed.table.entries() {
            covered[entry.offset() as usize..(entry.offset() + entry.bytes()) as usize].fill(true);
        }
        for at in parsed.header.data_offset() as usize..parsed.content_end as usize {
            assert!(covered[at] || file[at] == 0, "byte {at} of the gap is not zero");
        }
        assert!(parsed.table.get(PROJECTIONS[0]).is_some());
        assert_eq!(parsed.table.get("text/layers/0/mlp/gate_up.suh").is_some(), companions);
    }
}

#[test]
fn a_tiny_artifact_without_a_projection_lacks_it() {
    let file = TinyArtifact {
        omitted: PROJECTIONS[1],
        ..TinyArtifact::new(2, false)
    }
    .bytes()
    .unwrap();
    let parsed = parse(&file);
    assert!(parsed.table.get(PROJECTIONS[1]).is_none());
    assert!(parsed.table.get(PROJECTIONS[0]).is_some());
}

#[test]
fn tiny_artifacts_are_the_same_every_time_and_differ_by_seed() {
    let first = TinyArtifact::new(1, true).bytes().unwrap();
    assert_eq!(first, TinyArtifact::new(1, true).bytes().unwrap());
    assert_ne!(first, TinyArtifact::new(3, true).bytes().unwrap());
}

#[test]
fn tiny_artifacts_match_the_golden_files() {
    for (name, golden, seed, companions) in [
        (
            "trellis",
            &include_bytes!("fixtures/tiny-artifacts/trellis.edrl")[..],
            1,
            true,
        ),
        (
            "nvfp4",
            &include_bytes!("fixtures/tiny-artifacts/nvfp4.edrl")[..],
            2,
            false,
        ),
    ] {
        let written = TinyArtifact::new(seed, companions).bytes().unwrap();
        assert_eq!(written.len(), golden.len(), "{name}: length");
        let first_difference = written.iter().zip(golden).position(|(a, b)| a != b);
        assert_eq!(first_difference, None, "{name}: first differing byte");
    }
}

fn spec(name: &str, elements: u64) -> ObjectSpec {
    ObjectSpec {
        name: name.to_owned(),
        shape: vec![elements],
        dtype: TensorDataType::Fp32,
        format: WeightFormat::Fp32,
        layout: WeightLayout::ContiguousLeV1,
        bytes: elements * 4,
    }
}

fn entry(name: &str, elements: u64, offset: u64, bytes: u64) -> TensorEntry {
    TensorEntry::new(
        name,
        vec![elements],
        TensorDataType::Fp32,
        WeightFormat::Fp32,
        WeightLayout::ContiguousLeV1,
        offset,
        bytes,
    )
}

#[test]
fn the_table_refuses_what_the_parser_refuses() {
    assert!(matches!(
        encode_table(&[entry("a", 4, 0, 16), entry("a", 4, 16, 16)]),
        Err(EdrlError::DuplicateName { .. })
    ));
    assert!(matches!(
        encode_table(&[entry("a", 4, 0, 16), entry("b", 4, 8, 16)]),
        Err(EdrlError::Overlap { .. })
    ));
    assert!(matches!(
        encode_table(&[entry("a", 4, 0, 12)]),
        Err(EdrlError::SizeMismatch { .. })
    ));
    assert!(matches!(
        encode_table(&[entry("", 4, 0, 16)]),
        Err(EdrlError::Invalid { .. })
    ));
    // Touching payloads are fine.
    assert!(encode_table(&[entry("a", 4, 0, 16), entry("b", 4, 16, 16)]).is_ok());
}

#[test]
fn the_plan_aligns_each_payload_and_ends_at_the_last_byte() {
    let plan = ArtifactPlan::new(10, &[spec("a", 3), spec("b", 5)], 64).unwrap();
    let table_end = plan.header().data_offset();
    assert_eq!(table_end, HEADER_BYTES as u64 + 10 + 2 * (4 + 1 + 4 + 8 + 28) as u64);
    let [a, b] = plan.entries() else { panic!("two entries") };
    assert_eq!(a.offset(), table_end.div_ceil(64) * 64);
    assert_eq!(b.offset(), (a.offset() + 12).div_ceil(64) * 64);
    assert_eq!(plan.content_end(), b.offset() + 20);
    assert!(ArtifactPlan::new(10, &[spec("a", 3)], 0).is_err());
    assert_eq!(
        ArtifactPlan::new(10, &[], 4096).unwrap().content_end(),
        HEADER_BYTES as u64 + 10
    );
}

#[test]
fn the_metadata_refuses_what_the_parser_refuses() {
    let good = parse(FIXTURE).metadata;
    assert!(encode_metadata(&good).is_ok());
    let mut bad = good.clone();
    bad.rope_theta = f64::NAN;
    assert!(encode_metadata(&bad).is_err());
    let mut bad = good.clone();
    bad.layer_types.pop();
    assert!(encode_metadata(&bad).is_err());
    let mut bad = good.clone();
    bad.draft_depth = Some(0);
    assert!(encode_metadata(&bad).is_err());
    let mut bad = good;
    bad.vocab_size = 1_000_001;
    assert!(encode_metadata(&bad).is_err());
}

#[test]
fn the_tokenizer_footer_describes_the_body() {
    let section = encode_tokenizer_section(&TokenizerTexts {
        tokenizer: "é",
        tokenizer_config: "",
        chat_template: "t",
        special_tokens: "{}",
    })
    .unwrap();
    let (body, footer) = section.split_at(section.len() - FOOTER_BYTES);
    let footer = TokenizerFooter::parse(footer.try_into().unwrap()).unwrap().unwrap();
    assert_eq!(footer.body_length(), body.len() as u64);
    let parsed = TokenizerSection::parse(body.to_vec(), &footer).unwrap();
    assert_eq!(parsed.text("tokenizer.json"), Some("é"));
    assert_eq!(parsed.text("tokenizer_config.json"), Some(""));
}

#[test]
fn oversized_names_ranks_strings_and_headers_are_refused() {
    let long_name = "n".repeat((1 << 20) + 1);
    assert!(matches!(
        encode_table(&[entry(&long_name, 4, 0, 16)]),
        Err(EdrlError::Invalid { .. })
    ));
    let deep = TensorEntry::new(
        "deep",
        vec![1; 65],
        TensorDataType::Fp32,
        WeightFormat::Fp32,
        WeightLayout::ContiguousLeV1,
        0,
        4,
    );
    assert!(matches!(encode_table(&[deep]), Err(EdrlError::Invalid { .. })));

    let mut metadata = parse(FIXTURE).metadata;
    metadata.hidden_activation = "s".repeat((1 << 20) + 1);
    assert!(encode_metadata(&metadata).is_err());
}

#[test]
fn the_header_refuses_what_the_parser_refuses() {
    let table_end = HEADER_BYTES as u64 + 10 + 100;
    assert!(Header::new(10, 1, table_end).is_ok());
    assert!(Header::new(8 << 20, 1, (8 << 20) + 1000).is_ok());
    assert!(Header::new((8 << 20) + 1, 1, (8 << 20) + 1000).is_err());
    assert!(Header::new(10, 1_000_001, table_end).is_err());
    assert!(Header::new(10, 1, HEADER_BYTES as u64 + 9).is_err());
    assert!(Header::new(10, 1, u64::MAX).is_err());
    assert!(Header::new(u64::MAX, 1, u64::MAX).is_err());
    let header = Header::new(10, 1, table_end).unwrap();
    assert_eq!(Header::parse(&header.encode()).unwrap(), header);
}
