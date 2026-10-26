//! Parses `fixtures/tiny.edrl`, written by `fixtures/make_fixture.py` with the converter's own writers, and
//! variations of it. No artifact or GPU is needed.

use euhedral_inference_edrl::{
    EdrlError, FOOTER_BYTES, HEADER_BYTES, Header, LayerType, Metadata, ModelMetadata, ReadPlan, Region,
    TensorDataType, TensorEntry, TensorTable, TokenizerFooter, TokenizerSection, TrellisCompanions, WeightFormat,
    WeightLayout, content_end,
};

const FIXTURE: &[u8] = include_bytes!("fixtures/tiny.edrl");

struct Parsed {
    header: Header,
    metadata: ModelMetadata,
    table: TensorTable,
    footer: Option<TokenizerFooter>,
    content_end: u64,
}

fn range(bytes: &[u8], range: std::ops::Range<u64>) -> &[u8] {
    &bytes[range.start as usize..range.end as usize]
}

/// Parses `file` the way a loader does: header and footer first, then the regions they locate.
fn parse(file: &[u8]) -> Result<Parsed, EdrlError> {
    let file_len = file.len() as u64;
    let probe = ReadPlan::probe(file_len)?;
    let header = Header::parse(range(file, probe.range(Region::Header).unwrap()).try_into().unwrap())?;
    let footer = TokenizerFooter::parse(
        range(file, probe.range(Region::TokenizerFooter).unwrap())
            .try_into()
            .unwrap(),
    )?;
    let content_end = content_end(file_len, footer.as_ref())?;
    let plan = ReadPlan::new(&header, file_len, footer.as_ref())?;
    let metadata = Metadata::parse(range(file, plan.range(Region::Metadata).unwrap()))?;
    let table = TensorTable::parse(range(file, plan.range(Region::Table).unwrap()), &header, content_end)?;
    Ok(Parsed {
        header,
        metadata,
        table,
        footer,
        content_end,
    })
}

fn section(file: &[u8], parsed: &Parsed) -> Result<TokenizerSection, EdrlError> {
    let footer = parsed.footer.as_ref().expect("the file has a tokenizer section");
    TokenizerSection::parse(range(file, footer.body_range(file.len() as u64)?).to_vec(), footer)
}

fn entry<'a>(parsed: &'a Parsed, name: &str) -> &'a TensorEntry {
    parsed.table.get(name).unwrap_or_else(|| panic!("no tensor {name}"))
}

/// The position of the table entry `name`, found by its length-prefixed name.
fn entry_position(file: &[u8], name: &str) -> usize {
    let mut needle = (name.len() as u32).to_be_bytes().to_vec();
    needle.extend_from_slice(name.as_bytes());
    let hits: Vec<usize> = file
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "{name} must appear once");
    hits[0]
}

/// The position of the `offset` field of the table entry `name` (after name, rank, shape, dtype, format, layout).
fn offset_field(file: &[u8], name: &str, rank: usize) -> usize {
    entry_position(file, name) + 4 + name.len() + 4 + 8 * rank + 12
}

fn put_i64(file: &mut [u8], at: usize, value: i64) {
    file[at..at + 8].copy_from_slice(&value.to_be_bytes());
}

#[test]
fn parses_header_metadata_table() {
    let parsed = parse(FIXTURE).unwrap();

    let header = &parsed.header;
    assert_eq!(header.version(), 2);
    assert_eq!(header.metadata_offset(), HEADER_BYTES as u64);
    assert_eq!(header.tensor_count(), 11);
    assert_eq!(header.metadata_range().end, header.tensor_table_offset());
    assert_eq!(header.table_range().end, header.data_offset());

    let metadata = &parsed.metadata;
    assert_eq!(metadata.vocab_size, 1000);
    assert_eq!(metadata.hidden_size, 256);
    assert_eq!(metadata.num_hidden_layers, 4);
    assert_eq!(metadata.num_attention_heads, 8);
    assert_eq!(metadata.num_key_value_heads, 2);
    assert_eq!(metadata.attention_head_dim, 32);
    assert_eq!(metadata.intermediate_size, 512);
    assert_eq!(metadata.linear_conv_kernel_dim, 4);
    assert_eq!(metadata.rms_norm_epsilon, 1e-6);
    assert_eq!(metadata.rope_theta, 1e7);
    assert_eq!(metadata.partial_rotary_factor, 0.25);
    assert_eq!(metadata.max_position_embeddings, 4096);
    assert_eq!(metadata.hidden_activation, "silu");
    assert_eq!(
        metadata.layer_types,
        [
            LayerType::GatedDeltaNet,
            LayerType::GatedDeltaNet,
            LayerType::GatedDeltaNet,
            LayerType::FullAttention
        ]
    );
    assert!(!metadata.tie_word_embeddings);
    assert!(metadata.attention_output_gate);
    assert_eq!(metadata.mtp_layer_count, 1);
    assert_eq!(metadata.draft_depth, Some(4));

    let table = &parsed.table;
    let names: Vec<&str> = table.entries().iter().map(|entry| entry.name()).collect();
    assert_eq!(
        names,
        [
            "embed",
            "norm",
            "ids",
            "q3",
            "nvfp4",
            "nvfp4.sd4",
            "p2e2",
            "trellis",
            "trellis.suh",
            "trellis.svh",
            "trellis.segments"
        ]
    );
    let expected = [
        (
            "embed",
            vec![4, 8],
            TensorDataType::Bf16,
            WeightFormat::Bf16,
            WeightLayout::ContiguousLeV1,
            64,
        ),
        (
            "norm",
            vec![8],
            TensorDataType::Bf16,
            WeightFormat::Fp32,
            WeightLayout::ContiguousLeV1,
            32,
        ),
        (
            "ids",
            vec![3],
            TensorDataType::Int32,
            WeightFormat::I32,
            WeightLayout::ContiguousLeV1,
            12,
        ),
        (
            "q3",
            vec![2, 128],
            TensorDataType::Bf16,
            WeightFormat::Q3G64Fp16,
            WeightLayout::RowSplitK128V1,
            264,
        ),
        (
            "nvfp4",
            vec![2, 128],
            TensorDataType::Bf16,
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128V1,
            516,
        ),
        (
            "nvfp4.sd4",
            vec![2, 128],
            TensorDataType::Bf16,
            WeightFormat::Nvfp4,
            WeightLayout::RowSplitK128Sd4V1,
            532,
        ),
        (
            "p2e2",
            vec![2, 1024],
            TensorDataType::Bf16,
            WeightFormat::Q3G64Fp16,
            WeightLayout::RowSplitP2e2V1,
            1424,
        ),
        (
            "trellis",
            vec![8, 16, 24],
            TensorDataType::Bf16,
            WeightFormat::Trellis,
            WeightLayout::TrellisTile16V1,
            6144,
        ),
        (
            "trellis.suh",
            vec![2, 128],
            TensorDataType::Fp16,
            WeightFormat::Fp16,
            WeightLayout::ContiguousLeV1,
            512,
        ),
        (
            "trellis.svh",
            vec![256],
            TensorDataType::Fp16,
            WeightFormat::Fp16,
            WeightLayout::ContiguousLeV1,
            512,
        ),
        (
            "trellis.segments",
            vec![2],
            TensorDataType::Int32,
            WeightFormat::I32,
            WeightLayout::ContiguousLeV1,
            8,
        ),
    ];
    for (index, (name, shape, dtype, format, layout, bytes)) in expected.into_iter().enumerate() {
        let entry = &table.entries()[index];
        assert_eq!(entry.name(), name);
        assert_eq!(entry.shape(), shape, "{name}");
        assert_eq!(entry.dtype(), dtype, "{name}");
        assert_eq!(entry.format(), format, "{name}");
        assert_eq!(entry.layout(), layout, "{name}");
        assert_eq!(entry.bytes(), bytes, "{name}");
        assert_eq!(entry.offset() % 4096, 0, "{name} is block-aligned");
        assert!(entry.offset() >= header.data_offset(), "{name}");
        // The payload is where the entry says: the converter filled each with its 1-based index.
        let payload = range(FIXTURE, entry.range());
        if name != "trellis.segments" {
            assert!(payload.iter().all(|byte| *byte as usize == index + 1), "{name}");
        }
    }
    assert_eq!(table.len(), 11);
    assert!(table.get("absent").is_none());
    assert_eq!(
        parsed.content_end,
        FIXTURE.len() as u64 - parsed.footer.unwrap().body_length() - FOOTER_BYTES as u64
    );

    let section = section(FIXTURE, &parsed).unwrap();
    assert_eq!(section.text("tokenizer.json"), Some("{\"version\":\"1.0\"}"));
    assert_eq!(
        section.file("tokenizer_config.json"),
        Some(&b"{\"chat_template\":null}"[..])
    );
    assert_eq!(
        section.text("chat_template.jinja"),
        Some("{{ messages[0].content }} \u{e9}")
    );
    assert_eq!(
        section.text("special_tokens.json"),
        Some("{\"bos_token_id\":1,\"eos_token_id\":2,\"pad_token_id\":3}")
    );
    assert!(section.file("vocab.json").is_none());
}

#[test]
fn rejects_bad_magic() {
    let mut file = FIXTURE.to_vec();
    file[0] ^= 0xff;
    assert!(matches!(parse(&file), Err(EdrlError::BadMagic { .. })));

    // A file that is not an artifact at all reports the magic, not a later offset check.
    let text = [b'A'; HEADER_BYTES];
    assert_eq!(
        Header::parse(&text).err(),
        Some(EdrlError::BadMagic { found: 0x4141_4141 })
    );

    let mut file = FIXTURE.to_vec();
    file[7] = 3;
    assert_eq!(parse(&file).err(), Some(EdrlError::UnsupportedVersion { found: 3 }));

    let mut file = FIXTURE.to_vec();
    file[39] = 1; // the reserved field
    assert_eq!(
        parse(&file).err(),
        Some(EdrlError::Invalid {
            what: "header reserved field must be zero".into()
        })
    );

    assert!(matches!(
        ReadPlan::probe(HEADER_BYTES as u64 - 1),
        Err(EdrlError::Invalid { .. })
    ));
}

#[test]
fn rejects_overlapping_payloads() {
    let mut file = FIXTURE.to_vec();
    let embed = entry(&parse(&file).unwrap(), "embed").offset();
    // `ids` now starts inside `embed`.
    let at = offset_field(&file, "ids", 1);
    put_i64(&mut file, at, embed as i64 + 32);
    assert_eq!(
        parse(&file).err(),
        Some(EdrlError::Overlap {
            first: "embed".into(),
            second: "ids".into()
        })
    );

    // Touching payloads are not overlapping: `ids` begins where `embed` ends.
    put_i64(&mut file, at, embed as i64 + 64);
    assert!(parse(&file).is_ok());
}

#[test]
fn rejects_payloads_outside_the_content() {
    let parsed = parse(FIXTURE).unwrap();
    let last = parsed.table.entries().last().unwrap();
    let table_bytes = {
        let range = parsed.header.table_range();
        &FIXTURE[range.start as usize..range.end as usize]
    };
    // The same table against a file that ends inside the last payload.
    let short = last.offset() + last.bytes() - 1;
    assert!(matches!(
        TensorTable::parse(table_bytes, &parsed.header, short),
        Err(EdrlError::Invalid { .. })
    ));

    // A payload may not start in the table.
    let mut file = FIXTURE.to_vec();
    let at = offset_field(&file, "ids", 1);
    put_i64(&mut file, at, parsed.header.data_offset() as i64 - 4);
    assert!(matches!(parse(&file), Err(EdrlError::Invalid { .. })));

    // A table with a byte missing.
    assert!(matches!(
        TensorTable::parse(&table_bytes[1..], &parsed.header, parsed.content_end),
        Err(EdrlError::Invalid { .. })
    ));
}

#[test]
fn rejects_sizes_the_layout_does_not_allow() {
    let parsed = parse(FIXTURE).unwrap();
    for (name, rank, size) in [
        ("q3", 2, 265),
        ("nvfp4", 2, 517),
        ("nvfp4.sd4", 2, 516),
        ("trellis", 3, 6142),
        ("embed", 2, 62),
    ] {
        let mut file = FIXTURE.to_vec();
        // The byte size is the last field of the entry.
        let at = offset_field(&file, name, rank) + 8;
        put_i64(&mut file, at, size);
        assert!(
            matches!(parse(&file), Err(EdrlError::SizeMismatch { name: ref n, bytes }) if n == name && bytes == size as u64),
            "{name}"
        );
    }

    // P2E2 sizes are bounded, not exact: the payload is 4-byte words between empty and full.
    let p2e2 = entry(&parsed, "p2e2");
    let at = offset_field(FIXTURE, "p2e2", 2) + 8;
    let minimum = 1024 + 4 * 80;
    let maximum = minimum + 4 * (2 * 1024 / 16);
    for (size, accepted) in [
        (minimum - 4, false),
        (minimum, true),
        (minimum + 2, false),
        (maximum, true),
        (maximum + 4, false),
    ] {
        let mut file = FIXTURE.to_vec();
        put_i64(&mut file, at, size);
        // Keep the payload inside the content: it only ever shrinks or grows by a few hundred bytes of padding.
        let result = parse(&file);
        assert_eq!(
            result.is_ok(),
            accepted,
            "p2e2 of {size} bytes (stored {})",
            p2e2.bytes()
        );
    }
}

#[test]
fn rejects_formats_without_a_layout() {
    // Q3 values must be row-split; a contiguous Q3 object has no registered format.
    let mut file = FIXTURE.to_vec();
    let at = offset_field(&file, "embed", 2) - 8; // format field: three ordinals precede the offset
    file[at + 3] = 4; // WeightFormat::Q3G64Fp16
    assert!(matches!(parse(&file), Err(EdrlError::UnsupportedTensor { .. })));

    // Out-of-range ordinals.
    let mut file = FIXTURE.to_vec();
    let at = offset_field(&file, "embed", 2) - 12;
    file[at + 3] = 9; // TensorDataType
    assert!(matches!(parse(&file), Err(EdrlError::Invalid { .. })));
}

#[test]
fn rejects_bad_tokenizer_crc() {
    let parsed = parse(FIXTURE).unwrap();
    let footer = parsed.footer.unwrap();
    let body = footer.body_range(FIXTURE.len() as u64).unwrap();
    let mut file = FIXTURE.to_vec();
    file[body.start as usize + 20] ^= 1;
    let damaged = parse(&file).unwrap();
    assert_eq!(section(&file, &damaged).err(), Some(EdrlError::TokenizerCrc));

    // A body of the wrong length is refused before its CRC is looked at.
    assert!(matches!(
        TokenizerSection::parse(file[body.start as usize..body.end as usize - 1].to_vec(), &footer),
        Err(EdrlError::Truncated { .. })
    ));

    // A footer of another version is refused.
    let mut tail: [u8; FOOTER_BYTES] = FIXTURE[FIXTURE.len() - FOOTER_BYTES..].try_into().unwrap();
    tail[15] = 2;
    assert_eq!(
        TokenizerFooter::parse(&tail).err(),
        Some(EdrlError::TokenizerVersion { found: 2 })
    );

    // A body longer than the file is refused when the content end is computed.
    let mut tail: [u8; FOOTER_BYTES] = FIXTURE[FIXTURE.len() - FOOTER_BYTES..].try_into().unwrap();
    tail[..8].copy_from_slice(&(FIXTURE.len() as i64).to_be_bytes());
    let long = TokenizerFooter::parse(&tail).unwrap().unwrap();
    assert!(matches!(
        content_end(FIXTURE.len() as u64, Some(&long)),
        Err(EdrlError::Invalid { .. })
    ));
    tail[..8].copy_from_slice(&(-1i64).to_be_bytes());
    assert!(matches!(TokenizerFooter::parse(&tail), Err(EdrlError::Invalid { .. })));
}

#[test]
fn rejects_a_tokenizer_body_that_is_not_the_four_entries() {
    let parsed = parse(FIXTURE).unwrap();
    let footer = parsed.footer.unwrap();
    let body_range = footer.body_range(FIXTURE.len() as u64).unwrap();
    let body = range(FIXTURE, body_range).to_vec();

    // Re-seal a damaged body with a matching CRC: the structure checks must catch it.
    let reseal = |body: &[u8]| {
        let mut tail = [0u8; FOOTER_BYTES];
        tail[..8].copy_from_slice(&(body.len() as i64).to_be_bytes());
        tail[8..12].copy_from_slice(&crc32fast::hash(body).to_be_bytes());
        tail[12..16].copy_from_slice(&1u32.to_be_bytes());
        tail[16..].copy_from_slice(b"EDRLTOKN");
        TokenizerFooter::parse(&tail).unwrap().unwrap()
    };
    let mut renamed = body.clone();
    renamed[8] = b'T'; // "Tokenizer.json"
    assert!(matches!(
        TokenizerSection::parse(renamed.clone(), &reseal(&renamed)),
        Err(EdrlError::Invalid { .. })
    ));
    let mut trailing = body.clone();
    trailing.push(0);
    assert!(matches!(
        TokenizerSection::parse(trailing.clone(), &reseal(&trailing)),
        Err(EdrlError::Invalid { .. })
    ));
    let mut three = body.clone();
    three[3] = 3;
    assert!(matches!(
        TokenizerSection::parse(three.clone(), &reseal(&three)),
        Err(EdrlError::Invalid { .. })
    ));
    let mut not_utf8 = body.clone();
    let at = body.len() - 5; // inside special_tokens.json
    not_utf8[at] = 0xff;
    assert!(matches!(
        TokenizerSection::parse(not_utf8.clone(), &reseal(&not_utf8)),
        Err(EdrlError::Invalid { .. })
    ));
    assert!(TokenizerSection::parse(body.clone(), &reseal(&body)).is_ok());
}

#[test]
fn missing_tokenizer_section_is_none() {
    let parsed = parse(FIXTURE).unwrap();
    let content = &FIXTURE[..parsed.content_end as usize];
    let tail: [u8; FOOTER_BYTES] = content[content.len() - FOOTER_BYTES..].try_into().unwrap();
    assert_eq!(TokenizerFooter::parse(&tail), Ok(None));

    let without = parse(content).unwrap();
    assert!(without.footer.is_none());
    assert_eq!(without.content_end, content.len() as u64);
    assert_eq!(without.table, parsed.table);
    assert_eq!(content_end(content.len() as u64, None), Ok(content.len() as u64));

    // The plan then has no tokenizer ranges.
    let plan = ReadPlan::new(&without.header, content.len() as u64, None).unwrap();
    assert_eq!(plan.range(Region::TokenizerBody), None);
    assert_eq!(plan.range(Region::TokenizerFooter), None);
}

#[test]
fn draft_depth_optional() {
    let parsed = parse(FIXTURE).unwrap();
    let region = parsed.header.metadata_range();
    let metadata = range(FIXTURE, region);

    // Without the trailing int32, the artifact leaves the depth to the engine.
    let without = Metadata::parse(&metadata[..metadata.len() - 4]).unwrap();
    assert_eq!(without.draft_depth, None);
    assert_eq!(
        ModelMetadata {
            draft_depth: Some(4),
            ..without
        },
        parsed.metadata
    );

    for (depth, accepted) in [(1i32, true), (16, true), (0, false), (17, false), (-1, false)] {
        let mut bytes = metadata.to_vec();
        let at = bytes.len() - 4;
        bytes[at..].copy_from_slice(&depth.to_be_bytes());
        match Metadata::parse(&bytes) {
            Ok(parsed) => assert!(accepted && parsed.draft_depth == Some(depth as u8), "depth {depth}"),
            Err(_) => assert!(!accepted, "depth {depth}"),
        }
    }

    // A fifth byte after the depth, or a partial depth, is not a depth.
    let mut trailing = metadata.to_vec();
    trailing.push(0);
    assert!(Metadata::parse(&trailing).is_err());
    assert!(Metadata::parse(&metadata[..metadata.len() - 2]).is_err());
}

#[test]
fn rejects_malformed_metadata() {
    let parsed = parse(FIXTURE).unwrap();
    let metadata = range(FIXTURE, parsed.header.metadata_range()).to_vec();

    assert!(matches!(
        Metadata::parse(&metadata[..20]),
        Err(EdrlError::Truncated { .. })
    ));

    // 12 int32s, 3 doubles, one int32, then the activation string "silu".
    let activation = 12 * 4 + 3 * 8 + 4;
    let layer_count = activation + 4 + 4;
    let layer_types = layer_count + 4;

    let mut negative = metadata.clone();
    negative[0..4].copy_from_slice(&(-1i32).to_be_bytes());
    assert!(matches!(Metadata::parse(&negative), Err(EdrlError::Invalid { .. })));

    let mut nan = metadata.clone();
    nan[48..56].copy_from_slice(&f64::NAN.to_be_bytes());
    assert!(matches!(Metadata::parse(&nan), Err(EdrlError::Invalid { .. })));

    let mut not_utf8 = metadata.clone();
    not_utf8[activation + 4] = 0xff;
    assert!(matches!(Metadata::parse(&not_utf8), Err(EdrlError::Invalid { .. })));

    let mut bad_layer = metadata.clone();
    bad_layer[layer_types + 3] = 2;
    assert!(matches!(Metadata::parse(&bad_layer), Err(EdrlError::Invalid { .. })));

    // The layer count must match the hidden layer count.
    let mut miscounted = metadata.clone();
    miscounted[8..12].copy_from_slice(&5i32.to_be_bytes());
    assert!(matches!(Metadata::parse(&miscounted), Err(EdrlError::Invalid { .. })));
    assert_eq!(&metadata[layer_count..layer_count + 4], &4i32.to_be_bytes());

    // The booleans are 0 or 1.
    let mut flag = metadata.clone();
    let at = metadata.len() - 4 - 4 - 2 + 1;
    flag[at] = 2;
    assert!(matches!(Metadata::parse(&flag), Err(EdrlError::Invalid { .. })));
}

#[test]
fn trellis_segments_validated() {
    let parsed = parse(FIXTURE).unwrap();
    let segments = range(FIXTURE, entry(&parsed, "trellis.segments").range());
    // Little-endian int32, unlike the table.
    let ends = TrellisCompanions::parse(segments).unwrap();
    assert_eq!(ends, [128, 256]);

    let suh = entry(&parsed, "trellis.suh");
    let payload = entry(&parsed, "trellis");
    let n = payload.shape()[1] * 16;
    assert_eq!(n, 256);
    assert_eq!(TrellisCompanions::check_rows(&ends, n, suh.shape()[0]), Ok(()));
    assert!(
        TrellisCompanions::check_rows(&ends, n, 3).is_err(),
        "one suh row per segment"
    );
    assert!(
        TrellisCompanions::check_rows(&ends, 384, suh.shape()[0]).is_err(),
        "the last end is n"
    );

    let le = |values: &[i32]| -> Vec<u8> { values.iter().flat_map(|value| value.to_le_bytes()).collect() };
    assert_eq!(TrellisCompanions::parse(&le(&[128])), Ok(vec![128]));
    let eight: Vec<i32> = (1..=8).map(|i| i * 128).collect();
    assert_eq!(TrellisCompanions::parse(&le(&eight)).unwrap().len(), 8);

    let nine: Vec<i32> = (1..=9).map(|i| i * 128).collect();
    for (name, bytes) in [
        ("more than eight", le(&nine)),
        ("none", Vec::new()),
        ("not a multiple of 128", le(&[100])),
        ("equal ends", le(&[128, 128])),
        ("decreasing ends", le(&[256, 128])),
        ("zero", le(&[0])),
        ("negative", le(&[-128])),
        ("a partial word", vec![0, 1, 0, 0, 0]),
        ("big-endian", 128i32.to_be_bytes().to_vec()),
    ] {
        assert!(
            matches!(TrellisCompanions::parse(&bytes), Err(EdrlError::Invalid { .. })),
            "{name}"
        );
    }
}

#[test]
fn read_plan_lists_ranges_in_file_order() {
    let parsed = parse(FIXTURE).unwrap();
    let file_len = FIXTURE.len() as u64;

    let probe = ReadPlan::probe(file_len).unwrap();
    assert_eq!(
        probe
            .reads()
            .iter()
            .map(|read| (read.region, read.range.clone()))
            .collect::<Vec<_>>(),
        [
            (Region::Header, 0..48),
            (Region::TokenizerFooter, file_len - 24..file_len)
        ]
    );

    let plan = ReadPlan::new(&parsed.header, file_len, parsed.footer.as_ref()).unwrap();
    let regions: Vec<Region> = plan.reads().iter().map(|read| read.region).collect();
    assert_eq!(
        regions,
        // The header and the footer are what `probe` already fetched.
        [Region::Metadata, Region::Table, Region::TokenizerBody]
    );
    assert_eq!(plan.range(Region::Header), None);
    assert_eq!(plan.range(Region::TokenizerFooter), None);
    let starts: Vec<u64> = plan.reads().iter().map(|read| read.range.start).collect();
    assert!(
        starts.windows(2).all(|pair| pair[0] < pair[1]),
        "file order: {starts:?}"
    );
    for pair in plan.reads().windows(2) {
        assert!(pair[0].range.end <= pair[1].range.start, "no region overlaps the next");
    }
    assert_eq!(plan.range(Region::Metadata), Some(parsed.header.metadata_range()));
    assert_eq!(plan.range(Region::Table), Some(parsed.header.table_range()));
    assert_eq!(plan.range(Region::TokenizerBody).unwrap().end, file_len - 24);
    // The tokenizer body starts where the payloads end.
    assert_eq!(plan.range(Region::TokenizerBody).unwrap().start, parsed.content_end);

    // A file too short for its header's regions has no plan.
    assert!(ReadPlan::new(&parsed.header, 100, None).is_err());
}

#[test]
fn rejects_duplicate_tensor_names() {
    let mut file = FIXTURE.to_vec();
    // `nvfp4` becomes a second `embed`.
    let at = entry_position(&file, "nvfp4") + 4;
    file[at..at + 5].copy_from_slice(b"embed");
    assert_eq!(
        parse(&file).err(),
        Some(EdrlError::DuplicateName { name: "embed".into() })
    );
}

#[test]
fn a_hostile_tensor_count_is_a_truncated_table() {
    let parsed = parse(FIXTURE).unwrap();
    let mut file = FIXTURE.to_vec();
    file[32..36].copy_from_slice(&1_000_000i32.to_be_bytes());
    assert!(matches!(parse(&file), Err(EdrlError::Truncated { .. })));
    assert_eq!(parsed.header.tensor_count(), 11);
}
