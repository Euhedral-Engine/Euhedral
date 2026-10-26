//! Parses the header, metadata, table and footer of a real artifact and compares the tensor table with the one the
//! converter's Python reader (`tools/euhedral_artifacts/edrl.py`) reads. Needs `euhedral.qwen.rco-artifact` (in
//! `~/.gradle/gradle.properties`, or `EUHEDRAL_TEST_QWEN_RCO_ARTIFACT`) to name an RCO NVFP4 + trellis artifact, and
//! `python3`. Never reads a payload.
//!
//! ```text
//! cargo test -p euhedral-inference-edrl -- --ignored
//! ```

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;

use euhedral_inference_edrl::{
    Header, LayerType, Metadata, ReadPlan, Region, TensorTable, TokenizerFooter, TokenizerSection, content_end,
};
use serde_json::Value;

const KEY: &str = "euhedral.qwen.rco-artifact";

fn read_range(file: &mut File, range: Range<u64>) -> Vec<u8> {
    file.seek(SeekFrom::Start(range.start)).unwrap();
    let mut bytes = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut bytes).unwrap();
    bytes
}

/// The artifact's table as the Python reader prints it.
// A local test fixture: it waits for a child process, which the workspace bans in engine code.
#[allow(clippy::disallowed_methods)]
fn python_table(artifact: &Path) -> Vec<Value> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let tools = manifest.join("../../../tools");
    let output = Command::new("python3")
        .arg("-I")
        .arg(manifest.join("tests/fixtures/read_table.py"))
        .arg(tools)
        .arg(artifact)
        .output()
        .expect("python3 runs");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
#[ignore = "needs the real RCO artifact (euhedral.qwen.rco-artifact) and python3"]
fn real_artifact_inventory_matches_python() {
    let Some(path) = test_support::require_property(KEY) else {
        return;
    };
    let path = PathBuf::from(path);
    if !path.is_file() {
        println!("skipped: {KEY} names no file");
        return;
    }
    let mut file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len();

    // The reads a loader makes: header and footer, then what they locate.
    let probe = ReadPlan::probe(file_len).unwrap();
    let header = Header::parse(
        read_range(&mut file, probe.range(Region::Header).unwrap())
            .as_slice()
            .try_into()
            .unwrap(),
    )
    .unwrap();
    let footer = TokenizerFooter::parse(
        read_range(&mut file, probe.range(Region::TokenizerFooter).unwrap())
            .as_slice()
            .try_into()
            .unwrap(),
    )
    .unwrap()
    .expect("the artifact carries its tokenizer");
    let plan = ReadPlan::new(&header, file_len, Some(&footer)).unwrap();
    let metadata = Metadata::parse(&read_range(&mut file, plan.range(Region::Metadata).unwrap())).unwrap();
    let table = TensorTable::parse(
        &read_range(&mut file, plan.range(Region::Table).unwrap()),
        &header,
        content_end(file_len, Some(&footer)).unwrap(),
    )
    .unwrap();
    let section = TokenizerSection::parse(
        read_range(&mut file, plan.range(Region::TokenizerBody).unwrap()),
        &footer,
    )
    .unwrap();

    // The RCO artifact: 64 layers, full attention on 3, 7, ..., 63, draft depth 4.
    assert_eq!(metadata.num_hidden_layers, 64);
    let full_attention: Vec<usize> = metadata
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, layer)| **layer == LayerType::FullAttention)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(full_attention, (3..64).step_by(4).collect::<Vec<_>>());
    assert_eq!(metadata.draft_depth, Some(4));
    assert!(section.file("tokenizer.json").is_some_and(|json| !json.is_empty()));

    // The Python reader sees the same table: same names, shapes, ordinals, offsets and sizes.
    let expected = python_table(&path);
    assert_eq!(table.len(), expected.len());
    assert_eq!(table.len() as u32, header.tensor_count());
    let mut formats: BTreeMap<String, usize> = BTreeMap::new();
    for (entry, python) in table.entries().iter().zip(&expected) {
        assert_eq!(python["name"], entry.name());
        assert_eq!(python["shape"], serde_json::json!(entry.shape()), "{}", entry.name());
        assert_eq!(python["dtype"], entry.dtype() as u64, "{}", entry.name());
        assert_eq!(python["format"], entry.format() as u64, "{}", entry.name());
        assert_eq!(python["layout"], entry.layout() as u64, "{}", entry.name());
        assert_eq!(python["offset"], entry.offset(), "{}", entry.name());
        assert_eq!(python["bytes"], entry.bytes(), "{}", entry.name());
        *formats
            .entry(format!("{:?}/{:?}", entry.format(), entry.layout()))
            .or_default() += 1;
    }
    println!("{} tensors: {formats:?}", table.len());
}
