use super::*;
use crate::flash::sequence::lines;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

fn stage(table_bytes: u64) -> Result<EmbedStreams, FlashError> {
    stage_over(table_bytes, 4)
}

/// The stage over regions of `streams` streams, for a four-stream shape.
fn stage_over(table_bytes: u64, streams: u32) -> Result<EmbedStreams, FlashError> {
    let shape = StreamShape::new(64, 4).unwrap();
    let held = StreamShape::new(64, streams).unwrap();
    let regions = FlashRegions::over(at(0x1000000), 8, held, ScratchSizes::default()).unwrap();
    let table = Weight {
        address: at(0x100000),
        bytes: table_bytes,
    };
    EmbedStreams::new(table, 32, shape, regions)
}

#[test]
fn the_lookup_feeds_the_repeat_into_the_streams() {
    let stage = stage(32 * 64 * 2).unwrap();
    let ops = stage.ops(at(0x300000), 5).unwrap();
    let lines = lines(&ops);
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("euhedral_q4_embedding_bf16"), "{}", lines[0]);
    assert!(lines[1].starts_with("euhedral_q4_repeat_streams_bf16"), "{}", lines[1]);
    let embedded = stage.regions.embedded.get();
    assert!(
        lines[0].contains(&format!("p100000,p300000,p{embedded:x},i5,i64,i32")),
        "{}",
        lines[0]
    );
    assert!(
        lines[1].contains(&format!("p{embedded:x},p{:x},i5,i4,i64", stage.regions.state.get())),
        "{}",
        lines[1]
    );
}

#[test]
fn a_table_of_the_wrong_size_is_refused_with_its_size() {
    let error = stage(1000).unwrap_err().to_string();
    assert!(error.contains("1000"), "{error}");
}

#[test]
fn regions_of_another_stream_count_are_refused() {
    assert!(stage_over(32 * 64 * 2, 4).is_ok());
    assert!(stage_over(32 * 64 * 2, 2).is_err());
}
