use super::*;
use crate::flash::buffers::StreamShape;
use crate::flash::sequence::lines;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

fn weight(address: u64) -> Weight {
    Weight {
        address: at(address),
        bytes: 4096,
    }
}

/// Four heads of 32-value rows (24-byte records) into 128 values over two streams of 64.
fn shape() -> PleShape {
    PleShape::new(64, 2, 128, 4, 32, 24, 3, 2, 1e-6).unwrap()
}

fn apply() -> PleApply {
    let weights = PleWeights {
        key_proj: weight(0x100000),
        value_proj: weight(0x110000),
        norm_key: weight(0x120000),
        norm_query: weight(0x130000),
        norm_conv: weight(0x140000),
        convolution: weight(0x150000),
    };
    let streams = StreamShape::new(64, 2).unwrap();
    let regions = FlashRegions::over(at(0x4000000), 8, streams, 0, shape().scratch_bytes(8)).unwrap();
    PleApply::new(shape(), weights, at(0x190000), regions)
}

#[test]
fn the_apply_queues_javas_operations_in_javas_order() {
    let ops = apply().ops(3, false).unwrap();
    let names: Vec<_> = lines(&ops)
        .iter()
        .map(|line| line.split(' ').next().unwrap().to_owned())
        .collect();
    assert_eq!(
        names,
        [
            "euhedral_q4_ngram_expand_bf16",
            "nvfp4_linear",
            "euhedral_q4_grouped_rms_norm_bf16",
            "nvfp4_linear",
            "euhedral_q4_grouped_rms_norm_bf16",
            "euhedral_q4_ple_gate_bf16",
            "euhedral_q4_grouped_rms_norm_bf16",
            "euhedral_q4_ple_conv_bf16",
            "euhedral_q4_conv_history_bf16",
            "residual_add",
        ]
    );
}

#[test]
fn the_expand_reads_the_staged_records_and_the_embedding_feeds_both_projections() {
    let apply = apply();
    let scratch = apply.shape.carve(&mut Carver::new(apply.regions.layer_scratch), 3);
    let lines = lines(&apply.ops(3, false).unwrap());
    // Twelve records (3 rows of 4 heads) of 32 values in 24-byte records.
    assert!(
        lines[0].contains(&format!(
            "p{:x},p{:x},i12,i32,i24",
            scratch.records.get(),
            scratch.embedding.get()
        )),
        "{}",
        lines[0]
    );
    assert!(lines[1].starts_with(&format!(
        "nvfp4_linear p{:x},p100000,p{:x},i3,i128,i128",
        scratch.embedding.get(),
        scratch.key.get()
    )));
    assert!(lines[3].starts_with(&format!(
        "nvfp4_linear p{:x},p110000,p{:x},i3,i128,i64",
        scratch.embedding.get(),
        scratch.value.get()
    )));
    // The query is the streams, normed; the add folds the output into the streams in place.
    let state = apply.regions.state.get();
    assert!(
        lines[4].contains(&format!("p{state:x},p130000,p{:x}", scratch.query.get())),
        "{}",
        lines[4]
    );
    let output = apply.regions.ple_output.get();
    assert_eq!(
        lines[9],
        format!("residual_add p{state:x},p{output:x},p{state:x},i3,i128")
    );
}

#[test]
fn the_history_keeps_taps_less_one_rows_at_the_dilation() {
    let lines = lines(&apply().ops(3, false).unwrap());
    // Two taps' worth of dilation 2: four rows of history over 128 channels.
    assert!(lines[8].ends_with("i3,i128,i4"), "{}", lines[8]);
    assert!(lines[7].ends_with("i3,i128,i3,i2"), "{}", lines[7]);
}

#[test]
fn the_scratch_buffers_follow_javas_order_and_the_reported_size_holds_them() {
    let shape = shape();
    let rows = 5;
    let scratch = shape.carve(&mut Carver::new(at(0x400000)), rows);
    let starts = [
        scratch.records,
        scratch.embedding,
        scratch.key,
        scratch.key_normed,
        scratch.value,
        scratch.query,
        scratch.gated,
        scratch.gated_normed,
    ]
    .map(DevicePtr::get);
    assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(scratch.gated_normed.get() + bf16_bytes(rows, 128) <= 0x400000 + shape.scratch_bytes(rows));
}

#[test]
fn a_shape_the_table_and_the_streams_cannot_serve_is_refused() {
    // The heads' rows must concatenate to the embedding width.
    assert!(PleShape::new(64, 2, 100, 4, 32, 24, 3, 2, 1e-6).is_err());
    assert!(PleShape::new(64, 2, 128, 0, 32, 24, 3, 2, 1e-6).is_err());
    // The history the convolution keeps is at most 32 rows.
    assert!(PleShape::new(64, 2, 128, 4, 32, 24, 3, 17, 1e-6).is_err());
    assert!(PleShape::new(64, 2, 128, 4, 32, 24, 0, 2, 1e-6).is_err());
    assert!(PleShape::new(64, 0, 128, 4, 32, 24, 3, 2, 1e-6).is_err());
    // A record the expand kernel cannot read: not a multiple of four bytes, too small for its row, a row of partial blocks.
    assert!(PleShape::new(64, 2, 128, 4, 32, 22, 3, 2, 1e-6).is_err());
    assert!(PleShape::new(64, 2, 128, 4, 32, 20, 3, 2, 1e-6).is_err());
    assert!(PleShape::new(64, 2, 120, 4, 30, 24, 3, 2, 1e-6).is_err());
}
