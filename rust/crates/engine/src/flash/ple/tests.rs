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

fn nvfp4(address: u64) -> Nvfp4Weight {
    Nvfp4Weight {
        weight: weight(address),
        sd4: false,
    }
}

/// Four heads (two for each of the sizes 2 and 3) over 1000 rows.
fn ids() -> NgramIds {
    NgramIds::new(3, 2, &[3, 5, 7], &[100, 1000, 100, 1000], &[0, 100, 200, 300], -1).unwrap()
}

/// A table of rows of `row_width` values: 32 values in 24-byte records.
fn table(row_width: u32) -> NgramRows {
    let shard: Arc<dyn ShardBytes> = Arc::new(vec![0u8; 1 << 16]);
    NgramRows::new(vec![shard], 100, 100, row_width).unwrap()
}

/// Four heads of 32-value rows (24-byte records) into 128 values over two streams of 64.
fn shape() -> PleShape {
    PleShape::new(64, 2, 128, &ids(), &table(32), 3, 2, 1e-6).unwrap()
}

fn device() -> Nvfp4Device {
    Nvfp4Device::with_native_route(true, 8)
}

fn weights() -> PleWeights {
    PleWeights {
        key_proj: nvfp4(0x100000),
        value_proj: nvfp4(0x110000),
        norm_key: weight(0x120000),
        norm_query: weight(0x130000),
        norm_conv: weight(0x140000),
        convolution: weight(0x150000),
    }
}

/// The scratch `shape` takes for `capacity` rows on `device`.
fn sizes(shape: &PleShape, capacity: u32) -> ScratchSizes {
    ScratchSizes {
        layer: shape.scratch_bytes(capacity),
        expansion: shape.expansion_bytes(&device(), capacity),
        ..ScratchSizes::default()
    }
}

fn regions_of(shape: &PleShape, streams: u32, capacity: u32, sizes: ScratchSizes) -> FlashRegions {
    let held = StreamShape::new(shape.hidden, streams).unwrap();
    FlashRegions::over(at(0x4000000), capacity, held, sizes).unwrap()
}

fn apply() -> PleApply {
    let shape = shape();
    let regions = regions_of(&shape, 2, 8, sizes(&shape, 8));
    PleApply::new(shape, weights(), at(0x190000), regions, device()).unwrap()
}

#[test]
fn the_apply_queues_javas_operations_in_javas_order() {
    let ops = apply().ops(3).unwrap();
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
    let lines = lines(&apply.ops(3).unwrap());
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
    let lines = lines(&apply().ops(3).unwrap());
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
    let (ids, rows) = (ids(), table(32));
    let with =
        |embed_dim, taps, dilation, streams| PleShape::new(64, streams, embed_dim, &ids, &rows, taps, dilation, 1e-6);
    assert!(with(128, 3, 2, 2).is_ok());
    // The heads' rows must concatenate to the embedding width.
    assert!(with(100, 3, 2, 2).is_err());
    // The history the convolution keeps is at most 32 rows, and a convolution has at least two taps.
    assert!(with(128, 3, 17, 2).is_err());
    assert!(with(128, 0, 2, 2).is_err());
    assert!(with(128, 1, 2, 2).is_err());
    assert!(with(128, 3, 0, 2).is_err());
    assert!(with(128, 3, 2, 0).is_err());
}

#[test]
fn the_shape_takes_the_heads_and_the_record_from_the_id_function_and_the_store() {
    let shape = shape();
    assert_eq!((shape.heads, shape.row_width, shape.record_bytes), (4, 32, 24));
    // Two heads of 64 values make the same width from a store of wider rows.
    let two = NgramIds::new(2, 2, &[3, 5], &[100, 1000], &[0, 100], -1).unwrap();
    let wide = PleShape::new(64, 2, 128, &two, &table(64), 3, 2, 1e-6).unwrap();
    assert_eq!((wide.heads, wide.row_width), (2, 64));
    // The id function and the store disagree: two heads of 32 values are not the 128 the projections read.
    assert!(PleShape::new(64, 2, 128, &two, &table(32), 3, 2, 1e-6).is_err());
}

#[test]
fn regions_of_another_shape_or_with_too_little_scratch_are_refused() {
    let shape = shape();
    let full = sizes(&shape, 8);
    let build = |streams, sizes| {
        PleApply::new(
            shape,
            weights(),
            at(0x190000),
            regions_of(&shape, streams, 8, sizes),
            device(),
        )
        .is_ok()
    };
    assert!(build(2, full));
    assert!(!build(4, full), "streams of another count");
    assert!(!build(
        2,
        ScratchSizes {
            layer: full.layer - 1,
            ..full
        }
    ));
}

#[test]
fn regions_without_the_expansion_the_native_route_needs_are_refused() {
    // 256 values of embedding: its projections take the native route from nine rows.
    let two = NgramIds::new(2, 8, &[3, 5], &[100; 8], &[0; 8], -1).unwrap();
    let shape = PleShape::new(64, 2, 256, &two, &table(32), 3, 2, 1e-6).unwrap();
    let full = sizes(&shape, 16);
    assert!(full.expansion > 0);
    let build = |sizes| {
        let regions = regions_of(&shape, 2, 16, sizes);
        PleApply::new(shape, weights(), at(0x190000), regions, device()).is_ok()
    };
    assert!(build(full));
    assert!(!build(ScratchSizes {
        expansion: full.expansion - 1,
        ..full
    }));
}

#[test]
fn the_projections_of_nine_rows_or_more_run_on_the_native_route() {
    let two = NgramIds::new(2, 8, &[3, 5], &[100; 8], &[0; 8], -1).unwrap();
    let shape = PleShape::new(64, 2, 256, &two, &table(32), 3, 2, 1e-6).unwrap();
    let regions = regions_of(&shape, 2, 16, sizes(&shape, 16));
    let apply = PleApply::new(shape, weights(), at(0x190000), regions, device()).unwrap();
    let native = |rows| {
        let ops = apply.ops(rows).unwrap();
        [1, 3].map(|index| lines(&ops)[index].contains(",native"))
    };
    assert_eq!(native(8), [false; 2]);
    assert_eq!(native(9), [true; 2]);
}

#[test]
fn an_upload_lands_at_its_records_place_in_the_staged_records() {
    let apply = apply();
    let records = apply
        .shape
        .carve(&mut Carver::new(apply.regions.layer_scratch), 5)
        .records
        .get();
    // Records of 24 bytes: rows 3..7 of the ids start 72 bytes in and take 96.
    let (destination, bytes) = apply.upload_extent(5, &(3..7));
    assert_eq!((destination.get(), bytes), (records + 72, 96));
    let (first, bytes) = apply.upload_extent(5, &(0..3));
    assert_eq!((first.get(), bytes), (records, 72));
}

fn quantum_ids() -> PleIdsStage {
    PleIdsStage::new(Arc::new(ids()))
}

#[test]
fn the_ids_stage_refuses_a_token_count_that_is_not_the_rows_and_ids_computed_twice() {
    let stage = quantum_ids();
    let rows = PleRows::new(ids().fresh_context());
    assert!(stage.hash(&rows, &[1, 2], 3).is_err());
    assert!(rows.ids().is_none());
    stage.hash(&rows, &[1, 2, 3], 3).unwrap();
    assert_eq!(rows.ids().map(<[u64]>::len), Some(12));
    assert!(stage.hash(&rows, &[1, 2, 3], 3).is_err());
}

#[test]
fn the_ids_stage_refuses_a_context_of_the_wrong_length() {
    let rows = PleRows::new(vec![0; 5].into());
    assert!(quantum_ids().hash(&rows, &[1], 1).is_err());
    assert!(rows.next_context().is_none());
}

#[test]
fn a_quantum_the_apply_cannot_serve_is_refused_before_any_upload() {
    let apply = apply();
    let ids_for = |rows: u32| {
        let hand_over = PleRows::new(ids().fresh_context());
        quantum_ids()
            .hash(&hand_over, &vec![1; rows as usize], rows as usize)
            .unwrap();
        hand_over
    };
    let message =
        |rows: u32, hand_over: &PleRows| format!("{:?}", apply.parts(rows, hand_over).map(|_| ()).unwrap_err());
    // More rows than the regions hold.
    assert!(message(9, &ids_for(9)).contains("exceeds"));
    // Ids that were never computed.
    assert!(message(3, &PleRows::new(ids().fresh_context())).contains("before the ids"));
    // Ids of two rows for a quantum of three.
    assert!(message(3, &ids_for(2)).contains("expected 12"));
    // Ids of three rows, none of whose parts were gathered.
    assert!(message(3, &ids_for(3)).contains("part 0 of the n-gram gather never arrived"));
}
