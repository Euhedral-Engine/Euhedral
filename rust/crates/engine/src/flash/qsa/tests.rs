use super::*;
use crate::flash::linear::Expansion;
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

fn attention() -> AttentionConfig {
    AttentionConfig {
        num_heads: 8,
        num_kv_heads: 2,
        head_dim: 256,
        partial_rotary_factor: 0.25,
        rope_theta: 1e7,
        mrope_section: Vec::new(),
        mrope_interleaved: false,
        output_gate: "sigmoid".to_owned(),
        full_attention_interval: 4,
    }
}

fn qsa() -> QsaConfig {
    QsaConfig {
        indexer_budget: 2048,
        compress_ratio: 4,
        head_dim: 128,
        kv_heads: 1,
        num_heads: 4,
    }
}

fn shape_with(max_tokens: u32, score_bytes: u64) -> QsaShape {
    QsaShape::new(256, &attention(), &qsa(), 1e-6, max_tokens, score_bytes).unwrap()
}

fn body_of(shape: QsaShape) -> QsaBody {
    let weights = QsaWeights {
        query: nvfp4(0x100000),
        key: nvfp4(0x110000),
        value: nvfp4(0x120000),
        output: nvfp4(0x130000),
        query_norm: weight(0x140000),
        key_norm: weight(0x150000),
        index: weight(0x160000),
        index_query_norm: weight(0x170000),
        index_key_norm: weight(0x180000),
    };
    let buffers = QsaBuffers {
        block_keys: at(0x1000000),
        tails: at(0x2000000),
    };
    QsaBody::new(shape, weights, buffers, Nvfp4Device::with_native_route(true, 8))
}

fn body() -> QsaBody {
    body_of(shape_with(4096, DEFAULT_SCORE_BYTES))
}

fn chunk(start: u32, tail_slot: u32) -> ChunkAt {
    ChunkAt {
        start,
        key_table: at(0x3000000),
        value_table: at(0x3100000),
        tail_slot,
    }
}

fn call(body: &QsaBody, rows: u32, chunk: ChunkAt) -> BlockCall {
    BlockCall {
        input: at(0x200000),
        output: at(0x210000),
        rows,
        exact: false,
        scratch: at(0x4000000),
        expansion: Expansion {
            address: at(0x8000000),
            bytes: body.expansion_bytes(rows.max(64)),
        },
        chunk,
    }
}

fn names(body: &QsaBody, rows: u32, chunk: ChunkAt) -> Vec<String> {
    let ops = body.ops(&call(body, rows, chunk)).unwrap();
    lines(&ops)
        .iter()
        .map(|line| line.split(' ').next().unwrap().to_owned())
        .collect()
}

fn count(names: &[String], symbol: &str) -> usize {
    names.iter().filter(|name| name.as_str() == symbol).count()
}

const NORM: &str = "euhedral_q4_head_norm_rope_bf16";
const SCORES: &str = "euhedral_q4_qsa_scores";
const SELECT: &str = "euhedral_q4_qsa_select";

#[test]
fn a_row_that_lists_every_block_projects_norms_appends_and_attends_without_scoring() {
    // Position 100: no block completes, no row chooses; 101 keys split three ways and merge.
    let names = names(&body(), 1, chunk(100, 0));
    assert_eq!(
        names,
        [
            "nvfp4_linear",
            "nvfp4_linear",
            "nvfp4_linear",
            "euhedral_q4_linear_bf16",
            NORM,
            NORM,
            NORM,
            "euhedral_q4_qsa_kv_append",
            "euhedral_q4_qsa_tail_bf16",
            "euhedral_q4_qsa_attention",
            "euhedral_q4_qsa_merge",
            "nvfp4_linear",
        ]
    );
}

#[test]
fn a_decode_row_past_the_budget_pools_the_block_it_completes_then_scores_and_selects() {
    // Position 3003 ends block 750; the row sees 750 blocks and keeps 512.
    let names = names(&body(), 1, chunk(3003, 0));
    assert_eq!(
        names,
        [
            "nvfp4_linear",
            "nvfp4_linear",
            "nvfp4_linear",
            "euhedral_q4_linear_bf16",
            NORM,
            NORM,
            NORM,
            "euhedral_q4_qsa_kv_append",
            "euhedral_q4_qsa_pool_keys_bf16",
            NORM,
            "euhedral_q4_qsa_tail_bf16",
            SCORES,
            SELECT,
            "euhedral_q4_qsa_attention",
            "euhedral_q4_qsa_merge",
            "nvfp4_linear",
        ]
    );
}

#[test]
fn the_pooled_block_keys_are_normed_and_rotated_at_their_own_first_positions() {
    let body = body();
    let call = call(&body, 1, chunk(3003, 0));
    let ops = body.ops(&call).unwrap();
    let scratch = body.shape.carve(&mut Carver::new(call.scratch), 1);
    // The block ending at 3003 is block 750; its key is rotated as position 3000, with the weight of the key norm.
    let fresh = body.buffers.block_keys.offset(750 * 256);
    let expected = HeadNormRope {
        input: fresh,
        weight: at(0x180000),
        output: fresh,
        rows: 1,
        heads: 1,
        width: 128,
        in_row_stride: 128,
        in_head_stride: 128,
        out_row_stride: 128,
        out_head_stride: 128,
        rotary: 64,
        position: 3000,
        position_step: 4,
        epsilon: body.shape.epsilon,
        theta: body.shape.rope_theta,
    }
    .launch()
    .unwrap();
    assert_eq!(lines(&ops)[9], Op::Launch(expected).line());
    // The raw keys are the last 128 values of the indexer row, which sits after the four heads.
    let raw = scratch.index.offset(4 * 128 * 2);
    let pool = ops::pool_keys(raw, body.buffers.tails, body.buffers.block_keys, 1, 3003, 640)
        .unwrap()
        .unwrap();
    assert_eq!(lines(&ops)[8], Op::Launch(pool).line());
}

#[test]
fn the_tail_is_read_from_the_chunks_slot_and_written_to_the_other() {
    let body = body();
    for slot in [0u32, 1] {
        let call = call(&body, 2, chunk(8, slot));
        let ops = body.ops(&call).unwrap();
        let scratch = body.shape.carve(&mut Carver::new(call.scratch), 2);
        let raw = scratch.index.offset(4 * 128 * 2);
        let read = body.buffers.tails.offset(u64::from(slot) * 768);
        let written = body.buffers.tails.offset(u64::from(slot ^ 1) * 768);
        let expected = ops::tail(raw, read, written, 2, 8, 640).unwrap();
        assert!(lines(&ops).contains(&Op::Launch(expected).line()), "slot {slot}");
    }
}

#[test]
fn a_prefill_chunk_lists_the_early_rows_and_scores_the_rest_in_tiles_of_the_scores_scratch() {
    // 4096 tokens: a row of scores is 1024 floats; 100 rows of the 640-float stride the chunk ends at fit 256000 bytes.
    let body = body_of(shape_with(4096, 256_000));
    let rows = 2500;
    let names = names(&body, rows, chunk(0, 0));
    // Rows 0..2051 see at most 512 blocks and just list them: one select. The 449 choosing rows take 5 tiles.
    assert_eq!(count(&names, SELECT), 1 + 5);
    assert_eq!(count(&names, SCORES), 5);
    // 2500 rows are one split: no merge, and the pooling covers the 625 complete blocks.
    assert_eq!(count(&names, "euhedral_q4_qsa_merge"), 0);
    assert_eq!(count(&names, "euhedral_q4_qsa_pool_keys_bf16"), 1);
    let first_select = names.iter().position(|name| name == SELECT).unwrap();
    let first_scores = names.iter().position(|name| name == SCORES).unwrap();
    assert!(
        first_select < first_scores,
        "the listing rows are selected before the scored ones"
    );
}

#[test]
fn the_leading_select_lists_the_rows_before_the_first_one_that_chooses() {
    let body = body();
    let call = call(&body, 2500, chunk(0, 0));
    let ops = body.ops(&call).unwrap();
    let scratch = body.shape.carve(&mut Carver::new(call.scratch), 2500);
    let leading = ops::select(scratch.scores, scratch.ids, scratch.counts, 0, 2051, 0, 0, 512).unwrap();
    assert!(lines(&ops).contains(&Op::Launch(leading).line()));
    // The tile covers the 449 rows left; each sees up to block 625 (stride 640).
    let scores = Scores {
        queries: scratch.index,
        keys: body.buffers.block_keys,
        scores: scratch.scores,
        row_begin: 2051,
        tile_rows: 449,
        start: 0,
        query_row_stride: 640,
        score_stride: 640,
        blocks_total: 625,
    }
    .launch()
    .unwrap();
    assert!(lines(&ops).contains(&Op::Launch(scores).line()));
}

#[test]
fn a_chunk_that_starts_past_the_budget_scores_all_its_rows() {
    let names = names(&body(), 3, chunk(2100, 0));
    assert_eq!(count(&names, SELECT), 1);
    assert_eq!(count(&names, SCORES), 1);
}

#[test]
fn attention_reads_the_chunks_page_tables_and_the_ids_only_when_rows_were_selected() {
    let body = body();
    let listing = call(&body, 1, chunk(100, 0));
    let scratch = body.shape.carve(&mut Carver::new(listing.scratch), 1);
    let attention = |ids, counts, splits, start, partial| {
        Op::Launch(
            Attention {
                queries: scratch.query,
                gate: scratch.query.offset(512),
                key_pages: at(0x3000000),
                value_pages: at(0x3100000),
                ids,
                counts,
                partial,
                core: DevicePtr::NULL,
                gated: scratch.gated,
                rows: 1,
                query_heads: 8,
                key_heads: 2,
                start,
                splits,
                budget: 512,
                query_row_stride: 4096,
                query_head_stride: 512,
                gate_row_stride: 4096,
                gate_head_stride: 512,
            }
            .launch()
            .unwrap(),
        )
        .line()
    };
    let ops = body.ops(&listing).unwrap();
    assert!(lines(&ops).contains(&attention(DevicePtr::NULL, DevicePtr::NULL, 3, 100, scratch.partial)));
    let choosing = call(&body, 1, chunk(3003, 0));
    let ops = body.ops(&choosing).unwrap();
    // 2051 keys at most: 64 splits.
    assert!(lines(&ops).contains(&attention(scratch.ids, scratch.counts, 64, 3003, scratch.partial)));
}

#[test]
fn a_chunk_the_state_cannot_hold_or_the_cache_does_not_back_is_refused_with_the_values() {
    let body = body();
    let refuse = |rows: u32, chunk: ChunkAt| body.ops(&call(&body, rows, chunk)).unwrap_err().to_string();
    assert!(refuse(2, chunk(4095, 0)).contains("4097 tokens, past the 4096"));
    assert!(refuse(1, chunk(4096, 0)).contains("4097 tokens"));
    assert!(body.ops(&call(&body, 1, chunk(4095, 0))).is_ok());
    assert!(refuse(1, chunk(0, 2)).contains("reads slot 2"));
    let no_keys = ChunkAt {
        key_table: DevicePtr::NULL,
        ..chunk(0, 0)
    };
    let no_values = ChunkAt {
        value_table: DevicePtr::NULL,
        ..chunk(0, 0)
    };
    assert!(refuse(1, no_keys).contains("page tables"));
    assert!(refuse(1, no_values).contains("page tables"));
}

#[test]
fn the_shape_refuses_what_the_kernels_are_not_written_for_with_the_value() {
    let refused = |change: fn(&mut AttentionConfig, &mut QsaConfig), hidden: u32, max_tokens: u32| {
        let (mut attention, mut qsa) = (attention(), qsa());
        change(&mut attention, &mut qsa);
        QsaShape::new(hidden, &attention, &qsa, 1e-6, max_tokens, DEFAULT_SCORE_BYTES)
            .unwrap_err()
            .to_string()
    };
    let same = |_: &mut AttentionConfig, _: &mut QsaConfig| {};
    assert!(
        refused(|_, q| q.compress_ratio = 8, 256, 64).contains("over blocks of 4, got 4 heads of 128 over blocks of 8")
    );
    assert!(refused(|_, q| q.num_heads = 2, 256, 64).contains("got 2 heads"));
    assert!(refused(|_, q| q.head_dim = 64, 256, 64).contains("heads of 64"));
    assert!(refused(|_, q| q.kv_heads = 2, 256, 64).contains("got 2 key heads"));
    assert!(refused(|_, q| q.indexer_budget = 0, 256, 64).contains("got 0"));
    assert!(refused(|_, q| q.indexer_budget = 10, 256, 64).contains("got 10"));
    assert!(refused(|a, _| a.head_dim = 128, 256, 64).contains("got 128 with 32"));
    assert!(refused(|a, _| a.partial_rotary_factor = 0.5, 256, 64).contains("with 128"));
    assert!(refused(|a, _| a.num_heads = 34, 256, 64).contains("got 34 query heads over 2"));
    assert!(refused(|a, _| a.num_kv_heads = 0, 256, 64).contains("over 0 key heads"));
    assert!(refused(same, 256, 0).contains("got 0"));
    assert!(refused(same, 256, 1 << 24).contains("got 16777216"));
    assert!(refused(same, 252, 64).contains("hidden 252"));
    assert!(refused(same, 0, 64).contains("hidden 0"));
    assert!(
        QsaShape::new(256, &attention(), &qsa(), f64::NAN, 64, 0)
            .unwrap_err()
            .to_string()
            .contains("epsilon NaN")
    );
    assert!(QsaShape::new(256, &attention(), &qsa(), 1e-6, (1 << 24) - 1, 0).is_ok());
}

#[test]
fn the_scores_scratch_holds_at_least_one_row_of_the_longest_sequence_and_is_aligned() {
    // 4096 tokens are 1024 blocks: one row of 1024 floats.
    assert_eq!(shape_with(4096, 0).score_bytes(), 4096);
    assert_eq!(shape_with(4096, 5000).score_bytes(), 5120);
    assert_eq!(shape_with(4096, DEFAULT_SCORE_BYTES).score_bytes(), DEFAULT_SCORE_BYTES);
    // 1000 tokens are 250 blocks, a stride of 256 floats.
    assert_eq!(shape_with(1000, 0).score_bytes(), 1024);
    // One token still has a row of 32 floats, padded to the alignment.
    assert_eq!(shape_with(1, 0).score_bytes(), 256);
}

#[test]
fn the_scratch_buffers_do_not_overlap_are_aligned_and_fit_the_reported_size() {
    let body = body();
    for rows in [1u32, 7, 64, 513] {
        let base = 0x4000000;
        let scratch = body.shape.carve(&mut Carver::new(at(base)), rows);
        let starts = [
            scratch.query,
            scratch.key,
            scratch.key_normed,
            scratch.value,
            scratch.index,
            scratch.scores,
            scratch.ids,
            scratch.counts,
            scratch.partial,
            scratch.gated,
        ]
        .map(DevicePtr::get);
        assert!(starts.iter().all(|start| start % ALIGNMENT == 0), "rows {rows}");
        assert!(starts.windows(2).all(|pair| pair[0] <= pair[1]), "rows {rows}");
        assert!(scratch.gated.get() - base < body.scratch_bytes(rows), "rows {rows}");
        assert!(body.scratch_bytes(rows) >= body.shape.score_bytes());
    }
    assert!(body.scratch_bytes(64) > body.scratch_bytes(1));
}

#[test]
fn the_split_partials_are_sized_for_the_largest_row_count_that_splits() {
    let shape = shape_with(4096, DEFAULT_SCORE_BYTES);
    // Two key heads: one row aims at 256 splits, capped at 64; 8 rows give 32; from 129 rows none split.
    assert_eq!(shape.splits_cap(1), 64);
    assert_eq!(shape.splits_cap(8), 32);
    assert_eq!(shape.splits_cap(256), 1);
    assert_eq!(shape.splits_cap(129), 1);
    assert_eq!(shape.partial_bytes(300), shape.partial_bytes(128));
    assert_eq!(shape.partial_bytes(1), 8 * 64 * u64::from(PARTIAL_FLOATS) * 4);
    // Each row count's own need is within the bytes of a call that size.
    for rows in (1..=300).filter(|rows| shape.splits_cap(*rows) > 1) {
        let need = u64::from(rows) * 8 * u64::from(shape.splits_cap(rows)) * u64::from(PARTIAL_FLOATS) * 4;
        assert!(need <= shape.partial_bytes(rows), "rows {rows}");
    }
    assert_eq!(shape.splits_for(1, 100), 3);
    assert_eq!(shape.splits_for(1, 31), 1);
    assert_eq!(shape.splits_for(1, 100_000), 64);
}

#[test]
fn the_expansion_is_the_widest_of_the_layers_linears() {
    let body = body();
    let widest = body
        .shape
        .linear_shapes()
        .iter()
        .map(|(k, n)| body.device.expansion_bytes(64, *k, *n))
        .max()
        .unwrap();
    assert_eq!(body.expansion_bytes(64), widest);
}
