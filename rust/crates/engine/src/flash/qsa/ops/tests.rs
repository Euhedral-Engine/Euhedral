use super::*;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

const A: u64 = 0x1000;
const B: u64 = 0x2000;
const C: u64 = 0x3000;
const D: u64 = 0x4000;

fn refusal(error: FlashError) -> String {
    error.to_string()
}

fn norm_rope() -> HeadNormRope {
    HeadNormRope {
        input: at(A),
        weight: at(B),
        output: at(C),
        rows: 6,
        heads: 3,
        width: 256,
        in_row_stride: 768,
        in_head_stride: 256,
        out_row_stride: 768,
        out_head_stride: 256,
        rotary: 64,
        position: 100,
        position_step: 1,
        epsilon: 1e-6,
        theta: 1e7,
    }
}

#[test]
fn head_norm_rope_runs_a_warp_per_head_with_javas_arguments() {
    let launch = norm_rope().launch().unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_head_norm_rope_bf16");
    // ceil(6 rows x 3 heads / 4) CTAs of 128 threads, no dynamic shared memory.
    assert_eq!(
        (launch.grid, launch.block, launch.shared_bytes),
        ([5, 1, 1], [128, 1, 1], 0)
    );
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .int32(6)
        .int32(3)
        .int32(256)
        .int32(768)
        .int32(256)
        .int32(768)
        .int32(256)
        .int32(64)
        .int32(100)
        .int32(1)
        .float32(1e-6)
        .float32(1e7);
    assert_eq!(launch.args, expected);
}

#[test]
fn head_norm_rope_refuses_what_the_kernel_is_not_written_for_with_the_value() {
    let refused = |change: fn(&mut HeadNormRope)| {
        let mut launch = norm_rope();
        change(&mut launch);
        refusal(launch.launch().unwrap_err())
    };
    assert!(refused(|l| l.rows = 0).contains("rows must be positive"));
    assert!(refused(|l| l.width = 48).contains("got 48"));
    assert!(refused(|l| l.width = 288).contains("got 288"));
    assert!(refused(|l| l.rotary = 32).contains("got 32"));
    assert!(
        refused(|l| {
            l.width = 32;
            l.rotary = 64;
        })
        .contains("a head of 64 values or more, got 32")
    );
    // The last position is 16_777_000 + 6 x 1_000.
    assert!(
        refused(|l| {
            l.position = 16_777_000;
            l.position_step = 1_000;
        })
        .contains("reach 16783000")
    );
    assert!(refused(|l| l.input = at(A + 2)).contains("input 0x1002 is not 4-byte aligned"));
    // A head with no rotation takes any width the kernel serves.
    let mut plain = norm_rope();
    plain.width = 32;
    plain.rotary = 0;
    assert!(plain.launch().is_ok());
}

#[test]
fn pooling_runs_a_warp_per_completed_block_and_is_nothing_when_none_completes() {
    // Rows 5..8 at 6 complete one block (token 8 is the first of the next one): blocks 1 and 2 of the chunk 6..14.
    assert_eq!(completed_blocks(6, 8), 2);
    assert_eq!(completed_blocks(5, 3), 1);
    assert_eq!(completed_blocks(4, 3), 0);
    assert_eq!(completed_blocks(0, 1), 0);
    assert!(pool_keys(at(A), at(B), at(C), 3, 4, 640).unwrap().is_none());
    let launch = pool_keys(at(A), at(B), at(C), 8, 6, 640).unwrap().unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_pool_keys_bf16");
    assert_eq!((launch.grid, launch.block), ([1, 1, 1], [128, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .int32(2)
        .int32(6)
        .int32(640);
    assert_eq!(launch.args, expected);
    // Five blocks need two CTAs of four warps.
    assert_eq!(
        pool_keys(at(A), at(B), at(C), 20, 0, 640).unwrap().unwrap().grid,
        [2, 1, 1]
    );
    assert!(refusal(pool_keys(at(A), at(B + 2), at(C), 8, 6, 640).unwrap_err()).contains("tail 0x2002"));
}

#[test]
fn the_tail_is_written_beside_the_one_it_reads() {
    let launch = tail(at(A), at(B), at(C), 3, 6, 640).unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_tail_bf16");
    assert_eq!((launch.grid, launch.block), ([1, 1, 1], [128, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .int32(3)
        .int32(6)
        .int32(640);
    assert_eq!(launch.args, expected);
    assert!(refusal(tail(at(A), at(B), at(B), 3, 6, 640).unwrap_err()).contains("both are 0x2000"));
    assert!(tail(at(A), at(B), at(C), 0, 6, 640).is_err());
}

#[test]
fn scores_cover_64_blocks_by_32_rows_per_cta() {
    let scores = Scores {
        queries: at(A),
        keys: at(B),
        scores: at(C),
        row_begin: 40,
        tile_rows: 65,
        start: 1000,
        query_row_stride: 640,
        score_stride: 192,
        blocks_total: 129,
    };
    let launch = scores.launch().unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_scores");
    assert_eq!((launch.grid, launch.block), ([3, 3, 1], [256, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .int32(40)
        .int32(65)
        .int32(1000)
        .int32(640)
        .int32(192)
        .int32(129);
    assert_eq!(launch.args, expected);
    assert!(
        Scores {
            query_row_stride: 641,
            ..scores
        }
        .launch()
        .is_err()
    );
    assert!(
        Scores {
            blocks_total: 0,
            ..scores
        }
        .launch()
        .is_err()
    );
    assert!(
        Scores {
            keys: at(B + 1),
            ..scores
        }
        .launch()
        .is_err()
    );
}

#[test]
fn selection_runs_a_cta_of_1024_per_row() {
    let launch = select(at(A), at(B), at(C), 7, 33, 1000, 192, 512).unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_select");
    assert_eq!((launch.grid, launch.block), ([33, 1, 1], [1024, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .int32(7)
        .int32(1000)
        .int32(192)
        .int32(512);
    assert_eq!(launch.args, expected);
    assert!(select(at(A), at(B), at(C), 0, 0, 0, 0, 512).is_err());
    assert!(select(at(A), at(B), at(C), 0, 1, 0, 0, 0).is_err());
}

fn attention() -> Attention {
    Attention {
        queries: at(A),
        gate: at(A + 512),
        key_pages: at(B),
        value_pages: at(B + 0x100),
        ids: at(C),
        counts: at(C + 0x100),
        partial: at(D),
        core: DevicePtr::NULL,
        gated: at(D + 0x100),
        rows: 3,
        query_heads: 8,
        key_heads: 2,
        start: 900,
        splits: 4,
        budget: 512,
        query_row_stride: 4096,
        query_head_stride: 512,
        gate_row_stride: 4096,
        gate_head_stride: 512,
    }
}

#[test]
fn attention_runs_a_unit_per_row_key_head_and_split_with_the_shared_bytes_of_the_catalog() {
    let launch = attention().launch().unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_attention");
    assert_eq!(
        (launch.grid, launch.block, launch.shared_bytes),
        ([3 * 2 * 4, 1, 1], [96, 1, 1], 28224)
    );
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(A + 512))
        .pointer(at(B))
        .pointer(at(B + 0x100))
        .pointer(at(C))
        .pointer(at(C + 0x100))
        .pointer(at(D))
        .pointer(DevicePtr::NULL)
        .pointer(at(D + 0x100))
        .int32(3)
        .int32(8)
        .int32(2)
        .int32(900)
        .int32(4)
        .int32(512)
        .int32(4096)
        .int32(512)
        .int32(4096)
        .int32(512);
    assert_eq!(launch.args, expected);
}

#[test]
fn attention_refuses_a_group_too_wide_and_an_arrangement_it_cannot_run() {
    let refused = |change: fn(&mut Attention)| {
        let mut launch = attention();
        change(&mut launch);
        refusal(launch.launch().unwrap_err())
    };
    assert!(refused(|a| a.query_heads = 34).contains("got 34 over 2"));
    assert!(refused(|a| a.query_heads = 9).contains("got 9 over 2"));
    assert!(refused(|a| a.partial = DevicePtr::NULL).contains("4 splits need a partial buffer"));
    assert!(refused(|a| a.counts = DevicePtr::NULL).contains("ids and counts go together"));
    assert!(refused(|a| a.rows = 0).contains("rows must be positive"));
    // One split needs no partial, and no ids is every block.
    let mut dense = attention();
    dense.splits = 1;
    dense.partial = DevicePtr::NULL;
    dense.ids = DevicePtr::NULL;
    dense.counts = DevicePtr::NULL;
    assert!(dense.launch().is_ok());
}

#[test]
fn merge_runs_a_warp_per_row_and_head_and_kv_append_a_warp_per_row_and_key_head() {
    let launch = merge(at(A), at(B), DevicePtr::NULL, at(C), 3, 8, 4, 4096, 512).unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_merge");
    // ceil(3 x 8 / 4) CTAs of 128 threads.
    assert_eq!((launch.grid, launch.block), ([6, 1, 1], [128, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(DevicePtr::NULL)
        .pointer(at(C))
        .int32(3)
        .int32(8)
        .int32(4)
        .int32(4096)
        .int32(512);
    assert_eq!(launch.args, expected);
    assert!(merge(at(A), at(B), DevicePtr::NULL, at(C), 3, 8, 0, 4096, 512).is_err());

    let launch = kv_append(at(A), at(B), at(C), at(D), 5, 2, 512, 512, 77).unwrap();
    assert_eq!(launch.symbol, "euhedral_q4_qsa_kv_append");
    assert_eq!((launch.grid, launch.block), ([3, 1, 1], [128, 1, 1]));
    let expected = KernelArguments::new()
        .pointer(at(A))
        .pointer(at(B))
        .pointer(at(C))
        .pointer(at(D))
        .int32(5)
        .int32(2)
        .int32(512)
        .int32(512)
        .int32(77);
    assert_eq!(launch.args, expected);
    assert!(
        refusal(kv_append(at(A + 8), at(B), at(C), at(D), 5, 2, 512, 512, 77).unwrap_err()).contains("keys 0x1008")
    );
}
