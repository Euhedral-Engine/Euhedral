//! Temporary: removed together with the native launch path, once the kernels run through RustEE.
//!
//! The attention launches (cache append, norm and rotary, causal attention over both cache formats) through the C
//! wrappers and through their plans on a RustEE lane, compared byte for byte:
//! `cargo test -p euhedral-inference-cuda --test twin_attention -- --ignored --test-threads=1 --nocapture`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the tallies are the test's output")]
#![allow(clippy::too_many_arguments, reason = "the wrappers take their shapes as arguments")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::attention::{
    AppendShape, CacheFormat, CausalShape, RopeShape, plan_causal, plan_kv_append, plan_qk_norm_rope,
};
use euhedral_inference_cuda::launch::catalog::Launch;
use euhedral_inference_cuda::launch::load::KernelSources;
use euhedral_inference_cuda::{CudaLibrary, DevicePtr, ops};

use common::twin::{Case, Random, Tally, Twin, watchdog};

const QUERY_HEADS: u32 = 24;
const KEY_HEADS: u32 = 4;
const HEAD_DIM: u32 = 256;
const QUERY_WIDTH: u32 = QUERY_HEADS * HEAD_DIM;
const KEY_WIDTH: u32 = KEY_HEADS * HEAD_DIM;
/// Elements of one row of the query, key and value planes.
const ROW: usize = (QUERY_WIDTH + KEY_WIDTH) as usize;
const ROTARY_DIM: u32 = 64;
const EPSILON: f32 = 1e-6;
const THETA: f64 = 10_000_000.0;
const PAGE_TOKENS: usize = 256;

const ROWS: [u32; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 33, 48, 64, 65, 128, 512];
const STARTS: [u64; 10] = [0, 1, 47, 48, 1023, 1024, 1025, 2047, 2048, 2049];

// The buffers of a case, in order; the pages follow.
const QK_IN: usize = 0;
const GATE_VALUE: usize = 1;
const QK_OUT: usize = 2;
const QUERY_NORM: usize = 3;
const KEY_NORM: usize = 4;
const POSITION: usize = 5;
const HISTORY_QK: usize = 6;
const HISTORY_GV: usize = 7;
const CHUNK_POSITIONS: usize = 8;
const KEY_TABLE: usize = 9;
const VALUE_TABLE: usize = 10;
const SCRATCH: usize = 11;
const OUT: usize = 12;
const FIRST_PAGE: usize = 13;

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

fn format_name(format: CacheFormat) -> &'static str {
    match format {
        CacheFormat::Nvfp4 => "nvfp4",
        CacheFormat::Bf16 => "bf16",
    }
}

/// The buffers of a case for `rows` rows at `start`; `history` makes the earlier tokens' source rows too.
fn buffers(random: &mut Random, format: CacheFormat, rows: u32, start: u64, history: bool) -> (Vec<Vec<u8>>, usize) {
    let total = start as usize + rows as usize;
    let pages = total.div_ceil(PAGE_TOKENS);
    let row_bytes = match format {
        CacheFormat::Nvfp4 => 144,
        CacheFormat::Bf16 => 512,
    };
    let page_bytes = PAGE_TOKENS * KEY_HEADS as usize * row_bytes;
    let chunks = if history {
        (start as usize).div_ceil(PAGE_TOKENS).max(1)
    } else {
        1
    };
    let history_rows = if history {
        start.min(PAGE_TOKENS as u64) as usize
    } else {
        0
    };
    let norm = |random: &mut Random| {
        // Positive weights of about 1.
        random
            .bf16(HEAD_DIM as usize)
            .chunks(2)
            .flat_map(|v| [v[0], v[1] & 0x7f])
            .collect::<Vec<u8>>()
    };
    let scratch_rows = if rows <= 64 { rows as usize } else { 1 };
    let mut all = vec![
        random.bf16(rows as usize * ROW),
        random.bf16(rows as usize * ROW),
        vec![0; rows as usize * ROW * 2],
        norm(random),
        norm(random),
        start.to_le_bytes().to_vec(),
        random.bf16(history_rows.max(1) * ROW),
        random.bf16(history_rows.max(1) * ROW),
        (0..chunks as u64)
            .flat_map(|chunk| (chunk * PAGE_TOKENS as u64).to_le_bytes())
            .collect(),
        vec![0; pages * 8],
        vec![0; pages * 8],
        vec![0; scratch_rows * QUERY_HEADS as usize * 64 * 258 * 4],
        vec![0; rows as usize * QUERY_WIDTH as usize * 2],
    ];
    for _ in 0..2 * pages {
        all.push(vec![0; page_bytes]);
    }
    (all, pages)
}

fn tables(pages: usize) -> Vec<(usize, Vec<usize>)> {
    vec![
        (KEY_TABLE, (FIRST_PAGE..FIRST_PAGE + pages).collect()),
        (VALUE_TABLE, (FIRST_PAGE + pages..FIRST_PAGE + 2 * pages).collect()),
    ]
}

fn append_native(
    format: CacheFormat,
    submission: &euhedral_inference_cuda::Submission<'_>,
    qk: DevicePtr,
    gv: DevicePtr,
    p: &[DevicePtr],
    rows: u32,
    start: u64,
    position: DevicePtr,
) -> Result<(), euhedral_inference_cuda::CudaError> {
    let (keys, values) = (p[KEY_TABLE], p[VALUE_TABLE]);
    match format {
        CacheFormat::Nvfp4 => ops::attention_kv_append_nvfp4(
            submission,
            qk,
            gv,
            keys,
            values,
            rows,
            QUERY_WIDTH,
            KEY_WIDTH,
            start,
            position,
        ),
        CacheFormat::Bf16 => ops::attention_kv_append_bf16(
            submission,
            qk,
            gv,
            keys,
            values,
            rows,
            QUERY_WIDTH,
            KEY_WIDTH,
            start,
            position,
        ),
    }
}

fn rope_native(
    submission: &euhedral_inference_cuda::Submission<'_>,
    p: &[DevicePtr],
    rows: u32,
    start: u64,
) -> Result<(), euhedral_inference_cuda::CudaError> {
    ops::attention_qk_norm_rope_bf16(
        submission,
        p[QK_IN],
        p[QUERY_NORM],
        p[KEY_NORM],
        p[QK_OUT],
        rows,
        QUERY_HEADS,
        KEY_HEADS,
        HEAD_DIM,
        ROTARY_DIM,
        start,
        p[POSITION],
        EPSILON,
        THETA,
    )
}

fn rope_shape(p: &[DevicePtr], rows: u32, start: u64) -> RopeShape {
    RopeShape {
        query_key: p[QK_IN],
        query_norm: p[QUERY_NORM],
        key_norm: p[KEY_NORM],
        output: p[QK_OUT],
        position: p[POSITION],
        rows,
        query_heads: QUERY_HEADS,
        key_value_heads: KEY_HEADS,
        head_dim: HEAD_DIM,
        rotary_dim: ROTARY_DIM,
        start_position: start,
        epsilon: EPSILON,
        rope_theta: THETA,
    }
}

fn append_shape(
    p: &[DevicePtr],
    qk: DevicePtr,
    gv: DevicePtr,
    rows: u32,
    start: u64,
    position: DevicePtr,
) -> AppendShape {
    AppendShape {
        query_key: qk,
        gate_value: gv,
        key_cache: p[KEY_TABLE],
        value_cache: p[VALUE_TABLE],
        position,
        rows,
        query_width: QUERY_WIDTH,
        key_value_width: KEY_WIDTH,
        start_position: start,
    }
}

fn causal_shape(p: &[DevicePtr], rows: u32, start: u64) -> CausalShape {
    CausalShape {
        query_key: p[QK_OUT],
        gate: p[GATE_VALUE],
        keys: p[KEY_TABLE],
        values: p[VALUE_TABLE],
        output: p[OUT],
        scratch: p[SCRATCH],
        position: p[POSITION],
        rows,
        query_heads: QUERY_HEADS,
        key_heads: KEY_HEADS,
        head_dim: HEAD_DIM,
        cache_length: (start + u64::from(rows)) as u32,
        start,
    }
}

/// Causal attention after real appends: the history through the C library, then this quantum's norm and rotary and
/// append, and the attention itself is the pair under test.
fn causal_case<'a>(
    name: &'a str,
    random: &mut Random,
    format: CacheFormat,
    rows: u32,
    start: u64,
    modes: Modes,
) -> Case<'a> {
    let (buffers, pages) = buffers(random, format, rows, start, true);
    Case {
        name,
        buffers,
        prepare: Some(Box::new(move |submission, p| {
            for chunk in 0..start.div_ceil(PAGE_TOKENS as u64) {
                let count = (start - chunk * PAGE_TOKENS as u64).min(PAGE_TOKENS as u64) as u32;
                append_native(
                    format,
                    submission,
                    p[HISTORY_QK],
                    p[HISTORY_GV],
                    p,
                    count,
                    chunk * PAGE_TOKENS as u64,
                    p[CHUNK_POSITIONS].offset(chunk * 8),
                )?;
            }
            rope_native(submission, p, rows, start)?;
            append_native(
                format,
                submission,
                p[QK_OUT],
                p[GATE_VALUE],
                p,
                rows,
                start,
                p[POSITION],
            )
        })),
        native: Box::new(move |submission, p| {
            let call = match format {
                CacheFormat::Nvfp4 => ops::attention_causal_nvfp4,
                CacheFormat::Bf16 => ops::attention_causal_bf16,
            };
            let cache_length = (start + u64::from(rows)) as u32;
            call(
                submission,
                p[QK_OUT],
                p[GATE_VALUE],
                p[KEY_TABLE],
                p[VALUE_TABLE],
                p[OUT],
                rows,
                QUERY_HEADS,
                KEY_HEADS,
                HEAD_DIM,
                cache_length,
                start,
                p[POSITION],
                p[SCRATCH],
            )
        }),
        plan: Box::new(move |p| {
            let mut launches = Vec::new();
            plan_causal(format, &causal_shape(p, rows, start), modes, &mut |launch| {
                launches.push(launch)
            })
            .unwrap();
            launches
        }),
        outputs: vec![OUT],
        row_exact: modes.row_exact,
        tables: tables(pages),
        scratch: vec![SCRATCH],
    }
}

/// Norm and rotary then the append into empty pages: both launches in plan order, the rotated rows and the pages
/// compared.
fn append_case<'a>(
    name: &'a str,
    random: &mut Random,
    format: CacheFormat,
    rows: u32,
    start: u64,
    modes: Modes,
) -> Case<'a> {
    let (buffers, pages) = buffers(random, format, rows, start, false);
    let mut outputs = vec![QK_OUT];
    outputs.extend(FIRST_PAGE..FIRST_PAGE + 2 * pages);
    Case {
        name,
        buffers,
        prepare: None,
        native: Box::new(move |submission, p| {
            rope_native(submission, p, rows, start)?;
            append_native(
                format,
                submission,
                p[QK_OUT],
                p[GATE_VALUE],
                p,
                rows,
                start,
                p[POSITION],
            )
        }),
        plan: Box::new(move |p| {
            let mut launches: Vec<Launch> = Vec::new();
            plan_qk_norm_rope(&rope_shape(p, rows, start), modes, &mut |launch| launches.push(launch)).unwrap();
            let shape = append_shape(p, p[QK_OUT], p[GATE_VALUE], rows, start, p[POSITION]);
            plan_kv_append(format, &shape, &mut |launch| launches.push(launch)).unwrap();
            launches
        }),
        outputs,
        row_exact: modes.row_exact,
        tables: tables(pages),
        scratch: vec![SCRATCH],
    }
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn attention_plans_match_the_native_wrappers() {
    let Some(library) = library() else { return };
    if KernelSources::beside(&library)
        .expect("the product directory")
        .compiler()
        .is_none()
    {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let done = watchdog(360);
    let mut twin = Twin::open(&library).expect("both sides open on one device");
    let mut random = Random(0x5eed);
    let mut tally = Tally::default();
    let began = Instant::now();
    let probe = DevicePtr::NULL.offset(0x1000);
    let at = |count: usize| vec![probe; count];
    for exact_numerics in [false, true] {
        library.exact_numerics_for_tests(exact_numerics);
        for row_exact in [false, true] {
            let modes = Modes {
                row_exact,
                exact_numerics,
            };
            for format in [CacheFormat::Nvfp4, CacheFormat::Bf16] {
                for rows in ROWS {
                    println!(
                        "{:>6.1?} {} rows {rows} row_exact {row_exact} exact_numerics {exact_numerics}",
                        began.elapsed(),
                        format_name(format)
                    );
                    for start in STARTS {
                        let shape = format!(
                            "{} rows {rows} start {start} row_exact {row_exact} exact_numerics {exact_numerics}",
                            format_name(format)
                        );
                        let name = shape.clone();
                        // What the planner refuses it refuses before any launch; the probe addresses stand for buffers.
                        let mut sink = |_: Launch| {};
                        let planned =
                            plan_causal(format, &causal_shape(&at(FIRST_PAGE), rows, start), modes, &mut sink);
                        if planned.is_err() {
                            tally.refused("causal");
                        } else {
                            let case = causal_case(&name, &mut random, format, rows, start, modes);
                            let verdict = twin.run(&case).expect("the case runs");
                            tally.record("causal", &shape, verdict);
                        }
                        // The rope and append twins do not depend on exact numerics.
                        if !exact_numerics {
                            let rope = plan_qk_norm_rope(&rope_shape(&at(FIRST_PAGE), rows, start), modes, &mut sink);
                            let append = plan_kv_append(
                                format,
                                &append_shape(&at(FIRST_PAGE), probe, probe, rows, start, probe),
                                &mut sink,
                            );
                            if rope.is_err() || append.is_err() {
                                tally.refused("rope+append");
                            } else {
                                let case = append_case(&name, &mut random, format, rows, start, modes);
                                let verdict = twin.run(&case).expect("the case runs");
                                tally.record("rope+append", &shape, verdict);
                            }
                        }
                    }
                }
            }
        }
    }
    library.exact_numerics_for_tests(false);
    done.store(true, Ordering::Relaxed);
    tally.finish();
}
