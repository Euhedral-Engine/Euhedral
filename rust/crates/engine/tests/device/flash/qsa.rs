//! The sparse-attention launches of Flash-Next on a GPU, each against a host reference at a small synthetic shape: the
//! per-head norm and RoPE, the block-key pooling and raw tail across chunkings, the block scores, the exact top-k
//! selection (ties included) at a decode and a prefill row count, and the attention over the NVFP4 pages with its split
//! merge. The budget is a launch argument, so a few blocks make the rows choose at tens of tokens.

use std::iter::repeat_n;

use euhedral_inference_cuda::{DeviceBuffer, DevicePtr};
use euhedral_inference_engine::flash::qsa::ops::{
    Attention, HeadNormRope, Scores, kv_append, merge, pool_keys, select, tail,
};
use euhedral_inference_engine::flash::sequence::Op;
use euhedral_inference_engine::formats::shapes::Launch;

use super::{E2M1, e4m3, gaussians, on_device, queue, read, read_f32, round, sigmoid};
use crate::common;
use crate::linear_gpu::{Random, Rig, assert_close, relative_rms, rig, upload};

const EPSILON: f32 = 1e-6;
const THETA: f32 = 1e7;
/// One indexer row: four heads of 128 queries, then the raw key of 128 values.
const INDEX_ROW: usize = 640;
const KEY_HEADS: usize = 2;
const QUERY_HEADS: usize = 8;
const HEAD: usize = 256;
/// The `[q | gate]` row of a head, and the whole query row.
const QUERY_HEAD_STRIDE: usize = 2 * HEAD;
const QUERY_ROW: usize = QUERY_HEADS * QUERY_HEAD_STRIDE;
const PAGE_ROW_BYTES: usize = 144;

fn launch(rig: &Rig, launches: Vec<Launch>) {
    let ops: Vec<Op<'_>> = launches.into_iter().map(Op::Launch).collect();
    queue(rig, &ops);
}

fn read_i32(buffer: &DeviceBuffer, count: usize) -> Vec<i32> {
    let mut host = vec![0u8; count * 4];
    buffer.copy_to_host(0, &mut host).unwrap();
    host.as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// The normalized Sylvester transform of 256 values, as the cache stores keys and values.
fn hadamard(values: &mut [f32]) {
    let mut stride = 1;
    while stride < values.len() {
        for base in (0..values.len()).step_by(2 * stride) {
            for offset in base..base + stride {
                let (low, high) = (values[offset], values[offset + stride]);
                values[offset] = low + high;
                values[offset + stride] = low - high;
            }
        }
        stride *= 2;
    }
    for value in values.iter_mut() {
        *value *= 0.0625;
    }
}

/// The host norm and RoPE of one head (`QsaOps` rounds to BF16 after every step).
fn norm_rope(x: &[f32], weight: &[f32], rotary: usize, position: u32, theta: f32) -> Vec<f32> {
    let width = x.len();
    let sum: f32 = x.iter().map(|value| value * value).sum();
    let inverse = 1.0 / (sum / width as f32 + EPSILON).sqrt();
    let mut v: Vec<f32> = x
        .iter()
        .zip(weight)
        .map(|(x, w)| round((x * inverse) * (1.0 + w)))
        .collect();
    if rotary == 64 {
        for lane in 0..32 {
            let power = f64::from(theta).powf(2.0 * lane as f64 / 64.0) as f32;
            let angle = position as f32 * (1.0 / power);
            let (cosine, sine) = (round(angle.cos()), round(angle.sin()));
            let (first, second) = (v[lane], v[lane + 32]);
            v[lane] = round(round(first * cosine) + round(-second * sine));
            v[lane + 32] = round(round(second * cosine) + round(first * sine));
        }
    }
    v
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn head_norm_rope_matches_the_host_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(71);
    // Heads of 256 inside `[q | gate]` rows, normalized and rotated in place; the gate half stays as it is.
    let (rows, heads) = (3usize, 2usize);
    let row_stride = heads * QUERY_HEAD_STRIDE;
    let x = gaussians(&mut random, rows * row_stride, 1.0);
    let weight = gaussians(&mut random, HEAD, 0.5);
    let (x_d, weight_d) = (on_device(&rig, &x), on_device(&rig, &weight));
    let position = 1000u32;
    launch(
        &rig,
        vec![
            HeadNormRope {
                input: x_d.ptr(),
                weight: weight_d.ptr(),
                output: x_d.ptr(),
                rows: rows as u32,
                heads: heads as u32,
                width: HEAD as u32,
                in_row_stride: row_stride as u32,
                in_head_stride: QUERY_HEAD_STRIDE as u32,
                out_row_stride: row_stride as u32,
                out_head_stride: QUERY_HEAD_STRIDE as u32,
                rotary: 64,
                position,
                position_step: 1,
                epsilon: EPSILON,
                theta: THETA,
            }
            .launch()
            .unwrap(),
        ],
    );
    let got = read(&x_d, x.len());
    for row in 0..rows {
        for head in 0..heads {
            let at = row * row_stride + head * QUERY_HEAD_STRIDE;
            let wanted = norm_rope(&x[at..at + HEAD], &weight, 64, position + row as u32, THETA);
            assert_close(
                &wanted,
                &got[at..at + HEAD],
                HEAD,
                0.03,
                &format!("row {row} head {head}"),
            );
            assert_eq!(
                &got[at + HEAD..at + QUERY_HEAD_STRIDE],
                &x[at + HEAD..at + QUERY_HEAD_STRIDE],
                "the gate stays"
            );
        }
    }

    // Block keys of 128: rotated at 4 tokens apart, into another buffer.
    let keys = gaussians(&mut random, 5 * 128, 1.0);
    let norm = gaussians(&mut random, 128, 0.5);
    let (keys_d, norm_d, out_d) = (
        on_device(&rig, &keys),
        on_device(&rig, &norm),
        upload(&rig.library, &[0u8; 5 * 256]),
    );
    launch(
        &rig,
        vec![
            HeadNormRope {
                input: keys_d.ptr(),
                weight: norm_d.ptr(),
                output: out_d.ptr(),
                rows: 5,
                heads: 1,
                width: 128,
                in_row_stride: 128,
                in_head_stride: 128,
                out_row_stride: 128,
                out_head_stride: 128,
                rotary: 64,
                position: 40,
                position_step: 4,
                epsilon: EPSILON,
                theta: THETA,
            }
            .launch()
            .unwrap(),
        ],
    );
    let got = read(&out_d, keys.len());
    for row in 0..5usize {
        let wanted = norm_rope(&keys[row * 128..][..128], &norm, 64, 40 + 4 * row as u32, THETA);
        assert_close(
            &wanted,
            &got[row * 128..][..128],
            128,
            0.03,
            &format!("block key {row}"),
        );
    }
}

/// The host pooled key of block `block`: the mean of its four raw keys, in the kernel's order.
fn pooled(raw: &[Vec<f32>], block: usize) -> Vec<f32> {
    (0..128)
        .map(|d| {
            let mut sum = 0.0f32;
            for key in &raw[4 * block..4 * block + 4] {
                sum += key[d];
            }
            round(sum * 0.25)
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn pooling_and_the_tail_give_the_same_block_keys_for_any_chunking() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(72);
    let total = 22usize;
    let raw: Vec<Vec<f32>> = (0..total).map(|_| gaussians(&mut random, 128, 1.0)).collect();
    // Indexer rows of the whole sequence: the raw key sits after the four heads' queries.
    let mut index = vec![0.0f32; total * INDEX_ROW];
    for (token, key) in raw.iter().enumerate() {
        index[token * INDEX_ROW + 512..][..128].copy_from_slice(key);
    }
    let index_d = on_device(&rig, &index);
    let tails_d = upload(&rig.library, &vec![0u8; 2 * 3 * 256]);
    let wanted: Vec<Vec<f32>> = (0..total / 4).map(|block| pooled(&raw, block)).collect();
    for chunks in [vec![22usize], vec![6, 1, 15], vec![1; 22], vec![3, 3, 3, 3, 3, 3, 4]] {
        let blocks_d = upload(&rig.library, &vec![0u8; (total / 4) * 256]);
        let (mut start, mut slot) = (0usize, 0u64);
        for rows in &chunks {
            let raw_ptr = index_d.ptr().offset((start * INDEX_ROW * 2 + 512 * 2) as u64);
            let read_tail = tails_d.ptr().offset(slot * 768);
            let write_tail = tails_d.ptr().offset((slot ^ 1) * 768);
            let mut launches = Vec::new();
            if let Some(pool) = pool_keys(
                raw_ptr,
                read_tail,
                blocks_d.ptr(),
                *rows as u32,
                start as u32,
                INDEX_ROW as u32,
            )
            .unwrap()
            {
                launches.push(pool);
            }
            launches.push(
                tail(
                    raw_ptr,
                    read_tail,
                    write_tail,
                    *rows as u32,
                    start as u32,
                    INDEX_ROW as u32,
                )
                .unwrap(),
            );
            launch(&rig, launches);
            start += rows;
            slot ^= 1;
        }
        let got = read(&blocks_d, (total / 4) * 128);
        for (block, wanted) in wanted.iter().enumerate() {
            assert_eq!(
                &got[block * 128..][..128],
                wanted.as_slice(),
                "chunks {chunks:?} block {block}"
            );
        }
        // The incomplete block's raw keys are the tail: tokens 20 and 21.
        let tail_now = read(&tails_d, 2 * 3 * 128);
        let live = &tail_now[slot as usize * 384..];
        assert_eq!(&live[..128], raw[20].as_slice(), "chunks {chunks:?} tail token 20");
        assert_eq!(&live[128..256], raw[21].as_slice(), "chunks {chunks:?} tail token 21");
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn scores_match_the_host_reference_for_a_decode_row_and_a_tile_across_block_tiles() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(73);
    // (start, rows): one row at 3003; 40 rows at 240 see up to 70 blocks, so two CTAs of 64 blocks and two of 32 rows.
    for (start, rows) in [(3003usize, 1usize), (240, 40)] {
        let end = start + rows;
        let blocks = end / 4;
        let queries = gaussians(&mut random, rows * INDEX_ROW, 1.0);
        let keys = gaussians(&mut random, blocks * 128, 1.0);
        let stride = blocks.next_multiple_of(32);
        let (queries_d, keys_d) = (on_device(&rig, &queries), on_device(&rig, &keys));
        let scores_d = upload(&rig.library, &f32_bytes(&vec![-7.0f32; rows * stride]));
        launch(
            &rig,
            vec![
                Scores {
                    queries: queries_d.ptr(),
                    keys: keys_d.ptr(),
                    scores: scores_d.ptr(),
                    row_begin: 0,
                    tile_rows: rows as u32,
                    start: start as u32,
                    query_row_stride: INDEX_ROW as u32,
                    score_stride: stride as u32,
                    blocks_total: blocks as u32,
                }
                .launch()
                .unwrap(),
            ],
        );
        let got = read_f32(&scores_d, rows * stride);
        for row in 0..rows {
            let nb = (start + row + 1) / 4;
            let wanted: Vec<f32> = (0..nb)
                .map(|block| {
                    let mut sum = 0.0f32;
                    for head in 0..4 {
                        let dot: f32 = (0..128)
                            .map(|d| queries[row * INDEX_ROW + head * 128 + d] * keys[block * 128 + d])
                            .sum();
                        sum += dot.max(0.0);
                    }
                    sum * (1.0 / 128f32.sqrt())
                })
                .collect();
            assert_close(
                &wanted,
                &got[row * stride..][..nb],
                nb,
                0.01,
                &format!("start {start} row {row}"),
            );
            // Past the row's blocks the kernel writes nothing.
            assert!(
                got[row * stride + nb..(row + 1) * stride]
                    .iter()
                    .all(|score| *score == -7.0),
                "row {row} tail"
            );
        }
    }
}

/// The host selection: the `budget` highest scores (negatives count as zero), the lowest ids among equals, ascending.
fn host_select(scores: &[f32], budget: usize) -> Vec<i32> {
    let nb = scores.len();
    if nb <= budget {
        return (0..nb as i32).collect();
    }
    let mut order: Vec<usize> = (0..nb).collect();
    order.sort_by(|a, b| scores[*b].max(0.0).total_cmp(&scores[*a].max(0.0)).then(a.cmp(b)));
    let mut kept: Vec<i32> = order[..budget].iter().map(|id| *id as i32).collect();
    kept.sort_unstable();
    kept
}

/// The scores of a row of `nb` blocks: a few levels (heavy ties, negatives included), or distinct Gaussians.
fn scores_row(random: &mut Random, nb: usize, tied: bool) -> Vec<f32> {
    const LEVELS: [f32; 6] = [-1.0, 0.0, 0.5, 1.0, 1.0, 2.0];
    (0..nb)
        .map(|_| {
            if tied {
                LEVELS[(random.next_u64() % LEVELS.len() as u64) as usize]
            } else {
                random.gaussian() * 3.0
            }
        })
        .collect()
}

/// Runs the body's selection of a chunk (the listing rows, then the scored ones) and returns the ids and counts.
fn run_select(
    rig: &Rig,
    start: usize,
    rows: usize,
    budget: usize,
    stride: usize,
    scores: &[Vec<f32>],
) -> (Vec<i32>, Vec<i32>) {
    let first = (4 * budget + 3).saturating_sub(start).min(rows);
    let mut flat = vec![0.0f32; (rows - first).max(1) * stride];
    for (row, scores) in scores.iter().enumerate() {
        flat[row * stride..][..scores.len()].copy_from_slice(scores);
    }
    let scores_d = upload(&rig.library, &f32_bytes(&flat));
    let ids_d = upload(&rig.library, &i32_bytes(&vec![-1; rows * budget]));
    let counts_d = upload(&rig.library, &i32_bytes(&vec![-1; rows]));
    let mut launches = Vec::new();
    if first > 0 {
        launches.push(
            select(
                scores_d.ptr(),
                ids_d.ptr(),
                counts_d.ptr(),
                0,
                first as u32,
                start as u32,
                0,
                budget as u32,
            )
            .unwrap(),
        );
    }
    if first < rows {
        launches.push(
            select(
                scores_d.ptr(),
                ids_d.ptr(),
                counts_d.ptr(),
                first as u32,
                (rows - first) as u32,
                start as u32,
                stride as u32,
                budget as u32,
            )
            .unwrap(),
        );
    }
    launch(rig, launches);
    (read_i32(&ids_d, rows * budget), read_i32(&counts_d, rows))
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn selection_keeps_the_highest_scores_and_the_lowest_ids_among_ties_at_decode_and_prefill_rows() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(74);
    // (start, rows, budget, stride): a decode row, a prefill chunk whose first rows list every block, and a long row
    // whose 250 blocks need all four radix passes.
    for (start, rows, budget, stride) in [(39usize, 1usize, 4usize, 32usize), (0, 24, 4, 32), (1000, 3, 64, 256)] {
        for tied in [true, false] {
            let first = (4 * budget + 3).saturating_sub(start).min(rows);
            let scores: Vec<Vec<f32>> = (first..rows)
                .map(|row| scores_row(&mut random, (start + row + 1) / 4, tied))
                .collect();
            let (ids, counts) = run_select(&rig, start, rows, budget, stride, &scores);
            for row in 0..rows {
                let nb = (start + row + 1) / 4;
                let wanted = if row < first {
                    (0..nb.min(budget) as i32).collect()
                } else {
                    host_select(&scores[row - first], budget)
                };
                let label = format!("start {start} row {row} tied {tied}");
                assert_eq!(counts[row] as usize, wanted.len(), "{label} count");
                assert_eq!(&ids[row * budget..][..wanted.len()], wanted.as_slice(), "{label} ids");
            }
        }
    }
    // Every score equal: the first `budget` blocks, in order.
    let scores = vec![vec![1.5f32; 40]];
    let (ids, counts) = run_select(&rig, 159, 1, 8, 64, &scores);
    assert_eq!((counts[0], &ids[..8]), (8, [0, 1, 2, 3, 4, 5, 6, 7].as_slice()));
}

/// The 144-byte rows of `tokens` positions of one plane, decoded to the rotated values the kernel multiplies.
fn decode_plane(page: &[u8], tokens: usize) -> Vec<Vec<f32>> {
    let mut rows = Vec::new();
    for token in 0..tokens {
        for head in 0..KEY_HEADS {
            let row = &page[(token * KEY_HEADS + head) * PAGE_ROW_BYTES..][..PAGE_ROW_BYTES];
            let values = (0..HEAD)
                .map(|index| {
                    let byte = row[index / 2];
                    let code = if index % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let magnitude = E2M1[usize::from(code & 7)] * e4m3(row[128 + index / 16]);
                    if code & 8 == 0 { magnitude } else { -magnitude }
                })
                .collect();
            rows.push(values);
        }
    }
    rows
}

struct Fixture {
    queries: Vec<f32>,
    gate: Vec<f32>,
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// One attention call at `start` over `rows` rows, with the cache holding every position up to the last row.
/// `ids` is the per-row selected blocks (`None`: every block). Returns the device's gated output and the host's.
fn attend(
    rig: &Rig,
    random: &mut Random,
    start: usize,
    rows: usize,
    budget: usize,
    ids: Option<&[Vec<i32>]>,
    splits: usize,
) {
    let end = start + rows;
    let fixture = Fixture {
        queries: gaussians(random, rows * QUERY_ROW, 1.0),
        gate: gaussians(random, rows * QUERY_ROW, 1.0),
        keys: gaussians(random, end * KEY_HEADS * HEAD, 1.0),
        values: gaussians(random, end * KEY_HEADS * HEAD, 1.0),
    };
    // The query and gate share `[q | gate]` rows.
    let mut query_rows = vec![0.0f32; rows * QUERY_ROW];
    for row in 0..rows {
        for head in 0..QUERY_HEADS {
            let at = row * QUERY_ROW + head * QUERY_HEAD_STRIDE;
            query_rows[at..at + HEAD].copy_from_slice(&fixture.queries[(row * QUERY_HEADS + head) * HEAD..][..HEAD]);
            query_rows[at + HEAD..at + 2 * HEAD]
                .copy_from_slice(&fixture.gate[(row * QUERY_HEADS + head) * HEAD..][..HEAD]);
        }
    }
    let queries_d = on_device(rig, &query_rows);
    let (keys_d, values_d) = (on_device(rig, &fixture.keys), on_device(rig, &fixture.values));
    let page_bytes = 256 * KEY_HEADS * PAGE_ROW_BYTES;
    assert!(end <= 256, "one page");
    let (key_page, value_page) = (
        upload(&rig.library, &vec![0u8; page_bytes]),
        upload(&rig.library, &vec![0u8; page_bytes]),
    );
    let key_table = upload(&rig.library, &key_page.ptr().get().to_le_bytes());
    let value_table = upload(&rig.library, &value_page.ptr().get().to_le_bytes());
    launch(
        rig,
        vec![
            kv_append(
                keys_d.ptr(),
                values_d.ptr(),
                key_table.ptr(),
                value_table.ptr(),
                end as u32,
                KEY_HEADS as u32,
                (KEY_HEADS * HEAD) as u32,
                (KEY_HEADS * HEAD) as u32,
                0,
            )
            .unwrap(),
        ],
    );
    let mut host_keys = vec![0u8; page_bytes];
    let mut host_values = vec![0u8; page_bytes];
    key_page.copy_to_host(0, &mut host_keys).unwrap();
    value_page.copy_to_host(0, &mut host_values).unwrap();
    let (rotated_keys, rotated_values) = (decode_plane(&host_keys, end), decode_plane(&host_values, end));
    // The append is the codec of the cache: the decoded rows, rotated back, are the BF16 rows to the codec's accuracy.
    let mut back = Vec::new();
    for row in &rotated_keys {
        let mut row = row.clone();
        hadamard(&mut row);
        back.extend(row);
    }
    assert!(
        relative_rms(&fixture.keys, &back) < 0.2,
        "the cached keys are the appended keys"
    );

    let ids_flat: Vec<i32> = ids.map_or_else(Vec::new, |ids| {
        ids.iter()
            .flat_map(|row| row.iter().copied().chain(repeat_n(-1, budget - row.len())))
            .collect()
    });
    let counts: Vec<i32> = ids.map_or_else(Vec::new, |ids| ids.iter().map(|row| row.len() as i32).collect());
    let ids_d = upload(&rig.library, &i32_bytes(&ids_flat));
    let counts_d = upload(&rig.library, &i32_bytes(&counts));
    let heads_total = rows * QUERY_HEADS;
    let partial_d = upload(&rig.library, &vec![0u8; heads_total * splits * 258 * 4]);
    let gated_d = upload(&rig.library, &vec![0u8; rows * QUERY_HEADS * HEAD * 2]);
    let gate = queries_d.ptr().offset((HEAD * 2) as u64);
    let mut launches = vec![
        Attention {
            queries: queries_d.ptr(),
            gate,
            key_pages: key_table.ptr(),
            value_pages: value_table.ptr(),
            ids: if ids.is_some() { ids_d.ptr() } else { DevicePtr::NULL },
            counts: if ids.is_some() { counts_d.ptr() } else { DevicePtr::NULL },
            partial: if splits > 1 { partial_d.ptr() } else { DevicePtr::NULL },
            core: DevicePtr::NULL,
            gated: gated_d.ptr(),
            rows: rows as u32,
            query_heads: QUERY_HEADS as u32,
            key_heads: KEY_HEADS as u32,
            start: start as u32,
            splits: splits as u32,
            budget: budget as u32,
            query_row_stride: QUERY_ROW as u32,
            query_head_stride: QUERY_HEAD_STRIDE as u32,
            gate_row_stride: QUERY_ROW as u32,
            gate_head_stride: QUERY_HEAD_STRIDE as u32,
        }
        .launch()
        .unwrap(),
    ];
    if splits > 1 {
        launches.push(
            merge(
                partial_d.ptr(),
                gate,
                DevicePtr::NULL,
                gated_d.ptr(),
                rows as u32,
                QUERY_HEADS as u32,
                splits as u32,
                QUERY_ROW as u32,
                QUERY_HEAD_STRIDE as u32,
            )
            .unwrap(),
        );
    }
    launch(rig, launches);
    let got = read(&gated_d, rows * QUERY_HEADS * HEAD);

    let group = QUERY_HEADS / KEY_HEADS;
    for row in 0..rows {
        let position = start + row;
        let nb = (position + 1) / 4;
        let blocks: Vec<usize> = ids.map_or_else(
            || (0..nb).collect(),
            |ids| ids[row].iter().map(|id| *id as usize).collect(),
        );
        let tokens: Vec<usize> = blocks
            .iter()
            .flat_map(|block| 4 * block..4 * block + 4)
            .chain(4 * nb..=position)
            .collect();
        for head in 0..QUERY_HEADS {
            let kv = head / group;
            let mut query = fixture.queries[(row * QUERY_HEADS + head) * HEAD..][..HEAD].to_vec();
            hadamard(&mut query);
            let logits: Vec<f32> = tokens
                .iter()
                .map(|token| {
                    let key = &rotated_keys[token * KEY_HEADS + kv];
                    query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>() * 0.0625
                })
                .collect();
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = logits.iter().map(|logit| (logit - maximum).exp()).collect();
            let total: f32 = weights.iter().sum();
            let mut out = vec![0.0f32; HEAD];
            for (token, weight) in tokens.iter().zip(&weights) {
                for (out, value) in out.iter_mut().zip(&rotated_values[token * KEY_HEADS + kv]) {
                    *out += weight / total * value;
                }
            }
            hadamard(&mut out);
            let wanted: Vec<f32> = out
                .iter()
                .zip(&fixture.gate[(row * QUERY_HEADS + head) * HEAD..][..HEAD])
                .map(|(core, gate)| round(round(*core) * round(sigmoid(*gate))))
                .collect();
            let at = (row * QUERY_HEADS + head) * HEAD;
            assert_close(
                &wanted,
                &got[at..at + HEAD],
                HEAD,
                0.04,
                &format!("start {start} rows {rows} splits {splits} row {row} head {head}"),
            );
        }
    }
}

/// Random selections of `min(budget, nb)` blocks per row, ascending.
fn random_ids(random: &mut Random, start: usize, rows: usize, budget: usize) -> Vec<Vec<i32>> {
    (0..rows)
        .map(|row| {
            let nb = (start + row + 1) / 4;
            let mut pool: Vec<i32> = (0..nb as i32).collect();
            for at in 0..pool.len().min(budget) {
                let pick = at + (random.next_u64() % (pool.len() - at) as u64) as usize;
                pool.swap(at, pick);
            }
            let mut kept = pool[..pool.len().min(budget)].to_vec();
            kept.sort_unstable();
            kept
        })
        .collect()
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn attention_over_the_selected_blocks_and_the_tail_matches_the_host_reference_for_a_decode_row() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(75);
    // Position 41 sees 10 blocks and keeps 4, plus a tail of two: 18 keys in 3 splits, the last one empty.
    let ids = random_ids(&mut random, 41, 1, 4);
    attend(&rig, &mut random, 41, 1, 4, Some(&ids), 3);
    // Two splits that both hold keys, and one split.
    let ids = random_ids(&mut random, 100, 1, 8);
    attend(&rig, &mut random, 100, 1, 8, Some(&ids), 2);
    attend(&rig, &mut random, 100, 1, 8, Some(&ids), 1);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn attention_over_the_selected_blocks_and_the_tail_matches_the_host_reference_for_a_prefill_chunk() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(76);
    // Rows 30..41: the first see at most the budget's blocks and list them, the later ones choose.
    let ids = random_ids(&mut random, 30, 12, 8);
    attend(&rig, &mut random, 30, 12, 8, Some(&ids), 1);
    // The same rows in two splits through the merge.
    attend(&rig, &mut random, 30, 12, 8, Some(&ids), 2);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn attention_without_ids_is_the_dense_causal_prefix() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(77);
    attend(&rig, &mut random, 3, 5, 4, None, 1);
    attend(&rig, &mut random, 0, 70, 4, None, 2);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn appended_rows_land_at_their_positions_in_every_page() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(78);
    // Rows 250..260 cross the first page's end into the second; the table holds both pages.
    let rows = 10usize;
    let keys = gaussians(&mut random, rows * KEY_HEADS * HEAD, 1.0);
    let values = gaussians(&mut random, rows * KEY_HEADS * HEAD, 1.0);
    let (keys_d, values_d) = (on_device(&rig, &keys), on_device(&rig, &values));
    let page_bytes = 256 * KEY_HEADS * PAGE_ROW_BYTES;
    let pages: Vec<DeviceBuffer> = (0..4).map(|_| upload(&rig.library, &vec![0u8; page_bytes])).collect();
    let table = |planes: &[DeviceBuffer]| {
        let bytes: Vec<u8> = planes.iter().flat_map(|page| page.ptr().get().to_le_bytes()).collect();
        upload(&rig.library, &bytes)
    };
    let (key_table, value_table) = (table(&pages[..2]), table(&pages[2..]));
    launch(
        &rig,
        vec![
            kv_append(
                keys_d.ptr(),
                values_d.ptr(),
                key_table.ptr(),
                value_table.ptr(),
                rows as u32,
                KEY_HEADS as u32,
                (KEY_HEADS * HEAD) as u32,
                (KEY_HEADS * HEAD) as u32,
                250,
            )
            .unwrap(),
        ],
    );
    let mut first = vec![0u8; page_bytes];
    let mut second = vec![0u8; page_bytes];
    pages[0].copy_to_host(0, &mut first).unwrap();
    pages[1].copy_to_host(0, &mut second).unwrap();
    let rotated_first = decode_plane(&first, 256);
    let rotated_second = decode_plane(&second, 4);
    for row in 0..rows {
        let position = 250 + row;
        for head in 0..KEY_HEADS {
            let mut decoded = if position < 256 {
                rotated_first[position * KEY_HEADS + head].clone()
            } else {
                rotated_second[(position - 256) * KEY_HEADS + head].clone()
            };
            hadamard(&mut decoded);
            let wanted = &keys[(row * KEY_HEADS + head) * HEAD..][..HEAD];
            assert!(relative_rms(wanted, &decoded) < 0.2, "position {position} head {head}");
        }
    }
    // Position 249 and the slots after the last row stay as they were.
    assert!(
        first[..250 * KEY_HEADS * PAGE_ROW_BYTES].iter().all(|byte| *byte == 0),
        "before the chunk"
    );
}
