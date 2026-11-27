//! The mixture-of-experts launches of Flash-Next on a GPU, each against a host reference at a small synthetic shape:
//! the router with its readback, the expert stages (gate and up with SwiGLU, down, combine) over a description the
//! host planner wrote, and the shared expert's padded SwiGLU with the finish and the zero fill of a block without
//! routed pairs. The expert records are NVFP4 tensors built here, so the reference dequantizes the codes itself.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_inference_cuda::{DeviceBuffer, PinnedPool, PooledPinned};
use euhedral_inference_engine::flash::moe::ops::{
    ExpertGeometry, ExpertItems, expert_combine, expert_down, expert_gate_up, moe_finish, router, swiglu, swiglu_padded,
};
use euhedral_inference_engine::flash::moe::routing::{DescriptorLayout, Routing};
use euhedral_inference_engine::flash::sequence::Op;
use euhedral_inference_engine::formats::shapes::Launch;

use super::{E2M1, e4m3, gaussians, on_device, queue, read, round, sigmoid, silu};
use crate::common;
use crate::linear_gpu::{Random, Rig, assert_close, bf16, empty, rig, upload};

const HIDDEN: u32 = 128;
const INTER: u32 = 128;
const EXPERTS: u32 = 4;
const TOP_K: u32 = 2;
const ROWS: u32 = 3;
/// The block scale codes the tensors draw from: E4M3 exponents 6 and 7, so scales between 0.5 and 2.
const SCALE_CODES: Range<u8> = 0x30..0x40;
const GLOBAL: f32 = 0.03125;

fn launches(rig: &Rig, launches: Vec<Launch>) {
    let ops: Vec<Op<'_>> = launches.into_iter().map(Op::Launch).collect();
    queue(rig, &ops);
}

fn align256(bytes: usize) -> usize {
    bytes.next_multiple_of(256)
}

/// A plain NVFP4 tensor of `rows` rows of `k` values (`k` a multiple of 128) with random codes and scales: its bytes,
/// then its values as the kernel reads them before the global scale.
fn tensor(random: &mut Random, rows: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let scales_at = align256(rows * k / 2);
    let global_at = align256(scales_at + rows * k / 16);
    let mut bytes = vec![0u8; align256(global_at + 4)];
    let mut values = vec![0.0f32; rows * k];
    for row in 0..rows {
        for block in 0..k / 16 {
            let scale = SCALE_CODES.start + (random.next_u64() % u64::from(SCALE_CODES.len() as u8)) as u8;
            bytes[scales_at + row * k / 16 + block] = scale;
            for at in block * 16..block * 16 + 16 {
                let nibble = (random.next_u64() % 16) as u8;
                let byte = &mut bytes[row * k / 2 + at / 2];
                *byte |= nibble << (4 * (at % 2));
                let sign = if nibble & 8 == 0 { 1.0 } else { -1.0 };
                values[row * k + at] = sign * E2M1[usize::from(nibble & 7)] * e4m3(scale);
            }
        }
    }
    bytes[global_at..global_at + 4].copy_from_slice(&GLOBAL.to_le_bytes());
    (bytes, values)
}

/// `rows` rows of `values` (`k` wide) times the rows of `weights`, BF16-rounded after the global scale.
fn project(weights: &[f32], x: &[f32], n: usize, k: usize) -> Vec<f32> {
    (0..n)
        .map(|row| {
            let sum: f32 = (0..k).map(|at| weights[row * k + at] * x[at]).sum();
            round(sum * GLOBAL)
        })
        .collect()
}

fn pinned_words(pool: &Arc<PinnedPool>, bytes: usize) -> PooledPinned {
    pool.acquire(bytes).unwrap()
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn router_picks_the_top_experts_and_reads_them_back() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (rows, experts, k) = (3u32, 16u32, 4u32);
    let mut random = Random::new(21);
    let mut logits = gaussians(&mut random, (rows * experts) as usize, 2.0);
    // The last row is all ties: the lowest experts win and share the weight equally.
    logits[(2 * experts) as usize..].fill(0.5);
    let logits_device = on_device(&rig, &logits);
    let ids_device = empty(&rig.library, (rows * k) as usize * 4);
    let weights_device = empty(&rig.library, (rows * k) as usize * 2);
    let pool = PinnedPool::new(&rig.library, 1 << 20);
    let ids_host = pinned_words(&pool, (rows * k) as usize * 4);
    let weights_host = pinned_words(&pool, ((rows * k) as usize * 2).next_multiple_of(4));
    let launch = router(
        logits_device.ptr(),
        ids_device.ptr(),
        weights_device.ptr(),
        rows,
        experts,
        k,
    )
    .unwrap();
    queue(
        &rig,
        &[
            Op::Launch(launch),
            Op::Readback {
                destination: ids_host.buffer(),
                offset: 0,
                source: ids_device.ptr(),
                bytes: (rows * k) as usize * 4,
            },
            Op::Readback {
                destination: weights_host.buffer(),
                offset: 0,
                source: weights_device.ptr(),
                bytes: (rows * k) as usize * 2,
            },
        ],
    );
    for row in 0..rows as usize {
        let row_logits = &logits[row * experts as usize..][..experts as usize];
        let top = row_logits.iter().copied().fold(f32::MIN, f32::max);
        let exponentials: Vec<f32> = row_logits.iter().map(|value| (value - top).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        let probabilities: Vec<f32> = exponentials.iter().map(|value| value / total).collect();
        let mut order: Vec<usize> = (0..experts as usize).collect();
        order.sort_by(|a, b| probabilities[*b].total_cmp(&probabilities[*a]).then(a.cmp(b)));
        let picked = &order[..k as usize];
        let sum: f32 = picked.iter().map(|id| probabilities[*id]).sum();
        for (slot, id) in picked.iter().enumerate() {
            let at = row * k as usize + slot;
            let found = ids_host.buffer().word(at).load(Ordering::Relaxed) as i32;
            assert_eq!(found, *id as i32, "row {row} pick {slot}");
            let word = weights_host.buffer().word(at / 2).load(Ordering::Relaxed);
            let bits = (word >> (16 * (at % 2))) as u16;
            let wanted = round(probabilities[*id] / sum);
            let found = f32::from_bits(u32::from(bits) << 16);
            assert!(
                (found - wanted).abs() <= 0.01 * wanted,
                "row {row} weight {slot}: {found} against {wanted}"
            );
        }
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn expert_stages_over_a_planned_description_match_the_host_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (hidden, inter) = (HIDDEN as usize, INTER as usize);
    let mut random = Random::new(33);
    // The records: gate_up [2 inter][hidden], then down [hidden][inter], each at its 256-aligned offset.
    let mut records = Vec::new();
    let mut dequantized = Vec::new();
    let (mut gate_up_offset, mut down_offset, mut record_bytes) = (0, 0, 0);
    for _ in 0..EXPERTS {
        let (gate_up, gate_up_values) = tensor(&mut random, 2 * inter, hidden);
        let (down, down_values) = tensor(&mut random, hidden, inter);
        (gate_up_offset, down_offset) = (0, gate_up.len());
        record_bytes = gate_up.len() + down.len();
        records.extend_from_slice(&gate_up);
        records.extend_from_slice(&down);
        dequantized.push((gate_up_values, down_values));
    }
    let geometry = ExpertGeometry::new(HIDDEN, INTER, gate_up_offset as u64, down_offset as u64).unwrap();
    let experts_device = upload(&rig.library, &records);
    let x = gaussians(&mut random, ROWS as usize * hidden, 1.0);
    let x_device = on_device(&rig, &x);

    // The router's answer, planned by the host: row 0 takes experts 2 and 0, row 1 takes 2 and 3, row 2 takes 0 and 2.
    let ids = [2i32, 0, 2, 3, 0, 2];
    let weights = [0.625f32, 0.375, 0.5, 0.5, 0.75, 0.25];
    let layout = DescriptorLayout::new(EXPERTS, TOP_K, ROWS).unwrap();
    let mut routing = Routing::new(layout);
    routing.plan(ROWS, |i| ids[i], |i| bf16(weights[i])).unwrap();
    let pool = PinnedPool::new(&rig.library, 1 << 20);
    let staged = pinned_words(&pool, layout.bytes());
    routing.fill(&mut |word, value| staged.buffer().word(word).store(value, Ordering::Relaxed));
    for index in 0..routing.active_experts() {
        let expert = routing.active_expert(index).unwrap() as usize;
        let address = experts_device.ptr().get() + (expert * record_bytes) as u64;
        for (word, value) in layout.slot_words(index, address) {
            staged.buffer().word(word).store(value, Ordering::Relaxed);
        }
    }
    let description = empty(&rig.library, layout.bytes());
    let pairs = routing.pair_count();
    let act = empty(&rig.library, pairs as usize * inter * 2);
    let weighted = empty(&rig.library, pairs as usize * hidden * 2);
    let routed = upload(&rig.library, &vec![0xffu8; ROWS as usize * hidden * 2]);
    let at = |offset: usize| description.ptr().offset(offset as u64);

    queue(
        &rig,
        &[Op::Upload {
            destination: description.ptr(),
            source: staged.buffer(),
            offset: 0,
            bytes: layout.bytes(),
        }],
    );
    for index in 0..routing.active_experts() {
        let first = routing.item_start(index).unwrap() as usize;
        let items = ExpertItems {
            slots: at(layout.slots_offset()),
            items: at(layout.items_offset() + 16 * first),
            item_count: routing.item_count(index).unwrap(),
            pairs: at(layout.pairs_offset()),
        };
        launches(
            &rig,
            vec![
                expert_gate_up(&geometry, &items, x_device.ptr(), act.ptr()).unwrap(),
                expert_down(&geometry, &items, act.ptr(), weighted.ptr()).unwrap(),
            ],
        );
    }
    launches(
        &rig,
        vec![
            expert_combine(
                &geometry,
                weighted.ptr(),
                at(layout.row_offsets_offset()),
                at(layout.row_pairs_offset()),
                routed.ptr(),
                ROWS,
            )
            .unwrap(),
        ],
    );

    // The reference: every (row, expert) pair, added into its row in expert order from zero.
    let mut wanted = vec![0.0f32; ROWS as usize * hidden];
    for (expert, (gate_up, down)) in dequantized.iter().enumerate() {
        for row in 0..ROWS as usize {
            let Some(slot) = (0..TOP_K as usize).find(|slot| ids[row * 2 + slot] == expert as i32) else {
                continue;
            };
            let input = &x[row * hidden..][..hidden];
            let projected = project(gate_up, input, 2 * inter, hidden);
            let hidden_act: Vec<f32> = (0..inter)
                .map(|j| round(round(silu(projected[j])) * projected[inter + j]))
                .collect();
            let y = project(down, &hidden_act, hidden, inter);
            let weight = round(weights[row * 2 + slot]);
            for (column, value) in y.iter().enumerate() {
                let at = row * hidden + column;
                wanted[at] = round(wanted[at] + round(value * weight));
            }
        }
    }
    assert_close(&wanted, &read(&routed, ROWS as usize * hidden), hidden, 3e-2, "routed");
}

fn finished(
    rig: &Rig,
    routed: &DeviceBuffer,
    shared: &DeviceBuffer,
    gate: &DeviceBuffer,
    rows: u32,
    width: u32,
) -> Vec<f32> {
    let output = empty(&rig.library, (rows * width) as usize * 2);
    launches(
        rig,
        vec![moe_finish(routed.ptr(), shared.ptr(), gate.ptr(), output.ptr(), rows, width).unwrap()],
    );
    read(&output, (rows * width) as usize)
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn shared_swiglu_and_finish_match_the_host_reference_with_and_without_routed_pairs() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (rows, width, cols, padded) = (3u32, 128u32, 192u32, 256u32);
    let mut random = Random::new(45);
    let gate_values = gaussians(&mut random, (rows * cols) as usize, 1.5);
    let up_values = gaussians(&mut random, (rows * cols) as usize, 1.5);
    let gate = on_device(&rig, &gate_values);
    let up = on_device(&rig, &up_values);
    let padded_act = upload(&rig.library, &vec![0xffu8; (rows * padded) as usize * 2]);
    let plain_act = empty(&rig.library, (rows * cols) as usize * 2);
    launches(
        &rig,
        vec![
            swiglu_padded(gate.ptr(), up.ptr(), padded_act.ptr(), rows, cols, padded).unwrap(),
            swiglu(gate.ptr(), up.ptr(), plain_act.ptr(), rows * cols).unwrap(),
        ],
    );
    let wanted: Vec<f32> = gate_values
        .iter()
        .zip(&up_values)
        .map(|(gate, up)| round(round(silu(*gate)) * up))
        .collect();
    assert_close(&wanted, &read(&plain_act, wanted.len()), cols as usize, 1e-2, "swiglu");
    let stored = read(&padded_act, (rows * padded) as usize);
    for row in 0..rows as usize {
        let at = row * padded as usize;
        assert_close(
            &wanted[row * cols as usize..][..cols as usize],
            &stored[at..at + cols as usize],
            cols as usize,
            1e-2,
            "padded swiglu",
        );
        assert!(
            stored[at + cols as usize..at + padded as usize]
                .iter()
                .all(|value| *value == 0.0),
            "row {row} pad"
        );
    }

    let shared_values = gaussians(&mut random, (rows * width) as usize, 1.0);
    let routed_values = gaussians(&mut random, (rows * width) as usize, 1.0);
    let gate_raw = gaussians(&mut random, rows as usize, 2.0);
    let shared = on_device(&rig, &shared_values);
    let gate_raw_device = on_device(&rig, &gate_raw);
    let expected = |routed: &[f32]| -> Vec<f32> {
        (0..(rows * width) as usize)
            .map(|at| {
                let scale = round(sigmoid(gate_raw[at / width as usize]));
                round(routed[at] + round(scale * shared_values[at]))
            })
            .collect()
    };
    // A block with routed pairs adds the shared expert to the combined output.
    let routed = on_device(&rig, &routed_values);
    assert_close(
        &expected(&routed_values),
        &finished(&rig, &routed, &shared, &gate_raw_device, rows, width),
        width as usize,
        1e-2,
        "finish",
    );
    // A block without pairs zeroes stale scratch first, so only the shared part is left.
    let stale = upload(&rig.library, &vec![0xffu8; (rows * width) as usize * 2]);
    queue(
        &rig,
        &[Op::Zero {
            address: stale.ptr(),
            bytes: u64::from(rows * width) * 2,
        }],
    );
    assert_close(
        &expected(&vec![0.0; (rows * width) as usize]),
        &finished(&rig, &stale, &shared, &gate_raw_device, rows, width),
        width as usize,
        1e-2,
        "finish without pairs",
    );
}
