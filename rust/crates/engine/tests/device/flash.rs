//! The Flash-Next dense launches on a GPU, each against a host reference at a small synthetic shape. The tests are
//! `#[ignore]`d and run in the device suite: `cargo test -p euhedral-inference-engine --test device -- --ignored flash::`.
//!
//! The library comes from the `euhedral.cuda.library` property; a test skips with a message when it is absent.
//! The references follow the kernels' own roundings (BF16 after every step the kernel rounds), so the tolerances only
//! cover the order of the sums. The NVFP4 projections and the gated delta rule are shared kernels, covered by the
//! format and GDN suites; here the launches that are new to the Flash-Next table run.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test copies device memory and synchronizes"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "small synthetic shapes"
)]

use std::sync::Arc;

use euhedral_inference_cuda::{DeviceBuffer, DevicePtr, KernelOps, PinnedPool};
use euhedral_inference_edrl::{NgramLayout, TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::flash::Weight;
use euhedral_inference_engine::flash::buffers::{FlashRegions, ScratchSizes, StreamShape};
use euhedral_inference_engine::flash::embed::EmbedStreams;
use euhedral_inference_engine::flash::hyper::{HyperShape, HyperWeights};
use euhedral_inference_engine::flash::linear::{Expansion, Nvfp4Device, Nvfp4Weight};
use euhedral_inference_engine::flash::ops::{
    conv_history, gdn_control, gdn_conv, gdn_gated_norm, ngram_expand, ple_conv, ple_gate,
};
use euhedral_inference_engine::flash::ple::{
    NgramIds, NgramRows, PARTS, PleGatherStage, PleIdsStage, PleRows, ShardBytes,
};
use euhedral_inference_engine::flash::sequence::{Op, queue_all};
use euhedral_inference_engine::formats::{LinearMode, Nvfp4Linear};
use euhedral_inference_engine::model::{Residence, WeightRef};
use linear_gpu::{Random, Rig, assert_close, bf16, bf16_value, empty, rig, upload};

use crate::common;
use crate::linear_gpu;

fn round(value: f32) -> f32 {
    bf16_value(bf16(value))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

/// `count` Gaussian values scaled by `scale`, already BF16-exact.
fn gaussians(random: &mut Random, count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|_| round(random.gaussian() * scale)).collect()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| bf16(*value).to_le_bytes()).collect()
}

fn on_device(rig: &Rig, values: &[f32]) -> DeviceBuffer {
    upload(&rig.library, &bf16_bytes(values))
}

fn read(buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
    let mut host = vec![0u8; count * 2];
    buffer.copy_to_host(0, &mut host).unwrap();
    host.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_value(u16::from_le_bytes(*pair)))
        .collect()
}

fn read_f32(buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
    let mut host = vec![0u8; count * 4];
    buffer.copy_to_host(0, &mut host).unwrap();
    host.as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn queue(rig: &Rig, ops: &[Op<'_>]) {
    rig.run(|lane| queue_all(ops, lane as &mut dyn KernelOps));
}

fn weight(buffer: &DeviceBuffer, bytes: usize) -> Weight {
    Weight {
        address: buffer.ptr(),
        bytes: bytes as u64,
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn embedding_repeats_each_row_into_every_stream() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (vocabulary, hidden, streams, rows) = (32u32, 64u32, 3u32, 4u32);
    let mut random = Random::new(11);
    let table = gaussians(&mut random, (vocabulary * hidden) as usize, 1.0);
    let table_device = on_device(&rig, &table);
    let ids = [5u32, 0, 31, 5];
    let ids_device = upload(
        &rig.library,
        &ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<u8>>(),
    );
    let shape = StreamShape::new(hidden, streams).unwrap();
    let sizes = ScratchSizes {
        hyper: 256,
        layer: 256,
        expansion: 0,
    };
    let total = FlashRegions::bytes(rows, shape, sizes) as usize;
    let arena = empty(&rig.library, total);
    let regions = FlashRegions::over(arena.ptr(), rows, shape, sizes).unwrap();
    let state = regions.state;
    let embed = EmbedStreams::new(weight(&table_device, table.len() * 2), vocabulary, shape, regions).unwrap();
    queue(&rig, &embed.ops(ids_device.ptr(), rows).unwrap());
    let streamed = read_ptr(&arena, state, (rows * streams * hidden) as usize);
    for (row, id) in ids.iter().enumerate() {
        let wanted = &table[*id as usize * hidden as usize..][..hidden as usize];
        for stream in 0..streams as usize {
            let at = (row * streams as usize + stream) * hidden as usize;
            assert_eq!(&streamed[at..at + hidden as usize], wanted, "row {row} stream {stream}");
        }
    }
}

/// Reads `count` BF16 values at `ptr`, which lies inside `arena`.
fn read_ptr(arena: &DeviceBuffer, ptr: DevicePtr, count: usize) -> Vec<f32> {
    let offset = (ptr.get() - arena.ptr().get()) as usize;
    let mut host = vec![0u8; count * 2];
    arena.copy_to_host(offset, &mut host).unwrap();
    host.as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16_value(u16::from_le_bytes(*pair)))
        .collect()
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn hyper_mix_and_inject_match_the_host_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (streams, hidden, lowrank, rows) = (2usize, 64usize, 8usize, 3usize);
    let epsilon = 1e-6f32;
    let width = streams * hidden;
    let shape = HyperShape::new(streams as u32, hidden as u32, lowrank as u32, epsilon).unwrap();
    let mut random = Random::new(23);
    let state = gaussians(&mut random, rows * width, 1.0);
    let norm = gaussians(&mut random, width, 0.1);
    let down = gaussians(&mut random, lowrank * width, 0.2);
    let up = gaussians(&mut random, width * lowrank, 0.5);
    let inject = gaussians(&mut random, streams * width, 0.2);
    let block = gaussians(&mut random, rows * hidden, 1.0);

    let (state_d, norm_d, down_d, up_d, inject_d, block_d) = (
        on_device(&rig, &state),
        on_device(&rig, &norm),
        on_device(&rig, &down),
        on_device(&rig, &up),
        on_device(&rig, &inject),
        on_device(&rig, &block),
    );
    let scratch_d = empty(&rig.library, shape.scratch_bytes(rows as u32) as usize);
    let mixed_d = empty(&rig.library, rows * hidden * 2);
    let injected_d = empty(&rig.library, rows * width * 2);
    let weights = HyperWeights {
        norm: weight(&norm_d, norm.len() * 2),
        down: weight(&down_d, down.len() * 2),
        up: weight(&up_d, up.len() * 2),
        inject: Some(weight(&inject_d, inject.len() * 2)),
    };
    let scratch = shape.scratch(scratch_d.ptr(), rows as u32);
    let mut list = shape
        .mix(&weights, state_d.ptr(), &scratch, mixed_d.ptr(), rows as u32, false)
        .unwrap();
    list.push(
        shape
            .inject(state_d.ptr(), block_d.ptr(), &scratch, injected_d.ptr(), rows as u32)
            .unwrap(),
    );
    queue(&rig, &list);

    // The reference.
    let linear = |input: &[f32], weights: &[f32], k: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|j| round((0..k).map(|i| input[i] * weights[j * k + i]).sum()))
            .collect()
    };
    let (mut wanted_mixed, mut wanted_injected) = (Vec::new(), Vec::new());
    for row in 0..rows {
        let x = &state[row * width..(row + 1) * width];
        let mut normed = vec![0.0f32; width];
        for stream in 0..streams {
            let span = stream * hidden..(stream + 1) * hidden;
            let mean = x[span.clone()].iter().map(|v| v * v).sum::<f32>() / hidden as f32;
            let inverse = 1.0 / (mean + epsilon).sqrt();
            for d in span {
                normed[d] = round(x[d] * inverse * (1.0 + norm[d]));
            }
        }
        let low = linear(&normed, &down, width, lowrank);
        let activated: Vec<f32> = low.iter().map(|v| round(silu(round(v / streams as f32)))).collect();
        let gates = linear(&activated, &up, lowrank, width);
        for d in 0..hidden {
            let sum: f32 = (0..streams)
                .map(|s| round(round(sigmoid(gates[s * hidden + d])) * normed[s * hidden + d]))
                .sum();
            wanted_mixed.push(round(sum / streams as f32));
        }
        let raw = linear(&normed, &inject, width, streams);
        for stream in 0..streams {
            let weight = 2.0 * round(sigmoid(round(raw[stream] / streams as f32)));
            for d in 0..hidden {
                let value = block[row * hidden + d];
                wanted_injected.push(round(x[stream * hidden + d] + round(value * weight)));
            }
        }
    }
    assert_close(&wanted_mixed, &read(&mixed_d, rows * hidden), hidden, 0.05, "mix");
    assert_close(
        &wanted_injected,
        &read(&injected_d, rows * width),
        hidden,
        0.05,
        "inject",
    );
}

/// E2M1 magnitudes by code.
const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

fn e4m3(code: u8) -> f32 {
    let (exponent, mantissa) = (i32::from((code >> 3) & 15), f32::from(code & 7));
    if exponent == 0 {
        mantissa * 0.001_953_125
    } else {
        (1.0 + mantissa * 0.125) * 2f32.powi(exponent - 7)
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn ple_expand_gate_and_conv_match_the_host_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (rows, streams, width, taps, dilation) = (3usize, 2usize, 32usize, 3usize, 2usize);
    let channels = width;
    let history_rows = (taps - 1) * dilation;
    let mut random = Random::new(31);

    // A shard of four rows as the store lays it out (codes, then E4M3 block scales, per row; the global scale in the
    // trailer), gathered into records by the stage's own table: what the gather writes is what the kernel reads.
    let layout = NgramLayout::new(4, width as u64).unwrap();
    let row_bytes = width / 2 + width / 16;
    let global = 0.25 + random.uniform();
    let mut shard = vec![0u8; layout.byte_size() as usize];
    for row in 0..4 {
        let at = row * row_bytes;
        for byte in &mut shard[at..at + width / 2] {
            *byte = random.next_u64() as u8;
        }
        for scale in &mut shard[at + width / 2..at + row_bytes] {
            *scale = 0x30 + (random.next_u64() % 16) as u8;
        }
    }
    let trailer = layout.trailer_offset() as usize;
    shard[trailer..trailer + 4].copy_from_slice(&global.to_le_bytes());
    let source: Arc<dyn ShardBytes> = Arc::new(shard.clone());
    let table = NgramRows::new(vec![source], 4, 4, width as u32).unwrap();
    let record_bytes = table.record_bytes() as usize;
    let ids = [2u64, 0, 3];
    let mut records = vec![0u8; rows * record_bytes];
    table.gather(&ids, 0..rows, &mut records).unwrap();
    let mut expanded = Vec::new();
    for id in ids {
        let row = &shard[id as usize * row_bytes..(id as usize + 1) * row_bytes];
        for i in 0..width {
            let byte = row[i / 2];
            let nibble = if i % 2 == 1 { byte >> 4 } else { byte & 15 };
            let magnitude = E2M1[usize::from(nibble & 7)];
            let value = if nibble & 8 != 0 { -magnitude } else { magnitude };
            expanded.push(round(value * (e4m3(row[width / 2 + i / 16]) * global)));
        }
    }
    let records_d = upload(&rig.library, &records);
    let expanded_d = empty(&rig.library, rows * width * 2);
    queue(
        &rig,
        &[Op::Launch(
            ngram_expand(
                records_d.ptr(),
                expanded_d.ptr(),
                rows as u32,
                width as u32,
                record_bytes as u32,
            )
            .unwrap(),
        )],
    );
    assert_eq!(read(&expanded_d, rows * width), expanded, "ngram_expand");

    // The gate, over [row][stream][width] keys and queries and [row][width] values.
    let key = gaussians(&mut random, rows * streams * width, 1.0);
    let query = gaussians(&mut random, rows * streams * width, 1.0);
    let value = gaussians(&mut random, rows * width, 1.0);
    let (key_d, query_d, value_d) = (on_device(&rig, &key), on_device(&rig, &query), on_device(&rig, &value));
    let gated_d = empty(&rig.library, rows * streams * width * 2);
    queue(
        &rig,
        &[Op::Launch(
            ple_gate(
                key_d.ptr(),
                query_d.ptr(),
                value_d.ptr(),
                gated_d.ptr(),
                rows as u32,
                streams as u32,
                width as u32,
            )
            .unwrap(),
        )],
    );
    let mut wanted_gated = Vec::new();
    for pair in 0..rows * streams {
        let row = pair / streams;
        let dot = round(
            (0..width)
                .map(|d| round(key[pair * width + d] * query[pair * width + d]))
                .sum(),
        );
        let gate = round(dot * (1.0 / (width as f32).sqrt()));
        let magnitude = round(round(gate.abs().max(1e-6)).sqrt());
        let signed = if gate > 0.0 {
            magnitude
        } else if gate < 0.0 {
            -magnitude
        } else {
            0.0
        };
        let scale = round(sigmoid(signed));
        wanted_gated.extend((0..width).map(|d| round(scale * value[row * width + d])));
    }
    assert_close(
        &wanted_gated,
        &read(&gated_d, rows * streams * width),
        width,
        0.05,
        "ple_gate",
    );

    // The convolution with its history, then the history's update.
    let normed = gaussians(&mut random, rows * channels, 1.0);
    let gated = gaussians(&mut random, rows * channels, 1.0);
    let history = gaussians(&mut random, history_rows * channels, 1.0);
    let taps_w = gaussians(&mut random, channels * taps, 0.5);
    let (normed_d, gated_in_d, history_d, taps_d) = (
        on_device(&rig, &normed),
        on_device(&rig, &gated),
        on_device(&rig, &history),
        on_device(&rig, &taps_w),
    );
    let out_d = empty(&rig.library, rows * channels * 2);
    queue(
        &rig,
        &[
            Op::Launch(
                ple_conv(
                    normed_d.ptr(),
                    gated_in_d.ptr(),
                    history_d.ptr(),
                    taps_d.ptr(),
                    out_d.ptr(),
                    rows as u32,
                    channels as u32,
                    taps as u32,
                    dilation as u32,
                )
                .unwrap(),
            ),
            Op::Launch(
                conv_history(
                    normed_d.ptr(),
                    history_d.ptr(),
                    rows as u32,
                    channels as u32,
                    history_rows as u32,
                )
                .unwrap(),
            ),
        ],
    );
    let joined: Vec<f32> = history.iter().chain(&normed).copied().collect();
    let mut wanted_out = Vec::new();
    for t in 0..rows {
        for c in 0..channels {
            let sum: f32 = (0..taps)
                .map(|i| taps_w[c * taps + i] * joined[(t + dilation * i) * channels + c])
                .sum();
            wanted_out.push(round(gated[t * channels + c] + round(silu(round(sum)))));
        }
    }
    assert_close(&wanted_out, &read(&out_d, rows * channels), channels, 0.05, "ple_conv");
    let wanted_history = &joined[rows * channels..(rows + history_rows) * channels];
    assert_eq!(
        read(&history_d, history_rows * channels),
        wanted_history,
        "conv_history"
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_conv_control_and_gated_norm_match_the_host_reference() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (rows, channels, taps, heads, head_dim) = (3usize, 48usize, 4usize, 2usize, 16usize);
    let epsilon = 1e-6f32;
    let mut random = Random::new(47);

    let x = gaussians(&mut random, rows * channels, 1.0);
    let history = gaussians(&mut random, (taps - 1) * channels, 1.0);
    let taps_w = gaussians(&mut random, channels * taps, 0.5);
    let (x_d, history_d, taps_d) = (on_device(&rig, &x), on_device(&rig, &history), on_device(&rig, &taps_w));
    let convolved_d = empty(&rig.library, rows * channels * 2);

    let a = gaussians(&mut random, rows * heads, 1.0);
    let b = gaussians(&mut random, rows * heads, 1.0);
    let a_log = gaussians(&mut random, heads, 0.5);
    let dt_bias = gaussians(&mut random, heads, 0.5);
    let (a_d, b_d, a_log_d, dt_bias_d) = (
        on_device(&rig, &a),
        on_device(&rig, &b),
        on_device(&rig, &a_log),
        on_device(&rig, &dt_bias),
    );
    let alpha_d = empty(&rig.library, rows * heads * 4);
    let beta_d = empty(&rig.library, rows * heads * 4);

    let width = heads * head_dim;
    let core = gaussians(&mut random, rows * width, 1.0);
    let z = gaussians(&mut random, rows * width, 1.0);
    let norm = gaussians(&mut random, head_dim, 0.5);
    let (core_d, z_d, norm_d) = (on_device(&rig, &core), on_device(&rig, &z), on_device(&rig, &norm));
    let silu_d = empty(&rig.library, rows * width * 2);
    let sigmoid_d = empty(&rig.library, rows * width * 2);

    let norm_launch = |activation: u32, output: &DeviceBuffer| {
        Op::Launch(
            gdn_gated_norm(
                core_d.ptr(),
                z_d.ptr(),
                norm_d.ptr(),
                output.ptr(),
                rows as u32,
                heads as u32,
                head_dim as u32,
                epsilon,
                activation,
            )
            .unwrap(),
        )
    };
    queue(
        &rig,
        &[
            Op::Launch(
                gdn_conv(
                    x_d.ptr(),
                    history_d.ptr(),
                    taps_d.ptr(),
                    convolved_d.ptr(),
                    rows as u32,
                    channels as u32,
                    taps as u32,
                )
                .unwrap(),
            ),
            Op::Launch(
                conv_history(
                    x_d.ptr(),
                    history_d.ptr(),
                    rows as u32,
                    channels as u32,
                    taps as u32 - 1,
                )
                .unwrap(),
            ),
            Op::Launch(
                gdn_control(
                    a_d.ptr(),
                    b_d.ptr(),
                    a_log_d.ptr(),
                    dt_bias_d.ptr(),
                    alpha_d.ptr(),
                    beta_d.ptr(),
                    rows as u32,
                    heads as u32,
                )
                .unwrap(),
            ),
            norm_launch(0, &silu_d),
            norm_launch(1, &sigmoid_d),
        ],
    );

    let joined: Vec<f32> = history.iter().chain(&x).copied().collect();
    let mut wanted_conv = Vec::new();
    for t in 0..rows {
        for c in 0..channels {
            let sum: f32 = (0..taps)
                .map(|i| taps_w[c * taps + i] * joined[(t + i) * channels + c])
                .sum();
            wanted_conv.push(round(silu(round(sum))));
        }
    }
    assert_close(
        &wanted_conv,
        &read(&convolved_d, rows * channels),
        channels,
        0.05,
        "gdn_conv",
    );
    assert_eq!(
        read(&history_d, (taps - 1) * channels),
        &joined[rows * channels..],
        "conv_history keeps the last rows"
    );

    let (alpha, beta) = (read_f32(&alpha_d, rows * heads), read_f32(&beta_d, rows * heads));
    for index in 0..rows * heads {
        let head = index % heads;
        let shifted = a[index] + dt_bias[head];
        let softplus = if shifted > 20.0 { shifted } else { shifted.exp().ln_1p() };
        let wanted_alpha = (-a_log[head].exp() * softplus).exp();
        assert!(
            (alpha[index] - wanted_alpha).abs() <= 1e-5,
            "alpha {index}: {} against {wanted_alpha}",
            alpha[index]
        );
        assert_eq!(beta[index], round(sigmoid(b[index])), "beta {index}");
    }

    let mut wanted = [Vec::new(), Vec::new()];
    for head_row in 0..rows * heads {
        let span = head_row * head_dim..(head_row + 1) * head_dim;
        let mean = core[span.clone()].iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inverse = 1.0 / (mean + epsilon).sqrt();
        for (i, at) in span.enumerate() {
            let weighted = round(norm[i] * round(core[at] * inverse));
            wanted[0].push(round(weighted * silu(z[at])));
            wanted[1].push(round(weighted * sigmoid(z[at])));
        }
    }
    assert_close(
        &wanted[0],
        &read(&silu_d, rows * width),
        head_dim,
        0.05,
        "gated norm, silu",
    );
    assert_close(
        &wanted[1],
        &read(&sigmoid_d, rows * width),
        head_dim,
        0.05,
        "gated norm, sigmoid",
    );
}

/// The flash linear of a prefill, 40 rows over 512 values into 1024, against the format's own launch over the same
/// weight: the stage plans the native route, and what it queues is what the format queues. Below nine rows both run
/// the decode kernels. The weight is arbitrary bytes, so the outputs are compared bit for bit.
#[test]
#[ignore = "needs a GPU and the native library"]
fn nvfp4_projections_of_a_prefill_take_the_formats_route() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let (k, n) = (512usize, 1024usize);
    let mut random = Random::new(47);
    // The plain layout: the codes, then one E4M3 scale per 16 values, each plane to 256 bytes, then the global scale.
    let align = |value: usize| value.next_multiple_of(256);
    let size = align(align(n * k / 2) + n * k / 16) + 4;
    let mut bytes: Vec<u8> = (0..size).map(|_| random.next_u64() as u8 & 0x77).collect();
    bytes[size - 4..].copy_from_slice(&1.0f32.to_le_bytes());
    let weight_device = upload(&rig.library, &bytes);
    let weight_ref = WeightRef {
        name: "w".to_string(),
        shape: vec![n as u64, k as u64],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Nvfp4,
        layout: WeightLayout::RowSplitK128V1,
        ptr: weight_device.ptr(),
        bytes: bytes.len() as u64,
        residence: Residence::Device,
        trellis: None,
    };
    let device = Nvfp4Device::new(&rig.library);
    let format = Nvfp4Linear::new(&rig.library);
    for rows in [3usize, 9, 40, 70] {
        let x = on_device(&rig, &gaussians(&mut random, rows * k, 1.0));
        let wanted = rig.linear(
            &format,
            &weight_ref,
            &x,
            0,
            k,
            n,
            rows,
            LinearMode {
                row_exact: false,
                prefill: true,
            },
        );
        let expansion_bytes = device.expansion_bytes(rows as u32, k as u32, n as u32);
        let expansion = empty(&rig.library, expansion_bytes as usize);
        let y = empty(&rig.library, rows * n * 2);
        let op = device
            .linear(
                x.ptr(),
                Nvfp4Weight {
                    weight: weight(&weight_device, bytes.len()),
                    sd4: false,
                },
                y.ptr(),
                rows as u32,
                k as u32,
                n as u32,
                Expansion {
                    address: expansion.ptr(),
                    bytes: expansion_bytes,
                },
            )
            .unwrap();
        queue(&rig, &[op]);
        let mut host = vec![0u8; rows * n * 2];
        y.copy_to_host(0, &mut host).unwrap();
        let actual: Vec<u16> = host
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_le_bytes(*pair))
            .collect();
        assert_eq!(actual, wanted, "{rows} rows");
        assert_eq!(
            expansion_bytes > 0,
            rows >= 9 && rig.library.nvfp4_native_available(),
            "{rows} rows"
        );
    }
}

/// The gather stage refuses a part past the eight, a run before the ids, and a part published twice.
#[test]
#[ignore = "needs a GPU and the native library"]
fn ple_gather_refuses_a_missing_part_a_run_before_the_ids_and_a_second_publication() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let layout = NgramLayout::new(16, 32).unwrap();
    let shard: Arc<dyn ShardBytes> = Arc::new(vec![0u8; layout.byte_size() as usize]);
    let table = Arc::new(NgramRows::new(vec![shard], 16, 16, 32).unwrap());
    let ids = Arc::new(NgramIds::new(2, 1, &[3, 5], &[10], &[0], -1).unwrap());
    let pool = PinnedPool::new(&rig.library, 1 << 20);
    assert!(PleGatherStage::new(PARTS, Arc::clone(&table), Arc::clone(&pool)).is_err());
    let stage = PleGatherStage::new(0, table, pool).unwrap();
    let rows = PleRows::new(ids.fresh_context());
    assert!(stage.gather(&rows).is_err(), "before the ids");
    PleIdsStage::new(ids).hash(&rows, &[1; 12], 12).unwrap();
    stage.gather(&rows).unwrap();
    assert!(stage.gather(&rows).is_err(), "the part was published already");
}
