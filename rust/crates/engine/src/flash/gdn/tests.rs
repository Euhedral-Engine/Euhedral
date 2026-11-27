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

fn config() -> GdnConfig {
    GdnConfig {
        num_key_heads: 2,
        num_value_heads: 4,
        key_head_dim: 128,
        value_head_dim: 128,
        conv_kernel_dim: 4,
        ssm_dtype: "float32".to_owned(),
    }
}

fn body() -> GdnBody {
    body_on(Nvfp4Device::with_native_route(true, 8))
}

fn body_on(device: Nvfp4Device) -> GdnBody {
    let shape = GdnShape::new(256, &config(), "sigmoid", 1e-6).unwrap();
    let weights = GdnWeights {
        qkv: nvfp4(0x100000),
        z: nvfp4(0x110000),
        a: weight(0x120000),
        b: weight(0x130000),
        conv: weight(0x140000),
        a_log: weight(0x150000),
        dt_bias: weight(0x160000),
        norm: weight(0x170000),
        out: nvfp4(0x180000),
    };
    let state = GdnState {
        conv_history: at(0x190000),
        recurrent: at(0x1a0000),
    };
    GdnBody::new(shape, weights, state, device)
}

fn call(rows: u32) -> BlockCall {
    BlockCall {
        input: at(0x200000),
        output: at(0x210000),
        rows,
        exact: false,
        scratch: at(0x400000),
        expansion: Expansion {
            address: at(0x800000),
            bytes: body().expansion_bytes(rows.max(64)),
        },
    }
}

#[test]
fn the_block_queues_javas_operations_in_javas_order() {
    let ops = body().ops(&call(3)).unwrap();
    let names: Vec<_> = lines(&ops)
        .iter()
        .map(|line| line.split(' ').next().unwrap().to_owned())
        .collect();
    assert_eq!(
        names,
        [
            "nvfp4_linear",
            "nvfp4_linear",
            "euhedral_q4_linear_split8_bf16",
            "euhedral_q4_linear_split8_bf16",
            "euhedral_q4_gdn_conv_bf16",
            "euhedral_q4_conv_history_bf16",
            "euhedral_q4_gdn_control_bf16",
            "gdn_recurrence",
            "euhedral_q4_gdn_gated_norm_bf16",
            "nvfp4_linear",
        ]
    );
}

#[test]
fn the_projections_read_the_mixed_rows_and_the_last_writes_the_block_output() {
    let shape = body().shape;
    let ops = body().ops(&call(3)).unwrap();
    match (&ops[0], &ops[9]) {
        (
            Op::Nvfp4Linear {
                input,
                out_features,
                in_features,
                ..
            },
            Op::Nvfp4Linear {
                output,
                out_features: back,
                in_features: from,
                ..
            },
        ) => {
            assert_eq!(input.get(), 0x200000);
            assert_eq!((*in_features, *out_features), (256, shape.channels()));
            assert_eq!(output.get(), 0x210000);
            assert_eq!((*from, *back), (shape.value_width(), 256));
        }
        other => panic!("expected the two NVFP4 projections, got {other:?}"),
    }
}

#[test]
fn the_history_keeps_one_row_fewer_than_the_taps() {
    let ops = body().ops(&call(5)).unwrap();
    let conv = lines(&ops)[5].clone();
    assert!(conv.ends_with(&format!("i5,i{},i3", body().shape.channels())), "{conv}");
}

#[test]
fn the_scratch_buffers_do_not_overlap_and_fit_the_reported_size() {
    let body = body();
    let rows = 7;
    let scratch = body.shape.carve(&mut Carver::new(at(0x400000)), rows);
    let mut starts = [
        scratch.qkv,
        scratch.convolved,
        scratch.z,
        scratch.core,
        scratch.normed,
        scratch.a,
        scratch.b,
        scratch.alpha,
        scratch.beta,
    ]
    .map(DevicePtr::get);
    let sorted = {
        let mut copy = starts;
        copy.sort_unstable();
        copy
    };
    assert_eq!(starts, sorted, "the buffers follow Java's order");
    starts.sort_unstable();
    assert!(starts.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(scratch.beta.get() + f32_bytes(rows, 4) <= 0x400000 + body.shape.scratch_bytes(rows));
}

#[test]
fn a_layer_the_kernels_cannot_serve_is_refused() {
    let with = |change: fn(&mut GdnConfig)| {
        let mut config = config();
        change(&mut config);
        GdnShape::new(256, &config, "silu", 1e-6)
    };
    assert!(with(|config| config.key_head_dim = 64).is_err());
    assert!(with(|config| config.value_head_dim = 64).is_err());
    assert!(with(|config| config.num_value_heads = 3).is_err());
    assert!(with(|config| config.conv_kernel_dim = 1).is_err());
    assert!(with(|config| config.conv_kernel_dim = 33).is_err());
    assert!(GdnShape::new(256, &config(), "relu", 1e-6).is_err());
    assert!(GdnShape::new(250, &config(), "silu", 1e-6).is_err());
}

#[test]
fn the_activation_is_zero_for_silu_and_one_for_sigmoid() {
    let silu = GdnShape::new(256, &config(), "silu", 1e-6).unwrap();
    let sigmoid = GdnShape::new(256, &config(), "sigmoid", 1e-6).unwrap();
    assert_eq!((silu.activation, sigmoid.activation), (0, 1));
}

#[test]
fn the_projections_of_nine_rows_or_more_take_the_native_route_with_the_expansion_scratch() {
    let native =
        |ops: &OpList<'static>| [0, 1, 9].map(|index| matches!(&ops[index], Op::Nvfp4Linear { native: Some(_), .. }));
    assert_eq!(native(&body().ops(&call(8)).unwrap()), [false; 3]);
    assert_eq!(native(&body().ops(&call(9)).unwrap()), [true; 3]);
    let line = &lines(&body().ops(&call(9)).unwrap())[0];
    assert!(line.ends_with(",native p800000"), "{line}");
    // A device without the native route keeps the decode kernels at any row count.
    let plain = body_on(Nvfp4Device::with_native_route(false, 8));
    assert_eq!(native(&plain.ops(&call(40)).unwrap()), [false; 3]);
    assert_eq!(plain.expansion_bytes(40), 0);
}

#[test]
fn the_expansion_a_layer_reports_holds_its_largest_projection() {
    let body = body();
    let device = Nvfp4Device::with_native_route(true, 8);
    let shape = body.shape;
    let largest = [
        (shape.hidden, shape.channels()),
        (shape.hidden, shape.value_width()),
        (shape.value_width(), shape.hidden),
    ]
    .map(|(k, n)| device.expansion_bytes(40, k, n))
    .into_iter()
    .max()
    .unwrap();
    assert!(largest > 0);
    assert_eq!(body.expansion_bytes(40), largest);
    assert_eq!(body.expansion_bytes(8), 0);
}
