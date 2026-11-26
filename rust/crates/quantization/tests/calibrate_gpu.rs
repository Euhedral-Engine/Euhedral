//! Device test of the calibration's rounding: the fixture's linear rounded through the calibration's lanes, in slices
//! of rows, written to its file, equals the GPTQ library called directly on the same weight and Hessian. `#[ignore]`d
//! like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test calibrate_gpu -- --ignored --nocapture`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: it reads its fixture, pumps the lane's completions and stops on a failed unwrap"
)]

mod checkpoint_support;
mod gptq_rig;

use std::fs::read;
use std::path::PathBuf;
use std::sync::Arc;

use checkpoint_support::World;
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::calibrate::rounding::{Ended, Linear, RoundSettings, Services, round_all};
use euhedral_inference_quantization::calibrate::{CalibrateError, round_order};
use euhedral_inference_quantization::gptq::format::Format;
use euhedral_inference_quantization::gptq::state::QuantState;
use euhedral_inference_quantization::gptq::{GptqSettings, RoundRequest, TensorScales};
use euhedral_inference_quantization::npy::Npy;
use euhedral_inference_quantization::recipe::{Quantization, Recipe};
use gptq_rig::{range_of, rig};
use serde_json::Value;

const OBJECT: &str = "text/layers/9/gdn/query_key";

fn fixture(name: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../resources/fixtures/gptq-fixture");
    read(path.join(name)).unwrap()
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn a_linear_rounded_in_slices_by_the_calibration_equals_the_library_called_directly() {
    let cases: Value = serde_json::from_slice(&fixture("cases.json")).unwrap();
    let n = cases["rows"].as_u64().unwrap() as usize;
    let k = cases["columns"].as_u64().unwrap() as usize;
    let damp = cases["damp"].as_f64().unwrap() as f32;
    let weight_file = fixture("weight_bf16.npy");
    let hessian_file = fixture("hessian.npy");
    let weight_bytes = Npy::parse(&weight_file).unwrap().data.to_vec();
    let hessian_bytes = Npy::parse(&hessian_file).unwrap().data.to_vec();

    let recipe = Recipe::new(Quantization::Nvfp4, false, None).unwrap();
    let stored = recipe.format(OBJECT).unwrap();
    let format = Format::of(&stored.name, &stored.layout, None).unwrap();
    assert!(format.nvfp4, "the test needs a format with tensor scales");

    let mut rig = rig(GptqSettings::default());
    let weight = rig.upload(&weight_bytes);
    let hessian = rig.upload(&hessian_bytes);

    // Directly: the whole weight in one call.
    let factor = rig.factor(&hessian, k, damp, false).unwrap();
    let tensor: TensorScales = rig.tensor_scales(&weight, n, &factor, &format).unwrap();
    let direct = rig
        .round(RoundRequest {
            weight: range_of(&weight),
            n,
            factor: &factor,
            format: &format,
            tensor: &tensor,
            decoded_out: None,
        })
        .unwrap();
    drop(factor);

    // Through the calibration's rounding: slices of half the rows.
    let directory = tempfile::tempdir().unwrap();
    let world = World::start();
    let (planner, owner) = rig.planner();
    let services = Arc::new(Services {
        planner,
        owner,
        device: euhedral_memory::LocationId(0),
        pinned: euhedral_memory::LocationId(1),
        settings: RoundSettings {
            recipe,
            damp,
            act_order: false,
            slice_rows: n / 2,
        },
        out: directory.path().to_path_buf(),
        io: Arc::clone(&world.io),
        host: Arc::clone(&world.host),
    });
    let decoded = euhedral_inference_cuda::DeviceBuffer::alloc(&rig.library, 2 * n * k).unwrap();
    let held = Handoff::empty();
    held.send(decoded);
    let linears = vec![Linear {
        name: OBJECT.to_owned(),
        rows: n as u64,
        columns: k,
        weight: weight.addr(),
        hessian: hessian.addr(),
        decoded: held,
    }];
    let order = round_order(&[(OBJECT.to_owned(), (n as u64, k as u64))]);
    let lane = rig.take_lane();
    let ended = Handoff::<Result<Ended, CalibrateError>>::new();
    let sink = Arc::clone(&ended);
    // SAFETY: the weight and the Hessian are the test's buffers, live and untouched until the rounding has ended.
    unsafe {
        round_all(
            services,
            vec![lane],
            linears,
            order,
            Box::new(move |result| sink.send(result)),
        );
    }
    rig.pump_until(|| ended.is_set());
    let Ended { mut lanes, .. } = ended.take().unwrap().unwrap();
    rig.restore_lane(lanes.remove(0));
    world.finish();

    let written = read(directory.path().join(QuantState::file_name(OBJECT))).unwrap();
    let through = QuantState::from_npz(&written).unwrap();
    assert_eq!(
        through, direct,
        "the calibration's slices differ from one call of the library"
    );
}
