//! Proof only: GPTQ against the recorded bytes of the Java run on the fixture. Deleted before the final commit.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stdout, clippy::disallowed_methods, reason = "proof test")]

mod gptq_rig;

use std::fs::read;
use std::path::PathBuf;

use euhedral_inference_quantization::gptq::format::Format;
use euhedral_inference_quantization::gptq::{GptqSettings, RoundRequest, TensorScales};
use euhedral_inference_quantization::npy::Npy;
use euhedral_inference_quantization::npz::Npz;
use gptq_rig::rig;
use serde_json::Value;

#[test]
#[ignore = "needs a GPU and the goldens"]
fn matches_the_java_goldens_byte_for_byte() {
    let goldens = PathBuf::from(test_support::gradle_property("euhedral.quantization.goldens").expect("goldens key"))
        .join("gptq-fixture-java");
    let fixture = |name: &str| {
        read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../quantization/src/test/resources/gptq-fixture").join(name)).unwrap()
    };
    let cases: Value = serde_json::from_slice(&fixture("cases.json")).unwrap();
    let n = cases["rows"].as_u64().unwrap() as usize;
    let k = cases["columns"].as_u64().unwrap() as usize;
    let damp = cases["damp"].as_f64().unwrap() as f32;
    let weight_file = fixture("weight_bf16.npy");
    let hessian_file = fixture("hessian.npy");
    let (weight_npy, hessian_npy) = (Npy::parse(&weight_file).unwrap(), Npy::parse(&hessian_file).unwrap());
    let mut rig = rig(GptqSettings::default());
    let weight = rig.upload(weight_npy.data);
    let hessian = rig.upload(hessian_npy.data);
    let mut compared = 0;
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let format = Format::of(case["format"].as_str().unwrap(), case["layout"].as_str().unwrap(), None).unwrap();
        let factor = rig.factor(&hessian, k, damp, case["act_order"].as_bool().unwrap()).unwrap();
        let tensor = if format.nvfp4 {
            rig.tensor_scales(&weight, n, &factor, &format).unwrap()
        } else {
            TensorScales { global_scale: 0.0, table: None }
        };
        let state = rig
            .round(RoundRequest { weight: weight.addr(), n, factor: &factor, format: &format, tensor: &tensor, decoded_out: None })
            .unwrap();
        let mine = state.to_npz().unwrap();
        let theirs = read(goldens.join(format!("{name}.npz"))).unwrap();
        let (a, b) = (Npz::parse(&mine).unwrap(), Npz::parse(&theirs).unwrap());
        let names = |z: &Npz| z.entries().iter().map(|e| e.name.clone()).collect::<Vec<_>>();
        assert_eq!(names(&a), names(&b), "{name}: member names and order");
        for (x, y) in a.entries().iter().zip(b.entries()) {
            assert_eq!(x.bytes, y.bytes, "{name}: member {}", x.name);
        }
        assert_eq!(state.codes, read(goldens.join(format!("{name}.codes.u8"))).unwrap(), "{name} codes.u8");
        assert_eq!(state.scales, read(goldens.join(format!("{name}.scales.bin"))).unwrap(), "{name} scales.bin");
        println!("{name}: {} members identical, codes {} bytes, scales {} bytes", a.entries().len(), state.codes.len(), state.scales.len());
        compared += 1;
    }
    assert_eq!(compared, 8);
}
