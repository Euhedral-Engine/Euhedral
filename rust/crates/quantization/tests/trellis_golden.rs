//! Golden comparison of the trellis kernels (temporary proof).
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used, reason = "a test file")]
#![allow(clippy::print_stdout, reason = "the counts are the output")]

mod support;

use std::fs::read;
use std::path::PathBuf;

use euhedral_inference_quantization::npy::Npy;
use euhedral_inference_quantization::trellis::{
    Rate, Reconstruction, TILE_WEIGHTS, plan_decode, plan_pack, plan_reconstruct, plan_viterbi,
};
use support::rig;

fn fixture(name: &str) -> Vec<u8> {
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../quantization/src/test/resources/trellis-fixture");
    read(root.join(name)).unwrap()
}

fn golden(name: &str) -> Vec<u8> {
    let root = test_support::gradle_property("euhedral.quantization.goldens").expect("goldens key");
    read(PathBuf::from(root).join("trellis-tiles-java").join(name)).unwrap()
}

fn half(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f32::from(bits & 0x3ff);
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-24)
    } else {
        sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15)
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn tiles_match_exllamav3_and_the_java_goldens() {
    let rig = rig();
    let input = fixture("tiles-in.npy");
    let tiles_npy = Npy::parse(&input).unwrap();
    let count = 64usize;
    for name in ["2", "3", "4", "5", "6", "2.5", "3.5"] {
        let rate = Rate::new(name.parse().unwrap()).unwrap();
        let words = rate.words_per_tile();
        let tiles = rig.upload(tiles_npy.data);
        let indices = rig.buffer(2 * count * TILE_WEIGHTS);
        let decoded = rig.buffer(4 * count * TILE_WEIGHTS);
        let packed = rig.buffer(2 * count * words);
        let again = rig.buffer(4 * count * TILE_WEIGHTS);
        let scratch = rig.buffer(usize::try_from(rate.scratch_bytes(count as u64)).unwrap());
        let mut steps = plan_viterbi(
            rate,
            tiles.addr(),
            count as u32,
            indices.addr(),
            decoded.addr(),
            scratch.addr(),
            count as u32,
        )
        .unwrap();
        steps.extend(plan_pack(rate, indices.addr(), count as u32, packed.addr()).unwrap());
        steps.extend(plan_decode(rate, packed.addr(), count as u32, again.addr()).unwrap());
        rig.run(&steps);
        let got_idx = rig.download(&indices, 2 * count * TILE_WEIGHTS);
        let got_packed = rig.download(&packed, 2 * count * words);
        let got_decoded = rig.download(&decoded, 4 * count * TILE_WEIGHTS);
        let got_again = rig.download(&again, 4 * count * TILE_WEIGHTS);
        let (idx, pk, out) = (
            fixture(&format!("tiles-{name}-idx.npy")),
            fixture(&format!("tiles-{name}-packed.npy")),
            fixture(&format!("tiles-{name}-out.npy")),
        );
        assert_eq!(got_idx, Npy::parse(&idx).unwrap().data, "exllamav3 states, K {name}");
        assert_eq!(got_packed, Npy::parse(&pk).unwrap().data, "exllamav3 words, K {name}");
        let out = Npy::parse(&out).unwrap();
        let expected: Vec<u8> = out
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| half(u16::from_le_bytes(*c)).to_le_bytes())
            .collect();
        assert_eq!(got_decoded, expected, "exllamav3 values, K {name}");
        assert_eq!(got_again, got_decoded, "decode of the packed words, K {name}");
        assert_eq!(
            got_idx,
            golden(&format!("tiles-{name}-idx.u16")),
            "java states, K {name}"
        );
        assert_eq!(
            got_packed,
            golden(&format!("tiles-{name}-packed.u16")),
            "java words, K {name}"
        );
        assert_eq!(
            got_decoded,
            golden(&format!("tiles-{name}-decoded.f32")),
            "java values, K {name}"
        );
        println!("K {name}: states, words and values exact against exllamav3 and the Java golden ({count} tiles)");
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn the_reconstruction_matches_exllamav3_within_one_bf16_ulp() {
    let rig = rig();
    let meta = String::from_utf8(fixture("linear.json")).unwrap();
    let field = |key: &str| -> f64 {
        let at = meta.find(&format!("\"{key}\"")).unwrap();
        let rest = &meta[at..];
        let colon = rest.find(':').unwrap();
        rest[colon + 1..]
            .split([',', '\n', '}'])
            .next()
            .unwrap()
            .trim()
            .parse()
            .unwrap()
    };
    let (k, n) = (field("k") as usize, field("n") as usize);
    let rate = Rate::new(field("bits")).unwrap();
    let (trellis, suh, svh, loaded) = (
        fixture("linear-trellis.npy"),
        fixture("linear-suh.npy"),
        fixture("linear-svh.npy"),
        fixture("linear-loaded.npy"),
    );
    let words = rig.upload(Npy::parse(&trellis).unwrap().data);
    let su = rig.upload(Npy::parse(&suh).unwrap().data);
    let sv = rig.upload(Npy::parse(&svh).unwrap().data);
    let out = rig.buffer(2 * n * k);
    let scratch = rig.buffer(4 * k * n);
    let operands = Reconstruction {
        words: words.addr(),
        suh: su.addr(),
        svh: sv.addr(),
        k: k as u32,
        n: n as u32,
        out: out.addr(),
        scratch: scratch.addr(),
    };
    rig.run(&plan_reconstruct(rate, &operands).unwrap());
    let actual = rig.download(&out, 2 * n * k);
    let expected = Npy::parse(&loaded).unwrap();
    let (mut error, mut norm, mut max_error, mut max_value) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (a, e) in actual.as_chunks::<2>().0.iter().zip(expected.data.as_chunks::<2>().0) {
        let a = f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*a)) << 16));
        let e = f64::from(half(u16::from_le_bytes(*e)));
        error += (a - e) * (a - e);
        norm += e * e;
        max_error = max_error.max((a - e).abs());
        max_value = max_value.max(e.abs());
    }
    let relative = (error / norm).sqrt();
    let ulp = 2f64.powi(max_value.log2().floor() as i32 - 7);
    println!("reconstruct: relative RMS {relative:e} (bound 4e-3), max error {max_error:e}, bf16 ulp {ulp:e}");
    assert!(relative <= 4e-3 && max_error <= ulp);
}
