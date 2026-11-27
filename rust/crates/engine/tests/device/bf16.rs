//! The BF16 linear and embedding on a GPU, against a host reference, on small synthetic objects. The tests are
//! `#[ignore]`d and run under a card grant of a gigabyte or two; a test skips with a message when the native library
//! property (`euhedral.cuda.library`) is absent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test copies device memory and synchronizes"
)]

use crate::common;
use crate::linear_gpu;

use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{Bf16Linear, LinearMode, TokenEmbedding};
use euhedral_inference_engine::model::{Residence, WeightRef};
use linear_gpu::{Random, activations, assert_close, bf16, bf16_value, rig, upload, values};

const DECODE: LinearMode = LinearMode {
    row_exact: true,
    prefill: false,
};

/// A BF16 table of `rows` rows of `width` Gaussian values, as bytes, and its values.
fn table(random: &mut Random, rows: usize, width: usize) -> (Vec<u8>, Vec<f32>) {
    let bits: Vec<u16> = (0..rows * width).map(|_| bf16(random.gaussian())).collect();
    let bytes = bits.iter().flat_map(|value| value.to_le_bytes()).collect();
    (bytes, values(&bits))
}

fn weight(name: &str, shape: [usize; 2], buffer: &DeviceBuffer) -> WeightRef {
    WeightRef {
        name: name.to_string(),
        shape: shape.iter().map(|dim| *dim as u64).collect(),
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Bf16,
        layout: WeightLayout::ContiguousLeV1,
        ptr: buffer.ptr(),
        bytes: (shape[0] * shape[1] * 2) as u64,
        residence: Residence::Device,
        trellis: None,
    }
}

/// Every route of the row-exact GEMMs (split over K or not, one row tile or four) against the f32 sum of the products,
/// and every row of a many-row launch against the one-row launch of that row: the row-exact contract.
#[test]
#[ignore = "needs a GPU and the native library"]
fn bf16_linear_matches_the_host_sum_and_each_row_its_own_launch() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(7);
    // (n, k, rows): n % 32 != 0 splits; n <= 1536 splits; above 16 rows runs four row tiles.
    for (n, k, rows) in [
        (40, 64, 1),
        (2048, 128, 3),
        (1536, 96, 20),
        (2048, 96, 40),
        (72, 32, 17),
    ] {
        let (weight_bytes, weights) = table(&mut random, n, k);
        let device = upload(&rig.library, &weight_bytes);
        let weight = weight("w", [n, k], &device);
        let input = upload(&rig.library, &activations(&mut random, rows, k));
        let mut x = vec![0u8; rows * k * 2];
        input.copy_to_host(0, &mut x).unwrap();
        let x: Vec<f32> = x
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16_value(u16::from_le_bytes(*pair)))
            .collect();
        let wanted: Vec<f32> = (0..rows)
            .flat_map(|row| {
                let (x, weights) = (&x, &weights);
                (0..n).map(move |column| (0..k).map(|i| x[row * k + i] * weights[column * k + i]).sum::<f32>())
            })
            .collect();
        let all = rig.linear(&Bf16Linear, &weight, &input, 0, k, n, rows, DECODE);
        assert_close(&wanted, &values(&all), n, 0.02, &format!("n {n} k {k} rows {rows}"));
        for row in 0..rows {
            let alone = rig.linear(&Bf16Linear, &weight, &input, row, k, n, 1, DECODE);
            assert_eq!(
                &all[row * n..(row + 1) * n],
                &alone[..],
                "n {n} k {k}: row {row} differs from its own launch"
            );
        }
    }
}

/// The lookup copies each token's row of the table exactly, a repeated id gives the same row, and a table of another
/// format is refused before any launch.
#[test]
#[ignore = "needs a GPU and the native library"]
fn bf16_embedding_copies_the_token_rows() {
    let _card = common::card();
    let Some(rig) = rig() else { return };
    let mut random = Random::new(11);
    let (vocabulary, hidden) = (64usize, 96usize);
    let (bytes, _) = table(&mut random, vocabulary, hidden);
    let device = upload(&rig.library, &bytes);
    let weight = weight("text/token_embedding", [vocabulary, hidden], &device);
    let ids: [i32; 4] = [0, 63, 17, 17];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    let tokens = upload(&rig.library, &id_bytes);
    let output = upload(&rig.library, &vec![0u8; ids.len() * hidden * 2]);
    rig.run(|ops| TokenEmbedding.submit(ops, &weight, tokens.ptr(), output.ptr(), ids.len()));
    let mut host = vec![0u8; ids.len() * hidden * 2];
    output.copy_to_host(0, &mut host).unwrap();
    for (slot, id) in ids.iter().enumerate() {
        let row = *id as usize * hidden * 2;
        assert_eq!(
            &host[slot * hidden * 2..(slot + 1) * hidden * 2],
            &bytes[row..row + hidden * 2],
            "slot {slot}: the row of id {id}"
        );
    }
    let mut refused = weight.clone();
    refused.format = WeightFormat::Nvfp4;
    let made = rig.typed.table.submit_ops(rig.typed.lane(0), false, |ops| {
        TokenEmbedding.submit(ops, &refused, tokens.ptr(), output.ptr(), 1)
    });
    rig.typed.sync();
    assert!(made.is_err());
}
