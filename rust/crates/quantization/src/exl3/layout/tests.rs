#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use serde_json::json;

use super::*;

// Each case is a tie or an edge of the half-precision range; a mutant that rounds ties up, truncates, or moves the
// overflow threshold fails one of them.
#[test]
fn half_bits_rounds_to_nearest_even_and_saturates_to_infinity() {
    let ulp = 2f32.powi(-10);
    let cases: [(f32, u16); 14] = [
        (1.0, 0x3C00),
        (1.0 + ulp / 2.0, 0x3C00), // a tie goes to the even neighbour below
        (1.0 + 1.5 * ulp, 0x3C02), // a tie goes to the even neighbour above
        (1.0 + ulp / 2.0 + 1e-6, 0x3C01),
        (65504.0, 0x7BFF),
        (65519.9, 0x7BFF),
        (65520.0, 0x7C00),
        (2f32.powi(-24), 0x0001),
        (2f32.powi(-25), 0x0000),       // half the smallest subnormal ties to zero
        (1.5 * 2f32.powi(-24), 0x0002), // and a tie above it to two
        (2f32.powi(-14), 0x0400),
        (-0.0, 0x8000),
        (-2.0, 0xC000),
        (f32::NEG_INFINITY, 0xFC00),
    ];
    for (value, bits) in cases {
        assert_eq!(half_bits(value), bits, "{value}");
    }
    let nan = half_bits(f32::NAN);
    assert_eq!(nan & 0x7C00, 0x7C00);
    assert_ne!(nan & 0x03FF, 0, "a NaN stays a NaN");
}

// A mutant that clamps an overflow to the largest half, or lets it become infinity, fails.
#[test]
fn bf16_to_half_converts_and_refuses_a_finite_value_beyond_the_range() {
    let bf16 = |value: f32| ((value.to_bits() >> 16) as u16).to_le_bytes();
    let mut bytes: Vec<u8> = [1.0f32, -2.0, 0.0].into_iter().flat_map(bf16).collect();
    bf16_to_half("w", &mut bytes).unwrap();
    let halves: Vec<u16> = bytes
        .chunks(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    assert_eq!(halves, [0x3C00, 0xC000, 0x0000]);

    let mut huge: Vec<u8> = [1.0f32, 1.0e38].into_iter().flat_map(bf16).collect();
    let error = bf16_to_half("lm_head.weight", &mut huge).unwrap_err().to_string();
    assert!(error.contains("lm_head.weight"), "{error}");

    let mut infinite: Vec<u8> = bf16(f32::INFINITY).to_vec();
    bf16_to_half("w", &mut infinite).unwrap();
    assert_eq!(infinite, 0x7C00u16.to_le_bytes());
}

// The thresholds and the exponent form are where a different float printer disagrees.
#[test]
fn double_text_switches_to_the_exponent_form_outside_1e_3_to_1e7() {
    let cases = [
        (2.0, "2.0"),
        (2.5, "2.5"),
        (0.001, "0.001"),
        (0.000_999_9, "9.999E-4"),
        (1.0e-6, "1.0E-6"),
        (1234567.0, "1234567.0"),
        (9_999_999.0, "9999999.0"),
        (1.0e7, "1.0E7"),
        (1.5e10, "1.5E10"),
        (123456.789, "123456.789"),
        (-0.5, "-0.5"),
        (-0.0, "-0.0"),
        (0.0, "0.0"),
        (100.0, "100.0"),
    ];
    for (value, text) in cases {
        assert_eq!(double_text(value), text);
    }
}

// An empty array or object written as `[]` instead of `[ ]`, changes the bytes.
#[test]
fn pretty_puts_every_member_on_its_own_line() {
    let value = json!({"empty": [], "none": {}, "list": [1, 2.5], "nested": {"k": "v"}});
    let expected = "{\n    \"empty\": [ ],\n    \"none\": { },\n    \"list\": [\n        1,\n        2.5\n    ],\n    \"nested\": {\n        \"k\": \"v\"\n    }\n}";
    assert_eq!(pretty(&value), expected);
}

// A mutant that copies weights, pickled files or the index, or drops a tokenizer file, changes the list.
#[test]
fn auxiliary_files_are_the_ones_a_loader_needs() {
    let names = [
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "LICENSE",
        "old.bin",
        "weights.pth",
        "chat_template.jinja",
    ];
    assert_eq!(
        auxiliary_files(names),
        ["LICENSE", "chat_template.jinja", "tokenizer.json"]
    );
}

// A mutant that counts a repeated file twice doubles the total.
#[test]
fn a_file_written_again_replaces_its_entries() {
    let stored = |bytes| Stored {
        dtype: "F16".to_owned(),
        shape: vec![bytes / 2],
        bytes,
    };
    let mut ledger = Ledger::default();
    ledger.record_file("a.safetensors", &[("x".to_owned(), stored(8))]);
    ledger.record_file("b.safetensors", &[("y".to_owned(), stored(4))]);
    ledger.record_file(
        "a.safetensors",
        &[("x".to_owned(), stored(8)), ("z".to_owned(), stored(2))],
    );
    let index = ledger.index();
    assert_eq!(index["metadata"]["total_size"], 14);
    let order: Vec<&String> = index["weight_map"].as_object().unwrap().keys().collect();
    assert_eq!(order, ["y", "x", "z"], "a rewritten file moves to the end");
    assert_eq!(index["weight_map"]["x"], "a.safetensors");
}

// A run resumed from files an earlier run left records them after the MTP block it wrote itself; its index must still
// be the one of a run that wrote them in turn.
#[test]
fn the_index_lists_layers_by_number_then_the_mtp_block_whatever_the_order_they_were_recorded_in() {
    let stored = Stored {
        dtype: "F16".to_owned(),
        shape: vec![1],
        bytes: 2,
    };
    let record = |ledger: &mut Ledger, file: &str, tensor: &str| {
        ledger.record_file(file, &[(tensor.to_owned(), stored.clone())]);
    };
    let layer = |number: usize| format!("{TEXT}layers.{number}.safetensors");
    let mut ledger = Ledger::default();
    record(&mut ledger, "mtp.safetensors", "m");
    record(&mut ledger, &layer(10), "l10");
    record(&mut ledger, &layer(2), "l2");
    record(&mut ledger, "lm_head.safetensors", "h");
    record(&mut ledger, &layer(0), "l0");
    let index = ledger.index();
    let order: Vec<&String> = index["weight_map"].as_object().unwrap().keys().collect();
    assert_eq!(order, ["l0", "l2", "l10", "m", "h"]);
}

// A mutant that writes a made-up name for a dtype with no torch name gives a loader a dtype that does not exist.
#[test]
fn a_dtype_without_a_torch_name_is_stored_as_null() {
    let mut ledger = Ledger::default();
    let stored = |dtype: &str| Stored {
        dtype: dtype.to_owned(),
        shape: vec![4],
        bytes: 8,
    };
    ledger.record_file(
        "lm_head.safetensors",
        &[
            ("lm_head.weight".to_owned(), stored("F16")),
            ("lm_head.bias".to_owned(), stored("Q4")),
        ],
    );
    let storage = ledger.tensor_storage(0);
    let parts = &storage["lm_head"]["stored_tensors"];
    assert_eq!(parts["lm_head.weight"]["dtype"], "torch.float16");
    assert!(parts["lm_head.bias"]["dtype"].is_null(), "{parts}");
}
