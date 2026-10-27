//! The scoring arithmetic and the `.values` file, on small synthetic rows.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::time::{Duration, UNIX_EPOCH};

use euhedral_inference_bench::kld::{self, Reference, Row};
use euhedral_inference_bench::score::{FIELDS, percentile, score_row, standard_error, summary_lines, values_bytes};

/// BF16 little-endian bytes of values that BF16 holds exactly.
fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// A record whose log-probabilities are `p` (each a multiple of 1/16 from -16, within the 16-nat window).
fn record(p: &[f64]) -> Vec<u8> {
    let codes: Vec<u16> = p.iter().map(|p| ((p + 16.0) * 16.0).round() as u16).collect();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(1.0f32 / 16.0).to_le_bytes());
    bytes.extend_from_slice(&(-16.0f32).to_le_bytes());
    for code in &codes {
        bytes.extend_from_slice(&code.to_le_bytes());
    }
    bytes.resize(8 + 2 * 2 * p.len().div_ceil(2), 0);
    bytes
}

fn kl_by_hand(logits: &[f64], p: &[f64]) -> f64 {
    let max = logits.iter().copied().fold(f64::MIN, f64::max);
    let log_z = max + logits.iter().map(|q| (q - max).exp()).sum::<f64>().ln();
    p.iter()
        .zip(logits)
        .filter(|(p, _)| **p > -16.0)
        .map(|(p, q)| p.exp() * (p - (q - log_z)))
        .sum()
}

#[test]
fn kl_masks_floor_entries() {
    let logits = [2.0f32, 1.0, -1.0, 0.5, 0.0];
    let wide: Vec<f64> = logits.iter().map(|q| f64::from(*q)).collect();
    // The reference puts entries 3 and 4 on the floor (-16 exactly: a code of 0): they carry no mass and no term.
    let p = [-0.5, -1.25, -2.0, -16.0, -16.0];
    let bytes = record(&p);
    let row = Row::new(&bytes, 5);
    let values = score_row(&bf16(&logits), &row, 5, 1);
    assert!(
        (values[2] - kl_by_hand(&wide, &p)).abs() < 1e-12,
        "{} vs {}",
        values[2],
        kl_by_hand(&wide, &p)
    );

    // Moving the masked entries' logits changes the normaliser, not the masked terms: KL follows the by-hand sum, which
    // skips them, and a logit change at a masked entry moves it only through log Z.
    let moved = [2.0f32, 1.0, -1.0, 4.0, -3.0];
    let wide_moved: Vec<f64> = moved.iter().map(|q| f64::from(*q)).collect();
    let values_moved = score_row(&bf16(&moved), &row, 5, 1);
    assert!((values_moved[2] - kl_by_hand(&wide_moved, &p)).abs() < 1e-12);

    // An entry one code above the floor is counted.
    let mut above = p;
    above[3] = -16.0 + 1.0 / 16.0;
    let bytes_above = record(&above);
    let counted = score_row(&bf16(&logits), &Row::new(&bytes_above, 5), 5, 1);
    assert!((counted[2] - kl_by_hand(&wide, &above)).abs() < 1e-12);
    assert!(counted[2] != values[2]);
}

#[test]
fn a_row_scores_nll_agreement_and_probability_difference() {
    let logits = [3.0f32, 1.0, 0.0, -2.0];
    let wide: Vec<f64> = logits.iter().map(|q| f64::from(*q)).collect();
    let log_z = wide.iter().map(|q| q.exp()).sum::<f64>().ln();
    // The reference agrees on token 0 as the top one.
    let p = [-0.25, -2.0, -3.0, -8.0];
    let bytes = record(&p);
    let values = score_row(&bf16(&logits), &Row::new(&bytes, 4), 4, 1);
    assert!((values[0] - (log_z - 1.0)).abs() < 1e-12, "candidate NLL of token 1");
    assert_eq!(values[1], 2.0, "reference NLL of token 1");
    assert_eq!(values[3], 1.0, "both tops are token 0");
    assert!((values[4] - ((1.0 - log_z).exp() - (-2.0f64).exp())).abs() < 1e-12);

    // The reference prefers token 1: no agreement. A tie takes the first maximum on both sides.
    let other = record(&[-3.0, -0.25, -3.0, -8.0]);
    assert_eq!(score_row(&bf16(&logits), &Row::new(&other, 4), 4, 1)[3], 0.0);
    let tied = record(&[-1.0, -1.0, -3.0, -8.0]);
    assert_eq!(
        score_row(&bf16(&[1.0, 1.0, 0.0, 0.0]), &Row::new(&tied, 4), 4, 0)[3],
        1.0
    );
}

#[test]
fn values_file_layout_matches_java() {
    // Two chunks of three forced tokens: 2 * 3 * 5 values in step order, chunk after chunk.
    let chunks: Vec<Vec<f64>> = (0..2)
        .map(|chunk| (0..3 * FIELDS).map(|i| (chunk * 100 + i) as f64 + 0.5).collect())
        .collect();
    let bytes = values_bytes(&chunks, 3);
    assert_eq!(bytes.len(), 12 + 8 * 2 * 3 * 5);
    // `<3i`: chunks, scored tokens per chunk, values per token.
    assert_eq!(&bytes[..12], &[2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0]);
    assert_eq!(&bytes[12..20], &0.5f64.to_le_bytes());
    // Chunk 1, step 0, field 2 (kl): value 100 + 2 + 0.5 at (1 * 3 + 0) * 5 + 2 = 17
    let at = 12 + 8 * 17;
    assert_eq!(&bytes[at..at + 8], &102.5f64.to_le_bytes());
    let last = bytes.len() - 8;
    assert_eq!(&bytes[last..], &114.5f64.to_le_bytes());
}

#[test]
fn statistics_follow_the_java_scorer() {
    // NumPy's linear percentile.
    let sorted = [1.0, 2.0, 4.0, 8.0, 16.0];
    assert_eq!(percentile(&sorted, 50.0), 4.0);
    assert!((percentile(&sorted, 95.0) - (16.0 - 0.2 * 8.0)).abs() < 1e-12);
    assert_eq!(percentile(&sorted, 0.0), 1.0);
    // Chunk-level over several chunks, token-level over one.
    let chunks = [1.0, 3.0];
    let tokens = [0.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(standard_error(&chunks, &tokens), 1.0);
    assert!((standard_error(&[2.0], &tokens) - (2.5f64.sqrt() / 5.0f64.sqrt())).abs() < 1e-15);
}

#[test]
fn the_summary_has_the_scorers_lines() {
    let tokens: Vec<Vec<u32>> = vec![(0..8).collect(); 2];
    let bytes = kld::encode(8, 4, &tokens, |_, _| (1.0, 0.0, vec![0; 4]));
    let reference = Reference::parse(&bytes).unwrap();
    let values: Vec<Vec<f64>> = (0..2)
        .map(|chunk| {
            (0..3)
                .flat_map(|step| {
                    let k = (chunk * 3 + step) as f64;
                    [
                        1.0 + k / 10.0,
                        1.0,
                        0.01 * (k + 1.0),
                        f64::from(u8::from(step != 1)),
                        0.001 * k,
                    ]
                })
                .collect()
        })
        .collect();
    let when = UNIX_EPOCH + Duration::from_secs(1_790_424_000);
    let text = summary_lines("candidate", "ref.kld", "a.edrl", &reference, &values, when);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 5, "{text}");
    assert_eq!(
        lines[0],
        "# candidate 2026-09-26T12:00:00Z artifact a.edrl reference ref.kld"
    );
    assert_eq!(
        lines[1],
        "2 chunk(s) of 8, 3 forced tokens each (positions 5..7); reference ref.kld"
    );
    assert!(
        lines[2].starts_with("report") && lines[2].contains("KL(ref||c)"),
        "{}",
        lines[2]
    );
    // The reference's own row: mean NLL 1.0, perplexity e.
    assert!(
        lines[3].starts_with("ref.kld") && lines[3].contains("1.00000   2.7183"),
        "{}",
        lines[3]
    );
    let columns: Vec<&str> = lines[4].split_whitespace().collect();
    assert_eq!(columns[0], "candidate");
    // mean NLL over the six steps: 1 + mean(k)/10 = 1.25
    assert_eq!(columns[1], "1.25000");
    // top-1 agreement: 4 of 6 steps
    assert_eq!(columns[7], "0.6667");
    assert_eq!(columns.len(), 17);
}
