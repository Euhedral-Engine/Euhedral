//! The text a score prints: the mean NLL and KL of a candidate against a reference, with their standard errors and the
//! tails of the per-position distributions. The layout and the digits are the Java scorer's.

use std::fmt::Write;
use std::path::Path;

/// The float64 values the score kernel produces for one position: NLL, reference NLL, KL, top-1 agreement, and the
/// difference of the probabilities of the forced token.
pub const FIELDS: usize = 5;

/// What the summary needs to know of the reference.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    /// The tokens of a chunk.
    pub context: usize,
    /// The forced positions of a chunk.
    pub steps: usize,
}

/// The summary of one candidate against one reference: `scored` holds a vector per chunk of `steps * FIELDS` values.
///
/// `stamp` is the time the header names, already formatted. Returns `None` when there is nothing to summarise.
pub fn lines(
    name: &str,
    stamp: &str,
    path: &Path,
    artifact: &Path,
    shape: Shape,
    scored: &[Vec<f64>],
) -> Option<String> {
    let (chunks, steps) = (scored.len(), shape.steps);
    let n = chunks * steps;
    if n == 0 || scored.iter().any(|chunk| chunk.len() != steps * FIELDS) {
        return None;
    }
    let column = |field: usize| -> Vec<f64> {
        scored
            .iter()
            .flat_map(|chunk| chunk.as_chunks::<FIELDS>().0.iter().map(move |row| row[field]))
            .collect()
    };
    let (nll, reference_nll, kl, agree, dp) = (column(0), column(1), column(2), column(3), column(4));
    let difference: Vec<f64> = nll.iter().zip(&reference_nll).map(|(a, b)| a - b).collect();
    let per_chunk = |values: &[f64]| -> Vec<f64> { values.chunks_exact(steps).map(mean).collect() };
    let chunk_difference: Vec<f64> = per_chunk(&nll)
        .iter()
        .zip(per_chunk(&reference_nll))
        .map(|(a, b)| a - b)
        .collect();
    let (chunk_kl, chunk_agree) = (per_chunk(&kl), per_chunk(&agree));
    let (mut sorted_kl, mut sorted_dp) = (kl.clone(), dp.clone());
    sorted_kl.sort_by(f64::total_cmp);
    sorted_dp.sort_by(f64::total_cmp);
    let rms = (dp.iter().map(|value| value * value).sum::<f64>() / n as f64).sqrt();
    let reference_mean = mean(&reference_nll);
    let mean_nll = mean(&nll);
    let first = shape.context / 2 + 1;

    let mut text = String::new();
    // Writing to a String cannot fail.
    let _ = writeln!(
        text,
        "# {name} {stamp} artifact {} reference {}",
        artifact.display(),
        path.display()
    );
    let _ = writeln!(
        text,
        "{chunks} chunk(s) of {}, {steps} forced tokens each (positions {first}..{}); reference {}",
        shape.context,
        shape.context - 1,
        path.display()
    );
    let _ = writeln!(
        text,
        "{:<50} {:>9} {:>8} {:>9} {:>8} {:>11} {:>9} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "report",
        "mean NLL",
        "ppl",
        "dNLL",
        "+-SE",
        "KL(ref||c)",
        "+-SE",
        "top-1",
        "KL med",
        "KL p95",
        "KL p99",
        "KL max",
        "top1 SE",
        "dp mean",
        "dp RMS",
        "dp p1",
        "dp p99"
    );
    let _ = writeln!(
        text,
        "{:<50} {:>9.5} {:>8.4}",
        path.display(),
        reference_mean,
        reference_mean.exp()
    );
    let _ = writeln!(
        text,
        "{:<50} {:>9.5} {:>8.4} {:>+9.5} {:>8.5} {:>11.6} {:>9.6} {:>7.4} {:>8.5} {:>8.5} {:>8.5} {:>8.4} {:>8.5} \
         {:>+8.5} {:>8.5} {:>+8.5} {:>+8.5}",
        name,
        mean_nll,
        mean_nll.exp(),
        mean(&difference),
        standard_error(&chunk_difference, &difference),
        mean(&kl),
        standard_error(&chunk_kl, &kl),
        mean(&agree),
        percentile(&sorted_kl, 50.0),
        percentile(&sorted_kl, 95.0),
        percentile(&sorted_kl, 99.0),
        sorted_kl.last().copied().unwrap_or(f64::NAN),
        standard_error(&chunk_agree, &agree),
        mean(&dp),
        rms,
        percentile(&sorted_dp, 1.0),
        percentile(&sorted_dp, 99.0)
    );
    Some(text)
}

/// The mean of `values`.
pub fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// The standard error of the mean: chunk-level over several chunks (tokens within a document are correlated),
/// token-level over one.
pub fn standard_error(per_chunk: &[f64], per_token: &[f64]) -> f64 {
    let values = if per_chunk.len() > 1 { per_chunk } else { per_token };
    let centre = mean(values);
    let squares: f64 = values.iter().map(|value| (value - centre) * (value - centre)).sum();
    let count = values.len() as f64;
    (squares / (count - 1.0)).sqrt() / count.sqrt()
}

/// The `q`-th percentile of sorted values, by linear interpolation (NumPy's default).
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let position = q / 100.0 * (sorted.len() - 1) as f64;
    let below = position.floor() as usize;
    let above = (sorted.len() - 1).min(below + 1);
    sorted[below] + (position - below as f64) * (sorted[above] - sorted[below])
}

/// `seconds` since the Unix epoch as the UTC instant the header carries: `2026-01-31T12:00:00Z`.
pub fn instant(seconds: u64) -> String {
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    // Civil date from days since 1970-01-01 (proleptic Gregorian).
    let shifted = days as i64 + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3600,
        rest % 3600 / 60,
        rest % 60
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate_linearly_like_numpy() {
        let sorted = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 0.0), 1.0);
        assert_eq!(percentile(&sorted, 50.0), 2.5);
        assert_eq!(percentile(&sorted, 100.0), 4.0);
        assert!((percentile(&sorted, 95.0) - 3.85).abs() < 1e-12);
    }

    #[test]
    fn the_standard_error_is_per_chunk_over_several_chunks_and_per_token_over_one() {
        let tokens = [1.0, 3.0, 5.0, 7.0];
        let chunks = [2.0, 6.0];
        // sqrt(8) / sqrt(2) over the chunks; sqrt(20 / 3) / sqrt(4) over the tokens.
        assert!((standard_error(&chunks, &tokens) - 2.0).abs() < 1e-12);
        assert!((standard_error(&chunks[..1], &tokens) - (20.0f64 / 3.0).sqrt() / 2.0).abs() < 1e-12);
    }

    #[test]
    fn instants_are_utc_to_the_second() {
        assert_eq!(instant(0), "1970-01-01T00:00:00Z");
        assert_eq!(instant(951_782_400 + 86_399), "2000-02-29T23:59:59Z");
        assert_eq!(instant(1_795_000_000), "2026-11-18T11:06:40Z");
    }

    #[test]
    fn the_lines_keep_the_columns_and_the_digits_of_the_java_summary() {
        // Two chunks of two positions: the candidate is worse than the reference by half a nat everywhere, so the sign
        // of the NLL difference shows, and the probability differences are unequal either side of zero, so the sign
        // of those shows.
        let row = |kl: f64, dp: f64| [1.5, 1.0, kl, 1.0, dp];
        let chunk = |a: (f64, f64), b: (f64, f64)| row(a.0, a.1).into_iter().chain(row(b.0, b.1)).collect::<Vec<f64>>();
        let scored = vec![chunk((0.0, -0.2), (0.5, 0.0)), chunk((0.5, 0.0), (1.0, 0.4))];
        let shape = Shape { context: 6, steps: 2 };
        let text = lines(
            "cand",
            "2026-01-01T00:00:00Z",
            Path::new("ref.kld"),
            Path::new("a.edrl"),
            shape,
            &scored,
        )
        .unwrap();
        let rows: Vec<&str> = text.lines().collect();
        assert_eq!(rows[0], "# cand 2026-01-01T00:00:00Z artifact a.edrl reference ref.kld");
        assert_eq!(
            rows[1],
            "2 chunk(s) of 6, 2 forced tokens each (positions 4..5); reference ref.kld"
        );
        assert!(rows[2].starts_with("report"));
        assert_eq!(rows[3], format!("{:<50} {:>9} {:>8}", "ref.kld", "1.00000", "2.7183"));
        let data: Vec<&str> = rows[4].split_whitespace().collect();
        assert_eq!(
            data,
            [
                "cand", "1.50000", "4.4817", "+0.50000", "0.00000", "0.500000", "0.250000", "1.0000", "0.50000",
                "0.92500", "0.98500", "1.0000", "0.00000", "+0.05000", "0.22361", "-0.19400", "+0.38800"
            ]
        );
    }

    #[test]
    fn nothing_is_summarised_without_positions() {
        let shape = Shape { context: 6, steps: 2 };
        assert!(lines("n", "t", Path::new("r"), Path::new("a"), shape, &[]).is_none());
        assert!(lines("n", "t", Path::new("r"), Path::new("a"), shape, &[vec![0.0; 3]]).is_none());
    }
}
