//! The sampler's cases, run host-only. The reference selection draws from a sequential stream; this one draws only
//! by position, so a case that repeated a selection to see the stream advance repeats it over positions instead.

use std::cell::Cell;

use super::random::Random;
use super::reference::ReferenceSampler;
use super::{SamplingConfig, SamplingError, TokenFilter, TokenSampler, unit_draw};

const INF: f32 = f32::INFINITY;
const NEG_INF: f32 = f32::NEG_INFINITY;
const SERVED_VOCABULARY: usize = 248_320;

fn config(temperature: f32, top_k: u32, top_p: f32, seed: u64) -> SamplingConfig {
    SamplingConfig::new(temperature, top_k, top_p, seed, false).unwrap()
}

fn sampler(config: SamplingConfig, vocabulary: usize) -> TokenSampler {
    TokenSampler::new(config, vocabulary).unwrap()
}

fn sampled(temperature: f32, top_k: u32, top_p: f32, seed: u64, vocabulary: usize) -> TokenSampler {
    sampler(config(temperature, top_k, top_p, seed), vocabulary)
}

fn pick(sampler: &mut TokenSampler, row: &[f32], position: u64) -> u32 {
    sampler.select(row, position).unwrap()
}

/// The tokens a sampler picks from `row` at positions `first..first + length`.
fn picks(sampler: &mut TokenSampler, row: &[f32], first: u64, length: u64) -> Vec<u32> {
    (first..first + length)
        .map(|position| pick(sampler, row, position))
        .collect()
}

// ---- GenerationConfigTest

#[test]
fn rejects_invalid_sampling_ranges() {
    let refused = |temperature, top_k, top_p| SamplingConfig::new(temperature, top_k, top_p, 0, false);
    assert_eq!(refused(f32::NAN, 0, 1.0), Err(SamplingError::Temperature));
    assert_eq!(refused(INF, 0, 1.0), Err(SamplingError::Temperature));
    assert_eq!(refused(1.0, 0, 0.0), Err(SamplingError::TopP));
    assert_eq!(refused(1.0, 0, f32::NAN), Err(SamplingError::TopP));
    assert_eq!(refused(1.0, 0, INF), Err(SamplingError::TopP));
    assert_eq!(refused(1.0, 0, 1.1), Err(SamplingError::TopP));
    assert_eq!(refused(1.0, 0, -0.5), Err(SamplingError::TopP));
    assert!(refused(1.0, 0, 1.0).is_ok());
    assert!(refused(-1.0, 0, f32::MIN_POSITIVE).is_ok());
}

#[test]
fn greedy_configuration_holds_the_documented_values() {
    let greedy = SamplingConfig::greedy(9);
    assert_eq!(greedy, SamplingConfig::new(0.0, 0, 1.0, 9, true).unwrap());
    assert!(greedy.is_greedy());
    assert!(!config(1.0, 0, 1.0, 9).is_greedy());
    assert!(config(0.0, 0, 1.0, 9).is_greedy());
    assert!(SamplingConfig::new(1.0, 0, 1.0, 9, true).unwrap().is_greedy());
}

#[test]
fn a_vocabulary_of_zero_is_refused() {
    assert_eq!(
        TokenSampler::new(SamplingConfig::greedy(0), 0).unwrap_err(),
        SamplingError::EmptyVocabulary
    );
}

// ---- TokenSamplerTest

#[test]
fn greedy_selection_returns_the_maximum_logit_and_breaks_ties_by_lowest_token_id() {
    let mut greedy = sampler(SamplingConfig::greedy(7), 4);
    let mut nonpositive = sampled(-0.5, 0, 1.0, 7, 2);
    assert_eq!(pick(&mut greedy, &[-2.0, -1.0, 3.0, 0.0], 0), 2);
    assert_eq!(pick(&mut greedy, &[1.0, 1.0, -3.0, NEG_INF], 0), 0);
    assert_eq!(pick(&mut nonpositive, &[-1.0, 2.0], 0), 1);
}

#[test]
fn explicit_greedy_mode_uses_argmax_even_when_temperature_and_filters_would_sample() {
    let mut sampler = sampler(SamplingConfig::new(1.0, 1, 0.1, 7, true).unwrap(), 3);
    assert_eq!(pick(&mut sampler, &[-2.0, 4.0, 3.0], 0), 1);
}

#[test]
fn top_k_one_always_selects_the_highest_logit() {
    let mut sampler = sampled(1.0, 1, 1.0, 31, 3);
    assert_eq!(pick(&mut sampler, &[-2.0, 4.0, 3.0], 0), 1);
}

#[test]
fn top_k_treats_signed_zero_as_a_tie_and_prefers_the_lower_token_id() {
    let mut sampler = sampled(1.0, 1, 1.0, 31, 2);
    assert_eq!(pick(&mut sampler, &[-0.0, 0.0], 0), 0);
}

#[test]
fn top_k_breaks_a_tie_by_token_id_at_the_retention_cutoff() {
    let mut sampler = sampled(1.0, 2, 1.0, 31, 4);
    for token in picks(&mut sampler, &[5.0, 4.0, 4.0, 0.0], 0, 32) {
        assert!(token == 0 || token == 1);
    }
}

#[test]
fn top_k_retains_only_the_highest_k_tokens_with_stable_ties() {
    let mut sampler = sampled(1.0, 2, 1.0, 2, 5);
    for token in picks(&mut sampler, &[3.0, 2.0, 2.0, 4.0, 0.0], 0, 32) {
        assert!(token == 0 || token == 3);
    }
}

#[test]
fn top_k_then_top_p_restricts_sampling_to_the_filtered_nucleus() {
    let mut sampler = sampled(1.0, 2, 0.75, 2, 4);
    let tokens = picks(&mut sampler, &[4.0, 3.0, 2.0, 1.0], 0, 64);
    assert!(tokens.iter().all(|&token| token == 0 || token == 1));
    assert!(tokens.contains(&1));
}

#[test]
fn top_p_near_one_retains_the_tail_until_the_threshold_is_met() {
    let mut sampler = sampled(1.0, 0, 0.999_999, 42, 3);
    let tokens = picks(&mut sampler, &[0.0, 0.0, 0.0], 0, 96);
    for id in 0..3 {
        assert!(tokens.contains(&id), "token {id} never selected");
    }
}

#[test]
fn sampling_approximately_follows_the_normalized_uniform_distribution() {
    let mut sampler = sampled(1.0, 0, 1.0, 42, 4);
    let mut counts = [0_u32; 4];
    for token in picks(&mut sampler, &[0.0; 4], 0, 8192) {
        counts[token as usize] += 1;
    }
    assert!(counts.iter().all(|&count| count > 500 && count < 2_500), "{counts:?}");
}

#[test]
fn top_p_handles_large_alternating_vocabulary_ordering() {
    let row: Vec<f32> = (0..SERVED_VOCABULARY)
        .map(|id| if id % 2 == 0 { -(id as f32) } else { id as f32 })
        .collect();
    let mut sampler = sampled(1.0, 0, 0.5, 29, row.len());
    assert_eq!(pick(&mut sampler, &row, 0) as usize, row.len() - 1);
}

#[test]
fn top_p_keeps_the_smallest_prefix_that_meets_the_probability_threshold() {
    let mut narrow = sampled(1.0, 0, 0.5, 17, 3);
    let mut larger = sampled(1.0, 0, 0.8, 17, 3);
    let row = [10.0, 9.0, 0.0];
    assert!(picks(&mut narrow, &row, 0, 32).iter().all(|&token| token == 0));
    let tokens = picks(&mut larger, &row, 0, 32);
    assert!(tokens.iter().all(|&token| token == 0 || token == 1));
    assert!(tokens.contains(&1));
}

#[test]
fn top_p_resolves_an_exact_threshold_tie_by_lower_token_id() {
    let mut sampler = sampled(1.0, 0, 0.5, 17, 2);
    assert!(picks(&mut sampler, &[0.0, 0.0], 0, 32).iter().all(|&token| token == 0));
}

#[test]
fn temperature_changes_the_filtered_sampling_distribution() {
    let row = [2.0, 1.0, 0.0];
    let mut cold = sampled(0.1, 0, 0.8, 1, 3);
    let mut warm = sampled(10.0, 0, 0.8, 1, 3);
    assert!(picks(&mut cold, &row, 0, 32).iter().all(|&token| token == 0));
    assert!(picks(&mut warm, &row, 0, 32).iter().any(|&token| token != 0));
}

#[test]
fn independent_samplers_with_the_same_seed_produce_the_same_sequence() {
    let row = [0.1, 0.4, 0.9, 0.2];
    let mut first = sampled(1.0, 0, 1.0, 1234, 4);
    let mut second = sampled(1.0, 0, 1.0, 1234, 4);
    assert_eq!(picks(&mut first, &row, 0, 64), picks(&mut second, &row, 0, 64));
}

#[test]
fn different_seeds_can_produce_different_sequences() {
    let row = [0.0; 4];
    let mut first = sampled(1.0, 0, 1.0, 3, 4);
    let mut second = sampled(1.0, 0, 1.0, 11, 4);
    assert_ne!(picks(&mut first, &row, 0, 64), picks(&mut second, &row, 0, 64));
}

#[test]
fn greedy_selection_rejects_rows_with_no_selectable_logits() {
    let mut sampler = sampler(SamplingConfig::greedy(9), 2);
    assert_eq!(
        sampler.select(&[f32::NAN, NEG_INF], 0),
        Err(SamplingError::NoSelectableToken)
    );
}

#[test]
fn stochastic_selection_rejects_rows_with_no_selectable_logits() {
    let mut sampler = sampled(1.0, 1, 1.0, 9, 2);
    assert_eq!(
        sampler.select(&[f32::NAN, NEG_INF], 0),
        Err(SamplingError::NoSelectableToken)
    );
}

#[test]
fn top_k_larger_than_the_selectable_vocabulary_keeps_all_selectable_tokens() {
    let mut sampler = sampled(1.0, 4, 1.0, 9, 3);
    assert_eq!(pick(&mut sampler, &[f32::NAN, NEG_INF, 5.0], 0), 2);
}

#[test]
fn top_p_handles_underflowed_tail_probabilities() {
    let mut sampler = sampled(1.0, 0, 0.999_999, 9, 3);
    assert!(
        picks(&mut sampler, &[0.0, -10_000.0, -20_000.0], 0, 16)
            .iter()
            .all(|&token| token == 0)
    );
}

#[test]
fn sampling_accepts_positive_infinity_and_excludes_nan_and_negative_infinity() {
    let mut sampler = sampled(1.0, 0, 1.0, 5, 4);
    for token in picks(&mut sampler, &[f32::NAN, INF, 100.0, INF], 0, 32) {
        assert!(token == 1 || token == 3);
    }
}

#[test]
fn top_p_samples_positive_infinities_with_equal_weight() {
    let mut sampler = sampled(1.0, 0, 0.75, 42, 3);
    let tokens = picks(&mut sampler, &[INF, INF, 5.0], 0, 64);
    assert!(tokens.iter().all(|&token| token == 0 || token == 1));
    assert!(tokens.contains(&0));
    assert!(tokens.contains(&1));
}

#[test]
fn top_p_at_a_positive_infinity_cutoff_prefers_the_first_tied_token() {
    let mut sampler = sampled(1.0, 0, 0.5, 42, 3);
    assert!(
        picks(&mut sampler, &[INF, INF, 5.0], 0, 32)
            .iter()
            .all(|&token| token == 0)
    );
}

#[test]
fn selection_rejects_vocabulary_mismatches() {
    let mut sampler = sampler(SamplingConfig::greedy(9), 2);
    assert_eq!(
        sampler.select(&[1.0], 0),
        Err(SamplingError::RowWidth { expected: 2, got: 1 })
    );
    assert_eq!(
        sampler.select_filtered(&mut [1.0, 2.0, 3.0], &Modulo::new(1, 0), 0),
        Err(SamplingError::RowWidth { expected: 2, got: 3 })
    );
}

#[test]
fn greedy_selection_supports_the_real_vocabulary_size() {
    let mut row = vec![0.0; SERVED_VOCABULARY];
    row[SERVED_VOCABULARY - 1] = 2.0;
    let mut sampler = sampler(SamplingConfig::greedy(0), row.len());
    assert_eq!(pick(&mut sampler, &row, 0) as usize, SERVED_VOCABULARY - 1);
}

fn keyed_sequence(seed: u64, row: &[f32], length: u64, first: u64) -> Vec<u32> {
    picks(&mut sampled(1.0, 0, 1.0, seed, row.len()), row, first, length)
}

#[test]
fn keyed_draws_of_neighbouring_positions_are_independent() {
    let sequence = keyed_sequence(1234, &[0.0, 0.0], 4001, 0);
    let mut pairs = [[0_u32; 2]; 2];
    for pair in sequence.as_chunks::<2>().0 {
        pairs[pair[0] as usize][pair[1] as usize] += 1;
    }
    // 2000 pairs over four cells: each expects 500; a correlated stream would empty off-diagonal cells.
    for count in pairs.iter().flatten() {
        assert!(*count > 400 && *count < 600, "{pairs:?}");
    }
}

#[test]
fn keyed_draws_depend_on_the_seed() {
    let row = [0.0; 4];
    assert_ne!(keyed_sequence(1234, &row, 64, 0), keyed_sequence(1235, &row, 64, 0));
    assert_eq!(keyed_sequence(1234, &row, 64, 0), keyed_sequence(1234, &row, 64, 0));
}

#[test]
fn keyed_calls_ignore_earlier_calls_and_call_order() {
    let row = [0.0; 4];
    let expected = keyed_sequence(1234, &row, 64, 0);
    let mut interleaved = sampled(1.0, 0, 1.0, 1234, 4);
    let mut actual = vec![0; 64];
    for position in (0..64).rev() {
        pick(&mut interleaved, &row, position + 1000);
        actual[position as usize] = pick(&mut interleaved, &row, position);
        interleaved
            .select_filtered(&mut row.clone(), &Modulo::new(1, 0), position + 7)
            .unwrap();
    }
    assert_eq!(expected, actual);
}

// ---- TokenSamplerKeyedTest

#[test]
fn a_position_samples_the_same_token_in_any_order() {
    let published = config(1.0, 20, 0.95, 1234);
    let row: Vec<f32> = (0..64).map(|id| f64::from(1 + id).ln() as f32).collect();
    let mut forward = sampler(published, 64);
    let mut backward = sampler(published, 64);
    let a = picks(&mut forward, &row, 1000, 200);
    let mut b = vec![0; 200];
    for p in (0..200).rev() {
        b[p] = pick(&mut backward, &row, 1000 + p as u64);
    }
    assert_eq!(a, b);
}

#[test]
fn draws_are_uniform() {
    let mut bins = [0_u32; 10];
    for position in 0..100_000 {
        let u = unit_draw(77, position);
        assert!((0.0..1.0).contains(&u));
        bins[(u * 10.0) as usize] += 1;
    }
    let chi: f64 = bins.iter().map(|&n| (f64::from(n) - 10_000.0).powi(2) / 10_000.0).sum();
    assert!(chi < 30.0, "chi-square {chi}");
}

#[test]
fn keyed_sampling_follows_the_distribution() {
    let mut sampler = sampled(1.0, 0, 1.0, 5, 2);
    let row = [0.0, 3.0_f64.ln() as f32];
    let ones: u32 = (0..20_000).map(|position| pick(&mut sampler, &row, position)).sum();
    assert!((f64::from(ones) / 20_000.0 - 0.75).abs() < 0.015);
}

#[test]
fn greedy_ignores_the_position() {
    let mut sampler = sampler(SamplingConfig::greedy(1), 3);
    assert_eq!(pick(&mut sampler, &[0.0, 2.0, 1.0], 5), 1);
    assert_eq!(pick(&mut sampler, &[0.0, 2.0, 1.0], 6), 1);
}

#[test]
fn a_constrained_keyed_draw_matches_the_masked_row() {
    let config = config(1.0, 20, 0.9, 9);
    let mut random = Random::new(3);
    let filter = Modulo::negated(3, 0);
    for trial in 0..200 {
        let row: Vec<f32> = (0..100).map(|_| random.gaussian() as f32).collect();
        let mut masked = row.clone();
        filter.mask_disallowed(&mut masked);
        let expected = pick(&mut sampler(config, 100), &masked, trial);
        let actual = sampler(config, 100)
            .select_filtered(&mut row.clone(), &filter, trial)
            .unwrap();
        assert_eq!(expected, actual, "trial {trial}");
    }
}

#[test]
fn unit_draw_is_the_split_mix_stream_and_wraps() {
    // The finalizer's output for seed 0 at position 0 is SplitMix64's first output for state 0 (0xE220A8397B1DCDAF).
    assert_eq!(
        unit_draw(0, 0),
        (0xE220_A839_7B1D_CDAF_u64 >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    );
    // Positions and seeds at the top of the range wrap instead of overflowing.
    let u = unit_draw(u64::MAX, u64::MAX);
    assert!((0.0..1.0).contains(&u));
    assert_eq!(unit_draw(u64::MAX, u64::MAX), unit_draw(u64::MAX, u64::MAX));
}

// ---- ConstrainedTokenSamplerTest

const VOCABULARY: usize = 3000;

/// Allows the tokens whose id has `residue` modulo `modulus` (or every other one, when negated), counting its tests.
struct Modulo {
    modulus: u32,
    residue: u32,
    negated: bool,
    tested: Cell<usize>,
}

impl Modulo {
    fn new(modulus: u32, residue: u32) -> Self {
        Self {
            modulus,
            residue,
            negated: false,
            tested: Cell::new(0),
        }
    }

    fn negated(modulus: u32, residue: u32) -> Self {
        Self {
            negated: true,
            ..Self::new(modulus, residue)
        }
    }
}

impl TokenFilter for Modulo {
    fn allows(&self, id: u32) -> bool {
        self.tested.set(self.tested.get() + 1);
        (id % self.modulus == self.residue) != self.negated
    }
}

struct Nothing;

impl TokenFilter for Nothing {
    fn allows(&self, _: u32) -> bool {
        false
    }
}

/// Coarse values make ties common; a few NaN, negative infinities and signed zeros.
fn coarse_row(random: &mut Random) -> Vec<f32> {
    (0..VOCABULARY)
        .map(|_| match random.below(100) {
            0 => f32::NAN,
            1 => NEG_INF,
            2 => -0.0,
            _ => random.below(40) as f32 / 4.0 - 5.0,
        })
        .collect()
}

/// From permissive (most tokens) to restrictive (a handful), so both the ranked and the full paths run.
fn random_filter(random: &mut Random) -> Modulo {
    let restrictive = random.boolean();
    let modulus = 1 + random.below(if restrictive { 4 } else { 600 });
    let residue = random.below(modulus);
    Modulo::new(modulus, residue)
}

fn masked(row: &[f32], filter: &dyn TokenFilter) -> Vec<f32> {
    let mut masked = row.to_vec();
    filter.mask_disallowed(&mut masked);
    masked
}

#[test]
fn greedy_selection_equals_the_masked_argmax() {
    let mut random = Random::new(1);
    for trial in 0..400 {
        let row = coarse_row(&mut random);
        let filter = random_filter(&mut random);
        let expected = pick(
            &mut sampler(SamplingConfig::greedy(5), VOCABULARY),
            &masked(&row, &filter),
            0,
        );
        let actual = sampler(SamplingConfig::greedy(5), VOCABULARY).select_filtered(&mut row.clone(), &filter, 0);
        assert_eq!(Ok(expected), actual, "trial {trial}");
    }
}

#[test]
fn top_k_top_p_sampling_equals_the_masked_row_draw_for_draw() {
    let mut random = Random::new(2);
    for trial in 0..200 {
        let row = coarse_row(&mut random);
        let filter = random_filter(&mut random);
        let config = config(0.7 + random.float(), 20, 0.95, random.next_u64());
        let mut by_mask = sampler(config, VOCABULARY);
        let mut lazy = sampler(config, VOCABULARY);
        let masked_row = masked(&row, &filter);
        for position in 0..5 {
            assert_eq!(
                Ok(pick(&mut by_mask, &masked_row, position)),
                lazy.select_filtered(&mut row.clone(), &filter, position),
                "trial {trial}"
            );
        }
    }
}

#[test]
fn without_top_p_the_draw_stays_among_the_masked_top_k() {
    let mut random = Random::new(3);
    let mut sampler = sampled(1.0, 20, 1.0, 9, VOCABULARY);
    for trial in 0..200 {
        let row = coarse_row(&mut random);
        let filter = random_filter(&mut random);
        let masked_row = masked(&row, &filter);
        let selected = sampler.select_filtered(&mut row.clone(), &filter, trial).unwrap();
        assert!(filter.allows(selected));
        let better = masked_row
            .iter()
            .zip(0_u32..)
            .filter(|&(&logit, id)| {
                let own = masked_row[selected as usize];
                logit > own || (logit == own && id < selected)
            })
            .count();
        assert!(better < 20, "selected outside the top 20 allowed tokens");
    }
}

#[test]
fn an_unconstraining_filter_is_tested_only_on_the_best_candidates() {
    let mut random = Random::new(4);
    let everything = Modulo::new(1, 0);
    sampled(1.0, 20, 0.95, 1, VOCABULARY)
        .select_filtered(&mut coarse_row(&mut random), &everything, 0)
        .unwrap();
    assert_eq!(everything.tested.get(), 20);
    let everything = Modulo::new(1, 0);
    sampler(SamplingConfig::greedy(1), VOCABULARY)
        .select_filtered(&mut coarse_row(&mut random), &everything, 0)
        .unwrap();
    assert_eq!(everything.tested.get(), 1);
}

#[test]
fn a_filter_that_rejects_everything_fails() {
    let mut random = Random::new(5);
    let result =
        sampler(SamplingConfig::greedy(1), VOCABULARY).select_filtered(&mut coarse_row(&mut random), &Nothing, 0);
    assert_eq!(result, Err(SamplingError::NoSelectableToken));
}

#[test]
fn a_filtered_selection_after_a_masked_one_sees_the_row_it_was_given() {
    // The masked path overwrites the caller's row; the next call brings its own.
    let mut sampler = sampled(1.0, 2, 1.0, 4, 6);
    let filter = Modulo::new(6, 5);
    let mut row = [3.0, 2.0, 1.0, 0.5, 0.25, 0.1];
    assert_eq!(sampler.select_filtered(&mut row, &filter, 0), Ok(5));
    assert_eq!(row[..5], [NEG_INF; 5]);
}

// ---- bf16 rows

fn bf16(row: &[f32]) -> Vec<u8> {
    row.iter()
        .flat_map(|logit| ((logit.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

#[test]
fn a_bf16_row_selects_what_its_f32_row_selects() {
    let mut random = Random::new(6);
    let filter = Modulo::new(7, 3);
    let config = config(0.9, 20, 0.95, 77);
    let mut from_bytes = sampler(config, VOCABULARY);
    let mut from_floats = sampler(config, VOCABULARY);
    for position in 0..50 {
        let bytes = bf16(&coarse_row(&mut random));
        // The f32 row of the same bytes: BF16 is the upper half of an f32.
        let row: Vec<f32> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| f32::from_bits(u32::from(u16::from_le_bytes(b)) << 16))
            .collect();
        assert_eq!(
            from_bytes.select_bf16(&bytes, position, None),
            from_floats.select(&row, position)
        );
        assert_eq!(
            from_bytes.select_bf16(&bytes, position, Some(&filter)),
            from_floats.select_filtered(&mut row.clone(), &filter, position)
        );
    }
}

#[test]
fn a_bf16_row_of_the_wrong_size_is_refused() {
    let mut sampler = sampler(SamplingConfig::greedy(1), 3);
    assert_eq!(
        sampler.select_bf16(&[0; 5], 0, None),
        Err(SamplingError::RowWidth { expected: 3, got: 2 })
    );
    assert_eq!(sampler.select_bf16(&bf16(&[0.0, 1.0, 0.5]), 0, None), Ok(1));
}

// ---- TokenSamplerEquivalenceTest

const TEMPERATURES: [f32; 8] = [
    1.0,
    0.7,
    2.5,
    1.0e-3,
    1.0e-30,
    f32::from_bits(1),
    f32::MIN_POSITIVE,
    3.0e38,
];
const TOP_PS: [f32; 4] = [1.0, 0.95, 0.5, 1.0e-4];

fn special(random: &mut Random) -> f32 {
    match random.below(9) {
        0 => f32::NAN,
        1 => INF,
        2 => NEG_INF,
        3 => -0.0,
        4 => 0.0,
        5 => f32::from_bits(1),
        6 => -f32::from_bits(1),
        _ => random.gaussian() as f32,
    }
}

/// Realistic, tied, special-valued and sparse rows, cycling by row index.
fn equivalence_row(random: &mut Random, vocabulary: usize, index: usize) -> Vec<f32> {
    let mut row = vec![0.0_f32; vocabulary];
    match index % 6 {
        0 => row
            .iter_mut()
            .for_each(|logit| *logit = (random.gaussian() * 3.0) as f32),
        // Few distinct values: ties everywhere, including at the maximum and the top-k cutoff.
        1 => row.iter_mut().for_each(|logit| *logit = random.below(4) as f32 * 0.5),
        2 => row.iter_mut().for_each(|logit| *logit = special(random)),
        3 => {
            // Mostly unselectable: fewer candidates than top-k, sometimes none.
            row.iter_mut().for_each(|logit| {
                *logit = if random.below(8) == 0 {
                    random.gaussian() as f32
                } else {
                    NEG_INF
                }
            });
            if random.below(4) == 0 {
                row[random.below(vocabulary as u32) as usize] = f32::NAN;
            }
        }
        4 => row
            .iter_mut()
            .for_each(|logit| *logit = if random.boolean() { 0.0 } else { -0.0 }),
        _ => {
            let extremes = [
                f32::MAX,
                -f32::MAX,
                f32::from_bits(1),
                -f32::from_bits(1),
                f32::MIN_POSITIVE,
            ];
            row.iter_mut().for_each(|logit| {
                *logit = if random.below(3) == 0 {
                    extremes[random.below(5) as usize]
                } else {
                    (random.gaussian() * 1.0e3) as f32
                }
            });
        }
    }
    row
}

#[test]
fn greedy_selection_matches_the_reference_on_every_row_shape() {
    let mut random = Random::new(11);
    for vocabulary in [1, 2, 3, 7, 64, 1000, 4099] {
        let config = SamplingConfig::greedy(5);
        let mut sampler = sampler(config, vocabulary);
        let reference = ReferenceSampler::new(config, vocabulary);
        for index in 0..300 {
            let row = equivalence_row(&mut random, vocabulary, index);
            let position = index as u64;
            assert_eq!(
                reference.select(&row, unit_draw(5, position)),
                sampler.select(&row, position),
                "vocabulary {vocabulary}"
            );
        }
    }
}

#[test]
fn stochastic_selection_matches_the_reference_draw_for_draw() {
    let mut random = Random::new(23);
    let mut compared = 0;
    for vocabulary in [1_usize, 2, 5, 33, 600, 2049] {
        let top_ks = [
            0,
            1,
            2,
            5,
            20,
            vocabulary as u32 - 1,
            vocabulary as u32,
            vocabulary as u32 + 3,
        ];
        for temperature in TEMPERATURES {
            for top_k in top_ks {
                for top_p in TOP_PS {
                    let seed = random.next_u64();
                    let config = config(temperature, top_k, top_p, seed);
                    let mut sampler = sampler(config, vocabulary);
                    let reference = ReferenceSampler::new(config, vocabulary);
                    for index in 0..12 {
                        let row = equivalence_row(&mut random, vocabulary, index);
                        let position = random.next_u64() >> 1;
                        assert_eq!(
                            reference.select(&row, unit_draw(seed, position)),
                            sampler.select(&row, position),
                            "vocabulary {vocabulary} {config:?}"
                        );
                        compared += 1;
                    }
                }
            }
        }
    }
    assert!(compared > 10_000);
}

#[test]
fn stochastic_selection_matches_the_reference_at_the_served_vocabulary() {
    let mut random = Random::new(31);
    let config = config(1.0, 20, 0.95, 99);
    let mut sampler = sampler(config, SERVED_VOCABULARY);
    let reference = ReferenceSampler::new(config, SERVED_VOCABULARY);
    for index in 0..40 {
        let row = equivalence_row(&mut random, SERVED_VOCABULARY, index);
        let position = index as u64;
        assert_eq!(
            reference.select(&row, unit_draw(99, position)),
            sampler.select(&row, position)
        );
    }
}
