//! The decisions the quantizer takes on the host: the sign vectors, the tiles the global scale is searched on, the
//! candidate scales and the refinement of the best one. Pure functions of their inputs, so the choices are tested
//! without a device.

use std::array::from_fn;
use std::cmp::Ordering;

use crate::java_random::SplittableRandom;

/// The share of the mean Hessian diagonal added as damping, and (twice) per Cholesky retry.
pub const SIGMA_REG: f64 = 0.025;

/// The most times a failed factorization is retried with more damping.
pub const CHOLESKY_RETRIES: u32 = 10;

/// Rows of the Hessian that LDLQ groups: the error of a span is fed back to the earlier rows in one product.
pub(super) const STRIP_SPAN: u32 = 128;

/// Width, in tiles, of the wrapped diagonal the scale search samples.
const DIAGONAL_WIDTH: usize = 3;

/// The coarse candidates: 0.1 to 1.9 in steps of 0.2.
pub(super) const COARSE: usize = 10;

/// The fine candidates around the best coarse one.
pub(super) const FINE: usize = 5;

/// The spacing of the fine candidates.
const FINE_STEP: f64 = 0.075;

/// Every third sample is rounded for the coarse candidates, every sample for the fine ones.
pub(super) const COARSE_STRIDE: u32 = 3;

/// `count` signs, each `1.0` or `-1.0`, from the generator in the order the generator's boolean draws give them.
pub(super) fn signs(random: &mut SplittableRandom, count: usize) -> Vec<f32> {
    (0..count)
        .map(|_| if random.next_boolean() { 1.0 } else { -1.0 })
        .collect()
}

/// How far the feedback of the rounding error grows the weights, by rate: the sampled tiles are scaled up by this so
/// the search sees what LDLQ will round.
pub(super) fn drift(bits: f64) -> f32 {
    let drift: f64 = if bits == 2.0 {
        1.018
    } else if bits == 2.5 {
        1.009
    } else if bits == 3.0 {
        1.004
    } else {
        1.0
    };
    drift as f32
}

/// The tiles the global scale is searched on, as tile coordinates in the `[k / 16, n / 16]` tile grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Samples {
    pub(super) tile_k: Vec<i32>,
    pub(super) tile_n: Vec<i32>,
}

impl Samples {
    pub(super) fn count(&self) -> usize {
        self.tile_k.len()
    }
}

/// How many tiles are sampled from a grid of `tiles_k` by `tiles_n`: a wrapped diagonal of width three plus the
/// highest and the lowest energy tiles.
pub(super) fn sample_count(tiles_k: usize, tiles_n: usize) -> usize {
    let (diagonal, extreme) = diagonal_and_extreme(tiles_k, tiles_n);
    diagonal * DIAGONAL_WIDTH + 2 * extreme
}

fn diagonal_and_extreme(tiles_k: usize, tiles_n: usize) -> (usize, usize) {
    let diagonal = tiles_k.max(tiles_n);
    let total = tiles_k * tiles_n;
    let extreme = (diagonal * DIAGONAL_WIDTH / 16).max(8).min(total.div_ceil(2));
    (diagonal, extreme)
}

/// The sampled tiles: the wrapped diagonal, then the `extreme` highest-energy tiles (highest first), then the
/// `extreme` lowest. Ties in energy are broken by the tile's index.
pub(super) fn sample_tiles(tiles_k: usize, tiles_n: usize, energy: &[f32]) -> Samples {
    let total = tiles_k * tiles_n;
    debug_assert_eq!(energy.len(), total, "one energy per tile");
    let (diagonal, extreme) = diagonal_and_extreme(tiles_k, tiles_n);
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| compare(energy[a], energy[b]).then(a.cmp(&b)));
    let samples = diagonal * DIAGONAL_WIDTH + 2 * extreme;
    let (mut tile_k, mut tile_n) = (Vec::with_capacity(samples), Vec::with_capacity(samples));
    for i in 0..diagonal {
        for j in 0..DIAGONAL_WIDTH {
            tile_k.push((i % tiles_k) as i32);
            tile_n.push(((i + j) % tiles_n) as i32);
        }
    }
    for i in 0..extreme {
        let high = order[total - 1 - i];
        tile_k.push((high / tiles_n) as i32);
        tile_n.push((high % tiles_n) as i32);
    }
    for &low in &order[..extreme] {
        tile_k.push((low / tiles_n) as i32);
        tile_n.push((low % tiles_n) as i32);
    }
    Samples { tile_k, tile_n }
}

/// Orders energies with every NaN last, as a total order on the values.
fn compare(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => f64::from(a).total_cmp(&f64::from(b)),
    }
}

/// The coarse candidate scales.
pub(super) fn coarse_scales() -> [f64; COARSE] {
    from_fn(|i| 0.1 + 0.2 * i as f64)
}

/// The index of the smallest error; the first of equal ones.
pub(super) fn smallest(errors: &[f32]) -> usize {
    let mut best = 0;
    for (index, error) in errors.iter().enumerate().skip(1) {
        if *error < errors[best] {
            best = index;
        }
    }
    best
}

/// The fine candidate scales around `center`.
pub(super) fn fine_scales(center: f64) -> [f64; FINE] {
    from_fn(|i| center + FINE_STEP * (i as f64 - 2.0))
}

/// The global scale: the best fine candidate, moved by the minimum of the parabola through it and its neighbours
/// (at most half a step) when it is not at an end, and never below 0.01.
pub(super) fn refined_scale(fine: &[f64; FINE], errors: &[f32]) -> f32 {
    let best = smallest(errors);
    let mut offset = 0.0;
    if best > 0 && best < FINE - 1 {
        let (y0, y1, y2) = (
            f64::from(errors[best - 1]),
            f64::from(errors[best]),
            f64::from(errors[best + 1]),
        );
        let denominator = y0 - 2.0 * y1 + y2;
        offset = if denominator > 0.0 {
            0.5 * (y0 - y2) / denominator
        } else {
            0.0
        };
        offset = offset.clamp(-0.5, 0.5);
    }
    (fine[best] + offset * FINE_STEP).max(0.01) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_signs_follow_the_generator_one_bit_at_a_time() {
        let mut random = SplittableRandom::new(0);
        let drawn: Vec<bool> = {
            let mut again = SplittableRandom::new(0);
            (0..40).map(|_| again.next_boolean()).collect()
        };
        let signs = signs(&mut random, 40);
        for (sign, bit) in signs.iter().zip(&drawn) {
            assert_eq!(*sign, if *bit { 1.0 } else { -1.0 });
        }
        assert!(
            signs.contains(&1.0) && signs.contains(&-1.0),
            "a fair generator gives both"
        );
    }

    #[test]
    fn the_input_signs_are_drawn_before_the_output_signs() {
        let mut random = SplittableRandom::new(7);
        let first = signs(&mut random, 128);
        let second = signs(&mut random, 64);
        let mut together = SplittableRandom::new(7);
        let all = signs(&mut together, 192);
        assert_eq!(&all[..128], &first[..]);
        assert_eq!(&all[128..], &second[..]);
    }

    #[test]
    fn the_diagonal_wraps_and_the_extremes_are_the_hottest_and_coldest_tiles() {
        let (tiles_k, tiles_n) = (8usize, 4usize);
        let energy: Vec<f32> = (0..tiles_k * tiles_n).map(|i| ((i * 7) % 32) as f32).collect();
        let samples = sample_tiles(tiles_k, tiles_n, &energy);
        assert_eq!(samples.count(), sample_count(tiles_k, tiles_n));
        // eight diagonal rows of three, then 8 high and 8 low (a grid of 32 tiles holds at most 16 of each)
        assert_eq!(samples.count(), 8 * 3 + 2 * 8);
        for i in 0..8usize {
            for j in 0..3usize {
                assert_eq!(samples.tile_k[i * 3 + j], (i % tiles_k) as i32);
                assert_eq!(samples.tile_n[i * 3 + j], ((i + j) % tiles_n) as i32);
            }
        }
        let tile_energy = |at: usize| energy[samples.tile_k[at] as usize * tiles_n + samples.tile_n[at] as usize];
        assert_eq!(tile_energy(24), 31.0, "the hottest tile comes first");
        assert_eq!(tile_energy(24 + 8), 0.0, "then the coldest");
        for i in 1..8 {
            assert!(tile_energy(24 + i) <= tile_energy(24 + i - 1), "hottest first");
            assert!(tile_energy(24 + 8 + i) >= tile_energy(24 + 8 + i - 1), "coldest first");
        }
    }

    #[test]
    fn equal_energies_are_taken_in_tile_order() {
        let samples = sample_tiles(8, 8, &[1.0; 64]);
        let at = 8 * 3;
        // the coldest are the first tiles, the hottest the last ones, each tie broken by index
        assert_eq!((samples.tile_k[at], samples.tile_n[at]), (7, 7));
        let low = at + 8;
        assert_eq!((samples.tile_k[low], samples.tile_n[low]), (0, 0));
        assert_eq!((samples.tile_k[low + 1], samples.tile_n[low + 1]), (0, 1));
    }

    #[test]
    fn a_tiny_grid_cannot_sample_more_extremes_than_half_its_tiles() {
        // 8 x 8 weights is four tiles: the extremes are capped at (4 + 1) / 2
        assert_eq!(sample_count(2, 2), 2 * 3 + 2 * 2);
    }

    #[test]
    fn a_tie_between_candidates_takes_the_first() {
        assert_eq!(smallest(&[3.0, 1.0, 1.0, 2.0]), 1);
    }

    #[test]
    fn the_parabola_moves_the_scale_toward_the_lower_neighbour_and_never_past_half_a_step() {
        let fine = fine_scales(1.0);
        assert!((fine[2] - 1.0).abs() < 1e-12 && (fine[0] - 0.85).abs() < 1e-12);
        // a symmetric dip stays put
        assert_eq!(refined_scale(&fine, &[4.0, 1.0, 0.0, 1.0, 4.0]), 1.0);
        // a lower right neighbour moves it up
        let up = refined_scale(&fine, &[4.0, 3.0, 0.0, 0.5, 4.0]);
        assert!(up > 1.0 && up <= (1.0 + 0.5 * FINE_STEP) as f32 + 1e-6, "{up}");
        // the best at an end is not refined
        assert_eq!(refined_scale(&fine, &[0.0, 1.0, 2.0, 3.0, 4.0]), fine[0] as f32);
        // a flat parabola is not refined and the result is never below a hundredth
        assert_eq!(refined_scale(&fine_scales(0.0), &[2.0, 2.0, 1.0, 2.0, 2.0]), 0.01);
    }
}
