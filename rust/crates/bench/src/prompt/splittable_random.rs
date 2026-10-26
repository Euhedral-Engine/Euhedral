//! `java.util.SplittableRandom` for a fixed seed, so a seed picks the same words as the Java benchmark does.

/// The odd constant Java adds to the seed on every step (`SplittableRandom.GOLDEN_GAMMA`).
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// A `SplittableRandom` made by `new SplittableRandom(seed)`. It supports what the corpora draw: `nextInt()` and
/// `nextInt(bound)`.
#[derive(Debug, Clone)]
pub struct SplittableRandom {
    seed: u64,
}

impl SplittableRandom {
    pub fn new(seed: u64) -> SplittableRandom {
        SplittableRandom { seed }
    }

    /// `nextInt()`: the next 32 random bits.
    pub fn next_int(&mut self) -> i32 {
        self.seed = self.seed.wrapping_add(GOLDEN_GAMMA);
        mix32(self.seed)
    }

    /// `nextInt(bound)`: uniform in `0..bound`, drawing exactly as Java does (a power of two masks the low bits; any
    /// other bound rejects the draws that would bias the remainder). `None` when `bound` is not positive.
    pub fn next_int_bounded(&mut self, bound: i32) -> Option<i32> {
        if bound <= 0 {
            return None;
        }
        let mask = bound - 1;
        let mut draw = self.next_int();
        if bound & mask == 0 {
            return Some(draw & mask);
        }
        // `u >>> 1` is the draw without its sign bit.
        let mut shifted = ((draw as u32) >> 1) as i32;
        loop {
            draw = shifted % bound;
            // Java's int arithmetic wraps; a negative sum marks a draw from the biased tail.
            if shifted.wrapping_add(mask).wrapping_sub(draw) >= 0 {
                return Some(draw);
            }
            shifted = ((self.next_int() as u32) >> 1) as i32;
        }
    }
}

/// `SplittableRandom.mix32`.
fn mix32(mut z: u64) -> i32 {
    z = (z ^ (z >> 33)).wrapping_mul(0x62a9_d9ed_7997_05f5);
    ((z ^ (z >> 28)).wrapping_mul(0xcb24_d0a5_c88c_35b3) >> 32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 20_260_925;

    fn draws(bound: i32) -> Vec<i32> {
        let mut random = SplittableRandom::new(SEED);
        (0..16).map(|_| random.next_int_bounded(bound).unwrap()).collect()
    }

    // The expected values were printed by `jshell` (Java 25): `new SplittableRandom(20260925L)` and 16 calls of
    // `nextInt(bound)`.
    #[test]
    fn splittable_random_matches_java_sequence() {
        assert_eq!(draws(64), [36, 49, 6, 25, 30, 62, 47, 44, 46, 0, 12, 3, 3, 13, 44, 39]);
    }

    #[test]
    fn bounds_that_are_not_powers_of_two_match_java() {
        assert_eq!(draws(10), [0, 2, 9, 6, 1, 3, 1, 8, 1, 0, 2, 5, 1, 4, 4, 5]);
        assert_eq!(
            draws(1000),
            [
                90, 752, 499, 156, 951, 663, 791, 758, 111, 440, 102, 825, 81, 934, 14, 435
            ]
        );
    }

    #[test]
    fn large_bounds_match_java() {
        assert_eq!(
            draws(1 << 30),
            [
                530_872_356,
                840_419_505,
                309_815_174,
                290_280_665,
                302_996_254,
                530_667_326,
                120_267_759,
                952_185_516,
                1_026_040_750,
                1_016_673_408,
                710_608_204,
                330_744_003,
                641_408_515,
                866_256_397,
                1_045_432_556,
                1_068_751_399
            ]
        );
        assert_eq!(
            draws((1 << 30) + 1),
            [
                802_307_090,
                420_209_752,
                691_778_499,
                265_333_663,
                597_004_791,
                476_092_758,
                355_304_102,
                539_136_301,
                85_770_895,
                935_456_980,
                627_561_648,
                938_140_743,
                902_189_155,
                192_229_640,
                933_011_091,
                244_281_531
            ]
        );
    }

    #[test]
    fn next_int_matches_java() {
        let mut random = SplittableRandom::new(SEED);
        let drawn: Vec<i32> = (0..4).map(|_| random.next_int()).collect();
        assert_eq!(drawn, [1_604_614_180, 840_419_505, 1_383_556_998, -1_857_202_983]);
    }

    #[test]
    fn a_bound_that_is_not_positive_draws_nothing() {
        let mut random = SplittableRandom::new(SEED);
        assert_eq!(random.next_int_bounded(0), None);
        assert_eq!(random.next_int_bounded(-3), None);
        // No draw was consumed.
        assert_eq!(random.next_int(), 1_604_614_180);
    }
}
