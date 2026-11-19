//! Seeded pseudo-random generators in the style of the Java standard library's `java.util.Random` and
//! `java.util.SplittableRandom`: a seed gives the same draws every time. No test pins the sequence of a seed; a consumer
//! that needs it adds its own.

mod fdlibm;

const MULTIPLIER: i64 = 0x5_DEEC_E66D;
const ADDEND: i64 = 0xB;
const MASK: i64 = (1 << 48) - 1;
/// `1 / 2^53`, the step of `nextDouble`.
const DOUBLE_UNIT: f64 = 1.0 / (1u64 << 53) as f64;

/// `java.util.Random`: a 48-bit linear congruential generator.
#[derive(Debug, Clone)]
pub struct Random {
    seed: i64,
    next_gaussian: Option<f64>,
}

impl Random {
    /// `new Random(seed)`.
    pub fn new(seed: i64) -> Random {
        Random {
            seed: (seed ^ MULTIPLIER) & MASK,
            next_gaussian: None,
        }
    }

    /// `next(bits)`: the top `bits` of the next 48-bit state, as a signed int.
    fn next(&mut self, bits: u32) -> i32 {
        self.seed = self.seed.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND) & MASK;
        (self.seed >> (48 - bits)) as i32
    }

    /// `nextInt()`: the next 32 random bits.
    pub fn next_int(&mut self) -> i32 {
        self.next(32)
    }

    /// `nextInt(bound)`: uniform in `0..bound`. A power of two takes the high bits; any other bound rejects the draws
    /// that would bias the remainder. `None` when `bound` is not positive.
    pub fn next_int_bounded(&mut self, bound: i32) -> Option<i32> {
        if bound <= 0 {
            return None;
        }
        let mut bits = self.next(31);
        if bound & (bound - 1) == 0 {
            return Some(((i64::from(bound) * i64::from(bits)) >> 31) as i32);
        }
        loop {
            let value = bits % bound;
            // Java's int arithmetic wraps; a negative sum marks a draw from the biased tail.
            if bits.wrapping_sub(value).wrapping_add(bound - 1) >= 0 {
                return Some(value);
            }
            bits = self.next(31);
        }
    }

    /// `nextLong()`: two draws of 32 bits, the second added as a signed value.
    pub fn next_long(&mut self) -> i64 {
        (i64::from(self.next(32)) << 32).wrapping_add(i64::from(self.next(32)))
    }

    /// `nextBoolean()`.
    pub fn next_boolean(&mut self) -> bool {
        self.next(1) != 0
    }

    /// `nextDouble()`: uniform in `[0, 1)` with 53 random bits.
    pub fn next_double(&mut self) -> f64 {
        let high = i64::from(self.next(26)) << 27;
        (high + i64::from(self.next(27))) as f64 * DOUBLE_UNIT
    }

    /// `nextFloat()`: uniform in `[0, 1)` with 24 random bits.
    pub fn next_float(&mut self) -> f32 {
        self.next(24) as f32 / (1 << 24) as f32
    }

    /// `nextGaussian()`: the polar method, which makes two values per round and keeps the second.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(kept) = self.next_gaussian.take() {
            return kept;
        }
        loop {
            let v1 = 2.0 * self.next_double() - 1.0;
            let v2 = 2.0 * self.next_double() - 1.0;
            let s = v1 * v1 + v2 * v2;
            if s < 1.0 && s != 0.0 {
                let multiplier = (-2.0 * fdlibm::log(s) / s).sqrt();
                self.next_gaussian = Some(v2 * multiplier);
                return v1 * multiplier;
            }
        }
    }
}

/// The odd constant Java adds to the seed on every step (`SplittableRandom.GOLDEN_GAMMA`).
const GOLDEN_GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;

/// A `SplittableRandom`: made by `new SplittableRandom(seed)` or by [`split`](SplittableRandom::split).
#[derive(Debug, Clone)]
pub struct SplittableRandom {
    seed: u64,
    gamma: u64,
}

impl SplittableRandom {
    /// `new SplittableRandom(seed)`.
    pub fn new(seed: u64) -> SplittableRandom {
        SplittableRandom {
            seed,
            gamma: GOLDEN_GAMMA,
        }
    }

    fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(self.gamma);
        self.seed
    }

    /// `nextInt()`: the next 32 random bits.
    pub fn next_int(&mut self) -> i32 {
        let seed = self.next_seed();
        mix32(seed)
    }

    /// `nextLong()`.
    pub fn next_long(&mut self) -> i64 {
        let seed = self.next_seed();
        mix64(seed) as i64
    }

    /// `nextDouble()`: uniform in `[0, 1)` with 53 random bits.
    pub fn next_double(&mut self) -> f64 {
        (self.next_long() as u64 >> 11) as f64 * DOUBLE_UNIT
    }

    /// `nextBoolean()`: the sign of the next int.
    pub fn next_boolean(&mut self) -> bool {
        self.next_int() < 0
    }

    /// `split()`: a new generator whose seed and gamma come from this one's next draws.
    pub fn split(&mut self) -> SplittableRandom {
        let seed = self.next_long() as u64;
        let gamma = mix_gamma(self.next_seed());
        SplittableRandom { seed, gamma }
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

/// `SplittableRandom.mix64`.
fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// `SplittableRandom.mixGamma`: an odd gamma with enough bit transitions.
fn mix_gamma(mut z: u64) -> u64 {
    z = (z ^ (z >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    z = (z ^ (z >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    z = (z ^ (z >> 33)) | 1;
    if (z ^ (z >> 1)).count_ones() < 24 {
        z ^ 0xaaaa_aaaa_aaaa_aaaa
    } else {
        z
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: u64 = 20_260_925;

    fn draws(bound: i32) -> Vec<i32> {
        let mut random = SplittableRandom::new(SEED);
        (0..16).map(|_| random.next_int_bounded(bound).unwrap()).collect()
    }

    /// How many of each value in `0..bound` come up in `count` draws.
    fn histogram(bound: i32, count: usize) -> Vec<usize> {
        let mut random = SplittableRandom::new(SEED);
        let mut seen = vec![0; bound as usize];
        for _ in 0..count {
            seen[random.next_int_bounded(bound).unwrap() as usize] += 1;
        }
        seen
    }

    #[test]
    fn the_same_seed_gives_the_same_draws_and_another_seed_other_ones() {
        assert_eq!(draws(1000), draws(1000));
        let mut other = SplittableRandom::new(SEED + 1);
        let others: Vec<i32> = (0..16).map(|_| other.next_int_bounded(1000).unwrap()).collect();
        assert_ne!(draws(1000), others);
        let mut first = SplittableRandom::new(SEED);
        let mut second = SplittableRandom::new(SEED);
        for _ in 0..64 {
            assert_eq!(first.next_int(), second.next_int());
        }
    }

    #[test]
    fn a_bounded_draw_is_in_range_for_every_kind_of_bound() {
        for bound in [1, 2, 3, 10, 64, 1000, 1 << 30, (1 << 30) + 1, i32::MAX] {
            let mut random = SplittableRandom::new(SEED);
            for _ in 0..2000 {
                let drawn = random.next_int_bounded(bound).unwrap();
                assert!((0..bound).contains(&drawn), "bound {bound}: {drawn}");
            }
        }
    }

    #[test]
    fn bounded_draws_reach_every_value_about_equally() {
        // 64 is a power of two and 10 is not: both reach every value, none far from its fair share.
        for (bound, count) in [(64, 64_000), (10, 20_000), (7, 14_000)] {
            let seen = histogram(bound, count);
            let share = count / bound as usize;
            for (value, times) in seen.iter().enumerate() {
                assert!(
                    times * 10 > share * 8 && times * 10 < share * 12,
                    "bound {bound}: {value} came up {times} times, a fair share is {share}"
                );
            }
        }
    }

    #[test]
    fn the_draws_of_a_stream_do_not_repeat_and_use_both_signs() {
        let mut random = SplittableRandom::new(SEED);
        let drawn: Vec<i32> = (0..1000).map(|_| random.next_int()).collect();
        assert!(drawn.iter().any(|value| *value < 0) && drawn.iter().any(|value| *value > 0));
        let mut sorted = drawn.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert!(sorted.len() > 990, "{} distinct of 1000", sorted.len());
    }

    #[test]
    fn a_bound_that_is_not_positive_draws_nothing() {
        let mut random = SplittableRandom::new(SEED);
        assert_eq!(random.next_int_bounded(0), None);
        assert_eq!(random.next_int_bounded(-3), None);
        // No draw was consumed.
        let mut untouched = SplittableRandom::new(SEED);
        assert_eq!(random.next_int(), untouched.next_int());
    }
}
