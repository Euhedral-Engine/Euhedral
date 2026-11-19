//! The front of measured candidates over (KL, prefill microseconds per token, verification microseconds).

use std::cmp::Ordering;

/// What the front compares a measured candidate by.
pub trait Candidate {
    /// The KL divergence against the reference.
    fn kl(&self) -> f64;
    /// Prefill microseconds per token.
    fn prefill(&self) -> f64;
    /// Verification microseconds.
    fn decode(&self) -> f64;
    /// Whether the candidate fits the device.
    fn fits(&self) -> bool;
}

/// The positions in `measured` of the candidates that fit the device and that no other fitting candidate beats in all
/// three measures and in one strictly, by increasing KL. Of candidates equal in all three, the first measured stays
/// and the others are dropped; of equal KL the earlier position comes first.
pub fn front<C: Candidate>(measured: &[C]) -> Vec<usize> {
    let mut front = Vec::new();
    for (at, candidate) in measured.iter().enumerate() {
        if !candidate.fits() {
            continue;
        }
        let dominated = measured.iter().enumerate().any(|(other_at, other)| {
            other_at != at
                && other.fits()
                && (dominates(other, candidate) || (equal(other, candidate) && other_at < at))
        });
        if !dominated {
            front.push(at);
        }
    }
    front.sort_by(|a, b| compare(measured[*a].kl(), measured[*b].kl()));
    front
}

/// A total order with negative zero below zero and every NaN, whatever its sign bit, above every number and equal to
/// every other NaN.
fn compare(a: f64, b: f64) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

fn equal<C: Candidate>(a: &C, b: &C) -> bool {
    a.kl() == b.kl() && a.prefill() == b.prefill() && a.decode() == b.decode()
}

fn dominates<C: Candidate>(a: &C, b: &C) -> bool {
    let no_worse = a.kl() <= b.kl() && a.prefill() <= b.prefill() && a.decode() <= b.decode();
    no_worse && !equal(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Point {
        tag: &'static str,
        kl: f64,
        prefill: f64,
        decode: f64,
        fits: bool,
    }

    impl Candidate for Point {
        fn kl(&self) -> f64 {
            self.kl
        }
        fn prefill(&self) -> f64 {
            self.prefill
        }
        fn decode(&self) -> f64 {
            self.decode
        }
        fn fits(&self) -> bool {
            self.fits
        }
    }

    fn point(tag: &'static str, kl: f64, prefill: f64, decode: f64, fits: bool) -> Point {
        Point {
            tag,
            kl,
            prefill,
            decode,
            fits,
        }
    }

    fn tags(points: &[Point]) -> Vec<&'static str> {
        front(points).into_iter().map(|at| points[at].tag).collect()
    }

    #[test]
    fn keeps_the_non_dominated_feasible_points_by_kl() {
        let data = [
            point("a", 0.10, 10.0, 10.0, true),
            point("b", 0.05, 20.0, 20.0, true),
            point("c", 0.12, 12.0, 12.0, true), // dominated by a
            point("d", 0.02, 5.0, 5.0, false),  // would dominate all but does not fit
            point("e", 0.05, 20.0, 25.0, true), // dominated by b
            point("f", 0.20, 4.0, 30.0, true),  // fastest prefill
        ];
        assert_eq!(tags(&data), ["b", "a", "f"]);
    }

    #[test]
    fn an_empty_set_has_an_empty_front() {
        assert!(front::<Point>(&[]).is_empty());
    }

    #[test]
    fn of_equal_points_only_the_first_measured_stays() {
        let data = [
            point("first", 0.1, 1.0, 1.0, true),
            point("second", 0.1, 1.0, 1.0, true),
            point("third", 0.1, 1.0, 1.0, true),
        ];
        assert_eq!(tags(&data), ["first"]);
        // An equal point that does not fit does not hide the one that does.
        let data = [point("none", 0.1, 1.0, 1.0, false), point("fits", 0.1, 1.0, 1.0, true)];
        assert_eq!(tags(&data), ["fits"]);
    }

    #[test]
    fn points_of_equal_kl_keep_their_measured_order() {
        let data = [
            point("slow", 0.1, 9.0, 1.0, true),
            point("fast", 0.1, 1.0, 9.0, true),
            point("best", 0.05, 9.0, 9.0, true),
        ];
        assert_eq!(tags(&data), ["best", "slow", "fast"]);
    }

    #[test]
    fn a_nan_kl_sorts_after_every_number_whatever_its_sign_bit() {
        let negative = f64::from_bits(0xfff8_0000_0000_0000);
        let positive = f64::from_bits(0x7ff8_0000_0000_0000);
        assert!(negative.is_nan() && positive.is_nan());
        let data = [
            point("negative", negative, 1.0, 1.0, true),
            point("worse", 0.5, 7.0, 7.0, true),
            point("positive", positive, 2.0, 2.0, true),
            point("better", 0.1, 8.0, 8.0, true),
        ];
        assert_eq!(tags(&data), ["better", "worse", "negative", "positive"]);
    }
}
