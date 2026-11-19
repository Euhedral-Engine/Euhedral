//! The generators against draws printed by Java: each line of `tests/data/random-vectors.txt` is
//! `GENERATOR SEED METHOD: VALUES`.

#![allow(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "a test reads its fixture files and stops on a bad one"
)]

use std::fs;
use std::path::Path;

use euhedral_inference_quantization::java_random::{Random, SplittableRandom};

fn lines() -> Vec<(String, i64, String, Vec<i64>)> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/random-vectors.txt");
    let text = fs::read_to_string(path).unwrap();
    text.lines()
        .map(|line| {
            let (head, values) = line.split_once(':').unwrap();
            let mut words = head.split(' ');
            let generator = words.next().unwrap().to_string();
            let seed = words.next().unwrap().parse().unwrap();
            let method = words.next().unwrap().to_string();
            let values = values.split_whitespace().map(|v| v.parse().unwrap()).collect();
            (generator, seed, method, values)
        })
        .collect()
}

fn random_draws(seed: i64, method: &str, count: usize) -> Vec<i64> {
    let mut random = Random::new(seed);
    let bound = |name: &str| name.strip_prefix("bounded").map(|b| b.parse::<i32>().unwrap());
    match method {
        "int" => (0..count).map(|_| i64::from(random.next_int())).collect(),
        "long" => (0..count).map(|_| random.next_long()).collect(),
        "boolean" => (0..count).map(|_| i64::from(random.next_boolean())).collect(),
        "double" => (0..count).map(|_| random.next_double().to_bits() as i64).collect(),
        "gaussian" => (0..count).map(|_| random.next_gaussian().to_bits() as i64).collect(),
        "mixed" => (0..count / 3)
            .flat_map(|_| {
                let int = i64::from(random.next_int());
                let gaussian = random.next_gaussian().to_bits() as i64;
                let bounded = i64::from(random.next_int_bounded(10).unwrap());
                [int, gaussian, bounded]
            })
            .collect(),
        other => {
            let bound = bound(other).unwrap();
            (0..count)
                .map(|_| i64::from(random.next_int_bounded(bound).unwrap()))
                .collect()
        }
    }
}

fn splittable_draws(seed: i64, method: &str, count: usize) -> Vec<i64> {
    let mut random = SplittableRandom::new(seed as u64);
    match method {
        "int" => (0..count).map(|_| i64::from(random.next_int())).collect(),
        "long" => (0..count).map(|_| random.next_long()).collect(),
        "boolean" => (0..count).map(|_| i64::from(random.next_boolean())).collect(),
        "split" => {
            let mut drawn = Vec::new();
            for _ in 0..4 {
                let mut child = random.split();
                drawn.extend((0..8).map(|_| i64::from(child.next_int())));
                drawn.push(i64::from(random.next_int()));
                drawn.push(child.split().next_long());
            }
            drawn
        }
        other => {
            let bound: i32 = other.strip_prefix("bounded").unwrap().parse().unwrap();
            (0..count)
                .map(|_| i64::from(random.next_int_bounded(bound).unwrap()))
                .collect()
        }
    }
}

#[test]
fn every_draw_matches_java() {
    let lines = lines();
    assert!(lines.len() >= 100);
    for (generator, seed, method, expected) in lines {
        let drawn = match generator.as_str() {
            "random" => random_draws(seed, &method, expected.len()),
            "splittable" => splittable_draws(seed, &method, expected.len()),
            other => panic!("unknown generator {other}"),
        };
        assert_eq!(drawn, expected, "{generator} seed {seed} {method}");
    }
}
