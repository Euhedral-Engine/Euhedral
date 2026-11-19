//! The generators against the draws of the `random-vectors` golden: 10 seeds, 64 draws of each kind.

#![allow(
    clippy::disallowed_methods,
    clippy::print_stdout,
    clippy::unwrap_used,
    reason = "a test reads its golden files and stops on a bad one"
)]

use std::fs;
use std::path::{Path, PathBuf};

use euhedral_inference_quantization::java_random::{Random, SplittableRandom};
use test_support::require_property;

const DRAWS: usize = 64;

fn directory() -> Option<PathBuf> {
    let root = require_property("euhedral.quantization.goldens")?;
    let path = PathBuf::from(root).join("random-vectors");
    if !path.is_dir() {
        println!("skipped: no golden at {}", path.display());
        return None;
    }
    Some(path)
}

fn seeds(directory: &Path) -> Vec<i64> {
    let text = fs::read_to_string(directory.join("seeds.json")).unwrap();
    text.trim_matches(|c: char| c == '[' || c == ']' || c.is_whitespace())
        .split(',')
        .map(|seed| seed.trim().parse().unwrap())
        .collect()
}

fn words<const N: usize>(directory: &Path, name: &str) -> Vec<[u8; N]> {
    let bytes = fs::read(directory.join(name)).unwrap();
    bytes.as_chunks::<N>().0.to_vec()
}

fn ints(directory: &Path, name: &str) -> Vec<i32> {
    words::<4>(directory, name)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect()
}

fn longs(directory: &Path, name: &str) -> Vec<i64> {
    words::<8>(directory, name)
        .into_iter()
        .map(i64::from_le_bytes)
        .collect()
}

fn doubles(directory: &Path, name: &str) -> Vec<u64> {
    words::<8>(directory, name)
        .into_iter()
        .map(u64::from_le_bytes)
        .collect()
}

fn floats(directory: &Path, name: &str) -> Vec<u32> {
    words::<4>(directory, name)
        .into_iter()
        .map(u32::from_le_bytes)
        .collect()
}

#[test]
fn random_matches_the_golden() {
    let Some(dir) = directory() else { return };
    let seeds = seeds(&dir);
    let rows = |make: &dyn Fn(&mut Random, usize) -> u64| -> Vec<u64> {
        let mut out = Vec::new();
        for &seed in &seeds {
            let mut random = Random::new(seed);
            for index in 0..DRAWS {
                out.push(make(&mut random, index));
            }
        }
        out
    };
    let expected: Vec<u64> = ints(&dir, "random.nextInt.i32")
        .into_iter()
        .map(|v| v as u32 as u64)
        .collect();
    assert_eq!(rows(&|r, _| r.next_int() as u32 as u64), expected, "nextInt");
    let expected: Vec<u64> = ints(&dir, "random.nextInt-bounded.i32")
        .into_iter()
        .map(|v| v as u64)
        .collect();
    let bounded = {
        let mut out = Vec::new();
        for &seed in &seeds {
            let mut random = Random::new(seed);
            for index in 0..DRAWS {
                out.push(u64::from(random.next_int_bounded(1 + 37 * index as i32).unwrap() as u32));
            }
        }
        out
    };
    assert_eq!(bounded, expected, "nextInt(bound)");
    assert_eq!(
        rows(&|r, _| r.next_long() as u64),
        longs(&dir, "random.nextLong.i64")
            .into_iter()
            .map(|v| v as u64)
            .collect::<Vec<_>>(),
        "nextLong"
    );
    assert_eq!(
        rows(&|r, _| u64::from(r.next_boolean())),
        ints(&dir, "random.nextBoolean.i32")
            .into_iter()
            .map(|v| v as u64)
            .collect::<Vec<_>>(),
        "nextBoolean"
    );
    assert_eq!(
        rows(&|r, _| r.next_double().to_bits()),
        doubles(&dir, "random.nextDouble.f64")
    );
    assert_eq!(
        rows(&|r, _| r.next_gaussian().to_bits()),
        doubles(&dir, "random.nextGaussian.f64")
    );
    assert_eq!(
        rows(&|r, _| u64::from(r.next_float().to_bits())),
        floats(&dir, "random.nextFloat.f32")
            .into_iter()
            .map(u64::from)
            .collect::<Vec<_>>()
    );
}

#[test]
fn shuffles_match_the_golden() {
    let Some(dir) = directory() else { return };
    let seeds = seeds(&dir);
    let expected = ints(&dir, "random.shuffle.i32");
    let mut drawn = Vec::new();
    for &seed in &seeds {
        let mut random = Random::new(seed);
        let mut items: Vec<i32> = (0..100).collect();
        for i in (1..100).rev() {
            let j = random.next_int_bounded(i + 1).unwrap() as usize;
            items.swap(i as usize, j);
        }
        drawn.extend(items);
    }
    assert_eq!(drawn, expected);
}

#[test]
fn splittable_matches_the_golden() {
    let Some(dir) = directory() else { return };
    let seeds = seeds(&dir);
    let fresh = |seed: i64| SplittableRandom::new(seed as u64);
    let run = |draw: &mut dyn FnMut(&mut SplittableRandom, usize) -> u64| -> Vec<u64> {
        let mut out = Vec::new();
        for &seed in &seeds {
            let mut random = fresh(seed);
            for index in 0..DRAWS {
                out.push(draw(&mut random, index));
            }
        }
        out
    };
    assert_eq!(
        run(&mut |r, _| r.next_int() as u32 as u64),
        ints(&dir, "splittable.nextInt.i32")
            .into_iter()
            .map(|v| v as u32 as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        run(&mut |r, index| r.next_int_bounded(1 + 37 * index as i32).unwrap() as u32 as u64),
        ints(&dir, "splittable.nextInt-bounded.i32")
            .into_iter()
            .map(|v| v as u32 as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        run(&mut |r, _| r.next_long() as u64),
        longs(&dir, "splittable.nextLong.i64")
            .into_iter()
            .map(|v| v as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        run(&mut |r, _| u64::from(r.next_boolean())),
        ints(&dir, "splittable.nextBoolean.i32")
            .into_iter()
            .map(|v| v as u64)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        run(&mut |r, _| r.next_double().to_bits()),
        doubles(&dir, "splittable.nextDouble.f64")
    );
}
