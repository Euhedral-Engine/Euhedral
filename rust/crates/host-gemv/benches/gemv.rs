//! `gemv-probe`: absolute single-thread GB/s of each available instruction-set variant at M = 1..=8 activation rows.
//!
//! Parallelism belongs to the engine, so one call here runs on one thread. The weights are many random copies visited
//! round-robin, with a total size well past any cache, so every call reads cold weights from RAM (a hot, cached weight
//! copy would show decode-shape speeds that a real layer never gets). Throughput counts the bytes of the code and scale
//! planes a call reads.
//!
//! Options: `--rows N` (output rows of one tensor, default 4096), `--k K` (default 5120, a multiple of 32),
//! `--rotate-mib MIB` (total weight bytes across the copies, default 1024), `--passes P` (timed rotations, default 2).

#![allow(
    clippy::print_stdout,
    clippy::expect_used,
    reason = "a binary: its table on stdout is the result, and a failed setup aborts the run"
)]

use std::env;
use std::hint::black_box;
use std::time::Instant;

use euhedral_inference_host_gemv::contract::{f32_to_bf16, padded_k};
use euhedral_inference_host_gemv::{Isa, Nvfp4View, QuantizedRows, available, detected, gemv_nvfp4, quantize_rows};

struct Options {
    rows: usize,
    k: usize,
    rotate_mib: usize,
    passes: usize,
}

fn parse_options() -> Options {
    let mut options = Options {
        rows: 4096,
        k: 5120,
        rotate_mib: 1024,
        passes: 2,
    };
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().unwrap_or_else(|| panic!("{flag} needs a value"));
        let value: usize = value
            .parse()
            .unwrap_or_else(|_| panic!("{flag} needs a number, got {value}"));
        match flag.as_str() {
            "--rows" => options.rows = value,
            "--k" => options.k = value,
            "--rotate-mib" => options.rotate_mib = value,
            "--passes" => options.passes = value,
            other => panic!("unknown option {other}"),
        }
    }
    assert!(
        options.k > 0 && options.k.is_multiple_of(32),
        "--k must be a positive multiple of 32"
    );
    options
}

struct Splitmix(u64);

impl Splitmix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fill(&mut self, bytes: &mut [u8], valid_scales: bool) {
        for word in bytes.chunks_mut(8) {
            let v = self.next().to_le_bytes();
            word.copy_from_slice(&v[..word.len()]);
        }
        if valid_scales {
            // E4M3 scale bytes: no sign, never the NaN code.
            bytes.iter_mut().for_each(|b| *b = (*b & 0x7f).min(0x7e));
        }
    }
}

fn main() {
    let options = parse_options();
    let kp = padded_k(options.k);
    let code_bytes = options.rows * kp / 2;
    let scale_bytes = options.rows * kp / 16;
    let copy_bytes = code_bytes + scale_bytes;
    let copies = (options.rotate_mib * 1024 * 1024).div_ceil(copy_bytes).max(2);
    println!(
        "weights: {} rows x k {}, {} copies of {:.1} MiB ({:.0} MiB rotated); single thread",
        options.rows,
        options.k,
        copies,
        copy_bytes as f64 / (1024.0 * 1024.0),
        (copies * copy_bytes) as f64 / (1024.0 * 1024.0)
    );
    let mut rng = Splitmix(0x5eed);
    let mut codes = vec![0u8; copies * code_bytes];
    let mut scales = vec![0u8; copies * scale_bytes];
    rng.fill(&mut codes, false);
    rng.fill(&mut scales, true);
    let views: Vec<Nvfp4View<'_>> = (0..copies)
        .map(|c| {
            Nvfp4View::new(
                &codes[c * code_bytes..(c + 1) * code_bytes],
                &scales[c * scale_bytes..(c + 1) * scale_bytes],
                0.01,
                options.rows,
                options.k,
            )
            .expect("valid weights")
        })
        .collect();

    let x_bf16: Vec<u16> = (0..8 * options.k)
        .map(|i| f32_to_bf16(((rng.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * (1.0 + (i % 7) as f32)))
        .collect();

    // Quantising the activations is a single-thread step before the GEMV; its throughput is reported on its own.
    let mut quantized = QuantizedRows::new();
    let started = Instant::now();
    let repeats = 200;
    for _ in 0..repeats {
        quantize_rows(black_box(&x_bf16[..options.k]), 1, options.k, &mut quantized);
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "quantize (1 row, bf16 input): {:.2} GB/s",
        (repeats * options.k * 2) as f64 / seconds / 1e9
    );

    println!("detected: {}", detected().name());
    println!(
        "{:<14}{}",
        "variant",
        (1..=8).map(|m| format!("{:>9}", format!("M={m}"))).collect::<String>()
    );
    // The scalar twin is a correctness reference, not a kernel to time.
    for isa in available().into_iter().filter(|&isa| isa != Isa::Scalar) {
        let mut line = format!("{:<14}", isa.name());
        for m in 1..=8 {
            line.push_str(&format!(
                "{:>9.2}",
                measure(isa, &views, &x_bf16, options.k, m, options.passes)
            ));
        }
        println!("{line}  GB/s");
    }
}

fn measure(isa: Isa, views: &[Nvfp4View<'_>], x_bf16: &[u16], k: usize, m: usize, passes: usize) -> f64 {
    let mut x = QuantizedRows::new();
    quantize_rows(&x_bf16[..m * k], m, k, &mut x);
    let n = views[0].n();
    let mut y = vec![0.0f32; m * n];
    // One warm rotation, then `passes` timed rotations over all copies.
    for view in views {
        gemv_nvfp4(isa, view, &x, 0..n, &mut y);
    }
    let started = Instant::now();
    for _ in 0..passes {
        for view in views {
            gemv_nvfp4(isa, view, &x, 0..n, &mut y);
            black_box(&y);
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let bytes = views.iter().map(Nvfp4View::plane_bytes).sum::<usize>() * passes;
    bytes as f64 / seconds / 1e9
}
