//! Temporary: removed together with the native launch path, once the kernels run through RustEE.
//!
//! The table launches (the shared linear table's trellis kernels, the Flash-Next table's row-exact BF16 GEMMs) and the
//! Q3 and P2E2 embeddings through the C wrappers and through their plans on a RustEE lane, compared byte for byte:
//! `cargo test -p euhedral-inference-cuda --test twin_tables -- --ignored --test-threads=1 --nocapture`.
//!
//! The table launches are the ones the Java engine recorded for real layers (the engine's fixture): per distinct
//! sequence of kernels, the smallest case, replayed on device buffers filled with deterministic data, one buffer per
//! recorded base address. The two row-exact GEMMs the fixture does not record are added by hand.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the tallies are the test's output")]

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_inference_cuda::launch::catalog::{Launch, linear_table, qwen4_table};
use euhedral_inference_cuda::launch::embedding::{EmbedShape, plan_embed_q3, plan_embed_q3_p2e2};
use euhedral_inference_cuda::launch::load::KernelSources;
use euhedral_inference_cuda::launch::table::{Geometry, Table, plan_table_launch};
use euhedral_inference_cuda::{CudaLibrary, DevicePtr, ops};

use common::twin::{Case, Random, Tally, Twin, watchdog};

const FIXTURE: &str = include_str!("../../engine/tests/fixtures/linear_launches.txt");

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// An argument of a table launch: an address inside one of the case's buffers, or a 32-bit value.
#[derive(Clone, Copy)]
enum Arg {
    Pointer { buffer: usize, offset: u64 },
    Value(u32),
}

#[derive(Clone)]
struct TableLaunch {
    table: Table,
    name: String,
    grid: [u32; 3],
    block: [u32; 3],
    shared_bytes: u32,
    args: Vec<Arg>,
}

impl TableLaunch {
    fn index(&self) -> i32 {
        let entries = match self.table {
            Table::Linear => linear_table(),
            Table::Qwen4 => qwen4_table(),
        };
        entries
            .iter()
            .position(|spec| spec.name == self.name)
            .expect("a kernel of its table") as i32
    }

    fn words(&self, p: &[DevicePtr]) -> (Vec<u64>, Vec<u8>) {
        self.args
            .iter()
            .map(|arg| match *arg {
                Arg::Pointer { buffer, offset } => (p[buffer].offset(offset).get(), 8u8),
                Arg::Value(value) => (u64::from(value), 4u8),
            })
            .unzip()
    }

    fn geometry(&self) -> Geometry {
        Geometry {
            grid: self.grid,
            block: self.block,
            shared_bytes: self.shared_bytes,
        }
    }
}

struct Recorded {
    header: String,
    n: u64,
    k: u64,
    rows: u64,
    scratch: u64,
    /// The segment ends of the trellis object (the case's fifth buffer holds them as 32-bit integers).
    ends: Vec<u32>,
    launches: Vec<TableLaunch>,
}

fn triple(text: &str) -> [u32; 3] {
    let mut parts = text.split(',').map(|part| part.parse().unwrap());
    [parts.next().unwrap(), parts.next().unwrap(), parts.next().unwrap()]
}

/// The fixture's cases. A pointer `p<hex>` is a buffer (the digit above bit 32 names it, 1 to 7) and an offset.
fn recorded() -> Vec<Recorded> {
    let mut cases: Vec<Recorded> = Vec::new();
    for line in FIXTURE.lines() {
        if let Some(header) = line.strip_prefix("case ") {
            let number = |key: &str| -> u64 {
                let rest = &header[header.find(key).unwrap() + key.len()..];
                rest.split(' ').next().unwrap().parse().unwrap()
            };
            cases.push(Recorded {
                header: header.to_string(),
                n: number(" n="),
                k: number(" k="),
                rows: number(" rows="),
                scratch: 0,
                ends: header[header.find(" ends=[").unwrap() + 7..]
                    .split(']')
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|end| end.parse().unwrap())
                    .collect(),
                launches: Vec::new(),
            });
        } else if let Some(scratch) = line.strip_prefix("scratch=") {
            cases.last_mut().unwrap().scratch = scratch.split(',').next().unwrap().parse().unwrap();
        } else {
            let mut fields = line.split(' ');
            let name = fields.next().unwrap().to_string();
            let mut launch = TableLaunch {
                table: if name.starts_with("euhedral_q4_") {
                    Table::Qwen4
                } else {
                    Table::Linear
                },
                name,
                grid: [0; 3],
                block: [0; 3],
                shared_bytes: 0,
                args: Vec::new(),
            };
            for field in fields {
                let (key, value) = field.split_once('=').unwrap();
                match key {
                    "grid" => launch.grid = triple(value),
                    "block" => launch.block = triple(value),
                    "smem" => launch.shared_bytes = value.parse().unwrap(),
                    _ => {
                        launch.args = value
                            .split(',')
                            .filter(|argument| !argument.is_empty())
                            .map(|argument| match argument.split_at(1) {
                                ("p", hex) => {
                                    let address = u64::from_str_radix(hex, 16).unwrap();
                                    Arg::Pointer {
                                        buffer: (address >> 32) as usize - 1,
                                        offset: address & 0xffff_ffff,
                                    }
                                }
                                (_, decimal) => Arg::Value(decimal.parse().unwrap()),
                            })
                            .collect();
                    }
                }
            }
            cases.last_mut().unwrap().launches.push(launch);
        }
    }
    cases
}

/// The smallest recorded case of each distinct sequence of kernels.
fn smallest_per_sequence(cases: Vec<Recorded>) -> Vec<Recorded> {
    let mut best: BTreeMap<Vec<String>, Recorded> = BTreeMap::new();
    for case in cases {
        let key: Vec<String> = case.launches.iter().map(|launch| launch.name.clone()).collect();
        let size = case.n * case.k * case.rows.max(8) + case.scratch;
        match best.get(&key) {
            Some(held) if held.n * held.k * held.rows.max(8) + held.scratch <= size => {}
            _ => {
                best.insert(key, case);
            }
        }
    }
    best.into_values().collect()
}

/// A row-exact BF16 GEMM of the Flash-Next table by hand (`trellis.rs`'s launch: input, weights, output, rows, K, N).
fn row_exact(name: &str, rows: u32, k: u32, n: u32, split: bool) -> Recorded {
    let pointer = |buffer| Arg::Pointer { buffer, offset: 0 };
    Recorded {
        header: format!("{name} rows={rows} k={k} n={n}"),
        n: u64::from(n),
        k: u64::from(k),
        rows: u64::from(rows),
        scratch: 0,
        ends: Vec::new(),
        launches: vec![TableLaunch {
            table: Table::Qwen4,
            name: name.to_string(),
            grid: [if split { n / 8 } else { n / 32 }, rows.div_ceil(16), 1],
            block: [128, 1, 1],
            shared_bytes: 0,
            args: vec![
                pointer(0),
                pointer(1),
                pointer(2),
                Arg::Value(rows),
                Arg::Value(k),
                Arg::Value(n),
            ],
        }],
    }
}

/// Buffers for a replayed case: one per address base (seven), sized to hold what the case describes.
fn table_buffers(random: &mut Random, case: &Recorded) -> Vec<Vec<u8>> {
    // The trellis codes take about 2.5 bytes a weight; the others are activations, outputs and small tables.
    let small = (case.rows * case.n.max(case.k) * 4).max(4 << 20) as usize;
    let mut sizes = vec![small; 6];
    sizes[1] = (case.n * case.k * 3) as usize;
    sizes.push((case.scratch as usize).max(small));
    let mut buffers: Vec<Vec<u8>> = sizes.into_iter().map(|size| random.bf16(size / 2)).collect();
    for (slot, end) in buffers[4].as_chunks_mut::<4>().0.iter_mut().zip(&case.ends) {
        *slot = end.to_le_bytes();
    }
    buffers
}

fn round_up(value: u64, to: u64) -> u64 {
    value.div_ceil(to) * to
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn table_and_embedding_plans_match_the_native_wrappers() {
    let Some(library) = library() else { return };
    if KernelSources::beside(&library)
        .expect("the product directory")
        .compiler()
        .is_none()
    {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let done = watchdog(360);
    let mut twin = Twin::open(&library).expect("both sides open on one device");
    let mut random = Random(0x7ab1e);
    let mut tally = Tally::default();

    let mut cases = smallest_per_sequence(recorded());
    cases.push(row_exact("euhedral_q4_linear_tc_bf16", 8, 1024, 4096, false));
    cases.push(row_exact("euhedral_q4_linear_tc_split_bf16", 8, 1024, 1024, true));
    for case in &cases {
        let buffers = if case.scratch == 0 {
            // The hand-made GEMMs: input, weights, output.
            vec![
                random.bf16((case.rows * case.k) as usize),
                random.bf16((case.n * case.k) as usize),
                vec![0; (case.rows * case.n * 2) as usize],
            ]
        } else {
            table_buffers(&mut random, case)
        };
        let outputs: Vec<usize> = (0..buffers.len()).collect();
        let (native_launches, plan_launches) = (case.launches.clone(), case.launches.clone());
        let name = case.header.clone();
        let case_run = Case {
            name: &name,
            buffers,
            prepare: None,
            native: Box::new(move |submission, p| {
                for launch in &native_launches {
                    let (words, sizes) = launch.words(p);
                    let (grid, block, shared) = (launch.grid, launch.block, launch.shared_bytes);
                    match launch.table {
                        Table::Linear => {
                            ops::linear_launch(submission, launch.index(), grid, block, shared, &words, &sizes)?
                        }
                        Table::Qwen4 => {
                            ops::qwen4_launch(submission, launch.index(), grid, block, shared, &words, &sizes)?
                        }
                    }
                }
                Ok(())
            }),
            plan: Box::new(move |p| {
                let mut planned: Vec<Launch> = Vec::new();
                for launch in &plan_launches {
                    let (words, sizes) = launch.words(p);
                    plan_table_launch(
                        launch.table,
                        launch.index(),
                        launch.geometry(),
                        &words,
                        &sizes,
                        true,
                        &mut |l| planned.push(l),
                    )
                    .unwrap();
                }
                planned
            }),
            outputs,
            row_exact: false,
            tables: Vec::new(),
            scratch: Vec::new(),
        };
        println!("running {name}");
        let verdict = twin.run(&case_run).expect("the case runs");
        let sequence: Vec<&str> = case.launches.iter().map(|launch| launch.name.as_str()).collect();
        tally.record("table", &format!("{name} {sequence:?}"), verdict);
    }

    // Q3 embeddings: valid byte sizes by the tensor's geometry; P2E2 at its smallest and largest valid sizes.
    let q3_bytes = |vocabulary: u64, hidden: u64| {
        let groups = vocabulary * (hidden.div_ceil(128) * 128 / 64);
        round_up(groups * 24, 256) + groups * 2
    };
    let p2e2_bytes = |rows: u64, features: u64| {
        let codes = rows * features;
        let base = round_up(codes / 4, 256);
        let scale = round_up(base + 4 * rows, 256);
        let payload = round_up(scale + codes / 32, 256);
        let pad = 4 * 80;
        (payload + pad, payload + 4 * codes.div_ceil(16) + pad)
    };
    let mut embeddings: Vec<(&str, u32, u32, u32, u64)> = vec![
        ("q3", 5, 1000, 192, q3_bytes(1000, 192)),
        ("q3", 1, 3, 64, q3_bytes(3, 64)),
        ("q3", 33, 257, 640, q3_bytes(257, 640)),
    ];
    let (least, most) = p2e2_bytes(8, 1024);
    embeddings.push(("p2e2", 6, 8, 1024, least));
    embeddings.push(("p2e2", 9, 8, 1024, most));
    for (kind, tokens, vocabulary, hidden, bytes) in embeddings {
        let ids: Vec<u8> = (0..tokens)
            .flat_map(|_| ((random.word() % u64::from(vocabulary)) as i32).to_le_bytes())
            .collect();
        let mut table = random.bf16(bytes as usize / 2 + 1);
        table.truncate(bytes as usize);
        if kind == "p2e2" {
            // Two-bit codes of 0 to 2: a code of 3 escapes into the payload's extra words, which a random table
            // does not lay out.
            let codes = (u64::from(vocabulary) * u64::from(hidden) / 4) as usize;
            // Each row's first escape is the payload's first: the base words are zero.
            let base = round_up(codes as u64, 256) as usize;
            table[base..base + 4 * vocabulary as usize].fill(0);
            for byte in &mut table[..codes] {
                let mut value = 0u8;
                for field in 0..4 {
                    value |= ((random.word() % 3) as u8) << (2 * field);
                }
                *byte = value;
            }
        }
        let shape = format!("{kind} tokens {tokens} vocabulary {vocabulary} hidden {hidden} bytes {bytes}");
        let embed = move |p: &[DevicePtr]| EmbedShape {
            token_ids: p[0],
            embedding: p[1],
            hidden: p[2],
            token_count: tokens,
            vocabulary_size: vocabulary,
            hidden_size: hidden,
            embedding_bytes: bytes,
        };
        let case = Case {
            name: &shape,
            buffers: vec![ids, table, vec![0; tokens as usize * hidden as usize * 2]],
            prepare: None,
            native: Box::new(move |submission, p| {
                if kind == "q3" {
                    ops::embed_q3(submission, p[0], p[1], p[2], tokens, vocabulary, hidden, bytes)
                } else {
                    ops::embed_q3_p2e2(submission, p[0], p[1], p[2], tokens, vocabulary, hidden, bytes)
                }
            }),
            plan: Box::new(move |p| {
                let mut planned = Vec::new();
                let shape = embed(p);
                if kind == "q3" {
                    plan_embed_q3(&shape, &mut |l| planned.push(l)).unwrap();
                } else {
                    plan_embed_q3_p2e2(&shape, &mut |l| planned.push(l)).unwrap();
                }
                planned
            }),
            outputs: vec![2],
            row_exact: false,
            tables: Vec::new(),
            scratch: Vec::new(),
        };
        println!("running {shape}");
        let verdict = twin.run(&case).expect("the case runs");
        tally.record(&format!("embed {kind}"), &shape, verdict);
    }
    done.store(true, Ordering::Relaxed);
    tally.finish();
}
