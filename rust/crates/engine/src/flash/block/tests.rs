use super::*;
use crate::flash::Weight;
use crate::flash::buffers::{ScratchSizes, StreamShape};
use crate::flash::hyper::HyperShape;
use crate::flash::sequence::{Op, lines};

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

fn weight(address: u64) -> Weight {
    Weight {
        address: at(address),
        bytes: 0,
    }
}

fn site(inject: bool) -> HyperWeights {
    HyperWeights {
        norm: weight(0x100000),
        down: weight(0x110000),
        up: weight(0x120000),
        inject: inject.then(|| weight(0x130000)),
    }
}

/// A body that queues one residual add from its input to its output, so the tests read where the block sits.
#[derive(Debug)]
struct Marked;

/// The layer scratch the marked body asks for, per row.
const MARKED_SCRATCH_PER_ROW: u64 = 16;

impl BlockBody for Marked {
    fn scratch_bytes(&self, rows: u32) -> u64 {
        u64::from(rows) * MARKED_SCRATCH_PER_ROW
    }

    fn expansion_bytes(&self, _rows: u32) -> u64 {
        0
    }

    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError> {
        let mut list = OpList::new();
        list.push(Op::ResidualAdd {
            residual: call.input,
            delta: call.input,
            output: call.output,
            rows: call.rows,
            width: 64,
        });
        Ok(list)
    }
}

fn regions() -> (HyperShape, FlashRegions) {
    let hyper = HyperShape::new(4, 64, 8, 1e-6).unwrap();
    let shape = StreamShape::new(64, 4).unwrap();
    let sizes = ScratchSizes {
        hyper: hyper.scratch_bytes(8),
        layer: Marked.scratch_bytes(8),
        expansion: 0,
    };
    let regions = FlashRegions::over(at(0x4000000), 8, shape, sizes).unwrap();
    (hyper, regions)
}

#[test]
fn the_block_sits_between_the_two_mixes_and_the_injection_follows_it() {
    let (hyper, regions) = regions();
    let stage = BlockStage::new(hyper, site(true), site(true), Arc::new(Marked), regions).unwrap();
    let ops = stage.ops(3, false).unwrap();
    let lines = lines(&ops);
    let names: Vec<_> = lines.iter().map(|line| line.split(' ').next().unwrap()).collect();
    // Mix with its injection projection (6), the block, the injection, the second mix with its projection (6).
    assert_eq!(names.len(), 14);
    assert_eq!(names[6], "residual_add");
    assert_eq!(names[7], "euhedral_q4_hc_inject_bf16");
    assert_eq!(names[8], "euhedral_q4_grouped_rms_norm_bf16");
    let mixed = regions.mixed.get();
    let block = regions.block_output.get();
    let state = regions.state.get();
    let raw = hyper.scratch(regions.hyper_scratch, 3).raw_injection.get();
    assert!(
        lines[6].contains(&format!("p{mixed:x},p{mixed:x},p{block:x}")),
        "{}",
        lines[6]
    );
    // The mix before the block writes `mixed`; the injection folds the block's output into the state in place.
    assert!(lines[4].contains(&format!("p{mixed:x}")), "{}", lines[4]);
    // The injection reads the state, then the block's output, and writes the state: a swap of the two is a different sum.
    assert!(
        lines[7].contains(&format!("args=p{state:x},p{block:x},p{raw:x},p{state:x},i3")),
        "{}",
        lines[7]
    );
}

#[test]
fn a_layer_whose_site_lacks_the_injection_projection_is_refused() {
    let (hyper, regions) = regions();
    assert!(BlockStage::new(hyper, site(false), site(true), Arc::new(Marked), regions).is_err());
    assert!(BlockStage::new(hyper, site(true), site(false), Arc::new(Marked), regions).is_err());
}

#[test]
fn the_final_mix_reads_the_last_row_and_has_no_injection() {
    let (hyper, regions) = regions();
    let stage = FinalMix::new(hyper, site(false), regions).unwrap();
    let ops = stage.ops(5, false).unwrap();
    let lines = lines(&ops);
    assert_eq!(lines.len(), 5);
    let last = regions.state.get() + 4 * 256 * 2;
    assert!(lines[0].contains(&format!("p{last:x}")), "{}", lines[0]);
    assert!(
        lines[4].contains(&format!("p{:x}", regions.final_mixed.get())),
        "{}",
        lines[4]
    );
    assert!(FinalMix::new(hyper, site(true), regions).is_err());
}

#[test]
fn the_end_injection_folds_the_block_output_into_the_state() {
    let (hyper, regions) = regions();
    let op = EndInject::new(hyper, regions).unwrap().op(2).unwrap();
    let raw = hyper.scratch(regions.hyper_scratch, 2).raw_injection.get();
    let (state, block) = (regions.state.get(), regions.block_output.get());
    assert!(
        op.line()
            .contains(&format!("args=p{state:x},p{block:x},p{raw:x},p{state:x},i2")),
        "{}",
        op.line()
    );
}

#[test]
fn a_final_mix_of_no_rows_is_refused() {
    let (hyper, regions) = regions();
    let stage = FinalMix::new(hyper, site(false), regions).unwrap();
    assert!(stage.ops(0, false).is_err());
}

/// The regions of `sizes` for the four-stream shape, or of `streams` streams.
fn regions_of(streams: u32, sizes: ScratchSizes) -> FlashRegions {
    let shape = StreamShape::new(64, streams).unwrap();
    FlashRegions::over(at(0x4000000), 8, shape, sizes).unwrap()
}

#[test]
fn stages_over_regions_of_another_shape_or_with_too_little_scratch_are_refused() {
    let (hyper, _) = regions();
    let full = ScratchSizes {
        hyper: hyper.scratch_bytes(8),
        layer: Marked.scratch_bytes(8),
        expansion: 0,
    };
    let stages = |regions: FlashRegions| {
        [
            BlockStage::new(hyper, site(true), site(true), Arc::new(Marked), regions).is_ok(),
            EndInject::new(hyper, regions).is_ok(),
            FinalMix::new(hyper, site(false), regions).is_ok(),
        ]
    };
    assert_eq!(stages(regions_of(4, full)), [true; 3]);
    // Two streams where the mixer has four.
    assert_eq!(stages(regions_of(2, full)), [false; 3]);
    // One byte short of the mix scratch: the final mix reads one row only, so it still fits.
    let short = ScratchSizes {
        hyper: hyper.scratch_bytes(8) - 1,
        ..full
    };
    assert_eq!(stages(regions_of(4, short)), [false, false, true]);
    // The block body's scratch is the block stage's alone.
    let no_layer = ScratchSizes { layer: 0, ..full };
    assert_eq!(stages(regions_of(4, no_layer)), [false, true, true]);
}

/// A body that queues a residual add whose row count is the chunk's start, and whose width is the tail slot, so the
/// tests read what reached the body.
#[derive(Debug)]
struct ChunkReader;

impl BlockBody for ChunkReader {
    fn scratch_bytes(&self, _rows: u32) -> u64 {
        0
    }

    fn expansion_bytes(&self, _rows: u32) -> u64 {
        0
    }

    fn ops(&self, call: &BlockCall) -> Result<OpList<'static>, FlashError> {
        let mut list = OpList::new();
        list.push(Op::ResidualAdd {
            residual: call.chunk.key_table,
            delta: call.chunk.value_table,
            output: call.output,
            rows: call.chunk.start,
            width: call.chunk.tail_slot,
        });
        Ok(list)
    }
}

#[test]
fn the_chunk_a_caller_names_reaches_the_body_and_the_default_is_a_first_chunk_with_no_tables() {
    let (hyper, regions) = regions();
    let stage = BlockStage::new(hyper, site(true), site(true), Arc::new(ChunkReader), regions).unwrap();
    let chunk = ChunkAt {
        start: 40,
        key_table: at(0xa000),
        value_table: at(0xb000),
        tail_slot: 1,
    };
    let reached = |ops: &OpList<'static>| lines(ops)[6].clone();
    let named = Op::ResidualAdd {
        residual: at(0xa000),
        delta: at(0xb000),
        output: regions_output(&stage),
        rows: 40,
        width: 1,
    };
    assert_eq!(reached(&stage.ops_at(2, false, chunk).unwrap()), named.line());
    let first = Op::ResidualAdd {
        residual: DevicePtr::NULL,
        delta: DevicePtr::NULL,
        output: regions_output(&stage),
        rows: 0,
        width: 0,
    };
    assert_eq!(reached(&stage.ops(2, false).unwrap()), first.line());
}

fn regions_output(stage: &BlockStage) -> DevicePtr {
    stage.regions.block_output
}
