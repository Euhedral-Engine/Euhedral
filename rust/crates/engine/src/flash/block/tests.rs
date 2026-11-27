use super::*;
use crate::flash::Weight;
use crate::flash::buffers::StreamShape;
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

impl BlockBody for Marked {
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
    let regions = FlashRegions::over(at(0x4000000), 8, shape, hyper.scratch_bytes(8), 0).unwrap();
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
    assert!(
        lines[6].contains(&format!("p{mixed:x},p{mixed:x},p{block:x}")),
        "{}",
        lines[6]
    );
    // The mix before the block writes `mixed`; the injection folds the block's output into the state in place.
    assert!(lines[4].contains(&format!("p{mixed:x}")), "{}", lines[4]);
    assert!(lines[7].contains(&format!("p{block:x}")), "{}", lines[7]);
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
    let scratch = hyper.scratch(regions.hyper_scratch, 2);
    let op = hyper
        .inject(regions.state, regions.block_output, &scratch, regions.state, 2)
        .unwrap();
    assert!(op.line().starts_with("euhedral_q4_hc_inject_bf16"));
}
