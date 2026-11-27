use super::*;
use crate::flash::buffers::{ScratchSizes, StreamShape};
use crate::flash::sequence::lines;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

const HIDDEN: u32 = 64;
const VOCABULARY: u32 = 32;

fn regions() -> FlashRegions {
    let shape = StreamShape::new(HIDDEN, 4).unwrap();
    let sizes = ScratchSizes {
        hyper: 4096,
        layer: 4096,
        expansion: 0,
    };
    FlashRegions::over(at(0x4000000), 8, shape, sizes).unwrap()
}

fn weight(bytes: u64) -> Weight {
    Weight {
        address: at(0x100000),
        bytes,
    }
}

fn head() -> HeadStage {
    let bytes = u64::from(VOCABULARY) * u64::from(HIDDEN) * 2;
    HeadStage::new(weight(bytes), at(0x200000), &regions(), HIDDEN, VOCABULARY).unwrap()
}

#[test]
fn the_head_multiplies_the_final_mix_of_one_row_by_the_output_weights() {
    let regions = regions();
    let stage = head();
    let lines = lines(&stage.ops(false, Select::Row).unwrap());
    let expected = ops::linear_bf16(
        regions.final_mixed,
        at(0x100000),
        at(0x200000),
        1,
        HIDDEN,
        VOCABULARY,
        false,
    )
    .unwrap()
    .line();
    assert_eq!(lines, vec![expected]);
}

#[test]
fn a_greedy_head_selects_over_the_vocabulary_after_the_logits_and_names_the_key_to_copy() {
    let stage = head();
    let keys = at(0x300000);
    let lines = lines(&stage.ops(false, Select::Key(keys)).unwrap());
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1], format!("argmax p200000,i{VOCABULARY},p300000"));
    assert_eq!(stage.readback(Select::Key(keys)), (keys, 8));
}

#[test]
fn a_row_head_copies_the_whole_bf16_row() {
    let stage = head();
    assert_eq!(stage.readback(Select::Row), (at(0x200000), VOCABULARY as usize * 2));
}

#[test]
fn exact_numerics_reach_the_logits_launch() {
    let stage = head();
    let plain = lines(&stage.ops(false, Select::Row).unwrap());
    let exact = lines(&stage.ops(true, Select::Row).unwrap());
    assert_ne!(
        plain, exact,
        "an exact head must not use the kernel an inexact one does"
    );
}

#[test]
fn a_head_the_weights_or_the_logits_cannot_honour_is_refused() {
    let regions = regions();
    let right = u64::from(VOCABULARY) * u64::from(HIDDEN) * 2;
    let new = |head: Weight, logits: u64, hidden: u32, vocabulary: u32| {
        HeadStage::new(head, at(logits), &regions, hidden, vocabulary)
    };
    assert!(new(weight(right - 2), 0x200000, HIDDEN, VOCABULARY).is_err());
    assert!(new(weight(right), 0x200008, HIDDEN, VOCABULARY).is_err());
    assert!(new(weight(0), 0x200000, HIDDEN, 0).is_err());
    assert!(new(weight(0), 0x200000, 0, VOCABULARY).is_err());
    // A hidden width the BF16 kernels cannot take, or another than the regions hold, is refused at construction.
    assert!(new(weight(u64::from(VOCABULARY) * 12 * 2), 0x200000, 12, VOCABULARY).is_err());
    assert!(new(weight(u64::from(VOCABULARY) * 128 * 2), 0x200000, 128, VOCABULARY).is_err());
}

#[test]
fn the_host_reads_the_row_to_its_readback_when_the_quantum_has_one_else_the_key_to_the_selections_and_refuses_neither()
{
    let stage = head();
    let keys = at(0x300000);
    let row = VOCABULARY as usize * 2;
    assert_eq!(stage.route(1, Some((row, "row")), None).unwrap(), (Select::Row, "row"));
    assert_eq!(
        stage.route(1, Some((row, "row")), Some((keys, "key"))).unwrap(),
        (Select::Row, "row"),
        "the row readback wins when both are offered"
    );
    assert_eq!(
        stage.route(1, None, Some((keys, "key"))).unwrap(),
        (Select::Key(keys), "key")
    );
    assert!(stage.route::<&str>(1, None, None).is_err());
    let message = stage
        .route(1, Some((row + 2, "row")), Some((keys, "key")))
        .unwrap_err()
        .to_string();
    assert!(
        message.contains(&(row + 2).to_string()) && message.contains(&row.to_string()),
        "{message}"
    );
}

#[test]
fn a_head_refuses_a_quantum_that_scores_any_other_count_of_rows_than_one() {
    let stage = head();
    let row = VOCABULARY as usize * 2;
    for rows in [0, 2, 9] {
        let message = stage.route(rows, Some((row, ())), None).unwrap_err().to_string();
        assert!(message.contains(&format!("scores {rows}")), "{message}");
    }
}

#[test]
fn the_head_serves_a_quantum_of_one_segment_and_refuses_any_other_count_naming_it() {
    assert_eq!(single_segment(&[7]).unwrap(), &7);
    for count in [0usize, 2, 3] {
        let message = single_segment(&vec![0; count]).unwrap_err().to_string();
        assert!(message.contains(&format!("has {count} segments")), "{message}");
    }
}
