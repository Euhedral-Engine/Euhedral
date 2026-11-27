use super::*;
use crate::flash::buffers::StreamShape;
use crate::flash::moe::testing::{self, Source, at, regions, shape, weights};
use crate::flash::sequence::lines;

fn layer(padded: bool) -> MoeLayer {
    testing::layer(Arc::new(Source::default()), padded)
}

fn launch_lines(launches: &[Launch]) -> Vec<String> {
    launches.iter().map(Launch::line).collect()
}

#[test]
fn scratch_is_carved_in_order_from_the_block_scratch_of_the_regions() {
    let layer = layer(true);
    let (ids, weights) = layer.router_outputs();
    // The logits (4 rows x 8 experts x 2 bytes) come first at the start of the block scratch; the ids follow them
    // at the next 256 boundary, the routing weights after the ids' 8 pairs of 4 bytes.
    let first = regions(0).layer_scratch;
    assert_eq!(ids, first.offset(256));
    assert_eq!(weights, first.offset(512));
    assert_eq!(layer.descriptor().get() % 256, 0);
    assert!(layer.descriptor().get() > weights.get());
}

#[test]
fn the_route_is_the_router_linear_then_the_router() {
    let launches = layer(true).route_launches(3, false).unwrap();
    let first = regions(0).layer_scratch.get();
    let (logits, ids, weights) = (first, first + 256, first + 512);
    let mixed = regions(0).mixed.get();
    let lines = launch_lines(&launches);
    assert!(
        lines[0].contains(&format!("p{mixed:x},p100000,p{logits:x}")),
        "{}",
        lines[0]
    );
    assert_eq!(
        lines[1],
        format!(
            "euhedral_q4_router_bf16 grid=3,1,1 block=256,1,1 smem=0 args=p{logits:x},p{ids:x},p{weights:x},i3,i8,i2"
        )
    );
}

#[test]
fn the_shared_expert_reads_the_padded_down_weight_only_for_many_rows_and_when_it_has_one() {
    let names = |layer: &MoeLayer, rows: u32| -> Vec<String> {
        lines(&layer.shared_ops(rows, false).unwrap())
            .iter()
            .map(|line| line.split(' ').next().unwrap().to_owned())
            .collect()
    };
    let padded = layer(true);
    assert!(names(&padded, 9).contains(&"euhedral_q4_swiglu_padded_bf16".to_owned()));
    assert!(!names(&padded, 8).contains(&"euhedral_q4_swiglu_padded_bf16".to_owned()));
    assert!(names(&padded, 8).contains(&"euhedral_q4_swiglu_bf16".to_owned()));
    let plain = layer(false);
    assert!(!names(&plain, 9).contains(&"euhedral_q4_swiglu_padded_bf16".to_owned()));
    assert_eq!(names(&plain, 9).len(), 5);
}

#[test]
fn an_expert_reads_its_items_and_its_outputs_start_after_the_activations_of_the_block() {
    let layer = layer(true);
    // The expert's items start at the 3rd of the block's and number 3; the block has 5 pairs, so the weighted outputs
    // start 2 * 5 * 128 bytes after the activations.
    let launches = layer.expert_launches(2, 3, 5).unwrap();
    let layout = *shape().layout();
    let described = layer.descriptor().get();
    let (slots, items, pairs) = (
        described,
        described + layout.items_offset() as u64 + 32,
        described + layout.pairs_offset() as u64,
    );
    let (mixed, acts) = (regions(0).mixed.get(), 0x1000_4000);
    assert_eq!(layout.items_offset(), 64);
    assert_eq!(layer.descriptor().get(), 0x1000_3e00);
    assert_eq!(
        launch_lines(&launches),
        [
            format!(
                "euhedral_q4_expert_gate_up_swiglu_bf16 grid=4,3,1 block=128,1,1 smem=0 args=p{slots:x},p{items:x},p{pairs:x},p{mixed:x},p{acts:x},i0,i128,i128"
            ),
            format!(
                "euhedral_q4_expert_down_bf16 grid=1,3,1 block=128,1,1 smem=0 args=p{slots:x},p{items:x},p{pairs:x},p{acts:x},p{:x},i4096,i128,i128",
                acts + 1280
            ),
        ]
    );
}

#[test]
fn a_block_that_kept_no_pair_has_a_zero_expert_sum() {
    let layer = layer(true);
    let none = lines(&layer.finish_ops(3, 0).unwrap());
    assert_eq!(none.len(), 2);
    assert!(none[0].starts_with("zero p"), "{}", none[0]);
    assert!(none[0].ends_with("bytes=768"), "{}", none[0]);
    assert!(none[1].starts_with("euhedral_q4_moe_finish_bf16"));
    let some = lines(&layer.finish_ops(3, 4).unwrap());
    assert!(some[0].starts_with("euhedral_q4_expert_combine_bf16"));
    assert_eq!(some[1], none[1]);
}

#[test]
fn a_layer_refuses_regions_that_are_too_small_or_of_another_width() {
    let source: Arc<dyn ExpertSource> = Arc::new(Source::default());
    let device = Nvfp4Device::with_native_route(false, 8);
    let build =
        |regions: FlashRegions| MoeLayer::new(3, 5, shape(), weights(true), regions, device, Arc::clone(&source));
    let mut sizes = MoeLayer::scratch_sizes(&shape(), true, &device);
    sizes.layer -= 1;
    let small = FlashRegions::over(at(testing::REGIONS), 4, StreamShape::new(128, 2).unwrap(), sizes).unwrap();
    let error = build(small).unwrap_err().to_string();
    assert!(error.contains("layer 3") && error.contains("layer scratch"), "{error}");
    let sizes = MoeLayer::scratch_sizes(&shape(), true, &device);
    let narrow = FlashRegions::over(at(testing::REGIONS), 4, StreamShape::new(136, 2).unwrap(), sizes).unwrap();
    assert!(build(narrow).unwrap_err().to_string().contains("136"));
    let few = FlashRegions::over(at(testing::REGIONS), 2, StreamShape::new(128, 2).unwrap(), sizes).unwrap();
    assert!(build(few).unwrap_err().to_string().contains("routes up to 4 rows"));
}

#[test]
fn a_shape_refuses_a_shared_width_that_is_not_a_multiple_of_16() {
    let layout = *shape().layout();
    let geometry = *shape().geometry();
    assert!(MoeShape::new(layout, geometry, 0).is_err());
    assert!(
        MoeShape::new(layout, geometry, 201)
            .unwrap_err()
            .to_string()
            .contains("201")
    );
    assert_eq!(MoeShape::new(layout, geometry, 208).unwrap().padded_width(), 256);
}

#[test]
fn a_shape_the_router_cannot_serve_is_refused_naming_the_values_and_the_limits() {
    let geometry = *shape().geometry();
    let refused = |experts, top_k| {
        let layout = DescriptorLayout::new(experts, top_k, 4).unwrap();
        MoeShape::new(layout, geometry, 192).unwrap_err().to_string()
    };
    for (experts, top_k) in [(ROUTER_MAX_EXPERTS + 1, 2), (8, ROUTER_MAX_TOP_K + 1), (2, 3)] {
        let why = refused(experts, top_k);
        assert!(why.contains(&format!("{experts} experts with {top_k} picks")), "{why}");
        assert!(why.contains(&ROUTER_MAX_EXPERTS.to_string()), "{why}");
    }
    let limit = DescriptorLayout::new(ROUTER_MAX_EXPERTS, ROUTER_MAX_TOP_K, 4).unwrap();
    assert!(MoeShape::new(limit, geometry, 192).is_ok());
}
