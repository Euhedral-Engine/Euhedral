use super::*;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

const A: u64 = 0x1000;
const B: u64 = 0x2000;
const C: u64 = 0x3000;
const D: u64 = 0x4000;
const E: u64 = 0x5000;

fn items(count: u32) -> ExpertItems {
    ExpertItems {
        slots: at(0x10_000),
        items: at(0x10_100),
        item_count: count,
        pairs: at(0x10_200),
    }
}

fn refused(result: Result<Launch, FlashError>) -> String {
    result.unwrap_err().to_string()
}

#[test]
fn routing_and_shared_launches_carry_javas_arguments_in_javas_order() {
    let line = |launch: Result<Launch, FlashError>| launch.unwrap().line();
    assert_eq!(
        line(router(at(A), at(B), at(C), 7, 512, 10)),
        "euhedral_q4_router_bf16 grid=7,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i7,i512,i10"
    );
    assert_eq!(
        line(swiglu(at(A), at(B), at(C), 1025)),
        "euhedral_q4_swiglu_bf16 grid=5,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i1025"
    );
    assert_eq!(
        line(swiglu_padded(at(A), at(B), at(C), 9, 512, 768)),
        "euhedral_q4_swiglu_padded_bf16 grid=3,9,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i512,i768"
    );
    assert_eq!(
        line(moe_finish(at(A), at(B), at(C), at(D), 3, 2560)),
        "euhedral_q4_moe_finish_bf16 grid=10,3,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,i3,i2560"
    );
}

#[test]
fn expert_launches_carry_javas_arguments_in_javas_order() {
    let geometry = ExpertGeometry::flash_next();
    assert_eq!(
        expert_gate_up(&geometry, &items(2), at(A), at(B)).unwrap().line(),
        "euhedral_q4_expert_gate_up_swiglu_bf16 grid=20,2,1 block=128,1,1 smem=0 \
         args=p10000,p10100,p10200,p1000,p2000,i0,i2560,i640"
    );
    assert_eq!(
        expert_down(&geometry, &items(2), at(B), at(C)).unwrap().line(),
        "euhedral_q4_expert_down_bf16 grid=20,2,1 block=128,1,1 smem=0 \
         args=p10000,p10100,p10200,p2000,p3000,i1843456,i640,i2560"
    );
    assert_eq!(
        expert_combine(&geometry, at(C), at(D), at(E), at(A), 4).unwrap().line(),
        "euhedral_q4_expert_combine_bf16 grid=2,4,1 block=256,1,1 smem=0 args=p3000,p4000,p5000,p1000,i2560,i1"
    );
}

#[test]
fn the_expert_scratch_is_act_then_weighted_each_sixteen_byte_aligned() {
    let geometry = ExpertGeometry::flash_next();
    // 3 pairs of 640 BF16 values is 3840 bytes, a multiple of 16; 5 pairs of 2560 is 25600.
    assert_eq!(geometry.weighted_offset(3), 3840);
    assert_eq!(geometry.scratch_bytes(5), 6400 + 25600);
    // One pair of 128 BF16 values is 256 bytes.
    let narrow = ExpertGeometry::new(128, 128, 0, 256).unwrap();
    assert_eq!(narrow.weighted_offset(1), 256);
}

#[test]
fn each_launch_refuses_what_its_kernel_cannot_take() {
    assert!(refused(router(at(A), at(B), at(C), 0, 512, 10)).contains("rows must be positive"));
    assert!(refused(router(at(A), at(B), at(C), 1, 1025, 10)).contains("router shape 1 x 1025 top 10"));
    assert!(refused(router(at(A), at(B), at(C), 1, 512, 17)).contains("top 17"));
    assert!(refused(router(at(A), at(B), at(C), 1, 8, 9)).contains("top 9"));
    assert!(refused(swiglu(at(A), at(B), at(C), 0)).contains("count must be positive"));
    assert!(refused(swiglu_padded(at(A), at(B), at(C), 2, 512, 256)).contains("padded width 256 is below the 512"));
    assert!(refused(moe_finish(at(A), at(B), at(C), at(D), 1, 0)).contains("width must be positive"));
    let geometry = ExpertGeometry::flash_next();
    assert!(refused(expert_gate_up(&geometry, &items(0), at(A), at(B))).contains("item count must be positive"));
    assert!(refused(expert_gate_up(&geometry, &items(1), at(A + 8), at(B))).contains("input 0x1008 is not 16-byte"));
    assert!(refused(expert_down(&geometry, &items(1), at(A), DevicePtr::NULL)).contains("weighted 0x0"));
    assert!(refused(expert_combine(&geometry, at(A), at(B), at(C), at(D), 0)).contains("rows must be positive"));
    assert!(refused(expert_combine(&geometry, at(A), at(B), at(C + 4), at(D), 1)).contains("row pairs 0x3004"));
}

#[test]
fn a_geometry_refuses_extents_and_offsets_the_kernels_cannot_read() {
    let message = |hidden, inter, gate_up, down| {
        ExpertGeometry::new(hidden, inter, gate_up, down)
            .unwrap_err()
            .to_string()
    };
    assert!(message(0, 640, 0, 256).contains("hidden must be positive"));
    assert!(message(2560, 600, 0, 256).contains("multiples of 128, got 2560, 600"));
    assert!(message(2560, 640, 100, 256).contains("gate_up offset 100"));
    assert!(message(2560, 640, 0, 1 << 40).contains("down offset"));
    assert_eq!(
        ExpertGeometry::new(2560, 640, 0, 1_843_456).unwrap(),
        ExpertGeometry::flash_next()
    );
}
