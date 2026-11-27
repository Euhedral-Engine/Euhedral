use super::*;
use crate::flash::sequence::lines;

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

fn weights(inject: bool) -> HyperWeights {
    HyperWeights {
        norm: weight(0x100000),
        down: weight(0x110000),
        up: weight(0x120000),
        inject: inject.then(|| weight(0x130000)),
    }
}

#[test]
fn the_mix_queues_javas_operations_in_javas_order() {
    let shape = HyperShape::new(4, 64, 8, 1.0).unwrap();
    let scratch = shape.scratch(at(0x200000), 3);
    let ops = shape
        .mix(&weights(true), at(0x300000), &scratch, at(0x400000), 3, false)
        .unwrap();
    let symbols: Vec<_> = lines(&ops)
        .iter()
        .map(|line| line.split(' ').next().unwrap().to_owned())
        .collect();
    assert_eq!(
        symbols,
        [
            "euhedral_q4_grouped_rms_norm_bf16",
            // Three rows project 256 values to a narrow rank and a narrow stream count: the K-split kernel.
            "euhedral_q4_linear_split8_bf16",
            "euhedral_q4_scaled_silu_bf16",
            "euhedral_q4_linear_bf16",
            "euhedral_q4_hc_mix_bf16",
            "euhedral_q4_linear_split8_bf16",
        ]
    );
    let lines = lines(&ops);
    // The norm reads the streams and writes `normed`; the silu is scaled by the stream count over rows * lowrank values.
    assert!(lines[0].ends_with(&format!(
        "args=p300000,p100000,p{:x},i3,i4,i64,i1065353216",
        scratch.normed.get()
    )));
    assert!(lines[2].ends_with(&format!(
        "args=p{:x},p{:x},i24,i{}",
        scratch.down.get(),
        scratch.activated.get(),
        4f32.to_bits()
    )));
    // The projection up goes from the rank to the state width; the last projects the normed streams to the injection.
    assert!(lines[3].contains(&format!(
        "p{:x},p120000,p{:x},i3,i8,i256",
        scratch.activated.get(),
        scratch.up.get()
    )));
    assert!(lines[5].contains(&format!(
        "p{:x},p130000,p{:x},i3,i256,i4",
        scratch.normed.get(),
        scratch.raw_injection.get()
    )));
}

#[test]
fn a_site_without_an_injection_projection_queues_no_projection_for_it() {
    let shape = HyperShape::new(4, 64, 8, 1.0).unwrap();
    let scratch = shape.scratch(at(0x200000), 3);
    let ops = shape
        .mix(&weights(false), at(0x300000), &scratch, at(0x400000), 3, false)
        .unwrap();
    assert_eq!(ops.len(), 5, "the final mixer's site ends at the mix");
}

#[test]
fn the_inject_reads_the_raw_weights_the_mix_wrote() {
    let shape = HyperShape::new(4, 64, 8, 1.0).unwrap();
    let scratch = shape.scratch(at(0x200000), 3);
    let op = shape
        .inject(at(0x300000), at(0x500000), &scratch, at(0x300000), 3)
        .unwrap();
    assert_eq!(
        op.line(),
        format!(
            "euhedral_q4_hc_inject_bf16 grid=1,3,1 block=256,1,1 smem=0 args=p300000,p500000,p{:x},p300000,i3,i4,i64",
            scratch.raw_injection.get()
        )
    );
}

#[test]
fn scratch_buffers_are_disjoint_and_sized_for_the_rows() {
    let shape = HyperShape::new(4, 64, 8, 1.0).unwrap();
    let base = 0x200000;
    let scratch = shape.scratch(at(base), 5);
    let spans = [
        (scratch.normed, 5 * 256 * 2),
        (scratch.down, 5 * 8 * 2),
        (scratch.activated, 5 * 8 * 2),
        (scratch.up, 5 * 256 * 2),
        (scratch.raw_injection, 5 * 4 * 2),
    ];
    let mut end = base;
    for (start, bytes) in spans {
        assert!(start.get() >= end);
        end = start.get() + bytes;
    }
    assert!(end <= base + shape.scratch_bytes(5));
}

#[test]
fn a_shape_the_kernels_cannot_serve_is_refused() {
    for (streams, hidden, lowrank) in [(1, 64, 8), (0, 64, 8), (4, 0, 8), (4, 64, 0), (4, 64, 4), (3, 4, 8)] {
        let error = HyperShape::new(streams, hidden, lowrank, 1e-6).unwrap_err().to_string();
        assert!(
            error.contains(&format!("{streams} streams of {hidden} values at rank {lowrank}")),
            "{error}"
        );
    }
    assert!(HyperShape::new(4, 64, 8, f32::NAN).is_err());
}
