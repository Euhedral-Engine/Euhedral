//! The contracts of the pieces of the startup that need no device: the widened-weight lookup and the owner of a start
//! that never got a lattice.

use super::*;

fn job(layer: u32, offset: u64) -> PadJob {
    let source = Weight {
        // SAFETY: the address is only compared, never dereferenced.
        address: unsafe { DevicePtr::from_raw(0x1000) },
        bytes: 64,
    };
    PadJob {
        layer,
        source,
        target_offset: offset,
        target_bytes: 96,
    }
}

#[test]
fn a_widened_weight_is_found_by_layer_at_its_offset() {
    // SAFETY: the address is only compared, never dereferenced.
    let base = unsafe { DevicePtr::from_raw(0x10_0000) };
    let padded = PaddedDown {
        jobs: vec![job(3, 0), job(7, 256)],
        base: Some(base),
    };
    let weight = padded.weight(7).expect("layer 7 has a widened weight");
    assert_eq!(weight.address.get(), base.get() + 256);
    assert_eq!(weight.bytes, 96);
    assert!(padded.weight(4).is_none(), "layer 4 has none");
}

#[test]
fn no_widened_region_means_no_widened_weight() {
    let padded = PaddedDown {
        jobs: Vec::new(),
        base: None,
    };
    assert!(padded.weight(0).is_none());
}

#[test]
fn an_idle_owner_runs_the_free_and_returns_its_result() {
    assert_eq!(RuntimeOwner::idle().shutdown_with(|| 41 + 1), 42);
}
