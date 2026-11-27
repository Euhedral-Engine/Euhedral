use super::*;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

fn weight() -> Nvfp4Weight {
    Nvfp4Weight {
        weight: Weight {
            address: at(0x100000),
            bytes: 4096,
        },
        sd4: false,
    }
}

/// A linear of `rows` rows over 512 by 1024 values, with `bytes` of expansion scratch.
fn linear(device: &Nvfp4Device, rows: u32, bytes: u64) -> Result<Op<'static>, FlashError> {
    let expansion = Expansion {
        address: at(0x800000),
        bytes,
    };
    device.linear(at(0x200000), weight(), at(0x300000), rows, 512, 1024, expansion)
}

fn is_native(op: &Op<'_>) -> bool {
    matches!(op, Op::Nvfp4Linear { native: Some(_), .. })
}

#[test]
fn nine_rows_or_more_plan_the_native_route_and_fewer_plan_the_decode_route() {
    let device = Nvfp4Device::with_native_route(true, 8);
    let bytes = device.expansion_bytes(64, 512, 1024);
    assert!(bytes > 0);
    for rows in [1, 4, 8] {
        assert!(!is_native(&linear(&device, rows, bytes).unwrap()), "{rows} rows");
    }
    for rows in [9, 16, 64, 200] {
        assert!(
            is_native(&linear(&device, rows, device.expansion_bytes(rows, 512, 1024)).unwrap()),
            "{rows} rows"
        );
    }
}

#[test]
fn a_device_without_the_native_route_never_plans_it() {
    let device = Nvfp4Device::with_native_route(false, 8);
    assert_eq!(device.expansion_bytes(64, 512, 1024), 0);
    assert!(!is_native(&linear(&device, 64, 0).unwrap()));
}

#[test]
fn an_input_the_native_route_has_no_kernel_for_stays_on_the_decode_route() {
    let device = Nvfp4Device::with_native_route(true, 8);
    let expansion = Expansion {
        address: at(0x800000),
        bytes: 1 << 20,
    };
    // 192 inputs are not a multiple of 128.
    let op = device
        .linear(at(0x200000), weight(), at(0x300000), 16, 192, 1024, expansion)
        .unwrap();
    assert!(!is_native(&op));
}

#[test]
fn an_expansion_region_smaller_than_the_route_takes_is_refused_with_the_sizes() {
    let device = Nvfp4Device::with_native_route(true, 8);
    let needed = device.expansion_bytes(32, 512, 1024);
    assert!(linear(&device, 32, needed).is_ok());
    let message = linear(&device, 32, needed - 1).unwrap_err().to_string();
    assert!(message.contains(&needed.to_string()), "{message}");
}

#[test]
fn the_expansion_covers_the_larger_of_the_skinny_and_the_tile_kernel() {
    let device = Nvfp4Device::with_native_route(true, 8);
    let nine = device.expansion_bytes(9, 512, 1024);
    assert!(device.expansion_bytes(200, 512, 1024) >= nine);
    assert_eq!(device.expansion_bytes(8, 512, 1024), 0);
}
