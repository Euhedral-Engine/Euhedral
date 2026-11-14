use euhedral_gpu::{Args, GpuError, Launch as Geometry};

use super::*;

/// What the recording queue was asked, in order.
#[derive(Debug, PartialEq, Eq)]
enum Call {
    Copy {
        to: u64,
        from: u64,
        bytes: usize,
    },
    Pitched {
        to: u64,
        to_pitch: usize,
        from: u64,
        from_pitch: usize,
        row_bytes: usize,
        rows: usize,
    },
    Memset {
        at: u64,
        value: u8,
        bytes: usize,
    },
    ToDevice {
        to: u64,
        from: usize,
        bytes: usize,
    },
    ToHost {
        to: usize,
        from: u64,
        bytes: usize,
    },
}

#[derive(Default)]
struct Recording {
    calls: Vec<Call>,
}

impl LaneEnqueue for Recording {
    type Kernel = CudaKernel;

    unsafe fn launch(&mut self, _: &CudaKernel, _: Geometry, _: &Args) -> Result<(), GpuError> {
        panic!("no kernel is available in these tests, so nothing launches");
    }

    unsafe fn copy_to_device(&mut self, dst: u64, src: *const u8, bytes: usize) -> Result<(), GpuError> {
        self.calls.push(Call::ToDevice {
            to: dst,
            from: src as usize,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_to_host(&mut self, dst: *mut u8, src: u64, bytes: usize) -> Result<(), GpuError> {
        self.calls.push(Call::ToHost {
            to: dst as usize,
            from: src,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_on_device(&mut self, dst: u64, src: u64, bytes: usize) -> Result<(), GpuError> {
        self.calls.push(Call::Copy {
            to: dst,
            from: src,
            bytes,
        });
        Ok(())
    }

    unsafe fn copy_pitched_on_device(
        &mut self,
        dst: u64,
        dst_pitch: usize,
        src: u64,
        src_pitch: usize,
        row_bytes: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        // The lane's own refusal, which the engine side no longer repeats.
        if dst_pitch < row_bytes || src_pitch < row_bytes {
            return Err(GpuError::Invalid("pitch below the row".into()));
        }
        self.calls.push(Call::Pitched {
            to: dst,
            to_pitch: dst_pitch,
            from: src,
            from_pitch: src_pitch,
            row_bytes,
            rows,
        });
        Ok(())
    }

    unsafe fn memset(&mut self, dst: u64, value: u8, bytes: usize) -> Result<(), GpuError> {
        self.calls.push(Call::Memset { at: dst, value, bytes });
        Ok(())
    }
}

fn ptr(address: u64) -> DevicePtr {
    // SAFETY: nothing here dereferences an address.
    unsafe { DevicePtr::from_raw(address) }
}

fn status(result: Result<(), CudaError>) -> i32 {
    match result {
        Err(CudaError::Call { status, .. }) => status,
        other => panic!("expected a refusal with a status, got {other:?}"),
    }
}

#[test]
fn the_row_exact_selection_changes_later_plans_and_not_the_mode_the_quantum_began_in() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    assert!(!ops.row_exact());
    ops.set_row_exact(true);
    assert!(ops.modes().row_exact);
    assert!(!ops.row_exact(), "the quantum's own mode does not follow the selection");
    ops.set_row_exact(false);
    assert!(!ops.modes().row_exact);

    let mut ops = RusteeOps::new(
        &mut queue,
        &table,
        Modes {
            row_exact: true,
            exact_numerics: true,
        },
    );
    assert!(ops.row_exact() && ops.exact_numerics());
    ops.set_row_exact(false);
    assert!(ops.row_exact(), "it began row-exact");
    assert!(!ops.modes().row_exact);
}

#[test]
fn a_planner_refusal_is_the_status_the_native_wrapper_returns() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    // A null operand: invalid argument.
    assert_eq!(
        status(ops.residual_add_bf16(DevicePtr::NULL, ptr(256), ptr(512), 1, 8)),
        INVALID_ARGUMENT
    );
    assert_eq!(status(ops.argmax_bf16(ptr(256), 0, ptr(512))), INVALID_ARGUMENT);
    // The native NVFP4 route declines a width that is not a multiple of 128: the caller takes the other route.
    assert_eq!(
        status(ops.linear_nvfp4_native_bf16(ptr(256), ptr(512), ptr(768), ptr(1024), 16, 64, 64, 1 << 20, 1 << 20)),
        ROUTE_UNAVAILABLE
    );
    // A table kernel that is out of range.
    assert_eq!(
        status(ops.linear_launch(99, [1, 1, 1], [32, 1, 1], 0, &[], &[])),
        INVALID_ARGUMENT
    );
    // The native route also declines under exact numerics.
    let mut exact = RusteeOps::new(
        &mut queue,
        &table,
        Modes {
            row_exact: false,
            exact_numerics: true,
        },
    );
    assert_eq!(
        status(exact.linear_nvfp4_native_bf16(ptr(256), ptr(512), ptr(768), ptr(1024), 16, 128, 128, 1 << 20, 1 << 20)),
        ROUTE_UNAVAILABLE
    );
    assert!(queue.calls.is_empty(), "a refused call queued nothing");
}

#[test]
fn a_kernel_the_device_lacks_is_an_error_naming_it_and_queues_nothing() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    let error = ops
        .residual_add_bf16(ptr(256), ptr(512), ptr(768), 2, 64)
        .expect_err("no kernel is loaded");
    let CudaError::Argument(message) = error else {
        panic!("expected the reason, got {error:?}");
    };
    assert!(message.contains("euhedral_residual_add_bf16"), "{message}");
    assert!(message.contains("unavailable"), "{message}");
    assert!(queue.calls.is_empty());
}

#[test]
fn the_architecture_specific_table_kernels_answer_route_unavailable_without_the_target() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    // Entry 9 of the shared linear table is the block-scaled FP8 kernel, of an architecture-specific module.
    assert_eq!(
        status(ops.linear_launch(9, [1, 1, 1], [32, 1, 1], 0, &[], &[])),
        ROUTE_UNAVAILABLE
    );
    assert!(!ops.linear_kernel_available(9).unwrap());
    assert!(ops.linear_kernel_available(11).is_err(), "the table has 11 entries");
}

#[test]
fn copies_and_the_memset_are_the_lanes_own_operations_in_call_order() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    {
        let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
        ops.copy_device_to_device(ptr(0x1000), ptr(0x2000), 64).unwrap();
        ops.zero_device_memory(ptr(0x3000), 32).unwrap();
        // Pitched: one copy, the destination's pitch first.
        ops.copy_device_to_device_2d(ptr(0x4000), 256, ptr(0x5000), 128, 100, 3)
            .unwrap();
        // Equal pitches are the lane's to fold into one plain copy.
        ops.copy_device_to_device_2d(ptr(0x6000), 100, ptr(0x7000), 100, 100, 3)
            .unwrap();
        // Nothing to copy is not an error, as the native wrappers have it.
        ops.copy_device_to_device(ptr(0x8000), ptr(0x9000), 0).unwrap();
        ops.copy_device_to_device_2d(ptr(0x8000), 8, ptr(0x9000), 8, 8, 0)
            .unwrap();
    }
    assert_eq!(
        queue.calls,
        vec![
            Call::Copy {
                to: 0x1000,
                from: 0x2000,
                bytes: 64
            },
            Call::Memset {
                at: 0x3000,
                value: 0,
                bytes: 32
            },
            Call::Pitched {
                to: 0x4000,
                to_pitch: 256,
                from: 0x5000,
                from_pitch: 128,
                row_bytes: 100,
                rows: 3
            },
            Call::Pitched {
                to: 0x6000,
                to_pitch: 100,
                from: 0x7000,
                from_pitch: 100,
                row_bytes: 100,
                rows: 3
            },
        ]
    );
}

#[test]
fn a_copy_the_native_wrapper_refuses_is_refused() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    assert_eq!(
        status(ops.copy_device_to_device(DevicePtr::NULL, ptr(256), 8)),
        INVALID_ARGUMENT
    );
    assert_eq!(status(ops.zero_device_memory(ptr(256), 0)), INVALID_ARGUMENT);
    assert_eq!(status(ops.zero_device_memory(DevicePtr::NULL, 8)), INVALID_ARGUMENT);
    // A pitch below the row is the lane's refusal, passed on with nothing queued.
    assert!(matches!(
        ops.copy_device_to_device_2d(ptr(256), 4, ptr(512), 8, 8, 2),
        Err(CudaError::Argument(_))
    ));
    assert!(matches!(
        ops.copy_device_to_device_2d(ptr(256), 8, ptr(512), 4, 8, 2),
        Err(CudaError::Argument(_))
    ));
    assert_eq!(
        status(ops.copy_device_to_device_2d(DevicePtr::NULL, 8, ptr(512), 8, 8, 2)),
        INVALID_ARGUMENT
    );
    assert_eq!(
        status(ops.copy_device_to_device_2d(ptr(256), 8, DevicePtr::NULL, 8, 8, 2)),
        INVALID_ARGUMENT
    );
    // Zero bytes per row with rows to go is nothing to copy, not an error.
    ops.copy_device_to_device_2d(ptr(256), 8, ptr(512), 8, 0, 2).unwrap();
    assert!(queue.calls.is_empty());
}

#[test]
fn an_upload_and_a_readback_start_at_their_offset_in_the_host_bytes() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    {
        let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
        let whole = (0x7000 as *mut u8, 64);
        // SAFETY: the recording queue never dereferences the host addresses.
        unsafe {
            ops.upload(ptr(0x1000), whole, 16, 48, "buffer").unwrap();
            ops.download(whole, 8, ptr(0x2000), 16, "buffer").unwrap();
        }
    }
    assert_eq!(
        queue.calls,
        vec![
            Call::ToDevice {
                to: 0x1000,
                from: 0x7010,
                bytes: 48
            },
            Call::ToHost {
                to: 0x7008,
                from: 0x2000,
                bytes: 16
            },
        ]
    );
}

#[test]
fn an_upload_or_a_readback_the_native_wrapper_refuses_queues_nothing() {
    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let mut ops = RusteeOps::new(&mut queue, &table, Modes::default());
    let whole = (0x7000 as *mut u8, 64);
    // SAFETY: the recording queue never dereferences the host addresses, and nothing is queued.
    unsafe {
        assert_eq!(
            status(ops.upload(DevicePtr::NULL, whole, 0, 8, "buffer")),
            INVALID_ARGUMENT
        );
        assert_eq!(status(ops.upload(ptr(0x1000), whole, 0, 0, "buffer")), INVALID_ARGUMENT);
        assert_eq!(
            status(ops.download(whole, 0, DevicePtr::NULL, 8, "buffer")),
            INVALID_ARGUMENT
        );
        assert_eq!(
            status(ops.download(whole, 0, ptr(0x2000), 0, "buffer")),
            INVALID_ARGUMENT
        );
        // Outside the host bytes: one byte past the end, and an offset that overflows.
        assert!(matches!(
            ops.upload(ptr(0x1000), whole, 1, 64, "buffer"),
            Err(CudaError::Argument(_))
        ));
        assert!(matches!(
            ops.download(whole, usize::MAX, ptr(0x2000), 2, "buffer"),
            Err(CudaError::Argument(_))
        ));
    }
    assert!(queue.calls.is_empty());
}

/// A submission whose mode does not overlap launches (a prefill, or a decode past the overlap limit) launches a
/// kernel the catalog marks as an ordinary launch; one that does, launches it programmatically; and a kernel the
/// catalog does not mark never does.
#[test]
fn a_mode_without_pdl_launches_a_registered_kernel_without_overlap() {
    use crate::kernel;
    use crate::launch::catalog::Arguments;

    let table = KernelTable::all_unavailable(false);
    let mut queue = Recording::default();
    let launch = |kernel| Launch::new(kernel, [1, 1, 1], [32, 1, 1], 0, Arguments::new());
    let registered = launch(kernel!("euhedral_residual_add_bf16"));
    let unregistered = launch(kernel!("euhedral_nvfp4n_quantize_rows"));
    assert!(registered.programmatic && !unregistered.programmatic);

    let ops = RusteeOps::new(&mut queue, &table, Modes::default());
    assert!(ops.overlaps(&registered), "overlap is on until a mode turns it off");
    assert!(!ops.overlaps(&unregistered));
    let ops = ops.with_pdl(false);
    assert!(!ops.overlaps(&registered), "pdl false: not programmatic");
    assert!(!ops.overlaps(&unregistered));
    let ops = ops.with_pdl(true);
    assert!(ops.overlaps(&registered));
}
