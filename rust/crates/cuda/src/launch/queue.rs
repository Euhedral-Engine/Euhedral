//! Enqueueing a planned launch on a RustEE lane.
//!
//! [`enqueue`] takes the catalog's [`Launch`] (kernel, geometry, shared bytes and arguments as sized words) and the
//! [`KernelTable`] that [`load`](super::load::load) made, converts the words into RustEE's argument buffer and launches
//! on whatever [`LaneEnqueue`] it is given: the handle a lane's `submit` lends, a recording's, a shared lane's. Each
//! thread keeps one argument buffer and rewrites it in place, so after a thread's first launch nothing allocates.

use std::cell::RefCell;
use std::iter::repeat_n;

use euhedral_gpu::cuda::CudaKernel;
use euhedral_gpu::{
    Args, Dim3, GpuError, Interleave, L2Promotion, LaneEnqueue, Launch as Geometry, MAX_RANK, MapDataType, OobFill,
    Swizzle, TiledMap,
};
use thiserror::Error;

use crate::launch::catalog::{
    Launch, MAX_ARGUMENTS, MAX_TENSOR_MAPS, TENSOR_MAP_BYTES, TensorMapPlan, TensorMapSwizzle,
};
use crate::launch::load::{KernelTable, Slot, Unavailable};

thread_local! {
    /// The calling thread's argument buffer, rewritten for each launch.
    static ARGUMENTS: RefCell<Args> = RefCell::new(Args::new());
}

/// Why a planned launch was not enqueued.
#[derive(Debug, Error)]
pub enum EnqueueError {
    /// The kernel cannot be launched on this device.
    #[error("{name} is unavailable: {reason}")]
    Unavailable { name: &'static str, reason: Unavailable },
    #[error("{name} is launched with {count} arguments, the most is {MAX_ARGUMENTS}")]
    TooManyArguments { name: &'static str, count: usize },
    #[error("{name} is launched with {count} tensor maps, the most is {MAX_TENSOR_MAPS}")]
    TooManyTensorMaps { name: &'static str, count: usize },
    #[error("{name} is launched with an empty grid or block")]
    EmptyGeometry { name: &'static str },
    /// A word of a size other than 4 or 8 bytes.
    #[error("{name} argument {index} is {size} bytes wide, a word is 4 or 8")]
    WordSize { name: &'static str, index: usize, size: u8 },
    /// The driver reports no parameter sizes for the kernel, so the plan's arguments cannot be checked against it.
    #[error("{name}: the driver reports no parameter sizes to check the arguments against")]
    NoParameterSizes { name: &'static str },
    /// The plan's argument sizes (a tensor map is 128 bytes) differ from the kernel's parameters.
    #[error("{name} takes parameters of sizes {driver:?}, the plan passes {planned:?}")]
    ParameterSizes {
        name: &'static str,
        planned: Vec<u8>,
        driver: Vec<u32>,
    },
    /// RustEE refused the launch (arguments that do not fit the kernel's parameters, a poisoned device, a driver error).
    #[error("{name}: {source}")]
    Gpu {
        name: &'static str,
        #[source]
        source: GpuError,
    },
}

/// Enqueues `launch` on `queue`.
///
/// # Safety
/// The arguments are the kernel's, and every device address among them is valid for the kernel's accesses until the work
/// retires (the contract of [`LaneEnqueue::launch`]).
pub unsafe fn enqueue<Q>(queue: &mut Q, table: &KernelTable, launch: &Launch) -> Result<(), EnqueueError>
where
    Q: LaneEnqueue<Kernel = CudaKernel>,
{
    let name = launch.kernel.name();
    let resolved = match table.get(launch.kernel) {
        Slot::Ready(resolved) => resolved,
        Slot::Unavailable(reason) => {
            return Err(EnqueueError::Unavailable {
                name,
                reason: reason.clone(),
            });
        }
    };
    if launch.args.count() > MAX_ARGUMENTS {
        return Err(EnqueueError::TooManyArguments {
            name,
            count: launch.args.count(),
        });
    }
    if launch.grid.contains(&0) || launch.block.contains(&0) {
        return Err(EnqueueError::EmptyGeometry { name });
    }
    if launch.args.map_count() > MAX_TENSOR_MAPS {
        return Err(EnqueueError::TooManyTensorMaps {
            name,
            count: launch.args.map_count(),
        });
    }
    check_parameter_sizes(
        name,
        launch.args.map_count(),
        launch.args.sizes(),
        resolved.parameter_sizes(),
    )?;
    let geometry = Geometry {
        grid: dim(launch.grid),
        block: dim(launch.block),
        shared_bytes: launch.shared_bytes,
        programmatic: launch.programmatic,
    };
    ARGUMENTS.with_borrow_mut(|arguments| {
        arguments.clear();
        // The maps are the kernel's leading parameters; the driver encodes each for the launch, as the native wrapper does.
        for map in launch.args.maps() {
            let encoded = resolved
                .tensor_map(&tiled_map(map))
                .map_err(|source| EnqueueError::Gpu { name, source })?;
            arguments.push(encoded);
        }
        for (index, (&word, &size)) in launch.args.words().iter().zip(launch.args.sizes()).enumerate() {
            match size {
                4 => arguments.push(word as u32),
                8 => arguments.push(word),
                size => return Err(EnqueueError::WordSize { name, index, size }),
            };
        }
        // SAFETY: the caller's contract; the arguments were checked against the kernel's parameter sizes above.
        unsafe { queue.launch(resolved.kernel(), geometry, arguments) }
            .map_err(|source| EnqueueError::Gpu { name, source })
    })
}

/// The description the native wrapper passes to `cuTensorMapEncodeTiled` for a planned map (`byte_map`,
/// `nvfp4_linear.c`): bytes, unit element strides, no interleave, 128-byte L2 promotion, no out-of-bounds fill.
pub(crate) fn tiled_map(plan: &TensorMapPlan) -> TiledMap {
    let mut dims = [1; MAX_RANK];
    let mut strides = [0; MAX_RANK - 1];
    let mut box_dims = [1; MAX_RANK];
    for (target, source) in dims.iter_mut().zip(plan.dims()) {
        *target = *source;
    }
    for (target, source) in strides.iter_mut().zip(plan.strides()) {
        *target = *source;
    }
    for (target, source) in box_dims.iter_mut().zip(plan.box_dims()) {
        *target = *source;
    }
    TiledMap {
        address: plan.address,
        data_type: MapDataType::U8,
        rank: plan.rank,
        dims,
        strides,
        box_dims,
        element_strides: [1; MAX_RANK],
        interleave: Interleave::None,
        swizzle: match plan.swizzle {
            TensorMapSwizzle::None => Swizzle::None,
            TensorMapSwizzle::B64 => Swizzle::B64,
        },
        l2_promotion: L2Promotion::B128,
        oob_fill: OobFill::Zero,
    }
}

fn dim(extent: [u32; 3]) -> Dim3 {
    Dim3::new(extent[0], extent[1], extent[2])
}

/// Refuses a plan whose argument sizes (`maps` tensor maps of 128 bytes, then the words) are not the sizes of the
/// parameters the driver reports for the kernel, and a kernel for which the driver reports none.
fn check_parameter_sizes(
    name: &'static str,
    maps: usize,
    planned: &[u8],
    driver: Option<&[u32]>,
) -> Result<(), EnqueueError> {
    let driver = driver.ok_or(EnqueueError::NoParameterSizes { name })?;
    let sizes = || repeat_n(TENSOR_MAP_BYTES, maps).chain(planned.iter().copied());
    let same = sizes().count() == driver.len() && sizes().zip(driver).all(|(plan, &size)| u32::from(plan) == size);
    if same {
        Ok(())
    } else {
        Err(EnqueueError::ParameterSizes {
            name,
            planned: sizes().collect(),
            driver: driver.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use euhedral_gpu::TensorMap;

    use super::*;
    use crate::DevicePtr;
    use crate::launch::nvfp4::{Nvfp4Layout, Nvfp4Operands, nvfp4_byte_size, plan_native};

    #[test]
    fn matching_sizes_pass() {
        assert!(check_parameter_sizes("k", 0, &[8, 4, 4], Some(&[8, 4, 4])).is_ok());
        assert!(check_parameter_sizes("k", 0, &[], Some(&[])).is_ok());
    }

    #[test]
    fn differing_sizes_are_refused() {
        for (planned, driver) in [
            (&[8u8, 4][..], &[8u32, 8][..]),
            (&[8], &[8, 4]),
            (&[8, 4], &[8]),
            (&[4], &[128]),
        ] {
            assert!(matches!(
                check_parameter_sizes("k", 0, planned, Some(driver)),
                Err(EnqueueError::ParameterSizes { .. })
            ));
        }
    }

    #[test]
    fn a_driver_that_reports_no_sizes_is_refused() {
        assert!(matches!(
            check_parameter_sizes("k", 0, &[8], None),
            Err(EnqueueError::NoParameterSizes { .. })
        ));
    }

    /// A map as `byte_map` in `nvfp4_linear.c` describes it: bytes, unit element strides, no interleave, 128-byte L2
    /// promotion, no fill; `dims`, `strides` and `box_dims` as the C arrays hold them.
    fn c_map(address: u64, dims: &[u64], strides: &[u64], box_dims: &[u32], swizzle: Swizzle) -> TensorMap {
        let mut description = TiledMap {
            address,
            data_type: MapDataType::U8,
            rank: dims.len() as u32,
            dims: [1; MAX_RANK],
            strides: [0; MAX_RANK - 1],
            box_dims: [1; MAX_RANK],
            element_strides: [1; MAX_RANK],
            interleave: Interleave::None,
            swizzle,
            l2_promotion: L2Promotion::B128,
            oob_fill: OobFill::Zero,
        };
        description.dims[..dims.len()].copy_from_slice(dims);
        description.strides[..strides.len()].copy_from_slice(strides);
        description.box_dims[..box_dims.len()].copy_from_slice(box_dims);
        TensorMap::fake(&description).unwrap()
    }

    fn planned_maps(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout) -> Vec<TensorMap> {
        let (scratch, weights) = (0x4000, 0x2000);
        // SAFETY: the plan only does arithmetic on the address and never dereferences it.
        let at = |address: u64| unsafe { DevicePtr::from_raw(address) };
        let operands = Nvfp4Operands {
            input: at(0x1000),
            weights: at(weights),
            output: at(0x3000),
            rows,
            in_features,
            out_features,
            weights_byte_size: nvfp4_byte_size(out_features, in_features, layout),
        };
        let plan = plan_native(&operands, at(scratch), u64::MAX, false).unwrap();
        let tile = plan.launches().last().unwrap();
        assert_eq!(tile.args.map_count(), 3);
        tile.args
            .maps()
            .map(|map| TensorMap::fake(&tiled_map(map)).unwrap())
            .collect()
    }

    #[test]
    fn the_planned_tile_maps_are_the_bytes_the_native_wrapper_encodes() {
        // The three maps of `launch_native` (`nvfp4_linear.c`), worked by hand: the activation code planes
        // (K / 2 bytes, rows, 2 terms; strides K / 2 and K / 2 * rows; box 64 x 128 x 1; 64-byte swizzle), the weight
        // codes (K / 2, outputs; stride K / 2; box 64 x 128; 64-byte swizzle) and the weight scales
        // (K / 16 or K / 32 bytes, outputs; box 16 x 128; no swizzle) at the 256-aligned end of the codes.
        // 65 rows, 512 by 256, plain: K / 2 = 256, scales 32, the codes end at align256(256 * 256) = 65536.
        assert_eq!(
            planned_maps(65, 512, 256, Nvfp4Layout::Plain),
            [
                c_map(0x4000, &[256, 65, 2], &[256, 16640], &[64, 128, 1], Swizzle::B64),
                c_map(0x2000, &[256, 256], &[256], &[64, 128], Swizzle::B64),
                c_map(0x2000 + 65536, &[32, 256], &[32], &[16, 128], Swizzle::None),
            ]
        );
        // SD4 scale rows are K / 32 = 16 bytes.
        assert_eq!(
            planned_maps(129, 512, 256, Nvfp4Layout::Sd4),
            [
                c_map(0x4000, &[256, 129, 2], &[256, 33024], &[64, 128, 1], Swizzle::B64),
                c_map(0x2000, &[256, 256], &[256], &[64, 128], Swizzle::B64),
                c_map(0x2000 + 65536, &[16, 256], &[16], &[16, 128], Swizzle::None),
            ]
        );
        // 300 rows, 1024 by 512, plain: K / 2 = 512, scales 64, the codes end at align256(512 * 512) = 262144.
        assert_eq!(
            planned_maps(300, 1024, 512, Nvfp4Layout::Plain),
            [
                c_map(0x4000, &[512, 300, 2], &[512, 153_600], &[64, 128, 1], Swizzle::B64),
                c_map(0x2000, &[512, 512], &[512], &[64, 128], Swizzle::B64),
                c_map(0x2000 + 262_144, &[64, 512], &[64], &[16, 128], Swizzle::None),
            ]
        );
        // A different address or shape is a different map.
        assert_ne!(
            planned_maps(300, 1024, 512, Nvfp4Layout::Plain)[0],
            planned_maps(301, 1024, 512, Nvfp4Layout::Plain)[0]
        );
    }

    #[test]
    fn the_tensor_maps_are_counted_in_the_parameter_sizes() {
        assert!(check_parameter_sizes("k", 3, &[8, 4], Some(&[128, 128, 128, 8, 4])).is_ok());
        for (maps, driver) in [
            (2, &[128u32, 128, 128, 8, 4][..]),
            (3, &[128, 128, 8, 4]),
            (3, &[128, 128, 128, 4, 4]),
        ] {
            assert!(matches!(
                check_parameter_sizes("k", maps, &[8, 4], Some(driver)),
                Err(EnqueueError::ParameterSizes { .. })
            ));
        }
    }
}
