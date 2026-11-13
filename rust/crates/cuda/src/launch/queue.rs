//! Enqueueing a planned launch on a RustEE lane.
//!
//! [`enqueue`] takes the catalog's [`Launch`] (kernel, geometry, shared bytes and arguments as sized words) and the
//! [`KernelTable`] that [`load`](super::load::load) made, converts the words into RustEE's argument buffer and launches
//! on whatever [`LaneEnqueue`] it is given: the handle a lane's `submit` lends, a recording's, a shared lane's. Each
//! thread keeps one argument buffer and rewrites it in place, so after a thread's first launch nothing allocates.

use std::cell::RefCell;

use euhedral_gpu::cuda::CudaKernel;
use euhedral_gpu::{Args, Dim3, GpuError, LaneEnqueue, Launch as Geometry};
use thiserror::Error;

use crate::launch::catalog::{Launch, MAX_ARGUMENTS};
use crate::launch::load::{KernelTable, Slot, Unavailable, pin};

thread_local! {
    /// The calling thread's argument buffer, rewritten for each launch.
    static ARGUMENTS: RefCell<Args> = RefCell::new(Args::new());
}

/// Why a planned launch was not enqueued.
#[derive(Debug, Error)]
pub enum EnqueueError {
    /// The kernel cannot be launched at the pinned RustEE revision.
    #[error("{name} is unavailable: {reason}")]
    Unavailable { name: &'static str, reason: Unavailable },
    /// The launch needs more dynamic shared memory than the kernel's limit.
    #[error("{name} needs {needed} bytes of dynamic shared memory, its limit is {limit}")]
    SharedBytes {
        name: &'static str,
        needed: u32,
        limit: u32,
    },
    #[error("{name} is launched with {count} arguments, the most is {MAX_ARGUMENTS}")]
    TooManyArguments { name: &'static str, count: usize },
    #[error("{name} is launched with an empty grid or block")]
    EmptyGeometry { name: &'static str },
    /// A word of a size other than 4 or 8 bytes.
    #[error("{name} argument {index} is {size} bytes wide, a word is 4 or 8")]
    WordSize { name: &'static str, index: usize, size: u8 },
    /// The driver reports no parameter sizes for the kernel, so the plan's arguments cannot be checked against it.
    #[error("{name}: the driver reports no parameter sizes to check the arguments against")]
    NoParameterSizes { name: &'static str },
    /// The plan's argument sizes differ from the kernel's parameters.
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
    if launch.shared_bytes > resolved.max_shared_bytes() {
        return Err(EnqueueError::SharedBytes {
            name,
            needed: launch.shared_bytes,
            limit: resolved.max_shared_bytes(),
        });
    }
    if launch.args.count() > MAX_ARGUMENTS {
        return Err(EnqueueError::TooManyArguments {
            name,
            count: launch.args.count(),
        });
    }
    if launch.grid.contains(&0) || launch.block.contains(&0) {
        return Err(EnqueueError::EmptyGeometry { name });
    }
    check_parameter_sizes(name, launch.args.sizes(), resolved.parameter_sizes())?;
    // Seam: with programmatic dependent launch in RustEE, a launch whose `programmatic` flag is set (and whose caller
    // allows it) is made programmatic here. Until then the flag is not honoured: the kernel's `griddepcontrol.wait`
    // returns at once without a programmatic predecessor.
    const _: () = assert!(
        !pin::PROGRAMMATIC_LAUNCH,
        "fill the programmatic-launch seam, then remove this"
    );
    if pin::PROGRAMMATIC_LAUNCH && launch.programmatic {
        // Replace with the programmatic launch of the re-pinned RustEE.
    }
    let geometry = Geometry {
        grid: dim(launch.grid),
        block: dim(launch.block),
        shared_bytes: launch.shared_bytes,
        programmatic: false,
    };
    ARGUMENTS.with_borrow_mut(|arguments| {
        arguments.clear();
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

fn dim(extent: [u32; 3]) -> Dim3 {
    Dim3::new(extent[0], extent[1], extent[2])
}

/// Refuses a plan whose argument sizes are not the sizes of the parameters the driver reports for the kernel, and a
/// kernel for which the driver reports none.
fn check_parameter_sizes(name: &'static str, planned: &[u8], driver: Option<&[u32]>) -> Result<(), EnqueueError> {
    let driver = driver.ok_or(EnqueueError::NoParameterSizes { name })?;
    let same =
        planned.len() == driver.len() && planned.iter().zip(driver).all(|(&plan, &size)| u32::from(plan) == size);
    if same {
        Ok(())
    } else {
        Err(EnqueueError::ParameterSizes {
            name,
            planned: planned.to_vec(),
            driver: driver.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_sizes_pass() {
        assert!(check_parameter_sizes("k", &[8, 4, 4], Some(&[8, 4, 4])).is_ok());
        assert!(check_parameter_sizes("k", &[], Some(&[])).is_ok());
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
                check_parameter_sizes("k", planned, Some(driver)),
                Err(EnqueueError::ParameterSizes { .. })
            ));
        }
    }

    #[test]
    fn a_driver_that_reports_no_sizes_is_refused() {
        assert!(matches!(
            check_parameter_sizes("k", &[8], None),
            Err(EnqueueError::NoParameterSizes { .. })
        ));
    }
}
