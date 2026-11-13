//! Kernel launches as data, and the recorder that lists them.
//!
//! A linear's route is decided on the host as a list of [`Launch`]es: the kernel of a table, its grid, block and shared
//! bytes, and its arguments. The formats build that list and queue it, so what a test records is what the device ran.
//!
//! [`ShapeRecorder`] keeps the list a format queued, one line per launch, in the line format
//! `<symbol> grid=<x>,<y>,<z> block=<x>,<y>,<z> smem=<bytes> args=<arguments>` that the Java side prints from its own
//! launches (`tests/fixtures/linear_launches.jsh`): pointers as `p<hex>`, 32-bit values as `i<decimal>`. A format
//! records only when it was built for recording, which is for tests.

use std::cell::RefCell;
use std::fmt::Write;

use euhedral_inference_cuda::launch::PlanError;
use euhedral_inference_cuda::launch::catalog::{Arguments, Launch as PlannedLaunch};
use euhedral_inference_cuda::launch::table::{Geometry, plan_table_launch};
use euhedral_inference_cuda::{CudaError, DevicePtr, KernelOps};

pub use euhedral_inference_cuda::launch::catalog::MAX_ARGUMENTS;
pub use euhedral_inference_cuda::launch::table::Table;

/// The arguments of one launch: each a 64-bit word and the byte size (4 or 8) the kernel declares for it. The words
/// are the catalog's [`Arguments`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KernelArguments(Arguments);

impl KernelArguments {
    pub const fn new() -> Self {
        Self(Arguments::new())
    }

    /// A device address.
    pub fn pointer(self, address: DevicePtr) -> Self {
        Self(self.0.pointer(address))
    }

    /// A 32-bit value.
    pub fn int32(self, value: u32) -> Self {
        Self(self.0.u32(value))
    }

    /// How many arguments were added.
    pub fn count(&self) -> usize {
        self.0.count()
    }

    /// The arguments as the catalog's.
    pub fn arguments(&self) -> Arguments {
        self.0
    }
}

/// One kernel launch of a table, as the formats build it. [`Launch::planned`] is the same launch as the catalog's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub table: Table,
    /// The kernel's index in its table.
    pub kernel: i32,
    /// The kernel's name.
    pub symbol: &'static str,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic shared bytes.
    pub shared_bytes: u32,
    pub args: KernelArguments,
}

impl Launch {
    /// Queues the launch on `kernels`. The library checks the argument count and widths against the
    /// compiled kernel.
    pub fn queue(&self, kernels: &mut dyn KernelOps) -> Result<(), CudaError> {
        if self.args.count() > MAX_ARGUMENTS {
            return Err(CudaError::Argument(format!(
                "{} takes at most {MAX_ARGUMENTS} arguments, not {}",
                self.symbol,
                self.args.count()
            )));
        }
        let words = self.args.0.words();
        let sizes = self.args.0.sizes();
        match self.table {
            Table::Linear => kernels.linear_launch(self.kernel, self.grid, self.block, self.shared_bytes, words, sizes),
            Table::Qwen4 => kernels.qwen4_launch(self.kernel, self.grid, self.block, self.shared_bytes, words, sizes),
        }
    }

    /// The launch as a line of the recorder's format, with its arguments.
    pub fn line(&self) -> String {
        let mut line = self.geometry();
        line.push_str(" args=");
        let (words, sizes) = (self.args.0.words(), self.args.0.sizes());
        for (index, (&word, &size)) in words.iter().zip(sizes).enumerate() {
            if index > 0 {
                line.push(',');
            }
            // Writing to a String cannot fail.
            let _ = if size == 8 {
                write!(line, "p{word:x}")
            } else {
                write!(line, "i{word}")
            };
        }
        line
    }

    /// The launch as the catalog plans it: the table entry's kernel with the words typed by their sizes. The host-side
    /// refusals of the table launcher apply, for a device that has every kernel.
    pub fn planned(&self) -> Result<PlannedLaunch, PlanError> {
        if self.args.count() > MAX_ARGUMENTS {
            return Err(PlanError::InvalidArgument);
        }
        let mut planned = None;
        let geometry = Geometry {
            grid: self.grid,
            block: self.block,
            shared_bytes: self.shared_bytes,
        };
        plan_table_launch(
            self.table,
            self.kernel,
            geometry,
            self.args.0.words(),
            self.args.0.sizes(),
            // A launch is planned before it is queued and without a device, so the planner assumes the kernel is
            // loaded; the queue refuses a kernel the device lacks.
            true,
            &mut |launch| planned = Some(launch),
        )?;
        planned.ok_or(PlanError::InvalidArgument)
    }

    /// The launch as a line without its arguments (the launches the recorder sees only by route).
    pub fn geometry(&self) -> String {
        let [gx, gy, gz] = self.grid;
        let [bx, by, bz] = self.block;
        format!(
            "{} grid={gx},{gy},{gz} block={bx},{by},{bz} smem={}",
            self.symbol, self.shared_bytes
        )
    }
}

thread_local! {
    /// The lines the calling thread's recording formats queued since the last [`ShapeRecorder::take`].
    static RECORDED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// The launches the formats built for recording queued on this thread, in the order they were queued.
///
/// A submission is made and dropped inside one frame on one thread, so the thread is the unit of a recording: a test
/// that submits on its own thread takes the lines on that thread.
#[derive(Debug)]
pub struct ShapeRecorder;

impl ShapeRecorder {
    /// Appends a line.
    pub fn record(line: String) {
        RECORDED.with(|recorded| recorded.borrow_mut().push(line));
    }

    /// The lines recorded on this thread, which are cleared.
    pub fn take() -> Vec<String> {
        RECORDED.with(|recorded| recorded.take())
    }
}

/// The kernels of the shared linear table (`LinearKernel`, native/src/host/linear_kernels.c). The declaration order is
/// the native table's order, which is its ABI; a GPU test compares the names with the library's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearKernel {
    TrellisPrepass,
    TrellisMma,
    TrellisMmaRows,
    TrellisMmaRows64,
    TrellisEpilogue,
    TrellisDecodeBf16,
    TrellisEpilogueBf16,
    TrellisReference,
    TrellisMmaRowsFp8,
    TrellisMmaRows64Fp8,
    Nvfp4Embedding,
}

impl LinearKernel {
    /// Every kernel, in table order.
    pub const ALL: [LinearKernel; 11] = [
        LinearKernel::TrellisPrepass,
        LinearKernel::TrellisMma,
        LinearKernel::TrellisMmaRows,
        LinearKernel::TrellisMmaRows64,
        LinearKernel::TrellisEpilogue,
        LinearKernel::TrellisDecodeBf16,
        LinearKernel::TrellisEpilogueBf16,
        LinearKernel::TrellisReference,
        LinearKernel::TrellisMmaRowsFp8,
        LinearKernel::TrellisMmaRows64Fp8,
        LinearKernel::Nvfp4Embedding,
    ];

    /// The kernel's position in the table.
    pub fn index(self) -> usize {
        self as usize
    }

    /// The kernel's name.
    pub fn symbol(self) -> &'static str {
        match self {
            LinearKernel::TrellisPrepass => "euhedral_trellis_prepass",
            LinearKernel::TrellisMma => "euhedral_trellis_mma",
            LinearKernel::TrellisMmaRows => "euhedral_trellis_mma_rows",
            LinearKernel::TrellisMmaRows64 => "euhedral_trellis_mma_rows64",
            LinearKernel::TrellisEpilogue => "euhedral_trellis_epilogue",
            LinearKernel::TrellisDecodeBf16 => "euhedral_trellis_decode_bf16",
            LinearKernel::TrellisEpilogueBf16 => "euhedral_trellis_epilogue_bf16",
            LinearKernel::TrellisReference => "euhedral_trellis_reference",
            LinearKernel::TrellisMmaRowsFp8 => "euhedral_trellis_mma_rows_fp8",
            LinearKernel::TrellisMmaRows64Fp8 => "euhedral_trellis_mma_rows64_fp8",
            LinearKernel::Nvfp4Embedding => "euhedral_nvfp4_embedding",
        }
    }

    /// A launch of this kernel.
    pub fn launch(self, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, args: KernelArguments) -> Launch {
        Launch {
            table: Table::Linear,
            kernel: self as i32,
            symbol: self.symbol(),
            grid,
            block,
            shared_bytes,
            args,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pointer(address: u64) -> DevicePtr {
        // SAFETY: the tests never dereference it.
        unsafe { DevicePtr::from_raw(address) }
    }

    #[test]
    fn a_launch_is_the_catalogs_launch_of_the_same_table_entry() {
        let args = KernelArguments::new().pointer(pointer(0xabc0)).int32(7);
        let launch = LinearKernel::TrellisEpilogue.launch([3, 2, 1], [128, 1, 1], 64, args);
        let planned = launch.planned().unwrap();
        assert_eq!(planned.kernel.name(), launch.symbol);
        assert_eq!(
            (planned.grid, planned.block, planned.shared_bytes),
            ([3, 2, 1], [128, 1, 1], 64)
        );
        assert_eq!(planned.args.words(), &[0xabc0, 7]);
        assert_eq!(planned.args.sizes(), &[8, 4]);
        // The recorded line is the Java side's: pointers as p<hex>, 32-bit values as i<decimal>.
        assert_eq!(
            launch.line(),
            "euhedral_trellis_epilogue grid=3,2,1 block=128,1,1 smem=64 args=pabc0,i7"
        );
    }

    #[test]
    fn every_linear_kernel_names_the_catalogs_entry() {
        for kernel in LinearKernel::ALL {
            let planned = kernel
                .launch([1, 1, 1], [1, 1, 1], 0, KernelArguments::new())
                .planned()
                .unwrap();
            assert_eq!(planned.kernel.name(), kernel.symbol());
        }
    }

    #[test]
    fn too_many_arguments_do_not_plan() {
        let mut args = KernelArguments::new();
        for _ in 0..=MAX_ARGUMENTS {
            args = args.int32(1);
        }
        let launch = LinearKernel::TrellisPrepass.launch([1, 1, 1], [1, 1, 1], 0, args);
        assert_eq!(launch.planned(), Err(PlanError::InvalidArgument));
    }
}
