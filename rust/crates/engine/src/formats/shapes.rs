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

use euhedral_inference_cuda::{CudaError, DevicePtr, Submission, ops};

/// The most arguments one launch takes: the native launcher's limit (`EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS`).
pub const MAX_ARGUMENTS: usize = 32;

/// The arguments of one launch: each a 64-bit word and the byte size (4 or 8) the kernel declares for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelArguments {
    words: [u64; MAX_ARGUMENTS],
    sizes: [u8; MAX_ARGUMENTS],
    count: usize,
}

impl Default for KernelArguments {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelArguments {
    pub const fn new() -> Self {
        Self {
            words: [0; MAX_ARGUMENTS],
            sizes: [0; MAX_ARGUMENTS],
            count: 0,
        }
    }

    /// A device address.
    pub fn pointer(self, address: DevicePtr) -> Self {
        self.add(address.get(), 8)
    }

    /// A 32-bit value.
    pub fn int32(self, value: u32) -> Self {
        self.add(u64::from(value), 4)
    }

    fn add(mut self, word: u64, size: u8) -> Self {
        // An argument past the limit is counted and not stored; `Launch::queue` refuses the launch.
        if self.count < MAX_ARGUMENTS {
            self.words[self.count] = word;
            self.sizes[self.count] = size;
        }
        self.count += 1;
        self
    }

    /// How many arguments were added.
    pub fn count(&self) -> usize {
        self.count
    }

    fn stored(&self) -> usize {
        self.count.min(MAX_ARGUMENTS)
    }
}

/// The kernel tables a launch can go to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// The shared linear table (`euhedral_cuda_linear_launch`).
    Linear,
    /// The Flash-Next table (`euhedral_cuda_qwen4_launch`), which holds the row-exact BF16 GEMMs.
    Qwen4,
}

/// One kernel launch.
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
    /// Queues the launch on the submission's lane. The library checks the argument count and widths against the
    /// compiled kernel.
    pub fn queue(&self, submission: &Submission<'_>) -> Result<(), CudaError> {
        if self.args.count > MAX_ARGUMENTS {
            return Err(CudaError::Argument(format!(
                "{} takes at most {MAX_ARGUMENTS} arguments, not {}",
                self.symbol, self.args.count
            )));
        }
        let count = self.args.stored();
        let words = &self.args.words[..count];
        let sizes = &self.args.sizes[..count];
        match self.table {
            Table::Linear => ops::linear_launch(
                submission,
                self.kernel,
                self.grid,
                self.block,
                self.shared_bytes,
                words,
                sizes,
            ),
            Table::Qwen4 => ops::qwen4_launch(
                submission,
                self.kernel,
                self.grid,
                self.block,
                self.shared_bytes,
                words,
                sizes,
            ),
        }
    }

    /// The launch as a line of the recorder's format, with its arguments.
    pub fn line(&self) -> String {
        let mut line = self.geometry();
        line.push_str(" args=");
        for index in 0..self.args.stored() {
            if index > 0 {
                line.push(',');
            }
            let word = self.args.words[index];
            // Writing to a String cannot fail.
            let _ = if self.args.sizes[index] == 8 {
                write!(line, "p{word:x}")
            } else {
                write!(line, "i{word}")
            };
        }
        line
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
