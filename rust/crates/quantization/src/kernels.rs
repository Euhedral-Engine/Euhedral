//! The quantization kernels as NVRTC modules: the sources, the compile options, the kernel names of each module and
//! the loader that compiles a module once and resolves its kernels.
//!
//! The sources are compiled with `--std=c++17` and the device's own architecture (the backend adds
//! `--gpu-architecture`), with the CUDA header directory on the include path. These are not the engine's options
//! (`--std=c++14`): the quantization modules are compiled the same way wherever they are built, so the codes they
//! produce do not depend on the build.
//!
//! The compiler is RustEE's NVRTC facility, the bundled one the engine compiles with (one NVRTC serves the process, so
//! the loader never picks another), and a module compiles to machine code for the device or not at all: PTX is
//! refused. A compile runs on a thread of its own with the stack the engine's `kernels.compile.stack` sets. Where the
//! cuBLAS and cuSOLVER libraries are is not this module's concern.

mod source;

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaGpu, CudaModule, Nvrtc, Target};
use euhedral_gpu::{Args, CompileOptions, Gpu, GpuError, LaneEnqueue, Launch, Module as GpuModule};
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_cuda::launch::load::{DEFAULT_COMPILE_STACK_BYTES, KernelSources, LoadError, on_compile_stack};
use thiserror::Error;

pub use source::expand_includes;

/// A quantization kernel source and the kernels the library launches from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    /// GPTQ rounding: group choice, the NVFP4 and SD4 searches, the permutations and the Hessian helpers.
    Gptq,
    /// Trellis quantization: the rotations, the Viterbi encode, reconstruction and the scale sums.
    Trellis,
    /// Row scoring against reference logits.
    Score,
    /// Row gathering.
    Rows,
    /// The draft head's argmax.
    Mtp,
    /// The drafter's quantization and loss kernels.
    Drafter,
    /// The drafter's training operations.
    Train,
}

impl Module {
    /// Every module.
    pub const ALL: [Module; 7] = [
        Module::Gptq,
        Module::Trellis,
        Module::Score,
        Module::Rows,
        Module::Mtp,
        Module::Drafter,
        Module::Train,
    ];

    /// The source file's name, which NVRTC also logs the program under.
    pub fn file(self) -> &'static str {
        match self {
            Module::Gptq => "gptq.cu",
            Module::Trellis => "trellis.cu",
            Module::Score => "score.cu",
            Module::Rows => "rows.cu",
            Module::Mtp => "mtp.cu",
            Module::Drafter => "drafter.cu",
            Module::Train => "train.cu",
        }
    }

    /// The source text as it is in the repository, before its includes are expanded.
    pub fn source(self) -> &'static str {
        match self {
            Module::Gptq => include_str!("../../../kernels/quantization/gptq.cu"),
            Module::Trellis => include_str!("../../../kernels/quantization/trellis.cu"),
            Module::Score => include_str!("../../../kernels/quantization/score.cu"),
            Module::Rows => include_str!("../../../kernels/quantization/rows.cu"),
            Module::Mtp => include_str!("../../../kernels/quantization/mtp.cu"),
            Module::Drafter => include_str!("../../../kernels/quantization/drafter.cu"),
            Module::Train => include_str!("../../../kernels/quantization/train.cu"),
        }
    }

    /// The text the compiler is given: the source with its shipped includes expanded.
    pub fn expanded_source(self) -> String {
        expand_includes(self.source(), &source::header)
    }

    /// The kernels of this module the library launches, by their symbol names. A module holds more kernels than
    /// these (the ones no ported code launches); they are compiled and not resolved.
    pub fn kernels(self) -> &'static [&'static str] {
        match self {
            Module::Gptq => &GPTQ,
            Module::Trellis => &TRELLIS,
            Module::Score => &["score_rows"],
            Module::Rows => &["gather_rows"],
            Module::Mtp => &["argmax_rows"],
            Module::Drafter => &DRAFTER,
            Module::Train => &TRAIN,
        }
    }
}

const GPTQ: [&str; 16] = [
    "grouped_choose",
    "nvfp4_choose",
    "sd4_costs",
    "segment",
    "gather_group",
    "bf16_to_float_permuted",
    "float_to_bf16",
    "permute_symmetric",
    "read_diagonal",
    "write_diagonal",
    "mirror_lower",
    "mirror_upper",
    "amax_bf16",
    "slice_columns",
    "zero_columns",
    "fill_bytes",
];

const DRAFTER: [&str; 12] = [
    "absolute_maximum",
    "nvfp4_round",
    "nvfp4_decode_bf16",
    "drafter_softmax",
    "drafter_softmax_backward",
    "lineage_dot",
    "lineage_spread",
    "lineage_gather",
    "copy_columns",
    "draft_cross_entropy",
    "float_to_bf16",
    "q3_embedding_rows",
];

/// The drafter's subset of the training operations.
const TRAIN: [&str; 12] = [
    "adam",
    "bf16_to_float",
    "copy_floats",
    "fill",
    "gate_forward",
    "gate_backward",
    "qk_norm_rope_forward",
    "qk_norm_rope_backward",
    "rms_norm_forward",
    "rms_norm_backward",
    "swiglu_forward",
    "swiglu_backward",
];

/// The trellis kernels: for each rate K (integer 2..=6, half-step 2.5 and 3.5 named by their lower rate 2 and 3) the
/// Viterbi encode, the packing, the decode and the reconstruction, then the rotations, the scale and tile kernels
/// and the quantizer's helpers.
const TRELLIS: [&str; 64] = [
    "viterbi_k2",
    "pack_k2",
    "decode_k2",
    "viterbi_k3",
    "pack_k3",
    "decode_k3",
    "viterbi_k4",
    "pack_k4",
    "decode_k4",
    "viterbi_k5",
    "pack_k5",
    "decode_k5",
    "viterbi_k6",
    "pack_k6",
    "decode_k6",
    "viterbi_frac_k2",
    "pack_frac_k2",
    "decode_frac_k2",
    "viterbi_frac_k3",
    "pack_frac_k3",
    "decode_frac_k3",
    "reconstruct_k2",
    "reconstruct_k3",
    "reconstruct_k4",
    "reconstruct_k5",
    "reconstruct_k6",
    "reconstruct_frac_k2",
    "reconstruct_frac_k3",
    "hadamard128_rows",
    "hadamard128_columns",
    "strip_tiles",
    "scale_rows_fp16",
    "scale_columns_fp16",
    "transpose_to_bf16",
    "fill_floats",
    "copy_floats",
    "float_to_half",
    "axpy_floats",
    "subtract_floats",
    "sum_products",
    "strided_mean",
    "add_diagonal",
    "diagonal_io",
    "triangle",
    "scale_floats",
    "bf16_to_float_transposed",
    "invert_diagonal_blocks",
    "right_multiply_blocks",
    "column_rms",
    "row_rms",
    "output_scales",
    "input_scales",
    "tile_mean_square",
    "gather_tiles",
    "candidate_tiles",
    "candidate_mse",
    "strip_gather_sum",
    "strip_error",
    "column_dots",
    "row_dots",
    "column_ratio",
    "hessian_product",
    "clean_row_scales",
    "scale_scalar",
];

/// One kernel launch of a module: the kernel's name, its geometry and its arguments. Only the planners build one, so
/// its arguments always match the kernel's parameter list, and the module it names is the one that lists the kernel.
#[derive(Clone, Debug)]
pub struct Step {
    module: Module,
    kernel: &'static str,
    launch: Launch,
    args: Args,
}

impl Step {
    /// A launch of the kernel `name` of `module`, or `None` when the module's table does not list it.
    pub(crate) fn new(module: Module, name: &str, launch: Launch, args: Args) -> Option<Self> {
        let kernel = module.kernels().iter().copied().find(|listed| *listed == name)?;
        Some(Self {
            module,
            kernel,
            launch,
            args,
        })
    }

    /// The same step with `bytes` of dynamic shared memory.
    pub(crate) fn with_shared_bytes(mut self, bytes: u32) -> Self {
        self.launch.shared_bytes = bytes;
        self
    }

    /// The module the kernel belongs to.
    pub fn module(&self) -> Module {
        self.module
    }

    /// The kernel's symbol name.
    pub fn kernel(&self) -> &'static str {
        self.kernel
    }

    /// The grid, block and dynamic shared memory.
    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    /// The arguments, in the kernel's parameter order.
    pub fn args(&self) -> &Args {
        &self.args
    }
}

/// Why steps could not be enqueued.
#[derive(Debug, Error)]
pub enum StepError {
    /// A step names a kernel of another module than the one being enqueued.
    #[error("step for {kernel} belongs to {}, not the module being run", module.file())]
    WrongModule { module: Module, kernel: &'static str },
    /// The module has no resolved kernel of this name.
    #[error("no resolved kernel {0}")]
    NoKernel(&'static str),
    /// The lane refused a launch.
    #[error("device failure: {0}")]
    Gpu(#[from] GpuError),
}

/// Enqueues `steps` of `module` in order on `lane`, resolving each by name in `kernels`. Every step is looked up
/// before the first launch, so a step that does not belong leaves nothing on the lane.
///
/// On a lane failure the steps before it are already on the lane: their addresses must stay valid until the lane
/// retires them.
///
/// # Safety
/// Every address in the steps is valid for the access its kernel makes until the work retires, and no other work
/// uses those ranges meanwhile.
pub(crate) unsafe fn enqueue_steps<E, K>(
    module: Module,
    kernels: &HashMap<&'static str, K>,
    lane: &mut E,
    steps: &[Step],
) -> Result<(), StepError>
where
    E: LaneEnqueue<Kernel = K>,
{
    let resolved = steps
        .iter()
        .map(|step| {
            if step.module != module {
                return Err(StepError::WrongModule {
                    module: step.module,
                    kernel: step.kernel,
                });
            }
            kernels.get(step.kernel).ok_or(StepError::NoKernel(step.kernel))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (step, kernel) in steps.iter().zip(resolved) {
        // SAFETY: the caller vouches for the addresses; only the planners build a step, with the arguments in the
        // kernel's parameter order.
        unsafe { lane.launch(kernel, step.launch, &step.args) }?;
    }
    Ok(())
}

/// Where the compiler and the headers are, where compiled modules are cached, and the stack a compile runs on.
#[derive(Clone, Debug)]
pub struct ModuleOptions {
    /// The CUDA header directory (`cuda_bf16.h`, `cuda_fp16.h`), put on the include path.
    pub headers: PathBuf,
    /// The NVRTC library to load; the bundled one beside the engine library. `None` leaves the choice to the
    /// backend's search order, which only a test without a device may do.
    pub compiler: Option<PathBuf>,
    /// Where compiled modules are cached on disk; `None` compiles every time.
    pub cache_dir: Option<PathBuf>,
    /// The stack, in bytes, of the thread a compile runs on.
    pub compile_stack: usize,
}

impl ModuleOptions {
    /// Options with the headers at `headers`, no pinned compiler, no disk cache and the engine's default stack.
    pub fn new(headers: PathBuf) -> Self {
        Self {
            headers,
            compiler: None,
            cache_dir: None,
            compile_stack: DEFAULT_COMPILE_STACK_BYTES,
        }
    }

    /// The headers, the bundled compiler and the compile stack of `sources`: the places the engine's own kernels
    /// compile with, so one compiler serves both.
    pub fn from_sources(sources: &KernelSources) -> Self {
        Self {
            headers: sources.include().to_owned(),
            compiler: sources.compiler().map(ToOwned::to_owned),
            cache_dir: None,
            compile_stack: sources.compile_stack(),
        }
    }

    /// The options of the product `library` belongs to: its `cuda/include` and its bundled NVRTC.
    pub fn beside(library: &CudaLibrary) -> Result<Self, LoadError> {
        Ok(Self::from_sources(&KernelSources::beside(library)?))
    }

    /// The compile options of `module`.
    pub fn compile_options(&self, module: Module) -> CompileOptions {
        CompileOptions {
            name: module.file().to_owned(),
            options: vec!["--std=c++17".to_owned()],
            include_paths: vec![self.headers.clone()],
            headers: Vec::new(),
            cache_dir: self.cache_dir.clone(),
            compiler: self.compiler.clone(),
            arch_specific: false,
        }
    }
}

/// Why a module could not be loaded.
#[derive(Debug, Error)]
pub enum ModuleError {
    /// The source does not compile; the module and the compiler's log.
    #[error("{} does not compile:\n{log}", module.file())]
    Compile { module: Module, log: String },
    /// The module compiled and lacks a listed kernel.
    #[error("{} has no kernel {kernel}: {cause}", module.file())]
    Kernel {
        module: Module,
        kernel: &'static str,
        cause: GpuError,
    },
    /// No bundled NVRTC was found to compile with.
    #[error("no bundled NVRTC was found beside the library")]
    CompilerMissing,
    /// The compiler would give PTX for this device, which the driver compiles again; the codes would not be the
    /// ones every other build gives.
    #[error("the compiler targets PTX for this device, not its machine code")]
    NoMachineCode,
    /// The compile thread could not run.
    #[error("the compile thread failed: {0}")]
    Thread(LoadError),
    /// The compiler or the device failed before any module was compiled.
    #[error("the compiler cannot be used: {0}")]
    Compiler(GpuError),
    /// The device or the driver failed.
    #[error("{} cannot be loaded: {source}", module.file())]
    Gpu { module: Module, source: GpuError },
}

/// A compiled module with every kernel of its table resolved.
pub struct Compiled<M: GpuModule> {
    module: M,
    kernels: Vec<<M::Gpu as Gpu>::Kernel>,
    names: &'static [&'static str],
}

impl<M: GpuModule> Debug for Compiled<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Compiled")
            .field("kernels", &self.names.len())
            .finish_non_exhaustive()
    }
}

impl<M: GpuModule> Compiled<M> {
    /// The loaded module, for a later lookup of a kernel with its own limits.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// The kernel `name`, when the module's table lists it.
    pub fn kernel(&self, name: &str) -> Option<&<M::Gpu as Gpu>::Kernel> {
        let index = self.names.iter().position(|listed| *listed == name)?;
        self.kernels.get(index)
    }

    /// Every listed kernel with its name, in table order.
    pub fn kernels(&self) -> impl Iterator<Item = (&'static str, &<M::Gpu as Gpu>::Kernel)> {
        self.names.iter().copied().zip(self.kernels.iter())
    }
}

/// The compiled modules of one device: each module is compiled the first time it is asked for and shared after.
pub struct Modules<M: GpuModule> {
    gpu: Arc<M::Gpu>,
    options: ModuleOptions,
    loaded: HashMap<Module, Arc<Compiled<M>>>,
}

impl<M: GpuModule> Debug for Modules<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Modules")
            .field("loaded", &self.loaded.len())
            .finish_non_exhaustive()
    }
}

impl<M: GpuModule> Modules<M> {
    /// An empty cache for `gpu`.
    pub fn new(gpu: Arc<M::Gpu>, options: ModuleOptions) -> Self {
        Self {
            gpu,
            options,
            loaded: HashMap::new(),
        }
    }

    /// The compiled `module`, compiling it on the first call, on a thread with the options' stack. A first call
    /// blocks the caller for as long as NVRTC takes (seconds for the largest modules): it is a startup call, made
    /// before any frame runs.
    pub fn load(&mut self, module: Module) -> Result<Arc<Compiled<M>>, ModuleError> {
        if let Some(compiled) = self.loaded.get(&module) {
            return Ok(Arc::clone(compiled));
        }
        let (gpu, options) = (&self.gpu, &self.options);
        let compiled = on_compile_stack(options.compile_stack, || compile::<M>(gpu, options, module))
            .map_err(ModuleError::Thread)??;
        let compiled = Arc::new(compiled);
        self.loaded.insert(module, Arc::clone(&compiled));
        Ok(compiled)
    }

    /// Whether `module` has been compiled.
    pub fn is_loaded(&self, module: Module) -> bool {
        self.loaded.contains_key(&module)
    }
}

impl Modules<CudaModule> {
    /// The modules of the CUDA device `gpu`, compiled with the compiler `options` pin. The compiler must be the one
    /// the process already uses or the first it loads (one NVRTC serves the process), and must give machine code for
    /// the device.
    pub fn open(gpu: &Arc<CudaGpu>, options: ModuleOptions) -> Result<Self, ModuleError> {
        let compiler = options.compiler.as_deref().ok_or(ModuleError::CompilerMissing)?;
        let nvrtc = Nvrtc::shared(Some(compiler)).map_err(ModuleError::Compiler)?;
        check_target(nvrtc.target_for(gpu.arch()).map_err(ModuleError::Compiler)?)?;
        Ok(Self::new(Arc::clone(gpu), options))
    }
}

/// Refuses every target but machine code for the device's own architecture: PTX is compiled again by the driver, and
/// the architecture-specific target runs on one architecture only.
pub fn check_target(target: Target) -> Result<(), ModuleError> {
    match target {
        Target::Cubin(_) => Ok(()),
        Target::CubinSpecific(_) | Target::Ptx(_) => Err(ModuleError::NoMachineCode),
    }
}

/// Compiles `module` for `gpu` and resolves each kernel of its table.
pub fn compile<M: GpuModule>(
    gpu: &Arc<M::Gpu>,
    options: &ModuleOptions,
    module: Module,
) -> Result<Compiled<M>, ModuleError> {
    let loaded =
        M::from_source(gpu, &module.expanded_source(), &options.compile_options(module)).map_err(
            |error| match error {
                GpuError::Compile { log } => ModuleError::Compile { module, log },
                source => ModuleError::Gpu { module, source },
            },
        )?;
    let names = module.kernels();
    let mut kernels = Vec::with_capacity(names.len());
    for name in names {
        let kernel = loaded.kernel(name).map_err(|cause| match cause {
            GpuError::NotFound(_) => ModuleError::Kernel {
                module,
                kernel: name,
                cause,
            },
            source => ModuleError::Gpu { module, source },
        })?;
        kernels.push(kernel);
    }
    Ok(Compiled {
        module: loaded,
        kernels,
        names,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use euhedral_gpu::Kernel as _;
    use euhedral_gpu::fake::{FakeGpu, FakeModule};

    use super::*;

    /// Whether `source` defines the kernel `name`: a function of that name, or a macro that pastes a parameter onto
    /// a prefix of it and is instantiated with the rest of it (`viterbi_k##K` with `VITERBI(2)` defines `viterbi_k2`).
    fn defines(source: &str, name: &str) -> bool {
        if source.contains(&format!(" {name}(")) {
            return true;
        }
        let lines: Vec<&str> = source.lines().collect();
        lines.iter().enumerate().any(|(at, line)| {
            let Some(head) = line.strip_prefix("#define ") else {
                return false;
            };
            let Some((macro_name, rest)) = head.split_once('(') else {
                return false;
            };
            let Some((parameter, _)) = rest.split_once(')') else {
                return false;
            };
            let mut body = String::new();
            let mut index = at;
            while let Some(text) = lines.get(index) {
                body.push_str(text);
                if !text.trim_end().ends_with('\\') {
                    break;
                }
                index += 1;
            }
            (1..name.len()).filter(|end| name.is_char_boundary(*end)).any(|end| {
                let (prefix, suffix) = name.split_at(end);
                body.contains(&format!("{prefix}##{parameter}"))
                    && lines
                        .iter()
                        .any(|call| call.trim() == format!("{macro_name}({suffix})"))
            })
        })
    }

    #[test]
    fn a_macro_instantiation_defines_only_the_instantiated_names() {
        let source = "#define MAKE(K) void viterbi_k##K(int x) {}\nMAKE(2)\nMAKE(3)\nvoid plain(int x) {}\n";
        assert!(defines(source, "viterbi_k2") && defines(source, "viterbi_k3") && defines(source, "plain"));
        assert!(!defines(source, "viterbi_k9") && !defines(source, "viterbi_k"));
    }

    #[test]
    fn every_listed_kernel_is_defined_by_its_source_once_in_the_table() {
        for module in Module::ALL {
            let source = module.expanded_source();
            let mut seen = HashSet::new();
            for name in module.kernels() {
                assert!(defines(&source, name), "{} has no kernel {name}", module.file());
                assert!(seen.insert(*name), "{} lists {name} twice", module.file());
            }
        }
    }

    #[test]
    fn only_machine_code_for_the_device_is_accepted() {
        assert!(check_target(Target::Cubin(120)).is_ok());
        assert!(matches!(check_target(Target::Ptx(90)), Err(ModuleError::NoMachineCode)));
        assert!(matches!(
            check_target(Target::CubinSpecific(120)),
            Err(ModuleError::NoMachineCode)
        ));
    }

    #[test]
    fn the_trellis_source_includes_its_codebook_once_expanded() {
        let source = Module::Trellis.expanded_source();
        assert!(!source.contains("#include \"trellis/codebook.cuh\""));
        assert!(Module::Gptq.expanded_source() == Module::Gptq.source());
    }

    #[test]
    fn a_module_loads_once_and_resolves_every_listed_kernel() {
        let gpu = FakeGpu::new();
        for module in Module::ALL {
            for name in module.kernels() {
                gpu.define(name, |_, _, _| Ok(()));
            }
        }
        let mut cache = Modules::<FakeModule>::new(gpu, ModuleOptions::new(PathBuf::from("/cuda/include")));
        assert!(!cache.is_loaded(Module::Gptq));
        let first = cache.load(Module::Gptq).unwrap();
        let second = cache.load(Module::Gptq).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert!(cache.is_loaded(Module::Gptq));
        assert_eq!(first.kernels().count(), Module::Gptq.kernels().len());
        assert_eq!(first.kernel("mirror_upper").unwrap().name(), "mirror_upper");
        assert!(first.kernel("not_listed").is_none());
    }

    #[test]
    fn a_missing_kernel_names_the_module_and_the_kernel() {
        let gpu = FakeGpu::new();
        let error = compile::<FakeModule>(&gpu, &ModuleOptions::new(PathBuf::from("/h")), Module::Rows).unwrap_err();
        let text = error.to_string();
        assert!(text.contains("rows.cu") && text.contains("gather_rows"), "{text}");
    }
}
