//! The quantization kernels as NVRTC modules: the sources, the compile options, the kernel names of each module and
//! the loader that compiles a module once and resolves its kernels.
//!
//! The sources are compiled with `--std=c++17` and the device's own architecture (the backend adds
//! `--gpu-architecture`), with the CUDA header directory on the include path. These are not the engine's options
//! (`--std=c++14`): the quantization modules are compiled the same way wherever they are built, so the codes they
//! produce do not depend on the build.
//!
//! The compiler is RustEE's NVRTC facility. The loader takes the device and the places the headers and the compiler
//! are in ([`ModuleOptions`]); where the cuBLAS and cuSOLVER libraries are is not its concern.

mod source;

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::{CompileOptions, Gpu, GpuError, Module as GpuModule};
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
            Module::Gptq => include_str!("../../../../quantization/src/main/resources/cuda/gptq.cu"),
            Module::Trellis => include_str!("../../../../quantization/src/main/resources/cuda/trellis.cu"),
            Module::Score => include_str!("../../../../quantization/src/main/resources/cuda/score.cu"),
            Module::Rows => include_str!("../../../../quantization/src/main/resources/cuda/rows.cu"),
            Module::Mtp => include_str!("../../../../quantization/src/main/resources/cuda/mtp.cu"),
            Module::Drafter => include_str!("../../../../quantization/src/main/resources/cuda/drafter.cu"),
            Module::Train => include_str!("../../../../quantization/src/main/resources/cuda/train.cu"),
        }
    }

    /// The text the compiler is given: the source with its shipped includes expanded.
    pub fn expanded_source(self) -> String {
        expand_includes(self.source(), &source::header)
    }

    /// The kernels of this module the library launches, by their symbol names. A module holds more kernels than
    /// these (the ones only the commands that are not ported use); they are compiled and not resolved.
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

/// Where the compiler and the headers are, and where compiled modules are cached.
#[derive(Clone, Debug)]
pub struct ModuleOptions {
    /// The CUDA header directory (`cuda_bf16.h`, `cuda_fp16.h`), put on the include path.
    pub headers: PathBuf,
    /// The NVRTC library to load; `None` leaves the choice to the backend's search order.
    pub compiler: Option<PathBuf>,
    /// Where compiled modules are cached on disk; `None` compiles every time.
    pub cache_dir: Option<PathBuf>,
}

impl ModuleOptions {
    /// Options with the headers at `headers`, the backend's own compiler search and no disk cache.
    pub fn new(headers: PathBuf) -> Self {
        Self {
            headers,
            compiler: None,
            cache_dir: None,
        }
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
pub struct ModuleCache<M: GpuModule> {
    gpu: Arc<M::Gpu>,
    options: ModuleOptions,
    loaded: HashMap<Module, Arc<Compiled<M>>>,
}

impl<M: GpuModule> Debug for ModuleCache<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("ModuleCache")
            .field("loaded", &self.loaded.len())
            .finish_non_exhaustive()
    }
}

impl<M: GpuModule> ModuleCache<M> {
    /// An empty cache for `gpu`.
    pub fn new(gpu: Arc<M::Gpu>, options: ModuleOptions) -> Self {
        Self {
            gpu,
            options,
            loaded: HashMap::new(),
        }
    }

    /// The compiled `module`, compiling it on the first call. Compiling blocks the caller for as long as NVRTC takes
    /// and recurses deeply on the largest sources: it is a startup call, made on a thread with a large stack.
    pub fn load(&mut self, module: Module) -> Result<Arc<Compiled<M>>, ModuleError> {
        if let Some(compiled) = self.loaded.get(&module) {
            return Ok(Arc::clone(compiled));
        }
        let compiled = Arc::new(compile::<M>(&self.gpu, &self.options, module)?);
        self.loaded.insert(module, Arc::clone(&compiled));
        Ok(compiled)
    }

    /// Whether `module` has been compiled.
    pub fn is_loaded(&self, module: Module) -> bool {
        self.loaded.contains_key(&module)
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
        let kernel = loaded.kernel(name).map_err(|cause| ModuleError::Kernel {
            module,
            kernel: name,
            cause,
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

    /// Whether `source` defines the kernel `name`, directly or through a macro that pastes a suffix onto a prefix.
    fn defines(source: &str, name: &str) -> bool {
        source.contains(&format!(" {name}("))
            || (1..name.len()).any(|end| name.is_char_boundary(end) && source.contains(&format!(" {}##", &name[..end])))
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
    fn the_compile_options_are_the_quantization_ones() {
        let options = ModuleOptions::new(PathBuf::from("/cuda/include")).compile_options(Module::Trellis);
        assert_eq!(options.options, ["--std=c++17"]);
        assert_eq!(options.include_paths, [PathBuf::from("/cuda/include")]);
        assert_eq!(options.name, "trellis.cu");
        assert!(!options.arch_specific);
    }

    #[test]
    fn the_trellis_source_includes_its_codebook_once_expanded() {
        let source = Module::Trellis.expanded_source();
        assert!(!source.contains("#include \"trellis/codebook.cuh\""));
        assert!(source.contains("#include <cuda_fp16.h>"));
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
        let mut cache = ModuleCache::<FakeModule>::new(gpu, ModuleOptions::new(PathBuf::from("/cuda/include")));
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
