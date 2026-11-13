//! Compiling and loading the kernel modules through RustEE, and resolving every catalog kernel to a kernel handle.
//!
//! The kernel sources are the ones the native library installs unchanged beside itself, `<product>/share/euhedral_cuda`
//! (`<domain>/kernels.cu` and their headers), so the native library and this loader compile the identical files with
//! the identical compiler and options: the bundled NVRTC (`<product>/lib`), `--std=c++14`, the device's own
//! architecture as machine code, and the include paths `<product>/cuda/include`, the source's directory and the kernel
//! root, in the native loader's order. A module is compiled once, at startup, by [`load`]; the result is a
//! [`KernelTable`] indexed like the catalog.
//!
//! RustEE opens the primary context of the device the native library bound (one driver library, one context), so a
//! handle resolved here launches in the context the native library's buffers live in.
//!
//! What the pinned RustEE revision cannot do is named in [`pin`], the one place to edit when a later revision can.

use std::fmt;
use std::fs::{read_dir, read_to_string};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, Nvrtc, Target};
use euhedral_gpu::{CompileOptions, GpuError, Kernel, Module as _};
use thiserror::Error;

use crate::CudaLibrary;
use crate::launch::catalog::{KERNELS, KernelId, Module, lookup};

/// The dynamic shared memory a launch may use without an opt-in, in bytes (the driver's default per-block limit).
pub const DEFAULT_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

/// What the pinned RustEE revision lacks. Each constant is `false` at this revision; a later revision that provides
/// the capability flips it here, and the code that reads it (this file for the first two, `queue.rs` for the third)
/// has its one marked branch to fill in. Nothing else in the crate knows.
pub mod pin {
    /// Compiling for an architecture-specific target (`sm_XYa`). While `false`, every module of
    /// [`Module::architecture_specific`](crate::launch::catalog::Module::architecture_specific) is unavailable.
    pub const ARCHITECTURE_SPECIFIC_TARGET: bool = false;
    /// Raising a kernel's maximum dynamic shared memory above the default. While `false`, a launch that needs more than
    /// [`DEFAULT_DYNAMIC_SHARED_BYTES`](super::DEFAULT_DYNAMIC_SHARED_BYTES) is refused.
    pub const DYNAMIC_SHARED_OPT_IN: bool = false;
    /// Programmatic dependent launch. While `false`, a planned launch's `programmatic` flag is not honoured.
    pub const PROGRAMMATIC_LAUNCH: bool = false;
}

/// Why a kernel cannot be launched at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unavailable {
    /// The module compiles only for the architecture-specific target, which the pinned RustEE revision cannot make.
    ArchitectureSpecificTarget,
    /// The module compiled and does not contain the kernel's symbol.
    SymbolMissing,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Unavailable::ArchitectureSpecificTarget => "needs the architecture-specific compile target (sm_XYa)",
            Unavailable::SymbolMissing => "the compiled module has no such symbol",
        })
    }
}

/// A kernel handle and what launches of it may use.
#[derive(Clone, Debug)]
pub struct Resolved {
    kernel: CudaKernel,
    shared_opt_in_bytes: u32,
}

impl Resolved {
    pub fn kernel(&self) -> &CudaKernel {
        &self.kernel
    }

    /// The size in bytes of each parameter the compiled kernel declares; `None` when the driver does not report them.
    pub fn parameter_sizes(&self) -> Option<&[u32]> {
        self.kernel.parameter_sizes()
    }

    /// The dynamic shared bytes the native library sets as this kernel's maximum (the catalog's record of the need).
    pub fn shared_opt_in_bytes(&self) -> u32 {
        self.shared_opt_in_bytes
    }

    /// Whether launches of this kernel can need more than the default limit, which needs the opt-in.
    pub fn needs_shared_opt_in(&self) -> bool {
        self.shared_opt_in_bytes > DEFAULT_DYNAMIC_SHARED_BYTES
    }

    /// The most dynamic shared bytes a launch of this kernel may use at the pinned RustEE revision.
    pub fn max_shared_bytes(&self) -> u32 {
        if pin::DYNAMIC_SHARED_OPT_IN {
            self.shared_opt_in_bytes.max(DEFAULT_DYNAMIC_SHARED_BYTES)
        } else {
            DEFAULT_DYNAMIC_SHARED_BYTES
        }
    }
}

/// One catalog kernel: ready, or unavailable for a reason.
#[derive(Clone, Debug)]
pub enum Slot {
    Ready(Resolved),
    Unavailable(Unavailable),
}

/// How long one module took to compile (or fetch from the cache) and load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModuleTiming {
    pub module: Module,
    pub elapsed: Duration,
}

/// The resolved kernels, indexed like the catalog ([`KernelId::index`]).
#[derive(Debug)]
pub struct KernelTable {
    slots: Vec<Slot>,
    timings: Vec<ModuleTiming>,
}

impl KernelTable {
    pub fn get(&self, kernel: KernelId) -> &Slot {
        &self.slots[kernel.index()]
    }

    /// The kernel's handle, when it is ready.
    pub fn resolved(&self, kernel: KernelId) -> Option<&Resolved> {
        match self.get(kernel) {
            Slot::Ready(resolved) => Some(resolved),
            Slot::Unavailable(_) => None,
        }
    }

    /// Every catalog kernel with its slot, in catalog order.
    pub fn iter(&self) -> impl Iterator<Item = (KernelId, &Slot)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let id = lookup(KERNELS[index].name)?;
            debug_assert_eq!(id.index(), index);
            Some((id, slot))
        })
    }

    /// The compile and load time of each module that was compiled, in the order they were compiled.
    pub fn timings(&self) -> &[ModuleTiming] {
        &self.timings
    }
}

/// Why the modules could not be loaded.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("the kernel source {path} cannot be read: {source}")]
    Source {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("the library has no product directory around it")]
    Product,
    #[error("the compiler targets PTX for this device, not its machine code")]
    NoMachineCode,
    #[error(transparent)]
    Gpu(#[from] GpuError),
}

/// Where the kernel sources and the compiler are.
#[derive(Clone, Debug)]
pub struct KernelSources {
    share: PathBuf,
    include: PathBuf,
    compiler: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
}

impl KernelSources {
    /// The sources installed beside `library`: `share/euhedral_cuda`, `cuda/include` and the bundled NVRTC in `lib` (the
    /// search order of RustEE when there is none), with no cache.
    pub fn beside(library: &CudaLibrary) -> Result<Self, LoadError> {
        let lib = library.path().parent().ok_or(LoadError::Product)?;
        let product = lib.parent().ok_or(LoadError::Product)?;
        Ok(Self {
            share: product.join("share").join("euhedral_cuda"),
            include: product.join("cuda").join("include"),
            compiler: bundled_compiler(lib),
            cache_dir: None,
        })
    }

    /// Sources at explicit places.
    pub fn new(share: PathBuf, include: PathBuf, compiler: Option<PathBuf>) -> Self {
        Self {
            share,
            include,
            compiler,
            cache_dir: None,
        }
    }

    /// Caches compiled modules in `directory` (RustEE keys an entry by the source text, options, target and compiler;
    /// not by the included headers, so name a directory by the installed tree).
    #[must_use]
    pub fn with_cache_dir(mut self, directory: PathBuf) -> Self {
        self.cache_dir = Some(directory);
        self
    }

    /// The compiler library that will be used, when one was found beside the library.
    pub fn compiler(&self) -> Option<&Path> {
        self.compiler.as_deref()
    }

    fn options(&self, module: Module) -> CompileOptions {
        let root = module.root();
        // The order of the native loader: the CUDA headers, the source's directory, the kernel root.
        let domain = root.rsplit_once('/').map_or("", |(directory, _)| directory);
        CompileOptions {
            name: root.to_owned(),
            options: vec!["--std=c++14".to_owned()],
            include_paths: vec![self.include.clone(), self.share.join(domain), self.share.clone()],
            headers: Vec::new(),
            cache_dir: self.cache_dir.clone(),
            compiler: self.compiler.clone(),
        }
    }
}

/// The NVRTC library file in `lib` (not its builtins library), if there is one.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: finding the bundled compiler lists a directory once, before any frame runs"
)]
fn bundled_compiler(lib: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = read_dir(lib)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("libnvrtc.so") && !name.contains("builtins"))
        })
        .collect();
    found.sort();
    found.pop()
}

/// Compiles every module the catalog names that the pinned RustEE revision can, loads it, and resolves every catalog
/// kernel. Blocking: it is a startup call (compiling reads files and runs NVRTC).
pub fn load(gpu: &Arc<CudaGpu>, sources: &KernelSources) -> Result<KernelTable, LoadError> {
    // The compiler must produce machine code for this device; PTX would be compiled again by the driver, and the
    // native library never uses it.
    let nvrtc = Nvrtc::shared(sources.compiler())?;
    if !matches!(nvrtc.target_for(gpu.arch())?, Target::Cubin(_)) {
        return Err(LoadError::NoMachineCode);
    }
    let mut modules: Vec<(Module, Option<CudaModule>)> = Vec::new();
    let mut timings = Vec::new();
    for spec in &KERNELS {
        if modules.iter().any(|(module, _)| *module == spec.module) {
            continue;
        }
        let started = Instant::now();
        let compiled = compile(gpu, sources, spec.module)?;
        if compiled.is_some() {
            timings.push(ModuleTiming {
                module: spec.module,
                elapsed: started.elapsed(),
            });
        }
        modules.push((spec.module, compiled));
    }
    let slots = KERNELS
        .iter()
        .map(|spec| {
            let compiled = modules
                .iter()
                .find(|(module, _)| *module == spec.module)
                .and_then(|(_, compiled)| compiled.as_ref());
            let Some(module) = compiled else {
                return Slot::Unavailable(Unavailable::ArchitectureSpecificTarget);
            };
            match module.kernel(spec.name) {
                Ok(kernel) => {
                    debug_assert_eq!(kernel.name(), spec.name);
                    Slot::Ready(Resolved {
                        kernel,
                        shared_opt_in_bytes: spec.shared_opt_in_bytes,
                    })
                }
                Err(_) => Slot::Unavailable(Unavailable::SymbolMissing),
            }
        })
        .collect();
    Ok(KernelTable { slots, timings })
}

/// The module compiled and loaded, or `None` when the pinned RustEE revision cannot compile it for its target.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the kernel source is read once, before any frame runs"
)]
fn compile(gpu: &Arc<CudaGpu>, sources: &KernelSources, module: Module) -> Result<Option<CudaModule>, LoadError> {
    if module.architecture_specific() && !pin::ARCHITECTURE_SPECIFIC_TARGET {
        return Ok(None);
        // Seam: with the capability, compile for `sm_XYa` here (the device must be compute capability 12.x; on another
        // device the module stays unavailable) by setting the architecture-specific field of the options below.
    }
    let path = sources.share.join(module.root());
    let text = read_to_string(&path).map_err(|source| LoadError::Source { path, source })?;
    let compiled = CudaModule::from_source(gpu, &text, &sources.options(module))?;
    Ok(Some(compiled))
}
