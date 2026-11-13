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
use std::fs::{read, read_dir, read_to_string};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{Builder, scope};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, Nvrtc, Target};
use euhedral_gpu::{CompileOptions, GpuError, Kernel, Module as _};
use thiserror::Error;

use crate::CudaLibrary;
use crate::launch::catalog::{KERNELS, KernelId, Module, ids};

/// The dynamic shared memory a launch may use without an opt-in, in bytes (the driver's default per-block limit).
pub const DEFAULT_DYNAMIC_SHARED_BYTES: u32 = 48 * 1024;

/// The stack of the thread that compiles a module, in bytes unless the caller sets another: the native loader's own
/// (256 MiB), because NVRTC recurses deeply on the largest modules. The engine passes `kernels.compile.stack`.
pub const DEFAULT_COMPILE_STACK_BYTES: usize = 256 << 20;

/// The NVRTC library files of the bundled compiler start with this. This is the one place that names the platform's
/// library file name; remove the condition when the RustEE pin moves to a revision that loads on Windows.
#[cfg(target_os = "linux")]
const COMPILER_FILE_PREFIX: &str = "libnvrtc.so";

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
    /// Kernel arguments of 128 bytes (tensor maps). While `false`, the kernels that take them are unavailable.
    pub const TENSOR_MAP_ARGUMENTS: bool = false;
}

/// Why a kernel cannot be launched at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unavailable {
    /// The module compiles only for the architecture-specific target, which the pinned RustEE revision cannot make.
    ArchitectureSpecificTarget,
    /// The kernel takes tensor maps (128-byte parameters), which `Arguments` cannot carry yet.
    TensorMapArguments,
    /// The module does not compile; the module's name and the compiler's log.
    Compile { module: &'static str, log: String },
    /// The module compiled and does not contain the kernel's symbol.
    SymbolMissing,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Unavailable::ArchitectureSpecificTarget => "needs the architecture-specific compile target (sm_XYa)",
            Unavailable::TensorMapArguments => "takes tensor-map arguments, which the planned arguments cannot carry",
            Unavailable::Compile { module, log } => {
                return write!(formatter, "module {module} does not compile: {log}");
            }
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
        // Seam: with the capability, the opt-in is raised to `shared_opt_in_bytes` where the kernel is resolved.
        const _: () = assert!(
            !pin::DYNAMIC_SHARED_OPT_IN,
            "fill the shared-memory opt-in seam, then remove this"
        );
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
        ids().zip(&self.slots)
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
    #[error("no bundled NVRTC was found beside the library")]
    CompilerMissing,
    #[error("the NVRTC of this process is not the bundled one: {cause}")]
    CompilerNotBundled { cause: String },
    #[error("the NVRTC loaded is version {}.{} and the bundled one is {}.{}", loaded.0, loaded.1, bundled.0, bundled.1)]
    CompilerVersion { loaded: (i32, i32), bundled: (i32, i32) },
    #[error("the compile thread failed: {0}")]
    CompileThread(String),
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
    compile_stack: usize,
    /// The digest of the installed tree's manifest (every source and header), which names the cache directory.
    manifest: Option<u64>,
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
            compile_stack: DEFAULT_COMPILE_STACK_BYTES,
            manifest: manifest_digest(product),
        })
    }

    /// Sources at explicit places.
    pub fn new(share: PathBuf, include: PathBuf, compiler: Option<PathBuf>) -> Self {
        let manifest = share.parent().and_then(Path::parent).and_then(manifest_digest);
        Self {
            share,
            include,
            compiler,
            cache_dir: None,
            compile_stack: DEFAULT_COMPILE_STACK_BYTES,
            manifest,
        }
    }

    /// The stack of the compile thread, in bytes (the engine's `kernels.compile.stack`).
    #[must_use]
    pub fn with_compile_stack(mut self, bytes: usize) -> Self {
        self.compile_stack = bytes;
        self
    }

    /// Caches compiled modules under `directory`. RustEE keys an entry by the source text, options, target and
    /// compiler but not by the headers the source includes, so the entries live in a subdirectory named by the digest of
    /// the installed tree's manifest (`files.sha256`, which lists every source and header): another tree never reads
    /// them. Without a manifest there is no cache.
    #[must_use]
    pub fn with_cache_dir(mut self, directory: PathBuf) -> Self {
        self.cache_dir = Some(directory);
        self
    }

    /// The directory the cache entries live in, when there is a cache and a manifest to name it.
    pub fn cache_path(&self) -> Option<PathBuf> {
        let directory = self.cache_dir.as_ref()?;
        Some(directory.join(format!("tree-{:016x}", self.manifest?)))
    }

    #[cfg(test)]
    fn with_manifest(mut self, manifest: &[u8]) -> Self {
        self.manifest = Some(digest(manifest));
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
            cache_dir: self.cache_path(),
            compiler: self.compiler.clone(),
        }
    }
}

/// FNV-1a over `bytes`: stable across runs and toolchains, which a cache directory name needs.
fn digest(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    })
}

#[allow(
    clippy::disallowed_methods,
    reason = "startup: the manifest of the installed tree is read once, before any frame runs"
)]
fn manifest_digest(product: &Path) -> Option<u64> {
    read(product.join("files.sha256")).ok().map(|bytes| digest(&bytes))
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
                .is_some_and(|name| name.starts_with(COMPILER_FILE_PREFIX) && !name.contains("builtins"))
        })
        .collect();
    found.sort();
    found.pop()
}

/// What compiling one module gave.
enum Compiled {
    Module(CudaModule),
    Unavailable(Unavailable),
}

/// The `major.minor` the bundled builtins library's file name carries (`libnvrtc-builtins.so.13.1`).
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the bundled compiler's version is read from a directory listing once, before any frame runs"
)]
fn bundled_version(compiler: &Path) -> Option<(i32, i32)> {
    read_dir(compiler.parent()?).ok()?.find_map(|entry| {
        let name = entry.ok()?.file_name().into_string().ok()?;
        let version = name.strip_prefix("libnvrtc-builtins.so.")?;
        let mut parts = version.split('.');
        Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
    })
}

/// Compiles every module the catalog names that the pinned RustEE revision can, loads it, and resolves every catalog
/// kernel. A module that does not compile makes only its own kernels unavailable. Blocking: it is a startup call
/// (compiling reads files and runs NVRTC), and the compiling runs on a thread of its own with the stack
/// [`KernelSources::with_compile_stack`] gives it.
pub fn load(gpu: &Arc<CudaGpu>, sources: &KernelSources) -> Result<KernelTable, LoadError> {
    // The bundled compiler, and no other: the native library compiles with it, and the two must agree.
    let bundled = sources.compiler().ok_or(LoadError::CompilerMissing)?;
    let nvrtc = Nvrtc::shared(Some(bundled)).map_err(|error| match error {
        GpuError::Invalid(cause) => LoadError::CompilerNotBundled { cause },
        other => LoadError::Gpu(other),
    })?;
    if let Some(version) = bundled_version(bundled)
        && version != nvrtc.version()
    {
        return Err(LoadError::CompilerVersion {
            loaded: nvrtc.version(),
            bundled: version,
        });
    }
    // The compiler must produce machine code for this device; PTX would be compiled again by the driver, and the
    // native library never uses it.
    if !matches!(nvrtc.target_for(gpu.arch())?, Target::Cubin(_)) {
        return Err(LoadError::NoMachineCode);
    }
    let (modules, timings) = compile_all(gpu, sources)?;
    let mut slots = Vec::with_capacity(KERNELS.len());
    for spec in &KERNELS {
        if spec.tensor_map_arguments() && !pin::TENSOR_MAP_ARGUMENTS {
            // Seam: with tensor-map arguments in RustEE's `Args`, the kernel resolves like any other.
            const _: () = assert!(!pin::TENSOR_MAP_ARGUMENTS, "fill the tensor-map seam, then remove this");
            slots.push(Slot::Unavailable(Unavailable::TensorMapArguments));
            continue;
        }
        let compiled = modules
            .iter()
            .find(|(module, _)| *module == spec.module)
            .map(|(_, compiled)| compiled);
        let slot = match compiled {
            Some(Compiled::Module(module)) => match module.kernel(spec.name) {
                Ok(kernel) => {
                    debug_assert_eq!(kernel.name(), spec.name);
                    Slot::Ready(Resolved {
                        kernel,
                        shared_opt_in_bytes: spec.shared_opt_in_bytes,
                    })
                }
                Err(GpuError::NotFound(_)) => Slot::Unavailable(Unavailable::SymbolMissing),
                Err(other) => return Err(LoadError::Gpu(other)),
            },
            Some(Compiled::Unavailable(reason)) => Slot::Unavailable(reason.clone()),
            None => Slot::Unavailable(Unavailable::SymbolMissing),
        };
        slots.push(slot);
    }
    Ok(KernelTable { slots, timings })
}

type Compilations = (Vec<(Module, Compiled)>, Vec<ModuleTiming>);

/// Compiles each distinct module, in first-appearance order, on a thread with the configured stack.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: NVRTC needs a deep stack the caller's thread may not have, and loading waits for the compile by nature"
)]
fn compile_all(gpu: &Arc<CudaGpu>, sources: &KernelSources) -> Result<Compilations, LoadError> {
    scope(|scope| {
        let worker = Builder::new()
            .name("kernel-compile".to_owned())
            .stack_size(sources.compile_stack)
            .spawn_scoped(scope, || -> Result<Compilations, LoadError> {
                let mut modules: Vec<(Module, Compiled)> = Vec::new();
                let mut timings = Vec::new();
                for spec in &KERNELS {
                    if modules.iter().any(|(module, _)| *module == spec.module) {
                        continue;
                    }
                    let started = Instant::now();
                    let compiled = compile(gpu, sources, spec.module)?;
                    if matches!(compiled, Compiled::Module(_)) {
                        timings.push(ModuleTiming {
                            module: spec.module,
                            elapsed: started.elapsed(),
                        });
                    }
                    modules.push((spec.module, compiled));
                }
                Ok((modules, timings))
            })
            .map_err(|error| LoadError::CompileThread(error.to_string()))?;
        worker.join().map_err(|payload| {
            let what = payload
                .downcast_ref::<&str>()
                .map(ToString::to_string)
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic".to_owned());
            LoadError::CompileThread(what)
        })?
    })
}

/// The module compiled and loaded, or why it is unavailable. Only a source that does not compile is unavailable; every
/// other failure (an unreadable file, a driver error) fails the load.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the kernel source is read once, before any frame runs"
)]
fn compile(gpu: &Arc<CudaGpu>, sources: &KernelSources, module: Module) -> Result<Compiled, LoadError> {
    // Seam: with the capability, compile for `sm_XYa` here (the device must be compute capability 12.x; on another
    // device the module stays unavailable) by setting the architecture-specific field of the options below.
    const _: () = assert!(
        !pin::ARCHITECTURE_SPECIFIC_TARGET,
        "fill the architecture-specific seam, then remove this"
    );
    if module.architecture_specific() && !pin::ARCHITECTURE_SPECIFIC_TARGET {
        return Ok(Compiled::Unavailable(Unavailable::ArchitectureSpecificTarget));
    }
    let path = sources.share.join(module.root());
    let text = read_to_string(&path).map_err(|source| LoadError::Source { path, source })?;
    match CudaModule::from_source(gpu, &text, &sources.options(module)) {
        Ok(compiled) => Ok(Compiled::Module(compiled)),
        Err(GpuError::Compile { log }) => Ok(Compiled::Unavailable(Unavailable::Compile {
            module: module.root(),
            log,
        })),
        Err(other) => Err(LoadError::Gpu(other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sources() -> KernelSources {
        KernelSources::new(
            PathBuf::from("/p/share/euhedral_cuda"),
            PathBuf::from("/p/cuda/include"),
            None,
        )
    }

    #[test]
    fn the_compile_options_are_the_native_loaders() {
        let options = sources().options(Module::Gdn);
        assert_eq!(options.options, ["--std=c++14"]);
        assert!(options.headers.is_empty());
        let root = Module::Gdn.root();
        let domain = root.rsplit_once('/').map_or("", |(directory, _)| directory);
        assert_eq!(
            options.include_paths,
            [
                PathBuf::from("/p/cuda/include"),
                PathBuf::from("/p/share/euhedral_cuda").join(domain),
                PathBuf::from("/p/share/euhedral_cuda"),
            ]
        );
        assert_eq!(options.name, root);
    }

    #[test]
    fn the_cache_directory_follows_the_manifest_of_the_installed_tree() {
        let cached = |manifest: &[u8]| {
            sources()
                .with_cache_dir(PathBuf::from("/c"))
                .with_manifest(manifest)
                .cache_path()
        };
        let before = cached(b"aa  cuda/include/x.h\n");
        assert!(before.as_ref().is_some_and(|path| path.starts_with("/c")));
        assert_eq!(before, cached(b"aa  cuda/include/x.h\n"));
        assert_ne!(before, cached(b"ab  cuda/include/x.h\n"));
        assert_eq!(sources().with_manifest(b"aa").cache_path(), None);
    }

    #[test]
    fn the_unmanifested_tree_has_no_cache() {
        let mut bare = sources().with_cache_dir(PathBuf::from("/c"));
        bare.manifest = None;
        assert_eq!(bare.cache_path(), None);
    }

    #[test]
    fn the_compile_stack_defaults_to_the_native_size_and_is_set_by_the_caller() {
        assert_eq!(sources().compile_stack, 256 << 20);
        assert_eq!(sources().with_compile_stack(64 << 20).compile_stack, 64 << 20);
    }

    #[test]
    fn the_catalog_iterates_in_order_with_its_slots() {
        let table = KernelTable {
            slots: KERNELS
                .iter()
                .map(|_| Slot::Unavailable(Unavailable::SymbolMissing))
                .collect(),
            timings: Vec::new(),
        };
        assert_eq!(table.iter().count(), KERNELS.len());
        assert!(table.iter().enumerate().all(|(index, (id, _))| id.index() == index));
    }
}
