//! Compiling and loading the kernel modules through RustEE, and resolving every catalog kernel to a kernel handle.
//!
//! The kernel sources are the ones the native library installs unchanged beside itself, `<product>/share/euhedral_cuda`
//! (`<domain>/kernels.cu` and their headers), so the native library and this loader compile the identical files with
//! the identical compiler and options: the bundled NVRTC (`<product>/lib`), `--std=c++14`, the device's own
//! architecture as machine code (the architecture-specific variant for the modules that need it), and the include paths `<product>/cuda/include`, the source's directory and the kernel
//! root, in the native loader's order. A module is compiled once, at startup, by [`load`]; the result is a
//! [`KernelTable`] indexed like the catalog.
//!
//! RustEE opens the primary context of the device the native library bound (one driver library, one context), so a
//! handle resolved here launches in the context the native library's buffers live in.
//!

use std::fmt;
use std::fs::{read, read_dir, read_to_string};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{Builder, scope};
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaKernel, CudaModule, Nvrtc, Target};
use euhedral_gpu::{CompileOptions, GpuError, Kernel, KernelOptions, Module as _, TensorMap, TiledMap};
use thiserror::Error;

use crate::CudaLibrary;
use crate::launch::catalog::{KERNELS, KernelId, Module, ids};

/// The stack of the thread that compiles a module, in bytes unless the caller sets another: the native loader's own
/// (256 MiB), because NVRTC recurses deeply on the largest modules. The engine is to pass `kernels.compile.stack` when it loads its kernels through
/// RustEE, which it does not do yet.
pub const DEFAULT_COMPILE_STACK_BYTES: usize = 256 << 20;

/// Why a kernel cannot be launched at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unavailable {
    /// The module compiles only for the architecture-specific target, which exists on compute capability 12.x only.
    ArchitectureSpecificTarget,
    /// The module does not compile; the module's name and the compiler's log.
    Compile { module: &'static str, log: String },
    /// The module compiled and does not contain the kernel's symbol.
    SymbolMissing,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Unavailable::ArchitectureSpecificTarget => "needs the architecture-specific compile target (sm_XYa)",
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
    gpu: Arc<CudaGpu>,
}

impl Resolved {
    pub fn kernel(&self) -> &CudaKernel {
        &self.kernel
    }

    /// The size in bytes of each parameter the compiled kernel declares; `None` when the driver does not report them.
    pub fn parameter_sizes(&self) -> Option<&[u32]> {
        self.kernel.parameter_sizes()
    }

    /// The most dynamic shared bytes a launch of this kernel may use: the limit the kernel was opted in to, or the
    /// driver's own. `u32::MAX` when the driver reports none, which leaves the check to the driver at the launch.
    pub fn max_shared_bytes(&self) -> u32 {
        self.kernel.max_dynamic_shared_bytes().unwrap_or(u32::MAX)
    }

    /// Encodes a tensor map of the kernel's device.
    pub fn tensor_map(&self, description: &TiledMap) -> Result<TensorMap, GpuError> {
        TensorMap::tiled(&self.gpu, description)
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
            arch_specific: module.architecture_specific(),
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
                .is_some_and(platform::is_compiler_file)
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

/// The `major.minor` the bundled builtins library's file name carries.
#[allow(
    clippy::disallowed_methods,
    reason = "startup: the bundled compiler's version is read from a directory listing once, before any frame runs"
)]
fn bundled_version(compiler: &Path) -> Option<(i32, i32)> {
    read_dir(compiler.parent()?).ok()?.find_map(|entry| {
        let name = entry.ok()?.file_name().into_string().ok()?;
        platform::builtins_version(&name)
    })
}

/// The names of the bundled NVRTC files: the one place that knows a platform's library file names.
mod platform {
    /// Whether `name` is the NVRTC library (not its builtins library): `libnvrtc.so.13.1`.
    #[cfg(not(windows))]
    pub(super) fn is_compiler_file(name: &str) -> bool {
        name.starts_with("libnvrtc.so")
    }

    /// The `major.minor` of a builtins library file name: `libnvrtc-builtins.so.13.1`.
    #[cfg(not(windows))]
    pub(super) fn builtins_version(name: &str) -> Option<(i32, i32)> {
        let version = name.strip_prefix("libnvrtc-builtins.so.")?;
        let mut parts = version.split('.');
        Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
    }

    /// Whether `name` is the NVRTC library (not its builtins library): `nvrtc64_130_0.dll`.
    #[cfg(windows)]
    pub(super) fn is_compiler_file(name: &str) -> bool {
        name.starts_with("nvrtc64_") && name.ends_with(".dll")
    }

    /// The `major.minor` of a builtins library file name, digits of `major * 10 + minor`: `nvrtc-builtins64_131.dll`.
    #[cfg(windows)]
    pub(super) fn builtins_version(name: &str) -> Option<(i32, i32)> {
        let digits = name.strip_prefix("nvrtc-builtins64_")?.strip_suffix(".dll")?;
        let number: i32 = digits.parse().ok()?;
        Some((number / 10, number % 10))
    }
}

/// Compiles every module the catalog names, loads it, and resolves every catalog
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
        let compiled = modules
            .iter()
            .find(|(module, _)| *module == spec.module)
            .map(|(_, compiled)| compiled);
        let slot = match compiled {
            // The native library sets each kernel's dynamic shared maximum to the catalog's figure, which the plans' launches
            // need; RustEE raises the driver's limit to it (0 leaves the default).
            Some(Compiled::Module(module)) => match module.kernel_with(
                spec.name,
                KernelOptions {
                    max_dynamic_shared_bytes: spec.shared_opt_in_bytes,
                },
            ) {
                Ok(kernel) => {
                    debug_assert_eq!(kernel.name(), spec.name);
                    Slot::Ready(Resolved {
                        kernel,
                        gpu: Arc::clone(gpu),
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
    // The architecture-specific variant exists for compute capability 12.x (block-scaled FP4 `mma.sync`), as in the
    // native loader; on another device the module stays unavailable, as there.
    if module.architecture_specific() && gpu.arch().0 != 12 {
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
        assert!(!options.arch_specific);
    }

    #[test]
    fn the_architecture_specific_modules_compile_for_the_architecture_specific_target() {
        // The native loader gives these `--gpu-architecture=sm_XYa` in the same place of the same options.
        for module in [Module::Nvfp4Native, Module::TrellisFp8] {
            let options = sources().options(module);
            assert!(options.arch_specific, "{}", module.root());
            assert_eq!(options.options, ["--std=c++14"]);
            assert_eq!(options.include_paths.len(), 3);
        }
        assert!(
            KERNELS
                .iter()
                .all(|spec| sources().options(spec.module).arch_specific == spec.module.architecture_specific())
        );
    }

    #[test]
    fn the_compiler_files_are_told_from_the_builtins_library() {
        #[cfg(not(windows))]
        {
            assert!(platform::is_compiler_file("libnvrtc.so.13.1"));
            assert!(!platform::is_compiler_file("libnvrtc-builtins.so.13.1"));
            assert_eq!(platform::builtins_version("libnvrtc-builtins.so.13.1"), Some((13, 1)));
            assert_eq!(platform::builtins_version("libnvrtc.so.13.1"), None);
        }
        #[cfg(windows)]
        {
            assert!(platform::is_compiler_file("nvrtc64_130_0.dll"));
            assert!(!platform::is_compiler_file("nvrtc-builtins64_131.dll"));
            assert_eq!(platform::builtins_version("nvrtc-builtins64_131.dll"), Some((13, 1)));
            assert_eq!(platform::builtins_version("nvrtc64_130_0.dll"), None);
        }
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
