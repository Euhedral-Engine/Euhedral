//! Loading the CUDA library and binding the device.

use std::array;
use std::ffi::{CStr, CString, c_int};
use std::fmt;
use std::path::{self, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_graph::LaneId;
use libloading::Library;

use crate::device::DeviceInfo;
use crate::error::{CudaError, check};
use crate::lane::EventRing;
use crate::memory::AllocationId;
use crate::sys::EuhedralCuda;

/// The most lanes a `LanePool` holds.
pub(crate) const MAX_LANES: usize = 64;

/// Runtime libraries beside the CUDA library that load first, by full path. NVRTC opens its builtins library by name,
/// which resolves to the copy already loaded; Windows also resolves a DLL's imports without the library's own
/// directory, so its runtime and NVRTC load first too. Linux finds them through the library's `$ORIGIN`.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const RUNTIME: &[&str] = &["libnvrtc-builtins.so.13.1"];
#[cfg(windows)]
const RUNTIME: &[&str] = &["cudart64_13.dll", "nvrtc-builtins64_131.dll", "nvrtc64_130_0.dll"];
#[cfg(not(any(all(target_os = "linux", target_arch = "x86_64"), windows)))]
const RUNTIME: &[&str] = &[];

/// The loaded CUDA library: its entry points, the device it bound, and the event rings of the lanes created on it.
///
/// One per process in practice; a lane, a buffer or a graph keeps it alive through an `Arc`.
pub struct CudaLibrary {
    // `api` holds the library handle and drops before the runtime libraries it links.
    api: EuhedralCuda,
    _runtime: Vec<Library>,
    path: PathBuf,
    device: DeviceInfo,
    allocations: AtomicU64,
    rings: [OnceLock<EventRing>; MAX_LANES],
}

impl fmt::Debug for CudaLibrary {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaLibrary")
            .field("path", &self.path)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl CudaLibrary {
    /// Loads the library at `library` (`lib/libeuhedral_cuda.so` or `lib/euhedral_cuda.dll` of a native product).
    ///
    /// The product directory is the library's grandparent; its `cuda/include` holds the headers the kernels compile
    /// against. Loading preloads the runtime libraries beside the library, sets that include directory and reads the
    /// device, which binds it.
    pub fn load(library: &Path) -> Result<Arc<CudaLibrary>, CudaError> {
        let library = path::absolute(library).map_err(|_| CudaError::LibraryMissing(library.to_path_buf()))?;
        if !library.is_file() {
            return Err(CudaError::LibraryMissing(library));
        }
        let directory = library
            .parent()
            .ok_or_else(|| CudaError::LibraryMissing(library.clone()))?;
        let runtime = preload_runtime(&library, directory)?;
        // SAFETY: loading a shared library runs its initializers; this is the Euhedral CUDA library, which has none
        // that depend on this process's state, and its runtime libraries are already loaded.
        let api = unsafe { EuhedralCuda::new(&library) }.map_err(|source| CudaError::Load {
            path: library.clone(),
            source,
        })?;
        let headers = directory
            .parent()
            .map(|product| product.join("cuda").join("include"))
            .ok_or_else(|| CudaError::LibraryMissing(library.clone()))?;
        set_include_directory(&api, &headers)?;
        let device = read_device(&api)?;
        Ok(Arc::new(CudaLibrary {
            api,
            _runtime: runtime,
            path: library,
            device,
            allocations: AtomicU64::new(0),
            rings: array::from_fn(|_| OnceLock::new()),
        }))
    }

    /// The path this library was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The device the library bound.
    pub fn device(&self) -> DeviceInfo {
        self.device
    }

    /// Free device memory in bytes, now.
    pub fn free_bytes(&self) -> Result<u64, CudaError> {
        let (mut free, mut total) = (0u64, 0u64);
        // SAFETY: both pointers refer to live locals.
        check("euhedral_cuda_device_memory_info", unsafe {
            self.api.euhedral_cuda_device_memory_info(&mut free, &mut total)
        })?;
        Ok(free)
    }

    /// Whether the native Blackwell NVFP4 route runs on this device.
    pub fn nvfp4_native_available(&self) -> bool {
        // SAFETY: the call takes no arguments.
        unsafe { self.api.euhedral_cuda_nvfp4_native_available() != 0 }
    }

    /// Scratch bytes one native NVFP4 linear needs.
    pub fn nvfp4_native_scratch_bytes(&self, rows: u32, in_features: u32, out_features: u32) -> u64 {
        // SAFETY: the call takes plain integers.
        unsafe {
            self.api
                .euhedral_cuda_nvfp4_native_scratch_bytes(rows, in_features, out_features)
        }
    }

    /// How many kernels the shared linear table holds.
    pub fn linear_kernel_count(&self) -> usize {
        // SAFETY: the call takes no arguments.
        let count = unsafe { self.api.euhedral_cuda_linear_kernel_count() };
        usize::try_from(count).unwrap_or(0)
    }

    /// The name of the shared linear table's kernel `index`, or `None` outside the table.
    pub fn linear_kernel_name(&self, index: usize) -> Option<String> {
        let index = c_int::try_from(index).ok()?;
        // SAFETY: the call takes an integer; the library returns null or a NUL-terminated static name.
        let name = unsafe { self.api.euhedral_cuda_linear_kernel_name(index) };
        if name.is_null() {
            return None;
        }
        // SAFETY: `name` is non-null and NUL-terminated, and the table's names live as long as the library.
        Some(unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned())
    }

    /// Blocks until all device work finishes. For startup, shutdown and tests; nothing on a hot path waits.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, waiting driver call: the device synchronizes only at startup, shutdown and in tests"
    )]
    pub fn synchronize(&self) -> Result<(), CudaError> {
        // SAFETY: the call takes no arguments.
        check("euhedral_cuda_synchronize", unsafe {
            self.api.euhedral_cuda_synchronize()
        })
    }

    /// Selects exact numerics (every quantized linear on its scalar reference, every relaxed-order operator on its
    /// exact twin) for later launches in this process, and returns the previous selection. Process-wide.
    #[cfg(feature = "test-oracles")]
    pub fn exact_numerics_for_tests(&self, exact: bool) -> bool {
        // SAFETY: the call takes an integer.
        unsafe { self.api.euhedral_cuda_select_exact_numerics(c_int::from(exact)) != 0 }
    }

    pub(crate) fn api(&self) -> &EuhedralCuda {
        &self.api
    }

    /// The next allocation serial. A serial is never reused, so a buffer read by a captured graph is told apart from
    /// a later allocation at the same address.
    pub(crate) fn next_allocation(&self) -> AllocationId {
        AllocationId(self.allocations.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// Registers the event ring of lane `id`; each id registers once.
    pub(crate) fn register_ring(&self, id: LaneId, ring: EventRing) -> Result<(), (CudaError, EventRing)> {
        let Some(slot) = self.rings.get(usize::from(id.0)) else {
            return Err((CudaError::LaneOutOfRange(id), ring));
        };
        slot.set(ring).map_err(|ring| (CudaError::LaneInUse(id), ring))
    }

    /// The event ring of lane `id`.
    pub(crate) fn ring(&self, id: LaneId) -> Result<&EventRing, CudaError> {
        self.rings
            .get(usize::from(id.0))
            .and_then(OnceLock::get)
            .ok_or(CudaError::UnknownLane(id))
    }
}

/// Loads the runtime libraries beside `library`, failing with the missing ones listed.
fn preload_runtime(library: &Path, directory: &Path) -> Result<Vec<Library>, CudaError> {
    let paths: Vec<PathBuf> = RUNTIME.iter().map(|name| directory.join(name)).collect();
    let missing: Vec<PathBuf> = paths.iter().filter(|path| !path.is_file()).cloned().collect();
    if !missing.is_empty() {
        return Err(CudaError::RuntimeMissing {
            library: library.to_path_buf(),
            missing,
        });
    }
    paths
        .into_iter()
        .map(|path| {
            // SAFETY: a CUDA runtime library's initializers do not depend on this process's state.
            unsafe { Library::new(&path) }.map_err(|source| CudaError::Load { path, source })
        })
        .collect()
}

fn set_include_directory(api: &EuhedralCuda, headers: &Path) -> Result<(), CudaError> {
    let invalid = || CudaError::IncludeDirectory(headers.to_path_buf());
    let text = CString::new(headers.to_str().ok_or_else(invalid)?).map_err(|_| invalid())?;
    // SAFETY: `text` is a NUL-terminated string that outlives the call, which copies it.
    let status = unsafe { api.euhedral_cuda_set_include_directory(text.as_ptr()) };
    if status == 0 { Ok(()) } else { Err(invalid()) }
}

/// Reads the device; the first runtime call also creates its context.
fn read_device(api: &EuhedralCuda) -> Result<DeviceInfo, CudaError> {
    let (mut sm_count, mut major, mut minor) = (0i32, 0i32, 0i32);
    let (mut free, mut total) = (0u64, 0u64);
    // SAFETY: every pointer refers to a live local.
    unsafe {
        check(
            "euhedral_cuda_multiprocessor_count",
            api.euhedral_cuda_multiprocessor_count(&mut sm_count),
        )?;
        check(
            "euhedral_cuda_compute_capability",
            api.euhedral_cuda_compute_capability(&mut major, &mut minor),
        )?;
        check(
            "euhedral_cuda_device_memory_info",
            api.euhedral_cuda_device_memory_info(&mut free, &mut total),
        )?;
    }
    Ok(DeviceInfo {
        sm_count: u32::try_from(sm_count).unwrap_or(0),
        compute_capability: (u32::try_from(major).unwrap_or(0), u32::try_from(minor).unwrap_or(0)),
        total_bytes: total,
    })
}

#[cfg(test)]
mod tests {
    use std::fs::{self, File};

    use super::*;

    #[test]
    fn missing_library_reports_path() {
        let path = Path::new("no-such-product").join("lib").join("libeuhedral_cuda.so");
        let error = CudaLibrary::load(&path).unwrap_err();
        assert!(matches!(error, CudaError::LibraryMissing(_)), "{error:?}");
        let message = error.to_string();
        assert!(message.contains("no-such-product"), "{message}");
        assert!(message.contains("libeuhedral_cuda.so"), "{message}");
    }

    #[test]
    fn a_library_without_its_runtime_beside_it_names_what_is_missing() {
        let product = tempfile::tempdir().unwrap();
        let lib = product.path().join("lib");
        fs::create_dir(&lib).unwrap();
        let library = lib.join("libeuhedral_cuda.so");
        File::create(&library).unwrap();
        let error = CudaLibrary::load(&library).unwrap_err();
        let message = error.to_string();
        if RUNTIME.is_empty() {
            assert!(matches!(error, CudaError::Load { .. }), "{message}");
        } else {
            assert!(matches!(error, CudaError::RuntimeMissing { .. }), "{message}");
            assert!(message.contains(RUNTIME[0]), "{message}");
        }
    }
}
