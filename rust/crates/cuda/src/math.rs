//! cuBLAS and cuSOLVER, loaded from a directory of NVIDIA's math libraries, for the calls a lane makes on the stream it
//! lends.
//!
//! [`MathLibraries::load`] opens the libraries once, in the order their dependencies need: cuSPARSE and nvJitLink,
//! then cuBLAS (with cuBLASLt), then cuSOLVER. Each lane owns one [`LaneHandles`], because a handle is not shareable
//! across streams in flight. Every call selects the stream the lane lends on its handle and enqueues on it; nothing
//! here waits. The results (the `info` word of a factorisation, for one) are read after the lane's queue has retired
//! the work, the way the crate reads any other result.
//!
//! Library calls on a lent stream cannot be recorded into a graph, so these are for work that runs once per
//! submission.

use std::ffi::{c_int, c_void};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::mem::replace;
use std::path::Path;
use std::ptr::{from_ref, null_mut, with_exposed_provenance_mut};
use std::sync::Arc;

use euhedral_gpu::RawStream;
use libloading::Library;

use crate::error::CudaError;
use crate::memory::DevicePtr;

/// A cuBLAS operation applied to a matrix operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// `CUBLAS_OP_N`.
    None = 0,
    /// `CUBLAS_OP_T`.
    Transpose = 1,
}

/// Which triangle of a symmetric matrix a factorisation uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// `CUBLAS_FILL_MODE_LOWER`.
    Lower = 0,
    /// `CUBLAS_FILL_MODE_UPPER`.
    Upper = 1,
}

/// The element type of the product a BF16 GEMM writes. Accumulation is FP32 either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemmOutput {
    /// `CUDA_R_32F`.
    Fp32,
    /// `CUDA_R_16BF`: rounded once from the FP32 accumulator.
    Bf16,
}

/// How cuBLAS multiplies FP32 operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathMode {
    /// `CUBLAS_DEFAULT_MATH`: full FP32.
    Default = 0,
    /// `CUBLAS_TF32_TENSOR_OP_MATH`: inputs rounded to TF32 on the tensor cores.
    Tf32 = 3,
}

const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

type Handle = *mut c_void;
type Stream = *mut c_void;

type CreateFn = unsafe extern "C" fn(*mut Handle) -> c_int;
type DestroyFn = unsafe extern "C" fn(Handle) -> c_int;
type SetStreamFn = unsafe extern "C" fn(Handle, Stream) -> c_int;
type SetMathModeFn = unsafe extern "C" fn(Handle, c_int) -> c_int;
type GemmExFn = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const c_void,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    c_int,
    c_int,
    *const c_void,
    *mut c_void,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type SgemmFn = unsafe extern "C" fn(
    Handle,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
    *const f32,
    *const f32,
    c_int,
    *const f32,
    c_int,
    *const f32,
    *mut f32,
    c_int,
) -> c_int;
type BufferSizeFn = unsafe extern "C" fn(Handle, c_int, c_int, *mut f32, c_int, *mut c_int) -> c_int;
type FactorFn = unsafe extern "C" fn(Handle, c_int, c_int, *mut f32, c_int, *mut f32, c_int, *mut c_int) -> c_int;
type SolveFn =
    unsafe extern "C" fn(Handle, c_int, c_int, c_int, *const f32, c_int, *mut f32, c_int, *mut c_int) -> c_int;

/// The loaded libraries and the entry points the lanes call.
pub struct MathLibraries {
    cublas_create: CreateFn,
    cublas_destroy: DestroyFn,
    cublas_set_stream: SetStreamFn,
    cublas_set_math_mode: SetMathModeFn,
    cublas_gemm_ex: GemmExFn,
    cublas_sgemm: SgemmFn,
    solver_create: CreateFn,
    solver_destroy: DestroyFn,
    solver_set_stream: SetStreamFn,
    potrf_buffer_size: BufferSizeFn,
    potrf: FactorFn,
    potri_buffer_size: BufferSizeFn,
    potri: FactorFn,
    potrs: SolveFn,
    // The entry points above point into these; they are declared last so they drop last.
    _libraries: Vec<Library>,
}

impl Debug for MathLibraries {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("MathLibraries").finish_non_exhaustive()
    }
}

/// The libraries of the CUDA 13 redistributables, in load order: each one's dependencies come before it, so the loader
/// finds them already open by name without a search path.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const LIBRARIES: &[&str] = &[
    "libcusparse.so.12",
    "libnvJitLink.so.13",
    "libcublasLt.so.13",
    "libcublas.so.13",
    "libcusolver.so.12",
];
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
const LIBRARIES: &[&str] = &[];

impl MathLibraries {
    /// Opens the libraries in `directory`, which holds the redistributable libraries cuBLAS, cuBLASLt, cuSOLVER,
    /// cuSPARSE and nvJitLink. The CUDA driver must already be usable by this process (load the [`CudaLibrary`](crate::CudaLibrary) first).
    pub fn load(directory: &Path) -> Result<Arc<MathLibraries>, CudaError> {
        if LIBRARIES.is_empty() {
            return Err(CudaError::Unsupported(
                "cuBLAS and cuSOLVER are loaded on Linux x86-64 only",
            ));
        }
        let mut libraries = Vec::with_capacity(LIBRARIES.len());
        let mut missing = Vec::new();
        for name in LIBRARIES {
            let path = directory.join(name);
            if !path.is_file() {
                missing.push(path);
                continue;
            }
            // SAFETY: loading a shared library runs its initializers; these are NVIDIA's math libraries, whose
            // initializers do not depend on this process's state, and their dependencies are already open.
            let library = unsafe { Library::new(&path) }.map_err(|source| CudaError::Load {
                path: path.clone(),
                source,
            })?;
            libraries.push(library);
        }
        if !missing.is_empty() {
            return Err(CudaError::MathLibrariesMissing {
                directory: directory.to_path_buf(),
                missing,
            });
        }
        let cublas = &libraries[3];
        let solver = &libraries[4];
        Ok(Arc::new(MathLibraries {
            cublas_create: symbol(cublas, "cublasCreate_v2")?,
            cublas_destroy: symbol(cublas, "cublasDestroy_v2")?,
            cublas_set_stream: symbol(cublas, "cublasSetStream_v2")?,
            cublas_set_math_mode: symbol(cublas, "cublasSetMathMode")?,
            cublas_gemm_ex: symbol(cublas, "cublasGemmEx")?,
            cublas_sgemm: symbol(cublas, "cublasSgemm_v2")?,
            solver_create: symbol(solver, "cusolverDnCreate")?,
            solver_destroy: symbol(solver, "cusolverDnDestroy")?,
            solver_set_stream: symbol(solver, "cusolverDnSetStream")?,
            potrf_buffer_size: symbol(solver, "cusolverDnSpotrf_bufferSize")?,
            potrf: symbol(solver, "cusolverDnSpotrf")?,
            potri_buffer_size: symbol(solver, "cusolverDnSpotri_bufferSize")?,
            potri: symbol(solver, "cusolverDnSpotri")?,
            potrs: symbol(solver, "cusolverDnSpotrs")?,
            _libraries: libraries,
        }))
    }

    /// Creates the cuBLAS and cuSOLVER handles of one lane.
    ///
    /// Call it from the lane's own submission, where the device's context is current, and keep the result with the
    /// lane's state. When the second handle cannot be made, the first comes back with the error in [`HandlesFailed`],
    /// so no submission drops it (destroying a handle can wait for the device).
    pub fn lane_handles(self: &Arc<Self>) -> Result<LaneHandles, HandlesFailed> {
        let mut blas = null_mut();
        // SAFETY: the pointer refers to a live local the call writes a handle to.
        let created = status("cublasCreate_v2", unsafe { (self.cublas_create)(&mut blas) });
        if let Err(error) = created {
            return Err(HandlesFailed { error, leftover: None });
        }
        let mut handles = LaneHandles {
            libraries: Arc::clone(self),
            blas,
            solver: null_mut(),
        };
        // SAFETY: as above.
        let created = status("cusolverDnCreate", unsafe { (self.solver_create)(&mut handles.solver) });
        match created {
            Ok(()) => Ok(handles),
            Err(error) => Err(HandlesFailed {
                error,
                leftover: Some(handles),
            }),
        }
    }
}

/// Why [`MathLibraries::lane_handles`] failed, with the handle it had already made, if any. Close the leftover at
/// shutdown with [`LaneHandles::close`].
#[derive(Debug)]
pub struct HandlesFailed {
    /// The failed call.
    pub error: CudaError,
    /// The cuBLAS handle made before the failure, still to be closed.
    pub leftover: Option<LaneHandles>,
}

/// The entry point `name` of `library`.
fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T, CudaError> {
    // SAFETY: `T` is the function pointer type of the entry point in the library's header, and the pointer is used only
    // while the library, which the caller keeps, stays loaded.
    let symbol = unsafe { library.get::<T>(name.as_bytes()) };
    symbol
        .map(|symbol| *symbol)
        .map_err(|_| CudaError::Argument(format!("the math library has no entry point {name}")))
}

/// `Ok` for status 0, else the failed call.
fn status(call: &'static str, status: c_int) -> Result<(), CudaError> {
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Library { call, status })
    }
}

/// Reads the `info` word a factorisation or solve wrote, after the queue has retired it: zero is success, a negative
/// value names the bad argument and a positive one is the order of the leading minor that is not positive definite.
pub fn check_info(call: &'static str, info: i32) -> Result<(), CudaError> {
    if info == 0 {
        Ok(())
    } else {
        Err(CudaError::Info { call, info })
    }
}

/// One lane's cuBLAS and cuSOLVER handles.
///
/// Used by one submission at a time (the lane's), but it may move between the threads that submit. Destroying the
/// handles can wait for the whole device: at shutdown call [`close`](Self::close), which reports a failure; a plain
/// drop does the same and ignores it. Never drop one on a running lane.
pub struct LaneHandles {
    libraries: Arc<MathLibraries>,
    blas: Handle,
    solver: Handle,
}

// SAFETY: a handle may be used from any thread as long as one thread uses it at a time, which `&mut self` on every call
// guarantees. The pointers are not shared. A handle is used with streams of one device context only, and with a stream
// of the lane it belongs to, or after the work enqueued on any other stream has retired (the `# Safety` of each call
// says so): the type does not check this.
unsafe impl Send for LaneHandles {}

impl Debug for LaneHandles {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("LaneHandles").finish_non_exhaustive()
    }
}

impl LaneHandles {
    /// Destroys the handles, in order, and reports the first failure. This waits for the device: for shutdown.
    pub fn close(mut self) -> Result<(), CudaError> {
        self.destroy()
    }

    fn destroy(&mut self) -> Result<(), CudaError> {
        let (blas, solver) = (
            replace(&mut self.blas, null_mut()),
            replace(&mut self.solver, null_mut()),
        );
        let mut first = Ok(());
        if !solver.is_null() {
            // SAFETY: the handle came from the create call and is destroyed once; the libraries stay loaded through
            // `self.libraries`.
            first = status("cusolverDnDestroy", unsafe { (self.libraries.solver_destroy)(solver) });
        }
        if !blas.is_null() {
            // SAFETY: as above.
            let second = status("cublasDestroy_v2", unsafe { (self.libraries.cublas_destroy)(blas) });
            first = first.and(second);
        }
        first
    }
}

impl Drop for LaneHandles {
    fn drop(&mut self) {
        // A failure at drop has nowhere to go; `close` reports it.
        drop(self.destroy());
    }
}

/// The address of device memory as the C ABI takes it. The pointer is never dereferenced on the host.
fn address<T>(pointer: DevicePtr) -> *mut T {
    with_exposed_provenance_mut(usize::try_from(pointer.get()).unwrap_or(0))
}

/// A matrix operand of a product: its device address and leading dimension.
#[derive(Clone, Copy, Debug)]
pub struct Matrix {
    /// The first element.
    pub pointer: DevicePtr,
    /// The leading dimension, in elements.
    pub leading: i32,
}

/// The shape of a product `C = alpha * op(A) op(B) + beta * C`, column-major, `C` being `m` by `n`.
#[derive(Clone, Copy, Debug)]
pub struct Product {
    /// The operation applied to `A`.
    pub a_operation: Operation,
    /// The operation applied to `B`.
    pub b_operation: Operation,
    /// Rows of `op(A)` and `C`.
    pub m: i32,
    /// Columns of `op(B)` and `C`.
    pub n: i32,
    /// The shared dimension.
    pub k: i32,
    /// The scale of the product.
    pub alpha: f32,
    /// The scale of the old `C`.
    pub beta: f32,
}

impl LaneHandles {
    fn select(&mut self, stream: &RawStream<'_, *mut c_void>) -> Result<(), CudaError> {
        let raw = stream.as_raw();
        // SAFETY: the handles are live, and the stream is lent for this submission; the selection is replaced on the
        // next lend.
        unsafe {
            status("cublasSetStream_v2", (self.libraries.cublas_set_stream)(self.blas, raw))?;
            status(
                "cusolverDnSetStream",
                (self.libraries.solver_set_stream)(self.solver, raw),
            )
        }
    }

    /// Chooses whether FP32 products use the TF32 tensor cores. The choice stays on this handle, so it applies to every
    /// later FP32 GEMM on this lane until it is changed.
    pub fn set_math_mode(&mut self, mode: MathMode) -> Result<(), CudaError> {
        // SAFETY: the handle is live.
        status("cublasSetMathMode", unsafe {
            (self.libraries.cublas_set_math_mode)(self.blas, mode as c_int)
        })
    }

    /// Enqueues `C = alpha * op(A) op(B) + beta * C` with BF16 `A` and `B` and FP32 accumulation on `stream`. `C` is
    /// FP32, or BF16 with `output`.
    ///
    /// # Safety
    /// Every stream passed must come from the lane this handle belongs to, or the work enqueued on any other stream
    /// must have retired first; the device context must be the same.
    /// The three operands must be live device allocations of the sizes `product` and the leading dimensions imply, and
    /// stay live and untouched by other work until the queue retires this one; `A` and `B` hold BF16, `C` the
    /// element type `output` names.
    pub unsafe fn gemm_bf16(
        &mut self,
        stream: &RawStream<'_, *mut c_void>,
        product: &Product,
        a: Matrix,
        b: Matrix,
        c: Matrix,
        output: GemmOutput,
    ) -> Result<(), CudaError> {
        self.select(stream)?;
        let output_type = match output {
            GemmOutput::Fp32 => CUDA_R_32F,
            GemmOutput::Bf16 => CUDA_R_16BF,
        };
        let (alpha, beta) = (product.alpha, product.beta);
        // SAFETY: alpha and beta are live locals, read before the call returns (host pointer mode); the operands are
        // the caller's contract.
        status("cublasGemmEx", unsafe {
            (self.libraries.cublas_gemm_ex)(
                self.blas,
                product.a_operation as c_int,
                product.b_operation as c_int,
                product.m,
                product.n,
                product.k,
                from_ref(&alpha).cast(),
                address::<c_void>(a.pointer),
                CUDA_R_16BF,
                a.leading,
                address::<c_void>(b.pointer),
                CUDA_R_16BF,
                b.leading,
                from_ref(&beta).cast(),
                address::<c_void>(c.pointer),
                output_type,
                c.leading,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            )
        })
    }

    /// Enqueues the FP32 product `C = alpha * op(A) op(B) + beta * C` on `stream`.
    ///
    /// # Safety
    /// Every stream passed must come from the lane this handle belongs to, or the work enqueued on any other stream
    /// must have retired first; the device context must be the same.
    /// As [`gemm_bf16`](Self::gemm_bf16), with all three operands FP32.
    pub unsafe fn gemm_fp32(
        &mut self,
        stream: &RawStream<'_, *mut c_void>,
        product: &Product,
        a: Matrix,
        b: Matrix,
        c: Matrix,
    ) -> Result<(), CudaError> {
        self.select(stream)?;
        let (alpha, beta) = (product.alpha, product.beta);
        // SAFETY: alpha and beta are live locals, read before the call returns; the operands are the caller's contract.
        status("cublasSgemm_v2", unsafe {
            (self.libraries.cublas_sgemm)(
                self.blas,
                product.a_operation as c_int,
                product.b_operation as c_int,
                product.m,
                product.n,
                product.k,
                &alpha,
                address::<f32>(a.pointer),
                a.leading,
                address::<f32>(b.pointer),
                b.leading,
                &beta,
                address::<f32>(c.pointer),
                c.leading,
            )
        })
    }

    /// The workspace, in floats, that [`cholesky`](Self::cholesky) (or, with `inverse`, the inverse from a factor) needs
    /// for an `n` by `n` matrix. The answer is on the host when the call returns.
    ///
    /// # Safety
    /// Every stream passed must come from the lane this handle belongs to, or the work enqueued on any other stream
    /// must have retired first; the device context must be the same.
    /// `matrix` must be a live device allocation of `n * n` floats (it is not read).
    pub unsafe fn solver_workspace(
        &mut self,
        stream: &RawStream<'_, *mut c_void>,
        inverse: bool,
        fill: Fill,
        n: i32,
        matrix: DevicePtr,
    ) -> Result<i32, CudaError> {
        self.select(stream)?;
        let mut floats: c_int = 0;
        let (call, function) = if inverse {
            ("cusolverDnSpotri_bufferSize", self.libraries.potri_buffer_size)
        } else {
            ("cusolverDnSpotrf_bufferSize", self.libraries.potrf_buffer_size)
        };
        // SAFETY: `floats` is a live local the call writes; the matrix is the caller's contract.
        status(call, unsafe {
            function(self.solver, fill as c_int, n, address::<f32>(matrix), n, &mut floats)
        })?;
        Ok(floats)
    }

    /// Enqueues the Cholesky factor of the `n` by `n` matrix in place on `stream` (`potrf`), or, with `inverse`, the
    /// inverse from a factor (`potri`). `info` is a device int the call writes; read it after the queue retires the
    /// work, with [`check_info`].
    ///
    /// # Safety
    /// Every stream passed must come from the lane this handle belongs to, or the work enqueued on any other stream
    /// must have retired first; the device context must be the same.
    /// `matrix` must be a live device allocation of `n * n` floats, `workspace` one of `workspace_floats` floats
    /// (at least [`solver_workspace`](Self::solver_workspace)) and `info` one of an int; all stay live and untouched by
    /// other work until the queue retires this one.
    #[allow(clippy::too_many_arguments, reason = "the library's own parameters")]
    pub unsafe fn cholesky(
        &mut self,
        stream: &RawStream<'_, *mut c_void>,
        inverse: bool,
        fill: Fill,
        n: i32,
        matrix: DevicePtr,
        workspace: DevicePtr,
        workspace_floats: i32,
        info: DevicePtr,
    ) -> Result<(), CudaError> {
        self.select(stream)?;
        let (call, function) = if inverse {
            ("cusolverDnSpotri", self.libraries.potri)
        } else {
            ("cusolverDnSpotrf", self.libraries.potrf)
        };
        // SAFETY: the buffers are the caller's contract.
        status(call, unsafe {
            function(
                self.solver,
                fill as c_int,
                n,
                address::<f32>(matrix),
                n,
                address::<f32>(workspace),
                workspace_floats,
                address::<c_int>(info),
            )
        })
    }

    /// Enqueues the solve of `A X = B` in place in `b` (`n` by `columns`, column-major, leading dimension `n`) from the
    /// Cholesky factor of `A` that [`cholesky`](Self::cholesky) left in `factor` (`potrs`). `info` is a device int, read
    /// after the queue retires the work.
    ///
    /// # Safety
    /// Every stream passed must come from the lane this handle belongs to, or the work enqueued on any other stream
    /// must have retired first; the device context must be the same.
    /// `factor` must be a live device allocation of `n * n` floats, `b` one of `n * columns` floats and `info` one of an
    /// int; all stay live and untouched by other work until the queue retires this one.
    #[allow(clippy::too_many_arguments, reason = "the library's own parameters")]
    pub unsafe fn solve_cholesky(
        &mut self,
        stream: &RawStream<'_, *mut c_void>,
        fill: Fill,
        n: i32,
        columns: i32,
        factor: DevicePtr,
        b: DevicePtr,
        info: DevicePtr,
    ) -> Result<(), CudaError> {
        self.select(stream)?;
        // SAFETY: the buffers are the caller's contract.
        status("cusolverDnSpotrs", unsafe {
            (self.libraries.potrs)(
                self.solver,
                fill as c_int,
                n,
                columns,
                address::<f32>(factor),
                n,
                address::<f32>(b),
                n,
                address::<c_int>(info),
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_directory_names_every_library() {
        if LIBRARIES.is_empty() {
            return;
        }
        let Err(CudaError::MathLibrariesMissing { missing, .. }) = MathLibraries::load(Path::new("no-such-directory"))
        else {
            panic!("an empty directory must be refused as missing libraries");
        };
        assert_eq!(missing.len(), LIBRARIES.len());
    }
}
