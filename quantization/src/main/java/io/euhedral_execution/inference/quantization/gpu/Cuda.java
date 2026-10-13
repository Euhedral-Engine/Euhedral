package io.euhedral_execution.inference.quantization.gpu;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/// The CUDA driver API, NVRTC, cuBLAS and cuSOLVER for the module's own kernels and library calls, in the same
/// primary context as the engine's library (which uses the runtime API on device 0). Streams are the engine's
/// `cudaStream_t` handles (`euhedral_cuda_stream_create`), so module work and engine operations order on one stream.
public final class Cuda {

    private static final Linker LINKER = Linker.nativeLinker();
    private static final ValueLayout.OfLong PTR = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT;

    public static final int CUBLAS_OP_N = 0, CUBLAS_OP_T = 1;
    public static final int CUDA_R_32F = 0, CUDA_R_16BF = 14;
    public static final int CUBLAS_COMPUTE_32F = 68, CUBLAS_COMPUTE_32F_PEDANTIC = 69;
    public static final int CUBLAS_GEMM_DEFAULT = -1;
    public static final int FILL_LOWER = 0, FILL_UPPER = 1;

    private final Arena arena = Arena.ofShared();
    private final MethodHandle cuInit, cuDeviceGet, cuDeviceGetAttribute, cuDevicePrimaryCtxRetain, cuCtxSetCurrent;
    private final MethodHandle cuModuleLoadData, cuModuleGetFunction, cuLaunchKernel, cuStreamSynchronize;
    private final MethodHandle cuEventCreate, cuEventRecord, cuEventQuery, cuStreamWaitEvent, cuEventDestroy;
    private final MethodHandle nvrtcCreateProgram, nvrtcCompileProgram, nvrtcGetProgramLogSize, nvrtcGetProgramLog;
    private final MethodHandle nvrtcGetCUBINSize, nvrtcGetCUBIN, nvrtcDestroyProgram;
    private final MethodHandle cublasCreate, cublasSetStream, cublasGemmEx, cublasSgemm;
    private final MethodHandle cusolverDnCreate, cusolverDnSetStream, potrfBufferSize, potrf, potriBufferSize, potri;
    private final long context;
    private final String architecture;

    public Cuda() {
        SymbolLookup driver = SymbolLookup.libraryLookup("libcuda.so.1", arena);
        SymbolLookup nvrtc = SymbolLookup.libraryLookup("libnvrtc.so.13", arena);
        // cuBLAS and cuSOLVER come from the module's download (cudaMathLibraries), loaded by absolute path with their
        // dependencies first, so no library path needs them.
        SymbolLookup cublas = mathLibrary("libnvJitLink.so.13", "libcublasLt.so.13", "libcublas.so.13");
        SymbolLookup cusolver = mathLibrary("libcusolver.so.12");
        cuInit = bind(driver, "cuInit", INT, INT);
        cuDeviceGet = bind(driver, "cuDeviceGet", INT, PTR, INT);
        cuDeviceGetAttribute = bind(driver, "cuDeviceGetAttribute", INT, PTR, INT, INT);
        cuDevicePrimaryCtxRetain = bind(driver, "cuDevicePrimaryCtxRetain", INT, PTR, INT);
        cuCtxSetCurrent = bind(driver, "cuCtxSetCurrent", INT, PTR);
        cuModuleLoadData = bind(driver, "cuModuleLoadData", INT, PTR, PTR);
        cuModuleGetFunction = bind(driver, "cuModuleGetFunction", INT, PTR, PTR, PTR);
        cuLaunchKernel = bind(driver, "cuLaunchKernel", INT, PTR, INT, INT, INT, INT, INT, INT, INT, PTR, PTR, PTR);
        cuStreamSynchronize = bind(driver, "cuStreamSynchronize", INT, PTR);
        cuEventCreate = bind(driver, "cuEventCreate", INT, PTR, INT);
        cuEventRecord = bind(driver, "cuEventRecord", INT, PTR, PTR);
        cuEventQuery = bind(driver, "cuEventQuery", INT, PTR);
        cuStreamWaitEvent = bind(driver, "cuStreamWaitEvent", INT, PTR, PTR, INT);
        cuEventDestroy = bind(driver, "cuEventDestroy_v2", INT, PTR);
        nvrtcCreateProgram = bind(nvrtc, "nvrtcCreateProgram", INT, PTR, PTR, PTR, INT, PTR, PTR);
        nvrtcCompileProgram = bind(nvrtc, "nvrtcCompileProgram", INT, PTR, INT, PTR);
        nvrtcGetProgramLogSize = bind(nvrtc, "nvrtcGetProgramLogSize", INT, PTR, PTR);
        nvrtcGetProgramLog = bind(nvrtc, "nvrtcGetProgramLog", INT, PTR, PTR);
        nvrtcGetCUBINSize = bind(nvrtc, "nvrtcGetCUBINSize", INT, PTR, PTR);
        nvrtcGetCUBIN = bind(nvrtc, "nvrtcGetCUBIN", INT, PTR, PTR);
        nvrtcDestroyProgram = bind(nvrtc, "nvrtcDestroyProgram", INT, PTR);
        cublasCreate = bind(cublas, "cublasCreate_v2", INT, PTR);
        cublasSetStream = bind(cublas, "cublasSetStream_v2", INT, PTR, PTR);
        cublasGemmEx = bind(
                cublas,
                "cublasGemmEx",
                INT,
                PTR,
                INT,
                INT,
                INT,
                INT,
                INT,
                PTR,
                PTR,
                INT,
                INT,
                PTR,
                INT,
                INT,
                PTR,
                PTR,
                INT,
                INT,
                INT,
                INT);
        cublasSgemm = bind(
                cublas, "cublasSgemm_v2", INT, PTR, INT, INT, INT, INT, INT, PTR, PTR, INT, PTR, INT, PTR, PTR, INT);
        cusolverDnCreate = bind(cusolver, "cusolverDnCreate", INT, PTR);
        cusolverDnSetStream = bind(cusolver, "cusolverDnSetStream", INT, PTR, PTR);
        potrfBufferSize = bind(cusolver, "cusolverDnSpotrf_bufferSize", INT, PTR, INT, INT, PTR, INT, PTR);
        potrf = bind(cusolver, "cusolverDnSpotrf", INT, PTR, INT, INT, PTR, INT, PTR, INT, PTR);
        potriBufferSize = bind(cusolver, "cusolverDnSpotri_bufferSize", INT, PTR, INT, INT, PTR, INT, PTR);
        potri = bind(cusolver, "cusolverDnSpotri", INT, PTR, INT, INT, PTR, INT, PTR, INT, PTR);
        try (Arena local = Arena.ofConfined()) {
            check("cuInit", (int) cuInit.invokeExact(0));
            MemorySegment device = local.allocate(INT);
            check("cuDeviceGet", (int) cuDeviceGet.invokeExact(device.address(), 0));
            int ordinal = device.get(INT, 0);
            MemorySegment value = local.allocate(INT);
            check("cuDeviceGetAttribute", (int) cuDeviceGetAttribute.invokeExact(value.address(), 75, ordinal));
            int major = value.get(INT, 0);
            check("cuDeviceGetAttribute", (int) cuDeviceGetAttribute.invokeExact(value.address(), 76, ordinal));
            architecture = "sm_" + major + value.get(INT, 0);
            MemorySegment ctx = local.allocate(PTR);
            check("cuDevicePrimaryCtxRetain", (int) cuDevicePrimaryCtxRetain.invokeExact(ctx.address(), ordinal));
            context = ctx.get(PTR, 0);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
        bindThread();
    }

    private SymbolLookup mathLibrary(String... names) {
        String directory = System.getProperty("euhedral.quantization.cuda-libraries", "");
        SymbolLookup last = null;
        for (String name : names) {
            Path path = directory.isBlank() ? null : Path.of(directory, name);
            last = path != null && Files.isRegularFile(path)
                    ? SymbolLookup.libraryLookup(path, arena)
                    : SymbolLookup.libraryLookup(name, arena);
        }
        return last;
    }

    /// Makes the primary context current on the calling thread (each thread that calls the driver API needs it).
    public void bindThread() {
        try {
            check("cuCtxSetCurrent", (int) cuCtxSetCurrent.invokeExact(context));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    public String architecture() {
        return architecture;
    }

    /// A compiled kernel function.
    public record Function(long handle, String name) {}

    /// Compiles `source` with NVRTC for this device and returns its kernels by name. Headers resolve against
    /// EUHEDRAL_CUDA_INCLUDE_DIR (the CUDA headers the engine's build provides).
    public List<Function> compile(String sourceName, String source, List<String> functions) {
        String include = System.getenv("EUHEDRAL_CUDA_INCLUDE_DIR");
        try (Arena local = Arena.ofConfined()) {
            MemorySegment program = local.allocate(PTR);
            check("nvrtcCreateProgram", (int) nvrtcCreateProgram.invokeExact(
                    program.address(),
                    local.allocateFrom(source).address(),
                    local.allocateFrom(sourceName).address(),
                    0,
                    0L,
                    0L));
            long handle = program.get(PTR, 0);
            List<String> options = new ArrayList<>(List.of("--std=c++17", "--gpu-architecture=" + architecture));
            if (include != null && !include.isBlank()) options.add("-I" + include);
            MemorySegment optionArray = local.allocate(PTR, options.size());
            for (int i = 0; i < options.size(); i++)
                optionArray.setAtIndex(
                        PTR, i, local.allocateFrom(options.get(i)).address());
            int status = (int) nvrtcCompileProgram.invokeExact(handle, options.size(), optionArray.address());
            if (status != 0) {
                MemorySegment size = local.allocate(PTR);
                int ignored = (int) nvrtcGetProgramLogSize.invokeExact(handle, size.address());
                MemorySegment log = local.allocate(Math.max(1, size.get(PTR, 0)));
                ignored = (int) nvrtcGetProgramLog.invokeExact(handle, log.address());
                throw new IllegalStateException(sourceName + " does not compile:\n" + log.getString(0));
            }
            MemorySegment size = local.allocate(PTR);
            check("nvrtcGetCUBINSize", (int) nvrtcGetCUBINSize.invokeExact(handle, size.address()));
            MemorySegment cubin = local.allocate(size.get(PTR, 0));
            check("nvrtcGetCUBIN", (int) nvrtcGetCUBIN.invokeExact(handle, cubin.address()));
            MemorySegment programRef = local.allocate(PTR);
            programRef.set(PTR, 0, handle);
            check("nvrtcDestroyProgram", (int) nvrtcDestroyProgram.invokeExact(programRef.address()));
            MemorySegment module = local.allocate(PTR);
            check("cuModuleLoadData", (int) cuModuleLoadData.invokeExact(module.address(), cubin.address()));
            List<Function> result = new ArrayList<>();
            for (String name : functions) {
                MemorySegment function = local.allocate(PTR);
                check("cuModuleGetFunction " + name, (int) cuModuleGetFunction.invokeExact(
                        function.address(),
                        module.get(PTR, 0),
                        local.allocateFrom(name).address()));
                result.add(new Function(function.get(PTR, 0), name));
            }
            return result;
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    private final java.util.Map<String, List<Function>> compiled = new java.util.concurrent.ConcurrentHashMap<>();

    /// Compiles a kernel source shipped with the module (resource `/cuda/NAME`), once per name and function list.
    public List<Function> compileResource(String name, List<String> functions) {
        return compiled.computeIfAbsent(name + functions, key -> compileResourceNow(name, functions));
    }

    private List<Function> compileResourceNow(String name, List<String> functions) {
        try (var input = Cuda.class.getResourceAsStream("/cuda/" + name)) {
            if (input == null) throw new IllegalArgumentException("no kernel source " + name);
            return compile(name, new String(input.readAllBytes(), StandardCharsets.UTF_8), functions);
        } catch (java.io.IOException failure) {
            throw new java.io.UncheckedIOException(failure);
        }
    }

    /// Kernel arguments, each a 64-bit device address, a 32-bit int or a 32-bit float, in order.
    public static final class Arguments {
        private final List<Object> values = new ArrayList<>();

        public Arguments pointer(long address) {
            values.add(address);
            return this;
        }

        public Arguments integer(int value) {
            values.add(value);
            return this;
        }

        public Arguments real(float value) {
            values.add(value);
            return this;
        }
    }

    public void launch(
            Function function, int gridX, int gridY, int blockX, int shared, long stream, Arguments arguments) {
        try (Arena local = Arena.ofConfined()) {
            int n = arguments.values.size();
            MemorySegment pointers = local.allocate(PTR, Math.max(1, n));
            for (int i = 0; i < n; i++) {
                Object value = arguments.values.get(i);
                MemorySegment slot = local.allocate(8, 8);
                if (value instanceof Long address) slot.set(PTR, 0, address);
                else if (value instanceof Integer integer) slot.set(INT, 0, integer);
                else slot.set(ValueLayout.JAVA_FLOAT, 0, (Float) value);
                pointers.setAtIndex(PTR, i, slot.address());
            }
            check("cuLaunchKernel " + function.name(), (int) cuLaunchKernel.invokeExact(
                    function.handle(), gridX, gridY, 1, blockX, 1, 1, shared, stream, pointers.address(), 0L));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// A CUDA event without timing (CU_EVENT_DISABLE_TIMING).
    public long createEvent() {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment event = local.allocate(PTR);
            check("cuEventCreate", (int) cuEventCreate.invokeExact(event.address(), 2));
            return event.get(PTR, 0);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    public void recordEvent(long event, long stream) {
        try {
            check("cuEventRecord", (int) cuEventRecord.invokeExact(event, stream));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// Whether the work before the event's last record has completed; never waits.
    public boolean eventDone(long event) {
        try {
            int status = (int) cuEventQuery.invokeExact(event);
            if (status == 600) return false; // CUDA_ERROR_NOT_READY
            check("cuEventQuery", status);
            return true;
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// Orders work submitted to `stream` from now on behind the event's last record, on the device.
    public void waitEvent(long stream, long event) {
        try {
            check("cuStreamWaitEvent", (int) cuStreamWaitEvent.invokeExact(stream, event, 0));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    public void destroyEvent(long event) {
        try {
            check("cuEventDestroy", (int) cuEventDestroy.invokeExact(event));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    public void synchronize(long stream) {
        try {
            check("cuStreamSynchronize", (int) cuStreamSynchronize.invokeExact(stream));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    // ---- cuBLAS ----

    public long cublasHandle(long stream) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment handle = local.allocate(PTR);
            check("cublasCreate", (int) cublasCreate.invokeExact(handle.address()));
            check("cublasSetStream", (int) cublasSetStream.invokeExact(handle.get(PTR, 0), stream));
            return handle.get(PTR, 0);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// C (column-major m x n, FP32) = alpha * op(A) op(B) + beta * C with BF16 A and B, FP32 accumulation.
    public void gemmBf16(
            long handle,
            int transA,
            int transB,
            int m,
            int n,
            int k,
            float alpha,
            long a,
            int lda,
            long b,
            int ldb,
            float beta,
            long c,
            int ldc) {
        gemmBf16(handle, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, false);
    }

    /// As gemmBf16, with C BF16 (FP32 accumulation, rounded once) when `bf16Output`.
    public void gemmBf16(
            long handle,
            int transA,
            int transB,
            int m,
            int n,
            int k,
            float alpha,
            long a,
            int lda,
            long b,
            int ldb,
            float beta,
            long c,
            int ldc,
            boolean bf16Output) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment alphaValue = local.allocateFrom(ValueLayout.JAVA_FLOAT, alpha);
            MemorySegment betaValue = local.allocateFrom(ValueLayout.JAVA_FLOAT, beta);
            check("cublasGemmEx", (int) cublasGemmEx.invokeExact(
                    handle,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    alphaValue.address(),
                    a,
                    CUDA_R_16BF,
                    lda,
                    b,
                    CUDA_R_16BF,
                    ldb,
                    betaValue.address(),
                    c,
                    bf16Output ? CUDA_R_16BF : CUDA_R_32F,
                    ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// FP32 GEMM, column-major (cuBLAS's default math: no TF32).
    public void gemmFp32(
            long handle,
            int transA,
            int transB,
            int m,
            int n,
            int k,
            float alpha,
            long a,
            int lda,
            long b,
            int ldb,
            float beta,
            long c,
            int ldc) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment alphaValue = local.allocateFrom(ValueLayout.JAVA_FLOAT, alpha);
            MemorySegment betaValue = local.allocateFrom(ValueLayout.JAVA_FLOAT, beta);
            check("cublasSgemm", (int) cublasSgemm.invokeExact(
                    handle,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    alphaValue.address(),
                    a,
                    lda,
                    b,
                    ldb,
                    betaValue.address(),
                    c,
                    ldc));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    // ---- cuSOLVER ----

    public long cusolverHandle(long stream) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment handle = local.allocate(PTR);
            check("cusolverDnCreate", (int) cusolverDnCreate.invokeExact(handle.address()));
            check("cusolverDnSetStream", (int) cusolverDnSetStream.invokeExact(handle.get(PTR, 0), stream));
            return handle.get(PTR, 0);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// Workspace floats potrf (or potri with `inverse`) needs for an n x n matrix.
    public int solverWorkspace(long handle, boolean inverse, int fill, int n, long a) {
        try (Arena local = Arena.ofConfined()) {
            MemorySegment size = local.allocate(INT);
            int status = inverse
                    ? (int) potriBufferSize.invokeExact(handle, fill, n, a, n, size.address())
                    : (int) potrfBufferSize.invokeExact(handle, fill, n, a, n, size.address());
            check(inverse ? "cusolverDnSpotri_bufferSize" : "cusolverDnSpotrf_bufferSize", status);
            return size.get(INT, 0);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /// Cholesky factor (or, with `inverse`, the inverse from a factor) in place; `info` is a device int.
    public void cholesky(
            long handle, boolean inverse, int fill, int n, long a, long workspace, int workspaceFloats, long info) {
        try {
            int status = inverse
                    ? (int) potri.invokeExact(handle, fill, n, a, n, workspace, workspaceFloats, info)
                    : (int) potrf.invokeExact(handle, fill, n, a, n, workspace, workspaceFloats, info);
            check(inverse ? "cusolverDnSpotri" : "cusolverDnSpotrf", status);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    private static MethodHandle bind(SymbolLookup library, String name, ValueLayout result, ValueLayout... arguments) {
        MemorySegment symbol =
                library.find(name).orElseThrow(() -> new IllegalStateException("missing symbol " + name));
        return LINKER.downcallHandle(symbol, FunctionDescriptor.of(result, arguments));
    }

    private static void check(String call, int status) {
        if (status != 0) throw new IllegalStateException(call + " failed with status " + status);
    }

    private static RuntimeException rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) return runtime;
        if (failure instanceof Error error) throw error;
        return new IllegalStateException(failure);
    }
}
