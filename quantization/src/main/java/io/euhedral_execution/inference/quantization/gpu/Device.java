package io.euhedral_execution.inference.quantization.gpu;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

/// The engine's GPU memory and operations, the module's CUDA libraries, and one stream they all submit to: engine
/// operations see it selected on the calling thread (`euhedral_cuda_stream_select`), cuBLAS, cuSOLVER and the module's
/// kernels take its handle.
public final class Device implements AutoCloseable {

    public final CudaGpuMemory gpu;
    public final Cuda cuda;
    public final long stream;
    public final long cublas;
    private final MethodHandle create, select, clear, destroy;
    /// Each worker thread's own stream (lattice frames that run concurrently), destroyed with the device.
    private final java.util.Map<Thread, Long> threadStreams = new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.List<Long> extraStreams = new java.util.concurrent.CopyOnWriteArrayList<>();

    public Device(Path library) {
        this.gpu = new CudaGpuMemory(library);
        this.cuda = new Cuda();
        Arena global = Arena.global();
        SymbolLookup engine = SymbolLookup.libraryLookup(library, global);
        Linker linker = Linker.nativeLinker();
        create = linker.downcallHandle(
                engine.find("euhedral_cuda_stream_create").orElseThrow(), FunctionDescriptor.of(ValueLayout.JAVA_LONG));
        select = linker.downcallHandle(
                engine.find("euhedral_cuda_stream_select").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
        clear = linker.downcallHandle(
                engine.find("euhedral_cuda_stream_clear").orElseThrow(), FunctionDescriptor.ofVoid());
        destroy = linker.downcallHandle(
                engine.find("euhedral_cuda_stream_destroy").orElseThrow(),
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
        try {
            this.stream = (long) create.invokeExact();
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
        if (stream == 0) throw new IllegalStateException("could not create a CUDA stream");
        this.cublas = cuda.cublasHandle(stream);
        bindThread();
    }

    /// Binds the calling thread: the primary context for the module's driver calls, and the stream for the engine's
    /// operations.
    public void bindThread() {
        cuda.bindThread();
        try {
            clear.invokeExact(); // a thread selects one stream at a time
            int status = (int) select.invokeExact(stream);
            if (status != 0) throw new IllegalStateException("euhedral_cuda_stream_select failed: " + status);
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
    }

    /// Binds the calling thread to a stream of its own (created on first use) and returns it: for work that runs
    /// concurrently with the main stream, such as rounding objects on lattice workers.
    public long bindOwnStream() {
        long own = threadStreams.computeIfAbsent(Thread.currentThread(), thread -> {
            try {
                long created = (long) create.invokeExact();
                if (created == 0) throw new IllegalStateException("could not create a CUDA stream");
                return created;
            } catch (Throwable failure) {
                throw new IllegalStateException(failure);
            }
        });
        cuda.bindThread();
        try {
            clear.invokeExact(); // a thread selects one stream at a time
            int status = (int) select.invokeExact(own);
            if (status != 0) throw new IllegalStateException("euhedral_cuda_stream_select failed: " + status);
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
        return own;
    }

    /// A new stream (destroyed with the device).
    public long newStream() {
        try {
            long created = (long) create.invokeExact();
            if (created == 0) throw new IllegalStateException("could not create a CUDA stream");
            extraStreams.add(created);
            return created;
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
    }

    /// Binds the calling thread to `stream`: the primary context, and the stream for the engine's operations.
    public void select(long stream) {
        cuda.bindThread();
        try {
            clear.invokeExact();
            int status = (int) select.invokeExact(stream);
            if (status != 0) throw new IllegalStateException("euhedral_cuda_stream_select failed: " + status);
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        }
    }

    public void synchronize() {
        cuda.synchronize(stream);
    }

    /// y [rows, n] (BF16) = x [rows, k] (BF16) * w [n, k]^T with FP32 accumulation, rounded once.
    public void linear(long x, long w, long y, int rows, int k, int n) {
        cuda.gemmBf16(cublas, Cuda.CUBLAS_OP_T, Cuda.CUBLAS_OP_N, n, rows, k, 1.0f, w, k, x, k, 0.0f, y, n, true);
    }

    /// Column block of the symmetric Hessian accumulation.
    private static final int HESSIAN_BLOCK = 1024;

    /// The upper blocks of hessian [k, k] (FP32, column-major) += x^T x for x [rows, k] (BF16), FP32 accumulation:
    /// the product is symmetric, so only blocks on or above the diagonal are computed (about half the work); call
    /// [#symmetrize] once the accumulation is complete.
    public void accumulateHessian(long hessian, long x, int rows, int k) {
        for (int i = 0; i < k; i += HESSIAN_BLOCK) {
            int height = Math.min(HESSIAN_BLOCK, k - i);
            for (int j = i; j < k; j += HESSIAN_BLOCK) {
                int width = Math.min(HESSIAN_BLOCK, k - j);
                cuda.gemmBf16(
                        cublas,
                        Cuda.CUBLAS_OP_N,
                        Cuda.CUBLAS_OP_T,
                        height,
                        width,
                        rows,
                        1.0f,
                        x + 2L * i,
                        k,
                        x + 2L * j,
                        k,
                        1.0f,
                        hessian + 4L * ((long) j * k + i),
                        k);
            }
        }
    }

    /// Completes a Hessian accumulated by [#accumulateHessian]: its lower triangle from its upper one.
    public void symmetrize(long hessian, int k) {
        if (mirrorUpper == null) {
            mirrorUpper = cuda.compileResource("gptq.cu", java.util.List.of("mirror_upper"))
                    .getFirst();
        }
        long count = (long) k * k;
        cuda.launch(
                mirrorUpper,
                (int) ((count + 255) / 256),
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(hessian).integer(k));
    }

    private volatile Cuda.Function mirrorUpper;

    @Override
    public void close() {
        try {
            synchronize();
            clear.invokeExact();
            int ignored = (int) destroy.invokeExact(stream);
            for (long extra : extraStreams) {
                cuda.synchronize(extra);
                ignored = (int) destroy.invokeExact(extra);
            }
            for (long own : threadStreams.values()) {
                cuda.synchronize(own);
                ignored = (int) destroy.invokeExact(own);
            }
        } catch (Throwable failure) {
            throw new IllegalStateException(failure);
        } finally {
            gpu.close();
        }
    }
}
