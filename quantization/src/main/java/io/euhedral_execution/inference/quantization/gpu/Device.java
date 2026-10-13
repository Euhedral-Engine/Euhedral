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

    public void synchronize() {
        cuda.synchronize(stream);
    }

    /// y [rows, n] (BF16) = x [rows, k] (BF16) * w [n, k]^T with FP32 accumulation, rounded once.
    public void linear(long x, long w, long y, int rows, int k, int n) {
        cuda.gemmBf16(cublas, Cuda.CUBLAS_OP_T, Cuda.CUBLAS_OP_N, n, rows, k, 1.0f, w, k, x, k, 0.0f, y, n, true);
    }

    /// hessian [k, k] (FP32) += x^T x for x [rows, k] (BF16), FP32 accumulation.
    public void accumulateHessian(long hessian, long x, int rows, int k) {
        cuda.gemmBf16(cublas, Cuda.CUBLAS_OP_N, Cuda.CUBLAS_OP_T, k, k, rows, 1.0f, x, k, x, k, 1.0f, hessian, k);
    }

    @Override
    public void close() {
        try {
            synchronize();
            clear.invokeExact();
            int ignored = (int) destroy.invokeExact(stream);
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
