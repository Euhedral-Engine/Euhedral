package io.euhedral_execution.inference.core.gpu;

import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.model_loader.artifact.P2e2Layout;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongConsumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// FFM binding for the stable Euhedral CUDA C ABI.
///
/// CUDA device addresses remain opaque longs. They are converted to zero-size address segments only
/// inside the native calls and are never exposed as dereferenceable Java memory.
public final class CudaGpuMemory extends ExecutionGpu implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(CudaGpuMemory.class);
    private static final int MAX_CACHED_EVENTS = 256;
    /// EUHEDRAL_CUDA_KERNEL_UNAVAILABLE: an optional kernel did not load.
    private static final int KERNEL_UNAVAILABLE = -4;
    /// EUHEDRAL_CUDA_ROUTE_UNAVAILABLE: a specialized route does not apply; another one must run.
    private static final int ROUTE_UNAVAILABLE = -5;
    /// Expansion scratch above which a P2E2 linear is computed in output-row chunks. The largest
    /// region that expands, the streamed FFN's gate/up plus down (109 MB), fits.
    private static final long Q3_SCRATCH_LIMIT = 128L << 20;
    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle hostMalloc;
    private final MethodHandle hostFree;
    private final MethodHandle hostWeightsMalloc;
    private final MethodHandle hostWeightsFree;
    private final java.util.concurrent.ConcurrentHashMap<Long, Long> hostWeights =
            new java.util.concurrent.ConcurrentHashMap<>();
    private final MethodHandle deviceMemoryInfo;
    private final MethodHandle copyHostToDevice;
    private final MethodHandle copyUploadToDevice;
    private final MethodHandle copyDeviceToHost;
    private final MethodHandle copyDeviceToReadback;
    private final MethodHandle copyDeviceToDevice;
    private final MethodHandle embedQ3;
    private final MethodHandle synchronize;
    private final MethodHandle streamCreate;
    private final MethodHandle streamDestroy;
    private final MethodHandle streamSynchronize;
    private final MethodHandle streamSelect;
    private final MethodHandle streamClear;
    private final MethodHandle pdlSelect;
    private final MethodHandle rowExactSelect;
    /// Mirrors the native thread-local row-exact selection; the native NVFP4 route is never row-exact.
    private static final ThreadLocal<boolean[]> ROW_EXACT = ThreadLocal.withInitial(() -> new boolean[1]);
    private final MethodHandle eventCreate;
    private final MethodHandle eventRecord;
    private final MethodHandle streamWaitEvent;
    private final MethodHandle eventQuery;
    private final MethodHandle eventDestroy;
    private final MethodHandle completionNotify;
    private final MemorySegment completionCallback;
    private final ConcurrentHashMap<Long, Completion> completions = new ConcurrentHashMap<>();
    private final AtomicLong nextCompletion = new AtomicLong();
    private final AtomicInteger openStreams = new AtomicInteger();
    private final AtomicReference<Throwable> poisoned = new AtomicReference<>();
    private final MpmcQueue<Long> availableEvents = new MpmcQueue<>(64, 4);
    private final ConcurrentLinkedQueue<MemorySegment> pinnedUploads = new ConcurrentLinkedQueue<>();
    private final AtomicInteger pinnedUploadCount = new AtomicInteger();
    /// The size of every live device allocation, by address; [#allocate] is the only native allocator.
    private final ConcurrentHashMap<Long, Long> allocationSizes = new ConcurrentHashMap<>();
    private final AtomicLong allocatedBytes = new AtomicLong();
    /// High-water mark of [#allocatedBytes] since construction or the last [#resetPeakAllocatedBytes].
    private final AtomicLong peakAllocatedBytes = new AtomicLong();
    private final MethodHandle rmsNormBf16;
    private final MethodHandle rmsNormUnitOffsetBf16;
    private final MethodHandle linearQ3Bf16;
    private final MethodHandle linearQ3DecodeBf16;
    private final MethodHandle linearQ3PrefillBf16;
    private final Q3DispatchMode q3DispatchMode;
    private final int q3SmallRowThreshold;
    private final MethodHandle linearQuantizedBf16;
    private final MethodHandle linearBf16ToFloat;
    private final MethodHandle gdnControlFp32;
    private final MethodHandle gdnConvolutionBf16;
    private final MethodHandle gdnRecurrenceBf16;
    private final MethodHandle gdnGatedRmsNormBf16;
    private final MethodHandle residualAddBf16;
    private final MethodHandle residualRmsNormBf16;
    private final MethodHandle q3GateUpSwiGluBf16;
    private final MethodHandle q3FfnDownBf16;
    private final MethodHandle q3FfnDownSplitBf16;
    private final MethodHandle selectExactNumerics;
    private final MethodHandle q3FfnStreamedBf16;
    private final MethodHandle gdnProjectControlFp32;
    private final MethodHandle gdnProjectionsBf16;
    private final MethodHandle swiGluBf16;
    private final MethodHandle argmaxBf16;
    private final MethodHandle zeroDeviceMemory;
    private final MethodHandle attentionQkNormRopeBf16;
    private final MethodHandle attentionKvAppendNvfp4;
    private final MethodHandle attentionCausalNvfp4;
    private final MethodHandle linearQ3P2e2DecodeBf16;
    private final MethodHandle q3P2e2Expand;
    private final MethodHandle embedQ3P2e2;
    private final MethodHandle copyDeviceToDevice2d;
    private final MethodHandle linearNvfp4Bf16;
    private final MethodHandle nvfp4GateUpSwiGluBf16;
    private final MethodHandle nvfp4NativeAvailable;
    private final MethodHandle nvfp4ActivationBytes;
    private final MethodHandle linearNvfp4NativeBf16;
    private final MethodHandle nvfp4NativeGateUpSwiGluBf16;
    /// Whether the native Blackwell NVFP4 module loaded (null until first asked).
    private volatile Boolean nvfp4Native;
    /// The stream whose launches the current thread is submitting, or null for synchronous calls.
    private final ThreadLocal<CudaStream> submitting = new ThreadLocal<>();
    /// One device region, reused by every P2E2 route that expands its tensor for a row-split kernel.
    /// Each use waits for the previous one through `q3ScratchEvent`, recorded on the stream that used it.
    private final ReentrantLock q3ScratchLock = new ReentrantLock();
    private long q3ScratchAddress;
    private long q3ScratchBytes;
    private long q3ScratchEvent;
    private volatile boolean closed;

    public CudaGpuMemory(Path libraryPath) {
        this(libraryPath, Q3DispatchMode.AUTO, Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD);
    }

    public CudaGpuMemory(Path libraryPath, Q3DispatchMode q3DispatchMode, int q3SmallRowThreshold) {
        Objects.requireNonNull(libraryPath, "libraryPath");
        this.q3DispatchMode = Objects.requireNonNull(q3DispatchMode, "q3DispatchMode");
        if (q3SmallRowThreshold < 0) throw new IllegalArgumentException("Q3 threshold must not be negative");
        this.q3SmallRowThreshold = q3SmallRowThreshold;
        Arena loadedLibraryArena = Arena.ofShared();
        try {
            SymbolLookup symbols = SymbolLookup.libraryLookup(libraryPath, loadedLibraryArena);
            Linker linker = Linker.nativeLinker();
            this.arena = loadedLibraryArena;
            this.malloc = bind(linker, symbols, "euhedral_cuda_malloc", MALLOC);
            this.free = bind(linker, symbols, "euhedral_cuda_free", FREE);
            this.hostMalloc = bind(linker, symbols, "euhedral_cuda_host_malloc", MALLOC);
            this.hostFree = bind(linker, symbols, "euhedral_cuda_host_free", FREE);
            this.hostWeightsMalloc = bind(linker, symbols, "euhedral_cuda_host_weights_malloc", MALLOC);
            this.hostWeightsFree = bind(linker, symbols, "euhedral_cuda_host_weights_free", FREE);
            this.deviceMemoryInfo = bind(linker, symbols, "euhedral_cuda_device_memory_info", DEVICE_MEMORY_INFO);
            this.copyHostToDevice = bind(linker, symbols, "euhedral_cuda_copy_host_to_device", COPY);
            this.copyUploadToDevice = bind(linker, symbols, "euhedral_cuda_copy_upload_to_device", COPY);
            this.copyDeviceToHost = bind(linker, symbols, "euhedral_cuda_copy_device_to_host", COPY);
            this.copyDeviceToReadback = bind(linker, symbols, "euhedral_cuda_copy_device_to_readback", COPY);
            this.copyDeviceToDevice = bind(linker, symbols, "euhedral_cuda_copy_device_to_device", COPY);
            this.embedQ3 = bind(linker, symbols, "euhedral_cuda_embed_q3", EMBED_Q3);
            this.embedQ3P2e2 = symbols.find("euhedral_cuda_embed_q3_p2e2")
                    .map(symbol -> linker.downcallHandle(symbol, EMBED_Q3))
                    .orElse(null);
            this.linearQ3P2e2DecodeBf16 = symbols.find("euhedral_cuda_linear_q3_p2e2_decode_bf16")
                    .map(symbol -> linker.downcallHandle(symbol, LINEAR_Q3_BF16))
                    .orElse(null);
            this.q3P2e2Expand = symbols.find("euhedral_cuda_q3_p2e2_expand")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.linearNvfp4Bf16 = symbols.find("euhedral_cuda_linear_nvfp4_bf16")
                    .map(symbol -> linker.downcallHandle(symbol, LINEAR_Q3_BF16))
                    .orElse(null);
            this.nvfp4GateUpSwiGluBf16 = symbols.find("euhedral_cuda_nvfp4_gate_up_swiglu_bf16")
                    .map(symbol -> linker.downcallHandle(symbol, LINEAR_Q3_BF16))
                    .orElse(null);
            this.nvfp4NativeAvailable = symbols.find("euhedral_cuda_nvfp4_native_available")
                    .map(symbol -> linker.downcallHandle(symbol, FunctionDescriptor.of(ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.nvfp4ActivationBytes = symbols.find("euhedral_cuda_nvfp4_native_scratch_bytes")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.linearNvfp4NativeBf16 = symbols.find("euhedral_cuda_linear_nvfp4_native_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.nvfp4NativeGateUpSwiGluBf16 = symbols.find("euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.copyDeviceToDevice2d = symbols.find("euhedral_cuda_copy_device_to_device_2d")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.synchronize = bind(linker, symbols, "euhedral_cuda_synchronize", SYNCHRONIZE);
            this.streamCreate =
                    bind(linker, symbols, "euhedral_cuda_stream_create", FunctionDescriptor.of(ValueLayout.JAVA_LONG));
            this.streamDestroy = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_stream_destroy",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.streamSynchronize = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_stream_synchronize",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.streamSelect = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_stream_select",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.rowExactSelect = symbols.find("euhedral_cuda_row_exact_select")
                    .map(symbol -> linker.downcallHandle(symbol, FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.pdlSelect = symbols.find("euhedral_cuda_pdl_select")
                    .map(symbol -> linker.downcallHandle(symbol, FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.streamClear = bind(linker, symbols, "euhedral_cuda_stream_clear", FunctionDescriptor.ofVoid());
            this.eventCreate = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_completion_event_create",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG));
            this.eventRecord = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_completion_event_record",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            this.streamWaitEvent = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_stream_wait_event",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            this.eventQuery = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_completion_event_query",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.eventDestroy = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_completion_event_destroy",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.completionNotify = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_completion_notify",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            this.completionCallback = linker.upcallStub(
                    MethodHandles.lookup()
                            .findVirtual(
                                    CudaGpuMemory.class,
                                    "nativeCompletion",
                                    MethodType.methodType(void.class, long.class, int.class))
                            .bindTo(this),
                    FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
                    loadedLibraryArena);
            this.rmsNormBf16 = bind(linker, symbols, "euhedral_cuda_rms_norm_bf16", RMS_NORM_BF16);
            this.rmsNormUnitOffsetBf16 =
                    bind(linker, symbols, "euhedral_cuda_rms_norm_unit_offset_bf16", RMS_NORM_UNIT_OFFSET_BF16);
            this.linearQ3Bf16 = bind(linker, symbols, "euhedral_cuda_linear_q3_bf16", LINEAR_Q3_BF16);
            // An explicitly scalar owner can still inspect or execute an older native package.
            // Optimized policies fail at construction rather than silently falling back.
            this.linearQ3DecodeBf16 = q3DispatchMode == Q3DispatchMode.SCALAR
                            && symbols.find("euhedral_cuda_linear_q3_decode_bf16")
                                    .isEmpty()
                    ? null
                    : bind(linker, symbols, "euhedral_cuda_linear_q3_decode_bf16", LINEAR_Q3_BF16);
            this.linearQ3PrefillBf16 = q3DispatchMode == Q3DispatchMode.SCALAR
                            && symbols.find("euhedral_cuda_linear_q3_prefill_bf16")
                                    .isEmpty()
                    ? null
                    : bind(linker, symbols, "euhedral_cuda_linear_q3_prefill_bf16", LINEAR_Q3_BF16);
            this.linearQuantizedBf16 =
                    bind(linker, symbols, "euhedral_cuda_linear_quantized_bf16", LINEAR_QUANTIZED_BF16);
            this.linearBf16ToFloat = bind(linker, symbols, "euhedral_cuda_linear_bf16_to_float", LINEAR_BF16_TO_FLOAT);
            this.gdnControlFp32 = bind(linker, symbols, "euhedral_cuda_gdn_control_fp32", GDN_CONTROL_FP32);
            this.gdnConvolutionBf16 = bind(linker, symbols, "euhedral_cuda_gdn_convolution_bf16", GDN_CONVOLUTION_BF16);
            this.gdnRecurrenceBf16 = bind(linker, symbols, "euhedral_cuda_gdn_recurrence_bf16", GDN_RECURRENCE_BF16);
            this.gdnGatedRmsNormBf16 =
                    bind(linker, symbols, "euhedral_cuda_gdn_gated_rms_norm_bf16", GDN_GATED_RMS_NORM_BF16);
            this.q3FfnStreamedBf16 = symbols.find("euhedral_cuda_q3_ffn_streamed_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.q3FfnDownBf16 = symbols.find("euhedral_cuda_q3_ffn_down_bf16")
                    .map(symbol -> linker.downcallHandle(symbol, LINEAR_Q3_BF16))
                    .orElse(null);
            this.selectExactNumerics = symbols.find("euhedral_cuda_select_exact_numerics")
                    .map(symbol -> linker.downcallHandle(
                            symbol, FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.q3FfnDownSplitBf16 = symbols.find("euhedral_cuda_q3_ffn_down_split_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.q3GateUpSwiGluBf16 = symbols.find("euhedral_cuda_q3_gate_up_swiglu_bf16")
                    .map(symbol -> linker.downcallHandle(symbol, LINEAR_Q3_BF16))
                    .orElse(null);
            this.residualRmsNormBf16 = symbols.find("euhedral_cuda_residual_rms_norm_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_FLOAT)))
                    .orElse(null);
            this.gdnProjectControlFp32 = symbols.find("euhedral_cuda_gdn_project_control_fp32")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT)))
                    .orElse(null);
            this.gdnProjectionsBf16 = symbols.find("euhedral_cuda_gdn_projections_bf16")
                    .map(symbol -> linker.downcallHandle(
                            symbol,
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
            this.residualAddBf16 = bind(linker, symbols, "euhedral_cuda_residual_add_bf16", RESIDUAL_ADD_BF16);
            this.swiGluBf16 = bind(linker, symbols, "euhedral_cuda_swiglu_bf16", SWIGLU_BF16);
            this.argmaxBf16 = bind(linker, symbols, "euhedral_cuda_argmax_bf16", ARGMAX_BF16);
            this.zeroDeviceMemory = bind(linker, symbols, "euhedral_cuda_zero_device_memory", ZERO_DEVICE_MEMORY);
            this.attentionQkNormRopeBf16 =
                    bind(linker, symbols, "euhedral_cuda_attention_qk_norm_rope_bf16", ATTENTION_QK_NORM_ROPE_BF16);
            this.attentionKvAppendNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_kv_append_nvfp4", ATTENTION_KV_APPEND_NVFP4);
            this.attentionCausalNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_causal_nvfp4", ATTENTION_CAUSAL_NVFP4);
        } catch (RuntimeException exception) {
            loadedLibraryArena.close();
            throw exception;
        } catch (Throwable exception) {
            loadedLibraryArena.close();
            throw new GpuMemoryException("CUDA stream initialization failed", exception);
        }
    }

    @Override
    public GpuStream openStream() {
        ensureOpen();
        long handle;
        try {
            handle = (long) streamCreate.invokeExact();
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA stream creation invocation failed", failure);
        }
        if (handle == 0) throw new GpuMemoryException("CUDA stream creation failed");
        openStreams.incrementAndGet();
        return new CudaStream(handle);
    }

    @Override
    public void poison(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        if (poisoned.compareAndSet(null, failure)) {
            LOG.error("CUDA recovery failed; retaining GPU allocations until process restart", failure);
        }
        // Every armed boundary still reaches its owner, which now observes the poison and retains
        // the storage that its unproven device work may still reference.
        for (var entry : completions.entrySet()) announce(entry.getKey(), entry.getValue(), false);
    }

    @Override
    public void ensureHealthy() {
        ensureOpen();
    }

    /// CUDA's host callback only announces the boundary; it never calls CUDA or releases ownership.
    private void nativeCompletion(long ticket, int status) {
        Completion completion = completions.get(ticket);
        if (completion == null) return;
        completion.callbackStatus = status;
        announce(ticket, completion, true);
    }

    private static void announce(long ticket, Completion completion, boolean driverThread) {
        if (!completion.announced.compareAndSet(false, true)) return;
        try {
            completion.listener.retired(ticket, driverThread);
        } catch (Throwable failure) {
            LOG.error("CUDA retirement listener failed", failure);
        }
    }

    private void destroyEvent(long event) {
        try {
            int status = (int) eventDestroy.invokeExact(event);
            if (status != 0) LOG.warn("CUDA event destruction failed with status {}", status);
        } catch (Throwable failure) {
            LOG.warn("CUDA event destruction failed", failure);
        }
    }

    private void recycleEvent(long event) {
        if (availableEvents.size() < MAX_CACHED_EVENTS) availableEvents.offer(event);
        else destroyEvent(event);
    }

    private static final class Completion {
        private final long event;
        private final GpuStream.RetirementListener listener;
        private final AtomicBoolean announced = new AtomicBoolean();
        private volatile int callbackStatus;

        private Completion(long event, GpuStream.RetirementListener listener) {
            this.event = event;
            this.listener = listener;
        }
    }

    /// One persistent CUDA stream; a quantum owns it while its stages run.
    private final class CudaStream implements GpuStream {
        private final long handle;
        private boolean closed;

        private CudaStream(long handle) {
            this.handle = handle;
        }

        @Override
        public void submit(Runnable launches, boolean overlapPredecessor) {
            ensureOpen();
            select(overlapPredecessor);
            CudaStream previous = submitting.get();
            submitting.set(this);
            try {
                launches.run();
            } finally {
                submitting.set(previous);
                clear(overlapPredecessor);
            }
        }

        private void select(boolean overlapPredecessor) {
            int status;
            try {
                status = (int) streamSelect.invokeExact(this.handle);
                if (status == 0 && overlapPredecessor && pdlSelect != null) pdlSelect.invokeExact(1);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission stream selection invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA submission stream selection", status);
        }

        private void clear(boolean overlapPredecessor) {
            try {
                if (overlapPredecessor && pdlSelect != null) pdlSelect.invokeExact(0);
                streamClear.invokeExact();
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission stream clear invocation failed", failure);
            }
        }

        @Override
        public long notifyRetired(RetirementListener listener) {
            ensureOpen();
            Objects.requireNonNull(listener, "listener");
            long event = 0;
            long ticket = 0;
            Completion completion = null;
            try {
                Long cached = availableEvents.poll();
                event = cached == null ? (long) eventCreate.invokeExact() : cached;
                if (event == 0) throw new GpuMemoryException("CUDA event creation returned null");
                int status = (int) eventRecord.invokeExact(event, this.handle);
                if (status != 0) throw new GpuMemoryException("CUDA event record", status);
                ticket = nextCompletion.incrementAndGet();
                if (ticket <= 0) throw new IllegalStateException("CUDA completion identifiers exhausted");
                completion = new Completion(event, listener);
                completions.put(ticket, completion);
                status = (int) completionNotify.invokeExact(this.handle, completionCallback, ticket);
                if (status != 0) throw new GpuMemoryException("CUDA completion notification", status);
                return ticket;
            } catch (Throwable failure) {
                // A concurrent poison() already announced this boundary to its listener, whose owner
                // will confirm it and observe the poison. It stays armed; a throw would retire it twice.
                if (completion != null && !completion.announced.compareAndSet(false, true)) return ticket;
                // Nothing was armed. The caller must prove device retirement before releasing storage.
                if (ticket != 0) completions.remove(ticket);
                if (event != 0) destroyEvent(event);
                if (failure instanceof RuntimeException runtime) throw runtime;
                if (failure instanceof Error error) throw error;
                throw new GpuMemoryException("CUDA completion registration failed", failure);
            }
        }

        @Override
        public long openMarker() {
            ensureOpen();
            long event;
            try {
                event = (long) eventCreate.invokeExact();
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA marker creation invocation failed", failure);
            }
            if (event == 0) throw new GpuMemoryException("CUDA marker creation returned null");
            return event;
        }

        @Override
        public void mark(long marker) {
            int status;
            try {
                status = (int) eventRecord.invokeExact(marker, this.handle);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA marker record invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA marker record", status);
        }

        @Override
        public void await(long marker) {
            int status;
            try {
                status = (int) streamWaitEvent.invokeExact(this.handle, marker);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA marker wait invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA marker wait", status);
        }

        @Override
        public void closeMarker(long marker) {
            if (marker != 0) destroyEvent(marker);
        }

        @Override
        public Throwable confirmRetired(long ticket) {
            Completion completion = completions.remove(ticket);
            if (completion == null) return new IllegalStateException("unknown CUDA retirement ticket " + ticket);
            Throwable poison = poisoned.get();
            if (poison != null) return poison;
            int status = completion.callbackStatus;
            if (status == 0) {
                try {
                    status = (int) eventQuery.invokeExact(completion.event);
                } catch (Throwable failure) {
                    return retireFailed(new GpuMemoryException("CUDA completion event query failed", failure));
                }
            }
            if (status != 0) {
                destroyEvent(completion.event);
                return retireFailed(new GpuMemoryException("CUDA asynchronous stream", status));
            }
            recycleEvent(completion.event);
            return null;
        }

        /// A failed boundary cannot prove that its stream stopped reading quantum storage.
        private Throwable retireFailed(Throwable failure) {
            try {
                CudaGpuMemory.this.synchronize();
            } catch (RuntimeException | Error synchronizationFailure) {
                failure.addSuppressed(synchronizationFailure);
                poison(failure);
            }
            return failure;
        }

        @Override
        public void synchronize() {
            ensureOpen();
            int status;
            try {
                status = (int) streamSynchronize.invokeExact(this.handle);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA stream synchronization invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA stream synchronization", status);
        }

        @Override
        public void recover(Throwable failure) {
            try {
                synchronize();
            } catch (RuntimeException | Error synchronizationFailure) {
                if (synchronizationFailure != failure) failure.addSuppressed(synchronizationFailure);
                poison(failure);
            }
        }

        @Override
        public void close() {
            if (this.closed) return;
            int status;
            try {
                status = (int) streamDestroy.invokeExact(this.handle);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA stream destruction invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA stream destruction", status);
            this.closed = true;
            openStreams.decrementAndGet();
        }
    }

    @Override
    public long allocate(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        try {
            MemorySegment address = (MemorySegment) malloc.invokeExact(byteSize);
            long value = address.address();
            if (value == 0) throw new GpuMemoryException("CUDA allocation returned a null address");
            allocationSizes.put(value, byteSize);
            long live = allocatedBytes.addAndGet(byteSize);
            peakAllocatedBytes.accumulateAndGet(live, Math::max);
            return value;
        } catch (GpuMemoryException exception) {
            throw exception;
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA allocation invocation failed", throwable);
        }
    }

    /// Pinned staging memory is page-locked, so allocating it per quantum costs about a millisecond.
    /// Small buffers are retained for reuse; larger ones are freed on release. A quantum holds its
    /// token staging and, when its KV reservation grows, one page-table staging per full-attention
    /// layer until it retires, so the cache covers a growing quantum without allocating.
    private static final long PINNED_CACHE_BYTES = 64 * 1024;

    private static final int PINNED_CACHE_ENTRIES = 32;

    @Override
    public UploadBuffer allocateUploadBuffer(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        try {
            boolean cacheable = byteSize <= PINNED_CACHE_BYTES;
            long capacity = cacheable ? PINNED_CACHE_BYTES : byteSize;
            MemorySegment address = cacheable ? pinnedUploads.poll() : null;
            if (address != null) {
                pinnedUploadCount.decrementAndGet();
            } else {
                address = (MemorySegment) hostMalloc.invokeExact(capacity);
                if (address.address() == 0) throw new GpuMemoryException("CUDA pinned upload allocation returned null");
                address = address.reinterpret(capacity);
            }
            MemorySegment allocation = address;
            return new UploadBuffer(allocation.asSlice(0, byteSize), () -> releasePinnedUpload(allocation, cacheable));
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA pinned upload allocation failed", failure);
        }
    }

    private void releasePinnedUpload(MemorySegment allocation, boolean cacheable) {
        if (cacheable && !closed && pinnedUploadCount.incrementAndGet() <= PINNED_CACHE_ENTRIES) {
            pinnedUploads.offer(allocation);
            return;
        }
        if (cacheable) pinnedUploadCount.decrementAndGet();
        freePinned(allocation);
    }

    @Override
    public long allocateHostWeights(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        MemorySegment address;
        try {
            address = (MemorySegment) hostWeightsMalloc.invokeExact(byteSize);
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA host weight allocation failed", failure);
        }
        if (address.address() == 0) throw new GpuMemoryException("CUDA host weight allocation returned null");
        this.hostWeights.put(address.address(), byteSize);
        return address.address();
    }

    @Override
    public void freeHostWeights(long address) {
        ensureOpen();
        if (this.hostWeights.remove(address) == null)
            throw new IllegalArgumentException("not a host weight allocation: " + address);
        try {
            int status = (int) hostWeightsFree.invokeExact(MemorySegment.ofAddress(address));
            if (status != 0) throw new GpuMemoryException("CUDA host weight free", status);
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA host weight free invocation failed", failure);
        }
    }

    /// Bytes of pinned host memory currently held for host-backed weights.
    public long hostWeightBytes() {
        return this.hostWeights.values().stream().mapToLong(Long::longValue).sum();
    }

    @Override
    public void copyHostWeightsToDevice(long destination, long source, long byteSize) {
        ensureOpen();
        requireDeviceAddress(destination);
        if (source == 0 || byteSize <= 0) throw new IllegalArgumentException("invalid host weight copy");
        int status = invokeCopy(
                copyUploadToDevice,
                MemorySegment.ofAddress(destination),
                MemorySegment.ofAddress(source),
                byteSize,
                "host weight copy");
        if (status != 0) throw new GpuMemoryException("host weight copy", status);
    }

    @Override
    public void copyUploadToDevice(long destination, UploadBuffer upload) {
        ensureOpen();
        requireDeviceAddress(destination);
        Objects.requireNonNull(upload, "upload");
        MemorySegment source = upload.segment();
        int status = invokeCopy(
                copyUploadToDevice,
                MemorySegment.ofAddress(destination),
                source,
                source.byteSize(),
                "pinned host-to-device copy");
        if (status != 0) throw new GpuMemoryException("pinned host-to-device copy", status);
    }

    @Override
    public boolean completionProven() {
        return poisoned.get() == null;
    }

    /// Page-locked, so a queued copy into it needs no synchronization. The caller keeps it across the
    /// stream boundary that proves the copy retired.
    @Override
    public ReadbackBuffer allocateReadbackBuffer(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        MemorySegment address;
        try {
            address = (MemorySegment) hostMalloc.invokeExact(byteSize);
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA pinned readback allocation failed", failure);
        }
        if (address.address() == 0) throw new GpuMemoryException("CUDA pinned readback allocation returned null");
        MemorySegment allocation = address.reinterpret(byteSize);
        return new ReadbackBuffer(allocation, () -> freePinned(allocation));
    }

    @Override
    public void copyDeviceToReadback(ReadbackBuffer destination, long source, long byteSize) {
        ensureOpen();
        Objects.requireNonNull(destination, "destination");
        requireTransferSize(destination.segment(), byteSize, "destination");
        requireDeviceAddress(source);
        int status = invokeCopy(
                copyDeviceToReadback,
                destination.segment(),
                MemorySegment.ofAddress(source),
                byteSize,
                "pinned device-to-host copy");
        if (status != 0) throw new GpuMemoryException("pinned device-to-host copy", status);
    }

    private void freePinned(MemorySegment address) {
        ensureOpen();
        try {
            int status = (int) hostFree.invokeExact(address);
            if (status != 0) throw new GpuMemoryException("CUDA pinned host free", status);
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA pinned host free invocation failed", failure);
        }
    }

    @Override
    public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
        ensureOpen();
        requireTransferSize(source, byteSize, "source");
        requireDeviceAddress(destination);
        int status = invokeCopy(
                copyHostToDevice, MemorySegment.ofAddress(destination), source, byteSize, "host-to-device copy");
        if (status != 0) throw new GpuMemoryException("host-to-device copy", status);
    }

    @Override
    public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
        ensureOpen();
        requireTransferSize(destination, byteSize, "destination");
        requireDeviceAddress(source);
        int status = invokeCopy(
                copyDeviceToHost, destination, MemorySegment.ofAddress(source), byteSize, "device-to-host copy");
        if (status != 0) throw new GpuMemoryException("device-to-host copy", status);
    }

    @Override
    public void copyDeviceToDevice(long destination, long source, long byteSize) {
        ensureOpen();
        if (byteSize < 0) throw new IllegalArgumentException("byteSize must not be negative");
        requireAddresses(destination, source);
        int status = invokeCopy(
                copyDeviceToDevice,
                MemorySegment.ofAddress(destination),
                MemorySegment.ofAddress(source),
                byteSize,
                "device-to-device copy");
        if (status != 0) throw new GpuMemoryException("device-to-device copy", status);
    }

    @Override
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize) {
        ensureOpen();
        requireDeviceAddress(tokenIdsAddress);
        requireDeviceAddress(embeddingAddress);
        requireDeviceAddress(hiddenStateAddress);
        if (embeddingByteSize <= 0 || tokenCount <= 0 || vocabularySize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("Q3 embedding sizes must be positive");
        }
        if (hiddenSize % 64 != 0) {
            throw new IllegalArgumentException("Q3 embedding hidden size must be divisible by 64");
        }
        int status;
        try {
            status = (int) embedQ3.invokeExact(
                    MemorySegment.ofAddress(tokenIdsAddress),
                    MemorySegment.ofAddress(embeddingAddress),
                    MemorySegment.ofAddress(hiddenStateAddress),
                    tokenCount,
                    vocabularySize,
                    hiddenSize,
                    embeddingByteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q3 embedding invocation failed", throwable);
        }
        if (status != 0) {
            String operation = status == CUDA_FORMAT_MISMATCH ? "Q3 embedding format/layout mismatch" : "Q3 embedding";
            throw new GpuMemoryException(operation, status);
        }
    }

    @Override
    public void synchronize() {
        ensureOpen();
        int status;
        try {
            status = (int) synchronize.invokeExact();
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA device synchronization invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("CUDA device synchronization", status);
    }

    @Override
    public void rmsNormBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(inputAddress, weightAddress, outputAddress);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("RMS norm dimensions and epsilon are invalid");
        }
        int status;
        try {
            status = (int) rmsNormBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    width,
                    epsilon);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("BF16 RMS norm invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("BF16 RMS norm", status);
    }

    @Override
    public void rmsNormUnitOffsetBf16(
            long inputAddress, long weightAddress, long outputAddress, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(inputAddress, weightAddress, outputAddress);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("RMS norm dimensions and epsilon are invalid");
        }
        int status;
        try {
            status = (int) rmsNormUnitOffsetBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    width,
                    epsilon);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("unit-offset BF16 RMS norm invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("unit-offset BF16 RMS norm", status);
    }

    @Override
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        linearQ3Bf16(
                inputAddress,
                weightsAddress,
                outputAddress,
                rows,
                inFeatures,
                outFeatures,
                weightsByteSize,
                this.q3DispatchMode.select(rows, this.q3SmallRowThreshold));
    }

    /// Forces a Q3 path without changing this GPU owner's immutable production dispatch policy.
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            Q3DispatchMode mode) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        int status;
        try {
            MethodHandle kernel =
                    switch (Objects.requireNonNull(mode, "mode").select(rows, this.q3SmallRowThreshold)) {
                        case SCALAR -> this.linearQ3Bf16;
                        case DECODE -> this.linearQ3DecodeBf16;
                        case PREFILL -> this.linearQ3PrefillBf16;
                        case AUTO -> throw new AssertionError("unresolved Q3 dispatch");
                    };
            if (kernel == null)
                throw new GpuMemoryException("native package does not provide the requested Q3 path: " + mode);
            status = (int) kernel.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightsAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    inFeatures,
                    outFeatures,
                    weightsByteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q3 linear invocation failed", throwable);
        }
        if (status != 0) {
            String operation = status == CUDA_FORMAT_MISMATCH ? "Q3 linear format/layout mismatch" : "Q3 linear";
            throw new GpuMemoryException(operation, status);
        }
    }

    @Override
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            WeightLayout layout) {
        linearQ3Bf16(
                inputAddress,
                weightsAddress,
                outputAddress,
                rows,
                inFeatures,
                outFeatures,
                weightsByteSize,
                this.q3DispatchMode,
                layout);
    }

    /// [#linearQ3Bf16(long, long, long, int, int, int, long, Q3DispatchMode)] for either Q3 layout. A
    /// P2E2 tensor runs one row on its own decode kernel, bitwise identical to the row-split contiguous
    /// kernel, where that kernel would run; every other route expands it into the shared scratch and
    /// runs the row-split route on the expansion, so the outputs are the same bits in every case.
    public void linearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            Q3DispatchMode mode,
            WeightLayout layout) {
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            requireRowSplit(layout, "Q3 linear");
            linearQ3Bf16(
                    inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize, mode);
            return;
        }
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        Q3DispatchMode selected = Objects.requireNonNull(mode, "mode").select(rows, this.q3SmallRowThreshold);
        if (selected == Q3DispatchMode.DECODE && rows == 1 && linearQ3P2e2DecodeBf16 != null) {
            int status;
            try {
                status = (int) linearQ3P2e2DecodeBf16.invokeExact(
                        MemorySegment.ofAddress(inputAddress),
                        MemorySegment.ofAddress(weightsAddress),
                        MemorySegment.ofAddress(outputAddress),
                        rows,
                        inFeatures,
                        outFeatures,
                        weightsByteSize);
            } catch (Throwable throwable) {
                throw new GpuMemoryException("P2E2 Q3 decode invocation failed", throwable);
            }
            if (status == 0) return;
            if (status != ROUTE_UNAVAILABLE)
                throw new GpuMemoryException(q3Operation("P2E2 Q3 decode", status), status);
        }
        long expanded = P2e2Layout.expandedByteSize(outFeatures, inFeatures);
        if (expanded <= Q3_SCRATCH_LIMIT) {
            withQ3Scratch(expanded, scratch -> {
                expandQ3(weightsAddress, weightsByteSize, outFeatures, inFeatures, 0, outFeatures, scratch, expanded);
                linearQ3Bf16(inputAddress, scratch, outputAddress, rows, inFeatures, outFeatures, expanded, selected);
            });
            return;
        }
        // Output-row chunks, each written to the scratch and copied into place. Every Q3 linear kernel
        // computes an output column from its own weight row alone, and the chunk widths avoid the
        // shapes that select a shape-specific kernel, so the outputs equal those of the whole tensor.
        if (copyDeviceToDevice2d == null) throw new UnsupportedOperationException("native pitched copy unavailable");
        int chunk = linearChunkRows(rows, inFeatures);
        for (int first = 0; first < outFeatures; first += chunk) {
            int firstRow = first;
            int count = Math.min(chunk, outFeatures - first);
            long weights = P2e2Layout.expandedByteSize(count, inFeatures);
            long outputOffset = alignUp(weights, 256);
            withQ3Scratch(outputOffset + (long) rows * count * Short.BYTES, scratch -> {
                expandQ3(weightsAddress, weightsByteSize, outFeatures, inFeatures, firstRow, count, scratch, weights);
                linearQ3Bf16(inputAddress, scratch, scratch + outputOffset, rows, inFeatures, count, weights, selected);
                copyRows(
                        outputAddress + (long) firstRow * Short.BYTES,
                        (long) outFeatures * Short.BYTES,
                        scratch + outputOffset,
                        (long) count * Short.BYTES,
                        rows);
            });
        }
    }

    /// Output rows per chunk of a P2E2 linear too large to expand at once: a multiple of 1024 that
    /// fits the scratch limit together with its output and is no FFN or mixer output width.
    private static int linearChunkRows(int rows, int inFeatures) {
        long perRow = P2e2Layout.expandedByteSize(1024, inFeatures) + 1024L * rows * Short.BYTES;
        long chunk = Math.max(1, Q3_SCRATCH_LIMIT / perRow) * 1024;
        if (chunk == 5120 || chunk == 34816) chunk -= 1024;
        return (int) Math.max(1024, Math.min(chunk, Integer.MAX_VALUE / 2));
    }

    private static long alignUp(long value, long alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    private static String q3Operation(String operation, int status) {
        return status == CUDA_FORMAT_MISMATCH ? operation + " format/layout mismatch" : operation;
    }

    /// Runs `use` with the shared scratch of at least `bytes`, ordered after its previous use on any
    /// stream; growing it first drains the device.
    private void withQ3Scratch(long bytes, LongConsumer use) {
        q3ScratchLock.lock();
        try {
            ensureOpen();
            CudaStream stream = submitting.get();
            if (bytes > q3ScratchBytes) {
                synchronize();
                if (q3ScratchAddress != 0) free(q3ScratchAddress);
                q3ScratchAddress = 0;
                q3ScratchBytes = 0;
                long size = alignUp(bytes, 1L << 20);
                q3ScratchAddress = allocate(size);
                q3ScratchBytes = size;
                LOG.info("P2E2 expansion scratch: {} MiB", size >> 20);
            } else if (stream == null) {
                synchronize();
            } else if (q3ScratchEvent != 0) {
                stream.await(q3ScratchEvent);
            }
            use.accept(q3ScratchAddress);
            if (stream != null) {
                if (q3ScratchEvent == 0) {
                    try {
                        q3ScratchEvent = (long) eventCreate.invokeExact();
                    } catch (Throwable failure) {
                        throw new GpuMemoryException("CUDA event creation invocation failed", failure);
                    }
                    if (q3ScratchEvent == 0) throw new GpuMemoryException("CUDA event creation returned null");
                }
                stream.mark(q3ScratchEvent);
            }
        } finally {
            q3ScratchLock.unlock();
        }
    }

    private void expandQ3(
            long weights,
            long weightsByteSize,
            int rows,
            int inFeatures,
            int firstRow,
            int rowCount,
            long destination,
            long destinationBytes) {
        if (q3P2e2Expand == null) throw new UnsupportedOperationException("native P2E2 expansion unavailable");
        int status;
        try {
            status = (int) q3P2e2Expand.invokeExact(
                    MemorySegment.ofAddress(weights),
                    weightsByteSize,
                    rows,
                    inFeatures,
                    firstRow,
                    rowCount,
                    MemorySegment.ofAddress(destination),
                    destinationBytes);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("P2E2 expansion invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException(q3Operation("P2E2 expansion", status), status);
    }

    @Override
    public void copyRowsDeviceToDevice(
            long destination, long destinationPitch, long source, long sourcePitch, int rows) {
        ensureOpen();
        if (copyDeviceToDevice2d == null) throw new UnsupportedOperationException("native pitched copy unavailable");
        requireAddresses(destination, source);
        copyRows(destination, destinationPitch, source, sourcePitch, rows);
    }

    private void copyRows(long destination, long destinationPitch, long source, long sourcePitch, int rows) {
        int status;
        try {
            status = (int) copyDeviceToDevice2d.invokeExact(
                    MemorySegment.ofAddress(destination),
                    destinationPitch,
                    MemorySegment.ofAddress(source),
                    sourcePitch,
                    sourcePitch,
                    (long) rows);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("pitched device copy invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("pitched device copy", status);
    }

    @Override
    public void embedQ3(
            long tokenIdsAddress,
            long embeddingAddress,
            long embeddingByteSize,
            long hiddenStateAddress,
            int tokenCount,
            int vocabularySize,
            int hiddenSize,
            WeightLayout layout) {
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            super.embedQ3(
                    tokenIdsAddress,
                    embeddingAddress,
                    embeddingByteSize,
                    hiddenStateAddress,
                    tokenCount,
                    vocabularySize,
                    hiddenSize,
                    layout);
            return;
        }
        ensureOpen();
        requireAddresses(tokenIdsAddress, embeddingAddress, hiddenStateAddress);
        if (embeddingByteSize <= 0 || tokenCount <= 0 || vocabularySize <= 0 || hiddenSize <= 0) {
            throw new IllegalArgumentException("Q3 embedding sizes must be positive");
        }
        if (embedQ3P2e2 == null) throw new UnsupportedOperationException("native P2E2 embedding unavailable");
        int status;
        try {
            status = (int) embedQ3P2e2.invokeExact(
                    MemorySegment.ofAddress(tokenIdsAddress),
                    MemorySegment.ofAddress(embeddingAddress),
                    MemorySegment.ofAddress(hiddenStateAddress),
                    tokenCount,
                    vocabularySize,
                    hiddenSize,
                    embeddingByteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("P2E2 Q3 embedding invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException(q3Operation("P2E2 Q3 embedding", status), status);
    }

    @Override
    public void q3FfnStreamedBf16(
            long input,
            long gateWeights,
            long downWeights,
            long output,
            long slots,
            long accumulators,
            int rows,
            int hidden,
            int intermediate,
            long gateBytes,
            long downBytes,
            WeightLayout gateLayout,
            WeightLayout downLayout) {
        boolean gateP2e2 = Objects.requireNonNull(gateLayout, "gateLayout") == WeightLayout.ROW_SPLIT_P2E2_V1;
        boolean downP2e2 = Objects.requireNonNull(downLayout, "downLayout") == WeightLayout.ROW_SPLIT_P2E2_V1;
        if (!gateP2e2 && !downP2e2) {
            super.q3FfnStreamedBf16(
                    input,
                    gateWeights,
                    downWeights,
                    output,
                    slots,
                    accumulators,
                    rows,
                    hidden,
                    intermediate,
                    gateBytes,
                    downBytes,
                    gateLayout,
                    downLayout);
            return;
        }
        if (!gateP2e2) requireRowSplit(gateLayout, "streamed FFN region");
        if (!downP2e2) requireRowSplit(downLayout, "streamed FFN region");
        long gateExpanded = gateP2e2 ? P2e2Layout.expandedByteSize(2L * intermediate, hidden) : 0;
        long downOffset = alignUp(gateExpanded, 256);
        long downExpanded = downP2e2 ? P2e2Layout.expandedByteSize(hidden, intermediate) : 0;
        withQ3Scratch(downOffset + downExpanded, scratch -> {
            if (gateP2e2)
                expandQ3(gateWeights, gateBytes, 2 * intermediate, hidden, 0, 2 * intermediate, scratch, gateExpanded);
            if (downP2e2)
                expandQ3(downWeights, downBytes, hidden, intermediate, 0, hidden, scratch + downOffset, downExpanded);
            q3FfnStreamedBf16(
                    input,
                    gateP2e2 ? scratch : gateWeights,
                    downP2e2 ? scratch + downOffset : downWeights,
                    output,
                    slots,
                    accumulators,
                    rows,
                    hidden,
                    intermediate,
                    gateP2e2 ? gateExpanded : gateBytes,
                    downP2e2 ? downExpanded : downBytes);
        });
    }

    @Override
    public void q3FfnDownBf16(
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes,
            WeightLayout layout) {
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            super.q3FfnDownBf16(input, weights, output, rows, width, outputs, weightBytes, layout);
            return;
        }
        long expanded = P2e2Layout.expandedByteSize(outputs, width);
        withQ3Scratch(expanded, scratch -> {
            expandQ3(weights, weightBytes, outputs, width, 0, outputs, scratch, expanded);
            q3FfnDownBf16(input, scratch, output, rows, width, outputs, expanded);
        });
    }

    @Override
    public void q3FfnDownSplitBf16(
            long input,
            long weights,
            long output,
            long partials,
            int rows,
            int width,
            int outputs,
            long weightBytes,
            WeightLayout layout) {
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            super.q3FfnDownSplitBf16(input, weights, output, partials, rows, width, outputs, weightBytes, layout);
            return;
        }
        long expanded = P2e2Layout.expandedByteSize(outputs, width);
        withQ3Scratch(expanded, scratch -> {
            expandQ3(weights, weightBytes, outputs, width, 0, outputs, scratch, expanded);
            q3FfnDownSplitBf16(input, scratch, output, partials, rows, width, outputs, expanded);
        });
    }

    @Override
    public void q3GateUpSwiGluBf16(
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes,
            WeightLayout layout) {
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            super.q3GateUpSwiGluBf16(input, weights, output, rows, width, outputs, weightBytes, layout);
            return;
        }
        long expanded = P2e2Layout.expandedByteSize(outputs, width);
        withQ3Scratch(expanded, scratch -> {
            expandQ3(weights, weightBytes, outputs, width, 0, outputs, scratch, expanded);
            q3GateUpSwiGluBf16(input, scratch, output, rows, width, outputs, expanded);
        });
    }

    /// Rows from which an NVFP4 linear runs on native FP4 tensor cores: one row stays on the GEMV, which
    /// streams weights as fast and keeps BF16 activations (docs/NVFP4_NATIVE.md).
    static final int NVFP4_NATIVE_MIN_ROWS = 2;
    /// The paired gate/up region exists only in region views, from 64 rows.
    static final int NVFP4_NATIVE_REGION_MIN_ROWS = 64;

    /// Whether the native Blackwell NVFP4 kernels are loaded: an sm_12x device, and not disabled with
    /// EUHEDRAL_NVFP4_NATIVE=0.
    public boolean nvfp4NativeAvailable() {
        Boolean available = this.nvfp4Native;
        if (available == null) {
            ensureOpen();
            try {
                available = this.nvfp4NativeAvailable != null && (int) this.nvfp4NativeAvailable.invokeExact() != 0;
            } catch (Throwable failure) {
                throw new GpuMemoryException("native NVFP4 availability invocation failed", failure);
            }
            this.nvfp4Native = available;
        }
        return available;
    }

    /// Native-numerics NVFP4 decode ([ExecutionGpu#selectNvfp4NativeDecode]); EUHEDRAL_NVFP4_NATIVE_DECODE=1
    /// selects it at construction.
    private volatile boolean nvfp4NativeDecode = "1".equals(System.getenv("EUHEDRAL_NVFP4_NATIVE_DECODE"));

    @Override
    public void selectNvfp4NativeDecode(boolean enabled) {
        if (enabled && !nvfp4NativeAvailable())
            throw new UnsupportedOperationException("native NVFP4 kernels are unavailable");
        this.nvfp4NativeDecode = enabled;
    }

    @Override
    public void linearNvfp4Bf16(
            long input, long weights, long output, int rows, int inFeatures, int outFeatures, long weightBytes) {
        boolean route = this.nvfp4NativeDecode
                || rows >= NVFP4_NATIVE_MIN_ROWS && !ROW_EXACT.get()[0];
        if (route && inFeatures % 128 == 0 && nvfp4NativeAvailable()) {
            if (invokeNativeNvfp4(
                    "native NVFP4 linear",
                    this.linearNvfp4NativeBf16,
                    input,
                    weights,
                    output,
                    rows,
                    inFeatures,
                    outFeatures,
                    weightBytes)) return;
        }
        invokeNvfp4(
                "NVFP4 linear", linearNvfp4Bf16, input, weights, output, rows, inFeatures, outFeatures, weightBytes);
    }

    @Override
    public void nvfp4GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        if (rows >= NVFP4_NATIVE_REGION_MIN_ROWS
                && width % 128 == 0
                && !ROW_EXACT.get()[0]
                && nvfp4NativeAvailable()) {
            if (invokeNativeNvfp4(
                    "native NVFP4 gate/up SwiGLU region",
                    this.nvfp4NativeGateUpSwiGluBf16,
                    input,
                    weights,
                    output,
                    rows,
                    width,
                    outputs,
                    weightBytes)) return;
        }
        invokeNvfp4(
                "NVFP4 gate/up SwiGLU region",
                nvfp4GateUpSwiGluBf16,
                input,
                weights,
                output,
                rows,
                width,
                outputs,
                weightBytes);
    }

    /// Quantizes the activations into the shared scratch, ordered between lanes by its event, and runs
    /// `kernel` on native FP4 tensor cores. False when the route declined (exact numerics select the
    /// BF16-expansion kernels).
    private boolean invokeNativeNvfp4(
            String operation,
            MethodHandle kernel,
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes) {
        ensureOpen();
        requireAddresses(input, weights, output);
        long activationBytes;
        try {
            activationBytes = (long) this.nvfp4ActivationBytes.invokeExact(rows, width, outputs);
        } catch (Throwable failure) {
            throw new GpuMemoryException("NVFP4 activation size invocation failed", failure);
        }
        boolean[] ran = {false};
        withQ3Scratch(activationBytes, scratch -> {
            int status;
            try {
                status = (int) kernel.invokeExact(
                        MemorySegment.ofAddress(input),
                        MemorySegment.ofAddress(weights),
                        MemorySegment.ofAddress(output),
                        MemorySegment.ofAddress(scratch),
                        rows,
                        width,
                        outputs,
                        weightBytes,
                        activationBytes);
            } catch (Throwable throwable) {
                throw new GpuMemoryException(operation + " invocation failed", throwable);
            }
            if (status == ROUTE_UNAVAILABLE) return;
            if (status != 0) throw new GpuMemoryException(q3Operation(operation, status), status);
            ran[0] = true;
        });
        return ran[0];
    }

    private void invokeNvfp4(
            String operation,
            MethodHandle kernel,
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes) {
        ensureOpen();
        requireAddresses(input, weights, output);
        if (rows <= 0 || width <= 0 || outputs <= 0 || weightBytes <= 0)
            throw new IllegalArgumentException(operation + " dimensions and payload size must be positive");
        if (kernel == null) throw new UnsupportedOperationException("native " + operation + " unavailable");
        int status;
        try {
            status = (int) kernel.invokeExact(
                    MemorySegment.ofAddress(input),
                    MemorySegment.ofAddress(weights),
                    MemorySegment.ofAddress(output),
                    rows,
                    width,
                    outputs,
                    weightBytes);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException(q3Operation(operation, status), status);
    }

    @Override
    public void linearQ4Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        linearQuantizedBf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize, 4);
    }

    @Override
    public void linearQ5Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        linearQuantizedBf16(
                inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize, 5);
    }

    private void linearQuantizedBf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize,
            int bits) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("quantized linear dimensions and payload size must be positive");
        }
        int status;
        try {
            status = (int) linearQuantizedBf16.invokeExact(
                    MemorySegment.ofAddress(inputAddress),
                    MemorySegment.ofAddress(weightsAddress),
                    MemorySegment.ofAddress(outputAddress),
                    rows,
                    inFeatures,
                    outFeatures,
                    weightsByteSize,
                    bits);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("Q" + bits + " linear invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("Q" + bits + " linear", status);
    }

    @Override
    public void linearBf16ToFloat(
            long inputAddress, long weightsAddress, long outputAddress, int rows, int inFeatures, int outFeatures) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0) {
            throw new IllegalArgumentException("BF16 linear dimensions must be positive");
        }
        invokeLayer(
                "BF16-to-FP32 linear",
                linearBf16ToFloat,
                MemorySegment.ofAddress(inputAddress),
                MemorySegment.ofAddress(weightsAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                inFeatures,
                outFeatures);
    }

    @Override
    public void gdnControlFp32(
            long aProjectionAddress,
            long bProjectionAddress,
            long aLogAddress,
            long dtBiasAddress,
            long alphaOutputAddress,
            long betaOutputAddress,
            int rows,
            int heads) {
        ensureOpen();
        requireAddresses(
                aProjectionAddress,
                bProjectionAddress,
                aLogAddress,
                dtBiasAddress,
                alphaOutputAddress,
                betaOutputAddress);
        if (rows <= 0 || heads <= 0) throw new IllegalArgumentException("GDN control dimensions must be positive");
        invokeLayer(
                "GDN control",
                gdnControlFp32,
                MemorySegment.ofAddress(aProjectionAddress),
                MemorySegment.ofAddress(bProjectionAddress),
                MemorySegment.ofAddress(aLogAddress),
                MemorySegment.ofAddress(dtBiasAddress),
                MemorySegment.ofAddress(alphaOutputAddress),
                MemorySegment.ofAddress(betaOutputAddress),
                rows,
                heads);
    }

    @Override
    public void gdnConvolutionBf16(
            long queryKeyAddress,
            long valueZAddress,
            long convolutionWeightsAddress,
            long convolutionStateAddress,
            long outputAddress,
            int rows,
            int queryKeyWidth,
            int valueWidth,
            int convolutionWidth,
            int kernelSize) {
        ensureOpen();
        requireAddresses(
                queryKeyAddress, valueZAddress, convolutionWeightsAddress, convolutionStateAddress, outputAddress);
        if (rows <= 0
                || queryKeyWidth <= 0
                || valueWidth <= 0
                || convolutionWidth != queryKeyWidth + valueWidth
                || kernelSize < 2
                || kernelSize > 32) {
            throw new IllegalArgumentException("GDN convolution dimensions are invalid");
        }
        invokeLayer(
                "GDN convolution",
                gdnConvolutionBf16,
                MemorySegment.ofAddress(queryKeyAddress),
                MemorySegment.ofAddress(valueZAddress),
                MemorySegment.ofAddress(convolutionWeightsAddress),
                MemorySegment.ofAddress(convolutionStateAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                queryKeyWidth,
                valueWidth,
                convolutionWidth,
                kernelSize);
    }

    @Override
    public void gdnRecurrenceBf16(
            long convolvedAddress,
            long alphaAddress,
            long betaAddress,
            long recurrentStateAddress,
            long outputAddress,
            int rows,
            int keyHeads,
            int valueHeads,
            int keyHeadDim,
            int valueHeadDim,
            float outputScale) {
        ensureOpen();
        requireAddresses(convolvedAddress, alphaAddress, betaAddress, recurrentStateAddress, outputAddress);
        if (rows <= 0
                || keyHeads <= 0
                || valueHeads <= 0
                || valueHeads % keyHeads != 0
                || keyHeadDim != 128
                || valueHeadDim != 128
                || !Float.isFinite(outputScale)
                || outputScale <= 0) {
            throw new IllegalArgumentException("GDN recurrence dimensions and scale are invalid");
        }
        invokeLayer(
                "GDN recurrence",
                gdnRecurrenceBf16,
                MemorySegment.ofAddress(convolvedAddress),
                MemorySegment.ofAddress(alphaAddress),
                MemorySegment.ofAddress(betaAddress),
                MemorySegment.ofAddress(recurrentStateAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                keyHeads,
                valueHeads,
                keyHeadDim,
                valueHeadDim,
                outputScale);
    }

    @Override
    public void gdnGatedRmsNormBf16(
            long recurrentAddress,
            long valueZAddress,
            long normWeightAddress,
            long outputAddress,
            int rows,
            int valueHeads,
            int headDim,
            float epsilon) {
        ensureOpen();
        requireAddresses(recurrentAddress, valueZAddress, normWeightAddress, outputAddress);
        if (rows <= 0 || valueHeads <= 0 || headDim != 128 || !Float.isFinite(epsilon) || epsilon < 0) {
            throw new IllegalArgumentException("GDN gated RMSNorm dimensions and epsilon are invalid");
        }
        invokeLayer(
                "GDN gated RMSNorm",
                gdnGatedRmsNormBf16,
                MemorySegment.ofAddress(recurrentAddress),
                MemorySegment.ofAddress(valueZAddress),
                MemorySegment.ofAddress(normWeightAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                valueHeads,
                headDim,
                epsilon);
    }

    @Override
    public void q3FfnStreamedBf16(
            long input,
            long gateWeights,
            long downWeights,
            long output,
            long slots,
            long accumulators,
            int rows,
            int hidden,
            int intermediate,
            long gateBytes,
            long downBytes) {
        ensureOpen();
        requireAddresses(input, gateWeights, downWeights, output, slots, accumulators);
        if ((rows != 64 && rows != 1024) || hidden != 5120 || intermediate != 17408 || gateBytes <= 0 || downBytes <= 0)
            throw new IllegalArgumentException("streamed FFN geometry is not qualified");
        if (q3FfnStreamedBf16 == null) throw new UnsupportedOperationException("native streamed FFN unavailable");
        invokeLayer(
                "streamed FFN region",
                q3FfnStreamedBf16,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(gateWeights),
                MemorySegment.ofAddress(downWeights),
                MemorySegment.ofAddress(output),
                MemorySegment.ofAddress(slots),
                MemorySegment.ofAddress(accumulators),
                rows,
                hidden,
                intermediate,
                gateBytes,
                downBytes);
    }

    @Override
    public void q3FfnDownBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        ensureOpen();
        requireAddresses(input, weights, output);
        if (q3FfnDownBf16 == null) {
            linearQ3Bf16(input, weights, output, rows, width, outputs, weightBytes);
            return;
        }
        if (rows <= 0 || width <= 0 || outputs <= 0 || weightBytes <= 0)
            throw new IllegalArgumentException("invalid Q3 FFN down dimensions");
        invokeLayer(
                "Q3 FFN down",
                q3FfnDownBf16,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(weights),
                MemorySegment.ofAddress(output),
                rows,
                width,
                outputs,
                weightBytes);
    }

    /// Selects exact numerics process-wide: every relaxed-order kernel (contiguous Q3 decode, split-K
    /// FFN down) is replaced by its bitwise-exact counterpart for later launches. Returns the previous
    /// selection. For numerical comparisons against the oracle; `EUHEDRAL_EXACT=1` sets the default.
    @Override
    public void selectRowExact(boolean enabled) {
        if (this.rowExactSelect == null) {
            if (enabled) throw new UnsupportedOperationException("native package has no row-exact execution");
            return;
        }
        try {
            this.rowExactSelect.invokeExact(enabled ? 1 : 0);
        } catch (Throwable failure) {
            throw new GpuMemoryException("row-exact selection invocation failed", failure);
        }
        ROW_EXACT.get()[0] = enabled;
    }

    public boolean selectExactNumerics(boolean exact) {
        ensureOpen();
        if (selectExactNumerics == null)
            throw new UnsupportedOperationException("exact numerics selection is unavailable");
        try {
            return (int) selectExactNumerics.invokeExact(exact ? 1 : 0) != 0;
        } catch (Throwable failure) {
            throw new IllegalStateException("exact numerics selection failed", failure);
        }
    }

    @Override
    public void q3FfnDownSplitBf16(
            long input, long weights, long output, long partials, int rows, int width, int outputs, long weightBytes) {
        ensureOpen();
        requireAddresses(input, weights, output, partials);
        if (q3FfnDownSplitBf16 == null) {
            q3FfnDownBf16(input, weights, output, rows, width, outputs, weightBytes);
            return;
        }
        if (rows <= 0 || width <= 0 || outputs <= 0 || weightBytes <= 0)
            throw new IllegalArgumentException("invalid Q3 FFN down dimensions");
        invokeLayer(
                "Q3 split-K FFN down",
                q3FfnDownSplitBf16,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(weights),
                MemorySegment.ofAddress(output),
                MemorySegment.ofAddress(partials),
                rows,
                width,
                outputs,
                weightBytes);
    }

    @Override
    public void q3GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        ensureOpen();
        requireAddresses(input, weights, output);
        if (rows <= 0 || width <= 0 || width % 128 != 0 || outputs <= 0 || outputs % 32 != 0 || weightBytes <= 0)
            throw new IllegalArgumentException("invalid Q3 gate/up region dimensions");
        if (q3GateUpSwiGluBf16 == null) throw new UnsupportedOperationException("native Q3 gate/up region unavailable");
        invokeLayer(
                "Q3 gate/up SwiGLU region",
                q3GateUpSwiGluBf16,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(weights),
                MemorySegment.ofAddress(output),
                rows,
                width,
                outputs,
                weightBytes);
    }

    @Override
    public void residualRmsNormBf16(
            long residual, long delta, long weight, long hidden, long normalized, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(residual, delta, weight, hidden, normalized);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0)
            throw new IllegalArgumentException("invalid residual RMSNorm region dimensions");
        if (residualRmsNormBf16 == null)
            throw new UnsupportedOperationException("native residual RMSNorm region unavailable");
        invokeLayer(
                "residual RMSNorm region",
                residualRmsNormBf16,
                MemorySegment.ofAddress(residual),
                MemorySegment.ofAddress(delta),
                MemorySegment.ofAddress(weight),
                MemorySegment.ofAddress(hidden),
                MemorySegment.ofAddress(normalized),
                rows,
                width,
                epsilon);
    }

    @Override
    public void gdnProjectControlFp32(
            long input,
            long aWeight,
            long bWeight,
            long aLog,
            long dtBias,
            long g,
            long beta,
            int rows,
            int width,
            int heads) {
        ensureOpen();
        requireAddresses(input, aWeight, bWeight, aLog, dtBias, g, beta);
        if (rows <= 0 || width <= 0 || heads <= 0)
            throw new IllegalArgumentException("invalid GDN control region dimensions");
        if (gdnProjectControlFp32 == null)
            throw new UnsupportedOperationException("native GDN control region unavailable");
        invokeLayer(
                "GDN projection/control region",
                gdnProjectControlFp32,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(aWeight),
                MemorySegment.ofAddress(bWeight),
                MemorySegment.ofAddress(aLog),
                MemorySegment.ofAddress(dtBias),
                MemorySegment.ofAddress(g),
                MemorySegment.ofAddress(beta),
                rows,
                width,
                heads);
    }

    @Override
    public void gdnProjectionsBf16(
            long input,
            long q4Weights,
            long q5Weights,
            long queryKeyOutput,
            long valueZOutput,
            int rows,
            int hidden,
            int queryKeyWidth,
            int valueZWidth,
            long q4Bytes,
            long q5Bytes) {
        if (gdnProjectionsBf16 == null) {
            super.gdnProjectionsBf16(
                    input,
                    q4Weights,
                    q5Weights,
                    queryKeyOutput,
                    valueZOutput,
                    rows,
                    hidden,
                    queryKeyWidth,
                    valueZWidth,
                    q4Bytes,
                    q5Bytes);
            return;
        }
        ensureOpen();
        requireAddresses(input, q4Weights, q5Weights, queryKeyOutput, valueZOutput);
        if (rows <= 0 || hidden <= 0 || queryKeyWidth <= 0 || valueZWidth <= 0 || q4Bytes <= 0 || q5Bytes <= 0)
            throw new IllegalArgumentException("invalid GDN projection dimensions");
        invokeLayer(
                "GDN projections",
                gdnProjectionsBf16,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(q4Weights),
                MemorySegment.ofAddress(q5Weights),
                MemorySegment.ofAddress(queryKeyOutput),
                MemorySegment.ofAddress(valueZOutput),
                rows,
                hidden,
                queryKeyWidth,
                valueZWidth,
                q4Bytes,
                q5Bytes);
    }

    @Override
    public void residualAddBf16(long residualAddress, long deltaAddress, long outputAddress, int rows, int width) {
        ensureOpen();
        requireAddresses(residualAddress, deltaAddress, outputAddress);
        if (rows <= 0 || width <= 0) throw new IllegalArgumentException("residual dimensions must be positive");
        invokeLayer(
                "BF16 residual add",
                residualAddBf16,
                MemorySegment.ofAddress(residualAddress),
                MemorySegment.ofAddress(deltaAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                width);
    }

    @Override
    public boolean argmaxBf16(long logitsAddress, int count, long resultAddress) {
        ensureOpen();
        requireAddresses(logitsAddress, resultAddress);
        if (count <= 0) throw new IllegalArgumentException("logit count must be positive");
        int status;
        try {
            status = (int) argmaxBf16.invokeExact(
                    MemorySegment.ofAddress(logitsAddress), count, MemorySegment.ofAddress(resultAddress));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("BF16 argmax invocation failed", throwable);
        }
        if (status == KERNEL_UNAVAILABLE) return false;
        if (status != 0) throw new GpuMemoryException("BF16 argmax", status);
        return true;
    }

    @Override
    public void swiGluBf16(long gateUpAddress, long outputAddress, int rows, int intermediateSize) {
        ensureOpen();
        requireAddresses(gateUpAddress, outputAddress);
        if (rows <= 0 || intermediateSize <= 0)
            throw new IllegalArgumentException("SwiGLU dimensions must be positive");
        invokeLayer(
                "BF16 SwiGLU",
                swiGluBf16,
                MemorySegment.ofAddress(gateUpAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                intermediateSize);
    }

    @Override
    public void zeroDeviceMemory(long address, long byteSize) {
        ensureOpen();
        requireDeviceAddress(address);
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        invokeLayer("CUDA device memory zero", zeroDeviceMemory, MemorySegment.ofAddress(address), byteSize);
    }

    @Override
    public void attentionQkNormRopeBf16(
            long queryKeyAddress,
            long queryNormAddress,
            long keyNormAddress,
            long outputAddress,
            int rows,
            int queryHeads,
            int keyValueHeads,
            int headDim,
            int rotaryDim,
            long startPosition,
            float epsilon,
            double ropeTheta) {
        ensureOpen();
        requireAddresses(queryKeyAddress, queryNormAddress, keyNormAddress, outputAddress);
        if (rows <= 0
                || queryHeads <= 0
                || keyValueHeads <= 0
                || queryHeads % keyValueHeads != 0
                || headDim != 256
                || rotaryDim <= 0
                || rotaryDim > headDim
                || (rotaryDim & 1) != 0
                || startPosition < 0
                || !Float.isFinite(epsilon)
                || epsilon <= 0
                || !Double.isFinite(ropeTheta)
                || ropeTheta <= 0) {
            throw new IllegalArgumentException("attention Q/K normalization and RoPE dimensions are invalid");
        }
        invokeLayer(
                "Qwen attention Q/K normalization and RoPE",
                attentionQkNormRopeBf16,
                MemorySegment.ofAddress(queryKeyAddress),
                MemorySegment.ofAddress(queryNormAddress),
                MemorySegment.ofAddress(keyNormAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                queryHeads,
                keyValueHeads,
                headDim,
                rotaryDim,
                startPosition,
                epsilon,
                ropeTheta);
    }

    @Override
    public void attentionKvAppendNvfp4(
            long queryKeyAddress,
            long gateValueAddress,
            long keyCacheAddress,
            long valueCacheAddress,
            int rows,
            int queryWidth,
            int keyValueWidth,
            long startPosition) {
        ensureOpen();
        requireAddresses(queryKeyAddress, gateValueAddress, keyCacheAddress, valueCacheAddress);
        if (rows <= 0
                || queryWidth <= 0
                || keyValueWidth <= 0
                || queryWidth % keyValueWidth != 0
                || startPosition < 0) {
            throw new IllegalArgumentException("attention KV append dimensions are invalid");
        }
        Math.addExact(startPosition, rows);
        invokeLayer(
                "Qwen attention KV append",
                attentionKvAppendNvfp4,
                MemorySegment.ofAddress(queryKeyAddress),
                MemorySegment.ofAddress(gateValueAddress),
                MemorySegment.ofAddress(keyCacheAddress),
                MemorySegment.ofAddress(valueCacheAddress),
                rows,
                queryWidth,
                keyValueWidth,
                startPosition);
    }

    @Override
    public void attentionCausalNvfp4(
            long queryKeyAddress,
            long gateValueAddress,
            long keyCacheAddress,
            long valueCacheAddress,
            long outputAddress,
            int rows,
            int queryHeads,
            int keyValueHeads,
            int headDim,
            int cacheLength,
            long startPosition,
            long scratchAddress) {
        ensureOpen();
        requireAddresses(queryKeyAddress, gateValueAddress, keyCacheAddress, valueCacheAddress, outputAddress);
        if (rows <= 0
                || queryHeads <= 0
                || keyValueHeads <= 0
                || queryHeads % keyValueHeads != 0
                || headDim != 256
                || cacheLength <= 0
                || startPosition < 0
                || Math.addExact(startPosition, rows) > cacheLength) {
            throw new IllegalArgumentException("causal attention dimensions are invalid");
        }
        if (rows == 1) requireAddresses(scratchAddress);
        invokeLayer(
                "Qwen causal attention",
                attentionCausalNvfp4,
                MemorySegment.ofAddress(queryKeyAddress),
                MemorySegment.ofAddress(gateValueAddress),
                MemorySegment.ofAddress(keyCacheAddress),
                MemorySegment.ofAddress(valueCacheAddress),
                MemorySegment.ofAddress(outputAddress),
                rows,
                queryHeads,
                keyValueHeads,
                headDim,
                cacheLength,
                startPosition,
                MemorySegment.ofAddress(scratchAddress));
    }

    @Override
    public void free(long address) {
        ensureOpen();
        if (address == 0) return;
        int status;
        try {
            status = (int) free.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA free invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("CUDA free", status);
        Long size = allocationSizes.remove(address);
        if (size != null) allocatedBytes.addAndGet(-size);
    }

    /// Device bytes allocated through this binding and not yet freed: the resident footprint of the
    /// model, its sequences' persistent state, and the execution graphs' workspace storage. Unlike
    /// [#deviceMemoryInfo], it excludes other processes and CUDA's own context and kernel modules, and
    /// it stays readable after close.
    public long allocatedBytes() {
        return allocatedBytes.get();
    }

    /// The largest [#allocatedBytes] seen since construction or the last [#resetPeakAllocatedBytes]:
    /// transient allocations included. Stays readable after close.
    public long peakAllocatedBytes() {
        return peakAllocatedBytes.get();
    }

    /// Restarts the high-water mark at the current [#allocatedBytes].
    public void resetPeakAllocatedBytes() {
        peakAllocatedBytes.set(allocatedBytes.get());
    }

    /// Returns the current CUDA device's free and total memory, in bytes.
    public DeviceMemoryInfo deviceMemoryInfo() {
        ensureOpen();
        try (Arena queryArena = Arena.ofConfined()) {
            MemorySegment freeBytes = queryArena.allocate(Long.BYTES, Long.BYTES);
            MemorySegment totalBytes = queryArena.allocate(Long.BYTES, Long.BYTES);
            int status;
            try {
                status = (int) deviceMemoryInfo.invokeExact(freeBytes, totalBytes);
            } catch (Throwable throwable) {
                throw new GpuMemoryException("CUDA device memory query invocation failed", throwable);
            }
            if (status != 0) throw new GpuMemoryException("CUDA device memory query", status);
            long free = freeBytes.get(ValueLayout.JAVA_LONG, 0);
            long total = totalBytes.get(ValueLayout.JAVA_LONG, 0);
            if (free < 0 || total <= 0 || free > total) {
                throw new GpuMemoryException("CUDA device memory query returned invalid byte counts");
            }
            return new DeviceMemoryInfo(free, total);
        }
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        ensureHealthy();
        if (!completions.isEmpty()) throw new IllegalStateException("CUDA retirement boundaries did not drain");
        if (openStreams.get() != 0) throw new IllegalStateException("CUDA quantum streams are still open");
        // A retirement may be confirmed before its native host callback returns. Drain the device
        // before releasing the FFM upcall stub or unloading its library arena.
        synchronize();
        if (q3ScratchAddress != 0) free(q3ScratchAddress);
        q3ScratchAddress = 0;
        q3ScratchBytes = 0;
        if (q3ScratchEvent != 0) destroyEvent(q3ScratchEvent);
        q3ScratchEvent = 0;
        for (Long event; (event = availableEvents.poll()) != null; ) destroyEvent(event);
        for (MemorySegment pinned; (pinned = pinnedUploads.poll()) != null; ) freePinned(pinned);
        closed = true;
        arena.close();
    }

    private static int invokeCopy(
            MethodHandle handle,
            MemorySegment firstAddress,
            MemorySegment secondAddress,
            long byteSize,
            String operation) {
        try {
            return (int) handle.invokeExact(firstAddress, secondAddress, byteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
    }

    private void invokeLayer(String operation, MethodHandle handle, Object... arguments) {
        int status;
        try {
            status = (int) handle.invokeWithArguments(arguments);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException(operation, status);
    }

    private static void requireAddresses(long... addresses) {
        for (long address : addresses) requireDeviceAddress(address);
    }

    private static void requireDeviceAddress(long address) {
        if (address == 0) throw new IllegalArgumentException("device address must be non-null");
    }

    private void ensureOpen() {
        Throwable failure = poisoned.get();
        if (failure != null)
            throw new IllegalStateException("CUDA engine is poisoned; GPU ownership is retained", failure);
        if (closed) throw new IllegalStateException("CUDA memory binding is closed");
    }

    private static void requireTransferSize(MemorySegment hostSegment, long byteSize, String name) {
        Objects.requireNonNull(hostSegment, name);
        if (byteSize < 0 || byteSize > hostSegment.byteSize()) {
            throw new IllegalArgumentException(name + " does not contain byteSize bytes");
        }
    }

    /// A snapshot of device memory capacity, in bytes.
    public record DeviceMemoryInfo(long freeBytes, long totalBytes) {}
}
