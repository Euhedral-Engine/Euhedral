package io.euhedral_execution.inference.core.gpu;

import io.euhedral_execution.data_structures.queues.MpmcQueue;
import io.euhedral_execution.inference.core.artifact.P2e2Layout;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
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
    /// Expansion scratch above which a P2E2 linear is computed in output-row chunks. The largest layer
    /// tensor that expands, the FFN gate/up (72 MB), fits with its activation region.
    private static final long Q3_SCRATCH_LIMIT = 128L << 20;
    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle mallocAsync;
    private final MethodHandle freeAsync;
    /// Live allocations made in stream order; only these are freed in stream order.
    private final java.util.Set<Long> streamOrdered = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private final MethodHandle hostMalloc;
    private final MethodHandle hostFree;
    private final MethodHandle hostWeightsMalloc;
    private final MethodHandle hostWeightsFree;
    private final MethodHandle hostWeightsDevicePointer;
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
    private final MethodHandle submissionRecord;
    private final MethodHandle submissionCheck;
    private final MethodHandle submissionFinish;
    private final MethodHandle submissionMarkUnrecordable;
    private final MethodHandle submissionCheckBegin;
    private final MethodHandle submissionCheckEnd;
    private final MethodHandle graphCaptureBegin;
    private final MethodHandle graphCaptureEnd;
    private final MethodHandle graphLaunch;
    private final MethodHandle graphDestroy;
    /// The calling thread's submission mode while a stage runs: recording, checking, or neither.
    private static final ThreadLocal<int[]> SUBMISSION = ThreadLocal.withInitial(() -> new int[1]);

    private static final int RECORDING = 1;
    private static final int CHECKING = 2;
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
    /// Live device allocations and pinned readback buffers by start address: a serial per allocation.
    private final ConcurrentHashMap<Long, Long> allocationIds = new ConcurrentHashMap<>();
    private final AtomicLong allocationSerial = new AtomicLong();
    private final AtomicLong allocatedBytes = new AtomicLong();
    /// High-water mark of [#allocatedBytes] since construction or the last [#resetPeakAllocatedBytes].
    private final AtomicLong peakAllocatedBytes = new AtomicLong();
    private final MethodHandle rmsNormBf16;
    private final MethodHandle rmsNormUnitOffsetBf16;
    private final MethodHandle linearQ3Bf16;
    private final MethodHandle linearQ3ReferenceBf16;
    /// Rows up to which a quantized linear runs the decode kernels (one launch, every row bit for bit as a one-row
    /// call); more rows run the block-scaled FP8 route.
    private static final int Q3_DECODE_MAX_ROWS = 8;
    private final MethodHandle linearQuantizedBf16;
    private final MethodHandle linearBf16ToFloat;
    private final MethodHandle gdnControlFp32;
    private final MethodHandle gdnConvolutionBf16;
    private final MethodHandle gdnRecurrenceBf16;
    private final MethodHandle gdnGatedRmsNormBf16;
    private final MethodHandle residualAddBf16;
    private final MethodHandle residualRmsNormBf16;
    private final MethodHandle selectExactNumerics;
    /// Mirrors the native process-wide exact-numerics selection.
    private volatile boolean exactNumerics;
    private final MethodHandle gdnProjectControlFp32;
    private final MethodHandle swiGluBf16;
    private final MethodHandle dflashLinearBf16;
    private final MethodHandle dflashRmsNormBf16;
    private final MethodHandle dflashConvBf16;
    private final MethodHandle dflashContextKvBf16;
    private final MethodHandle dflashBlockQkBf16;
    private final MethodHandle dflashAttentionBf16;
    private final MethodHandle dflashSwiGluBf16;
    private final MethodHandle dflashTopKBf16;
    private final MethodHandle dflashSelectBf16;
    private final MethodHandle argmaxBf16;
    private final MethodHandle zeroDeviceMemory;
    private final MethodHandle tableLaunch;
    private final MethodHandle tableKernelCount;
    private final MethodHandle tableKernelName;
    private final MethodHandle attentionQkNormRopeBf16;
    private final MethodHandle attentionKvAppendNvfp4;
    private final MethodHandle attentionCausalNvfp4;
    private final MethodHandle linearQ3P2e2DecodeBf16;
    private final MethodHandle q3P2e2Expand;
    private final MethodHandle embedQ3P2e2;
    private final MethodHandle copyDeviceToDevice2d;
    private final MethodHandle linearNvfp4Bf16;
    private final MethodHandle nvfp4NativeAvailable;
    private final MethodHandle nvfp4ActivationBytes;
    private final MethodHandle linearNvfp4NativeBf16;
    private final MethodHandle nvfp4NativeGateUpSwiGluBf16;
    private final MethodHandle q3MxAvailable;
    private final MethodHandle q3MxScratchBytes;
    private final MethodHandle q3MxSelectSplitRows;
    private final MethodHandle linearQ3MxBf16;
    private final MethodHandle linearQ45MxBf16;
    private final MethodHandle q3MxGateUpSwiGluBf16;
    /// Whether the Q3 block-scaled FP8 prefill module loaded (null until first asked).
    private volatile Boolean q3Mx;

    /// Whether the native Blackwell NVFP4 module loaded (null until first asked).
    private volatile Boolean nvfp4Native;
    /// The stream whose launches the current thread is submitting, or null for synchronous calls.
    private final ThreadLocal<CudaStream> submitting = new ThreadLocal<>();
    /// The scratch region bound for the calling thread's submissions ([#withScratch]): {address, bytes, uses in
    /// progress}. A route that needs scratch takes it; the stages that bind it are ordered by their shape's edges.
    private static final ThreadLocal<long[]> SCRATCH = ThreadLocal.withInitial(() -> new long[3]);
    /// Uses of scratch that found no region bound and took a private stream-ordered one.
    private final java.util.concurrent.atomic.AtomicLong scratchFallbacks =
            new java.util.concurrent.atomic.AtomicLong();
    private volatile boolean closed;

    public CudaGpuMemory(Path libraryPath) {
        Objects.requireNonNull(libraryPath, "libraryPath");
        Arena loadedLibraryArena = Arena.ofShared();
        try {
            SymbolLookup symbols;
            try {
                symbols = SymbolLookup.libraryLookup(libraryPath, loadedLibraryArena);
            } catch (IllegalArgumentException unloadable) {
                // The loader does not say which library failed; when the file exists, it is one the file links.
                throw new GpuMemoryException(
                        "cannot load " + libraryPath + " or a library it links: the CUDA 13 "
                                + "runtime (libcudart, libnvrtc) must be on the library path and the NVIDIA driver (libcuda) "
                                + "installed",
                        unloadable);
            }
            Linker linker = Linker.nativeLinker();
            this.arena = loadedLibraryArena;
            this.malloc = bind(linker, symbols, "euhedral_cuda_malloc", MALLOC);
            this.free = bind(linker, symbols, "euhedral_cuda_free", FREE);
            this.mallocAsync = bind(linker, symbols, "euhedral_cuda_malloc_async", MALLOC);
            this.freeAsync = bind(linker, symbols, "euhedral_cuda_free_async", FREE);
            this.hostMalloc = bind(linker, symbols, "euhedral_cuda_host_malloc", MALLOC);
            this.hostFree = bind(linker, symbols, "euhedral_cuda_host_free", FREE);
            this.hostWeightsMalloc = bind(linker, symbols, "euhedral_cuda_host_weights_malloc", MALLOC);
            this.hostWeightsFree = bind(linker, symbols, "euhedral_cuda_host_weights_free", FREE);
            this.hostWeightsDevicePointer =
                    bind(linker, symbols, "euhedral_cuda_host_weights_device_pointer", DEVICE_MEMORY_INFO);
            this.deviceMemoryInfo = bind(linker, symbols, "euhedral_cuda_device_memory_info", DEVICE_MEMORY_INFO);
            this.copyHostToDevice = bind(linker, symbols, "euhedral_cuda_copy_host_to_device", COPY);
            this.copyUploadToDevice = bind(linker, symbols, "euhedral_cuda_copy_upload_to_device", COPY);
            this.copyDeviceToHost = bind(linker, symbols, "euhedral_cuda_copy_device_to_host", COPY);
            this.copyDeviceToReadback = bind(linker, symbols, "euhedral_cuda_copy_device_to_readback", COPY);
            this.copyDeviceToDevice = bind(linker, symbols, "euhedral_cuda_copy_device_to_device", COPY);
            this.embedQ3 = bind(linker, symbols, "euhedral_cuda_embed_q3", EMBED_Q3);
            this.embedQ3P2e2 = bind(linker, symbols, "euhedral_cuda_embed_q3_p2e2", EMBED_Q3);
            this.linearQ3P2e2DecodeBf16 =
                    bind(linker, symbols, "euhedral_cuda_linear_q3_p2e2_decode_bf16", LINEAR_Q3_BF16);
            this.q3P2e2Expand = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_q3_p2e2_expand",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG));
            this.linearNvfp4Bf16 = bind(linker, symbols, "euhedral_cuda_linear_nvfp4_bf16", LINEAR_Q3_BF16);
            this.nvfp4NativeAvailable = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_nvfp4_native_available",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT));
            this.nvfp4ActivationBytes = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_nvfp4_native_scratch_bytes",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.linearNvfp4NativeBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_linear_nvfp4_native_bf16",
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
                            ValueLayout.JAVA_LONG));
            this.nvfp4NativeGateUpSwiGluBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16",
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
                            ValueLayout.JAVA_LONG));
            this.q3MxAvailable =
                    bind(linker, symbols, "euhedral_cuda_q3_mx_available", FunctionDescriptor.of(ValueLayout.JAVA_INT));
            this.q3MxSelectSplitRows = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_q3_mx_select_split_rows",
                    FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
            this.q3MxScratchBytes = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_q3_mx_scratch_bytes",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            FunctionDescriptor q3MxDescriptor = FunctionDescriptor.of(
                    ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_INT,
                    ValueLayout.JAVA_LONG,
                    ValueLayout.JAVA_LONG);
            this.linearQ3MxBf16 = bind(linker, symbols, "euhedral_cuda_linear_q3_mx_bf16", q3MxDescriptor);
            this.linearQ45MxBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_linear_q45_mx_bf16",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG));
            this.q3MxGateUpSwiGluBf16 =
                    bind(linker, symbols, "euhedral_cuda_q3_mx_gate_up_swiglu_bf16", q3MxDescriptor);
            this.copyDeviceToDevice2d = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_copy_device_to_device_2d",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG,
                            ValueLayout.JAVA_LONG));
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
            this.rowExactSelect = bind(
                    linker, symbols, "euhedral_cuda_row_exact_select", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
            this.pdlSelect =
                    bind(linker, symbols, "euhedral_cuda_pdl_select", FunctionDescriptor.ofVoid(ValueLayout.JAVA_INT));
            this.streamClear = bind(linker, symbols, "euhedral_cuda_stream_clear", FunctionDescriptor.ofVoid());
            this.submissionRecord = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_submission_record",
                    FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
            this.submissionCheck = bind(linker, symbols, "euhedral_cuda_submission_check", FunctionDescriptor.ofVoid());
            this.submissionFinish = bind(
                    linker, symbols, "euhedral_cuda_submission_finish", FunctionDescriptor.of(ValueLayout.JAVA_LONG));
            this.submissionMarkUnrecordable =
                    bind(linker, symbols, "euhedral_cuda_submission_mark_unrecordable", FunctionDescriptor.ofVoid());
            this.submissionCheckBegin = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_submission_check_begin",
                    FunctionDescriptor.of(ValueLayout.JAVA_LONG));
            this.submissionCheckEnd = bind(
                    linker, symbols, "euhedral_cuda_submission_check_end", FunctionDescriptor.of(ValueLayout.JAVA_INT));
            this.graphCaptureBegin = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_graph_capture_begin",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
            this.graphCaptureEnd = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_graph_capture_end",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.ADDRESS));
            this.graphLaunch = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_graph_launch",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            this.graphDestroy = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_graph_destroy",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG));
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
            this.linearQ3ReferenceBf16 =
                    bind(linker, symbols, "euhedral_cuda_linear_q3_reference_bf16", LINEAR_Q3_BF16);
            this.linearQuantizedBf16 =
                    bind(linker, symbols, "euhedral_cuda_linear_quantized_bf16", LINEAR_QUANTIZED_BF16);
            this.linearBf16ToFloat = bind(linker, symbols, "euhedral_cuda_linear_bf16_to_float", LINEAR_BF16_TO_FLOAT);
            this.gdnControlFp32 = bind(linker, symbols, "euhedral_cuda_gdn_control_fp32", GDN_CONTROL_FP32);
            this.gdnConvolutionBf16 = bind(linker, symbols, "euhedral_cuda_gdn_convolution_bf16", GDN_CONVOLUTION_BF16);
            this.gdnRecurrenceBf16 = bind(linker, symbols, "euhedral_cuda_gdn_recurrence_bf16", GDN_RECURRENCE_BF16);
            this.gdnGatedRmsNormBf16 =
                    bind(linker, symbols, "euhedral_cuda_gdn_gated_rms_norm_bf16", GDN_GATED_RMS_NORM_BF16);
            this.selectExactNumerics = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_select_exact_numerics",
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
            this.residualRmsNormBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_residual_rms_norm_bf16",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_FLOAT));
            this.gdnProjectControlFp32 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_gdn_project_control_fp32",
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
                            ValueLayout.JAVA_INT));
            this.residualAddBf16 = bind(linker, symbols, "euhedral_cuda_residual_add_bf16", RESIDUAL_ADD_BF16);
            this.swiGluBf16 = bind(linker, symbols, "euhedral_cuda_swiglu_bf16", SWIGLU_BF16);
            ValueLayout.OfInt i32 = ValueLayout.JAVA_INT;
            ValueLayout.OfFloat f32 = ValueLayout.JAVA_FLOAT;
            java.lang.foreign.AddressLayout p = ValueLayout.ADDRESS;
            this.dflashLinearBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_linear_bf16",
                    FunctionDescriptor.of(i32, p, p, p, i32, i32, i32));
            this.dflashRmsNormBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_rms_norm_bf16",
                    FunctionDescriptor.of(i32, p, p, p, i32, i32, f32));
            this.dflashConvBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_conv_bf16",
                    FunctionDescriptor.of(i32, p, p, p, p, i32, i32, i32, i32, i32));
            this.dflashContextKvBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_context_kv_bf16",
                    FunctionDescriptor.of(i32, p, p, p, p, i32, p, i32, i32, i32, f32, f32));
            this.dflashBlockQkBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_block_qk_bf16",
                    FunctionDescriptor.of(i32, p, p, p, p, p, p, i32, p, i32, i32, i32, f32, f32));
            this.dflashAttentionBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_attention_bf16",
                    FunctionDescriptor.of(i32, p, p, p, p, p, p, p, i32, p, i32, i32, i32, i32));
            this.dflashSwiGluBf16 = bind(
                    linker, symbols, "euhedral_cuda_dflash_swiglu_bf16", FunctionDescriptor.of(i32, p, p, i32, i32));
            this.dflashTopKBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_topk_bf16",
                    FunctionDescriptor.of(i32, p, i32, i32, p, p, p));
            this.dflashSelectBf16 = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_dflash_select_bf16",
                    FunctionDescriptor.of(i32, p, p, p, p, p, p, i32, i32, p, p));
            this.argmaxBf16 = bind(linker, symbols, "euhedral_cuda_argmax_bf16", ARGMAX_BF16);
            this.zeroDeviceMemory = bind(linker, symbols, "euhedral_cuda_zero_device_memory", ZERO_DEVICE_MEMORY);
            this.tableLaunch = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_qwen4_launch",
                    FunctionDescriptor.of(
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.JAVA_INT,
                            ValueLayout.ADDRESS,
                            ValueLayout.ADDRESS,
                            ValueLayout.JAVA_INT));
            this.tableKernelCount = bind(
                    linker, symbols, "euhedral_cuda_qwen4_kernel_count", FunctionDescriptor.of(ValueLayout.JAVA_INT));
            this.tableKernelName = bind(
                    linker,
                    symbols,
                    "euhedral_cuda_qwen4_kernel_name",
                    FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
            this.attentionQkNormRopeBf16 =
                    bind(linker, symbols, "euhedral_cuda_attention_qk_norm_rope_bf16", ATTENTION_QK_NORM_ROPE_BF16);
            this.attentionKvAppendNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_kv_append_nvfp4", ATTENTION_KV_APPEND_NVFP4);
            this.attentionCausalNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_causal_nvfp4", ATTENTION_CAUSAL_NVFP4);
            if (!q3MxAvailable() || !nvfp4NativeAvailable())
                throw new GpuMemoryException("an NVIDIA Blackwell GPU (compute capability 12.x) is required");
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
                if (status == 0 && overlapPredecessor) pdlSelect.invokeExact(1);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission stream selection invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA submission stream selection", status);
        }

        private void clear(boolean overlapPredecessor) {
            try {
                if (overlapPredecessor) pdlSelect.invokeExact(0);
                streamClear.invokeExact();
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission stream clear invocation failed", failure);
            }
        }

        @Override
        public boolean capturesGraphs() {
            return true;
        }

        @Override
        public long submitRecording(
                Runnable launches, boolean overlapPredecessor, GpuStream shadow, boolean shadowOverlap) {
            ensureOpen();
            if (!(shadow instanceof CudaStream capture))
                throw new IllegalArgumentException("shadow is not a CUDA stream");
            RECORDING_SHADOW.set(capture);
            select(overlapPredecessor);
            CudaStream previous = submitting.get();
            submitting.set(this);
            int[] mode = SUBMISSION.get();
            try {
                submissionRecord.invokeExact(capture.handle, shadowOverlap ? 1 : 0);
            } catch (Throwable failure) {
                RECORDING_SHADOW.remove();
                submitting.set(previous);
                clear(overlapPredecessor);
                throw new GpuMemoryException("CUDA submission recording invocation failed", failure);
            }
            mode[0] = RECORDING;
            long hash;
            try {
                launches.run();
            } finally {
                mode[0] = 0;
                RECORDING_SHADOW.remove();
                hash = finishSubmission();
                submitting.set(previous);
                clear(overlapPredecessor);
            }
            return hash;
        }

        @Override
        public long beginChecking() {
            ensureOpen();
            long sink;
            try {
                sink = (long) submissionCheckBegin.invokeExact();
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission check invocation failed", failure);
            }
            if (sink == 0) throw new GpuMemoryException("CUDA submission check could not start capturing its sink");
            return sink;
        }

        @Override
        public long submitChecking(Runnable launches, long sink) {
            ensureOpen();
            int status;
            try {
                status = (int) streamSelect.invokeExact(sink);
                if (status == 0) submissionCheck.invokeExact();
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission check invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA submission check stream selection", status);
            CudaStream previous = submitting.get();
            submitting.set(this);
            int[] mode = SUBMISSION.get();
            mode[0] = CHECKING;
            long hash;
            try {
                launches.run();
            } finally {
                mode[0] = 0;
                hash = finishSubmission();
                submitting.set(previous);
                clear(false);
            }
            return hash;
        }

        @Override
        public boolean endChecking() {
            try {
                return (int) submissionCheckEnd.invokeExact() == 0;
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA submission check invocation failed", failure);
            }
        }

        @Override
        public void beginCapture() {
            ensureOpen();
            int status;
            try {
                status = (int) graphCaptureBegin.invokeExact(this.handle);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA graph capture invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA graph capture", status);
        }

        @Override
        public long endCapture() {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment exec = arena.allocate(ValueLayout.JAVA_LONG);
                int status = (int) graphCaptureEnd.invokeExact(this.handle, exec);
                if (status != 0) {
                    LOG.debug("CUDA graph capture ended with status {}", status);
                    return 0;
                }
                return exec.get(ValueLayout.JAVA_LONG, 0);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA graph capture invocation failed", failure);
            }
        }

        @Override
        public boolean launchGraph(long graph) {
            ensureOpen();
            launch(graph);
            return true;
        }

        private void launch(long graph) {
            int status;
            try {
                status = (int) graphLaunch.invokeExact(graph, this.handle);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA graph launch invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA graph launch", status);
        }

        @Override
        public void destroyGraph(long graph) {
            int status;
            try {
                status = (int) graphDestroy.invokeExact(graph);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA graph destruction invocation failed", failure);
            }
            if (status != 0) LOG.warn("CUDA graph destruction failed with status {}", status);
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
            track(value, byteSize);
            return value;
        } catch (GpuMemoryException exception) {
            throw exception;
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA allocation invocation failed", throwable);
        }
    }

    @Override
    public long allocateAsync(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        // A captured or checked submission cannot take a stream-ordered allocation: it becomes ordinary.
        if (SUBMISSION.get()[0] != 0) {
            markUnrecordable();
            return allocate(byteSize);
        }
        MemorySegment address;
        try {
            address = (MemorySegment) this.mallocAsync.invokeExact(byteSize);
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA stream-ordered allocation invocation failed", throwable);
        }
        long value = address.address();
        if (value == 0) throw new GpuMemoryException("CUDA stream-ordered allocation returned a null address");
        track(value, byteSize);
        this.streamOrdered.add(value);
        return value;
    }

    @Override
    public void freeAsync(long address) {
        ensureOpen();
        if (address == 0) return;
        boolean recordable = SUBMISSION.get()[0] == 0;
        if (!recordable) markUnrecordable();
        if (!this.streamOrdered.remove(address) || !recordable) {
            free(address);
            return;
        }
        int status;
        try {
            status = (int) this.freeAsync.invokeExact(MemorySegment.ofAddress(address));
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA stream-ordered free invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("CUDA stream-ordered free", status);
        untrack(address);
    }

    private void track(long address, long byteSize) {
        allocationSizes.put(address, byteSize);
        allocationIds.put(address, allocationSerial.incrementAndGet());
        long live = allocatedBytes.addAndGet(byteSize);
        peakAllocatedBytes.accumulateAndGet(live, Math::max);
    }

    private void untrack(long address) {
        Long size = allocationSizes.remove(address);
        allocationIds.remove(address);
        if (size != null) allocatedBytes.addAndGet(-size);
    }

    /// Pinned staging memory is page-locked, so allocating it per quantum costs about a millisecond.
    /// Small buffers are retained for reuse; larger ones are freed on release. A quantum holds its
    /// token staging and, when its KV reservation grows, one page-table staging per full-attention
    /// layer until it retires, so the cache covers a growing quantum without allocating.
    private static final long PINNED_CACHE_BYTES = 64 * 1024;

    private static final int PINNED_CACHE_ENTRIES = 32;

    private long finishSubmission() {
        try {
            return (long) submissionFinish.invokeExact();
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA submission finish invocation failed", failure);
        }
    }

    /// While a thread records: its shadow, and the recording's order of shared-scratch uses.
    private static final ThreadLocal<CudaStream> RECORDING_SHADOW = new ThreadLocal<>();

    private void markUnrecordable() {
        try {
            submissionMarkUnrecordable.invokeExact();
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA submission invocation failed", failure);
        }
    }

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
    public long hostWeightsDeviceAddress(long hostAddress) {
        ensureOpen();
        try (Arena queryArena = Arena.ofConfined()) {
            MemorySegment device = queryArena.allocate(Long.BYTES, Long.BYTES);
            int status;
            try {
                status = (int) hostWeightsDevicePointer.invokeExact(MemorySegment.ofAddress(hostAddress), device);
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA host weight mapping invocation failed", failure);
            }
            if (status != 0) throw new GpuMemoryException("CUDA host weight mapping", status);
            return device.get(ValueLayout.JAVA_LONG, 0);
        }
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
        allocationIds.put(allocation.address(), allocationSerial.incrementAndGet());
        return new ReadbackBuffer(allocation, () -> {
            allocationIds.remove(allocation.address());
            freePinned(allocation);
        });
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
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        if (rows > Q3_DECODE_MAX_ROWS
                && invokeQ3Mx(
                        "Q3 FP8 linear",
                        this.linearQ3MxBf16,
                        inputAddress,
                        weightsAddress,
                        outputAddress,
                        rows,
                        inFeatures,
                        outFeatures,
                        weightsByteSize)) return;
        invokeQ3(
                this.linearQ3Bf16,
                inputAddress,
                weightsAddress,
                outputAddress,
                rows,
                inFeatures,
                outFeatures,
                weightsByteSize);
    }

    /// The scalar reference kernel: the numerical oracle for the Q3 kernels, not a production route.
    public void referenceLinearQ3Bf16(
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        invokeQ3(
                this.linearQ3ReferenceBf16,
                inputAddress,
                weightsAddress,
                outputAddress,
                rows,
                inFeatures,
                outFeatures,
                weightsByteSize);
    }

    private void invokeQ3(
            MethodHandle kernel,
            long inputAddress,
            long weightsAddress,
            long outputAddress,
            int rows,
            int inFeatures,
            int outFeatures,
            long weightsByteSize) {
        int status;
        try {
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

    /// [#linearQ3Bf16(long, long, long, int, int, int, long)] for either Q3 layout. A
    /// P2E2 tensor runs up to eight rows on its own decode kernels, bitwise identical to the row-split contiguous
    /// kernels, where those kernels would run; every other route expands it into the shared scratch and
    /// runs the row-split route on the expansion, so the outputs are the same bits in every case.
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
        if (Objects.requireNonNull(layout, "layout") != WeightLayout.ROW_SPLIT_P2E2_V1) {
            requireRowSplit(layout, "Q3 linear");
            linearQ3Bf16(inputAddress, weightsAddress, outputAddress, rows, inFeatures, outFeatures, weightsByteSize);
            return;
        }
        ensureOpen();
        requireAddresses(inputAddress, weightsAddress, outputAddress);
        if (rows <= 0 || inFeatures <= 0 || outFeatures <= 0 || weightsByteSize <= 0) {
            throw new IllegalArgumentException("Q3 linear dimensions and payload size must be positive");
        }
        if (rows <= Q3_DECODE_MAX_ROWS) {
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
            long activations = q3MxReserveBytes(rows, inFeatures, outFeatures),
                    activationOffset = alignUp(expanded, 256);
            withQ3Scratch(activationOffset + activations, scratch -> {
                expandQ3(weightsAddress, weightsByteSize, outFeatures, inFeatures, 0, outFeatures, scratch, expanded);
                withQ3MxReserved(
                        scratch + activationOffset,
                        activations,
                        () -> linearQ3Bf16(
                                inputAddress, scratch, outputAddress, rows, inFeatures, outFeatures, expanded));
            });
            return;
        }
        // Output-row chunks, each written to the scratch and copied into place. Every Q3 linear kernel
        // computes an output column from its own weight row alone, and the chunk widths avoid the
        // shapes that select a shape-specific kernel, so the outputs equal those of the whole tensor.
        int chunk = linearChunkRows(rows, inFeatures);
        // The FP8 route's split-K choice, hence its summation order, follows the whole tensor, not the chunk.
        selectQ3MxSplitRows(outFeatures);
        try {
            for (int first = 0; first < outFeatures; first += chunk) {
                int firstRow = first;
                int count = Math.min(chunk, outFeatures - first);
                long weights = P2e2Layout.expandedByteSize(count, inFeatures);
                long outputOffset = alignUp(weights, 256);
                long activationOffset = alignUp(outputOffset + (long) rows * count * Short.BYTES, 256);
                long activations = q3MxReserveBytes(rows, inFeatures, count);
                withQ3Scratch(activationOffset + activations, scratch -> {
                    expandQ3(
                            weightsAddress,
                            weightsByteSize,
                            outFeatures,
                            inFeatures,
                            firstRow,
                            count,
                            scratch,
                            weights);
                    withQ3MxReserved(
                            scratch + activationOffset,
                            activations,
                            () -> linearQ3Bf16(
                                    inputAddress, scratch, scratch + outputOffset, rows, inFeatures, count, weights));
                    copyRows(
                            outputAddress + (long) firstRow * Short.BYTES,
                            (long) outFeatures * Short.BYTES,
                            scratch + outputOffset,
                            (long) count * Short.BYTES,
                            rows);
                });
            }
        } finally {
            selectQ3MxSplitRows(0);
        }
    }

    /// Names the weight rows of the whole tensor for the FP8 route's split choice on this thread (0 clears it).
    private void selectQ3MxSplitRows(int weightRows) {
        try {
            this.q3MxSelectSplitRows.invokeExact(weightRows);
        } catch (Throwable failure) {
            throw new GpuMemoryException("Q3 FP8 split selection invocation failed", failure);
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

    @Override
    public void q3GateUpSwiGluBf16(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        q3GateUpSwiGluBf16(input, weights, output, rows, width, outputs, weightBytes, WeightLayout.ROW_SPLIT_K128_V1);
    }

    /// Runs `use` with scratch of at least `bytes`: the region bound for this thread's submissions
    /// ([#withScratch]), whose uses the binding stages' edges order; or, for a route its stage did not declare (an
    /// unusual fallback, a wider quantum), a private region taken and returned in stream order on the submitting
    /// lane. A nested use (a composition around its own linear) gets the same bound region.
    private void withQ3Scratch(long bytes, LongConsumer use) {
        ensureOpen();
        long[] bound = SCRATCH.get();
        if (bound[0] != 0 && bound[1] >= bytes) {
            bound[2]++;
            try {
                use.accept(bound[0]);
            } finally {
                bound[2]--;
            }
            return;
        }
        this.scratchFallbacks.incrementAndGet();
        // A captured quantum cannot keep a private region's address.
        if (SUBMISSION.get()[0] == RECORDING) markUnrecordable();
        long region = allocateAsync(alignUp(bytes, 256));
        long previousAddress = bound[0], previousBytes = bound[1], previousDepth = bound[2];
        bound[0] = region;
        bound[1] = bytes;
        bound[2] = previousDepth + 1;
        try {
            use.accept(region);
        } finally {
            bound[0] = previousAddress;
            bound[1] = previousBytes;
            bound[2] = previousDepth;
            freeAsync(region);
        }
    }

    @Override
    public void withScratch(long address, long bytes, Runnable submit) {
        long[] bound = SCRATCH.get();
        long previousAddress = bound[0], previousBytes = bound[1];
        bound[0] = address;
        bound[1] = bytes;
        try {
            submit.run();
        } finally {
            bound[0] = previousAddress;
            bound[1] = previousBytes;
        }
    }

    /// Uses of scratch that found no bound region large enough, since this binding opened.
    public long scratchFallbacks() {
        return this.scratchFallbacks.get();
    }

    @Override
    public long scratchBytes(ScratchUse use, int rows, int inFeatures, int outFeatures, WeightLayout layout) {
        ensureOpen();
        boolean p2e2 = layout == WeightLayout.ROW_SPLIT_P2E2_V1;
        return switch (use) {
            case Q3_LINEAR ->
                p2e2 ? p2e2LinearScratch(rows, inFeatures, outFeatures) : mxScratch(rows, inFeatures, outFeatures);
            case MX_LINEAR -> mxScratch(rows, inFeatures, outFeatures);
            case Q3_GATE_UP -> {
                long composition = (long) rows * outFeatures * Short.BYTES;
                if (!p2e2) yield Math.max(composition, mxScratch(rows, inFeatures, outFeatures));
                long expanded = P2e2Layout.expandedByteSize(outFeatures, inFeatures);
                long fused = q3MxReserveBytes(rows, inFeatures, outFeatures);
                yield Math.max(composition, alignUp(expanded, 256) + (fused > 0 ? fused : composition));
            }
            case NVFP4_LINEAR -> rows >= NVFP4_NATIVE_MIN_ROWS ? nvfp4Activations(rows, inFeatures, outFeatures) : 0;
            case NVFP4_GATE_UP ->
                Math.max((long) rows * outFeatures * Short.BYTES, nvfp4Activations(rows, inFeatures, outFeatures));
        };
    }

    private long mxScratch(int rows, int inFeatures, int outFeatures) {
        if (rows < Q3_MX_MIN_ROWS || inFeatures % 128 != 0 || outFeatures % 128 != 0 || !q3MxAvailable()) return 0;
        return q3MxScratchBytes(rows, inFeatures, outFeatures);
    }

    private long p2e2LinearScratch(int rows, int inFeatures, int outFeatures) {
        if (rows <= Q3_DECODE_MAX_ROWS) return 0;
        long expanded = P2e2Layout.expandedByteSize(outFeatures, inFeatures);
        if (expanded <= Q3_SCRATCH_LIMIT)
            return alignUp(expanded, 256) + q3MxReserveBytes(rows, inFeatures, outFeatures);
        int chunk = linearChunkRows(rows, inFeatures);
        long weights = P2e2Layout.expandedByteSize(chunk, inFeatures);
        long outputOffset = alignUp(weights, 256);
        long activationOffset = alignUp(outputOffset + (long) rows * chunk * Short.BYTES, 256);
        return activationOffset + q3MxReserveBytes(rows, inFeatures, chunk);
    }

    private long nvfp4Activations(int rows, int inFeatures, int outFeatures) {
        if (!nvfp4NativeAvailable()) return 0;
        try {
            return (long) this.nvfp4ActivationBytes.invokeExact(rows, inFeatures, outFeatures);
        } catch (Throwable failure) {
            throw new GpuMemoryException("NVFP4 activation size invocation failed", failure);
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

    /// Rows from which an NVFP4 linear runs on native FP4 tensor cores: up to 8 rows stay on the decode kernels,
    /// which stream weights as fast, keep BF16 activations and are row-exact (docs/NVFP4_NATIVE.md).
    static final int NVFP4_NATIVE_MIN_ROWS = 9;
    /// The paired gate/up region exists only in region views, from 64 rows.
    static final int NVFP4_NATIVE_REGION_MIN_ROWS = 64;

    /// Rows from which a Q3, Q4 or Q5 linear runs on the block-scaled FP8 route (docs/PREFILL_MX.md); smaller
    /// quanta (speculative verification, drafting and prompt tails) run the decode kernels.
    static final int Q3_MX_MIN_ROWS = Q3_DECODE_MAX_ROWS + 1;

    /// Whether the block-scaled FP8 prefill kernels are available on this device (an sm_12x GPU).
    public boolean q3MxAvailable() {
        Boolean available = this.q3Mx;
        if (available == null) {
            ensureOpen();
            try {
                available = (int) this.q3MxAvailable.invokeExact() != 0;
            } catch (Throwable failure) {
                throw new GpuMemoryException("Q3 FP8 prefill availability invocation failed", failure);
            }
            this.q3Mx = available;
        }
        return available;
    }

    /// Runs `kernel` (a Q3 FP8 prefill route) with its quantized-activation scratch. False when the route
    /// declined (exact numerics, odd shapes), in which case nothing ran.
    private boolean invokeQ3Mx(
            String operation,
            MethodHandle kernel,
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes) {
        return invokeQ3Mx(operation, kernel, 3, input, weights, output, rows, width, outputs, weightBytes);
    }

    /// `bits` 3 runs `kernel`; 4 and 5 run the Q4/Q5 linear.
    private boolean invokeQ3Mx(
            String operation,
            MethodHandle kernel,
            int bits,
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes) {
        if (rows < Q3_MX_MIN_ROWS || ROW_EXACT.get()[0] || this.exactNumerics || !q3MxAvailable()) return false;
        if (width % 128 != 0 || outputs % 128 != 0) return false;
        ensureOpen();
        requireAddresses(input, weights, output);
        long scratchBytes = q3MxScratchBytes(rows, width, outputs);
        // Inside a P2E2 expansion the shared scratch holds the expanded weights; the expansion reserved a region
        // behind them for the activations (withExpandedWeights).
        if (SCRATCH.get()[2] > 0) {
            long[] reserved = q3MxReserved.get();
            if (reserved == null || reserved[1] < scratchBytes) return false;
            return runQ3Mx(
                    operation,
                    kernel,
                    bits,
                    input,
                    weights,
                    output,
                    rows,
                    width,
                    outputs,
                    weightBytes,
                    reserved[0],
                    scratchBytes);
        }
        boolean[] ran = {false};
        withQ3Scratch(
                scratchBytes,
                scratch -> ran[0] = runQ3Mx(
                        operation,
                        kernel,
                        bits,
                        input,
                        weights,
                        output,
                        rows,
                        width,
                        outputs,
                        weightBytes,
                        scratch,
                        scratchBytes));
        return ran[0];
    }

    private boolean runQ3Mx(
            String operation,
            MethodHandle kernel,
            int bits,
            long input,
            long weights,
            long output,
            int rows,
            int width,
            int outputs,
            long weightBytes,
            long scratch,
            long scratchBytes) {
        int status;
        try {
            status = bits == 3
                    ? (int) kernel.invokeExact(
                            MemorySegment.ofAddress(input),
                            MemorySegment.ofAddress(weights),
                            MemorySegment.ofAddress(output),
                            MemorySegment.ofAddress(scratch),
                            rows,
                            width,
                            outputs,
                            weightBytes,
                            scratchBytes)
                    : (int) this.linearQ45MxBf16.invokeExact(
                            bits,
                            MemorySegment.ofAddress(input),
                            MemorySegment.ofAddress(weights),
                            MemorySegment.ofAddress(output),
                            MemorySegment.ofAddress(scratch),
                            rows,
                            width,
                            outputs,
                            weightBytes,
                            scratchBytes);
        } catch (Throwable throwable) {
            throw new GpuMemoryException(operation + " invocation failed", throwable);
        }
        if (status == ROUTE_UNAVAILABLE) return false;
        if (status != 0) throw new GpuMemoryException(q3Operation(operation, status), status);
        return true;
    }

    /// Bytes of quantized-activation scratch the FP8 route needs for `rows` rows of `width` values.
    private long q3MxScratchBytes(int rows, int width, int outputs) {
        try {
            return (long) this.q3MxScratchBytes.invokeExact(rows, width, outputs);
        } catch (Throwable failure) {
            throw new GpuMemoryException("Q3 FP8 scratch size invocation failed", failure);
        }
    }

    /// The activation region reserved behind P2E2-expanded weights for the calling thread: {address, bytes}.
    private final ThreadLocal<long[]> q3MxReserved = new ThreadLocal<>();

    /// Bytes to reserve behind expanded weights for an FP8 route over `rows` rows of `width` values, or zero
    /// when that route will not run.
    private long q3MxReserveBytes(int rows, int width, int outputs) {
        if (rows < Q3_MX_MIN_ROWS || ROW_EXACT.get()[0] || this.exactNumerics || width % 128 != 0 || !q3MxAvailable())
            return 0;
        return q3MxScratchBytes(rows, width, outputs);
    }

    /// Runs `use` with the activation region [`address`, `address + bytes`) reserved for the FP8 route of the
    /// launches it makes (the caller holds the shared scratch and has placed expanded weights before `address`).
    private void withQ3MxReserved(long address, long bytes, Runnable use) {
        if (bytes == 0) {
            use.run();
            return;
        }
        long[] previous = q3MxReserved.get();
        q3MxReserved.set(new long[] {address, bytes});
        try {
            use.run();
        } finally {
            q3MxReserved.set(previous);
        }
    }

    /// Whether the native Blackwell NVFP4 kernels are loaded: an sm_12x device.
    public boolean nvfp4NativeAvailable() {
        Boolean available = this.nvfp4Native;
        if (available == null) {
            ensureOpen();
            try {
                available = (int) this.nvfp4NativeAvailable.invokeExact() != 0;
            } catch (Throwable failure) {
                throw new GpuMemoryException("native NVFP4 availability invocation failed", failure);
            }
            this.nvfp4Native = available;
        }
        return available;
    }

    @Override
    public void linearNvfp4Bf16(
            long input, long weights, long output, int rows, int inFeatures, int outFeatures, long weightBytes) {
        boolean route = rows >= NVFP4_NATIVE_MIN_ROWS && !ROW_EXACT.get()[0] && !this.exactNumerics;
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
        ensureOpen();
        requireAddresses(input, weights, output);
        if (rows <= 0 || width <= 0 || outputs <= 0 || outputs % 2 != 0 || weightBytes <= 0)
            throw new IllegalArgumentException("invalid NVFP4 gate/up region dimensions");
        if (rows >= NVFP4_NATIVE_REGION_MIN_ROWS
                && width % 128 == 0
                && !ROW_EXACT.get()[0]
                && !this.exactNumerics
                && invokeNativeNvfp4(
                        "native NVFP4 gate/up SwiGLU region",
                        this.nvfp4NativeGateUpSwiGluBf16,
                        input,
                        weights,
                        output,
                        rows,
                        width,
                        outputs,
                        weightBytes)) return;
        composeGateUp(
                output,
                rows,
                outputs,
                gateUp -> linearNvfp4Bf16(input, weights, gateUp, rows, width, outputs, weightBytes));
    }

    /// The gate/up region as two operations, for the oracle and for shapes the fused route declines: `linear`
    /// writes the rows of [gate | up] into a scratch region, SwiGLU reduces them into `output`.
    private void composeGateUp(long output, int rows, int outputs, LongConsumer linear) {
        long bytes = (long) rows * outputs * Short.BYTES;
        LongConsumer run = gateUp -> {
            linear.accept(gateUp);
            swiGluBf16(gateUp, output, rows, outputs / 2);
        };
        if (SCRATCH.get()[2] > 0) {
            long[] reserved = q3MxReserved.get();
            if (reserved == null || reserved[1] < bytes)
                throw new IllegalStateException("the gate/up composition needs a reserved scratch region");
            run.accept(reserved[0]);
        } else withQ3Scratch(bytes, run);
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
        ensureOpen();
        requireAddresses(input, weights, output);
        if (rows <= 0 || width <= 0 || width % 128 != 0 || outputs <= 0 || outputs % 32 != 0 || weightBytes <= 0)
            throw new IllegalArgumentException("invalid Q3 gate/up region dimensions");
        if (Objects.requireNonNull(layout, "layout") == WeightLayout.ROW_SPLIT_P2E2_V1) {
            if (rows <= Q3_DECODE_MAX_ROWS && !this.exactNumerics) {
                // The decode kernels read the compressed tensor in place; no FP8 region applies to so few rows.
                composeGateUp(
                        output,
                        rows,
                        outputs,
                        gateUp -> linearQ3Bf16(input, weights, gateUp, rows, width, outputs, weightBytes, layout));
                return;
            }
            long expanded = P2e2Layout.expandedByteSize(outputs, width);
            long fused = q3MxReserveBytes(rows, width, outputs);
            long reserve = fused > 0 ? fused : (long) rows * outputs * Short.BYTES;
            long reserveOffset = alignUp(expanded, 256);
            withQ3Scratch(reserveOffset + reserve, scratch -> {
                expandQ3(weights, weightBytes, outputs, width, 0, outputs, scratch, expanded);
                withQ3MxReserved(
                        scratch + reserveOffset,
                        reserve,
                        () -> q3GateUpSwiGluRowSplit(input, scratch, output, rows, width, outputs, expanded));
            });
            return;
        }
        requireRowSplit(layout, "Q3 gate/up SwiGLU region");
        q3GateUpSwiGluRowSplit(input, weights, output, rows, width, outputs, weightBytes);
    }

    /// One row-split gate/up region: the FP8 route fused, or the composition of a linear and SwiGLU.
    private void q3GateUpSwiGluRowSplit(
            long input, long weights, long output, int rows, int width, int outputs, long weightBytes) {
        if (invokeQ3Mx(
                "Q3 FP8 gate/up SwiGLU region",
                this.q3MxGateUpSwiGluBf16,
                input,
                weights,
                output,
                rows,
                width,
                outputs,
                weightBytes)) return;
        composeGateUp(
                output,
                rows,
                outputs,
                gateUp -> linearQ3Bf16(input, weights, gateUp, rows, width, outputs, weightBytes));
    }

    /// Quantizes the activations into the shared scratch, ordered between lanes by its event, and runs
    /// `kernel` on native FP4 tensor cores. False when the route declined (exact numerics).
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
        if (invokeQ3Mx(
                "Q" + bits + " FP8 linear",
                null,
                bits,
                inputAddress,
                weightsAddress,
                outputAddress,
                rows,
                inFeatures,
                outFeatures,
                weightsByteSize)) return;
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
    public void selectRowExact(boolean enabled) {
        try {
            this.rowExactSelect.invokeExact(enabled ? 1 : 0);
        } catch (Throwable failure) {
            throw new GpuMemoryException("row-exact selection invocation failed", failure);
        }
        ROW_EXACT.get()[0] = enabled;
    }

    /// The native Q3, Q4 and Q5 linears run row-exact multi-row twins of their one-row contiguous kernels
    /// (or one row at a time) when row-exact execution is selected.
    @Override
    public boolean rowExactQuantizedLinears() {
        return true;
    }

    @Override
    public boolean exactNumerics() {
        return this.exactNumerics;
    }

    /// Selects exact numerics process-wide: every quantized linear runs its scalar reference and every
    /// relaxed-order operator its exact twin, for later launches. Returns the previous selection. This is the
    /// numerical oracle for comparisons and tests; production never selects it.
    public boolean selectExactNumerics(boolean exact) {
        ensureOpen();
        try {
            boolean previous = (int) selectExactNumerics.invokeExact(exact ? 1 : 0) != 0;
            this.exactNumerics = exact;
            return previous;
        } catch (Throwable failure) {
            throw new IllegalStateException("exact numerics selection failed", failure);
        }
    }

    @Override
    public void residualRmsNormBf16(
            long residual, long delta, long weight, long hidden, long normalized, int rows, int width, float epsilon) {
        ensureOpen();
        requireAddresses(residual, delta, weight, hidden, normalized);
        if (rows <= 0 || width <= 0 || !Float.isFinite(epsilon) || epsilon < 0)
            throw new IllegalArgumentException("invalid residual RMSNorm region dimensions");
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

    private static MemorySegment at(long address) {
        requireDeviceAddress(address);
        return MemorySegment.ofAddress(address);
    }

    @Override
    public void dflashLinearBf16(long input, long weights, long output, int rows, int inFeatures, int outFeatures) {
        ensureOpen();
        invokeLayer(
                "DFlash2 BF16 linear",
                dflashLinearBf16,
                at(input),
                at(weights),
                at(output),
                rows,
                inFeatures,
                outFeatures);
    }

    @Override
    public void dflashRmsNormBf16(long input, long weight, long output, int rows, int width, float epsilon) {
        ensureOpen();
        invokeLayer("DFlash2 RMSNorm", dflashRmsNormBf16, at(input), at(weight), at(output), rows, width, epsilon);
    }

    @Override
    public void dflashConvBf16(
            long input, long dynamic, long base, long output, int rows, int width, int group, int taps, int part) {
        ensureOpen();
        invokeLayer(
                "DFlash2 dynamic convolution",
                dflashConvBf16,
                at(input),
                at(dynamic),
                at(base),
                at(output),
                rows,
                width,
                group,
                taps,
                part);
    }

    @Override
    public void dflashContextKvBf16(
            long kv,
            long keyNorm,
            long ringKeys,
            long ringValues,
            int rows,
            long position,
            int window,
            int keyValueHeads,
            int headDim,
            float epsilon,
            float theta) {
        ensureOpen();
        invokeLayer(
                "DFlash2 context keys",
                dflashContextKvBf16,
                at(kv),
                at(keyNorm),
                at(ringKeys),
                at(ringValues),
                rows,
                at(position),
                window,
                keyValueHeads,
                headDim,
                epsilon,
                theta);
    }

    @Override
    public void dflashBlockQkBf16(
            long query,
            long kv,
            long queryNorm,
            long keyNorm,
            long queryOut,
            long keyOut,
            int rows,
            long position,
            int heads,
            int keyValueHeads,
            int headDim,
            float epsilon,
            float theta) {
        ensureOpen();
        invokeLayer(
                "DFlash2 block queries and keys",
                dflashBlockQkBf16,
                at(query),
                at(kv),
                at(queryNorm),
                at(keyNorm),
                at(queryOut),
                at(keyOut),
                rows,
                at(position),
                heads,
                keyValueHeads,
                headDim,
                epsilon,
                theta);
    }

    @Override
    public void dflashAttentionBf16(
            long query,
            long blockKeys,
            long kv,
            long ringKeys,
            long ringValues,
            long output,
            long scratch,
            int rows,
            long position,
            int window,
            int heads,
            int keyValueHeads,
            int headDim) {
        ensureOpen();
        invokeLayer(
                "DFlash2 attention",
                dflashAttentionBf16,
                at(query),
                at(blockKeys),
                at(kv),
                at(ringKeys),
                at(ringValues),
                at(output),
                at(scratch),
                rows,
                at(position),
                window,
                heads,
                keyValueHeads,
                headDim);
    }

    @Override
    public void dflashSwiGluBf16(long gateUp, long output, int rows, int intermediate) {
        ensureOpen();
        invokeLayer("DFlash2 SwiGLU", dflashSwiGluBf16, at(gateUp), at(output), rows, intermediate);
    }

    @Override
    public void dflashTopKBf16(long logits, int rows, int vocabulary, long values, long indices, long scratch) {
        ensureOpen();
        invokeLayer(
                "DFlash2 top-k", dflashTopKBf16, at(logits), rows, vocabulary, at(values), at(indices), at(scratch));
    }

    @Override
    public void dflashSelectBf16(
            long hidden,
            long values,
            long indices,
            long predecessor,
            long successor,
            long anchor,
            int positions,
            int rank,
            long tokens,
            long scores) {
        ensureOpen();
        invokeLayer(
                "DFlash2 selector",
                dflashSelectBf16,
                at(hidden),
                at(values),
                at(indices),
                at(predecessor),
                at(successor),
                at(anchor),
                positions,
                rank,
                at(tokens),
                at(scores));
    }

    /// Argument staging of [#launchTableKernel]: off-heap word and size arrays, one pair per launching thread.
    private final ThreadLocal<MemorySegment[]> tableStaging = ThreadLocal.withInitial(() -> {
        Arena staging = Arena.ofAuto();
        return new MemorySegment[] {
            staging.allocate(8L * KernelArguments.MAX, 8), staging.allocate(KernelArguments.MAX, 8)
        };
    });

    @Override
    public void launchTableKernel(
            TableKernel kernel,
            int gridX,
            int gridY,
            int gridZ,
            int blockX,
            int blockY,
            int blockZ,
            int sharedBytes,
            KernelArguments arguments) {
        ensureOpen();
        Objects.requireNonNull(kernel, "kernel");
        if (gridX <= 0 || gridY <= 0 || gridZ <= 0 || blockX <= 0 || blockY <= 0 || blockZ <= 0 || sharedBytes < 0)
            throw new IllegalArgumentException("invalid launch geometry for " + kernel);
        MemorySegment[] staging = this.tableStaging.get();
        MemorySegment.copy(arguments.words(), 0, staging[0], ValueLayout.JAVA_LONG, 0, arguments.count());
        MemorySegment.copy(arguments.sizes(), 0, staging[1], ValueLayout.JAVA_BYTE, 0, arguments.count());
        int status;
        try {
            status = (int) this.tableLaunch.invokeExact(
                    kernel.index(),
                    gridX,
                    gridY,
                    gridZ,
                    blockX,
                    blockY,
                    blockZ,
                    sharedBytes,
                    staging[0],
                    staging[1],
                    arguments.count());
        } catch (Throwable throwable) {
            throw new GpuMemoryException("table kernel " + kernel + " invocation failed", throwable);
        }
        if (status != 0) throw new GpuMemoryException("table kernel " + kernel, status);
    }

    /// The native kernel table's names, in table order (a test compares them with the model's [TableKernel] enum).
    public java.util.List<String> tableKernelNames() {
        ensureOpen();
        try {
            int count = (int) this.tableKernelCount.invokeExact();
            java.util.List<String> names = new java.util.ArrayList<>();
            for (int i = 0; i < count; i++) {
                MemorySegment name = (MemorySegment) this.tableKernelName.invokeExact(i);
                names.add(name.reinterpret(256).getString(0));
            }
            return names;
        } catch (Throwable throwable) {
            throw new GpuMemoryException("kernel table query failed", throwable);
        }
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
            long positionAddress,
            float epsilon,
            double ropeTheta) {
        ensureOpen();
        requireAddresses(queryKeyAddress, queryNormAddress, keyNormAddress, outputAddress, positionAddress);
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
                MemorySegment.ofAddress(positionAddress),
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
            long startPosition,
            long positionAddress) {
        ensureOpen();
        requireAddresses(queryKeyAddress, gateValueAddress, keyCacheAddress, valueCacheAddress, positionAddress);
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
                startPosition,
                MemorySegment.ofAddress(positionAddress));
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
            long positionAddress,
            long scratchAddress) {
        ensureOpen();
        requireAddresses(
                queryKeyAddress, gateValueAddress, keyCacheAddress, valueCacheAddress, outputAddress, positionAddress);
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
                MemorySegment.ofAddress(positionAddress),
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
        this.streamOrdered.remove(address);
        untrack(address);
    }

    @Override
    public long allocationId(long address) {
        return allocationIds.getOrDefault(address, 0L);
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
