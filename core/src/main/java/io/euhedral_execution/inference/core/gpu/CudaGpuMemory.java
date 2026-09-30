package io.euhedral_execution.inference.core.gpu;

import io.euhedral_execution.data_structures.queues.MpmcQueue;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;

/// FFM binding for the stable Euhedral CUDA C ABI.
///
/// CUDA device addresses remain opaque longs. They are converted to zero-size address segments only
/// inside the native calls and are never exposed as dereferenceable Java memory.
public final class CudaGpuMemory extends ExecutionGpu implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(CudaGpuMemory.class.getName());
    private static final int MAX_CACHED_EVENTS = 256;
    private final Arena arena;
    private final MethodHandle malloc;
    private final MethodHandle free;
    private final MethodHandle hostMalloc;
    private final MethodHandle hostFree;
    private final MethodHandle deviceMemoryInfo;
    private final MethodHandle copyHostToDevice;
    private final MethodHandle copyUploadToDevice;
    private final MethodHandle copyDeviceToHost;
    private final MethodHandle copyDeviceToDevice;
    private final MethodHandle embedQ3;
    private final MethodHandle synchronize;
    private final MethodHandle streamCreate;
    private final MethodHandle streamDestroy;
    private final MethodHandle streamSynchronize;
    private final MethodHandle streamSelect;
    private final MethodHandle streamClear;
    private final MethodHandle eventCreate;
    private final MethodHandle eventRecord;
    private final MethodHandle eventQuery;
    private final MethodHandle eventDestroy;
    private final MethodHandle completionNotify;
    private final MemorySegment completionCallback;
    private final boolean asynchronous;
    private final long stream;
    private final ConcurrentHashMap<Long, Long> workerStreams = new ConcurrentHashMap<>();
    private final AtomicLong nextWorker = new AtomicLong();
    private final ThreadLocal<StreamSelection> selectedStreams = ThreadLocal.withInitial(StreamSelection::new);
    private final ConcurrentHashMap<Long, Completion> completions = new ConcurrentHashMap<>();
    private final AtomicLong nextCompletion = new AtomicLong();
    private final AtomicReference<Throwable> poisoned = new AtomicReference<>();
    private boolean workerStartupAborted;
    private final MpmcQueue<Long> availableEvents = new MpmcQueue<>(64, 4);
    private volatile Consumer<Runnable> completionSink;
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
    private final MethodHandle q3FfnStreamedBf16;
    private final MethodHandle attentionProducersNvfp4;
    private final MethodHandle gdnProjectControlFp32;
    private final MethodHandle gdnProjectionsBf16;
    private final MethodHandle swiGluBf16;
    private final MethodHandle zeroDeviceMemory;
    private final MethodHandle attentionQkNormRopeBf16;
    private final MethodHandle attentionKvAppendNvfp4;
    private final MethodHandle attentionCausalNvfp4;
    private volatile boolean closed;

    public CudaGpuMemory(Path libraryPath) {
        this(libraryPath, false);
    }

    public CudaGpuMemory(Path libraryPath, boolean asynchronous) {
        this(libraryPath, asynchronous, Q3DispatchMode.AUTO, Q3DispatchMode.DEFAULT_SMALL_ROW_THRESHOLD);
    }

    public CudaGpuMemory(
            Path libraryPath, boolean asynchronous, Q3DispatchMode q3DispatchMode, int q3SmallRowThreshold) {
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
            this.hostMalloc = asynchronous ? bind(linker, symbols, "euhedral_cuda_host_malloc", MALLOC) : null;
            this.hostFree = asynchronous ? bind(linker, symbols, "euhedral_cuda_host_free", FREE) : null;
            this.deviceMemoryInfo = bind(linker, symbols, "euhedral_cuda_device_memory_info", DEVICE_MEMORY_INFO);
            this.copyHostToDevice = bind(linker, symbols, "euhedral_cuda_copy_host_to_device", COPY);
            this.copyUploadToDevice =
                    asynchronous ? bind(linker, symbols, "euhedral_cuda_copy_upload_to_device", COPY) : null;
            this.copyDeviceToHost = bind(linker, symbols, "euhedral_cuda_copy_device_to_host", COPY);
            this.copyDeviceToDevice = bind(linker, symbols, "euhedral_cuda_copy_device_to_device", COPY);
            this.embedQ3 = bind(linker, symbols, "euhedral_cuda_embed_q3", EMBED_Q3);
            this.synchronize = bind(linker, symbols, "euhedral_cuda_synchronize", SYNCHRONIZE);
            this.streamCreate = asynchronous
                    ? bind(linker, symbols, "euhedral_cuda_stream_create", FunctionDescriptor.of(ValueLayout.JAVA_LONG))
                    : null;
            this.streamDestroy = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_stream_destroy",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG))
                    : null;
            this.streamSynchronize = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_stream_synchronize",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG))
                    : null;
            this.streamSelect = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_stream_select",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG))
                    : null;
            this.streamClear = asynchronous
                    ? bind(linker, symbols, "euhedral_cuda_stream_clear", FunctionDescriptor.ofVoid())
                    : null;
            this.eventCreate = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_completion_event_create",
                            FunctionDescriptor.of(ValueLayout.JAVA_LONG))
                    : null;
            this.eventRecord = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_completion_event_record",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG))
                    : null;
            this.eventQuery = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_completion_event_query",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG))
                    : null;
            this.eventDestroy = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_completion_event_destroy",
                            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.JAVA_LONG))
                    : null;
            this.completionNotify = asynchronous
                    ? bind(
                            linker,
                            symbols,
                            "euhedral_cuda_completion_notify",
                            FunctionDescriptor.of(
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_LONG))
                    : null;
            this.completionCallback = asynchronous
                    ? linker.upcallStub(
                            MethodHandles.lookup()
                                    .findVirtual(
                                            CudaGpuMemory.class,
                                            "nativeCompletion",
                                            MethodType.methodType(void.class, long.class, int.class))
                                    .bindTo(this),
                            FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT),
                            loadedLibraryArena)
                    : MemorySegment.NULL;
            this.asynchronous = asynchronous;
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
            this.attentionProducersNvfp4 = symbols.find("euhedral_cuda_attention_producers_nvfp4")
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
                                    ValueLayout.ADDRESS,
                                    ValueLayout.ADDRESS,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_INT,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_FLOAT,
                                    ValueLayout.JAVA_DOUBLE,
                                    ValueLayout.JAVA_LONG,
                                    ValueLayout.JAVA_LONG)))
                    .orElse(null);
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
            this.zeroDeviceMemory = bind(linker, symbols, "euhedral_cuda_zero_device_memory", ZERO_DEVICE_MEMORY);
            this.attentionQkNormRopeBf16 =
                    bind(linker, symbols, "euhedral_cuda_attention_qk_norm_rope_bf16", ATTENTION_QK_NORM_ROPE_BF16);
            this.attentionKvAppendNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_kv_append_nvfp4", ATTENTION_KV_APPEND_NVFP4);
            this.attentionCausalNvfp4 =
                    bind(linker, symbols, "euhedral_cuda_attention_causal_nvfp4", ATTENTION_CAUSAL_NVFP4);
            this.stream = asynchronous ? (long) this.streamCreate.invokeExact() : 0L;
            if (asynchronous && this.stream == 0L) throw new GpuMemoryException("CUDA stream creation failed");
        } catch (RuntimeException exception) {
            loadedLibraryArena.close();
            throw exception;
        } catch (Throwable exception) {
            loadedLibraryArena.close();
            throw new GpuMemoryException("CUDA stream initialization failed", exception);
        }
    }

    @Override
    public boolean asynchronous() {
        return this.asynchronous;
    }

    long workerStream(long worker) {
        return workerStreams.getOrDefault(worker, 0L);
    }

    @Override
    public synchronized long openWorker(int cpu) {
        ensureOpen();
        if (workerStartupAborted) throw new IllegalStateException("CUDA worker startup was aborted");
        if (!asynchronous) return cpu;
        long worker = nextWorker.incrementAndGet();
        if (worker <= 0) throw new IllegalStateException("CUDA worker identifiers exhausted");
        try {
            long created = (long) streamCreate.invokeExact();
            if (created == 0) throw new GpuMemoryException("CUDA worker stream creation failed");
            workerStreams.put(worker, created);
            return worker;
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA worker stream creation invocation failed", failure);
        }
    }

    @Override
    public synchronized void closeWorker(long worker) {
        if (!asynchronous) return;
        Long selected = workerStreams.get(worker);
        if (selected == null) return;
        try {
            int status = (int) streamDestroy.invokeExact(selected.longValue());
            if (status != 0) throw new GpuMemoryException("CUDA worker stream destruction", status);
            workerStreams.remove(worker, selected);
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA worker stream destruction invocation failed", failure);
        }
    }

    @Override
    public synchronized void ensureWorkersClosed() {
        if (!workerStreams.isEmpty()) throw new IllegalStateException("CUDA lattice workers have not closed");
    }

    @Override
    public synchronized void abortWorkerStartup() {
        workerStartupAborted = true;
        // No inference runner exists until load completes. Clones omitted from a partially
        // published shard have no close hook, so retire every pre-admission stream here.
        for (long worker : workerStreams.keySet()) closeWorker(worker);
    }

    @Override
    public void withWorker(long worker, Runnable operation) {
        if (!asynchronous) {
            operation.run();
            return;
        }
        long selected = workerStream(worker);
        if (selected == 0) throw new IllegalStateException("CUDA worker stream was not opened: " + worker);
        StreamSelection selection = selectedStreams.get();
        long previous = selection.worker;
        selection.worker = selected;
        try {
            operation.run();
        } finally {
            selection.worker = previous;
        }
    }

    @Override
    public void poison(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        if (poisoned.compareAndSet(null, failure)) {
            LOG.log(Level.SEVERE, "CUDA recovery failed; retaining GPU allocations until process restart", failure);
        }
        Throwable cause = poisoned.get();
        for (Completion completion : completions.values()) failPoisoned(completion, cause);
    }

    @Override
    public void ensureHealthy() {
        ensureOpen();
    }

    @Override
    public void bindCompletionSink(Consumer<Runnable> completionFrames) {
        if (!asynchronous) throw new IllegalStateException("asynchronous completion is disabled");
        ensureOpen();
        if (this.completionSink != null) throw new IllegalStateException("CUDA completion sink already attached");
        this.completionSink = Objects.requireNonNull(completionFrames, "completionFrames");
    }

    @Override
    public void submit(Runnable operation) {
        ensureOpen();
        if (!asynchronous) {
            operation.run();
            return;
        }
        StreamSelection selection = selectedStreams.get();
        long target = selection.worker == 0 ? stream : selection.worker;
        try {
            int status = (int) streamSelect.invokeExact(target);
            if (status != 0) throw new GpuMemoryException("CUDA submission stream selection", status);
            try {
                selection.submitted = target;
                operation.run();
            } finally {
                streamClear.invokeExact();
            }
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA submission stream invocation failed", failure);
        }
    }

    @Override
    public void prepare(Runnable initialization) {
        submit(initialization);
        if (!asynchronous) return;
        // Admission precedes graph publication. Finish initialization on its own stream
        // before any of the independent worker streams may read its buffers.
        StreamSelection selection = selectedStreams.get();
        long target = selection.submitted;
        selection.submitted = 0;
        try {
            int status = (int) streamSynchronize.invokeExact(target);
            if (status != 0) throw new GpuMemoryException("CUDA initialization stream synchronization", status);
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA initialization stream synchronization failed", failure);
        }
    }

    @Override
    public void deferCompletion(Runnable completed, Consumer<Throwable> failed) {
        ensureOpen();
        if (!asynchronous) throw new IllegalStateException("asynchronous completion is disabled");
        if (completionSink == null) throw new IllegalStateException("CUDA completion sink is not attached");
        long event = 0;
        long token = 0;
        StreamSelection selection = selectedStreams.get();
        long target = selection.submitted == 0 ? stream : selection.submitted;
        selection.submitted = 0;
        try {
            Long cached = availableEvents.poll();
            event = cached == null ? (long) eventCreate.invokeExact() : cached;
            if (event == 0) throw new GpuMemoryException("CUDA event creation returned null");
            int status = (int) eventRecord.invokeExact(event, target);
            if (status != 0) throw new GpuMemoryException("CUDA event record", status);
            token = nextCompletion.incrementAndGet();
            if (token <= 0) throw new IllegalStateException("CUDA completion identifiers exhausted");
            completions.put(token, new Completion(event, completed, failed, new AtomicBoolean()));
            ensureHealthy();
            status = (int) completionNotify.invokeExact(target, completionCallback, token);
            if (status != 0) throw new GpuMemoryException("CUDA completion notification", status);
        } catch (Throwable failure) {
            // Failed event registration cannot prove that a previously launched kernel stopped.
            // This explicit error-recovery barrier precedes workspace/frame cleanup.
            try {
                synchronize();
            } catch (RuntimeException | Error synchronizationFailure) {
                failure.addSuppressed(synchronizationFailure);
                poison(failure);
                Completion retained = token == 0 ? null : completions.get(token);
                if (retained == null) notifyFailed(failed, failure);
                else failPoisoned(retained, failure);
                return;
            }
            if (token != 0) completions.remove(token);
            if (event != 0) destroyEvent(event);
            notifyFailed(failed, failure);
        }
    }

    /// CUDA's host callback only publishes a ready frame; it never calls CUDA or finalizes ownership.
    private void nativeCompletion(long token, int status) {
        if (poisoned.get() != null) return;
        try {
            if (!completions.containsKey(token)) return;
            completionSink.accept(() -> finishCompletion(token, status));
        } catch (Throwable failure) {
            // CUDA host callbacks must not block on finalization or call CUDA. One-shot failure
            // propagation happens off the callback thread and never releases unproven buffers.
            try {
                Thread.ofVirtual().name("euhedral-cuda-failed-publication").start(() -> poison(failure));
            } catch (Throwable threadFailure) {
                failure.addSuppressed(threadFailure);
                poison(failure);
            }
        }
    }

    private void finishCompletion(long token, int callbackStatus) {
        Completion completion = completions.get(token);
        if (completion == null) return;
        if (poisoned.get() != null || completion.finalized().get()) return;
        if (callbackStatus != 0) {
            completeFailed(token, completion, new GpuMemoryException("CUDA asynchronous stream", callbackStatus));
            return;
        }
        int status;
        try {
            status = (int) eventQuery.invokeExact(completion.event());
        } catch (Throwable failure) {
            completeFailed(token, completion, failure);
            return;
        }
        if (status != 0) {
            completeFailed(token, completion, new GpuMemoryException("CUDA completion event", status));
            return;
        }
        if (!completion.finalized().compareAndSet(false, true)) return;
        completions.remove(token);
        if (availableEvents.size() < MAX_CACHED_EVENTS) availableEvents.offer(completion.event());
        else destroyEvent(completion.event());
        try {
            completion.completed().run();
        } catch (Throwable failure) {
            LOG.log(Level.SEVERE, "CUDA completion callback failed after event completion", failure);
        }
    }

    private void completeFailed(long token, Completion completion, Throwable failure) {
        try {
            synchronize();
        } catch (RuntimeException | Error synchronizationFailure) {
            failure.addSuppressed(synchronizationFailure);
            poison(failure);
            return;
        }
        if (!completion.finalized().compareAndSet(false, true)) return;
        completions.remove(token);
        destroyEvent(completion.event());
        notifyFailed(completion.failed(), failure);
    }

    private void failPoisoned(Completion completion, Throwable failure) {
        if (completion.finalized().compareAndSet(false, true)) notifyFailed(completion.failed(), failure);
    }

    private void notifyFailed(Consumer<Throwable> failed, Throwable failure) {
        try {
            failed.accept(failure);
        } catch (Throwable callbackFailure) {
            LOG.log(Level.SEVERE, "CUDA failed-completion callback failed", callbackFailure);
        }
    }

    private void destroyEvent(long event) {
        try {
            int status = (int) eventDestroy.invokeExact(event);
            if (status != 0) LOG.log(Level.WARNING, "CUDA event destruction failed with status {0}", status);
        } catch (Throwable failure) {
            LOG.log(Level.WARNING, "CUDA event destruction failed", failure);
        }
    }

    private record Completion(long event, Runnable completed, Consumer<Throwable> failed, AtomicBoolean finalized) {}

    private static final class StreamSelection {
        private long worker;
        private long submitted;
    }

    @Override
    public long allocate(long byteSize) {
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        try {
            MemorySegment address = (MemorySegment) malloc.invokeExact(byteSize);
            long value = address.address();
            if (value == 0) throw new GpuMemoryException("CUDA allocation returned a null address");
            return value;
        } catch (GpuMemoryException exception) {
            throw exception;
        } catch (Throwable throwable) {
            throw new GpuMemoryException("CUDA allocation invocation failed", throwable);
        }
    }

    @Override
    public UploadBuffer allocateUploadBuffer(long byteSize) {
        if (!asynchronous) return super.allocateUploadBuffer(byteSize);
        ensureOpen();
        if (byteSize <= 0) throw new IllegalArgumentException("byteSize must be positive");
        try {
            MemorySegment address = (MemorySegment) hostMalloc.invokeExact(byteSize);
            if (address.address() == 0) throw new GpuMemoryException("CUDA pinned upload allocation returned null");
            return new UploadBuffer(address.reinterpret(byteSize), () -> freePinnedUpload(address));
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA pinned upload allocation failed", failure);
        }
    }

    @Override
    public void copyUploadToDevice(long destination, UploadBuffer upload) {
        if (!asynchronous) {
            super.copyUploadToDevice(destination, upload);
            return;
        }
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

    private void freePinnedUpload(MemorySegment address) {
        ensureOpen();
        try {
            int status = (int) hostFree.invokeExact(address);
            if (status != 0) throw new GpuMemoryException("CUDA pinned upload free", status);
        } catch (GpuMemoryException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new GpuMemoryException("CUDA pinned upload free invocation failed", failure);
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
    public void attentionProducersNvfp4(
            long input,
            long q4,
            long q5,
            long queryNorm,
            long keyNorm,
            long queryKey,
            long gate,
            long keys,
            long values,
            int rows,
            int hidden,
            int queryHeads,
            int keyHeads,
            int headDim,
            int rotaryDim,
            long start,
            float epsilon,
            double theta,
            long q4Bytes,
            long q5Bytes) {
        ensureOpen();
        requireAddresses(input, q4, q5, queryNorm, keyNorm, queryKey, gate, keys, values);
        if (rows < 64
                || hidden <= 0
                || queryHeads <= 0
                || keyHeads <= 0
                || headDim != 256
                || rotaryDim <= 0
                || start < 0
                || q4Bytes <= 0
                || q5Bytes <= 0) throw new IllegalArgumentException("invalid attention producer geometry");
        if (attentionProducersNvfp4 == null)
            throw new UnsupportedOperationException("native attention producers unavailable");
        invokeLayer(
                "attention producer region",
                attentionProducersNvfp4,
                MemorySegment.ofAddress(input),
                MemorySegment.ofAddress(q4),
                MemorySegment.ofAddress(q5),
                MemorySegment.ofAddress(queryNorm),
                MemorySegment.ofAddress(keyNorm),
                MemorySegment.ofAddress(queryKey),
                MemorySegment.ofAddress(gate),
                MemorySegment.ofAddress(keys),
                MemorySegment.ofAddress(values),
                rows,
                hidden,
                queryHeads,
                keyHeads,
                headDim,
                rotaryDim,
                start,
                epsilon,
                theta,
                q4Bytes,
                q5Bytes);
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
        if (asynchronous) {
            if (!completions.isEmpty()) throw new IllegalStateException("CUDA completion frames did not drain");
            // A completion frame may run before its native host callback returns. Drain the
            // stream before releasing the FFM upcall stub or unloading its library arena.
            synchronize();
            for (Long event; (event = availableEvents.poll()) != null; ) destroyEvent(event);
            try {
                for (var entry : workerStreams.entrySet()) {
                    long workerStream = entry.getValue();
                    int status = (int) streamDestroy.invokeExact(workerStream);
                    if (status != 0) throw new GpuMemoryException("CUDA worker stream destruction", status);
                    workerStreams.remove(entry.getKey(), workerStream);
                }
                int status = (int) streamDestroy.invokeExact(stream);
                if (status != 0) throw new GpuMemoryException("CUDA stream destruction", status);
            } catch (GpuMemoryException failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new GpuMemoryException("CUDA stream destruction invocation failed", failure);
            }
        }
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
