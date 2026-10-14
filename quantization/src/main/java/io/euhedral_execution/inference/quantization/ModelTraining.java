package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.quantization.checkpoint.Overlay;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gptq.QuantState;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import io.euhedral_execution.inference.quantization.lattice.GpuCompletions;
import io.euhedral_execution.inference.quantization.model.ForwardPipeline;
import io.euhedral_execution.inference.quantization.model.LayerForward;
import io.euhedral_execution.inference.quantization.model.LayerWeights;
import io.euhedral_execution.inference.quantization.training.Ops;
import io.euhedral_execution.inference.quantization.training.TrainableLayer;
import java.io.EOFException;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.CompletableFuture;

/// End-to-end distillation of a quantized artifact's scales and norms: every decoder layer's quantized linears
/// train their group scales (codes fixed) and the layers' norms their weights, against the BF16 checkpoint's
/// own output distribution: at every position of the calibration sequences, the KL over its top-k tokens and
/// one bucket for the rest of the vocabulary (the teacher's tail mass is kept, so the student is not pushed to
/// zero probability outside the top k; `top_k_distillation`).
///
/// The teacher runs once: the BF16 layers over every sequence, then the final norm and LM head, keeping each
/// position's top-k log-probabilities. Each step then runs one sequence through all the layers on the training
/// kernels ([TrainableLayer], one of each kind on the device, filled per layer from host memory), keeping each
/// layer's input; takes the KL at the LM head (the checkpoint's, in BF16); and goes back through the layers,
/// recomputing each from its input, so that each layer's scales and norms take their Adam step (their moments
/// kept in host memory with them). Held-out sequences measure the mean KL per position. The result is written
/// as `train-blocks` writes it: a codes directory for `convert_checkpoint.py --codes-from` and an overlay
/// checkpoint with the trained norms.
///
/// A run resumes: the teacher's top-k is kept in `out/resume/teacher.bin`, and every held-out evaluation that
/// follows an update writes the whole training state (each parameter's values and Adam moments, the step, the
/// update count, the best held-out KL) to `out/resume/state`. A run with the same settings and tokens starts
/// from there; delete `out/resume` to start over.
public final class ModelTraining {

    /// Command-line settings. `codes` holds the starting rounding (all objects); `norms`, when set, the
    /// starting norms (a `train-blocks` output's `norms`); otherwise the checkpoint's.
    public record Settings(
            Path model,
            Path codes,
            Path norms,
            Path tokens,
            Path out,
            Path overlay,
            int sequences,
            int holdout,
            int epochs,
            float scaleRate,
            float normRate,
            int topK,
            int evaluateEvery,
            int probe,
            int accumulate,
            boolean tensorCores,
            int batch) {}

    /// Rows of the LM head's logits at a time.
    private static final int HEAD_ROWS = 256;

    private final Settings settings;
    private final Device device;
    private final Qwen38Objects checkpoint;
    /// The training kernels on the main stream, compiled by the run's first frame.
    private Ops ops;
    private final int hidden, vocabulary, layers;
    private final long start = System.nanoTime();
    private final Arena host = Arena.ofShared();
    private final List<Long> pinned = new ArrayList<>();

    /// A trainable parameter in pinned host memory: values and Adam moments.
    private record HostParameter(MemorySegment values, MemorySegment first, MemorySegment second) {}

    /// One quantized object: its codes (pinned), its state as read (format, layout), its trainable scales.
    private record HostLinear(
            String object, String role, int out, int in, MemorySegment codes, QuantState state, HostParameter scales) {}

    /// One decoder layer in host memory.
    private record HostLayer(
            int index,
            boolean attention,
            Map<String, HostLinear> linears,
            Map<String, HostParameter> norms,
            Map<String, MemorySegment> frozen) {}

    private final List<HostLayer> model = new ArrayList<>();

    /// What a resumed run must share with the one it resumes: the data, the starting state and the optimizer.
    private String identity;

    private ModelTraining(Settings settings, Device device, Qwen38Objects checkpoint) {
        this.settings = settings;
        this.device = device;
        this.checkpoint = checkpoint;
        this.hidden = checkpoint.config.hidden();
        this.vocabulary = checkpoint.config.vocabulary();
        this.layers = checkpoint.config.layers();
    }

    /// Starts the run as lattice frames and waits for it: nothing else runs on the calling thread.
    public static void run(Settings settings, Path library) throws Exception {
        try (Safetensors store = Safetensors.open(settings.model());
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(6)) {
            var training = new ModelTraining(settings, device, new Qwen38Objects(store, settings.model()));
            try {
                training.new Run(lattice).start().get();
            } finally {
                training.close();
            }
        }
    }

    private void log(String message) {
        System.out.printf(Locale.ROOT, "[%7.0f s] %s%n", (System.nanoTime() - start) / 1e9, message);
    }

    private void close() {
        for (long address : pinned) device.gpu.freeHostWeights(address);
        host.close();
    }

    /// The pinned chunk slices come from, and how much of it is used.
    private MemorySegment chunk = MemorySegment.NULL;
    private long chunkUsed;

    /// Pinned host memory (asynchronous copies need it), sliced from 1 GiB chunks, 256-byte aligned, zeroed.
    private MemorySegment pinned(long bytes) {
        long size = (bytes + 255) & ~255L;
        if (chunkUsed + size > chunk.byteSize()) {
            long chunkBytes = Math.max(1L << 30, size);
            long address = device.gpu.allocateHostWeights(chunkBytes);
            pinned.add(address);
            chunk = MemorySegment.ofAddress(address).reinterpret(chunkBytes);
            chunkUsed = 0;
        }
        MemorySegment slice = chunk.asSlice(chunkUsed, bytes);
        chunkUsed += size;
        slice.fill((byte) 0);
        return slice;
    }

    // ---- the run ----

    /// The run as a chain of lattice frames, each publishing the next: the setup, one frame per teacher layer
    /// (continued by its pipeline pass), the teacher's head pass (continued by the completion source), the
    /// student's frames ([Steps]), and the output. Only `start` and the caller's wait run outside them.
    private final class Run {
        final CalibrationLattice lattice;
        final CompletableFuture<Void> done = new CompletableFuture<>();
        final FrameSeeds seeds = new FrameSeeds();
        final GpuCompletions completions;
        int[][] tokens;
        int sequences, length;
        MemorySegment embedded, teacherIds, teacherLogProbs, stream;
        long streamAddress, batchStates;
        LayerForward forward;
        ForwardPipeline pipeline;
        Steps steps;

        Run(CalibrationLattice lattice) {
            this.lattice = lattice;
            this.completions = new GpuCompletions(device.cuda);
            lattice.attach(completions);
        }

        CompletableFuture<Void> start() {
            lattice.lake().admit(); // released when the run ends
            lattice.lake().publish(new Begin());
            return done.whenComplete((ignored, failure) -> lattice.lake().terminated());
        }

        /// A frame of the run: its work, or the run's failure.
        abstract class Frame extends io.euhedral_execution.core.frames.AbstractFrame {
            Frame() {
                super(FrameSeeds.ID_HASH);
                randomizeHash(seeds.next());
            }

            abstract void work() throws Exception;

            @Override
            public void execute() {
                if (done.isDone()) return;
                try {
                    device.select(device.stream);
                    work();
                } catch (Throwable failure) {
                    done.completeExceptionally(failure);
                }
            }

            @Override
            public void doFinallyWithError(Throwable failure) {
                done.completeExceptionally(failure);
            }
        }

        void publish(Frame frame) {
            lattice.lake().publish(frame);
        }

        /// Reads the tokens, embeds them, and starts the teacher's first layer.
        final class Begin extends Frame {
            @Override
            void work() throws IOException {
                Files.createDirectories(settings.out());
                ops = new Ops(device.cuda, device.stream, device.cublas);
                tokens = Calibrate.readTokens(settings.tokens(), settings.sequences());
                sequences = tokens.length;
                length = tokens[0].length;
                long positions = (long) sequences * length, sequenceBytes = 2L * hidden * length;
                int k = settings.topK();
                embedded = pinned(positions * 2 * hidden);
                teacherIds = pinned(4 * positions * k);
                teacherLogProbs = pinned(4 * positions * k);
                // The teacher's stream is freed with its pass; only the embedded sequences stay for the student.
                streamAddress = device.gpu.allocateHostWeights(sequenceBytes * sequences);
                stream = MemorySegment.ofAddress(streamAddress).reinterpret(sequenceBytes * sequences);
                long batchRows = (long) settings.batch() * length;
                batchStates = device.gpu.allocate(batchRows * 2 * hidden);
                embed(tokens, embedded, batchStates);
                identity = String.join(
                        " ",
                        "tokens=" + java.util.Arrays.deepHashCode(tokens),
                        "sequences=" + sequences,
                        "length=" + length,
                        "holdout=" + settings.holdout(),
                        "topK=" + k,
                        "accumulate=" + settings.accumulate(),
                        "scaleRate=" + settings.scaleRate(),
                        "normRate=" + settings.normRate(),
                        "codes=" + settings.codes().toAbsolutePath(),
                        "norms="
                                + (settings.norms() == null
                                        ? "-"
                                        : settings.norms().toAbsolutePath()));
                if (readTeacher(teacherIds, teacherLogProbs)) {
                    device.gpu.free(batchStates);
                    device.gpu.freeHostWeights(streamAddress);
                    log("teacher: top-" + k + " of every position read from " + teacherFile());
                    publish(new StudentLayer(0));
                    return;
                }
                MemorySegment.copy(embedded, 0, stream, 0, sequenceBytes * sequences);
                forward = new LayerForward(device, checkpoint.config, (int) batchRows);
                pipeline = new ForwardPipeline(device, forward, lattice.lake(), completions, settings.batch());
                publish(new TeacherLayer(0));
            }
        }

        /// Reads teacher layer `index` (and the student's starting state of it) and runs it over every
        /// sequence; its pipeline pass publishes the next layer's frame.
        final class TeacherLayer extends Frame {
            final int index;

            TeacherLayer(int index) {
                this.index = index;
            }

            @Override
            void work() throws IOException {
                LayerWeights layer;
                try (LayerWeights.Staged staged = LayerWeights.stage(checkpoint, index)) {
                    model.add(hostLayer(index, staged));
                    layer = LayerWeights.upload(device.gpu, staged);
                }
                pipeline.run(layer, streamAddress, batchStates, sequences, length, settings.batch(), null, true)
                        .whenComplete((ignored, failure) -> {
                            layer.close();
                            if (failure != null) done.completeExceptionally(failure);
                            else {
                                if (index % 8 == 7) log("teacher: layer " + index + " done");
                                publish(index + 1 < layers ? new TeacherLayer(index + 1) : new TeacherHead());
                            }
                        });
            }
        }

        /// Reads the student's starting state of layer `index` (the teacher's top-k already read).
        final class StudentLayer extends Frame {
            final int index;

            StudentLayer(int index) {
                this.index = index;
            }

            @Override
            void work() throws IOException {
                try (LayerWeights.Staged staged = LayerWeights.stage(checkpoint, index)) {
                    model.add(hostLayer(index, staged));
                }
                publish(index + 1 < layers ? new StudentLayer(index + 1) : new StudentStart(null, 0));
            }
        }

        /// The teacher's final norm and LM head over every position, enqueued at once: each chunk's top-k is
        /// copied to host memory behind it on the same stream; `StudentStart` follows the last copy.
        final class TeacherHead extends Frame {
            long head, finalNorm, normed, logits, ids, logProbs;

            @Override
            void work() throws IOException {
                forward.close();
                pipeline.close();
                int k = settings.topK();
                head = uploadBf16("lm_head.weight");
                finalNorm = uploadBf16("model.language_model.norm.weight");
                normed = device.gpu.allocate(2L * HEAD_ROWS * hidden);
                logits = device.gpu.allocate(4L * HEAD_ROWS * vocabulary);
                ids = device.gpu.allocate(4L * HEAD_ROWS * k);
                logProbs = device.gpu.allocate(4L * HEAD_ROWS * k);
                long positions = (long) sequences * length;
                for (long first = 0; first < positions; first += HEAD_ROWS) {
                    int rows = (int) Math.min(HEAD_ROWS, positions - first);
                    device.gpu.copyHostToDeviceAsync(
                            batchStates, streamAddress + first * 2 * hidden, 2L * rows * hidden);
                    device.gpu.rmsNormUnitOffsetBf16(
                            batchStates, finalNorm, normed, rows, hidden, (float) checkpoint.config.eps());
                    device.cuda.gemmBf16(
                            device.cublas,
                            Cuda.CUBLAS_OP_T,
                            Cuda.CUBLAS_OP_N,
                            vocabulary,
                            rows,
                            hidden,
                            1,
                            head,
                            hidden,
                            normed,
                            hidden,
                            0,
                            logits,
                            vocabulary,
                            false);
                    ops.topKLogSoftmax(logits, rows, vocabulary, k, ids, logProbs);
                    device.gpu.copyDeviceToHostAsync(teacherIds.address() + 4 * first * k, ids, 4L * rows * k);
                    device.gpu.copyDeviceToHostAsync(
                            teacherLogProbs.address() + 4 * first * k, logProbs, 4L * rows * k);
                }
                long finished = device.cuda.createEvent();
                device.cuda.recordEvent(finished, device.stream);
                completions.await(finished, new StudentStart(this, finished));
            }
        }

        /// Frees the teacher's buffers (none when its top-k was read) and starts the student's frames; `Finish`
        /// follows them.
        final class StudentStart extends Frame {
            final TeacherHead teacher;
            final long event;

            StudentStart(TeacherHead teacher, long event) {
                this.teacher = teacher;
                this.event = event;
            }

            @Override
            void work() throws IOException {
                if (teacher != null) {
                    device.cuda.destroyEvent(event);
                    for (long address : new long[] {
                        teacher.head,
                        teacher.finalNorm,
                        teacher.normed,
                        teacher.logits,
                        teacher.ids,
                        teacher.logProbs,
                        batchStates
                    }) device.gpu.free(address);
                    device.gpu.freeHostWeights(streamAddress);
                    writeTeacher(teacherIds, teacherLogProbs);
                    log("teacher: top-" + settings.topK() + " of every position kept in " + teacherFile());
                }
                steps = new Steps(lattice, embedded, teacherIds, teacherLogProbs, sequences, length, completions);
                steps.start().whenComplete((ignored, failure) -> {
                    if (failure != null) done.completeExceptionally(failure);
                    else publish(new Finish());
                });
            }
        }

        /// Frees the student and writes the output: the codes directory completed, the overlay checkpoint.
        final class Finish extends Frame {
            @Override
            void work() throws IOException {
                steps.close();
                finish();
                done.complete(null);
            }
        }
    }

    /// A layer's starting state in host memory: codes and scales from the codes directory, norms from `norms`
    /// or the checkpoint, the frozen GDN tensors from the checkpoint.
    private HostLayer hostLayer(int index, LayerWeights.Staged staged) throws IOException {
        boolean attention = checkpoint.config.fullAttention()[index];
        Map<String, HostLinear> linears = new LinkedHashMap<>();
        String prefix = "text/layers/" + index + "/";
        for (String name : staged.parts().keySet()) {
            if (!name.startsWith(prefix)) continue;
            try (Arena arena = Arena.ofConfined()) {
                QuantState state = QuantState.load(QuantState.stateFile(settings.codes(), name), arena);
                if (state.formatName().equals("NVFP4"))
                    throw new IllegalArgumentException(name + " is NVFP4: training takes grouped formats");
                MemorySegment codes = pinned(state.rows() * state.columns());
                MemorySegment.copy(state.codes(), 0, codes, 0, codes.byteSize());
                long count = state.scales().byteSize() / 2;
                HostParameter scales = parameter(count);
                for (long i = 0; i < count; i++)
                    scales.values()
                            .setAtIndex(
                                    ValueLayout.JAVA_FLOAT,
                                    i,
                                    Float.float16ToFloat(
                                            state.scales().getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i)));
                QuantState kept = new QuantState(
                        state.formatName(),
                        state.layout(),
                        state.rows(),
                        state.columns(),
                        codes,
                        null,
                        state.globalScale(),
                        state.table());
                linears.put(
                        name.substring(prefix.length()),
                        new HostLinear(
                                name,
                                name.substring(prefix.length()),
                                (int) state.rows(),
                                (int) state.columns(),
                                codes,
                                kept,
                                scales));
            }
        }
        Map<String, HostParameter> norms = new LinkedHashMap<>();
        Map<String, String> parts = Map.of(
                "input_norm",
                "inputNorm",
                "post_norm",
                "postNorm",
                "query_norm",
                "queryNorm",
                "key_norm",
                "keyNorm",
                "gdn_norm",
                "gdnNorm");
        for (var entry : BlockTraining.normTensors(index, attention).entrySet()) {
            Path trained = settings.norms() == null ? null : settings.norms().resolve(entry.getValue() + ".f32");
            float[] values = trained != null && Files.exists(trained)
                    ? readFloats(trained)
                    : bf16ToFloats(staged.parts().get(parts.get(entry.getKey())));
            HostParameter parameter = parameter(values.length);
            MemorySegment.copy(values, 0, parameter.values(), ValueLayout.JAVA_FLOAT, 0, values.length);
            norms.put(entry.getKey(), parameter);
        }
        Map<String, MemorySegment> frozen = new LinkedHashMap<>();
        if (!attention) {
            frozen.put("a_projection", floats(bf16ToFloats(staged.parts().get("aProjection"))));
            frozen.put("b_projection", floats(bf16ToFloats(staged.parts().get("bProjection"))));
            frozen.put("convolution", floats(bf16ToFloats(staged.parts().get("convolution"))));
            frozen.put("a_log", copy(staged.parts().get("aLog")));
            frozen.put("dt_bias", copy(staged.parts().get("dtBias")));
        }
        return new HostLayer(index, attention, linears, norms, frozen);
    }

    private HostParameter parameter(long count) {
        return new HostParameter(pinned(4 * count), pinned(4 * count), pinned(4 * count));
    }

    private MemorySegment floats(float[] values) {
        MemorySegment segment = pinned(4L * values.length);
        MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_FLOAT, 0, values.length);
        return segment;
    }

    private MemorySegment copy(MemorySegment source) {
        MemorySegment segment = pinned(source.byteSize());
        segment.copyFrom(source);
        return segment;
    }

    private static float[] bf16ToFloats(MemorySegment bf16) {
        float[] values = new float[(int) (bf16.byteSize() / 2)];
        for (int i = 0; i < values.length; i++)
            values[i] = Float.intBitsToFloat(bf16.getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i) << 16);
        return values;
    }

    private static float[] readFloats(Path file) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        float[] values = new float[bytes.remaining() / 4];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    private float[] readNormFloats(String tensor) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            long bytes = checkpoint.store().tensor(tensor).bytes();
            MemorySegment bf16 = arena.allocate(bytes, 16);
            checkpoint.store().read(tensor, 0, bf16);
            return bf16ToFloats(bf16);
        }
    }

    /// A BF16 checkpoint tensor on the device.
    private long uploadBf16(String tensor) throws IOException {
        long bytes = checkpoint.store().tensor(tensor).bytes();
        long address = device.gpu.allocate(bytes);
        try (Arena arena = Arena.ofConfined()) {
            long chunk = 1L << 28;
            MemorySegment buffer = arena.allocate(Math.min(chunk, bytes), 16);
            for (long at = 0; at < bytes; at += chunk) {
                long part = Math.min(chunk, bytes - at);
                checkpoint.store().read(tensor, at, buffer.asSlice(0, part));
                device.gpu.copyHostToDevice(address + at, buffer.asSlice(0, part), part);
            }
        }
        return address;
    }

    // ---- the student ----

    /// The student's training, asynchronous: a frame enqueues a whole step (or an evaluation) on two streams,
    /// and the lattice's completion source emits the next frame once the device is through it; no thread waits
    /// on the device.
    ///
    /// The compute stream runs the layers; the copy stream brings each layer's codes, scales, norms and frozen
    /// tensors from pinned host memory into one of two storage slots of its kind while the layer before it
    /// runs, saves each layer's input to host memory during the forward pass and brings it back during the
    /// backward pass. Events order the two: a slot is filled once the layer that last read it has run, and a
    /// layer runs once its slot is filled. Gradients add up on the device, in each layer's slice of one buffer,
    /// across the sequences of an update; an update streams each parameter's values and Adam moments through
    /// the device and back.
    private final class Steps implements AutoCloseable {

        /// One storage slot: a layer's parameters on the device, and the events that order its use.
        private final class Slot {
            final Map<String, Long> codes = new LinkedHashMap<>(), values = new LinkedHashMap<>();
            final Map<String, Long> frozen = new LinkedHashMap<>();
            final long loaded = device.cuda.createEvent(), released = device.cuda.createEvent();
            int layer = -1;
        }

        final CalibrationLattice lattice;
        final MemorySegment embedded, teacherIds, teacherLogProbs, saved;
        final int sequences, length, training;
        final long count, copyStream;
        final Ops compute;
        final TrainableLayer gdn, attention;
        /// [kind][parity]: kind 0 GDN, 1 attention.
        final Slot[][] slots = new Slot[2][2];
        final int[] slotParity;
        /// Each layer's accumulated gradients on the device, by parameter name.
        final List<Map<String, Long>> gradients = new ArrayList<>();
        final long gradientBuffer, head, finalNorm, inverse, normed, dn, embeddedBf16, logits, dLogits, chunkBf16;
        final long ids, logProbs, loss, scratchOutput, weight, weightGradient;
        final long[] activations = new long[2], inputs = new long[2], deltas = new long[2];
        final long[] saveDone = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long[] inputLoaded = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long[] inputReleased = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long[] applyLoaded = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long[] applyDone = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long[] applyReleased = {device.cuda.createEvent(), device.cuda.createEvent()};
        final long copyReady = device.cuda.createEvent(), computeReady = device.cuda.createEvent();
        final long computeDone = device.cuda.createEvent(), finished = device.cuda.createEvent();
        /// The head's teacher rows and the embedded input are overwritten only once the compute stream has read
        /// them.
        final long headReleased = device.cuda.createEvent(), embeddedReleased = device.cuda.createEvent();
        /// Adam's values and moments of one parameter at a time, two in flight.
        final long[][] applyBuffers = new long[2][3];
        final MemorySegment lossHost;
        final GpuCompletions completions;
        final CompletableFuture<Void> done = new CompletableFuture<>();
        final FrameSeeds seeds = new FrameSeeds();
        final List<Long> owned = new ArrayList<>();
        int step, update, pendingSequences, applyParity;
        double best = Double.POSITIVE_INFINITY, window;

        Steps(
                CalibrationLattice lattice,
                MemorySegment embedded,
                MemorySegment teacherIds,
                MemorySegment teacherLogProbs,
                int sequences,
                int length,
                GpuCompletions completions)
                throws IOException {
            this.lattice = lattice;
            this.embedded = embedded;
            this.teacherIds = teacherIds;
            this.teacherLogProbs = teacherLogProbs;
            this.sequences = sequences;
            this.length = length;
            this.training = sequences - settings.holdout();
            this.count = (long) length * hidden;
            this.copyStream = device.newStream();
            long handle = device.cuda.cublasHandle(device.stream);
            if (settings.tensorCores()) device.cuda.tensorCoreMath(handle);
            this.compute = new Ops(device.cuda, device.stream, handle);
            long largest = 0, largestParameter = 0;
            for (HostLayer layer : model)
                for (HostLinear linear : layer.linears().values()) {
                    largest = Math.max(largest, (long) linear.out() * linear.in());
                    largestParameter =
                            Math.max(largestParameter, linear.scales().values().byteSize() / 4);
                }
            this.weight = allocate(4 * largest);
            this.weightGradient = allocate(4 * largest);
            this.gdn = describe(false);
            this.attention = describe(true);
            for (int kind = 0; kind < 2; kind++)
                for (int parity = 0; parity < 2; parity++) slots[kind][parity] = slot(kind == 1);
            this.slotParity = new int[layers];
            int[] ordinal = new int[2];
            for (HostLayer layer : model) slotParity[layer.index()] = ordinal[layer.attention() ? 1 : 0]++ % 2;
            long total = 0;
            for (HostLayer layer : model) {
                for (HostLinear linear : layer.linears().values())
                    total += linear.scales().values().byteSize() / 4;
                for (HostParameter norm : layer.norms().values())
                    total += norm.values().byteSize() / 4;
            }
            this.gradientBuffer = allocate(4 * total);
            compute.zero(gradientBuffer, total);
            long at = gradientBuffer;
            for (HostLayer layer : model) {
                Map<String, Long> slices = new LinkedHashMap<>();
                for (HostLinear linear : layer.linears().values()) {
                    slices.put(linear.role() + "/scales", at);
                    at += linear.scales().values().byteSize();
                }
                for (var norm : layer.norms().entrySet()) {
                    slices.put(norm.getKey(), at);
                    at += norm.getValue().values().byteSize();
                }
                gradients.add(slices);
            }
            for (int i = 0; i < 2; i++) {
                activations[i] = allocate(4 * count);
                inputs[i] = allocate(4 * count);
                deltas[i] = allocate(4 * count);
                for (int j = 0; j < 3; j++) applyBuffers[i][j] = allocate(4 * largestParameter);
            }
            this.scratchOutput = allocate(4 * count);
            this.head = uploadBf16("lm_head.weight");
            owned.add(head);
            this.finalNorm = allocate(4L * hidden);
            Upload.floats(device.gpu, finalNorm, readNormFloats("model.language_model.norm.weight"));
            this.inverse = allocate(4L * length);
            this.normed = allocate(4 * count);
            this.dn = allocate(4 * count);
            this.embeddedBf16 = allocate(2 * count);
            this.logits = allocate(4L * HEAD_ROWS * vocabulary);
            this.dLogits = allocate(4L * HEAD_ROWS * vocabulary);
            this.chunkBf16 = allocate(2L * HEAD_ROWS * Math.max(hidden, vocabulary));
            int k = settings.topK();
            this.ids = allocate(4L * length * k);
            this.logProbs = allocate(4L * length * k);
            this.loss = allocate(8L * Math.max(sequences, 1));
            this.saved = pinned((long) layers * 4 * count);
            this.lossHost = pinned(8L * Math.max(sequences, 1));
            this.completions = completions;
        }

        private long allocate(long bytes) {
            long address = device.gpu.allocate(bytes);
            owned.add(address);
            return address;
        }

        private TrainableLayer describe(boolean attention) {
            HostLayer template = model.stream()
                    .filter(l -> l.attention() == attention)
                    .findFirst()
                    .orElseThrow();
            var c = checkpoint.config;
            var shape = new TrainableLayer.Shape(
                    attention,
                    c.hidden(),
                    c.intermediate(),
                    c.keyHeads(),
                    c.valueHeads(),
                    c.keyDim(),
                    c.valueDim(),
                    c.convKernel(),
                    c.heads(),
                    c.kvHeads(),
                    c.headDim(),
                    c.rotaryDim(),
                    c.theta(),
                    (float) c.eps());
            var layer = new TrainableLayer(device.gpu, compute, shape);
            for (HostLinear linear : template.linears().values())
                layer.describeLinear(linear.role(), linear.out(), linear.in(), 64);
            for (var norm : template.norms().entrySet())
                layer.describeNorm(
                        norm.getKey(), (int) (norm.getValue().values().byteSize() / 4));
            for (String role : template.frozen().keySet()) layer.describeFrozen(role);
            layer.shareWeights(weight, weightGradient);
            return layer;
        }

        private Slot slot(boolean attention) {
            HostLayer template = model.stream()
                    .filter(l -> l.attention() == attention)
                    .findFirst()
                    .orElseThrow();
            Slot slot = new Slot();
            for (HostLinear linear : template.linears().values()) {
                slot.codes.put(linear.role(), allocate((long) linear.out() * linear.in()));
                slot.values.put(
                        linear.role() + "/scales",
                        allocate(linear.scales().values().byteSize()));
            }
            for (var norm : template.norms().entrySet())
                slot.values.put(norm.getKey(), allocate(norm.getValue().values().byteSize()));
            for (var frozen : template.frozen().entrySet())
                slot.frozen.put(frozen.getKey(), allocate(frozen.getValue().byteSize()));
            return slot;
        }

        private Slot slotOf(int index) {
            return slots[model.get(index).attention() ? 1 : 0][slotParity[index]];
        }

        // ---- enqueueing; every method below runs on the frame that owns the run ----

        /// Selects the copy stream for the engine's asynchronous copies on this thread.
        private void onCopyStream() {
            device.select(copyStream);
        }

        private void toDevice(long target, MemorySegment source) {
            device.gpu.copyHostToDeviceAsync(target, source.address(), source.byteSize());
        }

        private void toHost(MemorySegment target, long source) {
            device.gpu.copyDeviceToHostAsync(target.address(), source, target.byteSize());
        }

        /// Fills layer `index`'s slot on the copy stream, once the layer that last used the slot has run.
        private void load(int index) {
            Slot slot = slotOf(index);
            HostLayer layer = model.get(index);
            device.cuda.waitEvent(copyStream, slot.released);
            for (HostLinear linear : layer.linears().values()) {
                toDevice(slot.codes.get(linear.role()), linear.codes());
                toDevice(
                        slot.values.get(linear.role() + "/scales"),
                        linear.scales().values());
            }
            for (var norm : layer.norms().entrySet())
                toDevice(slot.values.get(norm.getKey()), norm.getValue().values());
            for (var frozen : layer.frozen().entrySet()) toDevice(slot.frozen.get(frozen.getKey()), frozen.getValue());
            slot.layer = index;
            device.cuda.recordEvent(slot.loaded, copyStream);
        }

        /// The layer of `index`'s kind, computing from its slot (on the compute stream, behind the slot's
        /// load).
        private TrainableLayer use(int index) {
            Slot slot = slotOf(index);
            device.cuda.waitEvent(device.stream, slot.loaded);
            TrainableLayer layer = model.get(index).attention() ? attention : gdn;
            layer.bind(new TrainableLayer.Binding(slot.codes, slot.values, gradients.get(index), slot.frozen));
            return layer;
        }

        /// The forward pass of sequence `s`: the last layer's output in `activations[layers % 2]`; with `keep`,
        /// each layer's input saved to host memory.
        private long forward(int s, boolean keep) {
            device.cuda.waitEvent(copyStream, embeddedReleased);
            device.gpu.copyHostToDeviceAsync(embeddedBf16, embedded.address() + s * 2 * count, 2 * count);
            device.cuda.recordEvent(copyReady, copyStream);
            device.cuda.waitEvent(device.stream, copyReady);
            compute.bf16ToFloat(embeddedBf16, count, activations[0]);
            device.cuda.recordEvent(embeddedReleased, device.stream);
            load(0);
            for (int index = 0; index < layers; index++) {
                if (index + 1 < layers) load(index + 1);
                long input = activations[index % 2], output = activations[(index + 1) % 2];
                if (keep) {
                    // The copy stream saves the input once it is written; the layer writing `output` (the input of
                    // the layer before, two back) waits for that layer's save.
                    device.cuda.recordEvent(computeReady, device.stream);
                    device.cuda.waitEvent(copyStream, computeReady);
                    device.gpu.copyDeviceToHostAsync(saved.address() + index * 4 * count, input, 4 * count);
                    device.cuda.recordEvent(saveDone[index % 2], copyStream);
                    device.cuda.waitEvent(device.stream, saveDone[(index + 1) % 2]);
                }
                TrainableLayer layer = use(index);
                layer.forward(input, output, length, 0);
                device.cuda.recordEvent(slotOf(index).released, device.stream);
            }
            return activations[layers % 2];
        }

        /// The KL at the LM head for sequence `s` from the last layer's output `x` into `loss[slot]` (per
        /// position); with `gradient`, its gradient on `x` into `dh`.
        private void head(int s, long x, long dh, boolean gradient, int slot) {
            int k = settings.topK();
            long base = (long) s * length * k;
            device.cuda.waitEvent(copyStream, headReleased);
            device.gpu.copyHostToDeviceAsync(ids, teacherIds.address() + 4 * base, 4L * length * k);
            device.gpu.copyHostToDeviceAsync(logProbs, teacherLogProbs.address() + 4 * base, 4L * length * k);
            device.cuda.recordEvent(copyReady, copyStream);
            device.cuda.waitEvent(device.stream, copyReady);
            compute.rmsNormForward(x, finalNorm, normed, inverse, length, hidden, (float) checkpoint.config.eps(), 1);
            compute.zero(dn, count);
            for (int first = 0; first < length; first += HEAD_ROWS) {
                int rows = Math.min(HEAD_ROWS, length - first);
                compute.floatToBf16(normed + 4L * first * hidden, (long) rows * hidden, chunkBf16);
                device.cuda.gemmBf16(
                        compute.cublas,
                        Cuda.CUBLAS_OP_T,
                        Cuda.CUBLAS_OP_N,
                        vocabulary,
                        rows,
                        hidden,
                        1,
                        head,
                        hidden,
                        chunkBf16,
                        hidden,
                        0,
                        logits,
                        vocabulary,
                        false);
                compute.topKDistillation(
                        logits,
                        rows,
                        vocabulary,
                        k,
                        ids + 4L * first * k,
                        logProbs + 4L * first * k,
                        1.0f / length,
                        dLogits,
                        loss + 8L * slot);
                if (!gradient) continue;
                compute.floatToBf16(dLogits, (long) rows * vocabulary, chunkBf16);
                device.cuda.gemmBf16(
                        compute.cublas,
                        Cuda.CUBLAS_OP_N,
                        Cuda.CUBLAS_OP_N,
                        hidden,
                        rows,
                        vocabulary,
                        1,
                        head,
                        hidden,
                        chunkBf16,
                        vocabulary,
                        0,
                        dn + 4L * first * hidden,
                        hidden,
                        false);
            }
            device.cuda.recordEvent(headReleased, device.stream);
            if (gradient) {
                compute.zero(dh, count);
                compute.rmsNormBackward(x, finalNorm, inverse, dn, dh, 0, length, hidden, 1);
            }
        }

        /// One training step on sequence `s`: forward, head, backward; the gradients add to the accumulated
        /// ones.
        private void step(int s) {
            long x = forward(s, true);
            head(s, x, deltas[0], true, 0);
            load(layers - 1);
            loadInput(layers - 1);
            for (int index = layers - 1; index >= 0; index--) {
                if (index > 0) {
                    load(index - 1);
                    loadInput(index - 1);
                }
                device.cuda.waitEvent(device.stream, inputLoaded[index % 2]);
                TrainableLayer layer = use(index);
                long input = inputs[index % 2];
                layer.forward(input, scratchOutput, length, 0);
                long dy = deltas[(layers - 1 - index) % 2], dx = deltas[(layers - index) % 2];
                layer.backward(input, dy, index > 0 ? dx : 0, 0);
                device.cuda.recordEvent(slotOf(index).released, device.stream);
                device.cuda.recordEvent(inputReleased[index % 2], device.stream);
            }
        }

        /// Brings layer `index`'s saved input back into `inputs[index % 2]` on the copy stream.
        private void loadInput(int index) {
            device.cuda.waitEvent(copyStream, inputReleased[index % 2]);
            device.gpu.copyHostToDeviceAsync(inputs[index % 2], saved.address() + index * 4 * count, 4 * count);
            device.cuda.recordEvent(inputLoaded[index % 2], copyStream);
        }

        /// One Adam step of every parameter on the mean of the gradients accumulated over `sequencesSince`
        /// sequences; clears them.
        private void apply(int sequencesSince) {
            update++;
            for (HostLayer layer : model) {
                Map<String, Long> slices = gradients.get(layer.index());
                for (HostLinear linear : layer.linears().values())
                    adam(linear.scales(), slices.get(linear.role() + "/scales"), sequencesSince, settings.scaleRate());
                for (var norm : layer.norms().entrySet())
                    adam(norm.getValue(), slices.get(norm.getKey()), sequencesSince, settings.normRate());
            }
        }

        private void adam(HostParameter state, long gradient, int sequencesSince, float rate) {
            int parity = applyParity++ % 2;
            long[] buffers = applyBuffers[parity];
            long elements = state.values().byteSize() / 4;
            device.cuda.waitEvent(copyStream, applyReleased[parity]);
            toDevice(buffers[0], state.values());
            toDevice(buffers[1], state.first());
            toDevice(buffers[2], state.second());
            device.cuda.recordEvent(applyLoaded[parity], copyStream);
            device.cuda.waitEvent(device.stream, applyLoaded[parity]);
            compute.axpby(gradient, gradient, elements, 0, 1.0f / sequencesSince);
            compute.adam(buffers[0], gradient, buffers[1], buffers[2], elements, rate, 0.9f, 0.999f, 1e-12f, 0, update);
            device.cuda.recordEvent(applyDone[parity], device.stream);
            device.cuda.waitEvent(copyStream, applyDone[parity]);
            toHost(state.values(), buffers[0]);
            toHost(state.first(), buffers[1]);
            toHost(state.second(), buffers[2]);
            device.cuda.recordEvent(applyReleased[parity], copyStream);
        }

        /// Joins the compute stream into the copy stream, copies the losses out after it, and records
        /// `finished` there: once it is done, every copy and kernel enqueued so far is.
        private void finish(int slots) {
            device.cuda.recordEvent(computeDone, device.stream);
            device.cuda.waitEvent(copyStream, computeDone);
            device.gpu.copyDeviceToHostAsync(lossHost.address(), loss, 8L * slots);
            device.cuda.recordEvent(finished, copyStream);
        }

        // ---- the frames ----

        /// Publishes the first evaluation; `done` completes when the last frame has run.
        CompletableFuture<Void> start() throws IOException {
            Properties resumed = readState();
            if (resumed == null) {
                lattice.lake().publish(new Evaluate(true));
                return done;
            }
            step = Integer.parseInt(resumed.getProperty("step"));
            update = Integer.parseInt(resumed.getProperty("update"));
            best = Double.parseDouble(resumed.getProperty("best"));
            log(String.format(Locale.ROOT, "resumed at step %d (best held-out KL %.6f)", step, best));
            next();
            return done;
        }

        private double lossAt(int slot) {
            return lossHost.getAtIndex(ValueLayout.JAVA_DOUBLE, slot);
        }

        /// A frame of the run: its work enqueues device work or does host work; a failure ends the run.
        private abstract class Frame extends io.euhedral_execution.core.frames.AbstractFrame {
            Frame() {
                super(io.euhedral_execution.inference.core.runtime.graph.FrameSeeds.ID_HASH);
                randomizeHash(seeds.next());
            }

            abstract void work() throws Exception;

            @Override
            public void execute() {
                if (done.isDone()) return;
                try {
                    onCopyStream();
                    work();
                } catch (Throwable failure) {
                    done.completeExceptionally(failure);
                }
            }

            @Override
            public void doFinallyWithError(Throwable failure) {
                done.completeExceptionally(failure);
            }
        }

        /// Enqueues the forward pass and KL of every held-out sequence; `Evaluated` reads them.
        private final class Evaluate extends Frame {
            final boolean initial;

            Evaluate(boolean initial) {
                this.initial = initial;
            }

            @Override
            void work() {
                compute.zero(loss, (long) (sequences - training) * 2);
                for (int s = training; s < sequences; s++) head(s, forward(s, false), 0, false, s - training);
                finish(sequences - training);
                completions.await(finished, new Evaluated(initial));
            }
        }

        private final class Evaluated extends Frame {
            final boolean initial;

            Evaluated(boolean initial) {
                this.initial = initial;
            }

            @Override
            void work() throws IOException {
                double total = 0;
                for (int i = 0; i < sequences - training; i++) total += lossAt(i);
                double heldOut = total / (sequences - training) / length;
                if (initial) log(String.format(Locale.ROOT, "held-out KL %.6f before training", heldOut));
                else log(String.format(Locale.ROOT, "step %d: held-out KL %.6f", step, heldOut));
                if (heldOut < best) {
                    best = heldOut;
                    save();
                }
                if (!initial && pendingSequences == 0) writeState(step, update, best);
                next();
            }
        }

        /// The next frame after a step or an evaluation: another step, an evaluation, or the end.
        private void next() {
            int total = settings.epochs() * training;
            if (step >= total) {
                done.complete(null);
                return;
            }
            lattice.lake().publish(new Step());
        }

        /// Enqueues one training step (and, when due, the update); `Stepped` follows it.
        private final class Step extends Frame {
            @Override
            void work() {
                int s = step % training;
                step++;
                pendingSequences++;
                compute.zero(loss, 2);
                step(s);
                boolean last = step == settings.epochs() * training;
                if (pendingSequences == settings.accumulate() || step % training == 0) {
                    apply(pendingSequences);
                    pendingSequences = 0;
                }
                finish(1);
                completions.await(finished, new Stepped(last));
            }
        }

        private final class Stepped extends Frame {
            final boolean last;

            Stepped(boolean last) {
                this.last = last;
            }

            @Override
            void work() {
                window += lossAt(0) / length;
                if (step % 10 == 0) {
                    log(String.format(Locale.ROOT, "step %d: training KL %.6f (last 10)", step, window / 10));
                    window = 0;
                }
                boolean evaluate = last
                        || step % training == 0
                        || (settings.evaluateEvery() > 0 && step % settings.evaluateEvery() == 0);
                if (evaluate) lattice.lake().publish(new Evaluate(false));
                else next();
            }
        }

        @Override
        public void close() {
            device.select(device.stream);
            device.synchronize();
            for (long address : owned) device.gpu.free(address);
            gdn.close();
            attention.close();
        }
    }

    // ---- resuming ----

    private Path teacherFile() {
        return settings.out().resolve("resume").resolve("teacher.bin");
    }

    /// The teacher's top-k ids and log-probabilities, after a line naming the run's identity.
    private void writeTeacher(MemorySegment ids, MemorySegment logProbs) throws IOException {
        Path file = teacherFile(), partial = file.resolveSibling("teacher.bin.partial");
        Files.createDirectories(file.getParent());
        try (FileChannel channel = FileChannel.open(
                partial, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            write(channel, MemorySegment.ofArray((teacherIdentity() + "\n").getBytes(StandardCharsets.UTF_8)));
            write(channel, ids);
            write(channel, logProbs);
        }
        Files.move(partial, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    /// Reads the kept top-k into `ids` and `logProbs` if it is this run's; false otherwise.
    private boolean readTeacher(MemorySegment ids, MemorySegment logProbs) throws IOException {
        Path file = teacherFile();
        if (!Files.exists(file)) return false;
        byte[] header = (teacherIdentity() + "\n").getBytes(StandardCharsets.UTF_8);
        if (Files.size(file) != header.length + ids.byteSize() + logProbs.byteSize()) return false;
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
            MemorySegment kept = MemorySegment.ofArray(new byte[header.length]);
            read(channel, kept);
            if (!java.util.Arrays.equals(kept.toArray(ValueLayout.JAVA_BYTE), header)) return false;
            read(channel, ids);
            read(channel, logProbs);
        }
        return true;
    }

    /// The teacher depends on the data and the k it keeps, not on the student's settings.
    private String teacherIdentity() {
        return identity.replaceAll(" (holdout|accumulate|scaleRate|normRate|codes|norms)=\\S+", "");
    }

    /// Every layer's trainable parameters, in the order the state files hold them.
    private static List<HostParameter> parameters(HostLayer layer) {
        List<HostParameter> parameters = new ArrayList<>();
        for (HostLinear linear : layer.linears().values()) parameters.add(linear.scales());
        parameters.addAll(layer.norms().values());
        return parameters;
    }

    /// Writes the training state next to the last one and then replaces it, so a stop at any point leaves one
    /// whole state.
    private void writeState(int step, int update, double best) throws IOException {
        Path resume = settings.out().resolve("resume"), partial = resume.resolve("state.partial");
        Path state = resume.resolve("state"), old = resume.resolve("state.old");
        deleteTree(partial);
        Files.createDirectories(partial);
        for (HostLayer layer : model) {
            try (FileChannel channel = FileChannel.open(
                    partial.resolve("layer-" + layer.index() + ".bin"),
                    StandardOpenOption.CREATE_NEW,
                    StandardOpenOption.WRITE)) {
                for (HostParameter parameter : parameters(layer)) {
                    write(channel, parameter.values());
                    write(channel, parameter.first());
                    write(channel, parameter.second());
                }
            }
        }
        Properties progress = new Properties();
        progress.setProperty("identity", identity);
        progress.setProperty("step", Integer.toString(step));
        progress.setProperty("update", Integer.toString(update));
        progress.setProperty("best", Double.toString(best));
        try (Writer writer = Files.newBufferedWriter(partial.resolve("progress.properties"))) {
            progress.store(writer, null);
        }
        deleteTree(old);
        if (Files.exists(state)) Files.move(state, old, StandardCopyOption.ATOMIC_MOVE);
        Files.move(partial, state, StandardCopyOption.ATOMIC_MOVE);
        deleteTree(old);
        log("training state at step " + step + " kept in " + state);
    }

    /// Reads the kept training state into the host parameters if it is this run's; null otherwise.
    private Properties readState() throws IOException {
        Path resume = settings.out().resolve("resume"), state = resume.resolve("state");
        // A stop between the two moves leaves the last whole state as `state.old`.
        if (!Files.exists(state) && Files.exists(resume.resolve("state.old")))
            Files.move(resume.resolve("state.old"), state, StandardCopyOption.ATOMIC_MOVE);
        Path file = state.resolve("progress.properties");
        if (!Files.exists(file)) return null;
        Properties progress = new Properties();
        try (Reader reader = Files.newBufferedReader(file)) {
            progress.load(reader);
        }
        if (!identity.equals(progress.getProperty("identity"))) {
            log("the training state in " + state + " is another run's (" + progress.getProperty("identity")
                    + "); starting over");
            return null;
        }
        for (HostLayer layer : model) {
            try (FileChannel channel =
                    FileChannel.open(state.resolve("layer-" + layer.index() + ".bin"), StandardOpenOption.READ)) {
                for (HostParameter parameter : parameters(layer)) {
                    read(channel, parameter.values());
                    read(channel, parameter.first());
                    read(channel, parameter.second());
                }
                if (channel.position() != channel.size())
                    throw new IOException("layer " + layer.index() + "'s training state has the wrong size");
            }
        }
        return progress;
    }

    private static void write(FileChannel channel, MemorySegment segment) throws IOException {
        ByteBuffer buffer = segment.asByteBuffer();
        while (buffer.hasRemaining()) channel.write(buffer);
    }

    private static void read(FileChannel channel, MemorySegment segment) throws IOException {
        ByteBuffer buffer = segment.asByteBuffer();
        while (buffer.hasRemaining()) if (channel.read(buffer) < 0) throw new EOFException("training state ends early");
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (var paths = Files.walk(root)) {
            for (Path path : paths.sorted(java.util.Comparator.reverseOrder()).toList()) Files.delete(path);
        }
    }

    // ---- output ----

    /// Writes every layer's objects with their trained scales (float16) and its norms, as `train-blocks` does:
    /// the state of the best held-out KL so far.
    private void save() throws IOException {
        Files.createDirectories(settings.out().resolve("norms"));
        for (HostLayer layer : model) {
            for (HostLinear linear : layer.linears().values()) {
                long count = linear.scales().values().byteSize() / 4;
                MemorySegment half = MemorySegment.ofArray(new byte[(int) (2 * count)]);
                for (long i = 0; i < count; i++)
                    half.setAtIndex(
                            ValueLayout.JAVA_SHORT_UNALIGNED,
                            i,
                            Float.floatToFloat16(linear.scales().values().getAtIndex(ValueLayout.JAVA_FLOAT, i)));
                QuantState state = linear.state();
                new QuantState(
                                state.formatName(),
                                state.layout(),
                                state.rows(),
                                state.columns(),
                                linear.codes(),
                                half,
                                state.globalScale(),
                                state.table())
                        .save(QuantState.stateFile(settings.out(), linear.object()));
            }
            for (var entry :
                    BlockTraining.normTensors(layer.index(), layer.attention()).entrySet()) {
                MemorySegment values = layer.norms().get(entry.getKey()).values();
                Path file = settings.out().resolve("norms").resolve(entry.getValue() + ".f32");
                try (FileChannel channel = FileChannel.open(
                        file,
                        StandardOpenOption.CREATE,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.TRUNCATE_EXISTING)) {
                    channel.write(values.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN));
                }
            }
        }
        log("saved the scales and norms of the best held-out KL so far in " + settings.out());
    }

    /// Completes `out` with the objects it lacks (the heads, the MTP layer) and writes the overlay checkpoint.
    private void finish() throws IOException {
        try (var files = Files.list(settings.codes())) {
            for (Path file : (Iterable<Path>) files::iterator) {
                Path copy = settings.out().resolve(file.getFileName());
                if (file.getFileName().toString().endsWith(".npz") && !Files.exists(copy)) Files.copy(file, copy);
            }
        }
        if (settings.overlay() == null) return;
        Map<String, float[]> norms = new java.util.TreeMap<>();
        try (var files = Files.list(settings.out().resolve("norms"))) {
            for (Path file : (Iterable<Path>) files::iterator)
                norms.put(file.getFileName().toString().replaceAll("\\.f32$", ""), readFloats(file));
        }
        Overlay.write(settings.model(), settings.overlay(), norms);
        log("overlay checkpoint with " + norms.size() + " trained norms in " + settings.overlay());
    }

    // ---- embedding ----

    private void embed(int[][] tokens, MemorySegment states, long batchStates) throws IOException {
        var tensor = checkpoint.store().tensor("model.language_model.embed_tokens.weight");
        long table = uploadBf16(tensor.name());
        Cuda.Function gather =
                device.cuda.compileResource("rows.cu", List.of("gather_rows")).getFirst();
        int length = tokens[0].length;
        long sequenceBytes = 2L * hidden * length, idsDevice = device.gpu.allocate(4L * length);
        try {
            for (int s = 0; s < tokens.length; s++) {
                Upload.ints(device.gpu, idsDevice, tokens[s]);
                long vectors = (long) length * hidden / 8;
                device.cuda.launch(
                        gather,
                        (int) ((vectors + 255) / 256),
                        1,
                        256,
                        0,
                        device.stream,
                        new Cuda.Arguments()
                                .pointer(table)
                                .pointer(idsDevice)
                                .pointer(length)
                                .integer(hidden)
                                .pointer(batchStates));
                device.synchronize();
                device.gpu.copyDeviceToHost(
                        states.asSlice(s * sequenceBytes, sequenceBytes), batchStates, sequenceBytes);
            }
        } finally {
            device.gpu.free(idsDevice);
            device.gpu.free(table);
        }
        log(tokens.length + " sequences of " + length + " tokens embedded");
    }
}
