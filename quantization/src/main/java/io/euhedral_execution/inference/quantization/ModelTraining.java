package io.euhedral_execution.inference.quantization;

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
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/// End-to-end distillation of a quantized artifact's scales and norms: every decoder layer's quantized linears train
/// their group scales (codes fixed) and the layers' norms their weights, against the BF16 checkpoint's own output
/// distribution: at every position of the calibration sequences, the KL over its top-k tokens and one bucket for the
/// rest of the vocabulary (the teacher's tail mass is kept, so the student is not pushed to zero probability outside
/// the top k; `top_k_distillation`).
///
/// The teacher runs once: the BF16 layers over every sequence, then the final norm and LM head, keeping each
/// position's top-k log-probabilities. Each step then runs one sequence through all the layers on the training kernels
/// ([TrainableLayer], one of each kind on the device, filled per layer from host memory), keeping each layer's input;
/// takes the KL at the LM head (the checkpoint's, in BF16); and goes back through the layers, recomputing each from its
/// input, so that each layer's scales and norms take their Adam step (their moments kept in host memory with them).
/// Held-out sequences measure the mean KL per position. The result is written as `train-blocks` writes it: a codes
/// directory for `convert_checkpoint.py --codes-from` and an overlay checkpoint with the trained norms.
public final class ModelTraining {

    /// Command-line settings. `codes` holds the starting rounding (all objects); `norms`, when set, the starting norms
    /// (a `train-blocks` output's `norms`); otherwise the checkpoint's.
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
            int batch) {}

    /// Rows of the LM head's logits at a time.
    private static final int HEAD_ROWS = 256;

    private final Settings settings;
    private final Device device;
    private final Qwen38Objects checkpoint;
    private final Ops ops;
    private final int hidden, vocabulary, layers;
    private final long start = System.nanoTime();
    private final Arena host = Arena.ofShared();
    private final List<Long> pinned = new ArrayList<>();

    /// A trainable parameter in host memory: values and Adam moments.
    private record HostParameter(
            MemorySegment values, MemorySegment first, MemorySegment second, MemorySegment gradient) {}

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

    private ModelTraining(Settings settings, Device device, Qwen38Objects checkpoint) {
        this.settings = settings;
        this.device = device;
        this.checkpoint = checkpoint;
        this.ops = new Ops(device.cuda, device.stream, device.cublas);
        this.hidden = checkpoint.config.hidden();
        this.vocabulary = checkpoint.config.vocabulary();
        this.layers = checkpoint.config.layers();
    }

    public static void run(Settings settings, Path library) throws IOException {
        Files.createDirectories(settings.out());
        try (Safetensors store = Safetensors.open(settings.model());
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(6)) {
            var training = new ModelTraining(settings, device, new Qwen38Objects(store, settings.model()));
            try {
                training.train(lattice);
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

    private MemorySegment pinned(long bytes) {
        long address = device.gpu.allocateHostWeights(bytes);
        pinned.add(address);
        return MemorySegment.ofAddress(address).reinterpret(bytes);
    }

    // ---- the run ----

    private void train(CalibrationLattice lattice) throws IOException {
        int[][] tokens = Calibrate.readTokens(settings.tokens(), settings.sequences());
        int sequences = tokens.length, length = tokens[0].length, k = settings.topK();
        long positions = (long) sequences * length;
        MemorySegment embedded = pinned(positions * 2 * hidden);
        MemorySegment teacherIds = host.allocate(4 * positions * k, 4);
        MemorySegment teacherLogProbs = host.allocate(4 * positions * k, 4);
        teacher(lattice, tokens, embedded, teacherIds, teacherLogProbs);
        long rows = length;
        // The device side of the student: one trainable layer of each kind, the LM head and final norm, the buffers.
        var c = checkpoint.config;
        try (TrainableLayer gdn = storage(false);
                TrainableLayer attention = storage(true)) {
            long head = uploadBf16("lm_head.weight");
            long finalNorm = device.gpu.allocate(4L * hidden);
            float[] finalNormValues = readNormFloats("model.language_model.norm.weight");
            Upload.floats(device.gpu, finalNorm, finalNormValues);
            long count = rows * hidden;
            long x = device.gpu.allocate(4 * count),
                    y = device.gpu.allocate(4 * count),
                    dh = device.gpu.allocate(4 * count);
            long dx = device.gpu.allocate(4 * count),
                    normed = device.gpu.allocate(4 * count),
                    dn = device.gpu.allocate(4 * count);
            long inverse = device.gpu.allocate(4 * rows), embeddedBf16 = device.gpu.allocate(2 * count);
            long logits = device.gpu.allocate(4L * HEAD_ROWS * vocabulary),
                    dLogits = device.gpu.allocate(4L * HEAD_ROWS * vocabulary);
            long chunkBf16 = device.gpu.allocate(2L * HEAD_ROWS * Math.max(hidden, vocabulary));
            long ids = device.gpu.allocate(4L * rows * k), logProbs = device.gpu.allocate(4L * rows * k);
            long loss = device.gpu.allocate(8);
            MemorySegment saved = pinned((long) layers * 4 * count);
            Student student = new Student(
                    gdn,
                    attention,
                    head,
                    finalNorm,
                    x,
                    y,
                    dh,
                    dx,
                    normed,
                    dn,
                    inverse,
                    embeddedBf16,
                    logits,
                    dLogits,
                    chunkBf16,
                    ids,
                    logProbs,
                    loss,
                    saved,
                    embedded,
                    teacherIds,
                    teacherLogProbs,
                    length);
            int training = sequences - settings.holdout();
            double best = student.evaluate(training, sequences);
            log(String.format(Locale.ROOT, "held-out KL %.6f before training", best));
            // The output keeps the state of the best held-out KL measured so far, the starting state included.
            save();
            int step = 0;
            for (int epoch = 1; epoch <= settings.epochs(); epoch++) {
                double total = 0;
                for (int s = 0; s < training; s++) {
                    double kl = student.step(s, ++step);
                    total += kl;
                    if (step % settings.accumulate() == 0)
                        student.apply(settings.accumulate(), step / settings.accumulate());
                    if (step <= settings.probe()) {
                        student.forward(s, false);
                        log(String.format(
                                Locale.ROOT,
                                "probe step %d: sequence %d KL %.6f before its step, %.6f after",
                                step,
                                s,
                                kl,
                                student.headLoss(s, false)));
                    }
                    if (step % 10 == 0)
                        log(String.format(Locale.ROOT, "step %d: training KL %.6f (last 10)", step, total / 10));
                    if (step % 10 == 0) total = 0;
                    if (settings.evaluateEvery() > 0 && step % settings.evaluateEvery() == 0) {
                        double heldOut = student.evaluate(training, sequences);
                        log(String.format(Locale.ROOT, "step %d: held-out KL %.6f", step, heldOut));
                        if (heldOut < best) {
                            best = heldOut;
                            save();
                        }
                    }
                }
                if (step % settings.accumulate() != 0)
                    student.apply(step % settings.accumulate(), step / settings.accumulate() + 1);
                double heldOut = student.evaluate(training, sequences);
                log(String.format(Locale.ROOT, "epoch %d: held-out KL %.6f", epoch, heldOut));
                if (heldOut < best) {
                    best = heldOut;
                    save();
                }
            }
            for (long address : new long[] {
                head,
                finalNorm,
                x,
                y,
                dh,
                dx,
                normed,
                dn,
                inverse,
                embeddedBf16,
                logits,
                dLogits,
                chunkBf16,
                ids,
                logProbs,
                loss
            }) device.gpu.free(address);
        }
        finish();
    }

    // ---- the teacher ----

    /// Embeds every sequence, runs the BF16 layers over them (reading each layer's starting state into host memory on
    /// the way), and keeps the top-k log-probabilities of the BF16 model at every position.
    private void teacher(
            CalibrationLattice lattice,
            int[][] tokens,
            MemorySegment embedded,
            MemorySegment teacherIds,
            MemorySegment teacherLogProbs)
            throws IOException {
        int sequences = tokens.length, length = tokens[0].length, k = settings.topK();
        long sequenceBytes = 2L * hidden * length;
        // The teacher's stream is freed with the pass; only the embedded sequences stay for the student.
        long streamAddress = device.gpu.allocateHostWeights(sequenceBytes * sequences);
        MemorySegment stream = MemorySegment.ofAddress(streamAddress).reinterpret(sequenceBytes * sequences);
        long batchRows = (long) settings.batch() * length;
        long batchStates = device.gpu.allocate(batchRows * 2 * hidden);
        embed(tokens, embedded, batchStates);
        MemorySegment.copy(embedded, 0, stream, 0, sequenceBytes * sequences);
        GpuCompletions completions = new GpuCompletions(device.cuda);
        lattice.attach(completions);
        try (LayerForward forward = new LayerForward(device, checkpoint.config, (int) batchRows);
                ForwardPipeline pipeline =
                        new ForwardPipeline(device, forward, lattice.lake(), completions, settings.batch())) {
            for (int index = 0; index < layers; index++) {
                LayerWeights uploaded;
                try (LayerWeights.Staged staged = LayerWeights.stage(checkpoint, index)) {
                    model.add(hostLayer(index, staged));
                    uploaded = LayerWeights.upload(device.gpu, staged);
                }
                try (LayerWeights layer = uploaded) {
                    pipeline.run(layer, streamAddress, batchStates, sequences, length, settings.batch(), null, true)
                            .join();
                }
                if (index % 8 == 7) log("teacher: layer " + index + " done");
            }
        }
        // The final norm and LM head over every position.
        long head = uploadBf16("lm_head.weight");
        long finalNorm = uploadBf16("model.language_model.norm.weight");
        long normed = device.gpu.allocate(2L * HEAD_ROWS * hidden),
                logits = device.gpu.allocate(4L * HEAD_ROWS * vocabulary);
        long ids = device.gpu.allocate(4L * HEAD_ROWS * k), logProbs = device.gpu.allocate(4L * HEAD_ROWS * k);
        try {
            long positions = (long) sequences * length;
            for (long first = 0; first < positions; first += HEAD_ROWS) {
                int rows = (int) Math.min(HEAD_ROWS, positions - first);
                device.gpu.copyHostToDevice(
                        batchStates, stream.asSlice(first * 2 * hidden, 2L * rows * hidden), 2L * rows * hidden);
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
                device.synchronize();
                device.gpu.copyDeviceToHost(teacherIds.asSlice(4 * first * k, 4L * rows * k), ids, 4L * rows * k);
                device.gpu.copyDeviceToHost(
                        teacherLogProbs.asSlice(4 * first * k, 4L * rows * k), logProbs, 4L * rows * k);
            }
        } finally {
            for (long address : new long[] {head, finalNorm, normed, logits, ids, logProbs, batchStates})
                device.gpu.free(address);
            device.gpu.freeHostWeights(streamAddress);
        }
        log("teacher: top-" + k + " of every position kept");
    }

    /// A layer's starting state in host memory: codes and scales from the codes directory, norms from `norms` or the
    /// checkpoint, the frozen GDN tensors from the checkpoint.
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
        return new HostParameter(
                host.allocate(4 * count, 4),
                host.allocate(4 * count, 4),
                host.allocate(4 * count, 4),
                host.allocate(4 * count, 4));
    }

    private MemorySegment floats(float[] values) {
        MemorySegment segment = host.allocate(4L * values.length, 4);
        MemorySegment.copy(values, 0, segment, ValueLayout.JAVA_FLOAT, 0, values.length);
        return segment;
    }

    private MemorySegment copy(MemorySegment source) {
        MemorySegment segment = host.allocate(source.byteSize(), 16);
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

    /// A trainable layer of one kind with storage for every object, norm and frozen tensor of that kind.
    private TrainableLayer storage(boolean attention) {
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
        var layer = new TrainableLayer(device.gpu, ops, shape);
        for (HostLinear linear : template.linears().values())
            layer.addLinearStorage(linear.role(), linear.out(), linear.in(), 64);
        for (var entry : template.norms().entrySet())
            layer.addNormStorage(
                    entry.getKey(), (int) (entry.getValue().values().byteSize() / 4));
        for (var entry : template.frozen().entrySet())
            layer.addFrozenStorage(entry.getKey(), entry.getValue().byteSize() / 4);
        return layer;
    }

    // ---- the student ----

    private final class Student {
        final TrainableLayer gdn, attention;
        final long head, finalNorm, x, y, dh, dx, normed, dn, inverse, embeddedBf16, logits, dLogits, chunkBf16, ids;
        final long logProbs, loss;
        final MemorySegment saved, embedded, teacherIds, teacherLogProbs;
        final int length;

        Student(
                TrainableLayer gdn,
                TrainableLayer attention,
                long head,
                long finalNorm,
                long x,
                long y,
                long dh,
                long dx,
                long normed,
                long dn,
                long inverse,
                long embeddedBf16,
                long logits,
                long dLogits,
                long chunkBf16,
                long ids,
                long logProbs,
                long loss,
                MemorySegment saved,
                MemorySegment embedded,
                MemorySegment teacherIds,
                MemorySegment teacherLogProbs,
                int length) {
            this.gdn = gdn;
            this.attention = attention;
            this.head = head;
            this.finalNorm = finalNorm;
            this.x = x;
            this.y = y;
            this.dh = dh;
            this.dx = dx;
            this.normed = normed;
            this.dn = dn;
            this.inverse = inverse;
            this.embeddedBf16 = embeddedBf16;
            this.logits = logits;
            this.dLogits = dLogits;
            this.chunkBf16 = chunkBf16;
            this.ids = ids;
            this.logProbs = logProbs;
            this.loss = loss;
            this.saved = saved;
            this.embedded = embedded;
            this.teacherIds = teacherIds;
            this.teacherLogProbs = teacherLogProbs;
            this.length = length;
        }

        /// Fills the layer of `index`'s kind with its state.
        TrainableLayer load(int index) {
            HostLayer state = model.get(index);
            TrainableLayer layer = state.attention() ? attention : gdn;
            for (HostLinear linear : state.linears().values())
                layer.loadLinear(
                        linear.role(),
                        linear.codes(),
                        linear.scales().values(),
                        linear.scales().first(),
                        linear.scales().second());
            for (var entry : state.norms().entrySet())
                layer.loadNorm(
                        entry.getKey(),
                        entry.getValue().values(),
                        entry.getValue().first(),
                        entry.getValue().second());
            for (var entry : state.frozen().entrySet()) layer.loadFrozen(entry.getKey(), entry.getValue());
            return layer;
        }

        /// The model over sequence `s`, keeping each layer's input; leaves the last layer's output in `x`.
        void forward(int s, boolean keep) {
            long count = (long) length * hidden;
            device.gpu.copyHostToDevice(embeddedBf16, embedded.asSlice(s * 2 * count, 2 * count), 2 * count);
            ops.bf16ToFloat(embeddedBf16, count, x);
            for (int index = 0; index < layers; index++) {
                TrainableLayer layer = load(index);
                if (keep) device.gpu.copyDeviceToHost(saved.asSlice(index * 4 * count, 4 * count), x, 4 * count);
                layer.forward(x, y, length, 0);
                device.gpu.copyDeviceToDevice(x, y, 4 * count);
            }
        }

        /// The KL to the teacher at every position of sequence `s` from the last layer's output `x` (its mean, per
        /// position); with `gradient` also its gradient on `x` into `dh`.
        double headLoss(int s, boolean gradient) {
            long count = (long) length * hidden;
            int k = settings.topK();
            ops.rmsNormForward(x, finalNorm, normed, inverse, length, hidden, (float) checkpoint.config.eps(), 1);
            ops.zero(dn, count);
            ops.zero(loss, 2);
            long base = (long) s * length * k;
            device.gpu.copyHostToDevice(ids, teacherIds.asSlice(4 * base, 4L * length * k), 4L * length * k);
            device.gpu.copyHostToDevice(logProbs, teacherLogProbs.asSlice(4 * base, 4L * length * k), 4L * length * k);
            for (int first = 0; first < length; first += HEAD_ROWS) {
                int rows = Math.min(HEAD_ROWS, length - first);
                ops.floatToBf16(normed + 4L * first * hidden, (long) rows * hidden, chunkBf16);
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
                        chunkBf16,
                        hidden,
                        0,
                        logits,
                        vocabulary,
                        false);
                ops.topKDistillation(
                        logits,
                        rows,
                        vocabulary,
                        k,
                        ids + 4L * first * k,
                        logProbs + 4L * first * k,
                        1.0f / length,
                        dLogits,
                        loss);
                if (!gradient) continue;
                ops.floatToBf16(dLogits, (long) rows * vocabulary, chunkBf16);
                device.cuda.gemmBf16(
                        device.cublas,
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
            if (gradient) {
                ops.zero(dh, count);
                ops.rmsNormBackward(x, finalNorm, inverse, dn, dh, 0, length, hidden, 1);
            }
            return read(loss) / length;
        }

        /// Adds layer `index`'s gradients (left on the device by its backward pass) to its gradients in host memory.
        void accumulate(int index, TrainableLayer layer) {
            HostLayer state = model.get(index);
            for (HostLinear linear : state.linears().values())
                add(layer.linears.get(linear.role()).scales, linear.scales().gradient());
            for (var entry : state.norms().entrySet())
                add(layer.norms.get(entry.getKey()), entry.getValue().gradient());
        }

        private void add(TrainableLayer.Parameter parameter, MemorySegment sum) {
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment gradient = arena.allocate(4 * parameter.count, 4);
                device.gpu.copyDeviceToHost(gradient, parameter.gradient, gradient.byteSize());
                for (long i = 0; i < parameter.count; i++)
                    sum.setAtIndex(
                            ValueLayout.JAVA_FLOAT,
                            i,
                            sum.getAtIndex(ValueLayout.JAVA_FLOAT, i) + gradient.getAtIndex(ValueLayout.JAVA_FLOAT, i));
            }
        }

        /// One Adam step of every layer on the gradients accumulated since the last one (their mean over `count`
        /// sequences), which it then clears.
        void apply(int count, int update) {
            for (int index = 0; index < layers; index++) {
                HostLayer state = model.get(index);
                TrainableLayer layer = state.attention() ? attention : gdn;
                for (HostLinear linear : state.linears().values())
                    adam(layer.linears.get(linear.role()).scales, linear.scales(), count, settings.scaleRate(), update);
                for (var entry : state.norms().entrySet())
                    adam(layer.norms.get(entry.getKey()), entry.getValue(), count, settings.normRate(), update);
            }
        }

        private void adam(TrainableLayer.Parameter device, HostParameter state, int count, float rate, int update) {
            long bytes = 4 * device.count;
            ModelTraining.this.device.gpu.copyHostToDevice(device.values, state.values(), bytes);
            ModelTraining.this.device.gpu.copyHostToDevice(device.first, state.first(), bytes);
            ModelTraining.this.device.gpu.copyHostToDevice(device.second, state.second(), bytes);
            ModelTraining.this.device.gpu.copyHostToDevice(device.gradient, state.gradient(), bytes);
            ops.axpby(device.gradient, device.gradient, device.count, 0, 1.0f / count);
            ops.adam(
                    device.values,
                    device.gradient,
                    device.first,
                    device.second,
                    device.count,
                    rate,
                    0.9f,
                    0.999f,
                    1e-12f,
                    0,
                    update);
            ModelTraining.this.device.gpu.copyDeviceToHost(state.values(), device.values, bytes);
            ModelTraining.this.device.gpu.copyDeviceToHost(state.first(), device.first, bytes);
            ModelTraining.this.device.gpu.copyDeviceToHost(state.second(), device.second, bytes);
            state.gradient().fill((byte) 0);
        }

        /// One training step on sequence `s`; returns its mean KL.
        double step(int s, int step) {
            long count = (long) length * hidden;
            forward(s, true);
            double kl = headLoss(s, true);
            for (int index = layers - 1; index >= 0; index--) {
                TrainableLayer layer = load(index);
                device.gpu.copyHostToDevice(x, saved.asSlice(index * 4 * count, 4 * count), 4 * count);
                layer.forward(x, y, length, 0);
                layer.backward(x, dh, dx, 0);
                accumulate(index, layer);
                device.gpu.copyDeviceToDevice(dh, dx, 4 * count);
            }
            return kl;
        }

        double evaluate(int first, int sequences) {
            double total = 0;
            for (int s = first; s < sequences; s++) {
                forward(s, false);
                total += headLoss(s, false);
            }
            return total / (sequences - first);
        }
    }

    private double read(long loss) {
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment value = arena.allocate(8, 8);
            device.gpu.copyDeviceToHost(value, loss, 8);
            return value.get(ValueLayout.JAVA_DOUBLE, 0);
        }
    }

    // ---- output ----

    /// Writes every layer's objects with their trained scales (float16) and its norms, as `train-blocks` does: the
    /// state of the best held-out KL so far.
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
