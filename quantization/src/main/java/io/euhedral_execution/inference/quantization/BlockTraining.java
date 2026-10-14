package io.euhedral_execution.inference.quantization;

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
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/// Block-wise training of a quantized artifact's layers (block reconstruction): the calibration sequences run through
/// the BF16 checkpoint (the target stream) and through the layers decoded from a codes directory (the quantized
/// stream, as `calibrate` wrote the codes). At each layer to train, the layer's quantized linears train their group
/// scales and its norms their weights ([TrainableLayer]) so that the layer on the quantized stream's input matches the
/// BF16 layer's output on the target stream's input, by squared error; held-out sequences measure the error before
/// and after. The codes stay as GPTQ chose them.
public final class BlockTraining {

    /// Command-line settings: which layers train, on how many sequences (the last `holdout` held out), how long and
    /// how fast.
    public record Settings(
            Path model,
            Path codes,
            Path tokens,
            Set<Integer> layers,
            int sequences,
            int holdout,
            int epochs,
            float scaleRate,
            float normRate,
            int batch) {}

    private final Settings settings;
    private final Device device;
    private final Qwen38Objects checkpoint;
    private final Ops ops;
    private final int hidden;
    private final long start = System.nanoTime();

    private BlockTraining(Settings settings, Device device, Qwen38Objects checkpoint) {
        this.settings = settings;
        this.device = device;
        this.checkpoint = checkpoint;
        this.ops = new Ops(device.cuda, device.stream, device.cublas);
        this.hidden = checkpoint.config.hidden();
    }

    public static void run(Settings settings, Path library) throws IOException {
        try (Safetensors store = Safetensors.open(settings.model());
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(6)) {
            new BlockTraining(settings, device, new Qwen38Objects(store, settings.model())).train(lattice);
        }
    }

    private void log(String message) {
        System.out.printf(Locale.ROOT, "[%7.0f s] %s%n", (System.nanoTime() - start) / 1e9, message);
    }

    private void train(CalibrationLattice lattice) throws IOException {
        int[][] tokens = Calibrate.readTokens(settings.tokens(), settings.sequences());
        int sequences = tokens.length, length = tokens[0].length;
        long sequenceBytes = 2L * hidden * length, streamBytes = sequenceBytes * sequences;
        long targetAddress = device.gpu.allocateHostWeights(streamBytes);
        long quantizedAddress = device.gpu.allocateHostWeights(streamBytes);
        MemorySegment target = MemorySegment.ofAddress(targetAddress).reinterpret(streamBytes);
        MemorySegment quantized = MemorySegment.ofAddress(quantizedAddress).reinterpret(streamBytes);
        long batchRows = (long) settings.batch() * length;
        long batchStates = device.gpu.allocate(batchRows * 2 * hidden);
        GpuCompletions completions = new GpuCompletions(device.cuda);
        lattice.attach(completions);
        int last = settings.layers().stream().mapToInt(Integer::intValue).max().orElseThrow();
        try (LayerForward forward = new LayerForward(device, checkpoint.config, (int) batchRows);
                ForwardPipeline pipeline =
                        new ForwardPipeline(device, forward, lattice.lake(), completions, settings.batch())) {
            embed(tokens, target, batchStates);
            MemorySegment.copy(target, 0, quantized, 0, streamBytes);
            for (int index = 0; index <= last; index++) {
                LayerWeights uploaded;
                try (LayerWeights.Staged staged = LayerWeights.stage(checkpoint, index)) {
                    uploaded = LayerWeights.upload(device.gpu, staged);
                }
                try (LayerWeights layer = uploaded) {
                    // The target stream: the BF16 layer.
                    pipeline.run(layer, targetAddress, batchStates, sequences, length, settings.batch(), null, true)
                            .join();
                    boolean trains = settings.layers().contains(index);
                    if (trains) trainLayer(index, layer, target, quantized, sequences, length);
                    // The quantized stream: the layer decoded from its codes (untrained).
                    for (String name : List.copyOf(layer.objects.keySet())) layer.replace(name, decode(name));
                    pipeline.run(layer, quantizedAddress, batchStates, sequences, length, settings.batch(), null, true)
                            .join();
                    if (trains) {
                        log(String.format(
                                Locale.ROOT,
                                "layer %d: the engine's quantized layer (BF16) gives held-out squared error %.6g",
                                index,
                                heldOutError(quantized, target, sequences, length)));
                    }
                }
                log("layer " + index + " done");
            }
        } finally {
            device.gpu.free(batchStates);
            device.gpu.freeHostWeights(targetAddress);
            device.gpu.freeHostWeights(quantizedAddress);
        }
    }

    // ---- training one layer ----

    private void trainLayer(
            int index, LayerWeights layer, MemorySegment target, MemorySegment quantized, int sequences, int length)
            throws IOException {
        var c = checkpoint.config;
        boolean attention = c.fullAttention()[index];
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
        try (TrainableLayer trainable = new TrainableLayer(device.gpu, ops, shape);
                Arena arena = Arena.ofShared()) {
            List<Long> temporaries = new java.util.ArrayList<>(), codesOwned = new java.util.ArrayList<>();
            for (String name : layer.objects.keySet()) {
                QuantState state = QuantState.load(QuantState.stateFile(settings.codes(), name), arena);
                if (state.formatName().equals("NVFP4"))
                    throw new IllegalArgumentException(name + " is NVFP4: block training takes grouped formats");
                long codes = device.gpu.allocate(state.rows() * state.columns());
                codesOwned.add(codes);
                Upload.segment(device.gpu, codes, state.codes(), state.rows() * state.columns());
                long scales = uploadHalfScales(state, temporaries);
                trainable.addLinear(
                        name.substring(layer.prefix.length()),
                        (int) state.rows(),
                        (int) state.columns(),
                        64,
                        codes,
                        scales);
            }
            addNorm(trainable, "input_norm", layer.inputNorm, hidden, temporaries);
            addNorm(trainable, "post_norm", layer.postNorm, hidden, temporaries);
            if (attention) {
                addNorm(trainable, "query_norm", layer.queryNorm, c.headDim(), temporaries);
                addNorm(trainable, "key_norm", layer.keyNorm, c.headDim(), temporaries);
            } else {
                addNorm(trainable, "gdn_norm", layer.gdnNorm, c.valueDim(), temporaries);
                addFrozenBf16(
                        trainable, "a_projection", layer.aProjection, (long) c.valueHeads() * hidden, temporaries);
                addFrozenBf16(
                        trainable, "b_projection", layer.bProjection, (long) c.valueHeads() * hidden, temporaries);
                trainable.addFrozen("a_log", layer.aLog, c.valueHeads());
                trainable.addFrozen("dt_bias", layer.dtBias, c.valueHeads());
                long conv = (long) c.convKernel() * shape.convWidth();
                addFrozenBf16(trainable, "convolution", layer.convolution, conv, temporaries);
            }
            for (long address : temporaries) device.gpu.free(address);
            try {
                fit(index, trainable, target, quantized, sequences, length);
            } finally {
                for (long address : codesOwned) device.gpu.free(address);
            }
        }
    }

    /// A state's float16 scales as FP32 on the device (listed in `temporaries`).
    private long uploadHalfScales(QuantState state, List<Long> temporaries) {
        long count = state.scales().byteSize() / 2;
        float[] values = new float[(int) count];
        for (int i = 0; i < count; i++)
            values[i] = Float.float16ToFloat(state.scales().getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i));
        long scales = device.gpu.allocate(4 * count);
        temporaries.add(scales);
        Upload.floats(device.gpu, scales, values);
        return scales;
    }

    private void addNorm(TrainableLayer trainable, String role, long bf16, int width, List<Long> temporaries) {
        long values = device.gpu.allocate(4L * width);
        temporaries.add(values);
        ops.bf16ToFloat(bf16, width, values);
        trainable.addNorm(role, values, width);
    }

    private void addFrozenBf16(TrainableLayer trainable, String role, long bf16, long count, List<Long> temporaries) {
        long values = device.gpu.allocate(4 * count);
        temporaries.add(values);
        ops.bf16ToFloat(bf16, count, values);
        trainable.addFrozen(role, values, count);
    }

    /// Trains `trainable` on the first sequences (the quantized stream's input, the target stream's output) and
    /// reports the held-out squared error before and after each epoch.
    private void fit(
            int index,
            TrainableLayer trainable,
            MemorySegment target,
            MemorySegment quantized,
            int sequences,
            int length) {
        long rows = length, count = rows * hidden, sequenceBytes = 2 * count;
        long inputBf16 = device.gpu.allocate(sequenceBytes), targetBf16 = device.gpu.allocate(sequenceBytes);
        long x = device.gpu.allocate(4 * count), y = device.gpu.allocate(4 * count);
        long expected = device.gpu.allocate(4 * count), dy = device.gpu.allocate(4 * count);
        long loss = device.gpu.allocate(8);
        int training = sequences - settings.holdout();
        try {
            double before = evaluate(
                    trainable,
                    target,
                    quantized,
                    training,
                    sequences,
                    length,
                    inputBf16,
                    targetBf16,
                    x,
                    y,
                    expected,
                    loss);
            log(String.format(Locale.ROOT, "layer %d: held-out squared error %.6g before training", index, before));
            int step = 0;
            for (int epoch = 1; epoch <= settings.epochs(); epoch++) {
                double total = 0;
                for (int s = 0; s < training; s++) {
                    load(quantized, s, sequenceBytes, inputBf16, x, count);
                    load(target, s, sequenceBytes, targetBf16, expected, count);
                    trainable.forward(x, y, (int) rows, 0);
                    ops.zero(loss, 2);
                    ops.squaredError(y, expected, count, 1.0f / count, dy, loss);
                    trainable.backward(x, dy, 0, 0);
                    step++;
                    for (TrainableLayer.Parameter parameter : trainable.parameters()) {
                        float rate = parameter.name.endsWith("/scales") ? settings.scaleRate() : settings.normRate();
                        ops.adam(
                                parameter.values,
                                parameter.gradient,
                                parameter.first,
                                parameter.second,
                                parameter.count,
                                rate,
                                0.9f,
                                0.999f,
                                1e-12f,
                                0,
                                step);
                    }
                    total += read(loss);
                }
                double after = evaluate(
                        trainable,
                        target,
                        quantized,
                        training,
                        sequences,
                        length,
                        inputBf16,
                        targetBf16,
                        x,
                        y,
                        expected,
                        loss);
                log(String.format(
                        Locale.ROOT,
                        "layer %d epoch %d: training squared error %.6g, held-out %.6g (%+.1f%%)",
                        index,
                        epoch,
                        total / training,
                        after,
                        100 * (after - before) / before));
            }
        } finally {
            for (long address : new long[] {inputBf16, targetBf16, x, y, expected, dy, loss}) device.gpu.free(address);
        }
    }

    private double evaluate(
            TrainableLayer trainable,
            MemorySegment target,
            MemorySegment quantized,
            int first,
            int sequences,
            int length,
            long inputBf16,
            long targetBf16,
            long x,
            long y,
            long expected,
            long loss) {
        long count = (long) length * hidden, sequenceBytes = 2 * count;
        double total = 0;
        for (int s = first; s < sequences; s++) {
            load(quantized, s, sequenceBytes, inputBf16, x, count);
            load(target, s, sequenceBytes, targetBf16, expected, count);
            trainable.forward(x, y, length, 0);
            ops.zero(loss, 2);
            ops.squaredError(y, expected, count, 1.0f / count, 0, loss);
            total += read(loss) / count;
        }
        return total / (sequences - first);
    }

    /// The squared error of the quantized stream against the target stream over the held-out sequences (per value).
    private double heldOutError(MemorySegment quantized, MemorySegment target, int sequences, int length) {
        long count = (long) length * hidden, sequenceBytes = 2 * count;
        long a = device.gpu.allocate(sequenceBytes), b = device.gpu.allocate(sequenceBytes);
        long x = device.gpu.allocate(4 * count), t = device.gpu.allocate(4 * count), loss = device.gpu.allocate(8);
        try {
            double total = 0;
            for (int s = sequences - settings.holdout(); s < sequences; s++) {
                load(quantized, s, sequenceBytes, a, x, count);
                load(target, s, sequenceBytes, b, t, count);
                ops.zero(loss, 2);
                ops.squaredError(x, t, count, 1, 0, loss);
                total += read(loss) / count;
            }
            return total / settings.holdout();
        } finally {
            for (long address : new long[] {a, b, x, t, loss}) device.gpu.free(address);
        }
    }

    private void load(MemorySegment stream, int sequence, long sequenceBytes, long bf16, long floats, long count) {
        device.gpu.copyHostToDevice(bf16, stream.asSlice(sequence * sequenceBytes, sequenceBytes), sequenceBytes);
        ops.bf16ToFloat(bf16, count, floats);
    }

    private double read(long loss) {
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(8, 8);
            device.gpu.copyDeviceToHost(host, loss, 8);
            return host.get(ValueLayout.JAVA_DOUBLE, 0);
        }
    }

    // ---- the quantized stream's weights ----

    /// Object `name` decoded from its codes to BF16 on the device.
    private long decode(String name) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            QuantState state = QuantState.load(QuantState.stateFile(settings.codes(), name), arena);
            long n = state.rows(), k = state.columns();
            long codes = device.gpu.allocate(n * k), scales = 0, floats = device.gpu.allocate(4 * n * k);
            try {
                Upload.segment(device.gpu, codes, state.codes(), n * k);
                List<Long> scratch = new java.util.ArrayList<>();
                scales = uploadHalfScales(state, scratch);
                ops.dequantize(codes, scales, floats, n, (int) k, 64);
                long bf16 = device.gpu.allocate(2 * n * k);
                ops.floatToBf16(floats, n * k, bf16);
                device.synchronize();
                return bf16;
            } finally {
                device.gpu.free(codes);
                device.gpu.free(floats);
                if (scales != 0) device.gpu.free(scales);
            }
        }
    }

    // ---- embedding ----

    private void embed(int[][] tokens, MemorySegment states, long batchStates) throws IOException {
        var tensor = checkpoint.store().tensor("model.language_model.embed_tokens.weight");
        long table = device.gpu.allocate(tensor.bytes());
        Cuda.Function gather =
                device.cuda.compileResource("rows.cu", List.of("gather_rows")).getFirst();
        try (Arena arena = Arena.ofConfined()) {
            long chunk = 1L << 28;
            MemorySegment host = arena.allocate(chunk, 16);
            for (long at = 0; at < tensor.bytes(); at += chunk) {
                long bytes = Math.min(chunk, tensor.bytes() - at);
                checkpoint.store().read(tensor.name(), at, host.asSlice(0, bytes));
                device.gpu.copyHostToDevice(table + at, host.asSlice(0, bytes), bytes);
            }
            int length = tokens[0].length;
            long sequenceBytes = 2L * hidden * length, ids = device.gpu.allocate(4L * length);
            for (int s = 0; s < tokens.length; s++) {
                Upload.ints(device.gpu, ids, tokens[s]);
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
                                .pointer(ids)
                                .pointer(length)
                                .integer(hidden)
                                .pointer(batchStates));
                device.synchronize();
                device.gpu.copyDeviceToHost(
                        states.asSlice(s * sequenceBytes, sequenceBytes), batchStates, sequenceBytes);
            }
            device.gpu.free(ids);
        } finally {
            device.gpu.free(table);
        }
        log(tokens.length + " sequences of " + tokens[0].length + " tokens embedded");
    }
}
