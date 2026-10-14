package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gptq.Gptq;
import io.euhedral_execution.inference.quantization.gptq.QuantState;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import io.euhedral_execution.inference.quantization.lattice.GpuCompletions;
import io.euhedral_execution.inference.quantization.model.ForwardPipeline;
import io.euhedral_execution.inference.quantization.model.LayerForward;
import io.euhedral_execution.inference.quantization.model.LayerWeights;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/// GPTQ calibration of an artifact recipe on the engine's kernels: writes each quantized object's rounding to
/// OUT as the .npz files `convert_checkpoint.py --codes-from` takes.
///
/// The calibration sequences run through the checkpoint one decoder layer at a time: the layer's quantized
/// linears' input rows accumulate into their Hessians, every object is rounded by GPTQ in the format the recipe
/// stores it, the decoded weights replace the layer's, and the sequences run through it again to give the next
/// layer its inputs. Then the LM head is rounded under the final norm's output, the MTP layer on the rows the
/// engine feeds it (the normed hidden at p with the embedding of token p + 1), and the draft head under the MTP
/// output it reads. The hidden states stay in pinned host memory; every 8 layers they are checkpointed, and a
/// run with the same settings resumes there.
public final class Calibrate {

    /// Command-line settings.
    public record Settings(
            Path model,
            Recipe recipe,
            Path tokens,
            Path draftIdsFrom,
            Path out,
            boolean actOrder,
            float damp,
            int batch,
            int sequences,
            int layers,
            boolean skipMtp,
            Gptq.Nvfp4Search nvfp4) {}

    private static final int HEAD_ROWS = 16384;
    private static final int CHECKPOINT_EVERY = 8;
    private static final long CHECKPOINT_SLICE = 1L << 30;
    /// Objects of a layer rounded at once, each on its own stream (largest first): three fit a 16 GB card
    /// beside the layer and its Hessians.
    private static final int ROUNDING_FRAMES = 3;
    private static final int WORKERS = 6;

    private final Settings settings;
    private final Device device;
    private final CalibrationLattice lattice;
    /// The decoder layers' passes as lattice frames (set while calibrate runs).
    private ForwardPipeline pipeline;
    private final Map<Long, Gptq> streamGptq = new ConcurrentHashMap<>();
    private final Qwen38Objects checkpoint;
    private final Gptq gptq;
    private final Cuda.Function gatherRows;
    private final int hidden;
    private final long start = System.nanoTime();

    private Calibrate(Settings settings, Device device, Qwen38Objects checkpoint, CalibrationLattice lattice) {
        this.settings = settings;
        this.device = device;
        this.lattice = lattice;
        this.checkpoint = checkpoint;
        this.gptq = new Gptq(device.cuda, device.gpu, device.stream, settings.nvfp4());
        this.gatherRows =
                device.cuda.compileResource("rows.cu", List.of("gather_rows")).getFirst();
        this.hidden = checkpoint.config.hidden();
    }

    /// Starts the calibration on the lattice and waits for it: the setup, then each layer continuing the next,
    /// then the heads, all on the lattice's workers.
    public static void run(Settings settings, Path library) throws IOException {
        try (Safetensors store = Safetensors.open(settings.model());
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(WORKERS)) {
            var checkpoint = new Qwen38Objects(store, settings.model());
            lattice.submit(() -> {
                        try {
                            Files.createDirectories(settings.out());
                            device.select(device.stream);
                            return new Calibrate(settings, device, checkpoint, lattice).calibrate();
                        } catch (IOException failure) {
                            throw new UncheckedIOException(failure);
                        }
                    })
                    .thenCompose(chain -> chain)
                    .get();
        } catch (InterruptedException | java.util.concurrent.ExecutionException failure) {
            throw new IOException(failure);
        }
    }

    private void log(String message) {
        System.out.printf("[%7.0f s] %s%n", (System.nanoTime() - start) / 1e9, message);
    }

    /// Sets the calibration up and returns its chain: each layer once its read is done (the next one read while
    /// it runs), checkpoints, then the heads.
    private CompletableFuture<Void> calibrate() throws IOException {
        int[][] tokens = readTokens(settings.tokens(), settings.sequences());
        int sequences = tokens.length, length = tokens[0].length, batch = settings.batch();
        long rowBytes = 2L * hidden;
        long statesBytes = (long) sequences * length * rowBytes;
        long statesAddress = device.gpu.allocateHostWeights(statesBytes);
        MemorySegment states = MemorySegment.ofAddress(statesAddress).reinterpret(statesBytes);
        long batchRows = (long) batch * length;
        long batchStates = device.gpu.allocate(batchRows * rowBytes);
        GpuCompletions completions = new GpuCompletions(device.cuda);
        lattice.attach(completions);
        LayerForward forward = new LayerForward(device, checkpoint.config, (int) batchRows);
        ForwardPipeline pipeline = new ForwardPipeline(device, forward, lattice.lake(), completions, batch);
        this.pipeline = pipeline;
        int first = resume(states, sequences, length);
        if (first == 0) embed(tokens, states, batchStates);
        var reading = new AtomicReference<CompletableFuture<LayerWeights.Staged>>(stageLayer(first));
        CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
        for (int i = first; i < settings.layers(); i++) {
            int index = i;
            chain = chain.thenCompose(ignored -> reading.get()).thenCompose(staged -> {
                if (index + 1 < settings.layers()) reading.set(stageLayer(index + 1)); // read while this one runs
                return calibrateLayer(index, staged, forward, states, batchStates, sequences, length)
                        .thenRun(() -> {
                            if ((index + 1) % CHECKPOINT_EVERY == 0 && index + 1 < settings.layers()) {
                                try {
                                    saveCheckpoint(states, index + 1, sequences, length);
                                } catch (IOException failure) {
                                    throw new UncheckedIOException(failure);
                                }
                            }
                        });
            });
        }
        return chain.thenRun(() -> {
                    try {
                        device.select(device.stream);
                        dropCheckpoint();
                        heads(tokens, forward, states, batchStates, sequences, length);
                    } catch (IOException failure) {
                        throw new UncheckedIOException(failure);
                    }
                    log("done");
                })
                .whenComplete((ignored, failure) -> {
                    device.select(device.stream);
                    pipeline.close();
                    forward.close();
                    device.gpu.free(batchStates);
                    device.gpu.freeHostWeights(statesAddress);
                });
    }

    // ---- decoder layers ----

    /// Reads decoder layer `index` into host memory on a lattice worker.
    private CompletableFuture<LayerWeights.Staged> stageLayer(int index) {
        return lattice.submit(() -> {
            try {
                return LayerWeights.stage(checkpoint, index);
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        });
    }

    /// Layer `index`: its Hessians observed, its objects rounded and swapped in, then its pass over the states.
    private CompletableFuture<Void> calibrateLayer(
            int index,
            LayerWeights.Staged staged,
            LayerForward forward,
            MemorySegment states,
            long batchStates,
            int sequences,
            int length) {
        device.select(device.stream);
        LayerWeights layer;
        try (staged) {
            layer = LayerWeights.upload(device.gpu, staged);
        }
        return observeAll(layer, forward, states, batchStates, sequences, length)
                .thenCompose(hessians -> roundAll(layer, hessians))
                .thenCompose(ignored -> pipeline.run(
                        layer, states.address(), batchStates, sequences, length, settings.batch(), null, true))
                .whenComplete((ignored, failure) -> {
                    layer.close();
                    if (failure == null) log("layer " + index + " done");
                });
    }

    /// Runs the layer over every batch (its outputs discarded), accumulating each quantized object's Hessian;
    /// objects that read the same rows in the same call share one.
    private CompletableFuture<Map<String, Long>> observeAll(
            LayerWeights layer,
            LayerForward forward,
            MemorySegment states,
            long batchStates,
            int sequences,
            int length) {
        Map<String, Long> hessians = new LinkedHashMap<>();
        Map<String, String> shares = new HashMap<>();
        long[] previous = {0, 0};
        String[] previousName = {null};
        LayerForward.Observer observer = (name, rows, count, columns) -> {
            if (shares.containsKey(name)) return;
            if (!hessians.containsKey(name)) {
                if (previous[0] == rows && previous[1] == count && previousName[0] != null) {
                    shares.put(name, previousName[0]);
                    return;
                }
                long h = device.gpu.allocate(4L * columns * columns);
                device.gpu.zeroDeviceMemory(h, 4L * columns * columns);
                hessians.put(name, h);
            }
            device.accumulateHessian(hessians.get(name), rows, count, columns);
            previous[0] = rows;
            previous[1] = count;
            previousName[0] = name;
        };
        return pipeline.run(layer, states.address(), batchStates, sequences, length, settings.batch(), observer, false)
                .thenApply(ignored -> {
                    for (var share : shares.entrySet()) hessians.put(share.getKey(), hessians.get(share.getValue()));
                    return hessians;
                });
    }

    /// Rounds every quantized object of `layer` under its Hessian, saves its state and swaps in its decoded
    /// weights: each object is a lattice frame on its worker's own stream, at most ROUNDING_FRAMES at once,
    /// largest first.
    private CompletableFuture<Void> roundAll(LayerWeights layer, Map<String, Long> hessians) {
        device.select(device.stream);
        var mirrored = new HashSet<Long>();
        for (String name : layer.objects.keySet()) {
            long h = hessians.get(name);
            if (mirrored.add(h)) device.symmetrize(h, (int) layer.shapes.get(name)[1]);
        }
        device.synchronize(); // the Hessians are complete before other streams read them
        List<String> names = new ArrayList<>(layer.objects.keySet());
        names.sort(Comparator.comparingLong(
                (String name) -> -layer.shapes.get(name)[0] * layer.shapes.get(name)[1]));
        Map<String, Long> decoded = new ConcurrentHashMap<>();
        var next = new AtomicInteger();
        var failure = new AtomicReference<Throwable>();
        List<CompletableFuture<Void>> lanes = new ArrayList<>();
        for (int lane = 0; lane < Math.min(ROUNDING_FRAMES, names.size()); lane++) {
            lanes.add(lane(names, next, layer, hessians, decoded, failure));
        }
        return CompletableFuture.allOf(lanes.toArray(CompletableFuture[]::new)).thenRun(() -> {
            if (failure.get() != null) throw new IllegalStateException("rounding failed", failure.get());
            device.select(device.stream);
            for (var entry : decoded.entrySet()) layer.replace(entry.getKey(), entry.getValue());
            var freed = new HashSet<Long>();
            for (long h : hessians.values()) if (freed.add(h)) device.gpu.free(h);
        });
    }

    /// One rounding lane: rounds the next object in a frame, then continues with the next until none is left.
    private CompletableFuture<Void> lane(
            List<String> names,
            AtomicInteger next,
            LayerWeights layer,
            Map<String, Long> hessians,
            Map<String, Long> decoded,
            AtomicReference<Throwable> failure) {
        int index = next.getAndIncrement();
        if (index >= names.size() || failure.get() != null) return CompletableFuture.completedFuture(null);
        String name = names.get(index);
        return lattice.<Void>submit(() -> {
                    long stream = device.bindOwnStream();
                    Gptq own = streamGptq.computeIfAbsent(
                            stream, s -> new Gptq(device.cuda, device.gpu, s, settings.nvfp4()));
                    long[] shape = layer.shapes.get(name);
                    try {
                        decoded.put(
                                name,
                                roundObject(
                                        own,
                                        name,
                                        layer.objects.get(name),
                                        shape[0],
                                        (int) shape[1],
                                        hessians.get(name)));
                    } catch (Throwable error) {
                        failure.compareAndSet(null, error);
                    }
                    return null;
                })
                .thenCompose(ignored -> lane(names, next, layer, hessians, decoded, failure));
    }

    /// GPTQ of one object (BF16 [rows, k] on the device) under `hessian`, in row slices of HEAD_ROWS sharing
    /// one factor and tensor scales; saves its state and returns its decoded weights (BF16, device, owned by
    /// the caller).
    private long roundObject(String name, long weight, long rows, int k, long hessian) throws IOException {
        device.symmetrize(hessian, k); // on the main stream, where it was accumulated and the main Gptq reads it
        return roundObject(gptq, name, weight, rows, k, hessian);
    }

    private long roundObject(Gptq gptq, String name, long weight, long rows, int k, long hessian) throws IOException {
        Gptq.Format format = settings.recipe().format(name);
        Gptq.Factor factor = gptq.factor(hessian, k, settings.damp(), settings.actOrder());
        long decoded = device.gpu.allocate(2L * rows * k);
        try (Arena arena = Arena.ofConfined()) {
            Gptq.TensorScales tensor = format.nvfp4() ? gptq.tensorScales(weight, rows, k, format, factor) : null;
            int groups = k / format.group();
            MemorySegment codes = arena.allocate(rows * k);
            MemorySegment scales = arena.allocate((format.nvfp4() ? 1L : 2L) * rows * groups);
            QuantState last = null;
            for (long begin = 0; begin < rows; begin += HEAD_ROWS) {
                int count = (int) Math.min(HEAD_ROWS, rows - begin);
                try (Arena slice = Arena.ofConfined()) {
                    last = gptq.round(
                            weight + 2L * begin * k, count, factor, format, tensor, slice, decoded + 2L * begin * k);
                    MemorySegment.copy(
                            last.codes(), 0, codes, begin * k, last.codes().byteSize());
                    MemorySegment.copy(
                            last.scales(),
                            0,
                            scales,
                            (format.nvfp4() ? 1L : 2L) * begin * groups,
                            last.scales().byteSize());
                }
            }
            new QuantState(format.name(), last.layout(), rows, k, codes, scales, last.globalScale(), last.table())
                    .save(QuantState.stateFile(settings.out(), name));
            log(name + " " + format.name() + "/" + last.layout());
        } finally {
            device.gpu.free(factor.upper());
        }
        return decoded;
    }

    /// Runs `step` on each batch of whole sequences, copied into `batchStates`; with `write`, copies the result
    /// back. `states` is pinned, so the copies queue on the stream without stopping the host.
    private void batches(
            MemorySegment states,
            long batchStates,
            int sequences,
            int length,
            boolean write,
            java.util.function.IntConsumer step) {
        long sequenceBytes = 2L * hidden * length;
        for (int s = 0; s < sequences; s += settings.batch()) {
            int count = Math.min(settings.batch(), sequences - s);
            long host = states.address() + s * sequenceBytes, bytes = count * sequenceBytes;
            device.gpu.copyHostToDeviceAsync(batchStates, host, bytes);
            step.accept(count);
            if (write) device.gpu.copyDeviceToHostAsync(host, batchStates, bytes);
        }
        device.synchronize();
    }

    // ---- embedding, heads and MTP ----

    private long embeddingTable() throws IOException {
        var tensor = checkpoint.store().tensor("model.language_model.embed_tokens.weight");
        long table = device.gpu.allocate(tensor.bytes());
        long chunk = 1L << 28;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(chunk, 16);
            for (long at = 0; at < tensor.bytes(); at += chunk) {
                long bytes = Math.min(chunk, tensor.bytes() - at);
                checkpoint.store().read(tensor.name(), at, host.asSlice(0, bytes));
                device.gpu.copyHostToDevice(table + at, host.asSlice(0, bytes), bytes);
            }
        }
        return table;
    }

    private void gather(long table, int[] ids, long out) {
        long idsDevice = device.gpu.allocate(4L * ids.length);
        Upload.ints(device.gpu, idsDevice, ids);
        long vectors = (long) ids.length * hidden / 8;
        device.cuda.launch(
                gatherRows,
                (int) ((vectors + 255) / 256),
                1,
                256,
                0,
                device.stream,
                new Cuda.Arguments()
                        .pointer(table)
                        .pointer(idsDevice)
                        .pointer(ids.length)
                        .integer(hidden)
                        .pointer(out));
        device.synchronize();
        device.gpu.free(idsDevice);
    }

    private void embed(int[][] tokens, MemorySegment states, long batchStates) throws IOException {
        long table = embeddingTable();
        try {
            int length = tokens[0].length;
            long sequenceBytes = 2L * hidden * length;
            for (int s = 0; s < tokens.length; s++) {
                gather(table, tokens[s], batchStates);
                device.gpu.copyDeviceToHost(
                        states.asSlice(s * sequenceBytes, sequenceBytes), batchStates, sequenceBytes);
            }
        } finally {
            device.gpu.free(table);
        }
        log(tokens.length + " sequences of " + tokens[0].length + " tokens embedded");
    }

    private void heads(
            int[][] tokens, LayerForward forward, MemorySegment states, long batchStates, int sequences, int length)
            throws IOException {
        var c = checkpoint.config;
        float eps = (float) c.eps();
        long finalNorm = uploadTensor("model.language_model.norm.weight");
        long headHessian = zeroed(4L * hidden * hidden);
        long draftHessian = headHessian;
        long normed = device.gpu.allocate(2L * settings.batch() * length * hidden);
        try {
            if (settings.skipMtp()) {
                batches(states, batchStates, sequences, length, false, count -> {
                    device.gpu.rmsNormUnitOffsetBf16(batchStates, finalNorm, normed, count * length, hidden, eps);
                    device.accumulateHessian(headHessian, normed, count * length, hidden);
                });
            } else {
                draftHessian =
                        mtp(tokens, forward, states, batchStates, normed, finalNorm, headHessian, sequences, length);
            }
            var head = checkpoint.object("text/output_head");
            long headWeights = uploadObject(head);
            try {
                device.gpu.free(
                        roundObject("text/output_head", headWeights, head.rows(), (int) head.columns(), headHessian));
                int[] ids = draftIds();
                long draft = device.gpu.allocate(2L * ids.length * hidden);
                try {
                    gather(headWeights, ids, draft);
                    device.gpu.free(roundObject("text/draft_head", draft, ids.length, hidden, draftHessian));
                } finally {
                    device.gpu.free(draft);
                }
            } finally {
                device.gpu.free(headWeights);
            }
        } finally {
            device.gpu.free(normed);
            device.gpu.free(finalNorm);
            device.gpu.free(headHessian);
            if (draftHessian != headHessian) device.gpu.free(draftHessian);
        }
    }

    /// Accumulates the LM head's Hessian, rounds the MTP layer and returns the draft head's Hessian (the MTP
    /// output's).
    private long mtp(
            int[][] tokens,
            LayerForward forward,
            MemorySegment states,
            long batchStates,
            long normed,
            long finalNorm,
            long headHessian,
            int sequences,
            int length)
            throws IOException {
        var c = checkpoint.config;
        float eps = (float) c.eps();
        int rows = length - 1; // row p pairs the hidden at p with token p + 1
        long table = embeddingTable();
        long embeddingNorm = uploadTensor("mtp.pre_fc_norm_embedding.weight");
        long hiddenNorm = uploadTensor("mtp.pre_fc_norm_hidden.weight");
        long mtpNorm = uploadTensor("mtp.norm.weight");
        long batchRows = (long) settings.batch() * rows;
        long stem = device.gpu.allocate(4L * batchRows * hidden); // [rows, 2 * hidden]
        long scratch = device.gpu.allocate(2L * batchRows * hidden);
        long scratchNormed = device.gpu.allocate(2L * batchRows * hidden);
        long stemHessian = zeroed(4L * 4 * hidden * hidden);
        long rowBytes = 2L * hidden;
        try (LayerWeights layer = LayerWeights.mtp(device.gpu, checkpoint)) {
            var projection = checkpoint.object("mtp/input_projection");
            long projectionWeights = uploadObject(projection);
            // The stem of each batch: [RMSNorm(embedding of p + 1, embedding norm); RMSNorm(normed hidden p, hidden
            // norm)].
            java.util.function.BiConsumer<Integer, Integer> buildStem = (s, count) -> {
                device.gpu.rmsNormUnitOffsetBf16(batchStates, finalNorm, normed, count * length, hidden, eps);
                for (int q = 0; q < count; q++) {
                    device.gpu.copyDeviceToDevice(
                            scratch + q * rows * rowBytes, normed + (long) q * length * rowBytes, rows * rowBytes);
                }
                device.gpu.rmsNormUnitOffsetBf16(scratch, hiddenNorm, scratchNormed, count * rows, hidden, eps);
                device.gpu.copyRowsDeviceToDevice(stem + rowBytes, 2 * rowBytes, scratchNormed, rowBytes, count * rows);
                int[] next = new int[count * rows];
                for (int q = 0; q < count; q++) System.arraycopy(tokens[s + q], 1, next, q * rows, rows);
                gather(table, next, scratch);
                device.gpu.rmsNormUnitOffsetBf16(scratch, embeddingNorm, scratchNormed, count * rows, hidden, eps);
                device.gpu.copyRowsDeviceToDevice(stem, 2 * rowBytes, scratchNormed, rowBytes, count * rows);
            };
            long sequenceBytes = rowBytes * length;
            for (int s = 0; s < sequences; s += settings.batch()) {
                int count = Math.min(settings.batch(), sequences - s);
                MemorySegment slice = states.asSlice(s * sequenceBytes, count * sequenceBytes);
                device.gpu.copyHostToDevice(batchStates, slice, slice.byteSize());
                buildStem.accept(s, count);
                device.accumulateHessian(headHessian, normed, count * length, hidden);
                device.accumulateHessian(stemHessian, stem, count * rows, 2 * hidden);
            }
            long projectionDecoded = roundObject(
                    "mtp/input_projection",
                    projectionWeights,
                    projection.rows(),
                    (int) projection.columns(),
                    stemHessian);
            device.gpu.free(projectionWeights);
            // The MTP rows (input projection of the stem) replace the hidden states, `rows` per sequence.
            for (int s = 0; s < sequences; s += settings.batch()) {
                int count = Math.min(settings.batch(), sequences - s);
                MemorySegment slice = states.asSlice(s * sequenceBytes, count * sequenceBytes);
                device.gpu.copyHostToDevice(batchStates, slice, slice.byteSize());
                buildStem.accept(s, count);
                device.linear(stem, projectionDecoded, batchStates, count * rows, 2 * hidden, hidden);
                device.gpu.copyDeviceToHost(
                        slice.asSlice(0, count * rows * rowBytes), batchStates, count * rows * rowBytes);
            }
            device.gpu.free(projectionDecoded);
            device.gpu.free(table);
            Map<String, Long> hessians = observeMtp(layer, forward, states, batchStates, sequences, rows);
            roundAll(layer, hessians);
            long draftHessian = zeroed(4L * hidden * hidden);
            long packed = (long) rows * rowBytes;
            for (int s = 0; s < sequences; s += settings.batch()) {
                int count = Math.min(settings.batch(), sequences - s);
                device.gpu.copyHostToDevice(
                        batchStates, states.asSlice(s * sequenceBytes, count * packed), count * packed);
                forward.forward(layer, batchStates, count, rows, null);
                device.gpu.rmsNormUnitOffsetBf16(batchStates, mtpNorm, normed, count * rows, hidden, eps);
                device.accumulateHessian(draftHessian, normed, count * rows, hidden);
            }
            device.synchronize();
            log("MTP layer done");
            return draftHessian;
        } finally {
            for (long a : new long[] {embeddingNorm, hiddenNorm, mtpNorm, stem, scratch, scratchNormed, stemHessian}) {
                device.gpu.free(a);
            }
        }
    }

    private Map<String, Long> observeMtp(
            LayerWeights layer, LayerForward forward, MemorySegment states, long batchStates, int sequences, int rows) {
        Map<String, Long> hessians = new LinkedHashMap<>();
        long rowBytes = 2L * hidden, sequenceBytes = rowBytes * (rows + 1), packed = rowBytes * rows;
        LayerForward.Observer observer = (name, input, count, columns) -> {
            long h = hessians.computeIfAbsent(name, key -> zeroed(4L * columns * columns));
            device.accumulateHessian(h, input, count, columns);
        };
        for (int s = 0; s < sequences; s += settings.batch()) {
            int count = Math.min(settings.batch(), sequences - s);
            device.gpu.copyHostToDevice(batchStates, states.asSlice(s * sequenceBytes, count * packed), count * packed);
            forward.forward(layer, batchStates, count, rows, observer);
        }
        device.synchronize();
        return hessians;
    }

    // ---- helpers ----

    private long zeroed(long bytes) {
        long address = device.gpu.allocate(bytes);
        device.gpu.zeroDeviceMemory(address, bytes);
        return address;
    }

    private long uploadTensor(String name) throws IOException {
        long bytes = checkpoint.store().tensor(name).bytes();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(bytes, 16);
            checkpoint.readTensor(name, host);
            long address = device.gpu.allocate(bytes);
            device.gpu.copyHostToDevice(address, host, bytes);
            return address;
        }
    }

    private long uploadObject(Qwen38Objects.ObjectSource source) throws IOException {
        long address = device.gpu.allocate(source.bytes());
        long rowBytes = source.columns() * 2, rowsPerChunk = Math.max(1, (1L << 28) / rowBytes);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(rowsPerChunk * rowBytes, 16);
            for (long row = 0; row < source.rows(); row += rowsPerChunk) {
                long count = Math.min(rowsPerChunk, source.rows() - row);
                for (var piece : source.pieces()) {
                    if (piece.firstRow() != 0 || source.pieces().size() != 1) {
                        throw new IllegalArgumentException(
                                "chunked upload supports whole tensors only: " + source.name());
                    }
                }
                checkpoint
                        .store()
                        .read(source.pieces().getFirst().tensor(), row * rowBytes, host.asSlice(0, count * rowBytes));
                device.gpu.copyHostToDevice(
                        address + row * rowBytes, host.asSlice(0, count * rowBytes), count * rowBytes);
            }
        }
        return address;
    }

    private int[] draftIds() throws IOException {
        var artifact = ArtifactReader.read(settings.draftIdsFrom());
        for (var tensor : artifact.tensors()) {
            if (!tensor.name().equals("text/draft_head_token_ids")) continue;
            ByteBuffer bytes = ByteBuffer.allocate((int) tensor.byteSize()).order(ByteOrder.LITTLE_ENDIAN);
            try (FileChannel channel = FileChannel.open(settings.draftIdsFrom(), StandardOpenOption.READ)) {
                long position = tensor.dataOffset();
                while (bytes.hasRemaining()) {
                    int read = channel.read(bytes, position);
                    if (read < 0) throw new IOException("truncated draft ids");
                    position += read;
                }
            }
            int[] ids = new int[(int) (tensor.byteSize() / 4)];
            bytes.flip().asIntBuffer().get(ids);
            for (int id : ids) {
                if (id < 0 || id >= checkpoint.config.vocabulary())
                    throw new IOException("draft id out of range: " + id);
            }
            return ids;
        }
        throw new IOException(settings.draftIdsFrom() + " has no text/draft_head_token_ids");
    }

    /// tokens.npy: int32 [sequences, length] (gptq/calibration.py); the first `limit` sequences when limit > 0.
    static int[][] readTokens(Path path, int limit) throws IOException {
        byte[] bytes = Files.readAllBytes(path);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        int headerLength = Short.toUnsignedInt(buffer.getShort(8));
        String header = new String(bytes, 10, headerLength, java.nio.charset.StandardCharsets.US_ASCII);
        if (!header.contains("'<i4'")) throw new IOException(path + " is not int32");
        var shape = java.util.regex.Pattern.compile("\\((\\d+), (\\d+)\\)").matcher(header);
        if (!shape.find()) throw new IOException(path + ": no 2-D shape in " + header);
        int sequences = Integer.parseInt(shape.group(1)), length = Integer.parseInt(shape.group(2));
        if (limit > 0) sequences = Math.min(sequences, limit);
        var ints = ByteBuffer.wrap(bytes, 10 + headerLength, bytes.length - 10 - headerLength)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asIntBuffer();
        int[][] tokens = new int[sequences][length];
        for (int s = 0; s < sequences; s++) ints.get(tokens[s]);
        return tokens;
    }

    // ---- checkpoints ----

    private Path checkpointFile() {
        return settings.out().resolve("hidden.bin");
    }

    private String checkpointKey(int sequences, int length) {
        return settings.tokens().toAbsolutePath() + "|" + sequences + "|" + length + "|" + settings.recipe() + "|"
                + settings.actOrder() + "|" + settings.damp();
    }

    private void saveCheckpoint(MemorySegment states, int layersDone, int sequences, int length) throws IOException {
        Path partial = settings.out().resolve("hidden.bin.partial");
        try (FileChannel channel = FileChannel.open(
                partial, StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            // A ByteBuffer covers at most 2 GiB: write the states a slice at a time.
            for (long at = 0; at < states.byteSize(); at += CHECKPOINT_SLICE) {
                ByteBuffer buffer = states.asSlice(at, Math.min(CHECKPOINT_SLICE, states.byteSize() - at))
                        .asByteBuffer();
                while (buffer.hasRemaining()) channel.write(buffer);
            }
        }
        Files.move(partial, checkpointFile(), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
        Files.writeString(settings.out().resolve("hidden.key"), checkpointKey(sequences, length) + "|" + layersDone);
        log("checkpoint after layer " + (layersDone - 1));
    }

    private int resume(MemorySegment states, int sequences, int length) throws IOException {
        Path key = settings.out().resolve("hidden.key");
        if (!Files.isRegularFile(key) || !Files.isRegularFile(checkpointFile())) return 0;
        String saved = Files.readString(key);
        String prefix = checkpointKey(sequences, length) + "|";
        if (!saved.startsWith(prefix)) return 0;
        int layersDone = Integer.parseInt(saved.substring(prefix.length()).trim());
        try (FileChannel channel = FileChannel.open(checkpointFile(), StandardOpenOption.READ)) {
            long position = 0;
            for (long at = 0; at < states.byteSize(); at += CHECKPOINT_SLICE) {
                ByteBuffer buffer = states.asSlice(at, Math.min(CHECKPOINT_SLICE, states.byteSize() - at))
                        .asByteBuffer();
                while (buffer.hasRemaining()) {
                    int read = channel.read(buffer, position);
                    if (read < 0) return 0;
                    position += read;
                }
            }
        }
        log("resumed after layer " + (layersDone - 1));
        return layersDone;
    }

    private void dropCheckpoint() throws IOException {
        Files.deleteIfExists(checkpointFile());
        Files.deleteIfExists(settings.out().resolve("hidden.key"));
    }
}
