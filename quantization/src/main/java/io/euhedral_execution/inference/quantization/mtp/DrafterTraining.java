package io.euhedral_execution.inference.quantization.mtp;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.ResumeFile;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import io.euhedral_execution.inference.quantization.lattice.GpuCompletions;
import io.euhedral_execution.inference.quantization.training.Ops;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Random;
import java.util.concurrent.CompletableFuture;

/// Trains the MTP drafter on its corpus ([Corpus]): the base model's own greedy answers and the hidden of every row
/// it ran. The drafter starts from the checkpoint's MTP weights and trains every parameter, its projections rounded to
/// NVFP4 as they will be exported ([Drafter]), unrolled over `depths` drafts; each depth's loss is the cross-entropy
/// of the artifact's draft head (its NVFP4 shortlist rows, frozen) against the base model's choice, on answer tokens.
/// Token embeddings are the artifact's (Q3), as the decoder reads them.
///
/// Sequences are read in windows of `rows` lineages in order; a window's depth-1 rows attend the keys of up to
/// `keys` earlier positions of the sequence, kept from its earlier windows (their gradients stop at the window). A
/// window without answer tokens only runs depth 1 forward, for its keys. An update follows every `tokens` trained
/// lineages; every `holdout`-th sequence is held out, and each evaluation measures, over its lineages whose drafts all
/// have labels, how many leading drafts equal the base model's tokens: the drafts a verification would accept.
///
/// The best evaluation's parameters are kept in `out/best.bin`; a run resumes from `out/resume.bin`, overwritten at
/// every evaluation.
public final class DrafterTraining {

    public record Settings(
            Path corpus,
            Path model,
            Path artifact,
            Path out,
            int depths,
            int rows,
            int keys,
            int tokens,
            int epochs,
            float projectionRate,
            float normRate,
            int warmup,
            int holdout,
            int evaluateEvery,
            int limit) {}

    /// Rows of draft-head logits at a time.
    private static final int HEAD_ROWS = 128;
    /// Longest sequence of the corpus the key buffers hold (`generate` prompts are at most 4096 tokens, answers its
    /// `--max-tokens`).
    private static final int MAX_POSITIONS = 16384;

    private final Settings settings;
    private final Device device;
    private final Qwen38Objects checkpoint;
    private final long start = System.nanoTime();

    private DrafterTraining(Settings settings, Device device, Qwen38Objects checkpoint) {
        this.settings = settings;
        this.device = device;
        this.checkpoint = checkpoint;
    }

    /// Starts the run as lattice frames and waits for it: nothing else runs on the calling thread.
    public static void run(Settings settings, Path library) throws Exception {
        if (settings.depths() < 1 || settings.rows() < 1 || settings.keys() < settings.rows())
            throw new IllegalArgumentException("depths and rows must be positive, keys at least rows");
        try (Safetensors store = Safetensors.open(settings.model());
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(2)) {
            var training = new DrafterTraining(settings, device, new Qwen38Objects(store, settings.model()));
            Run run = training.new Run(lattice);
            try {
                run.start().get();
            } finally {
                run.close();
            }
        }
    }

    private void log(String message) {
        System.out.printf(Locale.ROOT, "[%7.0f s] %s%n", (System.nanoTime() - start) / 1e9, message);
    }

    /// One window of a sequence: lineages [a, a + count), keys from `firstKey`, and whether any of its drafts has a
    /// label.
    record Window(int sequence, int a, int count, int firstKey, boolean labelled) {}

    private final class Run implements AutoCloseable {
        final CalibrationLattice lattice;
        final CompletableFuture<Void> done = new CompletableFuture<>();
        final FrameSeeds seeds = new FrameSeeds();
        final GpuCompletions completions;
        final Arena arena = Arena.ofShared();
        final List<Long> deviceOwned = new ArrayList<>();
        final List<Long> hostOwned = new ArrayList<>();
        Ops ops;
        DrafterKernels kernels;
        Drafter drafter;
        int hidden, draftRows;
        long head, headBf16Input, logits, logitsBf16, embeddingTable, embeddingScales;
        int[] rowOfToken;
        List<Corpus.Entry> entries;
        MemorySegment tokenFile, labelFile, hiddenFile;
        List<Integer> training = new ArrayList<>(), heldOut = new ArrayList<>();
        // Per-window device inputs and outputs.
        long hiddenBf16, hiddenInput, ids, labels, predicted, labelLogProbabilities, loss;
        long[] embedded;
        MemorySegment hostPredicted, hostLabels, hostLoss;
        final long finished;
        // Progress.
        int epoch, cursor, step, pendingLineages;
        boolean evaluationDue;
        int[] order;
        double best = -1;
        ResumeFile resume;
        String identity;
        // Statistics since the last report.
        final double[] trainLoss = new double[16];
        final long[] trainCorrect = new long[16], trainCount = new long[16];

        Run(CalibrationLattice lattice) {
            this.lattice = lattice;
            this.completions = new GpuCompletions(device.cuda);
            lattice.attach(completions);
            this.finished = device.cuda.createEvent();
        }

        CompletableFuture<Void> start() {
            lattice.lake().admit();
            lattice.lake().publish(new Begin());
            return done.whenComplete((ignored, failure) -> lattice.lake().terminated());
        }

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

        long deviceBuffer(long bytes) {
            long address = device.gpu.allocate(Math.max(4, bytes));
            deviceOwned.add(address);
            return address;
        }

        MemorySegment pinned(long bytes) {
            long address = device.gpu.allocateHostWeights(Math.max(256, bytes));
            hostOwned.add(address);
            return MemorySegment.ofAddress(address).reinterpret(bytes);
        }

        // ------------------------------------------------------------ setup

        final class Begin extends Frame {
            @Override
            void work() throws IOException {
                Files.createDirectories(settings.out());
                ops = new Ops(device.cuda, device.stream, device.cublas);
                kernels = new DrafterKernels(device.cuda, device.stream);
                var config = checkpoint.config;
                hidden = config.hidden();
                var shape = new Drafter.Shape(
                        config.hidden(),
                        config.intermediate(),
                        config.heads(),
                        config.kvHeads(),
                        config.headDim(),
                        config.rotaryDim(),
                        config.theta(),
                        (float) config.eps());
                drafter = new Drafter(
                        device.gpu,
                        device.cuda,
                        device.cublas,
                        ops,
                        kernels,
                        shape,
                        settings.depths(),
                        settings.rows(),
                        settings.keys(),
                        MAX_POSITIONS);
                loadCheckpoint();
                loadArtifact();
                openCorpus();
                allocateWindow();
                identity = String.join(
                        " ",
                        "corpus=" + settings.corpus().toAbsolutePath(),
                        "sequences=" + entries.size(),
                        "model=" + settings.model().toAbsolutePath(),
                        "artifact=" + settings.artifact().toAbsolutePath(),
                        "depths=" + settings.depths(),
                        "rows=" + settings.rows(),
                        "keys=" + settings.keys(),
                        "tokens=" + settings.tokens(),
                        "epochs=" + settings.epochs(),
                        "rates=" + settings.projectionRate() + "/" + settings.normRate(),
                        "warmup=" + settings.warmup(),
                        "holdout=" + settings.holdout());
                resume = new ResumeFile(settings.out().resolve("resume.bin"));
                boolean resumed = readState();
                if (!resumed) order = shuffled(0);
                log(String.format(
                        Locale.ROOT,
                        "%d training and %d held-out sequences, %d draft-head rows%s",
                        training.size(),
                        heldOut.size(),
                        draftRows,
                        resumed ? ", resumed at epoch " + epoch + " step " + step : ""));
                publish(new Evaluate(true));
            }
        }

        void loadCheckpoint() throws IOException {
            try (Arena local = Arena.ofConfined()) {
                for (String name : Drafter.PROJECTIONS) {
                    var source = checkpoint.object(name);
                    MemorySegment host = local.allocate(source.bytes(), 256);
                    checkpoint.read(source, host);
                    upload(drafter.parameters.get(name), host);
                }
                Map<String, String> tensors = Map.of(
                        Drafter.EMBEDDING_NORM, "mtp.pre_fc_norm_embedding.weight",
                        Drafter.HIDDEN_NORM, "mtp.pre_fc_norm_hidden.weight",
                        Drafter.INPUT_NORM, "mtp.layers.0.input_layernorm.weight",
                        Drafter.POST_NORM, "mtp.layers.0.post_attention_layernorm.weight",
                        Drafter.QUERY_NORM, "mtp.layers.0.self_attn.q_norm.weight",
                        Drafter.KEY_NORM, "mtp.layers.0.self_attn.k_norm.weight",
                        Drafter.FINAL_NORM, "mtp.norm.weight");
                for (var entry : tensors.entrySet()) {
                    var p = drafter.parameters.get(entry.getKey());
                    MemorySegment host = local.allocate(2 * p.count, 256);
                    checkpoint.readTensor(entry.getValue(), host);
                    upload(p, host);
                }
            }
            drafter.measure();
        }

        /// BF16 host values into a parameter's FP32 values.
        void upload(Drafter.Parameter p, MemorySegment bf16) {
            long staging = device.gpu.allocate(2 * p.count);
            try {
                device.gpu.copyHostToDevice(staging, bf16, 2 * p.count);
                ops.bf16ToFloat(staging, p.count, p.values);
                device.synchronize();
            } finally {
                device.gpu.free(staging);
            }
        }

        void loadArtifact() throws IOException {
            Artifact artifact = ArtifactReader.read(settings.artifact());
            TensorDescriptor draftHead = descriptor(artifact, "text/draft_head");
            TensorDescriptor tokenIds = descriptor(artifact, "text/draft_head_token_ids");
            TensorDescriptor embedding = descriptor(artifact, "text/token_embedding");
            try (FileChannel channel = FileChannel.open(settings.artifact(), StandardOpenOption.READ);
                    Arena local = Arena.ofConfined()) {
                // The draft head, NVFP4 row-split-k128-v1, decoded to BF16 once.
                draftRows = Math.toIntExact(draftHead.shape()[0]);
                int k = Math.toIntExact(draftHead.shape()[1]), kPad = (k + 127) / 128 * 128;
                long codesBytes = (long) draftRows * kPad / 2;
                long scaleOffset = align(codesBytes), globalOffset = align(scaleOffset + (long) draftRows * kPad / 16);
                MemorySegment payload =
                        channel.map(FileChannel.MapMode.READ_ONLY, draftHead.dataOffset(), draftHead.byteSize(), local);
                long staging = device.gpu.allocate(draftHead.byteSize());
                try {
                    device.gpu.copyHostToDevice(staging, payload, draftHead.byteSize());
                    head = deviceBuffer(2L * draftRows * hidden);
                    kernels.nvfp4DecodeBf16(
                            staging, staging + scaleOffset, staging + globalOffset, draftRows, kPad, k, head);
                    device.synchronize();
                } finally {
                    device.gpu.free(staging);
                }
                MemorySegment ids =
                        channel.map(FileChannel.MapMode.READ_ONLY, tokenIds.dataOffset(), tokenIds.byteSize(), local);
                rowOfToken = new int[checkpoint.config.vocabulary()];
                Arrays.fill(rowOfToken, -1);
                for (int row = 0; row < draftRows; row++)
                    rowOfToken[ids.getAtIndex(ValueLayout.JAVA_INT_UNALIGNED, row)] = row;
                // The Q3 token embedding, read in place from pinned host memory.
                MemorySegment table =
                        channel.map(FileChannel.MapMode.READ_ONLY, embedding.dataOffset(), embedding.byteSize(), local);
                MemorySegment pinnedTable = pinned(embedding.byteSize());
                MemorySegment.copy(table, 0, pinnedTable, 0, embedding.byteSize());
                embeddingTable = pinnedTable.address();
                long vocabulary = embedding.shape()[0];
                int groups = (Math.toIntExact(embedding.shape()[1]) + 127) / 128 * 2;
                embeddingScales = align(vocabulary * groups * 24);
            }
        }

        TensorDescriptor descriptor(Artifact artifact, String name) {
            for (TensorDescriptor tensor : artifact.tensors()) if (tensor.name().equals(name)) return tensor;
            throw new IllegalArgumentException(settings.artifact() + " has no " + name);
        }

        static long align(long value) {
            return (value + 255) & ~255L;
        }

        void openCorpus() throws IOException {
            entries = Corpus.entries(settings.corpus());
            if (settings.limit() > 0 && entries.size() > settings.limit())
                entries = entries.subList(0, settings.limit());
            if (entries.isEmpty()) throw new IllegalArgumentException(settings.corpus() + " holds no sequences");
            for (int s = 0; s < entries.size(); s++) {
                if (entries.get(s).length() > MAX_POSITIONS)
                    throw new IllegalArgumentException(
                            "sequence " + entries.get(s).id() + " is too long");
                if (settings.holdout() > 0 && s % settings.holdout() == 0) heldOut.add(s);
                else training.add(s);
            }
            try (FileChannel tokens =
                            FileChannel.open(settings.corpus().resolve("tokens.bin"), StandardOpenOption.READ);
                    FileChannel labels =
                            FileChannel.open(settings.corpus().resolve("labels.bin"), StandardOpenOption.READ);
                    FileChannel rows =
                            FileChannel.open(settings.corpus().resolve("hidden.bin"), StandardOpenOption.READ)) {
                tokenFile = tokens.map(FileChannel.MapMode.READ_ONLY, 0, tokens.size(), arena);
                labelFile = labels.map(FileChannel.MapMode.READ_ONLY, 0, labels.size(), arena);
                hiddenFile = rows.map(FileChannel.MapMode.READ_ONLY, 0, rows.size(), arena);
            }
        }

        void allocateWindow() {
            int r = settings.rows(), depths = settings.depths();
            hiddenBf16 = deviceBuffer(2L * r * hidden);
            hiddenInput = deviceBuffer(4L * r * hidden);
            embedded = new long[depths];
            for (int k = 0; k < depths; k++) embedded[k] = deviceBuffer(4L * r * hidden);
            ids = deviceBuffer(4L * r * depths);
            labels = deviceBuffer(4L * r * depths);
            predicted = deviceBuffer(4L * r * depths);
            labelLogProbabilities = deviceBuffer(4L * r * depths);
            loss = deviceBuffer(8L * depths);
            headBf16Input = deviceBuffer(2L * r * hidden);
            logits = deviceBuffer(4L * HEAD_ROWS * draftRows);
            logitsBf16 = deviceBuffer(2L * HEAD_ROWS * draftRows);
            hostPredicted = pinned(4L * r * depths);
            hostLabels = pinned(4L * r * depths);
            hostLoss = pinned(8L * depths);
        }

        int[] shuffled(int seed) {
            int[] result = training.stream().mapToInt(Integer::intValue).toArray();
            Random random = new Random(20261019L + seed);
            for (int i = result.length - 1; i > 0; i--) {
                int j = random.nextInt(i + 1), t = result[i];
                result[i] = result[j];
                result[j] = t;
            }
            return result;
        }

        // ------------------------------------------------------------ windows

        List<Window> windows(int sequence) {
            Corpus.Entry entry = entries.get(sequence);
            int lineages = entry.length() - 1, r = settings.rows();
            List<Window> out = new ArrayList<>();
            for (int a = 0; a < lineages; a += r) {
                int count = Math.min(r, lineages - a);
                int firstKey = Math.max(0, a + count - settings.keys());
                boolean labelled = a + count - 1 + 2 + settings.depths() - 1 >= entry.prompt();
                out.add(new Window(sequence, a, count, firstKey, labelled));
            }
            return out;
        }

        int token(Corpus.Entry entry, int position) {
            return tokenFile.getAtIndex(ValueLayout.JAVA_INT_UNALIGNED, entry.token() + position);
        }

        /// The base model's greedy choice for `position`: what a draft of it should be.
        int label(Corpus.Entry entry, int position) {
            return labelFile.getAtIndex(ValueLayout.JAVA_INT_UNALIGNED, entry.token() + position);
        }

        /// Uploads the window's base hiddens, every depth's input tokens and labels (the draft-head row
        /// of the base model's greedy choice; on the device -1 for any other), and returns per depth on
        /// the host the draft-head rows of what a draft is checked against: the labels when training, the
        /// text's own tokens (what a verification of a sampled answer accepts) when `evaluation`; -1 for a
        /// prompt token or a position past the sequence, -2 for an answer token outside the head (a draft
        /// it rejects).
        int[][] stage(Window window, boolean evaluation) {
            Corpus.Entry entry = entries.get(window.sequence());
            int count = window.count(), depths = settings.depths(), length = entry.length();
            long bytes = 2L * count * hidden;
            device.gpu.copyHostToDevice(
                    hiddenBf16, hiddenFile.asSlice(2L * hidden * (entry.row() + window.a()), bytes), bytes);
            ops.bf16ToFloat(hiddenBf16, (long) count * hidden, hiddenInput);
            int[] tokenIds = new int[depths * count];
            int[][] checked = new int[depths][count];
            int[] flat = new int[depths * count];
            for (int k = 1; k <= depths; k++) {
                for (int r = 0; r < count; r++) {
                    int p = window.a() + r, input = p + k, target = p + k + 1;
                    tokenIds[(k - 1) * count + r] = input < length ? token(entry, input) : 0;
                    boolean answer = target < length && target >= entry.prompt();
                    int labelRow = answer ? rowOfToken[label(entry, target)] : -1;
                    flat[(k - 1) * count + r] = labelRow;
                    int row = !answer ? -1 : evaluation ? rowOfToken[token(entry, target)] : labelRow;
                    checked[k - 1][r] = answer && row < 0 ? -2 : row;
                }
            }
            Upload.ints(device.gpu, ids, tokenIds);
            Upload.ints(device.gpu, labels, flat);
            for (int k = 0; k < depths; k++)
                kernels.q3EmbeddingRows(
                        embeddingTable, ids + 4L * k * count, count, hidden, embeddingScales, embedded[k]);
            return checked;
        }

        /// Every depth's forward pass over the window (depth 1 only for an unlabelled one).
        long[] forward(Window window, int depths) {
            long[] outputs = new long[depths];
            long input = hiddenInput;
            for (int k = 1; k <= depths; k++) {
                outputs[k - 1] =
                        drafter.forward(k, window.a(), window.firstKey(), window.count(), input, embedded[k - 1]);
                input = outputs[k - 1];
            }
            return outputs;
        }

        /// Depth k's draft-head cross-entropy: argmax and label log-probabilities, loss, and (scaled by `scale`,
        /// when not 0) the gradient into the depth's output gradient.
        void head(int k, long output, int count, float scale) {
            int depthOffset = (k - 1) * count;
            kernels.floatToBf16(output, (long) count * hidden, headBf16Input);
            for (int first = 0; first < count; first += HEAD_ROWS) {
                int n = Math.min(HEAD_ROWS, count - first);
                device.cuda.gemmBf16(
                        device.cublas,
                        Cuda.CUBLAS_OP_T,
                        Cuda.CUBLAS_OP_N,
                        draftRows,
                        n,
                        hidden,
                        1,
                        head,
                        hidden,
                        headBf16Input + 2L * first * hidden,
                        hidden,
                        0,
                        logits,
                        draftRows);
                kernels.crossEntropy(
                        logits,
                        n,
                        draftRows,
                        labels + 4L * (depthOffset + first),
                        scale,
                        loss + 8L * (k - 1),
                        predicted + 4L * (depthOffset + first),
                        labelLogProbabilities + 4L * (depthOffset + first));
                if (scale == 0) continue;
                kernels.floatToBf16(logits, (long) n * draftRows, logitsBf16);
                device.cuda.gemmBf16(
                        device.cublas,
                        Cuda.CUBLAS_OP_N,
                        Cuda.CUBLAS_OP_N,
                        hidden,
                        n,
                        draftRows,
                        1,
                        head,
                        hidden,
                        logitsBf16,
                        draftRows,
                        1,
                        drafter.outputGradient(k) + 4L * first * hidden,
                        hidden);
            }
        }

        /// Enqueues the window's work: a training window's forward, losses and backward, an evaluation window's
        /// forward and argmaxes, or an unlabelled window's depth-1 forward; returns its labels.
        int[][] enqueue(Window window, boolean train) {
            int[][] labelRows = stage(window, !train);
            int depths = window.labelled() ? settings.depths() : 1;
            long[] outputs = forward(window, depths);
            if (!window.labelled()) return labelRows;
            ops.zero(loss, depths * 2L);
            float scale = train ? 1f / settings.tokens() : 0f;
            if (train) drafter.startBackward(window.firstKey(), window.a(), window.count());
            for (int k = 1; k <= depths; k++) head(k, outputs[k - 1], window.count(), scale);
            if (train)
                for (int k = depths; k >= 1; k--) drafter.backward(k, window.a(), window.firstKey(), window.count());
            return labelRows;
        }

        /// The window's argmaxes and losses, after its event.
        int[][] results(Window window) {
            int depths = settings.depths(), count = window.count();
            device.gpu.copyDeviceToHost(hostPredicted, predicted, 4L * depths * count);
            device.gpu.copyDeviceToHost(hostLoss, loss, 8L * depths);
            int[][] out = new int[depths][count];
            for (int k = 0; k < depths; k++)
                for (int r = 0; r < count; r++)
                    out[k][r] = hostPredicted.getAtIndex(ValueLayout.JAVA_INT, (long) k * count + r);
            return out;
        }

        // ------------------------------------------------------------ training

        /// Enqueues the windows of the next training sequence, all of them, and awaits them.
        final class Train extends Frame {
            @Override
            void work() {
                if (cursor == order.length) {
                    epoch++;
                    cursor = 0;
                    if (epoch == settings.epochs()) {
                        publish(new Evaluate(false, true));
                        return;
                    }
                    order = shuffled(epoch);
                }
                int sequence = order[cursor++];
                List<Window> windows = windows(sequence);
                publish(new TrainWindow(windows, 0));
            }
        }

        final class TrainWindow extends Frame {
            final List<Window> windows;
            final int index;

            TrainWindow(List<Window> windows, int index) {
                this.windows = windows;
                this.index = index;
            }

            @Override
            void work() {
                Window window = windows.get(index);
                int[][] labelRows = enqueue(window, true);
                device.cuda.recordEvent(finished, device.stream);
                completions.await(finished, new TrainWindowDone(windows, index, labelRows));
            }
        }

        final class TrainWindowDone extends Frame {
            final List<Window> windows;
            final int index;
            final int[][] labelRows;

            TrainWindowDone(List<Window> windows, int index, int[][] labelRows) {
                this.windows = windows;
                this.index = index;
                this.labelRows = labelRows;
            }

            @Override
            void work() throws IOException {
                Window window = windows.get(index);
                if (window.labelled()) {
                    int[][] argmax = results(window);
                    for (int k = 0; k < settings.depths(); k++) {
                        trainLoss[k] += hostLoss.getAtIndex(ValueLayout.JAVA_DOUBLE, k);
                        for (int r = 0; r < window.count(); r++) {
                            if (labelRows[k][r] < 0) continue;
                            trainCount[k]++;
                            if (argmax[k][r] == labelRows[k][r]) trainCorrect[k]++;
                        }
                    }
                    for (int r = 0; r < window.count(); r++) if (labelRows[0][r] != -1) pendingLineages++;
                }
                if (pendingLineages >= settings.tokens()) {
                    update();
                    if (settings.evaluateEvery() > 0 && step % settings.evaluateEvery() == 0) evaluationDue = true;
                }
                continueWith(windows, index + 1);
            }
        }

        /// The sequence's next window, or (its keys no longer needed) a due evaluation or the next sequence.
        void continueWith(List<Window> windows, int index) {
            if (index < windows.size()) {
                publish(new TrainWindow(windows, index));
            } else if (evaluationDue) {
                evaluationDue = false;
                publish(new Evaluate(false));
            } else {
                publish(new Train());
            }
        }

        /// One Adam step at the scheduled rate: linear warmup, then cosine decay to a tenth.
        void update() {
            step++;
            double total = Math.max(1.0, (double) settings.epochs() * expectedSteps());
            double rate = step <= settings.warmup()
                    ? (double) step / settings.warmup()
                    : 0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1.0, (step - settings.warmup()) / total)));
            drafter.step((float) (settings.projectionRate() * rate), (float) (settings.normRate() * rate), step);
            if (step % 10 == 0) {
                StringBuilder line = new StringBuilder(
                        String.format(Locale.ROOT, "step %d (epoch %d, %d/%d):", step, epoch, cursor, order.length));
                for (int k = 0; k < settings.depths(); k++) {
                    line.append(String.format(
                            Locale.ROOT,
                            " d%d loss %.3f top-1 %.3f",
                            k + 1,
                            trainLoss[k] / Math.max(1, trainCount[k]),
                            (double) trainCorrect[k] / Math.max(1, trainCount[k])));
                    trainLoss[k] = 0;
                    trainCorrect[k] = 0;
                    trainCount[k] = 0;
                }
                log(line.toString());
            }
            pendingLineages = 0;
        }

        long expectedSteps() {
            long lineages = 0;
            for (int s : training)
                lineages +=
                        Math.max(0, entries.get(s).length() - 1 - entries.get(s).prompt());
            return Math.max(1, lineages / settings.tokens());
        }

        // ------------------------------------------------------------ evaluation

        final class Evaluate extends Frame {
            final boolean initial, last;
            int sequence;
            List<Window> windows;
            int index;
            long lineages;
            final long[] accepted = new long[16];
            /// Lineages and accepted drafts of greedy [0] and sampled [1] answers.
            final long[] kindLineages = new long[2], kindAccepted = new long[2];
            final long[] reached = new long[16];

            Evaluate(boolean initial) {
                this(initial, false);
            }

            Evaluate(boolean initial, boolean last) {
                this.initial = initial;
                this.last = last;
            }

            @Override
            void work() {
                windows = windows(heldOut.get(sequence));
                index = 0;
                publish(new EvaluateWindow(this));
            }
        }

        final class EvaluateWindow extends Frame {
            final Evaluate evaluation;

            EvaluateWindow(Evaluate evaluation) {
                this.evaluation = evaluation;
            }

            @Override
            void work() {
                Window window = evaluation.windows.get(evaluation.index);
                int[][] labelRows = enqueue(window, false);
                device.cuda.recordEvent(finished, device.stream);
                completions.await(finished, new EvaluateWindowDone(evaluation, labelRows));
            }
        }

        final class EvaluateWindowDone extends Frame {
            final Evaluate evaluation;
            final int[][] labelRows;

            EvaluateWindowDone(Evaluate evaluation, int[][] labelRows) {
                this.evaluation = evaluation;
                this.labelRows = labelRows;
            }

            @Override
            void work() throws IOException {
                Evaluate e = evaluation;
                Window window = e.windows.get(e.index);
                if (window.labelled()) {
                    int[][] argmax = results(window);
                    int depths = settings.depths();
                    for (int r = 0; r < window.count(); r++) {
                        // Lineages whose every draft has a target: an answer token inside the sequence.
                        boolean complete = true;
                        for (int k = 0; k < depths; k++) complete &= labelRows[k][r] != -1;
                        if (!complete) continue;
                        e.lineages++;
                        int run = 0;
                        while (run < depths && labelRows[run][r] >= 0 && argmax[run][r] == labelRows[run][r]) run++;
                        e.accepted[run]++;
                        for (int k = 0; k < run; k++) e.reached[k]++;
                        int kind = entries.get(window.sequence()).sampled() ? 1 : 0;
                        e.kindLineages[kind]++;
                        e.kindAccepted[kind] += run;
                    }
                }
                e.index++;
                if (e.index == e.windows.size()) {
                    e.sequence++;
                    if (e.sequence == heldOut.size()) {
                        finish(e);
                        return;
                    }
                    e.windows = windows(heldOut.get(e.sequence));
                    e.index = 0;
                }
                publish(new EvaluateWindow(e));
            }
        }

        void finish(Evaluate e) throws IOException {
            double mean = 0;
            StringBuilder perDepth = new StringBuilder();
            for (int run = 0; run <= settings.depths(); run++) mean += (double) run * e.accepted[run];
            mean /= Math.max(1, e.lineages);
            for (int k = 0; k < settings.depths(); k++) {
                long before = k == 0 ? e.lineages : e.reached[k - 1];
                perDepth.append(
                        String.format(Locale.ROOT, " d%d %.3f", k + 1, (double) e.reached[k] / Math.max(1, before)));
            }
            log(String.format(
                    Locale.ROOT,
                    "%s: held-out accepted drafts %.4f per lineage (%d lineages), conditional%s; greedy %.4f (%d),"
                            + " sampled %.4f (%d)",
                    e.initial ? "before training" : "step " + step,
                    mean,
                    e.lineages,
                    perDepth,
                    (double) e.kindAccepted[0] / Math.max(1, e.kindLineages[0]),
                    e.kindLineages[0],
                    (double) e.kindAccepted[1] / Math.max(1, e.kindLineages[1]),
                    e.kindLineages[1]));
            if (mean > best) {
                best = mean;
                saveParameters(settings.out().resolve("best.bin"));
            }
            if (!e.initial) writeState();
            if (e.last) {
                log(String.format(Locale.ROOT, "best held-out accepted drafts %.4f", best));
                done.complete(null);
            } else {
                publish(new Train());
            }
        }

        // ------------------------------------------------------------ state

        /// Every parameter's values, then (with moments) its Adam moments, in parameter order.
        void saveParameters(Path file) throws IOException {
            try (ResumeFile out = new ResumeFile(file)) {
                long offset = ResumeFile.HEADER;
                for (Drafter.Parameter p : drafter.parameters.values())
                    offset = copyOut(out, offset, p.values, p.count);
                Properties header = new Properties();
                header.setProperty("identity", identity);
                header.setProperty("step", Integer.toString(step));
                header.setProperty("accepted", Double.toString(best));
                out.header(header);
            }
        }

        long copyOut(ResumeFile out, long offset, long address, long count) throws IOException {
            try (Arena local = Arena.ofConfined()) {
                MemorySegment host = local.allocate(4 * count, 256);
                device.gpu.copyDeviceToHost(host, address, 4 * count);
                return out.write(offset, host);
            }
        }

        long copyIn(ResumeFile in, long offset, long address, long count) throws IOException {
            try (Arena local = Arena.ofConfined()) {
                MemorySegment host = local.allocate(4 * count, 256);
                long next = in.read(offset, host);
                device.gpu.copyHostToDevice(address, host, 4 * count);
                return next;
            }
        }

        /// Overwrites resume.bin: parameters and moments, the epoch and its cursor, step and best.
        void writeState() throws IOException {
            long offset = ResumeFile.HEADER;
            for (Drafter.Parameter p : drafter.parameters.values()) {
                offset = copyOut(resume, offset, p.values, p.count);
                offset = copyOut(resume, offset, p.first, p.count);
                offset = copyOut(resume, offset, p.second, p.count);
            }
            Properties header = new Properties();
            header.setProperty("identity", identity);
            header.setProperty("epoch", Integer.toString(epoch));
            header.setProperty("cursor", Integer.toString(cursor));
            header.setProperty("step", Integer.toString(step));
            header.setProperty("best", Double.toString(best));
            resume.header(header);
        }

        boolean readState() throws IOException {
            Properties header = resume.header();
            if (!identity.equals(header.getProperty("identity"))) return false;
            long offset = ResumeFile.HEADER;
            for (Drafter.Parameter p : drafter.parameters.values()) {
                offset = copyIn(resume, offset, p.values, p.count);
                offset = copyIn(resume, offset, p.first, p.count);
                offset = copyIn(resume, offset, p.second, p.count);
            }
            drafter.measure();
            epoch = Integer.parseInt(header.getProperty("epoch"));
            cursor = Integer.parseInt(header.getProperty("cursor"));
            step = Integer.parseInt(header.getProperty("step"));
            best = Double.parseDouble(header.getProperty("best"));
            order = shuffled(epoch); // each epoch's order follows from its number
            return true;
        }

        @Override
        public void close() throws IOException {
            device.select(device.stream);
            device.synchronize();
            if (resume != null) resume.close();
            if (drafter != null) drafter.close();
            for (long address : deviceOwned) device.gpu.free(address);
            for (long address : hostOwned) device.gpu.freeHostWeights(address);
            arena.close();
        }
    }
}
