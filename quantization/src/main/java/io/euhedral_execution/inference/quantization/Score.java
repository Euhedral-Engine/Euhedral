package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/// Teacher-forced scoring of an artifact against llama.cpp BF16 references (`llama-perplexity
/// --kl-divergence-base`, docs/QUALITY.md). Each chunk of N reference tokens is forced on a fresh sequence: a prefill
/// of N / 2 + 1 tokens, then verifications of `rows` forced tokens, so that step k's logits row predicts the token the
/// reference's record k does. Every row is scored on the device where the engine left it (`score.cu`), against the
/// reference row a reader frame has already fetched into mapped host memory; nothing is copied. Several chunks run
/// at once on their own sequences. The summary line per reference: name, mean NLL, perplexity, dNLL and its standard
/// error, KL and its standard error, top-1 agreement, then KL median, p95, p99, maximum, the top-1 standard error and
/// delta-p mean, RMS, p1 and p99.
public final class Score {

    /// Command-line settings.
    public record Settings(
            Path artifact,
            List<Path> references,
            List<String> names,
            Path results,
            int rows,
            long hostBytes,
            int concurrency,
            int workers,
            Path library) {}

    private static final int RESULTS = 5; // nll, reference nll, kl, agree, delta-p
    private static final int THREADS = 256;

    private final Settings settings;
    private final Device device;
    private final Execution runtime;
    private final ExecutionPlan plan;
    private final Cuda.Function kernel;
    private final AtomicLong sequenceIds = new AtomicLong(1);

    private Score(Settings settings, Device device, Execution runtime, ExecutionPlan plan) {
        this.settings = settings;
        this.device = device;
        this.runtime = runtime;
        this.plan = plan;
        this.kernel =
                device.cuda.compileResource("score.cu", List.of("score_rows")).getFirst();
    }

    public static void run(Settings settings) throws Exception {
        var data = ArtifactReader.read(settings.artifact());
        try (Device device = new Device(settings.library());
                Qwen38Model model = Qwen38Model.load(
                        settings.artifact(),
                        data,
                        device.gpu,
                        ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(data, settings.hostBytes()));
                var lattice = CalibrationLattice.start(settings.workers(), java.util.Set.of(8, 9))) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice.terminal(), plan, device.gpu);
            try {
                var score = new Score(settings, device, runtime, plan);
                for (int r = 0; r < settings.references().size(); r++) {
                    score.reference(
                            settings.references().get(r), settings.names().get(r));
                }
            } finally {
                runtime.close();
            }
        }
    }

    /// Scores every chunk of `path` and appends the summary to the results file (and standard output).
    private void reference(Path path, String name) throws Exception {
        long started = System.nanoTime();
        try (Reference reference = Reference.open(path)) {
            int vocabulary = plan.weights().config().vocabSize();
            if (vocabulary != reference.vocabulary) {
                throw new IllegalArgumentException(
                        path + " has a vocabulary of " + reference.vocabulary + ", the artifact " + vocabulary);
            }
            var queue = new ConcurrentLinkedQueue<Integer>();
            for (int chunk = 0; chunk < reference.chunks; chunk++) queue.add(chunk);
            double[][] scored = new double[reference.chunks][];
            List<CompletableFuture<Void>> lanes = new ArrayList<>();
            for (int lane = 0; lane < Math.min(settings.concurrency(), reference.chunks); lane++) {
                var chain = new Chain(reference);
                lanes.add(chain.drain(queue, scored).whenComplete((ignored, failure) -> chain.close()));
            }
            CompletableFuture.allOf(lanes.toArray(CompletableFuture[]::new)).get();
            String summary = Summary.lines(name, path, settings.artifact(), reference, scored);
            System.out.print(summary);
            Files.writeString(
                    settings.results(),
                    summary,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.APPEND,
                    StandardOpenOption.WRITE);
            System.out.printf(
                    Locale.ROOT,
                    "scored %s: %d chunks in %.1f s%n",
                    name,
                    reference.chunks,
                    (System.nanoTime() - started) / 1e9);
        }
    }

    /// One sequence at a time, chunk after chunk: its mapped buffers (two reference row blocks, the targets and the
    /// results of the current chunk) and the stream its rows are scored on.
    private final class Chain implements AutoCloseable {
        final Reference reference;
        final long rowBytes;
        final long[] blocks = new long[2];
        final long targets, results, stream;

        Chain(Reference reference) {
            this.reference = reference;
            this.rowBytes = 2L * reference.width;
            for (int b = 0; b < 2; b++) blocks[b] = device.gpu.allocateHostWeights(settings.rows() * rowBytes);
            targets = device.gpu.allocateHostWeights(4L * reference.records);
            results = device.gpu.allocateHostWeights(8L * RESULTS * reference.records);
            stream = device.newStream();
        }

        CompletableFuture<Void> drain(ConcurrentLinkedQueue<Integer> queue, double[][] scored) {
            Integer chunk = queue.poll();
            if (chunk == null) return CompletableFuture.completedFuture(null);
            return chunk(chunk).thenCompose(values -> {
                scored[chunk] = values;
                return drain(queue, scored);
            });
        }

        /// The chunk's per-step results [records, RESULTS].
        CompletableFuture<double[]> chunk(int chunk) {
            int[] tokens = reference.tokens[chunk];
            int prefix = reference.context / 2 + 1, steps = reference.records;
            MemorySegment.ofAddress(targets)
                    .reinterpret(4L * steps)
                    .copyFrom(MemorySegment.ofArray(Arrays.copyOfRange(tokens, prefix, prefix + steps)));
            var sequence = new Sequence(sequenceIds.getAndIncrement());
            // Group 0 is step 0, from the prefill's last row; group g > 0 covers steps 1 + rows (g - 1) onwards.
            int groups = 1 + (steps - 1 + settings.rows() - 1) / settings.rows();
            // A group's reference rows are read while the quantum before it runs: block g % 2 was last scored by
            // group g - 2, finished by then.
            @SuppressWarnings("unchecked")
            CompletableFuture<Void>[] reads = new CompletableFuture[groups];
            reads[0] = read(chunk, 0, 0);
            if (groups > 1) reads[1] = read(chunk, 1, 1);
            CompletableFuture<Void> done = reads[0];
            for (int begin = 0; begin < prefix; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
                int[] part = Arrays.copyOfRange(
                        tokens, begin, Math.min(prefix, begin + InferenceConfig.PREFILL_CHUNK_TOKENS));
                boolean last = begin + InferenceConfig.PREFILL_CHUNK_TOKENS >= prefix;
                done = done.thenCompose(ignored -> submit(
                        sequence, Quantum.ExecutionKind.PREFILL, part, LogitsRequirement.LAST_TOKEN, last ? 0 : -1, 0));
            }
            for (int g = 1; g < groups; g++) {
                int group = g, first = 1 + settings.rows() * (g - 1), count = Math.min(settings.rows(), steps - first);
                int[] forced = Arrays.copyOfRange(tokens, prefix + first - 1, prefix + first - 1 + count);
                done = done.thenCompose(ignored -> {
                    if (group + 1 < groups) reads[group + 1] = read(chunk, group + 1, first + settings.rows());
                    return reads[group].thenCompose(alsoIgnored -> submit(
                            sequence,
                            Quantum.ExecutionKind.VERIFY,
                            forced,
                            LogitsRequirement.ALL_TOKENS,
                            group,
                            first));
                });
            }
            return done.handle((ignored, failure) -> {
                sequence.complete();
                if (failure != null) throw new IllegalStateException("chunk " + chunk + " failed", failure);
                double[] values = new double[RESULTS * steps];
                MemorySegment.copy(
                        MemorySegment.ofAddress(results).reinterpret(8L * RESULTS * steps),
                        ValueLayout.JAVA_DOUBLE,
                        0,
                        values,
                        0,
                        values.length);
                return values;
            });
        }

        /// Reads the reference rows of `group` (its first step `first`) into its block, on a worker.
        CompletableFuture<Void> read(int chunk, int group, int first) {
            int count = group == 0 ? 1 : Math.min(settings.rows(), reference.records - first);
            long block = blocks[group % 2];
            return runtime.onWorker(() -> {
                reference.read(chunk, first, count, block);
                return null;
            });
        }

        /// Runs one quantum; with `group` >= 0 its logits rows (all of them, or the last one) are scored as the
        /// steps of `group` starting at `first`.
        CompletableFuture<Void> submit(
                Sequence sequence,
                Quantum.ExecutionKind kind,
                int[] tokens,
                LogitsRequirement requirement,
                int group,
                int first) {
            var quantum = new Quantum(plan, sequence, kind, sequence.currentTokenPosition(), tokens, requirement);
            return runtime.submit(quantum, done -> {
                        try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                            if (group < 0) return;
                            int count = requirement == LogitsRequirement.ALL_TOKENS ? logits.tokenCount() : 1;
                            long row = logits.deviceAddress()
                                    + (long) (logits.tokenCount() - count) * logits.vocabularySize() * Short.BYTES;
                            score(row, group, first, count);
                        }
                    })
                    .thenAccept(outcome -> {
                        if (outcome.status() != Quantum.Status.SUCCESS) {
                            throw new IllegalStateException("quantum failed", outcome.failure());
                        }
                    });
        }

        /// Scores `count` logits rows from `row` against the block of `group`. The kernel reads the reference
        /// rows and writes its results in mapped host memory; it is waited for before the logits are released.
        void score(long row, int group, int first, int count) {
            device.cuda.bindThread();
            device.cuda.launch(
                    kernel,
                    count,
                    1,
                    THREADS,
                    0,
                    stream,
                    new Cuda.Arguments()
                            .pointer(row)
                            .pointer(device.gpu.hostWeightsDeviceAddress(blocks[group % 2]))
                            .integer(reference.width)
                            .integer(reference.vocabulary)
                            .pointer(device.gpu.hostWeightsDeviceAddress(targets) + 4L * first)
                            .pointer(device.gpu.hostWeightsDeviceAddress(results) + 8L * RESULTS * first));
            device.cuda.synchronize(stream);
        }

        @Override
        public void close() {
            for (long block : blocks) device.gpu.freeHostWeights(block);
            device.gpu.freeHostWeights(targets);
            device.gpu.freeHostWeights(results);
        }
    }

    /// A llama.cpp logits file: "_logits_", int32 context, vocabulary and chunk count, the chunks' tokens, then per
    /// chunk the records of its second half, each two FP32 values (scale, minimum) and a uint16 code per token.
    static final class Reference implements AutoCloseable {
        final FileChannel channel;
        final int context, vocabulary, chunks, records, width;
        final int[][] tokens;
        final long dataOffset;

        private Reference(FileChannel channel, int context, int vocabulary, int[][] tokens) {
            this.channel = channel;
            this.context = context;
            this.vocabulary = vocabulary;
            this.chunks = tokens.length;
            this.tokens = tokens;
            this.records = context - 1 - context / 2;
            this.width = 2 * ((vocabulary + 1) / 2) + 4;
            this.dataOffset = 20 + 4L * chunks * context;
        }

        static Reference open(Path path) throws IOException {
            FileChannel channel = FileChannel.open(path);
            try {
                ByteBuffer header = read(channel, 0, 20);
                byte[] magic = new byte[8];
                header.get(magic);
                if (!new String(magic, StandardCharsets.US_ASCII).equals("_logits_")) {
                    throw new IllegalArgumentException(path + " is not a llama.cpp logits file");
                }
                int context = header.getInt(), vocabulary = header.getInt(), chunks = header.getInt();
                ByteBuffer all = read(channel, 20, 4 * context * chunks);
                int[][] tokens = new int[chunks][context];
                for (int chunk = 0; chunk < chunks; chunk++)
                    all.asIntBuffer().position(chunk * context).get(tokens[chunk]);
                return new Reference(channel, context, vocabulary, tokens);
            } catch (IOException | RuntimeException failure) {
                channel.close();
                throw failure;
            }
        }

        private static ByteBuffer read(FileChannel channel, long offset, int size) throws IOException {
            ByteBuffer buffer = ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
            while (buffer.hasRemaining()) {
                if (channel.read(buffer, offset + buffer.position()) < 0) throw new IOException("truncated");
            }
            return buffer.flip();
        }

        /// Records `begin` to `begin + count` of `chunk` into the host memory at `address` (positioned reads: the
        /// references are never memory-mapped).
        void read(int chunk, int begin, int count, long address) {
            long bytes = 2L * width * count;
            ByteBuffer target =
                    MemorySegment.ofAddress(address).reinterpret(bytes).asByteBuffer();
            long offset = dataOffset + 2L * width * ((long) chunk * records + begin);
            try {
                while (target.hasRemaining()) {
                    if (channel.read(target, offset + target.position()) < 0) throw new IOException("truncated");
                }
            } catch (IOException failure) {
                throw new java.io.UncheckedIOException(failure);
            }
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    /// The summary of a reference's per-step results.
    static final class Summary {
        static String lines(String name, Path path, Path artifact, Reference reference, double[][] scored) {
            int chunks = scored.length, steps = reference.records, n = chunks * steps;
            double[] nll = new double[n],
                    referenceNll = new double[n],
                    kl = new double[n],
                    agree = new double[n],
                    dp = new double[n],
                    difference = new double[n];
            double[] chunkDifference = new double[chunks],
                    chunkKl = new double[chunks],
                    chunkAgree = new double[chunks];
            for (int c = 0; c < chunks; c++) {
                for (int s = 0; s < steps; s++) {
                    int i = c * steps + s;
                    double[] v = scored[c];
                    nll[i] = v[RESULTS * s];
                    referenceNll[i] = v[RESULTS * s + 1];
                    kl[i] = v[RESULTS * s + 2];
                    agree[i] = v[RESULTS * s + 3];
                    dp[i] = v[RESULTS * s + 4];
                    difference[i] = nll[i] - referenceNll[i];
                }
                chunkDifference[c] = mean(nll, c * steps, steps) - mean(referenceNll, c * steps, steps);
                chunkKl[c] = mean(kl, c * steps, steps);
                chunkAgree[c] = mean(agree, c * steps, steps);
            }
            StringBuilder b = new StringBuilder();
            b.append(String.format(
                    Locale.ROOT,
                    "# %s %s artifact %s reference %s%n",
                    name,
                    Instant.now().truncatedTo(ChronoUnit.SECONDS),
                    artifact,
                    path));
            b.append(String.format(
                    Locale.ROOT,
                    "%d chunk(s) of %d, %d forced tokens each (positions %d..%d); reference %s%n",
                    chunks,
                    reference.context,
                    steps,
                    reference.context / 2 + 1,
                    reference.context - 1,
                    path));
            b.append(String.format(
                    Locale.ROOT,
                    "%-50s %9s %8s %9s %8s %11s %9s %7s %8s %8s %8s %8s %8s %8s %8s %8s %8s%n",
                    "report",
                    "mean NLL",
                    "ppl",
                    "dNLL",
                    "+-SE",
                    "KL(ref||c)",
                    "+-SE",
                    "top-1",
                    "KL med",
                    "KL p95",
                    "KL p99",
                    "KL max",
                    "top1 SE",
                    "dp mean",
                    "dp RMS",
                    "dp p1",
                    "dp p99"));
            double referenceMean = mean(referenceNll, 0, n);
            b.append(String.format(Locale.ROOT, "%-50s %9.5f %8.4f%n", path, referenceMean, Math.exp(referenceMean)));
            double[] sortedKl = kl.clone(), sortedDp = dp.clone();
            Arrays.sort(sortedKl);
            Arrays.sort(sortedDp);
            double rms = 0;
            for (double value : dp) rms += value * value;
            double meanNll = mean(nll, 0, n);
            b.append(String.format(
                    Locale.ROOT,
                    "%-50s %9.5f %8.4f %+9.5f %8.5f %11.6f %9.6f %7.4f %8.5f %8.5f %8.5f %8.4f %8.5f %+8.5f %8.5f %+8.5f"
                            + " %+8.5f%n",
                    name,
                    meanNll,
                    Math.exp(meanNll),
                    mean(difference, 0, n),
                    standardError(chunkDifference, difference),
                    mean(kl, 0, n),
                    standardError(chunkKl, kl),
                    mean(agree, 0, n),
                    percentile(sortedKl, 50),
                    percentile(sortedKl, 95),
                    percentile(sortedKl, 99),
                    sortedKl[n - 1],
                    standardError(chunkAgree, agree),
                    mean(dp, 0, n),
                    Math.sqrt(rms / n),
                    percentile(sortedDp, 1),
                    percentile(sortedDp, 99)));
            return b.toString();
        }

        static double mean(double[] values, int from, int count) {
            double sum = 0;
            for (int i = from; i < from + count; i++) sum += values[i];
            return sum / count;
        }

        /// Chunk-level over several chunks (tokens within a document are correlated), token-level over one.
        static double standardError(double[] perChunk, double[] perToken) {
            double[] values = perChunk.length > 1 ? perChunk : perToken;
            double mean = mean(values, 0, values.length), sum = 0;
            for (double value : values) sum += (value - mean) * (value - mean);
            return Math.sqrt(sum / (values.length - 1)) / Math.sqrt(values.length);
        }

        /// NumPy's default (linear) percentile of sorted values.
        static double percentile(double[] sorted, double q) {
            double position = q / 100 * (sorted.length - 1);
            int below = (int) Math.floor(position);
            int above = Math.min(sorted.length - 1, below + 1);
            return sorted[below] + (position - below) * (sorted[above] - sorted[below]);
        }
    }
}
