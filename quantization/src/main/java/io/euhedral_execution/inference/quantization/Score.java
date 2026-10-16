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
import io.euhedral_execution.inference.core.state.KvFormat;
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
import java.util.concurrent.atomic.AtomicLong;

/// Teacher-forced scoring of an artifact against llama.cpp BF16 references (`llama-perplexity
/// --kl-divergence-base`, docs/QUALITY.md). Each chunk of N reference tokens is forced on a fresh sequence: a
/// prefill of N / 2 + 1 tokens, then verifications of `rows` forced tokens, so that step k's logits row
/// predicts the token the reference's record k does. Every row is scored on the device where the engine left it
/// (`score.cu`), against the reference row a reader frame has already fetched into mapped host memory; nothing
/// is copied. `batch` chunks run together, one sequence each, their verifications batched into one quantum (at
/// most 32 rows), so each weight is read once per step group of all of them. The summary line per reference:
/// name, mean NLL, perplexity, dNLL and its standard error, KL and its standard error, top-1 agreement, then KL
/// median, p95, p99, maximum, the top-1 standard error and delta-p mean, RMS, p1 and p99.
public final class Score {

    /// Command-line settings.
    public record Settings(
            Path artifact,
            List<Path> references,
            List<String> names,
            Path results,
            int rows,
            long hostBytes,
            int batch,
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
    private final long scoreStream;
    private final KvFormat kvFormat;

    private Score(Settings settings, Device device, Execution runtime, ExecutionPlan plan, KvFormat kvFormat) {
        this.settings = settings;
        this.kvFormat = kvFormat;
        this.device = device;
        this.runtime = runtime;
        this.plan = plan;
        device.cuda.bindThread(); // the worker's thread compiles the scorer's kernel; the engine selects streams here
        this.scoreStream = device.newStream();
        this.kernel =
                device.cuda.compileResource("score.cu", List.of("score_rows")).getFirst();
    }

    public static void run(Settings settings) throws Exception {
        if (settings.batch() < 1
                || settings.rows() < 1
                || settings.rows() > 8
                || settings.batch() * settings.rows() > 32)
            throw new IllegalArgumentException("--rows (at most 8) times --batch must be at most 32");
        var data = ArtifactReader.read(settings.artifact());
        int context = 0;
        for (Path path : settings.references())
            try (Reference reference = Reference.open(path)) {
                context = Math.max(context, reference.context);
            }
        try (Device device = new Device(settings.library());
                Qwen38Model model = Qwen38Model.load(
                        settings.artifact(),
                        data,
                        device.gpu,
                        ArtifactProfile.Speculation.NONE,
                        Residency.hostBacked(
                                data,
                                ArtifactProfile.Speculation.NONE,
                                device.gpu,
                                settings.hostBytes(),
                                settings.batch(),
                                context));
                var lattice = CalibrationLattice.start(settings.workers())) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice.terminal(), plan, device.gpu);
            try {
                // On the lattice's workers: the scorer's setup, then each reference in turn, each chunk group's
                // verifications continuing the next. This thread only waits.
                runtime.onWorker(() -> new Score(settings, device, runtime, plan, Residency.kvFormat(data)))
                        .thenCompose(score -> {
                            CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
                            for (int r = 0; r < settings.references().size(); r++) {
                                Path path = settings.references().get(r);
                                String name = settings.names().get(r);
                                chain = chain.thenCompose(ignored -> score.reference(path, name));
                            }
                            return chain;
                        })
                        .get();
            } finally {
                runtime.close();
            }
        }
    }

    /// Scores every chunk of `path` and appends the summary to the results file (and standard output);
    /// completes on a worker once the last group has.
    private CompletableFuture<Void> reference(Path path, String name) {
        long started = System.nanoTime();
        Reference reference;
        try {
            reference = Reference.open(path);
        } catch (IOException failure) {
            return CompletableFuture.failedFuture(failure);
        }
        int vocabulary = plan.weights().config().vocabSize();
        if (vocabulary != reference.vocabulary) {
            closeQuietly(reference);
            return CompletableFuture.failedFuture(new IllegalArgumentException(
                    path + " has a vocabulary of " + reference.vocabulary + ", the artifact " + vocabulary));
        }
        double[][] scored = new double[reference.chunks][];
        int batch = Math.min(settings.batch(), reference.chunks);
        Lane[] lanes = new Lane[batch];
        for (int k = 0; k < batch; k++) lanes[k] = new Lane(reference);
        CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
        for (int first = 0; first < reference.chunks; first += batch) {
            int[] chunks = new int[Math.min(batch, reference.chunks - first)];
            for (int k = 0; k < chunks.length; k++) chunks[k] = first + k;
            chain = chain.thenCompose(ignored -> new Group(reference, chunks, lanes).run(scored));
        }
        return chain.thenRun(() -> {
                    String summary = Summary.lines(name, path, settings.artifact(), reference, scored);
                    System.out.print(summary);
                    try {
                        Files.writeString(
                                settings.results(),
                                summary,
                                StandardOpenOption.CREATE,
                                StandardOpenOption.APPEND,
                                StandardOpenOption.WRITE);
                    } catch (IOException failure) {
                        throw new java.io.UncheckedIOException(failure);
                    }
                    System.out.printf(
                            Locale.ROOT,
                            "scored %s: %d chunks in %.1f s%n",
                            name,
                            reference.chunks,
                            (System.nanoTime() - started) / 1e9);
                })
                .whenComplete((ignored, failure) -> {
                    for (Lane lane : lanes) lane.close();
                    closeQuietly(reference);
                });
    }

    private static void closeQuietly(Reference reference) {
        try {
            reference.close();
        } catch (IOException ignored) {
            // A read-only file: nothing to lose.
        }
    }

    /// One slot of a group: the mapped buffers of the chunk it scores (two reference row blocks, the targets
    /// and the results).
    private final class Lane implements AutoCloseable {
        final long[] blocks = new long[2];
        final long targets, results;

        Lane(Reference reference) {
            for (int b = 0; b < 2; b++)
                blocks[b] = device.gpu.allocateHostWeights(settings.rows() * 2L * reference.width);
            targets = device.gpu.allocateHostWeights(4L * reference.records);
            results = device.gpu.allocateHostWeights(8L * RESULTS * reference.records);
        }

        @Override
        public void close() {
            for (long block : blocks) device.gpu.freeHostWeights(block);
            device.gpu.freeHostWeights(targets);
            device.gpu.freeHostWeights(results);
        }
    }

    /// Chunks scored together, one sequence each, in lockstep (every chunk of a reference has the same prefix
    /// and steps): each prefill on its own, then each step group of every chunk in one batched verification.
    private final class Group {
        final Reference reference;
        final int[] chunks;
        final Lane[] lanes;
        final Sequence[] sequences;
        final int prefix, steps, groups;

        Group(Reference reference, int[] chunks, Lane[] lanes) {
            this.reference = reference;
            this.chunks = chunks;
            this.lanes = lanes;
            this.sequences = new Sequence[chunks.length];
            this.prefix = reference.context / 2 + 1;
            this.steps = reference.records;
            // Group 0 is step 0, from the prefill's last row; group g > 0 covers steps 1 + rows (g - 1) onwards.
            this.groups = 1 + (steps - 1 + settings.rows() - 1) / settings.rows();
        }

        CompletableFuture<Void> run(double[][] scored) {
            for (int k = 0; k < chunks.length; k++) {
                int[] tokens = reference.tokens[chunks[k]];
                MemorySegment.ofAddress(lanes[k].targets)
                        .reinterpret(4L * steps)
                        .copyFrom(MemorySegment.ofArray(Arrays.copyOfRange(tokens, prefix, prefix + steps)));
                sequences[k] = new Sequence(sequenceIds.getAndIncrement());
                sequences[k].useKvFormat(kvFormat);
            }
            // A group's reference rows are read while the quantum before it runs: block g % 2 was last scored by
            // group g - 2, finished by then.
            @SuppressWarnings("unchecked")
            CompletableFuture<Void>[] reads = new CompletableFuture[groups];
            reads[0] = read(0, 0);
            if (groups > 1) reads[1] = read(1, 1);
            List<CompletableFuture<Void>> prefills = new ArrayList<>();
            for (int k = 0; k < chunks.length; k++) {
                int lane = k;
                int[] tokens = reference.tokens[chunks[k]];
                CompletableFuture<Void> chain = reads[0];
                for (int begin = 0; begin < prefix; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
                    int[] part = Arrays.copyOfRange(
                            tokens, begin, Math.min(prefix, begin + InferenceConfig.PREFILL_CHUNK_TOKENS));
                    boolean last = begin + InferenceConfig.PREFILL_CHUNK_TOKENS >= prefix;
                    chain = chain.thenCompose(ignored -> prefill(lane, part, last));
                }
                prefills.add(chain);
            }
            CompletableFuture<Void> done = CompletableFuture.allOf(prefills.toArray(CompletableFuture[]::new));
            for (int g = 1; g < groups; g++) {
                int group = g, first = 1 + settings.rows() * (g - 1), count = Math.min(settings.rows(), steps - first);
                done = done.thenCompose(ignored -> {
                    if (group + 1 < groups) reads[group + 1] = read(group + 1, first + settings.rows());
                    return reads[group].thenCompose(alsoIgnored -> verify(group, first, count));
                });
            }
            return done.handle((ignored, failure) -> {
                for (Sequence sequence : sequences) sequence.complete();
                if (failure != null)
                    throw new IllegalStateException("chunks " + Arrays.toString(chunks) + " failed", failure);
                for (int k = 0; k < chunks.length; k++) {
                    double[] values = new double[RESULTS * steps];
                    MemorySegment.copy(
                            MemorySegment.ofAddress(lanes[k].results).reinterpret(8L * RESULTS * steps),
                            ValueLayout.JAVA_DOUBLE,
                            0,
                            values,
                            0,
                            values.length);
                    scored[chunks[k]] = values;
                }
                return null;
            });
        }

        /// Reads every chunk's reference rows of `group` (its first step `first`) into the lanes' blocks, on
        /// workers.
        CompletableFuture<Void> read(int group, int first) {
            int count = group == 0 ? 1 : Math.min(settings.rows(), reference.records - first);
            CompletableFuture<?>[] reads = new CompletableFuture<?>[chunks.length];
            for (int k = 0; k < chunks.length; k++) {
                int chunk = chunks[k];
                long block = lanes[k].blocks[group % 2];
                reads[k] = runtime.onWorker(() -> {
                    reference.read(chunk, first, count, block);
                    return null;
                });
            }
            return CompletableFuture.allOf(reads);
        }

        /// One prefill quantum of lane `lane`'s sequence; the last one's logits row is step 0.
        CompletableFuture<Void> prefill(int lane, int[] tokens, boolean last) {
            Sequence sequence = sequences[lane];
            var quantum = new Quantum(
                    plan,
                    sequence,
                    Quantum.ExecutionKind.PREFILL,
                    sequence.currentTokenPosition(),
                    tokens,
                    last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE);
            return succeeded(runtime.submit(quantum, done -> {
                if (!last) return;
                try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                    long row = logits.deviceAddress()
                            + (long) (logits.tokenCount() - 1) * logits.vocabularySize() * Short.BYTES;
                    score(row, lane, 0, 0, 1);
                    device.cuda.synchronize(scoreStream);
                }
            }));
        }

        /// One batched verification of `count` forced tokens per chunk from step `first`, scored as `group`.
        CompletableFuture<Void> verify(int group, int first, int count) {
            List<Quantum.Part> parts = new ArrayList<>();
            for (int k = 0; k < chunks.length; k++) {
                int[] tokens = reference.tokens[chunks[k]];
                int[] forced = Arrays.copyOfRange(tokens, prefix + first - 1, prefix + first - 1 + count);
                parts.add(new Quantum.Part(sequences[k], sequences[k].currentTokenPosition(), forced));
            }
            var quantum = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
            return succeeded(runtime.submit(quantum, done -> {
                try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                    long rowBytes = (long) logits.vocabularySize() * Short.BYTES;
                    for (int k = 0; k < chunks.length; k++)
                        score(logits.deviceAddress() + (long) k * count * rowBytes, k, group, first, count);
                    device.cuda.synchronize(scoreStream);
                }
            }));
        }

        /// Scores `count` logits rows from `row` against lane `lane`'s block of `group`. The kernel reads the
        /// reference rows and writes its results in mapped host memory; the caller waits for it before the
        /// logits are released.
        void score(long row, int lane, int group, int first, int count) {
            device.cuda.bindThread();
            device.cuda.launch(
                    kernel,
                    count,
                    1,
                    THREADS,
                    0,
                    scoreStream,
                    new Cuda.Arguments()
                            .pointer(row)
                            .pointer(device.gpu.hostWeightsDeviceAddress(lanes[lane].blocks[group % 2]))
                            .integer(reference.width)
                            .integer(reference.vocabulary)
                            .pointer(device.gpu.hostWeightsDeviceAddress(lanes[lane].targets) + 4L * first)
                            .pointer(device.gpu.hostWeightsDeviceAddress(lanes[lane].results) + 8L * RESULTS * first));
        }
    }

    private static CompletableFuture<Void> succeeded(CompletableFuture<Quantum.Outcome> outcome) {
        return outcome.thenAccept(done -> {
            if (done.status() != Quantum.Status.SUCCESS)
                throw new IllegalStateException("quantum failed", done.failure());
        });
    }

    /// A llama.cpp logits file: "_logits_", int32 context, vocabulary and chunk count, the chunks' tokens, then
    /// per chunk the records of its second half, each two FP32 values (scale, minimum) and a uint16 code per
    /// token.
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

        /// Records `begin` to `begin + count` of `chunk` into the host memory at `address` (positioned reads:
        /// the references are never memory-mapped).
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
