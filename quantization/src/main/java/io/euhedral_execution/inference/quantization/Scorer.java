package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime;
import io.euhedral_execution.inference.core.settings.Tuning;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;

/// Teacher-forced scoring of an artifact against llama.cpp BF16 references (`llama-perplexity
/// --kl-divergence-base`, docs/QUALITY.md). Each chunk of N reference tokens is forced on a fresh sequence: a
/// prefill of N / 2 + 1 tokens, then verifications of `rows` forced tokens, so that step k's logits row
/// predicts the token the reference's record k does. Every row is scored on the device where the engine left it
/// (`score.cu`), against the reference row a reader frame has already fetched into mapped host memory; nothing
/// is copied. `batch` chunks run together, one sequence each, their verifications batched into one quantum (at
/// most the workspace's 512 rows, 32 with an NVFP4 object when the batch is auto-sized), so each weight is read
/// once per step group of all of them. The summary line per reference: name, mean NLL, perplexity, dNLL and its
/// standard error, KL and its standard error, top-1 agreement, then KL
/// median, p95, p99, maximum, the top-1 standard error and delta-p mean, RMS, p1 and p99.
public final class Scorer implements AutoCloseable {

    /// What a score of one artifact came to over all its chunks of all references: mean KL(reference || artifact) and
    /// its standard error (over chunks), top-1 agreement, the difference of mean NLL to the reference's and its
    /// standard error, and each chunk's mean KL in reference order.
    public record Result(double meanKl, double klSe, double top1, double dNll, double dNllSe, double[] chunkKl) {}

    /// One reference's per-step results, `values[chunk]` holding [RESULTS] doubles per forced token.
    record Scored(Path path, Reference reference, double[][] values, double seconds) {}

    /// Called on a worker as each reference completes, before the next starts.
    interface Listener {
        void scored(int index, Scored scored);
    }

    private static final int RESULTS = 5; // nll, reference nll, kl, agree, delta-p
    private static final int THREADS = 256;

    /// Most rows of one batched verification that an NVFP4 linear runs on its tensor-core decode kernels; beyond it
    /// row-exact verification sweeps the weights once per row (`ResidencyPlanner.NVFP4_ROW_EXACT_MAX_ROWS`, 8 rows
    /// in 4 groups in `nvfp4_linear.c`). Slower, not wrong: the auto-sized batch stays within it for an artifact
    /// with an NVFP4 object, and `--score-batch` may go past it.
    static final int NVFP4_TENSOR_CORE_ROWS = 32;
    /// Most rows of one quantum: the rows the workspace is sized for at load, the longest prefill chunk
    /// ([#execution] passes it as `maxRows`).
    static final int WORKSPACE_ROWS = InferenceConfig.PREFILL_CHUNK_TOKENS;
    /// Most rows of one sequence in a verification: the decode kernels' row-exact limit
    /// (`ExecutionPlan.DECODE_MAX_ROWS`); a longer verification takes the prefill route, which is not row-exact.
    static final int MAX_ROWS = 8;

    private final Device device;
    private final CalibrationLattice lattice;
    /// The device's free bytes when the scorer opened, before any artifact: every artifact is planned against it.
    /// What the first artifact leaves in use (kernel modules, the runtime's per-process state) stays for the
    /// process and is what the residency planner's reserve stands for; planning against the free bytes left after
    /// it would count it twice and push weights to host memory.
    private final long freeAtOpen;

    private Scorer(Device device, CalibrationLattice lattice) {
        this.device = device;
        this.lattice = lattice;
        this.freeAtOpen = device.gpu.deviceMemoryInfo().freeBytes();
    }

    /// One device and the calibration lattice with `workers` workers, for any number of artifacts to be scored.
    public static Scorer open(CudaLibraries library, int workers) {
        Device device = new Device(library);
        try {
            return new Scorer(device, CalibrationLattice.start(workers));
        } catch (RuntimeException | Error failure) {
            device.close();
            throw failure;
        }
    }

    /// The most rows one batched verification may have for `artifact`: the workspace's rows, and, when it has an
    /// NVFP4 object, the rows its linears run on tensor cores.
    static int rowLimit(Artifact artifact) {
        for (TensorDescriptor tensor : artifact.tensors())
            if (tensor.format() == WeightFormat.NVFP4) return NVFP4_TENSOR_CORE_ROWS;
        return WORKSPACE_ROWS;
    }

    /// The chunks to force together, at most `chunks`, `rows` rows each, `rowLimit` rows in all: the sequences
    /// the device holds, one needing `singleBytes` and each more `sequenceBytes` (their KV and state), of
    /// `freeBytes`; 1 when not even a second fits.
    static int maxBatch(long freeBytes, long singleBytes, long sequenceBytes, int rows, int rowLimit, int chunks) {
        if (rows < 1 || rows > rowLimit)
            throw new IllegalArgumentException("--rows must be at least 1 and at most " + rowLimit);
        if (sequenceBytes < 1) throw new IllegalArgumentException("a sequence needs memory");
        long memory = freeBytes <= singleBytes ? 1 : 1 + (freeBytes - singleBytes) / sequenceBytes;
        return (int) Math.max(1, Math.min(Math.min(memory, rowLimit / rows), chunks));
    }

    /// `batch` when `rows` rows of each of its chunks are at most `rowLimit` together.
    static int checkedBatch(int batch, int rows, int rowLimit) {
        if (batch < 1 || (long) batch * rows > rowLimit)
            throw new IllegalArgumentException("--batch times --rows must be at most " + rowLimit + " (the rows of a"
                    + " verification the workspace holds)");
        return batch;
    }

    /// The bytes free on the device now.
    public long freeDeviceBytes() {
        return device.gpu.deviceMemoryInfo().freeBytes();
    }

    @Override
    public void close() {
        try {
            lattice.close();
        } finally {
            device.close();
        }
    }

    /// Loads `artifact` (read from `artifactPath`, each object from the file [Artifact#pathOf] names), scores it
    /// against every reference and unloads it. `batch` chunks are forced together; 0 or less sizes it to the
    /// memory the device has left ([#maxBatch]).
    public Result score(
            Path artifactPath, Artifact artifact, List<Path> references, int rows, int batch, long hostBytes)
            throws Exception {
        return result(run(artifactPath, artifact, references, rows, batch, hostBytes, (index, scored) -> {}));
    }

    /// As [#score], the per-reference results kept, `listener` told of each as it completes.
    List<Scored> run(
            Path artifactPath,
            Artifact artifact,
            List<Path> references,
            int rows,
            int requested,
            long hostBytes,
            Listener listener)
            throws Exception {
        if (rows < 1 || rows > MAX_ROWS)
            throw new IllegalArgumentException("--rows must be at least 1 and at most " + MAX_ROWS);
        int[] dimensions = dimensions(references);
        int context = dimensions[0], chunks = dimensions[1];
        int batch = requested <= 0
                ? autoBatch(artifact, hostBytes, rows, context, chunks)
                : checkedBatch(requested, rows, WORKSPACE_ROWS);
        // At most `hostBytes` of weights in host memory (no limit when planned), the rest read from disk, so an
        // artifact larger than the device and host budget together (BF16) is scored as `reference` runs it.
        var residency =
                Residency.plan(artifact, ArtifactProfile.Speculation.NONE, freeAtOpen, hostBytes, batch, context);
        try (Qwen38Model model = Qwen38Model.load(
                artifactPath,
                artifact,
                device.gpu,
                ArtifactProfile.Speculation.NONE,
                residency.hostBacked(),
                residency.diskBacked())) {
            Forced forced = force(model, artifactPath, artifact, references, rows, batch, listener);
            try {
                return forced.scored().get();
            } finally {
                forced.runtime().close();
            }
        }
    }

    /// A model's scoring under way: the execution to close once `scored` is done, and the results.
    record Forced(Execution runtime, CompletableFuture<List<Scored>> scored) {}

    /// Scores a loaded `model` against every reference, on the lattice's workers: the scorer's setup, then each
    /// reference in turn, each chunk group's verifications continuing the next. The scoring completes with the
    /// per-reference results; the caller closes the execution, then `model`. Nothing here waits.
    Forced force(
            Qwen38Model model,
            Path artifactPath,
            Artifact artifact,
            List<Path> references,
            int rows,
            int batch,
            Listener listener) {
        List<Scored> all = new ArrayList<>();
        ExecutionPlan plan = new ExecutionPlan(model.weights(), model.staging(), model.window());
        Execution runtime = execution(lattice, plan, device, artifactPath, artifact);
        CompletableFuture<List<Scored>> results = runtime.onWorker(
                        () -> new Run(device, runtime, plan, Residency.kvFormat(artifact), rows, batch))
                .thenCompose(run -> {
                    CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
                    for (int r = 0; r < references.size(); r++) {
                        int index = r;
                        Path path = references.get(r);
                        chain = chain.thenCompose(ignored -> run.reference(path).thenAccept(scored -> {
                            all.add(scored);
                            listener.scored(index, scored);
                        }));
                    }
                    return chain.thenApply(ignored -> all);
                });
        return new Forced(runtime, results);
    }

    /// The device the scorer scores on.
    Device device() {
        return device;
    }

    /// The device's free bytes when the scorer opened, which every artifact is planned against.
    long freeAtOpen() {
        return freeAtOpen;
    }

    /// The longest context and the most chunks over `references`.
    static int[] dimensions(List<Path> references) throws IOException {
        int context = 0, chunks = 0;
        for (Path path : references)
            try (Reference reference = Reference.open(path)) {
                context = Math.max(context, reference.context);
                chunks = Math.max(chunks, reference.chunks);
            }
        return new int[] {context, chunks};
    }

    /// The batch the device allows for `chunks` chunks of `context` tokens at `rows` rows ([#maxBatch]). With
    /// `hostBytes` given the weights' residency is not the planner's, so the memory left is unknown: the batch is
    /// then 4 chunks at most.
    int autoBatch(Artifact artifact, long hostBytes, int rows, int context, int chunks) {
        int limit = rowLimit(artifact);
        if (hostBytes != Residency.PLANNED) return Math.max(1, Math.min(Math.min(4, limit / rows), chunks));
        long[] figures = Residency.sequenceFigures(artifact, ArtifactProfile.Speculation.NONE, freeAtOpen, context);
        if (figures[0] < 0) return 1;
        int batch = maxBatch(freeAtOpen, figures[0], figures[1], rows, limit, chunks);
        System.out.printf(Locale.ROOT, "scorer: %d chunk(s) of %d rows together%n", batch, rows);
        return batch;
    }

    /// The engine's execution of `plan` on `lattice`, reading disk-backed weights from the file each object of
    /// `artifact` comes from.
    static Execution execution(
            CalibrationLattice lattice, ExecutionPlan plan, Device device, Path artifactPath, Artifact artifact) {
        return new Execution(
                lattice.terminal(),
                plan,
                device.gpu,
                EuhedralInferenceRuntime.laneCount(),
                Tuning.defaults(),
                artifactPath,
                artifact.sources());
    }

    /// The pooled statistics of `scored`: chunks of every reference together.
    static Result result(List<Scored> scored) {
        List<double[]> kl = new ArrayList<>(), agree = new ArrayList<>(), difference = new ArrayList<>();
        List<Double> chunkKl = new ArrayList<>(), chunkDifference = new ArrayList<>();
        int rows = 0;
        for (Scored reference : scored) {
            int steps = reference.reference().records;
            for (double[] values : reference.values()) {
                double[] k = new double[steps], a = new double[steps], d = new double[steps];
                for (int s = 0; s < steps; s++) {
                    k[s] = values[RESULTS * s + 2];
                    a[s] = values[RESULTS * s + 3];
                    d[s] = values[RESULTS * s] - values[RESULTS * s + 1];
                }
                kl.add(k);
                agree.add(a);
                difference.add(d);
                chunkKl.add(Summary.mean(k, 0, steps));
                chunkDifference.add(Summary.mean(d, 0, steps));
                rows += steps;
            }
        }
        double[] perChunkKl = chunkKl.stream().mapToDouble(Double::doubleValue).toArray();
        double[] perChunkDifference =
                chunkDifference.stream().mapToDouble(Double::doubleValue).toArray();
        double[] allKl = concatenate(kl, rows),
                allAgree = concatenate(agree, rows),
                allDifference = concatenate(difference, rows);
        return new Result(
                Summary.mean(allKl, 0, rows),
                Summary.standardError(perChunkKl, allKl),
                Summary.mean(allAgree, 0, rows),
                Summary.mean(allDifference, 0, rows),
                Summary.standardError(perChunkDifference, allDifference),
                perChunkKl);
    }

    private static double[] concatenate(List<double[]> parts, int length) {
        double[] all = new double[length];
        int at = 0;
        for (double[] part : parts) {
            System.arraycopy(part, 0, all, at, part.length);
            at += part.length;
        }
        return all;
    }

    /// The scoring of one loaded model: the device, the forcing session, the scorer's kernel and stream.
    private static final class Run {
        private final Device device;
        private final Execution runtime;
        private final ExecutionPlan plan;
        private final Forcing.Session session;
        private final Cuda.Function kernel;
        private final long scoreStream;
        private final int rows;

        Run(Device device, Execution runtime, ExecutionPlan plan, KvFormat kvFormat, int rows, int batch) {
            this.rows = rows;
            this.device = device;
            this.runtime = runtime;
            this.plan = plan;
            this.session = new Forcing.Session(runtime, plan, kvFormat, rows, batch);
            device.cuda
                    .bindThread(); // the worker's thread compiles the scorer's kernel; the engine selects streams here
            this.scoreStream = device.newStream();
            this.kernel = device.cuda
                    .compileResource("score.cu", List.of("score_rows"))
                    .getFirst();
        }

        /// Scores every chunk of `path`; completes on a worker once the last group has.
        CompletableFuture<Scored> reference(Path path) {
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
            Lane[] lanes = new Lane[session.lanes(reference.chunks)];
            for (int k = 0; k < lanes.length; k++) lanes[k] = new Lane(reference);
            return session.force(reference.tokens, reference.context, 0, new ScoreSink(reference, lanes, scored))
                    .thenApply(ignored -> new Scored(path, reference, scored, (System.nanoTime() - started) / 1e9))
                    .whenComplete((ignored, failure) -> {
                        for (Lane lane : lanes) lane.close();
                        closeQuietly(reference);
                    });
        }

        /// One slot of a group: the mapped buffers of the chunk it scores (two reference row blocks, the targets
        /// and the results).
        private final class Lane implements AutoCloseable {
            final long[] blocks = new long[2];
            final long targets, results;

            Lane(Reference reference) {
                for (int b = 0; b < 2; b++) blocks[b] = device.gpu.allocateHostWeights(rows * 2L * reference.width);
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

        /// Scores each logits row on the device against the reference row a worker has read ahead.
        private final class ScoreSink implements Forcing.Sink {
            final Reference reference;
            final Lane[] lanes;
            final double[][] scored;

            ScoreSink(Reference reference, Lane[] lanes, double[][] scored) {
                this.reference = reference;
                this.lanes = lanes;
                this.scored = scored;
            }

            @Override
            public void begin(int[] chunks) {
                int prefix = reference.context / 2 + 1;
                for (int k = 0; k < chunks.length; k++) {
                    int[] tokens = reference.tokens[chunks[k]];
                    MemorySegment.ofAddress(lanes[k].targets)
                            .reinterpret(4L * reference.records)
                            .copyFrom(MemorySegment.ofArray(
                                    Arrays.copyOfRange(tokens, prefix, prefix + reference.records)));
                }
            }

            /// Reads every chunk's reference rows of `group` into the lanes' blocks, on workers.
            @Override
            public CompletableFuture<Void> read(int[] chunks, int group, int first, int count) {
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

            /// Scores `count` logits rows from `row` against lane `lane`'s block of `group`. The kernel reads the
            /// reference rows and writes its results in mapped host memory; the caller waits for it before
            /// the
            /// logits are released.
            @Override
            public void rows(long row, int lane, int group, int first, int count) {
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
                                .pointer(device.gpu.hostWeightsDeviceAddress(lanes[lane].results)
                                        + 8L * RESULTS * first));
            }

            @Override
            public void synchronize() {
                device.cuda.synchronize(scoreStream);
            }

            @Override
            public void end(int[] chunks) {
                int steps = reference.records;
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
            }
        }
    }

    private static void closeQuietly(Reference reference) {
        try {
            reference.close();
        } catch (IOException ignored) {
            // A read-only file: nothing to lose.
        }
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
