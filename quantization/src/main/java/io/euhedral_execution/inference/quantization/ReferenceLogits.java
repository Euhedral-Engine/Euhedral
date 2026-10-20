package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenKind;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;

/// The `reference` command: the logits of an artifact (the BF16 one, for the search's references) over a text, in
/// llama.cpp's `--kl-divergence-base` format ([ReferenceWriter]) so that [Scorer] reads them. The text is
/// tokenized by the engine's tokenizer, without BOS (Qwen adds none), and cut into consecutive chunks of `context`
/// tokens; each chunk is forced through the model as `score` forces one ([Forcing]), and the logits row at each
/// scored position is encoded as llama.cpp does. A stopped run continues after its last complete chunk.
public final class ReferenceLogits {

    /// Command-line settings; `chunks` 0 is as many whole chunks as the text holds.
    public record Settings(
            Path artifact,
            Path tokenizer,
            Path text,
            Path out,
            int context,
            int chunks,
            boolean parseSpecial,
            int rows,
            int batch,
            long hostBytes,
            int workers,
            CudaLibraries library) {}

    private ReferenceLogits() {}

    /// `text` as consecutive chunks of `context` tokens, at most `limit` of them when `limit` is positive. Control
    /// tokens in the text are single tokens, as the engine tokenizes them; without `parseSpecial` a text holding
    /// one is refused (llama.cpp reads their spelling as plain text).
    /// The model with the residency a reference over the BF16 artifact needs: as much in host memory as
    /// `--host-mib` allows, the rest read from disk.
    private static Qwen38Model load(Settings settings, Artifact artifact, Device device) throws IOException {
        var plan = Residency.plan(
                artifact,
                ArtifactProfile.Speculation.NONE,
                device.gpu,
                settings.hostBytes(),
                settings.batch(),
                settings.context());
        return Qwen38Model.load(
                settings.artifact(),
                artifact,
                device.gpu,
                ArtifactProfile.Speculation.NONE,
                plan.hostBacked(),
                plan.diskBacked());
    }

    static int[][] chunks(QwenTokenizer tokenizer, String text, int context, int limit, boolean parseSpecial) {
        int[] tokens = tokenizer.encodeText(text);
        if (!parseSpecial)
            for (int token : tokens)
                if (tokenizer.tokenKind(token) == TokenKind.CONTROL)
                    throw new IllegalArgumentException("the text holds the control token "
                            + tokenizer.decode(new int[] {token}) + "; pass --parse-special to make it one token");
        int whole = tokens.length / context;
        if (limit > 0 && limit > whole)
            throw new IllegalArgumentException("--chunks " + limit + " needs " + (long) limit * context
                    + " tokens, the text has " + tokens.length);
        int count = limit > 0 ? limit : whole;
        if (count < 1)
            throw new IllegalArgumentException(
                    "the text has " + tokens.length + " tokens, fewer than one chunk of " + context);
        int[][] chunks = new int[count][];
        for (int c = 0; c < count; c++) chunks[c] = Arrays.copyOfRange(tokens, c * context, (c + 1) * context);
        return chunks;
    }

    public static void run(Settings settings) throws Exception {
        if (settings.batch() < 1
                || settings.rows() < 1
                || settings.rows() > 8
                || settings.batch() * settings.rows() > 32)
            throw new IllegalArgumentException("--rows (at most 8) times --batch must be at most 32");
        if (settings.context() < 4) throw new IllegalArgumentException("--context is at least 4 tokens");
        QwenTokenizer tokenizer = QwenTokenizer.load(settings.tokenizer());
        int[][] chunks = chunks(
                tokenizer,
                Files.readString(settings.text(), StandardCharsets.UTF_8),
                settings.context(),
                settings.chunks(),
                settings.parseSpecial());
        System.out.printf(Locale.ROOT, "%d chunks of %d tokens%n", chunks.length, settings.context());
        var artifact = ArtifactReader.read(settings.artifact());
        try (Device device = new Device(settings.library());
                Qwen38Model model = load(settings, artifact, device);
                var lattice = CalibrationLattice.start(settings.workers())) {
            var plan = new ExecutionPlan(model.weights(), model.staging(), model.window());
            int vocabulary = plan.weights().config().vocabSize();
            try (ReferenceWriter writer =
                    ReferenceWriter.open(settings.out(), settings.context(), vocabulary, chunks)) {
                if (writer.completed() > 0)
                    System.out.printf(Locale.ROOT, "resuming after %d complete chunks%n", writer.completed());
                if (writer.completed() < chunks.length) {
                    var runtime = Scorer.execution(lattice, plan, device, settings.artifact(), artifact);
                    try {
                        // On the lattice's workers; this thread only waits.
                        runtime.onWorker(() -> new Forcing.Session(
                                        runtime, plan, Residency.kvFormat(artifact), settings.rows(), settings.batch()))
                                .thenCompose(session -> {
                                    int lanes = session.lanes(chunks.length - writer.completed());
                                    var sink = new WriteSink(device, vocabulary, settings.rows(), lanes, writer);
                                    return session.force(chunks, settings.context(), writer.completed(), sink)
                                            .whenComplete((ignored, failure) -> sink.close());
                                })
                                .get();
                    } finally {
                        runtime.close();
                    }
                }
                writer.finish();
            }
        }
        System.out.println("wrote " + settings.out());
    }

    /// Copies each logits row to host memory, encodes it and writes it, a group at a time; a group's chunks are
    /// recorded complete when the last of their rows is written.
    private static final class WriteSink implements Forcing.Sink, AutoCloseable {
        private final Device device;
        private final int vocabulary;
        private final ReferenceWriter writer;
        private final Arena arena = Arena.ofShared();
        private final MemorySegment[] hosts;
        private volatile int[] chunks;
        private final long started = System.nanoTime();

        WriteSink(Device device, int vocabulary, int rows, int lanes, ReferenceWriter writer) {
            this.device = device;
            this.vocabulary = vocabulary;
            this.writer = writer;
            this.hosts = new MemorySegment[lanes];
            for (int k = 0; k < lanes; k++) hosts[k] = arena.allocate(2L * rows * vocabulary, 8);
        }

        @Override
        public void begin(int[] chunks) {
            this.chunks = chunks;
        }

        @Override
        public CompletableFuture<Void> read(int[] chunks, int group, int first, int count) {
            return CompletableFuture.completedFuture(null);
        }

        @Override
        public void rows(long row, int lane, int group, int first, int count) {
            device.cuda.bindThread();
            MemorySegment host = hosts[lane];
            device.gpu.copyDeviceToHost(host, row, 2L * count * vocabulary);
            float[] logits = new float[vocabulary];
            try {
                for (int r = 0; r < count; r++) {
                    for (int i = 0; i < vocabulary; i++)
                        logits[i] = Float.intBitsToFloat((host.get(
                                                ValueLayout.JAVA_SHORT_UNALIGNED, 2L * ((long) r * vocabulary + i))
                                        & 0xffff)
                                << 16);
                    writer.write(chunks[lane], first + r, ReferenceWriter.encode(logits));
                }
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
        }

        @Override
        public void synchronize() {}

        @Override
        public void end(int[] done) {
            try {
                writer.complete(done[done.length - 1] + 1);
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            }
            System.out.printf(
                    Locale.ROOT,
                    "%d chunks complete (%.0f s)%n",
                    done[done.length - 1] + 1,
                    (System.nanoTime() - started) / 1e9);
        }

        @Override
        public void close() {
            arena.close();
        }
    }
}
