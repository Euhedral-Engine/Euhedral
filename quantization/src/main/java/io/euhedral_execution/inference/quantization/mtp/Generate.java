package io.euhedral_execution.inference.quantization.mtp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.ingest.AbstractIngestSink;
import io.euhedral_execution.data_structures.queues.MpscQueue;
import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.Residency;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Function;

/// The MTP drafter's training corpus ([Corpus]): the artifact's own greedy answers to chat prompts, with the
/// post-final-norm hidden of every row the base model ran, which is what seeds the drafter
/// (docs/MTP_CONTRACT.md §2).
///
/// Each prompt is rendered by the checkpoint's chat template (thinking on, as the template defaults) and
/// prefilled in the engine's 512-row chunks, as the server prefills it. Up to `batch` sequences then decode
/// together, one row each per batched verification quantum: every row is bit for bit what the artifact's own
/// greedy decode computes, and the weights are read once per step for all of them. A sequence ends at a stop
/// token or after `maxTokens`.
///
/// The run's bookkeeping belongs to a lattice source ([Generation]); quanta report to it through posted
/// records.
public final class Generate {

    /// CUDA's per-thread default stream: the argmax launches of concurrent quanta never share a stream.
    private static final long PER_THREAD_STREAM = 2;

    /// Command-line settings.
    public record Settings(
            Path artifact,
            List<Path> prompts,
            Path tokenizer,
            Path out,
            int maxTokens,
            int batch,
            long hostBytes,
            int workers,
            int limit,
            Path library) {}

    private final Settings settings;
    private final Device device;
    private final Execution runtime;
    private final ExecutionPlan plan;
    private final Cuda.Function argmax;
    private final int hidden, vocabulary;
    private final Set<Integer> stops;
    private final AtomicLong sequenceIds = new AtomicLong(1);
    private final KvFormat kvFormat;
    /// Mapped host memory the argmax kernel writes: one int per lane for prefills, one per row for the step.
    private final long prefillChoices, stepChoices;

    private Generate(
            Settings settings,
            Device device,
            Execution runtime,
            ExecutionPlan plan,
            Set<Integer> stops,
            KvFormat kvFormat) {
        this.settings = settings;
        this.kvFormat = kvFormat;
        this.device = device;
        this.runtime = runtime;
        this.plan = plan;
        this.stops = stops;
        this.hidden = plan.weights().config().hiddenSize();
        this.vocabulary = plan.weights().config().vocabSize();
        this.argmax =
                device.cuda.compileResource("mtp.cu", List.of("argmax_rows")).getFirst();
        this.prefillChoices = device.gpu.allocateHostWeights(4L * settings.batch());
        this.stepChoices = device.gpu.allocateHostWeights(4L * settings.batch());
    }

    /// Longest rendered prompt the residency plan makes room for, ahead of `maxTokens` of generation.
    private static final int PROMPT_TOKENS = 4096;

    public static void run(Settings settings) throws Exception {
        if (settings.batch() < 1 || settings.batch() > 32)
            throw new IllegalArgumentException("--batch is 1 to 32 sequences (one row each per step)");
        var data = ArtifactReader.read(settings.artifact());
        try (Device device = new Device(settings.library());
                Qwen38Model model = Qwen38Model.load(
                        settings.artifact(),
                        data,
                        device.gpu,
                        ArtifactProfile.Speculation.MTP,
                        // A rendered prompt is at most PROMPT_TOKENS long ahead of its generation.
                        Residency.hostBacked(
                                data,
                                ArtifactProfile.Speculation.MTP,
                                device.gpu,
                                settings.hostBytes(),
                                settings.batch(),
                                PROMPT_TOKENS + settings.maxTokens()));
                var lattice = CalibrationLattice.start(settings.workers())) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            if (!plan.drafts()) throw new IllegalArgumentException(settings.artifact() + " has no MTP drafter");
            var runtime = new Execution(lattice.terminal(), plan, device.gpu);
            try {
                // On the lattice: the prompts rendered and the generator set up, then the generation's frames. This
                // thread only waits.
                runtime.onWorker(() -> start(settings, device, runtime, plan, lattice, Residency.kvFormat(data)))
                        .thenCompose(generation -> generation.finished.whenComplete((ignored, failure) -> {
                            generation.complete();
                            try {
                                generation.writer.close();
                            } catch (IOException closing) {
                                throw new java.io.UncheckedIOException(closing);
                            }
                        }))
                        .get();
            } finally {
                runtime.close();
            }
        }
    }

    /// Renders the prompts not yet answered and attaches a generation of them to the lattice.
    private static Generation start(
            Settings settings,
            Device device,
            Execution runtime,
            ExecutionPlan plan,
            CalibrationLattice lattice,
            KvFormat kvFormat) {
        try {
            // The worker's thread compiles the generator's kernels; no stream stays selected on it, since the engine's
            // quanta select their own on the same workers.
            device.cuda.bindThread();
            QwenTokenizer tokenizer = QwenTokenizer.load(settings.tokenizer());
            QwenChatTemplate template = QwenChatTemplate.load(settings.tokenizer());
            Files.createDirectories(settings.out());
            List<Prompt> prompts = new ArrayList<>();
            try (Corpus.Writer existing = Corpus.Writer.open(settings.out(), 0)) {
                Set<String> done = existing.ids();
                for (Path file : settings.prompts()) {
                    for (Prompt prompt : Prompt.read(file, template, tokenizer)) {
                        if (!done.contains(prompt.id())) prompts.add(prompt);
                        if (settings.limit() > 0 && prompts.size() >= settings.limit()) break;
                    }
                }
                System.out.printf(Locale.ROOT, "%d prompts to answer, %d done%n", prompts.size(), done.size());
            }
            Corpus.Writer writer =
                    Corpus.Writer.open(settings.out(), plan.weights().config().hiddenSize());
            var generate = new Generate(settings, device, runtime, plan, tokenizer.generationEosTokenIds(), kvFormat);
            var generation = generate.new Generation(prompts, writer);
            lattice.attach(generation);
            return generation;
        } catch (IOException failure) {
            throw new java.io.UncheckedIOException(failure);
        }
    }

    /// A rendered prompt.
    record Prompt(String id, int[] tokens) {
        static List<Prompt> read(Path file, QwenChatTemplate template, QwenTokenizer tokenizer) throws IOException {
            ObjectMapper json = new ObjectMapper();
            List<Prompt> prompts = new ArrayList<>();
            for (String line : Files.readAllLines(file)) {
                if (line.isBlank()) continue;
                JsonNode record = json.readTree(line);
                List<QwenChatTemplate.Turn> turns = new ArrayList<>();
                for (JsonNode message : record.path("messages")) {
                    var role = QwenChatTemplate.Role.valueOf(
                            message.path("role").asText().toUpperCase(Locale.ROOT));
                    turns.add(new QwenChatTemplate.Turn(
                            role, message.path("content").asText(""), toolCalls(json, message)));
                }
                List<Map<String, Object>> tools = new ArrayList<>();
                for (JsonNode tool : record.path("tools")) tools.add(json.convertValue(tool, Map.class));
                String text = template.render(turns, tools, QwenChatTemplate.Thinking.MODEL_DEFAULT);
                prompts.add(new Prompt(record.path("id").asText(), tokenizer.encodeText(text)));
            }
            return prompts;
        }

        /// An assistant message's OpenAI `tool_calls` (`function.arguments` a JSON object or its text).
        @SuppressWarnings("unchecked")
        static List<QwenChatTemplate.ToolCall> toolCalls(ObjectMapper json, JsonNode message) throws IOException {
            List<QwenChatTemplate.ToolCall> calls = new ArrayList<>();
            for (JsonNode call : message.path("tool_calls")) {
                JsonNode function = call.path("function");
                JsonNode arguments = function.path("arguments");
                if (arguments.isTextual()) arguments = json.readTree(arguments.asText());
                calls.add(new QwenChatTemplate.ToolCall(
                        function.path("name").asText(), json.convertValue(arguments, LinkedHashMap.class)));
            }
            return calls;
        }
    }

    private Sequence sequence() {
        Sequence sequence = new Sequence(sequenceIds.getAndIncrement());
        sequence.useKvFormat(kvFormat);
        return sequence;
    }

    /// One sequence being answered: its tokens so far, and the hidden row of each token the base model ran, in
    /// host memory sized for the longest answer.
    private final class Slot implements AutoCloseable {
        final Prompt prompt;
        /// Its place in the prefill choices.
        final int lane;
        final Sequence sequence = sequence();
        final Arena arena = Arena.ofShared();
        final MemorySegment rows;
        int[] tokens;
        int length, ran;
        /// The token the next step feeds: the base model's greedy choice after the last row it ran.
        int next;

        Slot(Prompt prompt, int lane) {
            this.prompt = prompt;
            this.lane = lane;
            int capacity = prompt.tokens().length + settings.maxTokens();
            this.tokens = Arrays.copyOf(prompt.tokens(), capacity + 1);
            this.length = prompt.tokens().length;
            this.rows = arena.allocate((long) capacity * hidden * Short.BYTES, 16);
        }

        /// Copies `count` hidden rows from the sequence's draft seed rows, which the quantum that ran them
        /// wrote.
        void keep(int count) {
            long bytes = (long) count * hidden * Short.BYTES;
            long seeds = ((AttentionStates) sequence.kvCacheState()).seedRows();
            device.gpu.copyDeviceToHost(rows.asSlice((long) ran * hidden * Short.BYTES, bytes), seeds, bytes);
            ran += count;
        }

        boolean finished() {
            return stops.contains(next) || length - prompt.tokens().length >= settings.maxTokens();
        }

        @Override
        public void close() {
            sequence.complete();
            arena.close();
        }
    }

    /// The owner of a run: which prompts are answered, which sequences decode, whether a step is in flight, and
    /// the corpus. Its fields change only while a worker polls it, which the lattice does one thread at a time;
    /// quanta report what they did by posting records, which the poll applies in order before it starts
    /// prefills and the next step.
    final class Generation extends AbstractIngestSink {
        final CompletableFuture<Void> finished = new CompletableFuture<>();
        private final MpscQueue<Runnable> records = new MpscQueue<>(256, 4);
        private final List<Prompt> prompts;
        final Corpus.Writer writer;
        private final List<Slot> active = new ArrayList<>();
        private final ArrayDeque<Integer> lanes = new ArrayDeque<>();
        private final Delegate delegate = new Delegate();
        private int started, prefilling, answered;
        private boolean stepping;
        private long generated;
        private final long begun = System.nanoTime();

        Generation(List<Prompt> prompts, Corpus.Writer writer) {
            this.prompts = prompts;
            this.writer = writer;
            for (int lane = 0; lane < settings.batch(); lane++) lanes.add(lane);
        }

        /// Posts `record` for the poll to apply. Any thread; never blocks.
        void post(Runnable record) {
            if (!records.offer(record)) throw new IllegalStateException("the generation refused a record");
        }

        private void poll() {
            if (finished.isDone()) return;
            try {
                for (Runnable record; (record = records.poll()) != null; ) record.run();
                retire();
                while (started < prompts.size() && !lanes.isEmpty()) {
                    Slot slot = new Slot(prompts.get(started++), lanes.poll());
                    prefilling++;
                    prefill(slot)
                            .whenComplete((done, failure) -> post(() -> {
                                prefilling--;
                                if (failure != null) throw new IllegalStateException("a prefill failed", failure);
                                active.add(slot);
                            }));
                }
                if (!stepping && !active.isEmpty()) {
                    List<Slot> stepped = List.copyOf(active);
                    stepping = true;
                    step(stepped)
                            .whenComplete((choices, failure) -> post(() -> {
                                stepping = false;
                                if (failure != null) throw new IllegalStateException("a step failed", failure);
                                for (int k = 0; k < stepped.size(); k++) {
                                    Slot slot = stepped.get(k);
                                    slot.tokens[slot.length++] = slot.next;
                                    slot.next = choices[k];
                                }
                            }));
                }
                if (started == prompts.size() && active.isEmpty() && prefilling == 0 && !stepping) {
                    System.out.printf(Locale.ROOT, "%d answers, %d tokens%n", answered, generated);
                    finished.complete(null);
                }
            } catch (RuntimeException | Error failure) {
                finished.completeExceptionally(failure);
            }
        }

        /// Writes the sequences whose last choice ends them, and frees their lanes.
        private void retire() {
            if (stepping) return;
            for (var iterator = active.iterator(); iterator.hasNext(); ) {
                Slot slot = iterator.next();
                if (!slot.finished()) continue;
                iterator.remove();
                slot.tokens[slot.length++] = slot.next;
                writer.write(
                        slot.prompt.id(), slot.tokens, slot.length, slot.prompt.tokens().length, slot.rows, slot.ran);
                generated += slot.length - slot.prompt.tokens().length;
                lanes.add(slot.lane);
                slot.close();
                if (++answered % 16 == 0) {
                    System.out.printf(
                            Locale.ROOT,
                            "%d/%d answered, %d tokens generated, %.0f tok/s%n",
                            answered,
                            prompts.size(),
                            generated,
                            generated / ((System.nanoTime() - begun) / 1e9));
                }
            }
        }

        @Override
        public LatticeSource getDelegate() {
            return delegate;
        }

        @Override
        public void complete() {
            delegate.complete();
        }

        @Override
        public boolean isComplete() {
            return delegate.isComplete();
        }

        /// The lattice calls `request` and `pull` of a source one thread at a time: the generation's state
        /// needs no other protection.
        private final class Delegate extends AbstractIngestSink.Delegate {
            @Override
            public long hookOnPull(
                    Consumer<AbstractFrame> consumer, Function<AbstractFrame, Boolean> stopCondition, long demand) {
                poll();
                return 0;
            }

            @Override
            public void hookOnRequest(LatticeReceiver terminal, long demand) {
                poll();
            }
        }
    }

    /// Prefills a slot's prompt in the engine's chunks, keeping every row's hidden; the last chunk's logits
    /// give the first token to feed.
    private CompletableFuture<Void> prefill(Slot slot) {
        int[] prompt = slot.prompt.tokens();
        CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
        for (int begin = 0; begin < prompt.length; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
            int[] part = Arrays.copyOfRange(
                    prompt, begin, Math.min(prompt.length, begin + InferenceConfig.PREFILL_CHUNK_TOKENS));
            boolean last = begin + InferenceConfig.PREFILL_CHUNK_TOKENS >= prompt.length;
            chain = chain.thenCompose(ignored -> {
                var quantum = new Quantum(
                                plan,
                                slot.sequence,
                                Quantum.ExecutionKind.PREFILL,
                                slot.sequence.currentTokenPosition(),
                                part,
                                last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE)
                        .seedingDraft();
                return succeeded(runtime.submit(quantum, done -> {
                    slot.keep(part.length);
                    if (!last) return;
                    try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        long row = logits.deviceAddress() + (long) (logits.tokenCount() - 1) * vocabulary * Short.BYTES;
                        slot.next = choose(row, 1, prefillChoices + 4L * slot.lane)[0];
                    }
                }));
            });
        }
        return chain;
    }

    /// One batched verification of each slot's next token: keeps each row's hidden; completes with the next
    /// choices.
    private CompletableFuture<int[]> step(List<Slot> slots) {
        List<Quantum.Part> parts = new ArrayList<>(slots.size());
        for (Slot slot : slots)
            parts.add(new Quantum.Part(slot.sequence, slot.sequence.currentTokenPosition(), new int[] {slot.next}));
        var quantum = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS)
                .seedingDraft();
        int[][] chosen = new int[1][];
        return succeeded(runtime.submit(quantum, done -> {
                    try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        chosen[0] = choose(logits.deviceAddress(), slots.size(), stepChoices);
                    }
                    for (Slot slot : slots) slot.keep(1);
                }))
                .thenApply(ignored -> chosen[0]);
    }

    /// The greedy token of each of `rows` BF16 logits rows from `logits`, through the mapped ints at `out`, on
    /// the calling thread's own default stream.
    private int[] choose(long logits, int rows, long out) {
        device.cuda.bindThread();
        device.cuda.launch(
                argmax,
                rows,
                1,
                256,
                0,
                PER_THREAD_STREAM,
                new Cuda.Arguments()
                        .pointer(logits)
                        .integer(vocabulary)
                        .pointer(device.gpu.hostWeightsDeviceAddress(out)));
        device.cuda.synchronize(PER_THREAD_STREAM);
        int[] chosen = new int[rows];
        MemorySegment.copy(
                MemorySegment.ofAddress(out).reinterpret(4L * rows), ValueLayout.JAVA_INT, 0, chosen, 0, rows);
        return chosen;
    }

    private static CompletableFuture<Void> succeeded(CompletableFuture<Quantum.Outcome> outcome) {
        return outcome.thenAccept(done -> {
            if (done.status() != Quantum.Status.SUCCESS)
                throw new IllegalStateException("quantum failed", done.failure());
        });
    }
}
