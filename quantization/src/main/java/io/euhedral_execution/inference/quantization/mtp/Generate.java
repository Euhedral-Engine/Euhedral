package io.euhedral_execution.inference.quantization.mtp;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

/// The MTP drafter's training corpus ([Corpus]): the artifact's own greedy answers to chat prompts, with the
/// post-final-norm hidden of every row the base model ran, which is what seeds the drafter (docs/MTP_CONTRACT.md §2).
///
/// Each prompt is rendered by the checkpoint's chat template (thinking on, as the template defaults) and prefilled in
/// the engine's 512-row chunks, as the server prefills it. Up to `batch` sequences then decode together, one row each
/// per batched verification quantum: every row is bit for bit what the artifact's own greedy decode computes, and the
/// weights are read once per step for all of them. A sequence ends at a stop token or after `maxTokens`.
public final class Generate {

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
    private final long argmaxStream;
    private final int hidden, vocabulary;
    private final Set<Integer> stops;
    private final AtomicLong sequenceIds = new AtomicLong(1);
    /// Mapped host memory the argmax kernel writes, one int per row of a step.
    private final long choices;

    private Generate(Settings settings, Device device, Execution runtime, ExecutionPlan plan, Set<Integer> stops) {
        this.settings = settings;
        this.device = device;
        this.runtime = runtime;
        this.plan = plan;
        this.stops = stops;
        this.hidden = plan.weights().config().hiddenSize();
        this.vocabulary = plan.weights().config().vocabSize();
        this.argmax =
                device.cuda.compileResource("mtp.cu", List.of("argmax_rows")).getFirst();
        this.argmaxStream = device.newStream();
        this.choices = device.gpu.allocateHostWeights(4L * Math.max(settings.batch(), 1));
    }

    public static void run(Settings settings) throws Exception {
        if (settings.batch() < 1 || settings.batch() > 32)
            throw new IllegalArgumentException("--batch is 1 to 32 sequences (one row each per step)");
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
        var data = ArtifactReader.read(settings.artifact());
        try (Device device = new Device(settings.library());
                Qwen38Model model = Qwen38Model.load(
                        settings.artifact(),
                        data,
                        device.gpu,
                        ArtifactProfile.Speculation.MTP,
                        HostWeightSelection.select(data, settings.hostBytes()));
                var lattice = CalibrationLattice.start(settings.workers())) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            if (!plan.drafts()) throw new IllegalArgumentException(settings.artifact() + " has no MTP drafter");
            var runtime = new Execution(lattice.terminal(), plan, device.gpu);
            try (Corpus.Writer writer =
                    Corpus.Writer.open(settings.out(), plan.weights().config().hiddenSize())) {
                new Generate(settings, device, runtime, plan, tokenizer.generationEosTokenIds())
                        .answer(prompts, writer);
            } finally {
                runtime.close();
            }
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
                            role, message.path("content").asText("")));
                }
                List<Map<String, Object>> tools = new ArrayList<>();
                for (JsonNode tool : record.path("tools")) tools.add(json.convertValue(tool, Map.class));
                String text = template.render(turns, tools, QwenChatTemplate.Thinking.MODEL_DEFAULT);
                prompts.add(new Prompt(record.path("id").asText(), tokenizer.encodeText(text)));
            }
            return prompts;
        }
    }

    /// One sequence being answered: its tokens so far, and the hidden row of each token the base model ran, in
    /// host memory sized for the longest answer.
    private final class Slot implements AutoCloseable {
        final Prompt prompt;
        final Sequence sequence = new Sequence(sequenceIds.getAndIncrement());
        final Arena arena = Arena.ofShared();
        final MemorySegment rows;
        int[] tokens;
        int length, ran;
        /// The token the next step feeds: the base model's greedy choice after the last row it ran.
        int next;

        Slot(Prompt prompt) {
            this.prompt = prompt;
            int capacity = prompt.tokens().length + settings.maxTokens();
            this.tokens = Arrays.copyOf(prompt.tokens(), capacity + 1);
            this.length = prompt.tokens().length;
            this.rows = arena.allocate((long) capacity * hidden * Short.BYTES, 16);
        }

        /// Copies `count` hidden rows from the sequence's draft seed rows, which the quantum that ran them wrote.
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

    private void answer(List<Prompt> prompts, Corpus.Writer writer) throws Exception {
        var ready = new LinkedBlockingQueue<Object>();
        List<Slot> active = new ArrayList<>();
        int started = 0, prefilling = 0, finished = 0;
        long generated = 0, begun = System.nanoTime();
        while (true) {
            while (started < prompts.size() && active.size() + prefilling < settings.batch()) {
                Slot slot = new Slot(prompts.get(started++));
                prefilling++;
                prefill(slot).whenComplete((done, failure) -> ready.add(failure != null ? failure : slot));
            }
            if (active.isEmpty() && prefilling == 0) break;
            // Wait for a prefill only when nothing can step.
            Object arrived = active.isEmpty() ? ready.take() : ready.poll();
            while (arrived != null) {
                prefilling--;
                if (arrived instanceof Throwable failure) throw new IllegalStateException("a prefill failed", failure);
                active.add((Slot) arrived);
                arrived = ready.poll();
            }
            // A prompt whose first choice already ends it has no answer to keep.
            for (var iterator = active.iterator(); iterator.hasNext(); ) {
                Slot slot = iterator.next();
                if (!slot.finished()) continue;
                iterator.remove();
                slot.tokens[slot.length++] = slot.next;
                writer.write(
                        slot.prompt.id(), slot.tokens, slot.length, slot.prompt.tokens().length, slot.rows, slot.ran);
                generated += slot.length - slot.prompt.tokens().length;
                finished++;
                slot.close();
                if (finished % 16 == 0) {
                    double seconds = (System.nanoTime() - begun) / 1e9;
                    System.out.printf(
                            Locale.ROOT,
                            "%d/%d answered, %d tokens generated, %.0f tok/s%n",
                            finished,
                            prompts.size(),
                            generated,
                            generated / seconds);
                }
            }
            if (!active.isEmpty()) step(active);
        }
        System.out.printf(Locale.ROOT, "%d answers, %d tokens%n", finished, generated);
    }

    /// Prefills a slot's prompt in the engine's chunks, keeping every row's hidden; the last chunk's logits give the
    /// first token to feed.
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
                        slot.next = choose(row, 1)[0];
                    }
                }));
            });
        }
        return chain;
    }

    /// One batched verification of each active slot's next token: keeps each row's hidden and the next choices.
    private void step(List<Slot> active) throws Exception {
        List<Quantum.Part> parts = new ArrayList<>(active.size());
        for (Slot slot : active)
            parts.add(new Quantum.Part(slot.sequence, slot.sequence.currentTokenPosition(), new int[] {slot.next}));
        var quantum = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS)
                .seedingDraft();
        int[][] chosen = new int[1][];
        succeeded(runtime.submit(quantum, done -> {
                    try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        chosen[0] = choose(logits.deviceAddress(), active.size());
                    }
                    for (Slot slot : active) slot.keep(1);
                }))
                .get();
        for (int k = 0; k < active.size(); k++) {
            Slot slot = active.get(k);
            slot.tokens[slot.length++] = slot.next;
            slot.next = chosen[0][k];
        }
    }

    /// The greedy token of each of `rows` BF16 logits rows from `logits`.
    private synchronized int[] choose(long logits, int rows) {
        device.cuda.bindThread();
        device.cuda.launch(
                argmax,
                rows,
                1,
                256,
                0,
                argmaxStream,
                new Cuda.Arguments()
                        .pointer(logits)
                        .integer(vocabulary)
                        .pointer(device.gpu.hostWeightsDeviceAddress(choices)));
        device.cuda.synchronize(argmaxStream);
        int[] out = new int[rows];
        MemorySegment.copy(
                MemorySegment.ofAddress(choices).reinterpret(4L * rows), ValueLayout.JAVA_INT, 0, out, 0, rows);
        return out;
    }

    private static CompletableFuture<Void> succeeded(CompletableFuture<Quantum.Outcome> outcome) {
        return outcome.thenAccept(done -> {
            if (done.status() != Quantum.Status.SUCCESS)
                throw new IllegalStateException("quantum failed", done.failure());
        });
    }
}
