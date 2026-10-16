package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Quantum;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.state.KvFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/// Teacher forcing of token chunks through a loaded model, shared by the scorer and the reference writer. Each
/// chunk of N tokens is forced on a fresh sequence: a prefill of N / 2 + 1 tokens, then verifications of `rows`
/// forced tokens, so that step k's logits row (k = 0 .. N - 2 - N / 2) predicts the token at position N / 2 + 1 + k.
/// `batch` chunks run together, one sequence each, their verifications batched into one quantum (at most 32
/// rows), so each weight is read once per step group of all of them. What is done with a step's logits row, and
/// what is fetched ahead for it, is the [Sink]'s.
final class Forcing {

    private Forcing() {}

    /// What a command does with the rows of the chunks forced together (a group).
    interface Sink {
        /// The group's chunks are about to run; on a worker.
        void begin(int[] chunks);

        /// Fetches what the rows of `group` (`count` of them from step `first`) need, for every chunk of the
        /// group; completes on a worker. Group 0 is step 0 alone.
        CompletableFuture<Void> read(int[] chunks, int group, int first, int count);

        /// `count` rows of BF16 logits at device address `row`, of the group's chunk number `lane`, from step
        /// `first`. Called with the logits alive: the rows are consumed, or started to be, before it returns.
        void rows(long row, int lane, int group, int first, int count);

        /// Waits for what [#rows] started; called once after the rows of every lane of a quantum.
        void synchronize();

        /// The group's chunks have all been forced.
        void end(int[] chunks);
    }

    /// The forcing on one loaded model.
    static final class Session {
        private final Execution runtime;
        private final ExecutionPlan plan;
        private final KvFormat kvFormat;
        private final int rows, batch;
        private final AtomicLong sequenceIds = new AtomicLong(1);

        Session(Execution runtime, ExecutionPlan plan, KvFormat kvFormat, int rows, int batch) {
            this.runtime = runtime;
            this.plan = plan;
            this.kvFormat = kvFormat;
            this.rows = rows;
            this.batch = batch;
        }

        /// The chunks one group holds when `chunks` are to be forced: the sink needs this many slots.
        int lanes(int chunks) {
            return Math.min(batch, chunks);
        }

        /// Forces `tokens[from ..]`, `lanes(tokens.length - from)` chunks at a time; completes on a worker once the
        /// last group has.
        CompletableFuture<Void> force(int[][] tokens, int context, int from, Sink sink) {
            int lanes = lanes(tokens.length - from);
            CompletableFuture<Void> chain = CompletableFuture.completedFuture(null);
            for (int first = from; first < tokens.length; first += lanes) {
                int[] chunks = new int[Math.min(lanes, tokens.length - first)];
                for (int k = 0; k < chunks.length; k++) chunks[k] = first + k;
                chain = chain.thenCompose(ignored -> new Group(tokens, context, chunks, sink).run());
            }
            return chain;
        }

        /// Chunks forced together, one sequence each, in lockstep (every chunk has the same prefix and steps): each
        /// prefill on its own, then each step group of every chunk in one batched verification.
        private final class Group {
            final int[][] tokens;
            final int[] chunks;
            final Sink sink;
            final Sequence[] sequences;
            final int prefix, steps, groups;

            Group(int[][] tokens, int context, int[] chunks, Sink sink) {
                this.tokens = tokens;
                this.chunks = chunks;
                this.sink = sink;
                this.sequences = new Sequence[chunks.length];
                this.prefix = context / 2 + 1;
                this.steps = context - 1 - context / 2;
                // Group 0 is step 0, from the prefill's last row; group g > 0 covers steps 1 + rows (g - 1) onwards.
                this.groups = 1 + (steps - 1 + rows - 1) / rows;
            }

            CompletableFuture<Void> run() {
                sink.begin(chunks);
                for (int k = 0; k < chunks.length; k++) {
                    sequences[k] = new Sequence(sequenceIds.getAndIncrement());
                    sequences[k].useKvFormat(kvFormat);
                }
                // A group's rows are read while the quantum before it runs: block g % 2 was last used by group
                // g - 2, finished by then.
                @SuppressWarnings("unchecked")
                CompletableFuture<Void>[] reads = new CompletableFuture[groups];
                reads[0] = sink.read(chunks, 0, 0, 1);
                if (groups > 1) reads[1] = sink.read(chunks, 1, 1, Math.min(rows, steps - 1));
                List<CompletableFuture<Void>> prefills = new ArrayList<>();
                for (int k = 0; k < chunks.length; k++) {
                    int lane = k;
                    int[] chunk = tokens[chunks[k]];
                    CompletableFuture<Void> chain = reads[0];
                    for (int begin = 0; begin < prefix; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
                        int[] part = Arrays.copyOfRange(
                                chunk, begin, Math.min(prefix, begin + InferenceConfig.PREFILL_CHUNK_TOKENS));
                        boolean last = begin + InferenceConfig.PREFILL_CHUNK_TOKENS >= prefix;
                        chain = chain.thenCompose(ignored -> prefill(lane, part, last));
                    }
                    prefills.add(chain);
                }
                CompletableFuture<Void> done = CompletableFuture.allOf(prefills.toArray(CompletableFuture[]::new));
                for (int g = 1; g < groups; g++) {
                    int group = g, first = 1 + rows * (g - 1), count = Math.min(rows, steps - first);
                    done = done.thenCompose(ignored -> {
                        if (group + 1 < groups)
                            reads[group + 1] =
                                    sink.read(chunks, group + 1, first + rows, Math.min(rows, steps - (first + rows)));
                        return reads[group].thenCompose(alsoIgnored -> verify(group, first, count));
                    });
                }
                return done.handle((ignored, failure) -> {
                    for (Sequence sequence : sequences) sequence.complete();
                    if (failure != null)
                        throw new IllegalStateException("chunks " + Arrays.toString(chunks) + " failed", failure);
                    sink.end(chunks);
                    return null;
                });
            }

            /// One prefill quantum of lane `lane`'s sequence; the last one's logits row is step 0.
            CompletableFuture<Void> prefill(int lane, int[] part, boolean last) {
                Sequence sequence = sequences[lane];
                var quantum = new Quantum(
                        plan,
                        sequence,
                        Quantum.ExecutionKind.PREFILL,
                        sequence.currentTokenPosition(),
                        part,
                        last ? LogitsRequirement.LAST_TOKEN : LogitsRequirement.NONE);
                return succeeded(runtime.submit(quantum, done -> {
                    if (!last) return;
                    try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        long row = logits.deviceAddress()
                                + (long) (logits.tokenCount() - 1) * logits.vocabularySize() * Short.BYTES;
                        sink.rows(row, lane, 0, 0, 1);
                        sink.synchronize();
                    }
                }));
            }

            /// One batched verification of `count` forced tokens per chunk from step `first`, as `group`.
            CompletableFuture<Void> verify(int group, int first, int count) {
                List<Quantum.Part> parts = new ArrayList<>();
                for (int k = 0; k < chunks.length; k++) {
                    int[] forced =
                            Arrays.copyOfRange(tokens[chunks[k]], prefix + first - 1, prefix + first - 1 + count);
                    parts.add(new Quantum.Part(sequences[k], sequences[k].currentTokenPosition(), forced));
                }
                var quantum = Quantum.batch(plan, Quantum.ExecutionKind.VERIFY, parts, LogitsRequirement.ALL_TOKENS);
                return succeeded(runtime.submit(quantum, done -> {
                    try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        long rowBytes = (long) logits.vocabularySize() * Short.BYTES;
                        for (int k = 0; k < chunks.length; k++)
                            sink.rows(logits.deviceAddress() + (long) k * count * rowBytes, k, group, first, count);
                        sink.synchronize();
                    }
                }));
            }
        }

        private static CompletableFuture<Void> succeeded(CompletableFuture<Quantum.Outcome> outcome) {
            return outcome.thenAccept(done -> {
                if (done.status() != Quantum.Status.SUCCESS)
                    throw new IllegalStateException("quantum failed", done.failure());
            });
        }
    }
}
