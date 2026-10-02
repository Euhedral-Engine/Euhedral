package io.euhedral_execution.inference.benchmark.measure;

import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult.Throughput;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult.Timings;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult.Work;
import java.util.List;
import java.util.function.IntPredicate;

/// Converts raw boundaries into metrics. Every duration comes from its own boundary pair; no phase
/// is derived by subtracting another phase from an aggregate.
///
/// - `tokenization`: generate entry to prompt encoded.
/// - `prefill`: start of the first prefill quantum to successful execution of the last; excludes
///   tokenization and first-token sampling. Throughput denominator: prompt tokens executed.
/// - `firstTokenSample`: last prefill execution to first token selected (logits sampling).
/// - `timeToFirstToken`: generate entry to first token selected (tokenization, prefill, sampling).
/// - `decode`: start of the first decode quantum to selection by the last sampling decode quantum.
///   It includes host work between those quanta (output callback, incremental text decoding) and
///   excludes the final commit-only quantum. Throughput numerator: tokens sampled by decode quanta,
///   which excludes the first token (sampled by prefill) and counts a sampled terminator.
///   With speculative decoding, the window runs from first token selection to the last verification
///   step's execution (it includes drafting), and its numerator is the tokens those steps committed.
/// - `decodeQuantaSum`: sum of the sampling decode quanta alone, without host work between them
///   (speculative: the verification quanta alone).
/// - `finalCommit`: the last token's unsampled commit quantum.
/// - `timeToLastToken`: generate entry to selection of the last returned token.
/// - `endToEnd`: generate entry to return, including the commit, callbacks, and decoder flush.
///   Throughput: returned token IDs / endToEnd.
public final class Metrics {
    private Metrics() {}

    public record Outcome(String status, String reason, Work work, Timings timings, Throughput throughput) {}

    public static Outcome evaluate(
            Scenario scenario, IterationTiming timing, List<Integer> tokens, IntPredicate isEos, Throwable failure) {
        int prefillTokens = 0;
        for (int index = 0; index < timing.prefillCount; index++) prefillTokens += timing.prefillTokens[index];
        Integer promptTokens = timing.encoded ? timing.promptTokens : null;
        if (failure != null || !timing.entered || !timing.returned || tokens == null) {
            String reason = failure == null ? "generation_did_not_return" : failureReason(failure);
            Work work = new Work(promptTokens, timing.prefillCount, prefillTokens, null, null, null, null);
            return new Outcome(BenchmarkResult.FAILED, reason, work, null, null);
        }

        int sampledDecode = 0;
        int lastSampled = -1;
        int commitOnly = -1;
        long decodeQuantaSum = 0;
        for (int index = 0; index < timing.decodeCount; index++) {
            if (timing.decodeSampled[index]) {
                sampledDecode++;
                lastSampled = index;
                decodeQuantaSum += timing.decodeSelected[index] - timing.decodeStart[index];
            } else {
                commitOnly = index;
            }
        }
        boolean eos = !tokens.isEmpty() && isEos.test(tokens.getLast());
        // Speculative steps replace sampled decode quanta: each commits one or more output tokens.
        BenchmarkResult.Speculative speculative = null;
        long firstDecodeStart = timing.decodeCount > 0 ? timing.decodeStart[0] : 0;
        long lastSelected = lastSampled >= 0 ? timing.decodeSelected[lastSampled] : 0;
        if (timing.stepCount > 0) {
            long[] histogram = new long[8];
            int committed = 0;
            long verify = 0, draft = 0;
            for (int step = 0; step < timing.stepCount; step++) {
                histogram[Math.min(7, timing.stepAccepted[step])]++;
                committed += timing.stepOutputs[step];
                verify += timing.stepExecuted[step] - timing.stepStart[step];
                draft +=
                        timing.stepStart[step] - (step > 0 ? timing.stepExecuted[step - 1] : timing.firstSelectedNanos);
            }
            int depth = 0;
            for (int a = 0; a < histogram.length; a++) if (histogram[a] > 0) depth = a;
            speculative = new BenchmarkResult.Speculative(
                    timing.stepCount,
                    java.util.Arrays.copyOf(histogram, Math.max(depth + 1, 4)),
                    committed,
                    verify,
                    draft);
            sampledDecode = committed;
            lastSampled = timing.stepCount - 1;
            // The window starts at the first token: the last prompt chunk's MTP catch-up, which drafts
            // the first verification, runs after it and belongs to decode.
            firstDecodeStart = timing.firstSelectedNanos;
            lastSelected = timing.stepExecuted[timing.stepCount - 1];
            decodeQuantaSum = verify;
        }
        Work work = new Work(
                promptTokens,
                timing.prefillCount,
                prefillTokens,
                tokens.size(),
                sampledDecode,
                commitOnly >= 0 ? 1 : 0,
                eos,
                speculative);

        Long prefill = timing.prefillCount == 0
                ? null
                : timing.prefillExecuted[timing.prefillCount - 1] - timing.prefillStart[0];
        Long firstSample = timing.firstSelected && timing.prefillCount > 0
                ? timing.firstSelectedNanos - timing.prefillExecuted[timing.prefillCount - 1]
                : null;
        Long timeToFirst = timing.firstSelected ? timing.firstSelectedNanos - timing.entryNanos : null;
        Long decode = lastSampled >= 0 ? lastSelected - firstDecodeStart : null;
        Long finalCommit = commitOnly >= 0 ? timing.decodeExecuted[commitOnly] - timing.decodeStart[commitOnly] : null;
        Long timeToLast = lastSampled >= 0 ? Long.valueOf(lastSelected - timing.entryNanos) : timeToFirst;
        long endToEnd = timing.returnNanos - timing.entryNanos;
        Timings timings = new Timings(
                timing.encoded ? timing.encodedNanos - timing.entryNanos : null,
                prefill,
                firstSample,
                timeToFirst,
                decode,
                lastSampled >= 0 ? decodeQuantaSum : null,
                finalCommit,
                timeToLast,
                endToEnd);
        Throughput throughput = new Throughput(
                perSecond(prefillTokens, prefill),
                perSecond(sampledDecode, decode),
                perSecond(tokens.size(), endToEnd));

        String status = BenchmarkResult.SUCCESS;
        String reason = null;
        if (promptTokens == null || prefillTokens != promptTokens || timing.prefillCount == 0) {
            status = BenchmarkResult.FAILED;
            reason = "incomplete_prefill";
        } else {
            switch (scenario.kind()) {
                case PREFILL -> {
                    if (!tokens.isEmpty() || timing.firstSelected || timing.decodeCount != 0) {
                        status = BenchmarkResult.FAILED;
                        reason = "inconsistent_generation_boundaries";
                    }
                }
                case FIRST_TOKEN -> {
                    if (tokens.size() != 1 || !timing.firstSelected) {
                        status = BenchmarkResult.FAILED;
                        reason = "no_first_token";
                    } else if (sampledDecode != 0 || (eos ? timing.decodeCount != 0 : commitOnly != 0)) {
                        status = BenchmarkResult.FAILED;
                        reason = "inconsistent_generation_boundaries";
                    }
                }
                case PROMPT_TO_N, SUSTAINED_DECODE -> {
                    if (!timing.firstSelected
                            || tokens.isEmpty()
                            || sampledDecode != tokens.size() - 1
                            || (eos && commitOnly >= 0)
                            || (!eos && tokens.size() == scenario.requestedNewTokens() && commitOnly < 0)) {
                        status = BenchmarkResult.FAILED;
                        reason = "inconsistent_generation_boundaries";
                    } else if (eos) {
                        status = BenchmarkResult.INELIGIBLE;
                        reason = "eos_after_" + tokens.size() + "_of_" + scenario.requestedNewTokens() + "_tokens";
                    } else if (tokens.size() < scenario.requestedNewTokens()) {
                        status = BenchmarkResult.INELIGIBLE;
                        reason = "stopped_after_" + tokens.size() + "_of_" + scenario.requestedNewTokens() + "_tokens";
                    }
                }
            }
        }
        if (BenchmarkResult.FAILED.equals(status)) return new Outcome(status, reason, work, null, null);
        return new Outcome(status, reason, work, timings, throughput);
    }

    /// Returns null instead of dividing by zero or reporting a rate for no completed work.
    public static Double perSecond(long count, Long nanos) {
        if (count <= 0 || nanos == null || nanos <= 0) return null;
        return count * 1_000_000_000.0 / nanos;
    }

    private static String failureReason(Throwable failure) {
        var reason = new StringBuilder(failure.getClass().getSimpleName())
                .append(": ")
                .append(failure.getMessage());
        for (int depth = 0; depth < 8 && failure.getCause() != null && failure.getCause() != failure; depth++) {
            failure = failure.getCause();
            reason.append("; caused by ")
                    .append(failure.getClass().getSimpleName())
                    .append(": ")
                    .append(failure.getMessage());
        }
        return reason.toString();
    }
}
