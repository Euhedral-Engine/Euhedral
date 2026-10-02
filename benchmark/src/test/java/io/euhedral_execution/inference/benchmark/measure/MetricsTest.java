package io.euhedral_execution.inference.benchmark.measure;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import io.euhedral_execution.inference.benchmark.config.Scenario;
import io.euhedral_execution.inference.benchmark.prompt.PromptMaterial;
import io.euhedral_execution.inference.benchmark.result.BenchmarkResult;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetricsTest {
    private static final int EOS = 7;

    /// Entry 1000; five prompt tokens in chunks 2,2,1; three tokens selected; last one commit-only.
    static IterationTiming completedPromptToThree() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.entryNanos = 1_000;
        timing.promptEncoded(1_100, 5);
        timing.prefillQuantum(1_200, 1_500, 2);
        timing.prefillQuantum(1_510, 1_800, 2);
        timing.prefillQuantum(1_810, 2_000, 1);
        timing.firstTokenSelected(2_100, 1);
        timing.decodeQuantum(2_200, 2_300, 2_350, true, 2);
        timing.decodeQuantum(2_400, 2_500, 2_550, true, 3);
        timing.decodeQuantum(2_600, 2_700, 2_700, false, -1);
        timing.returned = true;
        timing.returnNanos = 2_800;
        return timing;
    }

    @Test
    void everyPhaseUsesItsOwnBoundariesAndActualWork() {
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.PROMPT_TO_N, 5, 3),
                completedPromptToThree(),
                List.of(1, 2, 3),
                id -> id == EOS,
                null);
        assertEquals(BenchmarkResult.SUCCESS, outcome.status());
        assertNull(outcome.reason());
        assertEquals(
                new BenchmarkResult.Work(
                        5, 3, 5, 3, 2, 1, false, null, BenchmarkResult.Work.outputSha256(List.of(1, 2, 3))),
                outcome.work());
        assertEquals(PromptMaterial.sha256("1,2,3"), outcome.work().outputSha256());
        assertEquals(
                new BenchmarkResult.Timings(100L, 800L, 100L, 1_100L, 350L, 300L, 100L, 1_550L, 1_800L),
                outcome.timings());
        // Prefill: 5 prompt tokens over 800 ns. Decode: tokens 2 and 3 over 350 ns; the first token
        // belongs to prefill and the unsampled final commit is excluded. End to end: 3 tokens over 1800 ns.
        assertEquals(new BenchmarkResult.Throughput(5e9 / 800, 2e9 / 350, 3e9 / 1_800), outcome.throughput());
    }

    /// Speculative decoding: two verification steps commit 3 and 2 tokens after the first, then the last
    /// token's commit-only quantum. The decode window starts at the first token, because the MTP
    /// catch-up that drafts the first verification runs after it.
    @Test
    void speculativeStepsCountTheirCommittedTokensAndDrafting() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.entryNanos = 1_000;
        timing.promptEncoded(1_100, 5);
        timing.prefillQuantum(1_200, 2_000, 5);
        timing.firstTokenSelected(2_100, 1);
        timing.speculativeStep(2_200, 2_400, 3, 2);
        timing.speculativeStep(2_500, 2_700, 2, 1);
        timing.decodeQuantum(2_750, 2_800, 2_800, false, -1);
        timing.returned = true;
        timing.returnNanos = 2_900;
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.PROMPT_TO_N, 5, 6),
                timing,
                List.of(1, 2, 3, 4, 5, 6),
                id -> id == EOS,
                null);
        assertEquals(BenchmarkResult.SUCCESS, outcome.status());
        var speculative = outcome.work().speculative();
        assertEquals(2, speculative.verifications());
        assertEquals(
                List.of(0L, 1L, 1L, 0L),
                java.util.Arrays.stream(speculative.acceptedDrafts()).boxed().toList());
        assertEquals(5, speculative.committedTokens());
        assertEquals(400, speculative.verifyNanos());
        assertEquals(200, speculative.draftNanos());
        assertEquals(5, outcome.work().decodeSampledTokens());
        assertEquals(1, outcome.work().finalCommitQuanta());
        assertEquals(
                new BenchmarkResult.Timings(100L, 800L, 100L, 1_100L, 600L, 400L, 50L, 1_700L, 1_900L),
                outcome.timings());
        assertEquals(new BenchmarkResult.Throughput(5e9 / 800, 5e9 / 600, 6e9 / 1_900), outcome.throughput());
    }

    @Test
    void prefillOnlyHasNoFirstTokenDecodeOrOutputRate() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.entryNanos = 0;
        timing.promptEncoded(10, 4);
        timing.prefillQuantum(20, 420, 4);
        timing.returned = true;
        timing.returnNanos = 500;
        var outcome = Metrics.evaluate(new Scenario(Scenario.Kind.PREFILL, 4, 0), timing, List.of(), id -> false, null);
        assertEquals(BenchmarkResult.SUCCESS, outcome.status());
        assertNull(outcome.timings().timeToFirstToken());
        assertNull(outcome.timings().firstTokenSample());
        assertNull(outcome.timings().decode());
        assertNull(outcome.timings().finalCommit());
        assertEquals(400L, outcome.timings().prefill());
        assertEquals(4e9 / 400, outcome.throughput().prefillTokensPerSecond());
        assertNull(outcome.throughput().decodeTokensPerSecond());
        assertNull(outcome.throughput().endToEndOutputTokensPerSecond(), "no output tokens, no output rate");
    }

    @Test
    void earlyEosRecordsActualTokensAndMarksTheSampleIneligible() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.promptEncoded(10, 2);
        timing.prefillQuantum(20, 40, 2);
        timing.firstTokenSelected(50, 1);
        timing.decodeQuantum(60, 80, 90, true, EOS);
        timing.returned = true;
        timing.returnNanos = 100;
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.SUSTAINED_DECODE, 2, 256), timing, List.of(1, EOS), id -> id == EOS, null);
        assertEquals(BenchmarkResult.INELIGIBLE, outcome.status());
        assertEquals("eos_after_2_of_256_tokens", outcome.reason());
        assertEquals(2, outcome.work().generatedTokens());
        assertEquals(1, outcome.work().decodeSampledTokens());
        assertEquals(0, outcome.work().finalCommitQuanta(), "a terminator is never committed");
        assertEquals(true, outcome.work().eosObserved());
        assertEquals(1e9 / 30, outcome.throughput().decodeTokensPerSecond());
    }

    @Test
    void stoppingShortWithoutEosIsAlsoIneligible() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.promptEncoded(1, 1);
        timing.prefillQuantum(2, 3, 1);
        timing.firstTokenSelected(4, 1);
        timing.returned = true;
        timing.returnNanos = 5;
        var outcome =
                Metrics.evaluate(new Scenario(Scenario.Kind.PROMPT_TO_N, 1, 4), timing, List.of(1), id -> false, null);
        assertEquals(BenchmarkResult.INELIGIBLE, outcome.status());
        assertEquals("stopped_after_1_of_4_tokens", outcome.reason());
    }

    @Test
    void firstTokenScenarioAcceptsATerminatorAsTheFirstToken() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.promptEncoded(1, 1);
        timing.prefillQuantum(2, 3, 1);
        timing.firstTokenSelected(9, EOS);
        timing.returned = true;
        timing.returnNanos = 10;
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.FIRST_TOKEN, 1, 1), timing, List.of(EOS), id -> id == EOS, null);
        assertEquals(BenchmarkResult.SUCCESS, outcome.status());
        assertEquals(9L, outcome.timings().timeToFirstToken());
    }

    @Test
    void firstNonTerminalTokenMustBeCommittedBeforeSuccess() {
        var timing = new IterationTiming(1, 1);
        timing.markEntry(0);
        timing.promptEncoded(1, 2);
        timing.prefillQuantum(2, 3, 2);
        timing.firstTokenSelected(4, 1);
        timing.markReturn(5);
        var scenario = new Scenario(Scenario.Kind.FIRST_TOKEN, 2, 1);
        var incomplete = Metrics.evaluate(scenario, timing, List.of(1), id -> false, null);
        assertEquals(BenchmarkResult.FAILED, incomplete.status());
        timing.decodeQuantum(5, 6, 6, false, -1);
        timing.markReturn(7);
        var complete = Metrics.evaluate(scenario, timing, List.of(1), id -> false, null);
        assertEquals(BenchmarkResult.SUCCESS, complete.status());
    }

    @Test
    void failuresReportNoTimingsOrThroughput() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.promptEncoded(1, 3);
        timing.prefillQuantum(2, 3, 2);
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.PREFILL, 3, 0),
                timing,
                null,
                id -> false,
                new IllegalStateException("Qwen execution quantum failed"));
        assertEquals(BenchmarkResult.FAILED, outcome.status());
        assertEquals("IllegalStateException: Qwen execution quantum failed", outcome.reason());
        assertNull(outcome.timings());
        assertNull(outcome.throughput());
        assertEquals(new BenchmarkResult.Work(3, 1, 2, null, null, null, null), outcome.work());
    }

    @Test
    void failureReasonRetainsTheUnderlyingExecutionError() {
        var failure = new IllegalStateException(
                "Qwen execution quantum failed", new IllegalStateException("CUDA_ERROR_UNSUPPORTED_PTX_VERSION"));
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.PREFILL, 3, 0), new IterationTiming(1, 1), null, id -> false, failure);
        assertEquals(
                "IllegalStateException: Qwen execution quantum failed; caused by IllegalStateException: "
                        + "CUDA_ERROR_UNSUPPORTED_PTX_VERSION",
                outcome.reason());
    }

    @Test
    void incompletePrefillIsAFailureWithoutRates() {
        var timing = new IterationTiming(1, 1);
        timing.entered = true;
        timing.promptEncoded(1, 3);
        timing.prefillQuantum(2, 3, 2);
        timing.returned = true;
        timing.returnNanos = 4;
        var outcome = Metrics.evaluate(new Scenario(Scenario.Kind.PREFILL, 3, 0), timing, List.of(), id -> false, null);
        assertEquals(BenchmarkResult.FAILED, outcome.status());
        assertEquals("incomplete_prefill", outcome.reason());
        assertNull(outcome.throughput());
    }

    @Test
    void completeTokenCountWithoutFirstSelectionIsNotAValidMeasurement() {
        var timing = completedPromptToThree();
        timing.firstSelected = false;
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.PROMPT_TO_N, 5, 3), timing, List.of(1, 2, 3), id -> false, null);
        assertEquals(BenchmarkResult.FAILED, outcome.status());
        assertNull(outcome.throughput());
    }

    @Test
    void completeTokenCountWithoutFinalCommitIsNotAValidMeasurement() {
        var timing = completedPromptToThree();
        timing.decodeCount--;
        var outcome = Metrics.evaluate(
                new Scenario(Scenario.Kind.SUSTAINED_DECODE, 5, 3), timing, List.of(1, 2, 3), id -> false, null);
        assertEquals(BenchmarkResult.FAILED, outcome.status());
        assertNull(outcome.throughput());
    }

    @Test
    void prefillOnlyCannotReportSampledTokens() {
        var timing = new IterationTiming(1, 1);
        timing.markEntry(0);
        timing.promptEncoded(1, 2);
        timing.prefillQuantum(2, 3, 2);
        timing.markReturn(4);
        var outcome =
                Metrics.evaluate(new Scenario(Scenario.Kind.PREFILL, 2, 0), timing, List.of(1), id -> false, null);
        assertEquals(BenchmarkResult.FAILED, outcome.status());
        assertNull(outcome.throughput());
    }

    @Test
    void ratesNeverDivideByZero() {
        assertNull(Metrics.perSecond(5, 0L));
        assertNull(Metrics.perSecond(5, -1L));
        assertNull(Metrics.perSecond(5, null));
        assertNull(Metrics.perSecond(0, 100L));
        assertEquals(2.0e9, Metrics.perSecond(2, 1L));
    }

    @Test
    void timingBuffersGrowBeyondTheirInitialCapacity() {
        var timing = new IterationTiming(1, 1);
        for (int index = 0; index < 10; index++) timing.prefillQuantum(index, index + 1, 1);
        for (int index = 0; index < 10; index++) timing.decodeQuantum(index, index, index, true, 1);
        assertEquals(10, timing.prefillCount);
        assertEquals(10, timing.decodeCount);
    }
}
