package io.euhedral_execution.inference.benchmark.measure;

import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import java.util.Arrays;

/// Raw boundaries for one `generate` call. `entry`/`returned` are taken immediately around
/// `QwenGenerationSession.generate`; everything else comes from the session timing hook.
/// Arrays are preallocated so recording does not allocate for typical runs.
public final class IterationTiming implements GenerationTimingListener {
    boolean entered;
    boolean returned;
    long entryNanos;
    long returnNanos;
    boolean encoded;
    long encodedNanos;
    int promptTokens;
    int restoredTokens;
    long restoreNanos;
    int prefillCount;
    long[] prefillStart;
    long[] prefillExecuted;
    int[] prefillTokens;
    boolean firstSelected;
    long firstSelectedNanos;
    int decodeCount;
    long[] decodeStart;
    long[] decodeExecuted;
    long[] decodeSelected;
    boolean[] decodeSampled;
    int stepCount;
    long[] stepStart = new long[16];
    long[] stepExecuted = new long[16];
    int[] stepOutputs = new int[16];
    int[] stepAccepted = new int[16];
    int[] stepRejection = new int[16];

    public IterationTiming(int expectedPrefillQuanta, int expectedDecodeQuanta) {
        int prefill = Math.max(1, expectedPrefillQuanta);
        int decode = Math.max(1, expectedDecodeQuanta);
        this.prefillStart = new long[prefill];
        this.prefillExecuted = new long[prefill];
        this.prefillTokens = new int[prefill];
        this.decodeStart = new long[decode];
        this.decodeExecuted = new long[decode];
        this.decodeSelected = new long[decode];
        this.decodeSampled = new boolean[decode];
    }

    /// Marks the instant immediately before `QwenGenerationSession.generate` is called.
    public void markEntry() {
        markEntry(System.nanoTime());
    }

    /// Marks the instant immediately after `QwenGenerationSession.generate` returns.
    public void markReturn() {
        markReturn(System.nanoTime());
    }

    /// Records entry at a caller-supplied time from the same clock as the listener timestamps.
    public void markEntry(long nanos) {
        this.entered = true;
        this.entryNanos = nanos;
    }

    /// Records return at a caller-supplied time from the same clock as the listener timestamps.
    public void markReturn(long nanos) {
        this.returnNanos = nanos;
        this.returned = true;
    }

    @Override
    public void promptEncoded(long nanos, int promptTokens) {
        this.encoded = true;
        this.encodedNanos = nanos;
        this.promptTokens = promptTokens;
    }

    @Override
    public void prefixRestored(int tokens, long nanos) {
        this.restoredTokens = tokens;
        this.restoreNanos = nanos;
    }

    @Override
    public void prefillQuantum(long startNanos, long executedNanos, int tokens) {
        if (this.prefillCount == this.prefillStart.length) {
            int size = this.prefillCount * 2;
            this.prefillStart = Arrays.copyOf(this.prefillStart, size);
            this.prefillExecuted = Arrays.copyOf(this.prefillExecuted, size);
            this.prefillTokens = Arrays.copyOf(this.prefillTokens, size);
        }
        this.prefillStart[this.prefillCount] = startNanos;
        this.prefillExecuted[this.prefillCount] = executedNanos;
        this.prefillTokens[this.prefillCount++] = tokens;
    }

    @Override
    public void firstTokenSelected(long nanos, int tokenId) {
        this.firstSelected = true;
        this.firstSelectedNanos = nanos;
    }

    @Override
    public void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts) {
        speculativeStep(startNanos, executedNanos, outputs, acceptedDrafts, -1);
    }

    @Override
    public void speculativeStep(long startNanos, long executedNanos, int outputs, int acceptedDrafts, int rejection) {
        if (this.stepCount == this.stepStart.length) {
            int size = this.stepCount * 2;
            this.stepStart = Arrays.copyOf(this.stepStart, size);
            this.stepExecuted = Arrays.copyOf(this.stepExecuted, size);
            this.stepOutputs = Arrays.copyOf(this.stepOutputs, size);
            this.stepAccepted = Arrays.copyOf(this.stepAccepted, size);
            this.stepRejection = Arrays.copyOf(this.stepRejection, size);
        }
        this.stepStart[this.stepCount] = startNanos;
        this.stepExecuted[this.stepCount] = executedNanos;
        this.stepOutputs[this.stepCount] = outputs;
        this.stepRejection[this.stepCount] = rejection;
        this.stepAccepted[this.stepCount++] = acceptedDrafts;
    }

    /// Drafting quanta by phase: count and summed nanoseconds.
    final java.util.Map<String, long[]> draftPhases = new java.util.TreeMap<>();

    @Override
    public void draftQuantum(String phase, long startNanos, long executedNanos) {
        long[] totals = this.draftPhases.computeIfAbsent(phase, ignored -> new long[2]);
        totals[0]++;
        totals[1] += executedNanos - startNanos;
    }

    @Override
    public void decodeQuantum(
            long startNanos, long executedNanos, long selectedNanos, boolean sampled, int selectedTokenId) {
        if (this.decodeCount == this.decodeStart.length) {
            int size = this.decodeCount * 2;
            this.decodeStart = Arrays.copyOf(this.decodeStart, size);
            this.decodeExecuted = Arrays.copyOf(this.decodeExecuted, size);
            this.decodeSelected = Arrays.copyOf(this.decodeSelected, size);
            this.decodeSampled = Arrays.copyOf(this.decodeSampled, size);
        }
        this.decodeStart[this.decodeCount] = startNanos;
        this.decodeExecuted[this.decodeCount] = executedNanos;
        this.decodeSelected[this.decodeCount] = selectedNanos;
        this.decodeSampled[this.decodeCount++] = sampled;
    }
}
