package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.testing.SharedQwen38;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Teacher-forced prefill and decode with every relaxed-order kernel against the exact oracle.
///
/// Two sequences receive the same tokens: a prefix by prefill, then one forced token per decode
/// quantum. Before each quantum the process-wide exact-numerics selection is switched, so sequence A
/// always runs the bitwise-exact kernels and sequence B every kernel whose FP32 accumulation order
/// was relaxed (contiguous Q3 decode, split-K FFN down, ...). Their errors combine, so this bounds
/// the cumulative effect.
/// Each step compares the final hidden state (before the final norm) and the logits row. Errors
/// must stay small and must not grow with position: the recurrent GDN and KV state carry any
/// difference forward, so progressive drift would show as a rising trend. With
/// `euhedral.numerics.drift.candidate=exact` both sequences run the exact kernels, which must agree bitwise.
@ModelGroup.CompactQ3
class RelaxedNumericsDriftCudaIntegrationTest {

    private static final int BURN_IN = 128;
    /// Logit lead of the exact top-1 token over the relaxed one at or below which a flip counts as a near-tie.
    private static final double NEAR_TIE = 0.25;

    private record Step(short[] hidden, float[] logits) {}

    private record Metrics(
            int position,
            double hiddenRelative,
            double hiddenMax,
            double logitMax,
            double logitRelative,
            double kl,
            boolean topAgrees,
            // Exact logit of the exact top-1 token minus that of the relaxed top-1 token (0 when they agree).
            double flipLead) {}

    @Test
    @Timeout(value = 600, unit = TimeUnit.SECONDS)
    void relaxedKernelsStayWithinBoundedErrorOfTheExactOracle() throws Throwable {
        // 256 steps per starting position: the 128-step burn-in, then two 64-step halves for the no-growth check, in
        // eight 32-step windows. Each decode step of the two sequences costs about 0.12 s.
        int steps = Integer.getInteger("euhedral.numerics.drift.steps", 256);
        // Calibration: "exact" runs the exact kernels on both sequences (determinism floor).
        boolean candidate = "exact".equals(System.getProperty("euhedral.numerics.drift.candidate", ""));
        Path tokenizerDirectory =
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"));
        String prefixes = System.getProperty("euhedral.numerics.drift.prefix", "256,384,512");
        QwenTokenizer tokenizer = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(tokenizerDirectory));
        List<Double> peaks = new ArrayList<>();
        List<Double> growths = new ArrayList<>();
        for (String value : prefixes.split(",")) {
            int prefixLength = Integer.parseInt(value.trim());
            List<Metrics> metrics = measure(tokenizer, prefixLength, steps, candidate);
            report(metrics);
            assertEquals(steps, metrics.size());
            // Outputs must agree at every starting position.
            double meanKl = metrics.stream().mapToDouble(Metrics::kl).average().orElseThrow();
            long agreements = metrics.stream().filter(Metrics::topAgrees).count();
            var decided = metrics.stream()
                    .filter(m -> !m.topAgrees() && m.flipLead() > NEAR_TIE)
                    .map(Metrics::flipLead)
                    .toList();
            assertTrue(meanKl < 1.0e-2, "mean KL divergence " + meanKl + " after prefix " + prefixLength);
            // Top-1 flips where the exact logits nearly tie depend on how many near-ties the weights give the text
            // (3 to 17 per 256 positions over the NVFP4 artifacts measured on 2026-10-13, almost all leads of 0 to
            // 0.25, two BF16 steps at these logits); a wrong kernel flips decided positions.
            assertTrue(
                    decided.size() <= 0.02 * metrics.size(),
                    "top-1 flips with an exact lead above " + NEAR_TIE + ": " + decided + " after prefix "
                            + prefixLength);
            int window = Math.max(1, metrics.size() / 8);
            double peak = 0;
            for (int start = 0; start < metrics.size(); start += window) {
                var part = metrics.subList(start, Math.min(metrics.size(), start + window));
                peak = Math.max(
                        peak,
                        part.stream()
                                .mapToDouble(Metrics::hiddenRelative)
                                .average()
                                .orElseThrow());
            }
            // Prefill is identical, so the error starts near zero and rises to its equilibrium level within about
            // 150 positions (calibration); after that burn-in, compare the later half with the earlier one.
            int burnIn = Math.min(BURN_IN, metrics.size() / 2);
            var settled = metrics.subList(burnIn, metrics.size());
            int half = settled.size() / 2;
            double early = settled.subList(0, half).stream()
                    .mapToDouble(Metrics::hiddenRelative)
                    .average()
                    .orElseThrow();
            double late = settled.subList(half, settled.size()).stream()
                    .mapToDouble(Metrics::hiddenRelative)
                    .average()
                    .orElseThrow();
            peaks.add(peak);
            growths.add(late / (early + 1.0e-4));
            System.out.printf(
                    "prefix %d: peak window hidden error %.4f, late/early %.3f, mean KL %.2e, top-1 %d/%d, decided flips %d%n",
                    prefixLength, peak, late / (early + 1.0e-4), meanKl, agreements, metrics.size(), decided.size());
        }
        // Single-ulp BF16 differences settle at a 5-9% hidden error in the 64-layer recurrent model; where one
        // window peaks depends on the tokens and the weights. For the q3 artifacts measured on 2026-10-12 the peak
        // window ran from 8% to 16% and the late/early ratio up to 1.7 with the starting position, so the
        // hidden-state bounds apply to the median over the starting positions: a relaxed kernel that is wrong raises
        // every position.
        double peak = median(peaks);
        double growth = median(growths);
        assertTrue(peak < 0.12, "median peak window hidden error " + peak + " over " + peaks);
        assertTrue(growth <= 1.5, "median hidden error growth " + growth + " over " + growths);
    }

    private static double median(List<Double> values) {
        var sorted = values.stream().sorted().toList();
        int n = sorted.size();
        return n % 2 == 1 ? sorted.get(n / 2) : (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2;
    }

    /// Teacher-forces `prefixLength` tokens by prefill and `steps` more by decode on an exact and a relaxed sequence.
    private static List<Metrics> measure(QwenTokenizer tokenizer, int prefixLength, int steps, boolean candidate)
            throws Throwable {
        int[] tokens = forcedTokens(tokenizer, prefixLength + steps);
        List<Metrics> metrics = new ArrayList<>();
        var loaded = SharedQwen38.q3();
        CudaGpuMemory gpu = loaded.gpu();
        Qwen38Model model = loaded.model();
        try (var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights());
            var runtime = new Execution(lattice, plan, gpu);
            var exact = new Sequence(1);
            var contiguous = new Sequence(2);
            try {
                int[] prefix = java.util.Arrays.copyOf(tokens, prefixLength);
                select(gpu, false, candidate);
                run(runtime, gpu, plan, exact, Quantum.ExecutionKind.PREFILL, prefix);
                select(gpu, true, candidate);
                run(runtime, gpu, plan, contiguous, Quantum.ExecutionKind.PREFILL, prefix);
                for (int step = 0; step < steps; step++) {
                    int[] token = {tokens[prefixLength + step]};
                    select(gpu, false, candidate);
                    Step a = run(runtime, gpu, plan, exact, Quantum.ExecutionKind.DECODE, token);
                    select(gpu, true, candidate);
                    Step b = run(runtime, gpu, plan, contiguous, Quantum.ExecutionKind.DECODE, token);
                    metrics.add(compare(prefixLength + step, a, b));
                }
            } finally {
                gpu.selectExactNumerics(false);
                exact.complete();
                contiguous.complete();
                runtime.close();
            }
        }
        return metrics;
    }

    /// Numerics for sequence A (the oracle: exact) or B (relaxed; exact when calibrating with `calibrate`).
    private static void select(CudaGpuMemory gpu, boolean sequenceB, boolean calibrate) {
        gpu.selectExactNumerics(!sequenceB || calibrate);
    }

    /// A fixed text: the README and three docs as they were when the bounds were calibrated. The window error
    /// depends on the tokens, and reading the live docs let an edit to the README fail the test.
    private static int[] forcedTokens(QwenTokenizer tokenizer, int count) throws Exception {
        String text;
        try (var input = RelaxedNumericsDriftCudaIntegrationTest.class.getResourceAsStream("drift-forced-text.md")) {
            text = new String(input.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
        int[] all = tokenizer.encodeText(text);
        if (all.length < count) throw new IllegalStateException("only " + all.length + " forced tokens available");
        return java.util.Arrays.copyOf(all, count);
    }

    private static Step run(
            Execution runtime,
            CudaGpuMemory gpu,
            ExecutionPlan plan,
            Sequence sequence,
            Quantum.ExecutionKind kind,
            int[] tokens)
            throws Exception {
        AtomicReference<Step> captured = new AtomicReference<>();
        var context = new Quantum(
                plan, sequence, kind, sequence.currentTokenPosition(), tokens, LogitsRequirement.LAST_TOKEN);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined()) {
                        var workspace = done.workspace();
                        int hiddenWidth = plan.weights().config().hiddenSize();
                        long hiddenAddress = workspace.address(ExecutionPlan.Buffer.FINAL_HIDDEN_STATE)
                                + (long) (tokens.length - 1) * hiddenWidth * Short.BYTES;
                        short[] hidden = shorts(gpu, arena, hiddenAddress, hiddenWidth);
                        try (DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                            int vocabulary = logits.vocabularySize();
                            short[] row = shorts(
                                    gpu,
                                    arena,
                                    logits.deviceAddress()
                                            + (long) (logits.tokenCount() - 1) * vocabulary * Short.BYTES,
                                    vocabulary);
                            float[] values = new float[vocabulary];
                            for (int i = 0; i < vocabulary; i++) values[i] = Float.intBitsToFloat(row[i] << 16);
                            captured.set(new Step(hidden, values));
                        }
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }

    private static short[] shorts(CudaGpuMemory gpu, Arena arena, long device, int count) {
        MemorySegment host = arena.allocate((long) count * Short.BYTES, Short.BYTES);
        gpu.copyDeviceToHost(host, device, host.byteSize());
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN), 0, values, 0, count);
        return values;
    }

    private static Metrics compare(int position, Step a, Step b) {
        double hiddenNorm = 0, hiddenDifference = 0, hiddenMax = 0;
        for (int i = 0; i < a.hidden().length; i++) {
            double x = Float.intBitsToFloat(a.hidden()[i] << 16), y = Float.intBitsToFloat(b.hidden()[i] << 16);
            hiddenNorm += x * x;
            hiddenDifference += (x - y) * (x - y);
            hiddenMax = Math.max(hiddenMax, Math.abs(x - y));
        }
        float[] p = a.logits(), q = b.logits();
        double logitNorm = 0,
                logitDifference = 0,
                logitMax = 0,
                maxP = Double.NEGATIVE_INFINITY,
                maxQ = Double.NEGATIVE_INFINITY;
        int topP = 0, topQ = 0;
        for (int i = 0; i < p.length; i++) {
            logitNorm += (double) p[i] * p[i];
            logitDifference += (double) (p[i] - q[i]) * (p[i] - q[i]);
            logitMax = Math.max(logitMax, Math.abs(p[i] - q[i]));
            if (p[i] > maxP) {
                maxP = p[i];
                topP = i;
            }
            if (q[i] > maxQ) {
                maxQ = q[i];
                topQ = i;
            }
        }
        double sumP = 0, sumQ = 0;
        for (int i = 0; i < p.length; i++) {
            sumP += Math.exp(p[i] - maxP);
            sumQ += Math.exp(q[i] - maxQ);
        }
        double logZp = maxP + Math.log(sumP), logZq = maxQ + Math.log(sumQ), kl = 0;
        for (int i = 0; i < p.length; i++) {
            double logPi = p[i] - logZp;
            kl += Math.exp(logPi) * (logPi - (q[i] - logZq));
        }
        return new Metrics(
                position,
                Math.sqrt(hiddenDifference / hiddenNorm),
                hiddenMax,
                logitMax,
                Math.sqrt(logitDifference / logitNorm),
                Math.max(kl, 0),
                topP == topQ,
                p[topP] - p[topQ]);
    }

    private static void report(List<Metrics> metrics) throws Exception {
        String path = System.getProperty("euhedral.numerics.drift.report");
        if (path != null && !path.isBlank()) {
            StringBuilder csv =
                    new StringBuilder("position,hidden_relative,hidden_max,logit_max,logit_relative,kl,top_agrees\n");
            for (Metrics m : metrics) {
                csv.append(m.position())
                        .append(',')
                        .append(m.hiddenRelative())
                        .append(',')
                        .append(m.hiddenMax())
                        .append(',')
                        .append(m.logitMax())
                        .append(',')
                        .append(m.logitRelative())
                        .append(',')
                        .append(m.kl())
                        .append(',')
                        .append(m.topAgrees() ? 1 : 0)
                        .append('\n');
            }
            Files.writeString(Path.of(path), csv);
        }
        int window = Math.max(1, metrics.size() / 8);
        for (int start = 0; start < metrics.size(); start += window) {
            var part = metrics.subList(start, Math.min(metrics.size(), start + window));
            System.out.printf(
                    "positions %5d-%5d hidden rel mean %.3e max %.3e | logit max %.3e rel %.3e | KL mean %.2e max %.2e | top-1 %d/%d%n",
                    part.getFirst().position(),
                    part.getLast().position(),
                    part.stream().mapToDouble(Metrics::hiddenRelative).average().orElseThrow(),
                    part.stream().mapToDouble(Metrics::hiddenRelative).max().orElseThrow(),
                    part.stream().mapToDouble(Metrics::logitMax).max().orElseThrow(),
                    part.stream().mapToDouble(Metrics::logitRelative).average().orElseThrow(),
                    part.stream().mapToDouble(Metrics::kl).average().orElseThrow(),
                    part.stream().mapToDouble(Metrics::kl).max().orElseThrow(),
                    part.stream().filter(Metrics::topAgrees).count(),
                    part.size());
        }
    }
}
