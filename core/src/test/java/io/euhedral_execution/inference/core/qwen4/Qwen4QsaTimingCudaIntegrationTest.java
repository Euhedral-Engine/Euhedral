package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.open;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// Times the QSA layer on the real layer-3 weights: the whole block and its stages for 1, 64 and 512 rows
/// at
/// histories from 1K to 262K tokens, the history built by running the layer itself over random rows.
/// Prints a
/// table; asserts nothing (the numbers go to docs/FLASH_NEXT_QSA.md). Enable with the environment
/// variable EUHEDRAL_QWEN4_TIMING=1.
class Qwen4QsaTimingCudaIntegrationTest {

    private static final int[] HISTORIES = {1024, 8192, 32768, 131072, 261632};
    private static final int[] ROWS = {1, 64, 512};

    private static double timeMs(CudaGpuMemory gpu, GpuStream stream, int reps, Runnable launches) {
        stream.submit(launches, false);
        gpu.synchronize();
        long begin = System.nanoTime();
        stream.submit(
                () -> {
                    for (int i = 0; i < reps; i++) launches.run();
                },
                false);
        gpu.synchronize();
        return (System.nanoTime() - begin) / 1e6 / reps;
    }

    /// Random rows of magnitude 0.5 to 1 with random signs: cheap to make, enough for timing.
    private static void fillRandom(MemorySegment segment, SplittableRandom rng, int rows) {
        for (long i = 0; i < (long) rows * 2560; i++) {
            int r = rng.nextInt();
            short bits = (short) (0x3f00 | (r & 0x7f) | ((r >>> 8) & 0x8000));
            segment.setAtIndex(ValueLayout.JAVA_SHORT, i, bits);
        }
    }

    @Test
    void timings() throws IOException {
        assumeTrue(System.getenv("EUHEDRAL_QWEN4_TIMING") != null, "set EUHEDRAL_QWEN4_TIMING=1 to run the timings");
        SplittableRandom rng = new SplittableRandom(5);
        int maxTokens = 262144;
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined();
                Qwen4QsaTestSupport.Loaded loaded = Qwen4QsaTestSupport.Loaded.load(gpu, 3);
                GpuStream stream = gpu.openStream()) {
            Qwen4QsaLayer layer =
                    new Qwen4QsaLayer(Qwen4QsaLayer.Config.of(loaded.artifact().config(), maxTokens));
            Qwen4QsaLayer.Weights w = loaded.weights();
            int maxRows = 512;
            long base = gpu.allocate(layer.scratchBytes(maxRows));
            Qwen4QsaLayer.Scratch s = layer.scratch(base, maxRows);
            long input = gpu.allocate((long) maxRows * 2560 * 2), output = gpu.allocate((long) maxRows * 2560 * 2);
            System.out.printf(
                    "scratch for %d rows: %.1f MiB (scores %.1f MiB); state of %d tokens: %.1f MiB; allocated %.1f MiB%n",
                    maxRows,
                    layer.scratchBytes(maxRows) / 1048576.0,
                    (s.ids() - s.scores()) / 1048576.0,
                    maxTokens,
                    (Qwen4QsaState.indexerBytes(maxTokens) + 576L * maxTokens) / 1048576.0,
                    gpu.allocatedBytes() / 1048576.0);
            try (Qwen4QsaState state = new Qwen4QsaState(gpu, 512, maxTokens)) {
                MemorySegment chunk = arena.allocate((long) maxRows * 2560 * 2, 16);
                for (int history : HISTORIES) {
                    while (state.length() < history) {
                        int step = Math.min(maxRows, history - state.length());
                        fillRandom(chunk, rng, step);
                        gpu.copyHostToDevice(input, chunk, (long) step * 2560 * 2);
                        layer.run(gpu, w, state, input, step, output, s, 0);
                        gpu.synchronize();
                        state.commit();
                    }
                    fillRandom(chunk, rng, maxRows);
                    for (int rows : ROWS) {
                        gpu.copyHostToDevice(input, chunk, (long) rows * 2560 * 2);
                        int start = state.length();
                        stream.submit(() -> layer.run(gpu, w, state, input, rows, output, s, 0), false);
                        gpu.synchronize();
                        // stages on the data of this run (all but the in-place norms leave it as it is)
                        int reps = rows == 1 ? 50 : 10;
                        double projections = timeMs(gpu, stream, reps, () -> {
                            layer.project(gpu, w, input, rows, s);
                        });
                        double append =
                                timeMs(gpu, stream, reps, () -> layer.appendKeyValues(gpu, state, rows, start, s));
                        double indexKeys =
                                timeMs(gpu, stream, reps, () -> layer.indexKeys(gpu, w, state, rows, start, s));
                        double select = timeMs(gpu, stream, reps, () -> layer.select(gpu, state, rows, start, s));
                        boolean selecting = layer.selects(rows, start);
                        double attend =
                                timeMs(gpu, stream, reps, () -> layer.attend(gpu, state, rows, start, s, selecting, 0));
                        double oProj = timeMs(
                                gpu,
                                stream,
                                reps,
                                () -> gpu.linearNvfp4Bf16(
                                        s.gated(), w.oProj(), output, rows, 6144, 2560, w.oProjBytes()));
                        state.discard();
                        // whole block: reserve, run, retire, discard (so every repetition sees this history)
                        int whole = rows == 1 ? 30 : 8;
                        double total = 0;
                        for (int i = 0; i < whole + 2; i++) {
                            long begin = System.nanoTime();
                            stream.submit(() -> layer.run(gpu, w, state, input, rows, output, s, 0), false);
                            gpu.synchronize();
                            if (i >= 2) total += System.nanoTime() - begin;
                            state.discard();
                        }
                        System.out.printf(
                                "history %6d rows %3d: block %7.3f ms | qkv+index proj %6.3f append %6.3f index keys %6.3f"
                                        + " select %7.3f attention %7.3f o_proj %6.3f%n",
                                history,
                                rows,
                                total / whole / 1e6,
                                projections,
                                append,
                                indexKeys,
                                select,
                                attend,
                                oProj);
                    }
                }
                System.out.printf("peak allocated %.1f MiB%n", gpu.peakAllocatedBytes() / 1048576.0);
            } finally {
                gpu.free(output);
                gpu.free(input);
                gpu.free(base);
            }
        }
    }
}
