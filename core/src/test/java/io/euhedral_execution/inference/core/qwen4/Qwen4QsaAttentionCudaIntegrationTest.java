package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.decodeCache;
import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.downloadShorts;
import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.open;
import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.random;
import static io.euhedral_execution.inference.core.qwen4.Qwen4QsaTestSupport.upload;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.r;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.sigmoid;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/// The sparse attention kernel and its split merge against a double-precision reference over the decoded
/// NVFP4
/// cache: the dense causal prefix (every block selected), random selections of 512 blocks, rows that drop
/// their own
/// block, and key splits including empty ones.
class Qwen4QsaAttentionCudaIntegrationTest {

    private static final int QUERY_HEADS = 24, KEY_HEADS = 2, GROUP = 12, ROW = QUERY_HEADS * 256;

    /// Runs one case and returns {relative RMS of core, largest difference of core in RMS units}.
    private static double[] run(
            CudaGpuMemory gpu,
            Arena arena,
            SplittableRandom rng,
            String name,
            int start,
            int rows,
            String selection,
            int splits,
            int warps) {
        int tokens = start + rows;
        try (Qwen4QsaState state = new Qwen4QsaState(gpu, KEY_HEADS * 256, tokens)) {
            // K and V rows of every position, appended in chunks of up to 512 rows.
            for (int begin = 0; begin < tokens; begin += 512) {
                int count = Math.min(512, tokens - begin);
                long k = upload(gpu, arena, random(rng, count * 512, 1.0));
                long v = upload(gpu, arena, random(rng, count * 512, 1.0));
                try {
                    int at = state.beginChunk(count);
                    Qwen4QsaOps.kvAppend(
                            gpu, k, v, state.keyPages(), state.valuePages(), count, KEY_HEADS, 512, 512, at);
                    state.submitted();
                    state.commit();
                } finally {
                    gpu.free(v);
                    gpu.free(k);
                }
            }
            double[][][] keys = new double[KEY_HEADS][][], values = new double[KEY_HEADS][][];
            for (int h = 0; h < KEY_HEADS; h++) {
                keys[h] = decodeCache(gpu, arena, state.keyPages(), tokens, KEY_HEADS, h);
                values[h] = decodeCache(gpu, arena, state.valuePages(), tokens, KEY_HEADS, h);
            }
            short[] queries = random(rng, rows * ROW, 1.0), gates = random(rng, rows * ROW, 1.0);
            int[][] ids = new int[rows][];
            long idAddress = 0, countAddress = 0;
            if (!selection.equals("dense")) {
                int[] flatIds = new int[rows * 512], counts = new int[rows];
                for (int r = 0; r < rows; r++) {
                    int p = start + r, nb = (p + 1) / 4;
                    int[] pick;
                    if (nb <= 512) {
                        pick = java.util.stream.IntStream.range(0, nb).toArray();
                    } else {
                        // a random set of 512 of the blocks; the "own" selection drops the row's own block
                        boolean dropOwn = selection.equals("drop-own") && (p + 1) % 4 == 0;
                        int[] pool = java.util.stream.IntStream.range(0, dropOwn ? nb - 1 : nb)
                                .toArray();
                        for (int i = pool.length - 1; i > 0; i--) {
                            int j = rng.nextInt(i + 1);
                            int t = pool[i];
                            pool[i] = pool[j];
                            pool[j] = t;
                        }
                        pick = Arrays.copyOf(pool, 512);
                        Arrays.sort(pick);
                    }
                    ids[r] = pick;
                    counts[r] = pick.length;
                    System.arraycopy(pick, 0, flatIds, r * 512, pick.length);
                }
                idAddress = upload(gpu, arena, flatIds);
                countAddress = upload(gpu, arena, counts);
            }
            long q = upload(gpu, arena, queries), gate = upload(gpu, arena, gates);
            long core = gpu.allocate((long) rows * ROW * 2), gated = gpu.allocate((long) rows * ROW * 2);
            long partial = gpu.allocate((long) rows * QUERY_HEADS * splits * 258 * 4);
            try {
                Qwen4QsaOps.attention(
                        gpu,
                        q,
                        gate,
                        state.keyPages(),
                        state.valuePages(),
                        idAddress,
                        countAddress,
                        splits > 1 ? partial : 0,
                        core,
                        gated,
                        rows,
                        QUERY_HEADS,
                        KEY_HEADS,
                        start,
                        splits,
                        512,
                        ROW,
                        256,
                        ROW,
                        256,
                        warps);
                if (splits > 1) Qwen4QsaOps.merge(gpu, partial, gate, core, gated, rows, QUERY_HEADS, splits, ROW, 256);
                short[] actualCore = downloadShorts(gpu, arena, core, (long) rows * ROW);
                short[] actualGated = downloadShorts(gpu, arena, gated, (long) rows * ROW);
                double[] expected = new double[rows * ROW];
                short[] expectedGated = new short[rows * ROW];
                for (int row = 0; row < rows; row++)
                    for (int head = 0; head < QUERY_HEADS; head++) {
                        int kh = head / GROUP;
                        double[] out = Qwen4QsaReference.attend(
                                queries,
                                row * ROW + head * 256,
                                keys[kh],
                                values[kh],
                                start + row,
                                ids[row],
                                ids[row] == null ? 0 : ids[row].length);
                        System.arraycopy(out, 0, expected, row * ROW + head * 256, 256);
                        for (int d = 0; d < 256; d++) {
                            int at = row * ROW + head * 256 + d;
                            float g = r(sigmoid(bf(gates[at])));
                            expectedGated[at] = bits(bf(actualCore[at]) * g);
                        }
                    }
                double rms = Qwen4QsaReference.relativeRms(expected, actualCore);
                double worst = 0, scale = 0;
                for (double e : expected) scale += e * e;
                scale = Math.sqrt(scale / expected.length);
                for (int i = 0; i < expected.length; i++)
                    worst = Math.max(worst, Math.abs(bf(actualCore[i]) - expected[i]) / scale);
                assertTrue(rms < 3e-3, name + ": core relative RMS " + rms);
                assertTrue(worst < 0.05, name + ": core worst difference " + worst + " RMS");
                // the gate is exactly bf16(core * bf16(sigmoid(gate)))
                for (int i = 0; i < expectedGated.length; i++)
                    assertTrue(
                            Math.abs(bf(expectedGated[i]) - bf(actualGated[i]))
                                    <= Math.abs(bf(expectedGated[i])) * 0.0079 + 1e-30,
                            name + ": gated element " + i);
                return new double[] {rms, worst};
            } finally {
                gpu.free(partial);
                gpu.free(gated);
                gpu.free(core);
                gpu.free(gate);
                gpu.free(q);
                if (idAddress != 0) {
                    gpu.free(countAddress);
                    gpu.free(idAddress);
                }
            }
        }
    }

    @Test
    void denseCausalAttentionMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(21);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int splits : new int[] {1, 8}) {
                double[] a = run(gpu, arena, rng, "5 rows at 32, splits " + splits, 32, 5, "dense", splits, 2);
                double[] b = run(gpu, arena, rng, "1 row at 0, splits " + splits, 0, 1, "dense", splits, 1);
                double[] c = run(gpu, arena, rng, "7 rows at 1500, splits " + splits, 1500, 7, "dense", splits, 2);
                double[] d = run(gpu, arena, rng, "decode at 2000, splits " + splits, 2000, 1, "dense", splits, 4);
                System.out.printf("dense splits %d: rms %.2e %.2e %.2e %.2e%n", splits, a[0], b[0], c[0], d[0]);
            }
        }
    }

    @Test
    void selectedAttentionMatchesTheReference() {
        SplittableRandom rng = new SplittableRandom(22);
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            for (int splits : new int[] {1, 8, 64}) {
                double[] a =
                        run(gpu, arena, rng, "selected 6 rows at 2600, splits " + splits, 2600, 6, "random", splits, 2);
                double[] b = run(
                        gpu, arena, rng, "drop-own 8 rows at 2600, splits " + splits, 2600, 8, "drop-own", splits, 2);
                double[] c = run(
                        gpu,
                        arena,
                        rng,
                        "crossing the budget at 2040, splits " + splits,
                        2040,
                        24,
                        "random",
                        splits,
                        4);
                System.out.printf("selected splits %d: rms %.2e %.2e %.2e%n", splits, a[0], b[0], c[0]);
            }
        }
    }
}
