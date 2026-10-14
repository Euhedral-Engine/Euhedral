package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/// `score.cu` against the arithmetic of tools/compare_reference.py in float64 on the host: random BF16 logits rows
/// against random 16-bit reference rows, a vocabulary that does not fill the last block of threads, and a tie for the
/// maximum on both sides.
class ScoreCudaIntegrationTest {

    @Test
    void rowsMatchTheHostArithmetic() {
        int rows = 5, vocabulary = 1001, width = 2 * ((vocabulary + 1) / 2) + 4;
        var random = new Random(7);
        short[] logits = new short[rows * vocabulary];
        short[] reference = new short[rows * width];
        int[] targets = new int[rows];
        for (int r = 0; r < rows; r++) {
            for (int i = 0; i < vocabulary; i++) {
                logits[r * vocabulary + i] = (short) (Float.floatToIntBits((float) (random.nextGaussian() * 3)) >>> 16);
            }
            float scale = 16f / 65535, minimum = (float) (-16 - random.nextDouble());
            int scaleBits = Float.floatToIntBits(scale), minimumBits = Float.floatToIntBits(minimum);
            reference[r * width] = (short) scaleBits;
            reference[r * width + 1] = (short) (scaleBits >>> 16);
            reference[r * width + 2] = (short) minimumBits;
            reference[r * width + 3] = (short) (minimumBits >>> 16);
            for (int i = 0; i < vocabulary; i++) reference[r * width + 4 + i] = (short) random.nextInt(65536);
            targets[r] = random.nextInt(vocabulary);
        }
        // Ties: two equal maxima on each side of row 0, the first of them the one counted.
        logits[17] = logits[900] = (short) (Float.floatToIntBits(40f) >>> 16);
        reference[4 + 900] = reference[4 + 17] = (short) 65535;
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (Device device = new Device(library);
                Arena arena = Arena.ofConfined()) {
            long logitsDevice = device.gpu.allocate(2L * logits.length);
            long referenceDevice = device.gpu.allocate(2L * reference.length);
            long targetsDevice = device.gpu.allocate(4L * rows);
            long out = device.gpu.allocate(8L * 5 * rows);
            Upload.segment(device.gpu, logitsDevice, copy(arena, logits), 2L * logits.length);
            Upload.segment(device.gpu, referenceDevice, copy(arena, reference), 2L * reference.length);
            Upload.ints(device.gpu, targetsDevice, targets);
            var kernel = device.cuda
                    .compileResource("score.cu", List.of("score_rows"))
                    .getFirst();
            device.cuda.launch(
                    kernel,
                    rows,
                    1,
                    256,
                    0,
                    device.stream,
                    new Cuda.Arguments()
                            .pointer(logitsDevice)
                            .pointer(referenceDevice)
                            .integer(width)
                            .integer(vocabulary)
                            .pointer(targetsDevice)
                            .pointer(out));
            device.synchronize();
            MemorySegment host = arena.allocate(8L * 5 * rows, 8);
            device.gpu.copyDeviceToHost(host, out, host.byteSize());
            for (int r = 0; r < rows; r++) {
                double[] expected = expected(logits, reference, r, vocabulary, width, targets[r]);
                for (int k = 0; k < 5; k++) {
                    double actual = host.getAtIndex(ValueLayout.JAVA_DOUBLE, 5L * r + k);
                    assertEquals(expected[k], actual, 1e-9 * Math.max(1, Math.abs(expected[k])), "row " + r + " " + k);
                }
            }
            for (long address : new long[] {logitsDevice, referenceDevice, targetsDevice, out}) {
                device.gpu.free(address);
            }
        }
    }

    private static MemorySegment copy(Arena arena, short[] values) {
        MemorySegment segment = arena.allocate(2L * values.length, 2);
        MemorySegment.copy(MemorySegment.ofArray(values), 0, segment, 0, segment.byteSize());
        return segment;
    }

    /// nll, reference nll, KL over the entries above the -16 floor, top-1 agreement, delta-p.
    private static double[] expected(short[] logits, short[] reference, int r, int vocabulary, int width, int target) {
        double[] q = new double[vocabulary], p = new double[vocabulary];
        int base = r * width;
        float scale = Float.intBitsToFloat((reference[base] & 0xffff) | (reference[base + 1] & 0xffff) << 16);
        float minimum = Float.intBitsToFloat((reference[base + 2] & 0xffff) | (reference[base + 3] & 0xffff) << 16);
        double max = Double.NEGATIVE_INFINITY;
        int qArg = 0, pArg = 0;
        for (int i = 0; i < vocabulary; i++) {
            q[i] = Float.intBitsToFloat(logits[r * vocabulary + i] << 16);
            p[i] = (double) scale * (reference[base + 4 + i] & 0xffff) + minimum;
            if (q[i] > q[qArg]) qArg = i;
            if (p[i] > p[pArg]) pArg = i;
            max = Math.max(max, q[i]);
        }
        double sum = 0;
        for (double value : q) sum += Math.exp(value - max);
        double logZ = max + Math.log(sum), kl = 0;
        for (int i = 0; i < vocabulary; i++) {
            if (p[i] > -16) kl += Math.exp(p[i]) * (p[i] - (q[i] - logZ));
        }
        double qTarget = q[target] - logZ;
        return new double[] {-qTarget, -p[target], kl, qArg == pArg ? 1 : 0, Math.exp(qTarget) - Math.exp(p[target])};
    }
}
