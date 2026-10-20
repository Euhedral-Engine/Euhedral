package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gptq.Npy;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/// exllamav3's own quantized linear (trellis, suh, svh) reconstructed by the module equals exllamav3's loader
/// reconstruction, up to BF16 rounding of the result.
class ReconstructCudaIntegrationTest {

    @Test
    void reconstructsLikeExllamav3() throws Exception {
        JsonNode meta;
        try (InputStream input = getClass().getResourceAsStream("/trellis-fixture/linear.json")) {
            meta = new ObjectMapper().readTree(input);
        }
        int k = meta.get("k").asInt(), n = meta.get("n").asInt();
        double bits = meta.get("bits").asDouble();
        Npy trellis = TrellisKernelsCudaIntegrationTest.read("linear-trellis.npy");
        Npy suh = TrellisKernelsCudaIntegrationTest.read("linear-suh.npy");
        Npy svh = TrellisKernelsCudaIntegrationTest.read("linear-svh.npy");
        short[] expected =
                TrellisKernelsCudaIntegrationTest.shorts(TrellisKernelsCudaIntegrationTest.read("linear-loaded.npy"));
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda(library);
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            long wordBytes = trellis.data().remaining();
            long words = gpu.allocate(wordBytes), su = gpu.allocate(2L * k), sv = gpu.allocate(2L * n);
            long out = gpu.allocate(2L * n * k), scratch = gpu.allocate(4L * k * n);
            Upload.segment(gpu, words, MemorySegment.ofBuffer(trellis.data()), wordBytes);
            Upload.segment(gpu, su, MemorySegment.ofBuffer(suh.data()), 2L * k);
            Upload.segment(gpu, sv, MemorySegment.ofBuffer(svh.data()), 2L * n);
            kernels.reconstruct(words, su, sv, k, n, bits, out, scratch);
            cuda.synchronize(0L);
            short[] actual = TrellisKernelsCudaIntegrationTest.download(gpu, arena, out, n * k);
            double error = 0, norm = 0, maxError = 0, maxValue = 0;
            for (int i = 0; i < actual.length; i++) {
                float a = Float.intBitsToFloat(actual[i] << 16), e = Float.float16ToFloat(expected[i]);
                error += (a - e) * (a - e);
                norm += e * e;
                maxError = Math.max(maxError, Math.abs(a - e));
                maxValue = Math.max(maxValue, Math.abs(e));
            }
            assertEquals(0.0, Math.sqrt(error / norm), 4e-3, "relative RMS against exllamav3's reconstruction");
            // One BF16 ulp at the largest magnitude (8 mantissa bits: 2^-7 relative to the binade's base), plus the
            // reference's own fp16 rounding at each of its four half-precision steps.
            double bf16Ulp = Math.scalb(1.0, Math.getExponent(maxValue) - 7);
            System.out.println(
                    "reconstruct: max |error| " + maxError + ", max |value| " + maxValue + ", bf16 ulp " + bf16Ulp);
            assertTrue(maxError <= bf16Ulp, "max abs error " + maxError + " above one BF16 ulp " + bf16Ulp);
            for (long a : new long[] {words, su, sv, out, scratch}) gpu.free(a);
        }
    }

    @Test
    void hadamardIsItsOwnInverse() throws Exception {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda(library);
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            int rows = 256, cols = 384;
            float[] x = new float[rows * cols];
            var random = new java.util.SplittableRandom(3);
            for (int i = 0; i < x.length; i++) x[i] = (float) random.nextGaussian();
            long d = gpu.allocate(4L * x.length);
            Upload.floats(gpu, d, x);
            kernels.hadamardRows(d, rows, cols);
            kernels.hadamardColumns(d, rows, cols);
            kernels.hadamardColumns(d, rows, cols);
            kernels.hadamardRows(d, rows, cols);
            cuda.synchronize(0L);
            float[] back = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, d, x.length);
            for (int i = 0; i < x.length; i++) assertEquals(x[i], back[i], 1e-5f, "element " + i);
            gpu.free(d);
        }
    }

    /// A single block transform against the explicit Sylvester matrix, rows and columns.
    @Test
    void hadamardMatchesSylvesterMatrix() throws Exception {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda(library);
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            int rows = 128, cols = 128;
            float[] x = new float[rows * cols];
            var random = new java.util.SplittableRandom(5);
            for (int i = 0; i < x.length; i++) x[i] = (float) random.nextGaussian();
            long d = gpu.allocate(4L * x.length);
            Upload.floats(gpu, d, x);
            kernels.hadamardRows(d, rows, cols);
            cuda.synchronize(0L);
            float[] byRows = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, d, x.length);
            Upload.floats(gpu, d, x);
            kernels.hadamardColumns(d, rows, cols);
            cuda.synchronize(0L);
            float[] byColumns = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, d, x.length);
            double scale = 1 / Math.sqrt(128);
            for (int i = 0; i < 128; i++)
                for (int c = 0; c < 128; c++) {
                    double rowSum = 0, columnSum = 0;
                    for (int j = 0; j < 128; j++) {
                        double h = Integer.bitCount(i & j) % 2 == 0 ? scale : -scale;
                        rowSum += h * x[j * cols + c];
                        columnSum += h * x[c * cols + j];
                    }
                    assertEquals(rowSum, byRows[i * cols + c], 1e-4, "rows " + i + "," + c);
                    assertEquals(columnSum, byColumns[c * cols + i], 1e-4, "columns " + c + "," + i);
                }
            gpu.free(d);
        }
    }

    /// toTiles and fromTiles are inverse permutations of a 16-row strip, in the tensor-core order.
    @Test
    void stripTilesRoundTrip() throws Exception {
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda(library);
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            int k = 48, n = 64, first = 16;
            float[] w = new float[k * n];
            for (int i = 0; i < w.length; i++) w[i] = i;
            long dw = gpu.allocate(4L * w.length), dt = gpu.allocate(4L * (n / 16) * 256);
            Upload.floats(gpu, dw, w);
            kernels.toTiles(dw, k, n, first, 16, dt);
            cuda.synchronize(0L);
            float[] tiles = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, dt, (n / 16) * 256);
            for (int tile = 0; tile < n / 16; tile++)
                for (int pos = 0; pos < 256; pos++) {
                    int e = TrellisKernels.TENSOR_CORE_ORDER[pos];
                    assertEquals((first + e / 16) * n + tile * 16 + e % 16, tiles[tile * 256 + pos], 0f);
                }
            Upload.floats(gpu, dw, new float[w.length]);
            kernels.fromTiles(dt, k, n, first, dw);
            cuda.synchronize(0L);
            float[] back = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, dw, w.length);
            for (int i = 0; i < w.length; i++)
                assertEquals(i / n >= first && i / n < first + 16 ? w[i] : 0f, back[i], 0f);
            gpu.free(dw);
            gpu.free(dt);
        }
    }
}
