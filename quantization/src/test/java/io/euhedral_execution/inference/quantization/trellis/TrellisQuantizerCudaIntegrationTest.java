package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

/// One linear quantized by the module against exllamav3's quantize_exl3 on the same weights and Hessian (frozen
/// fixture): the proxy error tr(E H Eᵀ) / tr(W H Wᵀ) within 10% of exllamav3's (different random signs), and the
/// returned tensors reconstruct to the returned decoded weights.
class TrellisQuantizerCudaIntegrationTest {

    @Test
    void proxyErrorMatchesExllamav3() throws Exception {
        var meta = new ObjectMapper().readTree(metadata());
        int k = meta.get("k").asInt(), n = meta.get("n").asInt();
        try (Fixture f = new Fixture(k, n)) {
            var result = f.quantizer.quantize(f.weight, n, k, f.hessian, 4096, 3.0, 0, true);
            double expected = meta.get("proxy_err").asDouble();
            System.out.println("proxy error " + result.proxyError() + " against exllamav3's " + expected
                    + ", global scale " + result.globalScale() + " against "
                    + meta.get("g_scale").asDouble());
            assertEquals(expected, result.proxyError(), 0.10 * expected, "proxy error");
            f.gpu.free(result.decoded());
        }
    }

    @Test
    void degenerateHessianFallsBack() throws Exception {
        try (Fixture f = new Fixture(1024, 512)) {
            f.gpu.zeroDeviceMemory(f.hessian, 4L * 1024 * 1024);
            var result = f.quantizer.quantize(f.weight, 512, 1024, f.hessian, 0, 3.0, 0, true);
            assertTrue(Double.isFinite(result.proxyError()) || Double.isNaN(result.proxyError()));
            for (short s : result.suh()) assertTrue(Float.isFinite(Float.float16ToFloat(s)), "finite suh");
            f.gpu.free(result.decoded());
        }
    }

    @Test
    void illConditionedHessianRetries() throws Exception {
        try (Fixture f = new Fixture(1024, 512)) {
            // rank-deficient H: the fixture's H with its first 512 rows and columns zeroed
            f.zeroLeading(512);
            var result = f.quantizer.quantize(f.weight, 512, 1024, f.hessian, 4096, 3.0, 0, true);
            assertTrue(Double.isFinite(result.proxyError()), "quantized after damping retries");
            f.gpu.free(result.decoded());
        }
    }

    @Test
    void indefiniteHessianRetriesWithMoreDamping() throws Exception {
        try (Fixture f = new Fixture(1024, 512)) {
            f.subtractFromDiagonal(0, 12.0f);
            var result = f.quantizer.quantize(f.weight, 512, 1024, f.hessian, 4096, 3.0, 0, true);
            assertTrue(Double.isFinite(result.proxyError()), "quantized after damping retries");
            f.gpu.free(result.decoded());
        }
    }

    @Test
    void tensorsReconstructToDecoded() throws Exception {
        try (Fixture f = new Fixture(1024, 512)) {
            var result = f.quantizer.quantize(f.weight, 512, 1024, f.hessian, 4096, 2.5, 7, true);
            short[] decoded = f.download(result.decoded(), 512 * 1024);
            short[] rebuilt = f.reconstruct(result, 2.5);
            for (int i = 0; i < decoded.length; i++) assertEquals(decoded[i], rebuilt[i], "element " + i);
            f.gpu.free(result.decoded());
        }
    }

    private static byte[] metadata() throws Exception {
        try (InputStream input =
                TrellisQuantizerCudaIntegrationTest.class.getResourceAsStream("/trellis-fixture/linear.json")) {
            return input.readAllBytes();
        }
    }

    /// The fixture's weights and Hessian on the device, a quantizer on the default stream.
    static final class Fixture implements AutoCloseable {
        final CudaGpuMemory gpu;
        final Cuda cuda = new Cuda();
        final TrellisQuantizer quantizer;
        final long weight, hessian;
        final int k, n;
        final Arena arena = Arena.ofConfined();

        Fixture(int k, int n) throws Exception {
            this.k = k;
            this.n = n;
            gpu = new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
            quantizer = new TrellisQuantizer(cuda, gpu, 0L);
            Npy w = TrellisKernelsCudaIntegrationTest.read("linear-w.npy");
            Npy h = TrellisKernelsCudaIntegrationTest.read("linear-h.npy");
            weight = gpu.allocate(2L * n * k);
            short[] half = TrellisKernelsCudaIntegrationTest.shorts(w);
            short[] bf16 = new short[half.length];
            for (int i = 0; i < half.length; i++)
                bf16[i] = (short) (Float.floatToIntBits(Float.float16ToFloat(half[i])) >>> 16);
            Upload.shorts(gpu, weight, bf16);
            hessian = gpu.allocate(4L * k * k);
            Upload.segment(gpu, hessian, MemorySegment.ofBuffer(h.data()), 4L * k * k);
        }

        void zeroLeading(int count) {
            float[] h = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, hessian, k * k);
            for (int i = 0; i < k; i++) for (int j = 0; j < k; j++) if (i < count || j < count) h[i * k + j] = 0f;
            Upload.floats(gpu, hessian, h);
        }

        /// Lowers one diagonal entry by `share` times the mean diagonal, making H indefinite.
        void subtractFromDiagonal(int index, float share) {
            float[] h = TrellisKernelsCudaIntegrationTest.downloadFloats(gpu, arena, hessian, k * k);
            double mean = 0;
            for (int i = 0; i < k; i++) mean += h[i * k + i] / k;
            h[index * k + index] -= (float) (share * mean);
            Upload.floats(gpu, hessian, h);
        }

        short[] download(long address, int count) {
            return TrellisKernelsCudaIntegrationTest.download(gpu, arena, address, count);
        }

        short[] reconstruct(TrellisQuantizer.Result r, double bits) {
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            long words = gpu.allocate(2L * r.trellis().length), su = gpu.allocate(2L * k), sv = gpu.allocate(2L * n);
            long out = gpu.allocate(2L * n * k), scratch = gpu.allocate(4L * k * n);
            Upload.shorts(gpu, words, r.trellis());
            Upload.shorts(gpu, su, r.suh());
            Upload.shorts(gpu, sv, r.svh());
            kernels.reconstruct(words, su, sv, k, n, bits, out, scratch);
            cuda.synchronize(0L);
            short[] values = download(out, n * k);
            for (long a : new long[] {words, su, sv, out, scratch}) gpu.free(a);
            return values;
        }

        @Override
        public void close() {
            quantizer.close();
            gpu.free(weight);
            gpu.free(hessian);
            arena.close();
            gpu.close();
        }
    }
}
