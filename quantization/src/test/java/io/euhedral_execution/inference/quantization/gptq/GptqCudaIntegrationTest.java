package io.euhedral_execution.inference.quantization.gptq;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/// The GPU GPTQ against tools/gptq on one object in every format, with and without act-order (fixture written by
/// quantization/src/test/python/make_gptq_fixture.py). The tensor scales match exactly; the codes come from a factor
/// cuSOLVER computes in its own order, so they are compared by agreement and by the output error under H.
class GptqCudaIntegrationTest {

    private static final float[] E2M1 = {0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f};

    @Test
    void roundsLikeToolsGptq() throws Exception {
        JsonNode fixture = new ObjectMapper().readTree(resource("cases.json"));
        int n = fixture.get("rows").asInt(), k = fixture.get("columns").asInt();
        float damp = (float) fixture.get("damp").asDouble();
        Npy weight = Npy.read(resource("weight_bf16.npy"));
        Npy hessian = Npy.read(resource("hessian.npy"));
        float[] h = new float[k * k];
        hessian.data().asFloatBuffer().get(h);
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda();
            Gptq gptq = new Gptq(cuda, gpu, 0L);
            long weightDevice = gpu.allocate(2L * n * k), hessianDevice = gpu.allocate(4L * k * k);
            Upload.segment(gpu, weightDevice, MemorySegment.ofBuffer(weight.data()), 2L * n * k);
            Upload.floats(gpu, hessianDevice, h);
            float[] w = new float[n * k];
            for (int i = 0; i < n * k; i++)
                w[i] = Float.intBitsToFloat(weight.data().getShort(2 * i) << 16);
            for (JsonNode c : fixture.get("cases")) {
                String name = c.get("name").asText();
                var format =
                        Gptq.Format.of(c.get("format").asText(), c.get("layout").asText(), null);
                var factor =
                        gptq.factor(hessianDevice, k, damp, c.get("act_order").asBoolean());
                var tensor = format.nvfp4() ? gptq.tensorScales(weightDevice, n, k, format, factor) : null;
                QuantState state = gptq.round(weightDevice, n, factor, format, tensor, arena, 0L);
                gpu.free(factor.upper());
                Npy codes = Npy.read(resource(name + "-codes.npy"));
                Npy scales = Npy.read(resource(name + "-scales.npy"));
                if (format.nvfp4()) {
                    assertEquals((float) c.get("global_scale").asDouble(), state.globalScale(), name);
                    if (format.sd4()) {
                        byte[] table = new byte[16];
                        for (int i = 0; i < 16; i++)
                            table[i] = (byte) c.get("table").get(i).asInt();
                        assertArrayEquals(table, state.table(), name + " table");
                    }
                }
                float[] expected = decode(
                        format, codes, scales, (float) c.get("global_scale").asDouble(), state.table(), n, k);
                float[] actual = decode(format, state, n, k);
                long agree = 0;
                for (int i = 0; i < n * k; i++) {
                    if (codes.data().get(i) == state.codes().get(ValueLayout.JAVA_BYTE, i)) agree++;
                }
                double pythonError = outputError(w, expected, h, n, k), nativeError = outputError(w, actual, h, n, k);
                double ratio = nativeError / pythonError;
                System.out.printf(
                        "%s: codes agree %.4f, output error %.6g (tools/gptq %.6g), ratio %.4f%n",
                        name, (double) agree / (n * k), nativeError, pythonError, ratio);
                assertTrue(Math.abs(ratio - 1) < 0.03, name + ": output error ratio " + ratio);
                assertTrue(agree >= 0.9 * n * k, name + ": codes agree on " + agree + " of " + n * k);
            }
        }
    }

    private static java.io.InputStream resource(String name) {
        return GptqCudaIntegrationTest.class.getResourceAsStream("/gptq-fixture/" + name);
    }

    /// Decoded weights of a stored rounding, as the kernels compute them.
    private static float[] decode(Gptq.Format format, Npy codes, Npy scales, float global, byte[] table, int n, int k) {
        float[] out = new float[n * k];
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < k; j++) {
                int code = codes.data().get(r * k + j);
                out[r * k + j] = value(format, code, scaleOf(format, scales.data(), r, j, k), global, table);
            }
        }
        return out;
    }

    private static float[] decode(Gptq.Format format, QuantState state, int n, int k) {
        float[] out = new float[n * k];
        var buffer = state.scales().asByteBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < k; j++) {
                int code = state.codes().get(ValueLayout.JAVA_BYTE, (long) r * k + j);
                out[r * k + j] =
                        value(format, code, scaleOf(format, buffer, r, j, k), state.globalScale(), state.table());
            }
        }
        return out;
    }

    private static int scaleOf(Gptq.Format format, java.nio.ByteBuffer scales, int r, int j, int k) {
        int groups = k / format.group();
        int g = j / format.group();
        return format.nvfp4()
                ? Byte.toUnsignedInt(scales.get(r * groups + g))
                : Short.toUnsignedInt(scales.getShort(2 * (r * groups + g)));
    }

    private static float value(Gptq.Format format, int code, int scale, float global, byte[] table) {
        if (!format.nvfp4()) return code * Float.float16ToFloat((short) scale);
        int e4m3 = table != null ? Byte.toUnsignedInt(table[scale]) : scale;
        float magnitude =
                e4m3 < 8 ? (e4m3 & 7) / 8f * 0.015625f : (1 + (e4m3 & 7) / 8f) * (float) Math.pow(2, (e4m3 >> 3) - 7);
        float decode = magnitude * global;
        return E2M1[code & 7] * ((code & 8) != 0 ? -1f : 1f) * decode;
    }

    /// sum over rows of (w - q) H (w - q)^T: the output error over the inputs H summarizes.
    private static double outputError(float[] w, float[] q, float[] h, int n, int k) {
        double total = 0;
        double[] d = new double[k];
        for (int r = 0; r < n; r++) {
            for (int j = 0; j < k; j++) d[j] = (double) w[r * k + j] - q[r * k + j];
            for (int a = 0; a < k; a++) {
                double row = 0;
                for (int b = 0; b < k; b++) row += h[a * k + b] * d[b];
                total += d[a] * row;
            }
        }
        return total;
    }
}
