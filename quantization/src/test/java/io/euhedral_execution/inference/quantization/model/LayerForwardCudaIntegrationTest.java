package io.euhedral_execution.inference.quantization.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Layers 0 (GDN) and 3 (attention) on the engine's kernels against the BF16 forward of a PyTorch reference of the
/// calibration (tools/gptq, last at commit 9ff02525) on the same input (frozen fixture, by default in
/// /mnt/shared/qwen38-quant/quality/fixtures/forward-fixture; `euhedral.quantization.forward-fixture` names another).
/// Each layer's update of the hidden states is compared. The engine keeps attention's keys and values in its NVFP4
/// cache, so layer 3 is held to the reference with the same cache codec, and its distance from plain BF16 is reported.
class LayerForwardCudaIntegrationTest {

    @Test
    void layersMatchThePythonCalibrationModel() throws Exception {
        String checkpoint = System.getProperty("euhedral.quantization.checkpoint", "");
        Path fixture = Path.of(System.getProperty(
                "euhedral.quantization.forward-fixture", "/mnt/shared/qwen38-quant/quality/fixtures/forward-fixture"));
        assumeTrue(!checkpoint.isBlank() && Files.isDirectory(fixture), "needs the checkpoint and the forward fixture");
        float[] input = bf16(fixture.resolve("input.npy"));
        int sequences = 2, length = 256, hidden = 5120, rows = sequences * length;
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (Safetensors store = Safetensors.open(Path.of(checkpoint));
                Device device = new Device(library);
                Arena arena = Arena.ofConfined()) {
            var objects = new Qwen38Objects(store, Path.of(checkpoint));
            try (LayerForward forward = new LayerForward(device, objects.config, rows)) {
                long states = device.gpu.allocate(2L * rows * hidden);
                MemorySegment host = arena.allocate(2L * rows * hidden);
                for (int index : new int[] {0, 3}) {
                    MemorySegment.copy(raw(fixture.resolve("input.npy")), 0, host, 0, host.byteSize());
                    device.gpu.copyHostToDevice(states, host, host.byteSize());
                    List<String> observed = new ArrayList<>();
                    try (LayerWeights layer = LayerWeights.layer(device.gpu, objects, index)) {
                        forward.forward(
                                layer, states, sequences, length, (name, at, count, columns) -> observed.add(name));
                        device.synchronize();
                    }
                    device.gpu.copyDeviceToHost(host, states, host.byteSize());
                    double plain =
                            difference(host, input, bf16(fixture.resolve("layer" + index + ".npy")), rows * hidden);
                    double relative = index == 0
                            ? plain
                            : difference(host, input, bf16(fixture.resolve("layer3-kv.npy")), rows * hidden);
                    System.out.printf(
                            "layer %d: relative difference of the update %.4f (from plain BF16 %.4f), observed %s%n",
                            index, relative, plain, observed);
                    assertEquals(objects.layerObjects(index), observed);
                    // The engine's attention multiplies probabilities and values in FP16 (1.85% on this fixture).
                    assertTrue(relative < (index == 0 ? 0.01 : 0.03), "layer " + index + ": " + relative);
                }
                device.gpu.free(states);
            }
        }
    }

    /// |actual update - expected update| / |expected update| over the hidden states.
    private static double difference(MemorySegment actual, float[] input, float[] expected, int count) {
        double difference = 0, norm = 0;
        for (int i = 0; i < count; i++) {
            float value = Float.intBitsToFloat(actual.getAtIndex(ValueLayout.JAVA_SHORT, i) << 16);
            double reference = expected[i] - input[i];
            difference += Math.pow((value - input[i]) - reference, 2);
            norm += reference * reference;
        }
        return Math.sqrt(difference / norm);
    }

    private static MemorySegment raw(Path npy) throws Exception {
        byte[] bytes = Files.readAllBytes(npy);
        int header = Short.toUnsignedInt(
                ByteBuffer.wrap(bytes, 8, 2).order(ByteOrder.LITTLE_ENDIAN).getShort());
        return MemorySegment.ofArray(java.util.Arrays.copyOfRange(bytes, 10 + header, bytes.length));
    }

    private static float[] bf16(Path npy) throws Exception {
        MemorySegment data = raw(npy);
        float[] out = new float[(int) (data.byteSize() / 2)];
        for (int i = 0; i < out.length; i++)
            out[i] = Float.intBitsToFloat(data.getAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i) << 16);
        return out;
    }
}
