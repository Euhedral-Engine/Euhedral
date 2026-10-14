package io.euhedral_execution.inference.quantization.training;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/// A whole layer's backward pass against its forward pass, at small dimensions, for both kinds: along a random
/// direction v of each parameter group (and of the input), the analytic derivative grad . v equals the central
/// difference of the squared-error loss (computed in double on the device).
class TrainableLayerCudaIntegrationTest {

    private static Device device;
    private static Ops ops;
    private static final List<Long> allocations = new ArrayList<>();
    private final Random random = new Random(5);

    @BeforeAll
    static void start() {
        device = new Device(Path.of(System.getProperty("euhedral.cuda.library")));
        ops = new Ops(device.cuda, device.stream, device.cublas);
    }

    @AfterAll
    static void stop() {
        for (long address : allocations) device.gpu.free(address);
        device.close();
    }

    private long upload(float[] values) {
        long address = device.gpu.allocate(4L * values.length);
        allocations.add(address);
        Upload.floats(device.gpu, address, values);
        return address;
    }

    private float[] random(int count, double scale) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float) (random.nextGaussian() * scale);
        return values;
    }

    private static float[] download(long address, long count) {
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(4 * count, 4);
            device.gpu.copyDeviceToHost(host, address, host.byteSize());
            return host.toArray(ValueLayout.JAVA_FLOAT);
        }
    }

    private void addLinear(TrainableLayer layer, String role, int out, int in, int group) {
        byte[] codes = new byte[out * in];
        for (int i = 0; i < codes.length; i++) codes[i] = (byte) (random.nextInt(8) - 4);
        long device = TrainableLayerCudaIntegrationTest.device.gpu.allocate(codes.length);
        allocations.add(device);
        Upload.bytes(TrainableLayerCudaIntegrationTest.device.gpu, device, codes);
        float[] scales = new float[out * (in / group)];
        for (int i = 0; i < scales.length; i++) scales[i] = (float) ((0.5 + random.nextDouble()) * 0.4 / Math.sqrt(in));
        layer.addLinear(role, out, in, group, device, upload(scales));
    }

    private TrainableLayer layer(boolean attention) {
        var shape = new TrainableLayer.Shape(attention, 64, 96, 2, 4, 32, 32, 4, 4, 2, 32, 16, 1e7, 1e-6f);
        var layer = new TrainableLayer(device.gpu, ops, shape);
        int h = shape.hidden(), group = 32;
        if (attention) {
            addLinear(layer, "attention/query_key", shape.qkWidth(), h, group);
            addLinear(layer, "attention/gate_value", shape.qkWidth(), h, group);
            addLinear(layer, "attention/output", h, shape.mixerWidth(), group);
            layer.addNorm("query_norm", upload(random(shape.headDim(), 0.2)), shape.headDim());
            layer.addNorm("key_norm", upload(random(shape.headDim(), 0.2)), shape.headDim());
        } else {
            addLinear(layer, "gdn/query_key", shape.qkWidth(), h, group);
            addLinear(layer, "gdn/value_z", 2 * shape.valueWidth(), h, group);
            addLinear(layer, "gdn/output", h, shape.valueWidth(), group);
            layer.addNorm("gdn_norm", upload(random(shape.valueDim(), 0.2)), shape.valueDim());
            layer.addFrozen("a_projection", upload(random(shape.valueHeads() * h, 0.1)), (long) shape.valueHeads() * h);
            layer.addFrozen("b_projection", upload(random(shape.valueHeads() * h, 0.1)), (long) shape.valueHeads() * h);
            layer.addFrozen("a_log", upload(random(shape.valueHeads(), 0.3)), shape.valueHeads());
            layer.addFrozen("dt_bias", upload(random(shape.valueHeads(), 0.3)), shape.valueHeads());
            long conv = (long) shape.convKernel() * shape.convWidth();
            layer.addFrozen("convolution", upload(random((int) conv, 0.4)), conv);
        }
        addLinear(layer, "mlp/gate_up", 2 * shape.intermediate(), h, group);
        addLinear(layer, "mlp/down", h, shape.intermediate(), group);
        float[] ones = new float[h];
        for (int i = 0; i < h; i++) ones[i] = (float) (0.1 * random.nextGaussian());
        layer.addNorm("input_norm", upload(ones), h);
        layer.addNorm("post_norm", upload(random(h, 0.1)), h);
        return layer;
    }

    /// The loss sum (layer(x) - target)^2 in double.
    private double loss(TrainableLayer layer, long x, long y, long target, long count, long scalar, int rows) {
        layer.forward(x, y, rows, 100);
        ops.zero(scalar, 2);
        ops.squaredError(y, target, count, 1, 0, scalar);
        device.synchronize();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(8, 8);
            device.gpu.copyDeviceToHost(host, scalar, 8);
            return host.get(ValueLayout.JAVA_DOUBLE, 0);
        }
    }

    @Test
    void gdnLayer() {
        check(false);
    }

    @Test
    void attentionLayer() {
        check(true);
    }

    private void check(boolean attention) {
        int rows = 40;
        try (TrainableLayer layer = layer(attention)) {
            int h = layer.shape.hidden();
            long count = (long) rows * h;
            long x = upload(random((int) count, 1)), y = upload(new float[(int) count]);
            long target = upload(random((int) count, 1)), dy = upload(new float[(int) count]);
            long dx = upload(new float[(int) count]), scalar = upload(new float[2]);
            loss(layer, x, y, target, count, scalar, rows);
            ops.squaredError(y, target, count, 1, dy, scalar);
            ops.zero(dx, count);
            layer.backward(x, dy, dx, 100);
            List<String> failures = new ArrayList<>();
            for (TrainableLayer.Parameter parameter : layer.parameters())
                directional(
                        layer,
                        parameter.name,
                        parameter.values,
                        parameter.gradient,
                        parameter.count,
                        x,
                        y,
                        target,
                        count,
                        scalar,
                        rows,
                        failures);
            directional(layer, "input", x, dx, count, x, y, target, count, scalar, rows, failures);
            assertTrue(failures.isEmpty(), String.join("; ", failures));
        }
    }

    private void directional(
            TrainableLayer layer,
            String name,
            long values,
            long gradient,
            long count,
            long x,
            long y,
            long target,
            long outputs,
            long scalar,
            int rows,
            List<String> failures) {
        float[] original = download(values, count), grad = download(gradient, count);
        double scale = 0;
        for (float v : original) scale = Math.max(scale, Math.abs(v));
        float[] direction = random((int) count, 1);
        double analytic = 0;
        for (int i = 0; i < count; i++) analytic += grad[i] * direction[i];
        double epsilon = 1e-3 * Math.max(scale, 1e-2);
        float[] shifted = new float[(int) count];
        for (int i = 0; i < count; i++) shifted[i] = (float) (original[i] + epsilon * direction[i]);
        Upload.floats(device.gpu, values, shifted);
        double up = loss(layer, x, y, target, outputs, scalar, rows);
        for (int i = 0; i < count; i++) shifted[i] = (float) (original[i] - epsilon * direction[i]);
        Upload.floats(device.gpu, values, shifted);
        double down = loss(layer, x, y, target, outputs, scalar, rows);
        Upload.floats(device.gpu, values, original);
        double numeric = (up - down) / (2 * epsilon);
        double error = Math.abs(numeric - analytic) / Math.max(Math.abs(numeric), 1e-6);
        System.out.printf(
                "%s %s: analytic %.6g, numeric %.6g, relative error %.2e%n",
                layer.shape.attention() ? "attention" : "gdn", name, analytic, numeric, error);
        if (error > 2e-2) failures.add(name + " " + analytic + " vs " + numeric);
    }
}
