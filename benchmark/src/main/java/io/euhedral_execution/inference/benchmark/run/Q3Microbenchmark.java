package io.euhedral_execution.inference.benchmark.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.Q3DispatchMode;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactDenseFfnWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Focused packed-weight operator screen, not an end-to-end generation measurement.
/// Uses real artifact matrices and deterministic BF16 activations; all paths use the same operands.
public final class Q3Microbenchmark {
    private Q3Microbenchmark() {}

    public static void run(BenchmarkOptions options, String selectedMatrix, Integer selectedRows) throws Exception {
        if (options.output().toString().endsWith(".json"))
            throw new IllegalArgumentException("Q3 screens require a JSONL output path, not .json");
        if (Files.exists(options.output())) throw new IllegalArgumentException("Q3 output already exists");
        if (selectedRows != null && selectedRows <= 0) throw new IllegalArgumentException("rows must be positive");
        if (selectedMatrix != null
                && !List.of("mixer-output", "mlp-gate-up", "mlp-down", "vocabulary")
                        .contains(selectedMatrix)) throw new IllegalArgumentException("unknown Q3 matrix");
        String capacityFailure = GpuCapacity.check(options.cudaLibrary(), options.artifact(), options.gpuHeadroomMiB());
        if (capacityFailure != null) throw new IllegalArgumentException(capacityFailure);
        Files.createDirectories(options.output().toAbsolutePath().getParent());
        var artifact = QwenArtifactReader.read(options.artifact());
        try (var gpu = new CudaGpuMemory(options.cudaLibrary());
                var model = QwenModel.load(options.artifact(), artifact, gpu);
                var writer = Files.newBufferedWriter(options.output(), StandardOpenOption.CREATE_NEW)) {
            var layer = model.weights().layers()[0];
            var mixer = (QwenCompactGatedDeltaNetWeights) layer.mixer();
            var ffn = (QwenCompactDenseFfnWeights) layer.ffn();
            Map<String, TensorHandle> matrices = new LinkedHashMap<>();
            matrices.put("mixer-output", mixer.output());
            matrices.put("mlp-gate-up", ffn.gateUp());
            matrices.put("mlp-down", ffn.down());
            matrices.put("vocabulary", model.weights().lmHead());
            var json = new ObjectMapper();
            for (var entry : matrices.entrySet()) {
                if (selectedMatrix != null && !selectedMatrix.equals(entry.getKey())) continue;
                TensorHandle weight = entry.getValue();
                int k = Math.toIntExact(weight.shape()[1]), n = Math.toIntExact(weight.shape()[0]);
                for (int rows :
                        selectedRows == null ? new int[] {1, 2, 4, 8, 16, 32, 256, 512} : new int[] {selectedRows}) {
                    // Full vocabulary rows are no longer a generation workload. Keep its screen bounded.
                    if (selectedRows == null && entry.getKey().equals("vocabulary") && rows > 32) continue;
                    try (Arena arena = Arena.ofConfined()) {
                        var hostInput = arena.allocate((long) rows * k * Short.BYTES);
                        for (int i = 0; i < rows * k; i++) {
                            float value = (float) (Math.sin(i * 0.017) * 0.7 + Math.cos(i * 0.031) * 0.3);
                            int bits = Float.floatToRawIntBits(value);
                            hostInput.setAtIndex(
                                    ValueLayout.JAVA_SHORT, i, (short) ((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16));
                        }
                        long x = gpu.allocate(hostInput.byteSize());
                        long y = 0;
                        long eviction = 0;
                        try {
                            y = gpu.allocate((long) rows * n * Short.BYTES);
                            eviction = gpu.allocate(128L * 1024 * 1024);
                            gpu.copyHostToDevice(x, hostInput, hostInput.byteSize());
                            gpu.linearQ3Bf16(
                                    x,
                                    weight.deviceAddress(),
                                    y,
                                    rows,
                                    k,
                                    n,
                                    weight.byteSize(),
                                    Q3DispatchMode.SCALAR,
                                    weight.layout());
                            var reference = arena.allocate((long) rows * n * Short.BYTES);
                            gpu.copyDeviceToHost(reference, y, reference.byteSize());
                            var actual = arena.allocate(reference.byteSize());
                            for (var mode :
                                    List.of(Q3DispatchMode.SCALAR, Q3DispatchMode.DECODE, Q3DispatchMode.PREFILL)) {
                                for (int warmup = 0; warmup < options.warmup(); warmup++)
                                    gpu.linearQ3Bf16(
                                            x,
                                            weight.deviceAddress(),
                                            y,
                                            rows,
                                            k,
                                            n,
                                            weight.byteSize(),
                                            mode,
                                            weight.layout());
                                long[] samples = new long[options.iterations()];
                                for (int iteration = 0; iteration < samples.length; iteration++) {
                                    // Larger than this target GPU's L2; excluded from the timed boundary.
                                    gpu.zeroDeviceMemory(eviction, 128L * 1024 * 1024);
                                    long started = System.nanoTime();
                                    gpu.linearQ3Bf16(
                                            x,
                                            weight.deviceAddress(),
                                            y,
                                            rows,
                                            k,
                                            n,
                                            weight.byteSize(),
                                            mode,
                                            weight.layout());
                                    samples[iteration] = System.nanoTime() - started;
                                }
                                gpu.copyDeviceToHost(actual, y, actual.byteSize());
                                var record = new LinkedHashMap<String, Object>();
                                record.put("schema", "euhedral-inference.q3-microbenchmark");
                                record.put("artifact", options.artifact().toString());
                                record.put("matrix", entry.getKey());
                                record.put("weight", weight.name());
                                record.put("rows", rows);
                                record.put("k", k);
                                record.put("n", n);
                                record.put("mode", mode);
                                record.put("input", "deterministic-sin-cos-bf16");
                                record.put("cachePolicy", "128MiB-device-clear-before-each-sample");
                                record.put("boundary", "synchronous-native-call-including-launch");
                                record.put("samplesNanos", samples);
                                var errors = errors(reference, actual);
                                boolean valid = accepts(mode, errors);
                                record.put("status", valid ? "success" : "failed");
                                record.put("error", errors);
                                writer.write(json.writeValueAsString(record));
                                writer.newLine();
                                writer.flush();
                                if (!valid)
                                    throw new IllegalStateException("Q3 numerical gate failed: " + entry.getKey()
                                            + " rows=" + rows + " " + mode + " " + errors);
                                System.out.println(
                                        entry.getKey() + " rows=" + rows + " " + mode + " " + Arrays.toString(samples));
                            }
                        } finally {
                            if (eviction != 0) gpu.free(eviction);
                            if (y != 0) gpu.free(y);
                            gpu.free(x);
                        }
                    }
                }
            }
        }
    }

    static Map<String, Object> errors(MemorySegment expected, MemorySegment actual) {
        long count = expected.byteSize() / Short.BYTES, different = 0, outsideRounding = 0;
        double squareSum = 0, absoluteSum = 0, max = 0;
        long[] absoluteBins = new long[6];
        for (long i = 0; i < count; i++) {
            short a = expected.getAtIndex(ValueLayout.JAVA_SHORT, i), b = actual.getAtIndex(ValueLayout.JAVA_SHORT, i);
            double error =
                    Math.abs(Float.intBitsToFloat((a & 0xffff) << 16) - Float.intBitsToFloat((b & 0xffff) << 16));
            if (!Double.isFinite(error)) throw new IllegalStateException("nonfinite Q3 result");
            if (a != b) different++;
            boolean adjacent = (a < 0) == (b < 0) && Math.abs((a & 0xffff) - (b & 0xffff)) <= 1;
            if (error > 0.001 && !adjacent) outsideRounding++;
            squareSum += error * error;
            absoluteSum += error;
            max = Math.max(max, error);
            absoluteBins[
                    error == 0 ? 0 : error <= 0.001 ? 1 : error <= 0.01 ? 2 : error <= 0.1 ? 3 : error <= 1 ? 4 : 5]++;
        }
        return Map.of(
                "outsideBf16Rounding",
                outsideRounding,
                "count",
                count,
                "different",
                different,
                "maxAbsolute",
                max,
                "meanAbsolute",
                absoluteSum / count,
                "rmse",
                Math.sqrt(squareSum / count),
                "absoluteBins",
                absoluteBins,
                "binUpperBounds",
                List.of("0", "0.001", "0.01", "0.1", "1", "infinity"));
    }

    static boolean accepts(Q3DispatchMode mode, Map<String, Object> errors) {
        // Decode preserves reference order; WMMA changes FP32 summation order.
        return (long) errors.get(mode == Q3DispatchMode.PREFILL ? "outsideBf16Rounding" : "different") == 0;
    }
}
