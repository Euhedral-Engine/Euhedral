package io.euhedral_execution.inference.benchmark.run;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.euhedral_execution.inference.benchmark.config.BenchmarkOptions;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.QwenModel;
import io.euhedral_execution.inference.core.model_loader.artifact.QwenArtifactReader;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactAttentionWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.QwenCompactGatedDeltaNetWeights;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

/// Real packed-weight Q4/Q5 screen. Each invocation uses one forced native mode;
/// compare the emitted BF16 output against a separate SCALAR invocation.
public final class Q45Microbenchmark {
    private Q45Microbenchmark() {}

    public static void run(BenchmarkOptions options, String selectedMatrix, Integer selectedRows) throws Exception {
        String mode = System.getenv("EUHEDRAL_Q45_DISPATCH");
        if (!"SCALAR".equals(mode) && !"DECODE".equals(mode) && !"PREFILL".equals(mode) && !"PREFILL64".equals(mode))
            throw new IllegalArgumentException("q45 requires EUHEDRAL_Q45_DISPATCH=SCALAR|DECODE|PREFILL|PREFILL64");
        if (options.output().toString().endsWith(".json") || Files.exists(options.output()))
            throw new IllegalArgumentException("q45 requires a new JSONL output path");
        if (selectedRows != null && selectedRows <= 0) throw new IllegalArgumentException("rows must be positive");
        String capacityFailure = GpuCapacity.check(options.cudaLibrary(), options.artifact(), options.gpuHeadroomMiB());
        if (capacityFailure != null) throw new IllegalArgumentException(capacityFailure);
        Files.createDirectories(options.output().toAbsolutePath().getParent());
        var artifact = QwenArtifactReader.read(options.artifact());
        try (var gpu = new CudaGpuMemory(options.cudaLibrary());
                var model = QwenModel.load(options.artifact(), artifact, gpu);
                var writer = Files.newBufferedWriter(options.output(), StandardOpenOption.CREATE_NEW)) {
            var gdn = (QwenCompactGatedDeltaNetWeights)
                    model.weights().layers()[0].mixer();
            var attention =
                    (QwenCompactAttentionWeights) model.weights().layers()[3].mixer();
            Map<String, TensorHandle> matrices = new LinkedHashMap<>();
            matrices.put("gdn-q4", gdn.queryKey());
            matrices.put("gdn-q5", gdn.valueZ());
            matrices.put("attention-q4", attention.queryKey());
            matrices.put("attention-q5", attention.gateValue());
            if (selectedMatrix != null && !matrices.containsKey(selectedMatrix))
                throw new IllegalArgumentException("unknown Q4/Q5 matrix: " + selectedMatrix);
            var json = new ObjectMapper();
            for (var entry : matrices.entrySet()) {
                if (selectedMatrix != null && !selectedMatrix.equals(entry.getKey())) continue;
                TensorHandle weight = entry.getValue();
                int k = Math.toIntExact(weight.shape()[1]), n = Math.toIntExact(weight.shape()[0]);
                int bits = entry.getKey().endsWith("q4") ? 4 : 5;
                for (int rows : selectedRows == null
                        ? new int[] {1, 2, 4, 8, 9, 12, 16, 32, 256, 512}
                        : new int[] {selectedRows}) {
                    try (Arena arena = Arena.ofConfined()) {
                        var hostInput = arena.allocate((long) rows * k * Short.BYTES);
                        for (int i = 0; i < rows * k; i++) {
                            float value = (float) (Math.sin(i * 0.017) * 0.7 + Math.cos(i * 0.031) * 0.3);
                            int raw = Float.floatToRawIntBits(value);
                            hostInput.setAtIndex(
                                    ValueLayout.JAVA_SHORT, i, (short) ((raw + 0x7fff + ((raw >>> 16) & 1)) >>> 16));
                        }
                        long x = gpu.allocate(hostInput.byteSize());
                        long y = 0;
                        long eviction = 0;
                        try {
                            y = gpu.allocate((long) rows * n * Short.BYTES);
                            eviction = gpu.allocate(128L * 1024 * 1024);
                            gpu.copyHostToDevice(x, hostInput, hostInput.byteSize());
                            for (int warmup = 0; warmup < options.warmup(); warmup++)
                                linear(gpu, bits, x, weight, y, rows, k, n);
                            long[] samples = new long[options.iterations()];
                            for (int iteration = 0; iteration < samples.length; iteration++) {
                                gpu.zeroDeviceMemory(eviction, 128L * 1024 * 1024);
                                long started = System.nanoTime();
                                linear(gpu, bits, x, weight, y, rows, k, n);
                                samples[iteration] = System.nanoTime() - started;
                            }
                            var hostOutput = arena.allocate((long) rows * n * Short.BYTES);
                            gpu.copyDeviceToHost(hostOutput, y, hostOutput.byteSize());
                            byte[] bytes = hostOutput.toArray(ValueLayout.JAVA_BYTE);
                            var outputFile = options.output()
                                    .resolveSibling(options.output().getFileName() + "." + entry.getKey() + "." + rows
                                            + ".bf16");
                            Files.write(outputFile, bytes, StandardOpenOption.CREATE_NEW);
                            var record = new LinkedHashMap<String, Object>();
                            record.put("schema", "euhedral-inference.q45-operator-screen");
                            record.put("artifact", options.artifact().toString());
                            record.put("matrix", entry.getKey());
                            record.put("weight", weight.name());
                            record.put("bits", bits);
                            record.put("rows", rows);
                            record.put("k", k);
                            record.put("n", n);
                            record.put("mode", mode);
                            record.put("outputFile", outputFile.toString());
                            record.put("input", "deterministic-sin-cos-bf16");
                            record.put("cachePolicy", "128MiB-device-clear-before-each-sample");
                            record.put("boundary", "synchronous-native-call-including-launch");
                            record.put("samplesNanos", samples);
                            writer.write(json.writeValueAsString(record));
                            writer.newLine();
                            writer.flush();
                            System.out.println(
                                    entry.getKey() + " rows=" + rows + " " + mode + " " + Arrays.toString(samples));
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

    private static void linear(
            CudaGpuMemory gpu, int bits, long x, TensorHandle weight, long y, int rows, int k, int n) {
        if (bits == 4) gpu.linearQ4Bf16(x, weight.deviceAddress(), y, rows, k, n, weight.byteSize());
        else gpu.linearQ5Bf16(x, weight.deviceAddress(), y, rows, k, n, weight.byteSize());
    }
}
