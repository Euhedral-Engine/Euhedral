package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.error;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.TensorLoader;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Artifact;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ArtifactReader;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Config;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Tensor;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Shared pieces of the Flash-Next GPU tests over the real artifact and the reference fixtures.
final class Qwen4TestSupport {

    private Qwen4TestSupport() {}

    static Path artifactPath() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
    }

    static Path fixtureRoot() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.fixtures", System.getProperty("user.home") + "/fixtures/flash-next"));
    }

    static boolean hasArtifact() {
        return Files.isRegularFile(artifactPath());
    }

    static CudaGpuMemory openGpu() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    /// Fixed objects of the artifact on the device, loaded on first use and freed on close.
    static final class Weights implements AutoCloseable {
        private final Path path = artifactPath();
        private final Qwen4Artifact artifact;
        private final CudaGpuMemory gpu;
        private final Map<String, TensorHandle> loaded = new HashMap<>();

        Weights(CudaGpuMemory gpu) throws IOException {
            this.gpu = gpu;
            this.artifact = Qwen4ArtifactReader.read(this.path);
        }

        Qwen4Artifact artifact() {
            return this.artifact;
        }

        Path path() {
            return this.path;
        }

        TensorHandle get(String name) throws IOException {
            TensorHandle handle = this.loaded.get(name);
            if (handle == null) {
                Qwen4Tensor tensor = this.artifact
                        .tensor(name)
                        .orElseThrow(() -> new IllegalArgumentException("artifact has no tensor " + name));
                handle = TensorLoader.load(this.path, tensor.descriptor(), this.gpu);
                this.loaded.put(name, handle);
            }
            return handle;
        }

        long address(String name) throws IOException {
            return get(name).deviceAddress();
        }

        long bytes(String name) throws IOException {
            return get(name).byteSize();
        }

        Qwen4Weight weight(String name) throws IOException {
            TensorHandle handle = get(name);
            return Qwen4Weight.of(handle.deviceAddress(), handle.byteSize());
        }

        @Override
        public void close() {
            for (TensorHandle handle : this.loaded.values()) this.gpu.free(handle.deviceAddress());
            this.loaded.clear();
        }
    }

    static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * 2));
        MemorySegment host = arena.allocate(Math.max(16, (long) values.length * 2), 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * 2);
        return address;
    }

    static short[] downloadBf16(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(Math.max(16, (long) count * 2), 16);
        gpu.copyDeviceToHost(host, address, (long) count * 2);
        short[] values = new short[count];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, count);
        return values;
    }

    static float[] downloadFloat(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(Math.max(16, (long) count * 4), 16);
        gpu.copyDeviceToHost(host, address, (long) count * 4);
        float[] values = new float[count];
        MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, values, 0, count);
        return values;
    }

    static Path fixtures(String name) {
        return Qwen4TestSupport.fixtureRoot().resolve(name);
    }

    static float epsilon(Qwen4Artifact artifact) {
        return (float) artifact.config().text().rmsNormEpsilon();
    }

    static Qwen4HyperConnection hyperConnection(Qwen4Artifact artifact) {
        Qwen4Config config = artifact.config();
        return new Qwen4HyperConnection(
                config.hyperConnection().count(),
                config.text().hiddenSize(),
                config.hyperConnection().lowrank(),
                epsilon(artifact));
    }

    static Qwen4HyperConnection.Weights hcWeights(Qwen4TestSupport.Weights weights, String prefix) throws Exception {
        return new Qwen4HyperConnection.Weights(
                weights.weight(prefix + "/hc_norm"),
                weights.weight(prefix + "/input_mix_down"),
                weights.weight(prefix + "/input_mix_up"),
                weights.artifact().tensor(prefix + "/block_inject").isPresent()
                        ? weights.weight(prefix + "/block_inject")
                        : null);
    }

    /// The injection weights the reference reports (2 * sigmoid(raw / streams)) from the engine's raw projection.
    static short[] injectionFromRaw(short[] raw, int streams) {
        short[] weights = new short[raw.length];
        for (int i = 0; i < raw.length; i++)
            weights[i] = bits(2.0f * Qwen4Reference.r(Qwen4Reference.sigmoid(Qwen4Reference.r(bf(raw[i]) / streams))));
        return weights;
    }

    /// Compares the hyper-connection of `prefix` (attn_hc or mlp_hc) of layer `layer` in chunk `chunk`.
    static void checkHyperConnection(
            CudaGpuMemory gpu,
            Arena arena,
            Qwen4TestSupport.Weights weights,
            ReferenceFixtures fixture,
            Qwen4TestSupport.Report report,
            int chunk,
            int layer,
            String prefix,
            String inputName,
            double bound)
            throws Exception {
        Qwen4HyperConnection connection = hyperConnection(weights.artifact());
        String at = "c" + chunk + "/L" + layer + "/" + prefix + "/";
        short[] state = fixture.bf16(inputName);
        int rows = state.length / connection.stateWidth();
        long stateAddress = upload(gpu, arena, state);
        long scratchAddress = gpu.allocate(connection.scratchBytes(rows));
        long mixedAddress = gpu.allocate((long) rows * connection.hidden() * 2);
        try {
            var scratch = connection.scratch(scratchAddress, rows);
            connection.mix(
                    gpu,
                    hcWeights(weights, "text/layers/" + layer + "/" + prefix),
                    stateAddress,
                    scratch,
                    mixedAddress,
                    rows);
            int width = connection.stateWidth();
            report.check(
                    at + "normed",
                    error(fixture.bf16(at + "normed"), downloadBf16(gpu, arena, scratch.normed(), rows * width)),
                    bound);
            report.check(
                    at + "down",
                    error(fixture.bf16(at + "down"), downloadBf16(gpu, arena, scratch.down(), rows * 320)),
                    bound);
            report.check(
                    at + "up",
                    error(fixture.bf16(at + "up"), downloadBf16(gpu, arena, scratch.up(), rows * width)),
                    bound);
            report.check(
                    at + "mixed",
                    error(
                            fixture.bf16(at + "mixed"),
                            downloadBf16(gpu, arena, mixedAddress, rows * connection.hidden())),
                    bound);
            report.check(
                    at + "injection",
                    error(
                            fixture.bf16(at + "injection"),
                            injectionFromRaw(downloadBf16(gpu, arena, scratch.rawInjection(), rows * 4), 4)),
                    bound);
        } finally {
            gpu.free(mixedAddress);
            gpu.free(scratchAddress);
            gpu.free(stateAddress);
        }
    }

    /// Error of an engine tensor against the reference, in the units that matter for BF16 data.
    record Error(double relativeRms, double maxAbsOverRms, int count) {
        @Override
        public String toString() {
            return String.format("rel-rms %.3e max|d|/rms %.3e (%d)", this.relativeRms, this.maxAbsOverRms, this.count);
        }
    }

    static Error error(short[] expected, short[] actual) {
        if (expected.length != actual.length)
            throw new AssertionError("length " + actual.length + " != " + expected.length);
        double sumDiff = 0, sumRef = 0, maxDiff = 0;
        for (int i = 0; i < expected.length; i++) {
            double e = bf(expected[i]), d = bf(actual[i]) - e;
            sumDiff += d * d;
            sumRef += e * e;
            maxDiff = Math.max(maxDiff, Math.abs(d));
        }
        double rms = Math.sqrt(sumRef / Math.max(1, expected.length));
        return new Error(Math.sqrt(sumDiff / Math.max(sumRef, 1e-300)), rms == 0 ? 0 : maxDiff / rms, expected.length);
    }

    static Error error(float[] expected, float[] actual) {
        double sumDiff = 0, sumRef = 0, maxDiff = 0;
        for (int i = 0; i < expected.length; i++) {
            double e = expected[i], d = actual[i] - e;
            sumDiff += d * d;
            sumRef += e * e;
            maxDiff = Math.max(maxDiff, Math.abs(d));
        }
        double rms = Math.sqrt(sumRef / Math.max(1, expected.length));
        return new Error(Math.sqrt(sumDiff / Math.max(sumRef, 1e-300)), rms == 0 ? 0 : maxDiff / rms, expected.length);
    }

    /// Collects errors by tensor name and checks them against bounds at the end, so one run reports everything.
    static final class Report {
        private final String title;
        private final List<String> lines = new ArrayList<>();
        private final List<String> failures = new ArrayList<>();
        private final Map<String, Double> worst = new HashMap<>();

        Report(String title) {
            this.title = title;
        }

        /// Records `error` for `name` and fails the report when its relative RMS exceeds `bound`.
        void check(String name, Error error, double bound) {
            this.lines.add(String.format("%-44s %s (bound %.1e)", name, error, bound));
            this.worst.merge(name.replaceAll("^c\\d+/", ""), error.relativeRms(), Math::max);
            if (!(error.relativeRms() <= bound)) this.failures.add(name + ": " + error + " exceeds " + bound);
        }

        void finish() {
            System.out.println("== " + this.title + " ==");
            if (Boolean.getBoolean("euhedral.qwen4.verbose")) for (String line : this.lines) System.out.println(line);
            System.out.println("-- worst relative RMS per tensor --");
            this.worst.entrySet().stream()
                    .sorted(Map.Entry.comparingByKey())
                    .forEach(e -> System.out.printf("%-40s %.3e%n", e.getKey(), e.getValue()));
            if (!this.failures.isEmpty())
                throw new AssertionError(this.title + ": " + this.failures.size() + " tensors out of bounds:\n"
                        + String.join("\n", this.failures));
        }
    }
}
