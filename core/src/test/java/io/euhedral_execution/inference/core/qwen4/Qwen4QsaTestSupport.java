package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bits;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model_loader.TensorLoader;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4Artifact;
import io.euhedral_execution.inference.core.model_loader.qwen4.Qwen4ArtifactReader;
import io.euhedral_execution.inference.core.scheduling.AttentionKvState;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/// Device helpers of the QSA GPU tests: opening the GPU, uploads and downloads of arrays, the
/// decoded contents of a sequence's KV pages.
final class Qwen4QsaTestSupport {

    private Qwen4QsaTestSupport() {}

    static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    static short[] random(SplittableRandom rng, int count, double scale) {
        short[] values = new short[count];
        for (int i = 0; i < count; i++) {
            double u = rng.nextDouble(), v = rng.nextDouble();
            double normal = Math.sqrt(-2 * Math.log(Math.max(u, 1e-12))) * Math.cos(2 * Math.PI * v);
            values[i] = bits((float) (normal * scale));
        }
        return values;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, short[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * Short.BYTES));
        MemorySegment host = arena.allocate((long) values.length * Short.BYTES, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_SHORT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * Short.BYTES);
        return address;
    }

    static long upload(CudaGpuMemory gpu, Arena arena, int[] values) {
        long address = gpu.allocate(Math.max(16, (long) values.length * Integer.BYTES));
        MemorySegment host = arena.allocate((long) values.length * Integer.BYTES, 16);
        MemorySegment.copy(values, 0, host, ValueLayout.JAVA_INT, 0, values.length);
        gpu.copyHostToDevice(address, host, (long) values.length * Integer.BYTES);
        return address;
    }

    static short[] downloadShorts(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Short.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Short.BYTES);
        short[] values = new short[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_SHORT, 0, values, 0, values.length);
        return values;
    }

    static int[] downloadInts(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Integer.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Integer.BYTES);
        int[] values = new int[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_INT, 0, values, 0, values.length);
        return values;
    }

    static float[] downloadFloats(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count * Float.BYTES, 16);
        gpu.copyDeviceToHost(host, address, count * Float.BYTES);
        float[] values = new float[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_FLOAT, 0, values, 0, values.length);
        return values;
    }

    static byte[] downloadBytes(CudaGpuMemory gpu, Arena arena, long address, long count) {
        MemorySegment host = arena.allocate(count, 16);
        gpu.copyDeviceToHost(host, address, count);
        byte[] values = new byte[Math.toIntExact(count)];
        MemorySegment.copy(host, ValueLayout.JAVA_BYTE, 0, values, 0, values.length);
        return values;
    }

    /// The effective (unrotated, dequantized) K or V rows of one KV head of the first `tokens`
    /// positions held by a state's page table (`table`: the device table of page addresses),
    /// `[token][256]`.
    static double[][] decodeCache(CudaGpuMemory gpu, Arena arena, long table, int tokens, int heads, int head) {
        int pages = (tokens + AttentionKvState.PAGE_TOKENS - 1) / AttentionKvState.PAGE_TOKENS;
        MemorySegment addresses = arena.allocate((long) pages * Long.BYTES, 8);
        gpu.copyDeviceToHost(addresses, table, (long) pages * Long.BYTES);
        double[][] rows = new double[tokens][];
        long pageBytes = (long) AttentionKvState.PAGE_TOKENS * heads * AttentionKvState.HEAD_ROW_BYTES;
        for (int page = 0; page < pages; page++) {
            byte[] bytes = downloadBytes(gpu, arena, addresses.getAtIndex(ValueLayout.JAVA_LONG, page), pageBytes);
            for (int t = page * AttentionKvState.PAGE_TOKENS;
                    t < Math.min(tokens, (page + 1) * AttentionKvState.PAGE_TOKENS);
                    t++) {
                int offset = ((t % AttentionKvState.PAGE_TOKENS) * heads + head) * AttentionKvState.HEAD_ROW_BYTES;
                rows[t] = Qwen4QsaReference.decodeRow(bytes, offset);
            }
        }
        return rows;
    }

    /// As [#decodeCache] for the listed tokens only (their pages are read, other rows are null).
    static double[][] decodeTokens(
            CudaGpuMemory gpu, Arena arena, long table, int tokens, int heads, int head, boolean[] wanted) {
        int pages = (tokens + AttentionKvState.PAGE_TOKENS - 1) / AttentionKvState.PAGE_TOKENS;
        MemorySegment addresses = arena.allocate((long) pages * Long.BYTES, 8);
        gpu.copyDeviceToHost(addresses, table, (long) pages * Long.BYTES);
        double[][] rows = new double[tokens][];
        long pageBytes = (long) AttentionKvState.PAGE_TOKENS * heads * AttentionKvState.HEAD_ROW_BYTES;
        MemorySegment page = arena.allocate(pageBytes, 16);
        byte[] bytes = new byte[Math.toIntExact(pageBytes)];
        for (int index = 0; index < pages; index++) {
            int first = index * AttentionKvState.PAGE_TOKENS,
                    last = Math.min(tokens, first + AttentionKvState.PAGE_TOKENS);
            boolean any = false;
            for (int t = first; t < last && !any; t++) any = wanted[t];
            if (!any) continue;
            gpu.copyDeviceToHost(page, addresses.getAtIndex(ValueLayout.JAVA_LONG, index), pageBytes);
            MemorySegment.copy(page, ValueLayout.JAVA_BYTE, 0, bytes, 0, bytes.length);
            for (int t = first; t < last; t++)
                if (wanted[t]) {
                    int offset = ((t % AttentionKvState.PAGE_TOKENS) * heads + head) * AttentionKvState.HEAD_ROW_BYTES;
                    rows[t] = Qwen4QsaReference.decodeRow(bytes, offset);
                }
        }
        return rows;
    }

    /// Rows of magnitude 0.5 to 1 with random signs: cheap to make, enough for timing and scale
    /// tests.
    static void fillRandom(MemorySegment segment, SplittableRandom rng, int rows, int width) {
        for (long i = 0; i < (long) rows * width; i++) {
            int r = rng.nextInt();
            short bits = (short) (0x3f00 | (r & 0x7f) | ((r >>> 8) & 0x8000));
            segment.setAtIndex(ValueLayout.JAVA_SHORT, i, bits);
        }
    }

    static Path artifactPath() {
        return Path.of(System.getProperty(
                "euhedral.qwen4.artifact", "/home/brandon/models/qwen3_8_flash_next/qwen3_8_flash_next_nvfp4.edrl"));
    }

    /// The device weights of one QSA layer, freed on close.
    record Loaded(Qwen4QsaLayer.Weights weights, Qwen4Artifact artifact, List<Long> owned, CudaGpuMemory gpu)
            implements AutoCloseable {

        static Loaded load(CudaGpuMemory gpu, int layer) throws IOException {
            Path path = artifactPath();
            assumeTrue(Files.isRegularFile(path), "no Flash-Next artifact at " + path);
            Qwen4Artifact artifact = Qwen4ArtifactReader.read(path);
            String base = "text/layers/" + layer + "/attention/";
            List<Long> owned = new ArrayList<>();
            TensorHandle[] h = new TensorHandle[9];
            String[] names = {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
                "indexer/index_qk_proj",
                "indexer/q_layernorm",
                "indexer/k_layernorm"
            };
            for (int i = 0; i < names.length; i++) {
                h[i] = TensorLoader.load(
                        path, artifact.tensor(base + names[i]).orElseThrow().descriptor(), gpu);
                owned.add(h[i].deviceAddress());
            }
            Qwen4Weight[] w = new Qwen4Weight[h.length];
            for (int i = 0; i < h.length; i++) w[i] = Qwen4Weight.of(h[i].deviceAddress(), h[i].byteSize());
            Qwen4QsaLayer.Weights weights =
                    new Qwen4QsaLayer.Weights(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8]);
            return new Loaded(weights, artifact, owned, gpu);
        }

        @Override
        public void close() {
            for (Long address : this.owned) this.gpu.free(address);
        }
    }
}
