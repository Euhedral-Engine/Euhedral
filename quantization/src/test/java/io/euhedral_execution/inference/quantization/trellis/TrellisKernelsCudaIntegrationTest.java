package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.quantization.gptq.Npy;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Upload;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/// The trellis kernels against exllamav3's own quantize_tiles and packing (frozen fixture, tools/trellis): identical
/// Viterbi states, decoded values and packed words for every rate the conversion uses.
class TrellisKernelsCudaIntegrationTest {

    @ParameterizedTest
    @ValueSource(strings = {"2", "3", "4", "5", "6", "2.5", "3.5"})
    void matchesExllamav3(String rate) throws Exception {
        run(rate, 64);
    }

    /// The same tiles through a scratch of 16 tiles: chunking is exact.
    @ParameterizedTest
    @ValueSource(strings = {"2", "2.5", "4", "3.5"})
    void chunkedScratchIsExact(String rate) throws Exception {
        run(rate, 16);
    }

    private void run(String rate, int scratchTiles) throws Exception {
        double bits = Double.parseDouble(rate);
        Npy in = read("tiles-in.npy");
        short[] idx = shorts(read("tiles-" + rate + "-idx.npy"));
        short[] out = shorts(read("tiles-" + rate + "-out.npy"));
        short[] packed = shorts(read("tiles-" + rate + "-packed.npy"));
        int count = 64, words = TrellisKernels.wordsPerTile(bits);
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Arena arena = Arena.ofConfined()) {
            Cuda cuda = new Cuda();
            TrellisKernels kernels = new TrellisKernels(cuda, 0L);
            long tiles = gpu.allocate(4L * count * 256), indices = gpu.allocate(2L * count * 256);
            long decoded = gpu.allocate(4L * count * 256), packedDevice = gpu.allocate(2L * count * words);
            long decodedAgain = gpu.allocate(4L * count * 256);
            long scratch = gpu.allocate(TrellisKernels.scratchBytes(scratchTiles, bits));
            Upload.segment(gpu, tiles, MemorySegment.ofBuffer(in.data()), 4L * count * 256);
            kernels.viterbi(tiles, count, bits, indices, decoded, scratch, scratchTiles);
            kernels.pack(indices, count, bits, packedDevice);
            kernels.decode(packedDevice, count, bits, decodedAgain);
            cuda.synchronize(0L);
            assertArrayEquals(idx, download(gpu, arena, indices, count * 256), "states, K " + rate);
            assertArrayEquals(packed, download(gpu, arena, packedDevice, count * words), "words, K " + rate);
            float[] values = downloadFloats(gpu, arena, decoded, count * 256);
            float[] again = downloadFloats(gpu, arena, decodedAgain, count * 256);
            for (int i = 0; i < values.length; i++) {
                assertEquals(Float.float16ToFloat(out[i]), values[i], 0f, "value " + i + ", K " + rate);
                assertEquals(values[i], again[i], 0f, "decode " + i + ", K " + rate);
            }
            for (long a : new long[] {tiles, indices, decoded, packedDevice, decodedAgain, scratch}) gpu.free(a);
        }
    }

    static Npy read(String name) throws Exception {
        try (InputStream input =
                TrellisKernelsCudaIntegrationTest.class.getResourceAsStream("/trellis-fixture/" + name)) {
            return Npy.read(input);
        }
    }

    static short[] shorts(Npy npy) {
        short[] values = new short[npy.data().remaining() / 2];
        npy.data().asShortBuffer().get(values);
        return values;
    }

    static short[] download(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(2L * count);
        gpu.copyDeviceToHost(host, address, 2L * count);
        return host.toArray(ValueLayout.JAVA_SHORT);
    }

    static float[] downloadFloats(CudaGpuMemory gpu, Arena arena, long address, int count) {
        MemorySegment host = arena.allocate(4L * count);
        gpu.copyDeviceToHost(host, address, 4L * count);
        return host.toArray(ValueLayout.JAVA_FLOAT);
    }
}
