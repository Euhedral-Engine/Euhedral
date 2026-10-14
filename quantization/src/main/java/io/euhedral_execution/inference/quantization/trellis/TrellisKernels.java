package io.euhedral_execution.inference.quantization.trellis;

import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// The trellis kernels (resources/cuda/trellis.cu) on one stream: Viterbi over 256-weight tiles (mul1 codebook,
/// 16-bit state, K bits per weight, 2.5 and 3.5 alternating K and K + 1), packing into exllamav3's int16 words, and
/// decoding words back to values. Tiles are in tensor-core element order; every pointer is a device address.
public final class TrellisKernels {

    private static final int[] INTEGER = {2, 3, 4, 5, 6};
    private static final int[] HALF_STEP = {2, 3};
    private static final int FRAC_THREADS = 512;

    private final Cuda cuda;
    private final long stream;
    private final Map<String, Cuda.Function> functions = new HashMap<>();

    public TrellisKernels(Cuda cuda, long stream) {
        this.cuda = cuda;
        this.stream = stream;
        List<String> names = new ArrayList<>();
        for (int k : INTEGER) for (String kind : List.of("viterbi_k", "pack_k", "decode_k")) names.add(kind + k);
        for (int k : HALF_STEP)
            for (String kind : List.of("viterbi_frac_k", "pack_frac_k", "decode_frac_k")) names.add(kind + k);
        for (Cuda.Function f : cuda.compileResource("trellis.cu", names)) functions.put(f.name(), f);
        for (int k : INTEGER) {
            Cuda.Function f = functions.get("viterbi_k" + k);
            int shared = viterbiShared(k);
            if (shared > 48 * 1024) cuda.allowSharedMemory(f, shared);
        }
    }

    /// Words (int16) per packed 256-weight tile.
    public static int wordsPerTile(double bits) {
        validate(bits);
        return (int) Math.round(16 * bits);
    }

    /// Bytes of Viterbi scratch for `count` tiles. Integer rates keep one history byte per position and state,
    /// `[count, 256, 65536 >> K]`; half-step rates also keep two cost arrays and 16-bit history (K the lower rate).
    public static long scratchBytes(int count, double bits) {
        validate(bits);
        int k = (int) Math.floor(bits);
        long edges = 65536L >> k;
        if (bits == k) return (long) count * 256 * edges;
        return (long) count * (2 * 2 * edges + 2 * 256 * edges);
    }

    private static void validate(double bits) {
        int k = (int) Math.floor(bits);
        boolean supported = bits != k ? (bits == 2.5 || bits == 3.5) : k >= 2 && k <= 6;
        if (!supported) throw new IllegalArgumentException("unsupported rate " + bits);
    }

    private static void validate(int count) {
        if (count < 0) throw new IllegalArgumentException("negative tile count " + count);
    }

    /// Tiles whose Viterbi scratch fits in `bytes` (at least one).
    public static int scratchTiles(long bytes, double bits) {
        return (int) Math.max(1, Math.min(Integer.MAX_VALUE, bytes / scratchBytes(1, bits)));
    }

    private Cuda.Function function(String kind, double bits) {
        validate(bits);
        int k = (int) Math.floor(bits);
        boolean frac = bits != k;
        return functions.get(kind + (frac ? "_frac_k" : "_k") + k);
    }

    /// Quantizes `count` float32 tiles `[count, 256]` to trellis states (uint16 `indices`) and the float32
    /// codebook values they decode to. `scratch` holds `scratchBytes(scratchTiles, bits)`. Tiles run in chunks of
    /// at most `scratchTiles` (and of the resident wave) on the stream, reusing the scratch; blocks are
    /// independent, so chunking does not change any result.
    public void viterbi(
            long tiles, int count, double bits, long indices, long decoded, long scratch, int scratchTiles) {
        validate(count);
        if (count == 0) return;
        if (scratchTiles < 1) throw new IllegalArgumentException("scratch holds no tile");
        int k = (int) Math.floor(bits);
        boolean frac = bits != k;
        int threads = frac ? FRAC_THREADS : viterbiThreads(k);
        int shared = frac ? fracShared() : viterbiShared(k);
        Cuda.Function f = function("viterbi", bits);
        int wave = Math.max(1, cuda.blocksPerMultiprocessor(f, threads, shared)) * cuda.multiprocessors();
        int chunk = Math.min(scratchTiles, wave);
        for (int start = 0; start < count; start += chunk) {
            int n = Math.min(chunk, count - start);
            long offset = 256L * start;
            cuda.launch(
                    f,
                    n,
                    1,
                    threads,
                    shared,
                    stream,
                    new Cuda.Arguments()
                            .pointer(tiles + 4 * offset)
                            .integer(n)
                            .pointer(indices + 2 * offset)
                            .pointer(decoded + 4 * offset)
                            .pointer(scratch));
        }
    }

    private static int viterbiThreads(int k) {
        return k == 5 || k == 6 ? 512 : 1024;
    }

    /// Dynamic shared memory of the integer-rate Viterbi: two fp16 cost arrays, the tile and the reduction slots.
    static int viterbiShared(int k) {
        int edges = 65536 >> k;
        return 2 * edges * 2 + 256 * 2 + 64 + 128;
    }

    private static int fracShared() {
        return 256 * 2 + 32 * 4 + 128;
    }

    /// Packs uint16 states `[count, 256]` into `[count, 16 * bits]` int16 words, MSB first.
    public void pack(long indices, int count, double bits, long words) {
        validate(count);
        if (count == 0) return;
        boolean frac = bits != Math.floor(bits);
        int blocks = frac ? (count + 127) / 128 : count;
        cuda.launch(
                function("pack", bits),
                blocks,
                1,
                128,
                0,
                stream,
                new Cuda.Arguments().pointer(indices).integer(count).pointer(words));
    }

    /// Decodes packed words back to float32 codebook values `[count, 256]`.
    public void decode(long words, int count, double bits, long decoded) {
        validate(count);
        if (count == 0) return;
        cuda.launch(
                function("decode", bits),
                count,
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(words).integer(count).pointer(decoded));
    }
}
