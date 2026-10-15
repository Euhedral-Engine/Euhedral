package io.euhedral_execution.inference.quantization.trellis;

import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).
///
/// The trellis kernels (resources/cuda/trellis.cu) on one stream: Viterbi over 256-weight tiles (mul1 codebook,
/// 16-bit state, K bits per weight, 2.5 and 3.5 alternating K and K + 1), packing into exllamav3's int16 words, and
/// decoding words back to values, and the reconstruction of a quantized linear (decode, signed 128-point Hadamards,
/// BF16 `nn.Linear` weight). Tiles are in tensor-core element order; every pointer is a device address.
public final class TrellisKernels {

    private static final int[] INTEGER = {2, 3, 4, 5, 6};
    private static final int[] HALF_STEP = {2, 3};
    private static final int FRAC_THREADS = 512;

    /// Position in a tensor-core ordered tile to its row-major element (`row * 16 + column`) of the 16x16 tile.
    public static final int[] TENSOR_CORE_ORDER = tensorCoreOrder();

    /// Kernels of the per-linear quantizer (Hessian preparation, block LDL, regularization, scale search, LDLQ strips,
    /// refit), launched by [TrellisQuantizer].
    private static final List<String> QUANTIZER_KERNELS = List.of(
            "fill_floats",
            "copy_floats",
            "float_to_half",
            "axpy_floats",
            "subtract_floats",
            "sum_products",
            "strided_mean",
            "add_diagonal",
            "diagonal_io",
            "triangle",
            "scale_floats",
            "bf16_to_float_transposed",
            "invert_diagonal_blocks",
            "right_multiply_blocks",
            "column_rms",
            "row_rms",
            "output_scales",
            "input_scales",
            "tile_mean_square",
            "gather_tiles",
            "candidate_tiles",
            "candidate_mse",
            "strip_gather_sum",
            "strip_error",
            "column_dots",
            "row_dots",
            "column_ratio",
            "hessian_product",
            "clean_row_scales",
            "scale_scalar");

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
        for (int k : INTEGER) names.add("reconstruct_k" + k);
        for (int k : HALF_STEP) names.add("reconstruct_frac_k" + k);
        names.addAll(List.of(
                "hadamard128_rows",
                "hadamard128_columns",
                "strip_tiles",
                "scale_rows_fp16",
                "scale_columns_fp16",
                "transpose_to_bf16"));
        names.addAll(QUANTIZER_KERNELS);
        for (Cuda.Function f : cuda.compileResource("trellis.cu", names)) functions.put(f.name(), f);
        for (int k : INTEGER) {
            Cuda.Function f = functions.get("viterbi_k" + k);
            int shared = viterbiShared(k);
            if (shared > 48 * 1024) cuda.allowSharedMemory(f, shared);
        }
    }

    /// A quantizer kernel by name.
    Cuda.Function kernel(String name) {
        return functions.get(name);
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

    /// Tiles whose Viterbi scratch fits in `bytes`; throws when `bytes` holds less than one tile at `bits`.
    public static int scratchTiles(long bytes, double bits) {
        long tile = scratchBytes(1, bits);
        if (bytes < tile)
            throw new IllegalArgumentException("Viterbi scratch of " + bytes + " bytes holds no tile at " + bits
                    + " bits (one needs " + tile + ")");
        return (int) Math.min(Integer.MAX_VALUE, bytes / tile);
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

    private static int[] tensorCoreOrder() {
        int[] order = new int[256];
        for (int t = 0; t < 32; t++) {
            int r0 = (t % 4) * 2, c0 = t / 4;
            int[] rows = {r0, r0 + 1, r0 + 8, r0 + 9};
            for (int j = 0; j < 8; j++) order[8 * t + j] = rows[j % 4] * 16 + c0 + (j / 4) * 8;
        }
        return order;
    }

    /// In place, every 128-row block of every column of the float32 row-major `x` `[rows, cols]` becomes
    /// `H128 * block / sqrt(128)` (Sylvester order).
    public void hadamardRows(long x, int rows, int cols) {
        validateBlocks(rows, "rows");
        if (cols < 0) throw new IllegalArgumentException("negative cols " + cols);
        if (rows == 0 || cols == 0) return;
        cuda.launch(
                functions.get("hadamard128_rows"),
                (cols + 31) / 32,
                rows / 128,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(x).integer(rows).integer(cols));
    }

    /// In place, every 128-column block of every row of the float32 row-major `x` `[rows, cols]` becomes
    /// `H128 * block / sqrt(128)`.
    public void hadamardColumns(long x, int rows, int cols) {
        validateBlocks(cols, "cols");
        if (rows < 0) throw new IllegalArgumentException("negative rows " + rows);
        if (rows == 0 || cols == 0) return;
        cuda.launch(
                functions.get("hadamard128_columns"),
                cols / 128,
                rows,
                128,
                0,
                stream,
                new Cuda.Arguments().pointer(x).integer(rows).integer(cols));
    }

    private static void validateBlocks(int extent, String name) {
        if (extent < 0 || extent % 128 != 0)
            throw new IllegalArgumentException(name + " " + extent + " not a multiple of 128");
    }

    /// Gathers rows `[firstRow, firstRow + stripRows)` of the float32 `w` `[k, n]` (`stripRows` is 16) into `n / 16`
    /// tiles `[n / 16, 256]` in tensor-core order.
    public void toTiles(long w, int k, int n, int firstRow, int stripRows, long tiles) {
        strip(w, k, n, firstRow, stripRows, tiles, 0);
    }

    /// The inverse of [#toTiles]: scatters the tiles back into the strip of `w`.
    public void fromTiles(long tiles, int k, int n, int firstRow, long w) {
        strip(w, k, n, firstRow, 16, tiles, 1);
    }

    private void strip(long w, int k, int n, int firstRow, int stripRows, long tiles, int inverse) {
        if (stripRows != 16) throw new IllegalArgumentException("strip of " + stripRows + " rows, not 16");
        if (n <= 0 || n % 16 != 0) throw new IllegalArgumentException("n " + n + " not a multiple of 16");
        if (firstRow < 0 || firstRow + 16 > k) throw new IllegalArgumentException("strip outside " + k + " rows");
        cuda.launch(
                functions.get("strip_tiles"),
                n / 16,
                1,
                256,
                0,
                stream,
                new Cuda.Arguments()
                        .pointer(w)
                        .integer(n)
                        .integer(firstRow)
                        .pointer(tiles)
                        .integer(inverse));
    }

    /// Rebuilds the BF16 `nn.Linear` weight `[n, k]` (`out`) of an exllamav3 linear from its packed trellis
    /// `[k / 16, n / 16, 16 * bits]` int16 words and fp16 `suh` `[k]`, `svh` `[n]`:
    /// `W = had_r(had_l(Q) * suh[:, None]) * svh[None, :]` with `Q` the decoded `[k, n]` matrix. `scratch` holds
    /// `4 * k * n` bytes (float32).
    public void reconstruct(long words, long suh, long svh, int k, int n, double bits, long out, long scratch) {
        validate(bits);
        validateBlocks(k, "k");
        validateBlocks(n, "n");
        cuda.launch(
                function("reconstruct", bits),
                (k / 16) * (n / 16),
                1,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(words).integer(n).pointer(scratch));
        hadamardRows(scratch, k, n);
        Cuda.Arguments rows =
                new Cuda.Arguments().pointer(scratch).pointer(suh).integer(k).integer(n);
        cuda.launch(functions.get("scale_rows_fp16"), (n + 255) / 256, k, 256, 0, stream, rows);
        hadamardColumns(scratch, k, n);
        Cuda.Arguments columns =
                new Cuda.Arguments().pointer(scratch).pointer(svh).integer(k).integer(n);
        cuda.launch(functions.get("scale_columns_fp16"), (n + 255) / 256, k, 256, 0, stream, columns);
        cuda.launch(
                functions.get("transpose_to_bf16"),
                (n + 31) / 32,
                (k + 31) / 32,
                256,
                0,
                stream,
                new Cuda.Arguments().pointer(scratch).integer(k).integer(n).pointer(out));
    }
}
