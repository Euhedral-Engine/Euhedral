package io.euhedral_execution.inference.quantization.mtp;

import io.euhedral_execution.inference.quantization.gpu.Cuda;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// The drafter training's own kernels (`drafter.cu`) on one stream: NVFP4 rounding, the unrolled drafts' attention
/// pieces, the draft head's cross-entropy and the Q3 embedding rows. Tensors are FP32 and row-major.
final class DrafterKernels {
    private static final int THREADS = 256;
    private static final List<String> KERNELS = List.of(
            "absolute_maximum",
            "nvfp4_round",
            "nvfp4_decode_bf16",
            "drafter_softmax",
            "drafter_softmax_backward",
            "lineage_dot",
            "lineage_spread",
            "lineage_gather",
            "copy_columns",
            "draft_cross_entropy",
            "float_to_bf16",
            "q3_embedding_rows");

    private final Cuda cuda;
    private final long stream;
    private final Map<String, Cuda.Function> functions = new HashMap<>();

    DrafterKernels(Cuda cuda, long stream) {
        this.cuda = cuda;
        this.stream = stream;
        List<Cuda.Function> compiled = cuda.compileResource("drafter.cu", KERNELS);
        for (int i = 0; i < KERNELS.size(); i++) functions.put(KERNELS.get(i), compiled.get(i));
    }

    private void launch(String kernel, long blocks, int threads, Cuda.Arguments arguments) {
        cuda.launch(functions.get(kernel), Math.toIntExact(blocks), 1, threads, 0, stream, arguments);
    }

    private void elementwise(String kernel, long count, Cuda.Arguments arguments) {
        launch(kernel, (count + THREADS - 1) / THREADS, THREADS, arguments);
    }

    /// `out` (one unsigned, already zero) = the bits of the largest |x|.
    void absoluteMaximum(long x, long count, long out) {
        long blocks = Math.min(1024, (count + THREADS - 1) / THREADS);
        launch(
                "absolute_maximum",
                blocks,
                THREADS,
                new Cuda.Arguments().pointer(x).pointer(count).pointer(out));
    }

    /// `out` = w [rows, k] rounded to NVFP4 under the global scale of `amaxBits`; when `packed` is not 0, also the
    /// artifact's packed codes [rows, k / 2] and scale codes [rows, k / 16]. k is a multiple of 16.
    void nvfp4Round(long w, long out, long packed, long scales, long rows, int k, long amaxBits) {
        long blocks = rows * (k / 16);
        elementwise(
                "nvfp4_round",
                blocks,
                new Cuda.Arguments()
                        .pointer(w)
                        .pointer(out)
                        .pointer(packed)
                        .pointer(scales)
                        .pointer(blocks)
                        .pointer(amaxBits));
    }

    /// BF16 [rows, k] of an artifact's NVFP4 rows (`kPad` the padded row length of its planes).
    void nvfp4DecodeBf16(long packed, long scales, long global, long rows, int kPad, int k, long out) {
        elementwise(
                "nvfp4_decode_bf16",
                rows * k,
                new Cuda.Arguments()
                        .pointer(packed)
                        .pointer(scales)
                        .pointer(global)
                        .pointer(rows)
                        .integer(kPad)
                        .integer(k)
                        .pointer(out));
    }

    void softmax(
            long scores,
            int ldS,
            int visible0,
            long extras,
            int extraStride,
            int extraCount,
            int heads,
            int rows,
            float scale) {
        launch(
                "drafter_softmax",
                (long) heads * rows,
                THREADS,
                new Cuda.Arguments()
                        .pointer(scores)
                        .integer(ldS)
                        .integer(visible0)
                        .pointer(extras)
                        .integer(extraStride)
                        .integer(extraCount)
                        .integer(rows)
                        .real(scale));
    }

    void softmaxBackward(
            long probabilities,
            long dScores,
            int ldS,
            int visible0,
            long extraProbabilities,
            long dExtras,
            int extraStride,
            int extraCount,
            int heads,
            int rows,
            float scale) {
        launch(
                "drafter_softmax_backward",
                (long) heads * rows,
                THREADS,
                new Cuda.Arguments()
                        .pointer(probabilities)
                        .pointer(dScores)
                        .integer(ldS)
                        .integer(visible0)
                        .pointer(extraProbabilities)
                        .pointer(dExtras)
                        .integer(extraStride)
                        .integer(extraCount)
                        .integer(rows)
                        .real(scale));
    }

    /// out[(h rows + r) stride + slot] = a[r, h d ..] . b[r, (h / group) d ..].
    void lineageDot(
            long a, int lda, long b, int ldb, int heads, int group, int d, int rows, long out, int stride, int slot) {
        launch(
                "lineage_dot",
                (long) rows * heads,
                d,
                new Cuda.Arguments()
                        .pointer(a)
                        .integer(lda)
                        .pointer(b)
                        .integer(ldb)
                        .integer(heads)
                        .integer(group)
                        .integer(d)
                        .integer(rows)
                        .pointer(out)
                        .integer(stride)
                        .integer(slot));
    }

    /// y[r, h d + i] += c[(h rows + r) stride + slot] x[r, (h / group) d + i].
    void lineageSpread(
            long c, int stride, int slot, long x, int ldx, long y, int ldy, int heads, int group, int d, int rows) {
        launch(
                "lineage_spread",
                (long) rows * heads,
                d,
                new Cuda.Arguments()
                        .pointer(c)
                        .integer(stride)
                        .integer(slot)
                        .pointer(x)
                        .integer(ldx)
                        .pointer(y)
                        .integer(ldy)
                        .integer(heads)
                        .integer(group)
                        .integer(d)
                        .integer(rows));
    }

    /// y[r, kv d + i] += sum over the group of kv of c[(h rows + r) stride + slot] x[r, h d + i].
    void lineageGather(
            long c, int stride, int slot, long x, int ldx, long y, int ldy, int heads, int group, int d, int rows) {
        launch(
                "lineage_gather",
                (long) rows * (heads / group),
                d,
                new Cuda.Arguments()
                        .pointer(c)
                        .integer(stride)
                        .integer(slot)
                        .pointer(x)
                        .integer(ldx)
                        .pointer(y)
                        .integer(ldy)
                        .integer(heads)
                        .integer(group)
                        .integer(d)
                        .integer(rows));
    }

    /// y[r, c] = (or += when `add`) x[r, c] for c < columns.
    void copyColumns(long x, int ldx, long y, int ldy, int rows, int columns, boolean add) {
        elementwise(
                "copy_columns",
                (long) rows * columns,
                new Cuda.Arguments()
                        .pointer(x)
                        .integer(ldx)
                        .pointer(y)
                        .integer(ldy)
                        .integer(rows)
                        .integer(columns)
                        .integer(add ? 1 : 0));
    }

    /// Per row of logits [rows, vocabulary] in place: (softmax - onehot(label)) scale, or 0 without a label; loss (a
    /// double) += -log p(label); predicted[r] = the argmax; labelLogProbability[r] = log p(label).
    void crossEntropy(
            long logits,
            int rows,
            int vocabulary,
            long labels,
            float scale,
            long loss,
            long predicted,
            long labelLogProbability) {
        launch(
                "draft_cross_entropy",
                rows,
                THREADS,
                new Cuda.Arguments()
                        .pointer(logits)
                        .integer(vocabulary)
                        .pointer(labels)
                        .real(scale)
                        .pointer(loss)
                        .pointer(predicted)
                        .pointer(labelLogProbability));
    }

    void floatToBf16(long x, long count, long out) {
        elementwise(
                "float_to_bf16",
                count,
                new Cuda.Arguments().pointer(x).pointer(count).pointer(out));
    }

    /// FP32 [rows, width] embedding rows of `ids` from a Q3G64 table (pinned host memory read in place).
    void q3EmbeddingRows(long table, long ids, int rows, int width, long scaleOffset, long out) {
        elementwise(
                "q3_embedding_rows",
                (long) rows * width,
                new Cuda.Arguments()
                        .pointer(table)
                        .pointer(ids)
                        .integer(rows)
                        .integer(width)
                        .pointer(scaleOffset)
                        .pointer(out));
    }
}
