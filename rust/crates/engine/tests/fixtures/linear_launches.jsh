// The launches Java makes for the Qwen3.8 trellis linears: kernel, grid, block, shared bytes and arguments, and the
// scratch the format plans, for the shapes of the RCO artifact's layers over row counts and trellis thresholds.
//
// Regenerate (Java 25, from the repository root, after `./gradlew :core:compileJava :core:compileTestJava`; CLASSPATH
// is the core module's test runtime classpath):
//   jshell --class-path "$CLASSPATH" rust/crates/engine/tests/fixtures/linear_launches.jsh > rust/crates/engine/tests/fixtures/linear_launches.txt
//
// One case per block: a `case` line, then `scratch=` (the format's planned bytes, then the decode-type route's), then
// one line per launch. Pointers print as `p<hex>`, 32-bit values as `i<decimal>`.

import io.euhedral_execution.inference.core.artifact.*;
import io.euhedral_execution.inference.core.artifact.format.*;
import io.euhedral_execution.inference.core.gpu.*;
import io.euhedral_execution.inference.core.model.qwen38.*;
import java.util.*;

class RecordingGpu extends MemoryGpu {
    TrellisSettings settings;
    boolean fp8;
    boolean exact;
    final StringBuilder out = new StringBuilder();

    @Override public TrellisSettings trellisSettings() { return settings; }
    @Override public boolean exactNumerics() { return exact; }
    @Override public boolean tableKernelAvailable(TableKernel kernel) {
        return fp8 || !kernel.symbol().endsWith("_fp8");
    }
    @Override public void launchTableKernel(TableKernel kernel, int gx, int gy, int gz, int bx, int by, int bz,
            int smem, KernelArguments arguments) {
        out.append(kernel.symbol()).append(" grid=").append(gx).append(',').append(gy).append(',').append(gz)
                .append(" block=").append(bx).append(',').append(by).append(',').append(bz)
                .append(" smem=").append(smem).append(" args=");
        for (int i = 0; i < arguments.count(); i++) {
            if (i > 0) out.append(',');
            long word = arguments.words()[i];
            if (arguments.sizes()[i] == 8) out.append("p").append(Long.toHexString(word));
            else out.append("i").append(word);
        }
        out.append('\n');
    }
}

final long X = 0x100000000L, WORDS = 0x200000000L, SUH = 0x300000000L, SVH = 0x400000000L, ENDS = 0x500000000L,
        Y = 0x600000000L, SCRATCH = 0x700000000L;

void emit(RecordingGpu gpu, String name, int n, int k, int w, int[] ends, int rows, boolean rowExact) {
    var format = LinearFormats.find(WeightFormat.TRELLIS, WeightLayout.TRELLIS_TILE16_V1).orElseThrow();
    var weights = List.of(
            new TensorHandle(name, new long[] {k / 16, n / 16, w}, TensorDataType.BF16, WeightFormat.TRELLIS,
                    WeightLayout.TRELLIS_TILE16_V1, WORDS, (long) k * n * w / 128),
            new TensorHandle(name + ".suh", new long[] {ends.length, k}, TensorDataType.FP16, WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1, SUH, (long) ends.length * k * 2),
            new TensorHandle(name + ".svh", new long[] {n}, TensorDataType.FP16, WeightFormat.FP16,
                    WeightLayout.CONTIGUOUS_LE_V1, SVH, (long) n * 2),
            new TensorHandle(name + ".segments", new long[] {ends.length}, TensorDataType.INT32, WeightFormat.I32,
                    WeightLayout.CONTIGUOUS_LE_V1, ENDS, ends.length * 4L));
    long planned = format.scratchBytes(gpu, ScratchUse.TRELLIS_LINEAR, rows, k, n, weights);
    long decode = format.decodeScratchBytes(gpu, ScratchUse.TRELLIS_LINEAR, rows, k, n, weights);
    gpu.out.append("case ").append(name).append(" n=").append(n).append(" k=").append(k).append(" w=").append(w)
            .append(" ends=").append(Arrays.toString(ends).replace(" ", "")).append(" rows=").append(rows)
            .append(" rowExact=").append(rowExact).append(" fp8=").append(gpu.fp8).append(" exact=").append(gpu.exact)
            .append(" target=").append(gpu.settings.targetCtas()).append(" mma=").append(gpu.settings.mmaMaxRows())
            .append(" wholeK=").append(gpu.settings.wholeKRows()).append(" wide=").append(gpu.settings.wholeKWideRows())
            .append('\n').append("scratch=").append(planned).append(',').append(decode).append('\n');
    format.linear(gpu, new LinearLaunch(ExecutionPlan.Kind.TRELLIS_LINEAR, weights, X, Y, rows, k, n, SCRATCH, planned,
            rowExact, false, ends));
}

record Shape(String name, int n, int k, int w, int[] ends) {}

var shapes = List.of(
        new Shape("gdn/query_key", 4096, 5120, 40, new int[] {4096}),
        new Shape("gdn/value_z", 12288, 5120, 40, new int[] {6144, 12288}),
        new Shape("gdn/output", 5120, 6144, 40, new int[] {5120}),
        new Shape("mlp/gate_up", 34816, 5120, 48, new int[] {17408, 34816}),
        new Shape("mlp/down", 5120, 17408, 56, new int[] {5120}),
        new Shape("attention/query_key", 7168, 5120, 64, new int[] {3584, 7168}));

// (target blocks, tensor-core max rows, whole-K rows, wide rows, rows)
var variants = List.of(
        new int[][] {{560, 512, 64, 128}, {1, 2, 3, 8, 9, 32, 63, 64, 65, 127, 128, 512, 513, 1024}},
        new int[][] {{384, 512, 64, 128}, {1, 8, 64, 128}},
        new int[][] {{560, 16, 32, 64}, {8, 16, 17, 31, 32, 64, 100}},
        new int[][] {{1120, 512, 16, 32}, {4, 16, 32, 33}});

var gpu = new RecordingGpu();
for (var variant : variants) {
    gpu.settings = new TrellisSettings(variant[0][1], variant[0][0], variant[0][2], variant[0][3]);
    for (var shape : shapes)
        for (int rows : variant[1])
            for (boolean rowExact : new boolean[] {false, true})
                for (boolean fp8 : new boolean[] {true, false}) {
                    // The device has the FP8 kernels or it does not; a verification never takes them either way.
                    if (rowExact && !fp8) continue;
                    gpu.fp8 = fp8;
                    emit(gpu, shape.name(), shape.n(), shape.k(), shape.w(), shape.ends(), rows, rowExact);
                }
}
// The scalar reference under exact numerics.
gpu.settings = new TrellisSettings(512, 560, 64, 128);
gpu.fp8 = true;
gpu.exact = true;
for (var shape : shapes)
    for (int rows : new int[] {1, 8, 100})
        emit(gpu, shape.name(), shape.n(), shape.k(), shape.w(), shape.ends(), rows, false);
System.out.print(gpu.out);
/exit
