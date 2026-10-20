package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.artifact.TensorDataReader;
import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.TensorLoader;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormat;
import io.euhedral_execution.inference.core.artifact.format.LinearFormats;
import io.euhedral_execution.inference.core.artifact.format.LinearLaunch;
import io.euhedral_execution.inference.core.artifact.format.TrellisLayout;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Times one launch of each option's linear for every distinct projection shape of the sources, at each row count,
/// on the device.
///
/// A launch is built the way the engine builds the linear of a stage: the kind from the object's
/// [LinearFormat#kind], the scratch from [LinearFormat#scratchUse] and [LinearFormat#scratchBytes], and the trellis
/// segment ends read from the object's `.segments` companion, so the route a row count takes is the route the engine
/// takes for it (a trellis launch of at most `euhedral.trellis.mma-max-rows` rows runs the tensor-core route). Rows
/// of 2 to [#VERIFY_MAX_ROWS] are launched as a verification, as the engine does. The payloads are the options' real
/// ones; the launches rotate over enough copies to leave the weights out of the L2 cache.
public final class Cost {

    /// The most rows of a launch that is a verification, not a prefill chunk: the MTP drafts verify at most this
    /// many positions.
    static final int VERIFY_MAX_ROWS = 8;

    /// BF16 0.0078125, the value the input rows are filled with; the timing does not depend on the contents.
    private static final short INPUT_VALUE = 0x3C00;

    /// The two host-mapped 32-bit flags of a timed run: start, then done.
    private static final long FLAG_BYTES = 4096;

    private static final VarHandle FLAG = ValueLayout.JAVA_INT.varHandle();

    private static final ValueLayout.OfShort SHORT =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Cost() {}

    /// The input and output features of a linear.
    public record Shape(int in, int out) {}

    /// One measurement: a launch of `option`'s linear of `shape` over `rows` rows.
    public record Key(Shape shape, Option option, int rows) {}

    /// The features of the linear stored as `descriptor`: `[n, k]` for the row-split formats, and, for trellis
    /// tiles, `[k/16, n/16, w]`.
    public static Shape shape(TensorDescriptor descriptor) {
        long[] shape = descriptor.shape();
        if (descriptor.format() == WeightFormat.TRELLIS && shape.length == 3)
            return new Shape(Math.toIntExact(shape[0] * 16), Math.toIntExact(shape[1] * 16));
        if (shape.length != 2)
            throw new IllegalArgumentException(descriptor.name() + " is not a linear: shape rank " + shape.length);
        return new Shape(Math.toIntExact(shape[1]), Math.toIntExact(shape[0]));
    }

    /// Measures every distinct projection shape of `sources` for every option, at each of `rows`, and writes the
    /// table (with no fixed terms) to `out`.
    ///
    /// @param library the CUDA libraries
    /// @param repeats timed launches per measurement, round-robin over the copies
    /// @param rotateBytes the least payload bytes the copies of an object span
    public static CostTable run(
            CudaLibraries library, Sources sources, List<Integer> rows, Path out, int repeats, long rotateBytes)
            throws IOException {
        if (repeats < 1) throw new IllegalArgumentException("repeats must be positive: " + repeats);
        if (rows.isEmpty()) throw new IllegalArgumentException("no row counts");
        Map<Shape, String> firstOfShape = new LinkedHashMap<>();
        for (String name : sources.projections())
            firstOfShape.putIfAbsent(shape(sources.descriptor(sources.base(), name)), name);
        Map<Key, Double> micros = new HashMap<>();
        try (Device device = new Device(library)) {
            for (var entry : firstOfShape.entrySet()) {
                for (Option option : Option.values()) {
                    if (!sources.options().contains(option)) continue;
                    for (var measured : measure(device, sources, option, entry.getValue(), rows, repeats, rotateBytes)
                            .entrySet()) {
                        micros.put(new Key(entry.getKey(), option, measured.getKey()), measured.getValue());
                        System.err.println(option.key() + " " + entry.getKey().in() + " x "
                                + entry.getKey().out() + " rows " + measured.getKey() + ": " + measured.getValue()
                                + " us");
                    }
                }
            }
        }
        CostTable table = new CostTable(micros, 0, Map.of());
        table.write(out);
        return table;
    }

    /// The microseconds per launch of `option`'s object `name`, by row count.
    private static Map<Integer, Double> measure(
            Device device,
            Sources sources,
            Option option,
            String name,
            List<Integer> rows,
            int repeats,
            long rotateBytes)
            throws IOException {
        List<TensorDescriptor> object = sources.objectWithCompanions(option, name);
        Shape shape = shape(object.getFirst());
        long payload = object.stream().mapToLong(TensorDescriptor::byteSize).sum();
        int copies = (int) Math.max(1, Math.ceilDiv(rotateBytes, payload));
        CudaGpuMemory gpu = device.gpu;
        LinearFormat format = LinearFormats.of(object.getFirst());
        ExecutionPlan.Kind kind = format.kind(ExecutionPlan.Kind.Q3_LINEAR);
        int[] ends = object.size() > 1 ? segmentEnds(sources.path(option), object) : null;
        int maxRows = rows.stream().mapToInt(Integer::intValue).max().orElseThrow();
        List<List<TensorHandle>> weights = new ArrayList<>();
        List<Long> buffers = new ArrayList<>();
        try {
            for (int copy = 0; copy < copies; copy++) {
                List<TensorHandle> handles = new ArrayList<>();
                weights.add(handles);
                for (TensorDescriptor tensor : object)
                    handles.add(TensorLoader.load(sources.path(option), tensor, gpu));
            }
            long input = gpu.allocate((long) maxRows * shape.in() * Short.BYTES);
            buffers.add(input);
            long output = gpu.allocate((long) maxRows * shape.out() * Short.BYTES);
            buffers.add(output);
            fill(gpu, input, (long) maxRows * shape.in());
            Map<Integer, Double> result = new LinkedHashMap<>();
            for (int count : rows) {
                boolean multiRow = count > 1;
                ScratchUse use = format.scratchUse(kind, multiRow);
                long scratchBytes = use == null
                        ? 0
                        : format.scratchBytes(gpu, use, count, shape.in(), shape.out(), weights.getFirst());
                long scratch = 0;
                if (scratchBytes > 0) {
                    scratch = gpu.allocate(scratchBytes);
                    buffers.add(scratch);
                }
                boolean verify = count > 1 && count <= VERIFY_MAX_ROWS;
                List<LinearLaunch> launches = new ArrayList<>();
                for (List<TensorHandle> handles : weights)
                    launches.add(new LinearLaunch(
                            kind,
                            handles,
                            input,
                            output,
                            count,
                            shape.in(),
                            shape.out(),
                            scratch,
                            scratchBytes,
                            verify,
                            false,
                            ends));
                for (LinearLaunch launch : launches) format.linear(gpu, launch); // warmup: every copy once
                device.synchronize();
                result.put(count, gpuMicros(device, gpu, format, launches, repeats));
                if (scratch != 0) {
                    gpu.free(scratch);
                    buffers.remove(scratch);
                }
            }
            return result;
        } finally {
            for (long address : buffers) gpu.free(address);
            for (List<TensorHandle> handles : weights)
                for (TensorHandle handle : handles) gpu.free(handle.deviceAddress());
        }
    }

    /// The device time of one launch, `repeats` launches round-robin over `launches`: the stream first waits on a
    /// host-mapped flag while every launch is queued, then runs them back to back and raises a second flag, so the
    /// time between releasing the first and seeing the second is the device's alone. The engine runs decode steps
    /// as captured graphs, which launch with no host cost between kernels; wall clock around the launch calls would
    /// add the host's per-launch cost instead.
    private static double gpuMicros(
            Device device, CudaGpuMemory gpu, LinearFormat format, List<LinearLaunch> launches, int repeats) {
        long flags = gpu.allocateHostWeights(FLAG_BYTES);
        try {
            MemorySegment host = MemorySegment.ofAddress(flags).reinterpret(FLAG_BYTES);
            FLAG.setVolatile(host, 0L, 0);
            FLAG.setVolatile(host, (long) Integer.BYTES, 0);
            long device0 = gpu.hostWeightsDeviceAddress(flags);
            gpu.waitHostFlag(device0, 1);
            for (int repeat = 0; repeat < repeats; repeat++) format.linear(gpu, launches.get(repeat % launches.size()));
            gpu.writeHostFlag(device0 + Integer.BYTES, 1);
            long start = System.nanoTime();
            FLAG.setVolatile(host, 0L, 1);
            while ((int) FLAG.getVolatile(host, (long) Integer.BYTES) != 1) Thread.onSpinWait();
            long elapsed = System.nanoTime() - start;
            device.synchronize();
            return elapsed / 1e3 / repeats;
        } finally {
            gpu.freeHostWeights(flags);
        }
    }

    /// The host-known segment ends in the `.segments` companion of `object` (trellis).
    private static int[] segmentEnds(Path artifact, List<TensorDescriptor> object) throws IOException {
        TensorDescriptor segments = object.getLast();
        if (!segments.name().endsWith(".segments")) return null;
        try (Arena arena = Arena.ofConfined()) {
            return TrellisLayout.readEnds(TensorDataReader.read(artifact, segments, arena));
        }
    }

    private static void fill(CudaGpuMemory gpu, long address, long elements) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(elements * Short.BYTES, Short.BYTES);
            for (long i = 0; i < elements; i++) host.set(SHORT, i * Short.BYTES, INPUT_VALUE);
            gpu.copyHostToDevice(address, host, elements * Short.BYTES);
        }
    }
}
