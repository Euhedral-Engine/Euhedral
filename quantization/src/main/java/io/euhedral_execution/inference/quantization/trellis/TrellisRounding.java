package io.euhedral_execution.inference.quantization.trellis;

import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.gpu.Cuda;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.model.LayerWeights;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/// The trellis rounding of a decoder layer inside the calibration: every quantized linear of the layer is a lattice
/// frame on its own stream (largest first, [#lanes] at once, sized from the free device memory) that gathers the
/// linear's BF16 rows out of
/// the layer's fused objects, quantizes them under the Hessian of the object it reads, and keeps the result. When
/// every linear is done the decoded rows replace the linear's rows in the objects (the propagation then runs through
/// the quantized weights) and the layer is written to the checkpoint. Also quantizes the MTP linears, uncalibrated.
///
/// The Hessians arrive as `observeAll` accumulates them (sums of `xᵀx` over `count` rows); they are completed and
/// divided by `count` in place here, as the quantizer expects.
public final class TrellisRounding implements AutoCloseable {

    /// Seed of the MTP linears' signs (a decoder layer's is its index).
    static final long MTP_SEED = 64;

    private static final String[] MTP_LINEARS = {
        "mtp.fc",
        "mtp.layers.0.self_attn.q_proj",
        "mtp.layers.0.self_attn.k_proj",
        "mtp.layers.0.self_attn.v_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.gate_proj",
        "mtp.layers.0.mlp.up_proj",
        "mtp.layers.0.mlp.down_proj"
    };

    private final Device device;
    private final CalibrationLattice lattice;
    private final Qwen38Objects checkpoint;
    private final Exl3Writer writer;
    private final Map<String, Double> bits;
    private final Consumer<String> log;
    private final Map<Long, TrellisQuantizer> streamQuantizers = new ConcurrentHashMap<>();
    private final Cuda.Function scale;
    private final Integer fixedLanes;
    private final long viterbiScratchBytes;
    private final long reserveBytes;

    public TrellisRounding(
            Device device,
            CalibrationLattice lattice,
            Qwen38Objects checkpoint,
            Exl3Writer writer,
            Map<String, Double> bits,
            Consumer<String> log,
            Integer fixedLanes,
            long viterbiScratchMib,
            long reserveMib) {
        this.fixedLanes = fixedLanes;
        this.viterbiScratchBytes = viterbiScratchMib << 20;
        this.reserveBytes = reserveMib << 20;
        this.device = device;
        this.lattice = lattice;
        this.checkpoint = checkpoint;
        this.writer = writer;
        this.bits = bits;
        this.log = log;
        this.scale = device.cuda
                .compileResource("trellis.cu", List.of("scale_scalar"))
                .getFirst();
        log.accept("trellis rounding: "
                + (fixedLanes != null ? fixedLanes + " lanes" : "lanes sized from free device memory")
                + ", Viterbi scratch " + viterbiScratchMib + " MiB, memory reserve " + reserveMib
                + " MiB, output scales on for every linear");
    }

    @Override
    public void close() {
        for (TrellisQuantizer quantizer : streamQuantizers.values()) quantizer.close();
        streamQuantizers.clear();
    }

    private TrellisQuantizer quantizer(long stream) {
        return streamQuantizers.computeIfAbsent(
                stream, s -> new TrellisQuantizer(device.cuda, device.gpu, s, viterbiScratchBytes));
    }

    /// Quantizes the layer's linears under `hessians` (by object name, accumulated over `count` rows; freed here),
    /// swaps the decoded rows into `layer` and writes layer `index`. Completes on a lattice worker.
    public CompletableFuture<Void> round(int index, LayerWeights layer, Map<String, Long> hessians, long count) {
        device.select(device.stream);
        var unique = new HashSet<Long>();
        for (String name : layer.objects.keySet()) {
            long h = hessians.get(name);
            if (unique.add(h)) {
                int k = (int) layer.shapes.get(name)[1];
                device.symmetrize(h, k);
                device.cuda.launch(
                        scale,
                        (int) Math.min(4096, ((long) k * k + 255) / 256),
                        1,
                        256,
                        0,
                        device.stream,
                        new Cuda.Arguments()
                                .pointer(h)
                                .pointer((long) k * k)
                                .real((float) count)
                                .integer(1));
            }
        }
        device.synchronize(); // the Hessians are complete and normalised before other streams read them

        List<Linears.Linear> linears = new ArrayList<>();
        for (Linears.Linear linear : Linears.layer(checkpoint, index)) if (linear.quantized()) linears.add(linear);
        linears.sort(Comparator.comparingLong((Linears.Linear l) -> -(long) l.rows() * l.columns()));
        Map<String, TrellisQuantizer.Result> results = new ConcurrentHashMap<>();
        var next = new AtomicInteger();
        var failure = new AtomicReference<Throwable>();
        List<CompletableFuture<Void>> lanes = new ArrayList<>();
        int width = lanes(index, linears);
        for (int lane = 0; lane < width; lane++)
            lanes.add(lane(index, linears, next, layer, hessians, count, results, failure));
        return CompletableFuture.allOf(lanes.toArray(CompletableFuture[]::new)).thenRun(() -> {
            device.select(device.stream);
            try {
                if (failure.get() != null) throw new IllegalStateException("rounding failed", failure.get());
                for (Linears.Linear linear : linears) {
                    TrellisQuantizer.Result result = results.get(linear.tensor());
                    long rowBytes = 2L * linear.columns();
                    for (Linears.Placement p : Linears.placements(checkpoint, index, linear.tensor())) {
                        device.gpu.copyDeviceToDevice(
                                layer.objects.get(p.object()) + p.objectRow() * rowBytes,
                                result.decoded() + p.linearRow() * rowBytes,
                                p.rows() * rowBytes);
                    }
                }
                device.synchronize();
                writer.layer(index, results, bits);
            } catch (IOException error) {
                throw new UncheckedIOException(error);
            } finally {
                for (var result : results.values()) device.gpu.free(result.decoded());
                var freed = new HashSet<Long>();
                for (long h : hessians.values()) if (freed.add(h)) device.gpu.free(h);
            }
        });
    }

    /// Peak device bytes of quantizing `linear` beyond its Hessian: the BF16 weight (`2kn`), the quantizer's working
    /// set (`16kn + 4k²`: four FP32 `[k, n]` matrices and the `[k, k]` factor or refit system) and its Viterbi
    /// scratch.
    long peakBytes(Linears.Linear linear) {
        long k = linear.columns(), n = linear.rows();
        return 18 * k * n + 4 * k * k + viterbiScratchBytes;
    }

    /// How many linears (largest first) run at once: the `--rounding-lanes` value when given, else as many as fit in
    /// the free device memory less the reserve and the BF16 decoded weights every linear of the layer keeps until
    /// the swap, by the estimated peak of the largest ones (at least 1). Logs the free memory and the choice.
    private int lanes(int index, List<Linears.Linear> linears) {
        long free = device.cuda.freeMemory();
        long held = 0;
        for (Linears.Linear linear : linears) held += 2L * linear.rows() * linear.columns();
        long budget = free - held - reserveBytes;
        int idle = streamQuantizers.size(); // their scratch is already allocated and reused by the first lanes
        int width = 0;
        long used = 0;
        if (fixedLanes != null) width = Math.min(fixedLanes, linears.size());
        else
            while (width < linears.size()) {
                long need = peakBytes(linears.get(width)) - (width < idle ? viterbiScratchBytes : 0);
                if (used + need > budget) break;
                used += need;
                width++;
            }
        width = Math.max(1, width);
        log.accept(String.format(
                "layer %d: %d MiB free, %d MiB held for decoded weights, %d MiB reserve, %d rounding lanes%s",
                index, free >> 20, held >> 20, reserveBytes >> 20, width, fixedLanes != null ? " (set)" : ""));
        return width;
    }

    private CompletableFuture<Void> lane(
            int index,
            List<Linears.Linear> linears,
            AtomicInteger next,
            LayerWeights layer,
            Map<String, Long> hessians,
            long count,
            Map<String, TrellisQuantizer.Result> results,
            AtomicReference<Throwable> failure) {
        int at = next.getAndIncrement();
        if (at >= linears.size() || failure.get() != null) return CompletableFuture.completedFuture(null);
        Linears.Linear linear = linears.get(at);
        return lattice.<Void>submit(() -> {
                    try {
                        long stream = device.bindOwnStream();
                        results.put(
                                linear.tensor(),
                                roundLinear(index, linear, layer, hessians, count, quantizer(stream), stream));
                    } catch (Throwable error) {
                        failure.compareAndSet(null, error);
                    }
                    return null;
                })
                .thenCompose(ignored -> lane(index, linears, next, layer, hessians, count, results, failure));
    }

    private TrellisQuantizer.Result roundLinear(
            int index,
            Linears.Linear linear,
            LayerWeights layer,
            Map<String, Long> hessians,
            long count,
            TrellisQuantizer quantizer,
            long stream) {
        long started = System.nanoTime();
        double linearBits = bits.get(linear.tensor());
        long rowBytes = 2L * linear.columns();
        long weight = device.gpu.allocate(rowBytes * linear.rows());
        try {
            for (Linears.Placement p : Linears.placements(checkpoint, index, linear.tensor())) {
                device.gpu.copyDeviceToDevice(
                        weight + p.linearRow() * rowBytes,
                        layer.objects.get(p.object()) + p.objectRow() * rowBytes,
                        p.rows() * rowBytes);
            }
            device.cuda.synchronize(stream);
            Long hessian = hessians.get(linear.hessianObject());
            if (hessian == null) throw new IllegalStateException("no Hessian for " + linear.hessianObject());
            TrellisQuantizer.Result result = quantizer.quantize(
                    weight, linear.rows(), linear.columns(), hessian, count, linearBits, index, true);
            log.accept(String.format(
                    "%s %s bits, proxy %.5f, refit failures %d, %.1f s",
                    linear.tensor(),
                    linearBits,
                    result.proxyError(),
                    result.refitFailures(),
                    (System.nanoTime() - started) / 1e9));
            return result;
        } finally {
            device.gpu.free(weight);
        }
    }

    /// Quantizes the eight MTP linears of the source checkpoint without a Hessian at [Exl3Writer#MTP_BITS] and
    /// writes `mtp.safetensors`; completes on a lattice worker.
    public CompletableFuture<Void> mtp() {
        return lattice.<Void>submit(() -> {
            long stream = device.bindOwnStream();
            TrellisQuantizer quantizer = quantizer(stream);
            Map<String, TrellisQuantizer.Result> results = new LinkedHashMap<>();
            try {
                for (String linear : MTP_LINEARS) {
                    long started = System.nanoTime();
                    var tensor = checkpoint.store().tensor(linear + ".weight");
                    long weight = device.gpu.allocate(tensor.bytes());
                    try (Arena arena = Arena.ofConfined()) {
                        MemorySegment host = arena.allocate(tensor.bytes(), 16);
                        checkpoint.readTensor(linear + ".weight", host);
                        device.gpu.copyHostToDevice(weight, host, tensor.bytes());
                        var result = quantizer.quantize(
                                weight,
                                (int) tensor.shape()[0],
                                (int) tensor.shape()[1],
                                0,
                                0,
                                Exl3Writer.MTP_BITS,
                                MTP_SEED,
                                true);
                        device.gpu.free(result.decoded());
                        results.put(linear, result);
                        log.accept(String.format(
                                "%s %d bits, proxy %.5f, refit failures %d, %.1f s",
                                linear,
                                Exl3Writer.MTP_BITS,
                                result.proxyError(),
                                result.refitFailures(),
                                (System.nanoTime() - started) / 1e9));
                    } finally {
                        device.gpu.free(weight);
                    }
                }
                writer.mtp(results);
            } catch (IOException error) {
                throw new UncheckedIOException(error);
            }
            return null;
        });
    }
}
