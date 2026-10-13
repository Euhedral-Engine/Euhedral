package io.euhedral_execution.inference.quantization.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gpu.Device;
import io.euhedral_execution.inference.quantization.lattice.GpuCompletions;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/// The frame pipeline (sequences on their own streams, frames on the lattice) gives the hidden states of the
/// sequential forward bit for bit, for a GDN layer and an attention layer, in batches of one sequence (chained
/// batches) and of two (sequences overlapping on two streams), and observes as many rows.
class ForwardPipelineCudaIntegrationTest {

    @Test
    void pipelineEqualsTheSequentialForward() throws Exception {
        String checkpoint = System.getProperty("euhedral.quantization.checkpoint", "");
        Path fixture = Path.of(System.getProperty("euhedral.quantization.forward-fixture", "build/forward-fixture"));
        assumeTrue(!checkpoint.isBlank() && Files.isDirectory(fixture), "needs the checkpoint and the forward fixture");
        int sequences = 2, length = 256, hidden = 5120, rows = sequences * length;
        long bytes = 2L * rows * hidden;
        byte[] input = Files.readAllBytes(fixture.resolve("input.npy"));
        int header = 10 + Short.toUnsignedInt((short) ((input[8] & 0xff) | (input[9] & 0xff) << 8));
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        try (Safetensors store = Safetensors.open(Path.of(checkpoint));
                Device device = new Device(library);
                CalibrationLattice lattice = CalibrationLattice.start(4, Set.of(8, 9));
                Arena arena = Arena.ofConfined()) {
            var objects = new Qwen38Objects(store, Path.of(checkpoint));
            GpuCompletions completions = new GpuCompletions(device.cuda);
            lattice.attach(completions);
            long pinned = device.gpu.allocateHostWeights(bytes);
            MemorySegment states = MemorySegment.ofAddress(pinned).reinterpret(bytes);
            MemorySegment expected = arena.allocate(bytes);
            long sequential = device.gpu.allocate(bytes), batchStates = device.gpu.allocate(bytes);
            try (LayerForward forward = new LayerForward(device, objects.config, rows);
                    ForwardPipeline pipeline =
                            new ForwardPipeline(device, forward, lattice.lake(), completions, sequences)) {
                for (int index : new int[] {0, 3})
                    for (int batch = sequences; batch >= 1; batch--) {
                        try (LayerWeights layer = LayerWeights.layer(device.gpu, objects, index)) {
                            MemorySegment.copy(MemorySegment.ofArray(input), header, expected, 0, bytes);
                            device.gpu.copyHostToDevice(sequential, expected, bytes);
                            AtomicInteger sequentialRows = new AtomicInteger();
                            // The reference runs in batches of the same size: cuBLAS may sum other shapes differently.
                            for (int first = 0; first < sequences; first += batch) {
                                forward.forward(
                                        layer,
                                        sequential + 2L * first * length * hidden,
                                        Math.min(batch, sequences - first),
                                        length,
                                        (n, a, count, c) -> sequentialRows.addAndGet(count));
                            }
                            device.synchronize();
                            device.gpu.copyDeviceToHost(expected, sequential, bytes);
                            MemorySegment.copy(MemorySegment.ofArray(input), header, states, 0, bytes);
                            AtomicInteger pipelineRows = new AtomicInteger();
                            pipeline.run(
                                            layer,
                                            pinned,
                                            batchStates,
                                            sequences,
                                            length,
                                            batch,
                                            (n, a, count, c) -> pipelineRows.addAndGet(count),
                                            true)
                                    .join();
                            long differing = 0;
                            double difference = 0, norm = 0;
                            for (long i = 0; i < bytes / 2; i++) {
                                short got = states.getAtIndex(ValueLayout.JAVA_SHORT, i);
                                short want = expected.getAtIndex(ValueLayout.JAVA_SHORT, i);
                                if (got != want) differing++;
                                double x = Float.intBitsToFloat(got << 16), y = Float.intBitsToFloat(want << 16);
                                difference += (x - y) * (x - y);
                                norm += y * y;
                            }
                            double relative = Math.sqrt(difference / norm);
                            System.out.printf(
                                    "layer %d, batches of %d: %d of %d values differ, relative %.2e%n",
                                    index, batch, differing, bytes / 2, relative);
                            assertEquals(0, differing, "layer " + index + ", batches of " + batch);
                            assertEquals(sequentialRows.get(), pipelineRows.get(), "observed rows, layer " + index);
                        }
                    }
            } finally {
                device.gpu.free(sequential);
                device.gpu.free(batchStates);
                device.gpu.freeHostWeights(pinned);
            }
        }
    }
}
