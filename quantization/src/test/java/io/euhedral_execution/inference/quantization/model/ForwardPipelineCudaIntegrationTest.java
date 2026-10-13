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
                    for (int batch = 1; batch <= sequences; batch++) {
                        try (LayerWeights layer = LayerWeights.layer(device.gpu, objects, index)) {
                            MemorySegment.copy(MemorySegment.ofArray(input), header, expected, 0, bytes);
                            device.gpu.copyHostToDevice(sequential, expected, bytes);
                            AtomicInteger sequentialRows = new AtomicInteger();
                            forward.forward(
                                    layer,
                                    sequential,
                                    sequences,
                                    length,
                                    (n, a, count, c) -> sequentialRows.addAndGet(count));
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
                            for (long i = 0; i < bytes / 2; i++) {
                                if (states.getAtIndex(ValueLayout.JAVA_SHORT, i)
                                        != expected.getAtIndex(ValueLayout.JAVA_SHORT, i)) differing++;
                            }
                            System.out.printf(
                                    "layer %d, batches of %d: %d of %d values differ%n",
                                    index, batch, differing, bytes / 2);
                            assertEquals(0, differing, "layer " + index);
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
