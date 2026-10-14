package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Stream waits on and writes of host-mapped flags: a copy queued behind a wait runs only once the host raises the
/// flag, the write after it lowers the flag, and both repeat in a captured graph.
class StreamFlagCudaIntegrationTest {

    private static Path library() {
        String selected = System.getProperty("euhedral.cuda.library");
        assumeTrue(selected != null && Files.isRegularFile(Path.of(selected)), "a CUDA library is required");
        return Path.of(selected);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aCapturedCopyWaitsForItsHostFlagAndLowersIt() throws Exception {
        try (var gpu = new CudaGpuMemory(library())) {
            assertTrue(gpu.streamMemoryOps());
            long arena = gpu.allocateHostWeights(8192);
            MemorySegment host = MemorySegment.ofAddress(arena).reinterpret(8192);
            long flag = arena;
            long source = arena + 4096;
            for (int i = 0; i < 64; i++) host.set(ValueLayout.JAVA_INT, 4096 + 4L * i, i + 1);
            long flagDevice = gpu.hostWeightsDeviceAddress(flag);
            long target = gpu.allocate(256);
            GpuStream capture = gpu.openStream();
            GpuStream run = gpu.openStream();
            try {
                capture.beginCapture();
                capture.submit(
                        () -> {
                            gpu.waitHostFlag(flagDevice, 1);
                            gpu.copyHostWeightsToDevice(target, source, 256);
                            gpu.writeHostFlag(flagDevice, 0);
                        },
                        false);
                long graph = capture.endCapture();
                assertNotEquals(0L, graph);
                for (int replay = 0; replay < 2; replay++) {
                    gpu.zeroDeviceMemory(target, 256);
                    assertTrue(run.launchGraph(graph));
                    Thread.sleep(200);
                    assertArrayEquals(new int[64], download(gpu, target), "the copy waits for its flag");
                    VarHandle.fullFence();
                    host.set(ValueLayout.JAVA_INT, 0, 1);
                    VarHandle.fullFence();
                    run.synchronize();
                    assertArrayEquals(IntStream.rangeClosed(1, 64).toArray(), download(gpu, target));
                    VarHandle.fullFence();
                    assertEquals(0, host.get(ValueLayout.JAVA_INT, 0), "the graph lowers the flag after its copy");
                }
                capture.destroyGraph(graph);
            } finally {
                run.close();
                capture.close();
                gpu.free(target);
                gpu.freeHostWeights(arena);
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void aFlagWaitNeedsASelectedStream() {
        try (var gpu = new CudaGpuMemory(library())) {
            long arena = gpu.allocateHostWeights(4096);
            try {
                long flagDevice = gpu.hostWeightsDeviceAddress(arena);
                assertThrows(GpuMemoryException.class, () -> gpu.waitHostFlag(flagDevice, 1));
                assertThrows(GpuMemoryException.class, () -> gpu.writeHostFlag(flagDevice, 0));
            } finally {
                gpu.freeHostWeights(arena);
            }
        }
    }

    private static int[] download(CudaGpuMemory gpu, long address) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment bytes = arena.allocate(256);
            gpu.copyDeviceToHost(bytes, address, 256);
            return bytes.toArray(ValueLayout.JAVA_INT);
        }
    }
}
