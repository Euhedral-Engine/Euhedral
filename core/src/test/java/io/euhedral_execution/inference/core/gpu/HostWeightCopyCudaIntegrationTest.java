package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/// Host weight arenas are page-locked, and the asynchronous host copies accept only page-locked memory of the binding:
/// a pageable range would make the driver copy synchronously on the submitting worker.
class HostWeightCopyCudaIntegrationTest {

    private static final int PAGEABLE = 0;
    private static final int PAGE_LOCKED = 1;
    private static final long ARENA_BYTES = 8L << 20;
    private static final int BYTES = 4096;

    private static Path library() {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(libraryPath), "native CUDA library is missing: " + libraryPath);
        return libraryPath;
    }

    @Test
    void hostWeightArenasArePageLockedFromEndToEnd() {
        try (CudaGpuMemory gpu = new CudaGpuMemory(library());
                Arena pageable = Arena.ofConfined()) {
            long arena = gpu.allocateHostWeights(ARENA_BYTES);
            try {
                assertEquals(PAGE_LOCKED, gpu.memoryType(arena));
                assertEquals(PAGE_LOCKED, gpu.memoryType(arena + ARENA_BYTES - 1));
            } finally {
                gpu.freeHostWeights(arena);
            }
            assertEquals(PAGEABLE, gpu.memoryType(pageable.allocate(BYTES, 4096).address()));
        }
    }

    @Test
    void copiesHostWeightsInsideTheirArena() {
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            long arena = gpu.allocateHostWeights(ARENA_BYTES);
            long device = gpu.allocate(BYTES);
            try (Arena host = Arena.ofConfined()) {
                long source = arena + ARENA_BYTES - BYTES;
                MemorySegment weights = MemorySegment.ofAddress(source).reinterpret(BYTES);
                for (int i = 0; i < BYTES; i++) weights.set(ValueLayout.JAVA_BYTE, i, (byte) (i * 31 + 7));
                gpu.copyHostWeightsToDevice(device, source, BYTES);
                MemorySegment back = host.allocate(BYTES, 1);
                gpu.copyDeviceToHost(back, device, BYTES);
                assertArrayEquals(weights.toArray(ValueLayout.JAVA_BYTE), back.toArray(ValueLayout.JAVA_BYTE));
            } finally {
                gpu.free(device);
                gpu.freeHostWeights(arena);
            }
        }
    }

    @Test
    void refusesPageableAndFreedSources() {
        try (CudaGpuMemory gpu = new CudaGpuMemory(library());
                Arena pageable = Arena.ofConfined()) {
            long device = gpu.allocate(BYTES);
            try {
                long heap = pageable.allocate(BYTES, 4096).address();
                assertThrows(IllegalArgumentException.class, () -> gpu.copyHostWeightsToDevice(device, heap, BYTES));
                assertThrows(IllegalArgumentException.class, () -> gpu.copyHostToDeviceAsync(device, heap, BYTES));
                assertThrows(IllegalArgumentException.class, () -> gpu.copyDeviceToHostAsync(heap, device, BYTES));

                long arena = gpu.allocateHostWeights(ARENA_BYTES);
                // A range that runs past the arena's end reads memory the arena does not pin.
                assertThrows(
                        IllegalArgumentException.class,
                        () -> gpu.copyHostWeightsToDevice(device, arena + ARENA_BYTES - BYTES / 2, BYTES));
                gpu.freeHostWeights(arena);
                assertThrows(IllegalArgumentException.class, () -> gpu.copyHostWeightsToDevice(device, arena, BYTES));
            } finally {
                gpu.free(device);
            }
        }
    }

    @Test
    void uploadBuffersRemainValidSources() {
        try (CudaGpuMemory gpu = new CudaGpuMemory(library())) {
            long device = gpu.allocate(BYTES);
            try (ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(BYTES)) {
                gpu.copyHostToDeviceAsync(device, upload.segment().address(), BYTES);
            } finally {
                gpu.free(device);
            }
        }
    }
}
