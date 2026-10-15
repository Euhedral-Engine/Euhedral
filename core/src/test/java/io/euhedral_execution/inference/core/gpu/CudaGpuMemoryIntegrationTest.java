package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class CudaGpuMemoryIntegrationTest {

    @Test
    void copiesDeterministicBytesThroughCudaMemory() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertTrue(Files.isRegularFile(libraryPath), "native CUDA library is missing: " + libraryPath);

        byte[] expected = new byte[64];
        for (int i = 0; i < expected.length; i++) {
            expected[i] = (byte) (i * 37 + 11);
        }

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate(expected.length, 1);
            MemorySegment destination = arena.allocate(expected.length, 1);
            source.asByteBuffer().put(expected);
            long device = gpu.allocate(expected.length);
            try {
                gpu.copyHostToDevice(device, source, expected.length);
                gpu.copyDeviceToHost(destination, device, expected.length);

                ByteBuffer result = destination.asByteBuffer();
                byte[] actual = new byte[expected.length];
                result.get(actual);
                assertArrayEquals(expected, actual);
            } finally {
                gpu.free(device);
            }
        }
    }

    @Test
    void releasedPinnedBuffersAreReusedInsteadOfFreed() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            long readbackBytes = 3L << 20;
            ExecutionGpu.ReadbackBuffer readback = gpu.allocateReadbackBuffer(readbackBytes);
            long readbackAddress = readback.segment().address();
            readback.close();
            ExecutionGpu.ReadbackBuffer again = gpu.allocateReadbackBuffer(readbackBytes);
            assertEquals(readbackAddress, again.segment().address(), "a released readback buffer serves the next one");
            again.close();

            long uploadBytes = 1L << 20;
            ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(uploadBytes);
            long uploadAddress = upload.segment().address();
            upload.close();
            ExecutionGpu.UploadBuffer next = gpu.allocateUploadBuffer(uploadBytes);
            assertEquals(uploadAddress, next.segment().address(), "a released upload buffer serves the next one");
            ExecutionGpu.UploadBuffer concurrent = gpu.allocateUploadBuffer(uploadBytes);
            assertTrue(
                    concurrent.segment().address() != next.segment().address(),
                    "a buffer in use is not handed out twice");
            next.close();
            concurrent.close();
        }
    }

    /// Pinned buffers come in power-of-two size classes: a released one serves any request of its class;
    /// releasing twice pools it once; a pooled buffer is no valid copy range; the pool keeps at most its cap
    /// and sets the rest aside for a caller's thread to free.
    @Test
    void thePinnedPoolKeepsSizeClassesUpToItsCap() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        assertEquals(1L << 20, CudaGpuMemory.pinnedCapacity((1L << 20) - 7));
        assertEquals(2L << 20, CudaGpuMemory.pinnedCapacity((1L << 20) + 1));
        assertEquals(64L << 10, CudaGpuMemory.pinnedCapacity(100));

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            ExecutionGpu.UploadBuffer upload = gpu.allocateUploadBuffer(900_000);
            long address = upload.segment().address();
            upload.close();
            upload.close();
            assertEquals(1L << 20, gpu.pinnedPoolBytes(), "a buffer released twice is pooled once");
            long device = gpu.allocate(4096);
            try {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> gpu.copyHostToDeviceAsync(device, address, 4096),
                        "a pooled buffer is no longer a copy range");
            } finally {
                gpu.free(device);
            }
            ExecutionGpu.UploadBuffer first = gpu.allocateUploadBuffer(1_000_000);
            ExecutionGpu.UploadBuffer second = gpu.allocateUploadBuffer(1_000_000);
            assertEquals(address, first.segment().address(), "a request of the same class reuses the buffer");
            assertEquals(0, gpu.pinnedPoolBytes());
            first.close();
            second.close();
            assertEquals(2L << 20, gpu.pinnedPoolBytes());

            ExecutionGpu.ReadbackBuffer large = gpu.allocateReadbackBuffer(CudaGpuMemory.PINNED_POOL_BYTES);
            assertEquals(CudaGpuMemory.PINNED_POOL_BYTES, large.segment().byteSize());
            large.close();
            assertEquals(1, gpu.pinnedPoolOverflows(), "a buffer the pool has no room for is set aside");
            assertEquals(2L << 20, gpu.pinnedPoolBytes());
            // Freed on a caller's thread, which may wait for the device.
            gpu.trimPinnedPool();
        }
    }

    @Test
    void validatesJavaArguments() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));

        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);
                Arena arena = Arena.ofConfined()) {
            MemorySegment host = arena.allocate(8, 1);

            assertThrows(IllegalArgumentException.class, () -> gpu.allocate(0));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyHostToDevice(0, host, 1));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyDeviceToHost(host, 0, 1));
            assertThrows(IllegalArgumentException.class, () -> gpu.copyHostToDevice(1, host, 9));
        }
    }

    @Test
    void rejectsUseAfterClose() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        CudaGpuMemory gpu = new CudaGpuMemory(libraryPath);

        gpu.close();

        assertThrows(IllegalStateException.class, () -> gpu.allocate(1));
        assertThrows(IllegalStateException.class, () -> gpu.free(0));
        gpu.close();
    }
}
