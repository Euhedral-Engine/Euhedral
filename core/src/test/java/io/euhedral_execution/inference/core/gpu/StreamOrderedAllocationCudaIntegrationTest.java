package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Stream-ordered allocation: memory taken and returned in a lane's order, without blocking the host.
class StreamOrderedAllocationCudaIntegrationTest {

    private static Path library() {
        String selected = System.getProperty("euhedral.cuda.library");
        assumeTrue(selected != null && Files.isRegularFile(Path.of(selected)), "a CUDA library is required");
        return Path.of(selected);
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void anAllocationOnALaneIsWrittenReadAndFreedInStreamOrder() {
        try (var gpu = new CudaGpuMemory(library());
                var arena = Arena.ofConfined()) {
            GpuStream stream = gpu.openStream();
            try {
                long before = gpu.allocatedBytes();
                int bytes = 1 << 20;
                byte[] pattern = new byte[bytes];
                for (int i = 0; i < bytes; i++) pattern[i] = (byte) (i * 31 + 7);
                var upload = gpu.allocateUploadBuffer(bytes);
                upload.segment().copyFrom(java.lang.foreign.MemorySegment.ofArray(pattern));
                var readback = gpu.allocateReadbackBuffer(bytes);
                stream.submit(
                        () -> {
                            long address = gpu.allocateAsync(bytes);
                            gpu.copyUploadToDevice(address, upload);
                            gpu.copyDeviceToReadback(readback, address, bytes);
                            gpu.freeAsync(address);
                        },
                        false);
                stream.synchronize();
                assertArrayEquals(pattern, readback.segment().toArray(ValueLayout.JAVA_BYTE));
                assertEquals(before, gpu.allocatedBytes(), "the stream-ordered allocation was returned");
                upload.close();
                readback.close();
            } finally {
                stream.close();
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void freeingOnAStreamDoesNotBlockTheHost() {
        try (var gpu = new CudaGpuMemory(library())) {
            GpuStream stream = gpu.openStream();
            long large = gpu.allocate(1L << 30);
            try {
                long[] elapsed = new long[1];
                stream.submit(
                        () -> {
                            long address = gpu.allocateAsync(1 << 20);
                            for (int pass = 0; pass < 8; pass++) gpu.zeroDeviceMemory(large, 1L << 30);
                            long started = System.nanoTime();
                            gpu.freeAsync(address);
                            elapsed[0] = System.nanoTime() - started;
                        },
                        false);
                stream.synchronize();
                assertTrue(
                        elapsed[0] < TimeUnit.MILLISECONDS.toNanos(5),
                        "freeAsync blocked the host for " + elapsed[0] / 1_000 + " us");
            } finally {
                stream.close();
                gpu.free(large);
            }
        }
    }
}
