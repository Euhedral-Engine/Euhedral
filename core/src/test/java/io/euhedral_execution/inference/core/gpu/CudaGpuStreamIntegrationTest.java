package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Quantum-owned CUDA streams: ordering, the driver-callback retirement boundary, and independence of
/// allocation and copies from work that is still pending on a stream.
class CudaGpuStreamIntegrationTest {

    private static Path library() {
        String selected = System.getProperty("euhedral.cuda.library");
        assumeTrue(selected != null && Files.isRegularFile(Path.of(selected)), "a CUDA library is required");
        return Path.of(selected);
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void streamsMustCloseBeforeTheGpu() {
        var gpu = new CudaGpuMemory(library());
        GpuStream first = gpu.openStream();
        GpuStream second = gpu.openStream();
        second.close();
        assertThrows(IllegalStateException.class, gpu::close, "an open quantum stream still owns device ordering");
        first.close();
        gpu.close();
        assertThrows(IllegalStateException.class, gpu::openStream);
    }

    @Test
    @Timeout(value = 20, unit = TimeUnit.SECONDS)
    void driverCallbackOnlyAnnouncesAndAnOrdinaryThreadConfirmsRetirement() throws Exception {
        try (var gpu = new CudaGpuMemory(library());
                var arena = Arena.ofConfined()) {
            GpuStream stream = gpu.openStream();
            long address = gpu.allocate(64);
            try {
                var announced = new CountDownLatch(1);
                var driverThread = new AtomicReference<Boolean>();
                var ticket = new AtomicLong();
                stream.submit(() -> gpu.zeroDeviceMemory(address, 64), false);
                long armed = stream.notifyRetired((retired, onDriver) -> {
                    driverThread.set(onDriver);
                    ticket.set(retired);
                    announced.countDown();
                });
                assertTrue(announced.await(10, TimeUnit.SECONDS), "CUDA did not announce the boundary");
                assertEquals(Boolean.TRUE, driverThread.get(), "the announcement arrives on the driver thread");
                assertEquals(armed, ticket.get());
                assertNull(stream.confirmRetired(armed), "every submitted operation retired");
                var bytes = arena.allocate(64);
                gpu.copyDeviceToHost(bytes, address, 64);
                assertArrayEquals(new byte[64], bytes.toArray(ValueLayout.JAVA_BYTE));
                for (int boundary = 0; boundary < 8; boundary++) {
                    var next = new CountDownLatch(1);
                    long nextTicket = stream.notifyRetired((retired, onDriver) -> next.countDown());
                    assertTrue(next.await(10, TimeUnit.SECONDS));
                    assertNull(stream.confirmRetired(nextTicket));
                }
                assertTrue(stream.confirmRetired(armed) instanceof IllegalStateException, "tickets confirm once");
            } finally {
                gpu.free(address);
                stream.close();
            }
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void allocationUploadAndDeviceCopiesDoNotWaitForAHeldRetirementCallback() throws Exception {
        try (var gpu = new CudaGpuMemory(library());
                var worker = Executors.newSingleThreadExecutor()) {
            GpuStream held = gpu.openStream();
            GpuStream other = gpu.openStream();
            var entered = new CountDownLatch(1);
            var release = new CountDownLatch(1);
            long source = gpu.allocate(64);
            long destination = gpu.allocate(64);
            var upload = gpu.allocateUploadBuffer(64);
            upload.segment().fill((byte) 0x5a);
            try {
                held.submit(() -> gpu.zeroDeviceMemory(source, 64), false);
                long ticket = held.notifyRetired((retired, onDriver) -> {
                    entered.countDown();
                    try {
                        if (!release.await(10, TimeUnit.SECONDS)) throw new AssertionError("callback gate timed out");
                    } catch (InterruptedException failure) {
                        Thread.currentThread().interrupt();
                        throw new AssertionError(failure);
                    }
                });
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                try {
                    long allocated = worker.submit(() -> gpu.allocate(64)).get(2, TimeUnit.SECONDS);
                    worker.submit(() -> other.submit(() -> gpu.copyUploadToDevice(destination, upload), false))
                            .get(2, TimeUnit.SECONDS);
                    worker.submit(() -> other.submit(() -> gpu.copyDeviceToDevice(destination, source, 64), false))
                            .get(2, TimeUnit.SECONDS);
                    gpu.free(allocated);
                } finally {
                    release.countDown();
                }
                other.synchronize();
                held.synchronize();
                assertNull(held.confirmRetired(ticket));
            } finally {
                upload.close();
                gpu.free(destination);
                gpu.free(source);
                held.close();
                other.close();
            }
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void oneStreamPreservesUploadAndCopyOrderingAcrossSubmittingThreads() throws Exception {
        try (var gpu = new CudaGpuMemory(library());
                var arena = Arena.ofConfined();
                var first = Executors.newSingleThreadExecutor();
                var second = Executors.newSingleThreadExecutor()) {
            GpuStream stream = gpu.openStream();
            long bytes = 8L << 20;
            var host = arena.allocate(bytes);
            host.fill((byte) 0x5a);
            var readback = arena.allocate(64);
            long source = gpu.allocate(bytes);
            long destination = gpu.allocate(bytes);
            try {
                for (int iteration = 0; iteration < 8; iteration++) {
                    // Successive stages of one quantum may run on different workers; stream order holds.
                    first.submit(() -> stream.submit(() -> gpu.zeroDeviceMemory(source, bytes), false))
                            .get(5, TimeUnit.SECONDS);
                    second.submit(() -> stream.submit(() -> gpu.copyHostToDevice(source, host, bytes), false))
                            .get(5, TimeUnit.SECONDS);
                    first.submit(() -> stream.submit(() -> gpu.copyDeviceToDevice(destination, source, bytes), false))
                            .get(5, TimeUnit.SECONDS);
                    stream.synchronize();
                    gpu.copyDeviceToHost(readback, destination + bytes - 64, 64);
                    for (byte value : readback.toArray(ValueLayout.JAVA_BYTE)) assertEquals((byte) 0x5a, value);
                }
            } finally {
                gpu.free(destination);
                gpu.free(source);
                stream.close();
            }
        }
    }

    @Test
    @Timeout(value = 15, unit = TimeUnit.SECONDS)
    void operationsWithoutASelectedStreamRemainSynchronousFixtures() {
        try (var gpu = new CudaGpuMemory(library());
                var arena = Arena.ofConfined()) {
            long address = gpu.allocate(64);
            try {
                var host = arena.allocate(64);
                host.fill((byte) 0x11);
                gpu.copyHostToDevice(address, host, 64);
                gpu.zeroDeviceMemory(address, 64);
                var readback = arena.allocate(64);
                readback.fill((byte) 0x7f);
                gpu.copyDeviceToHost(readback, address, 64);
                assertArrayEquals(new byte[64], readback.toArray(ValueLayout.JAVA_BYTE));
                assertNotEquals(0L, address);
            } finally {
                gpu.free(address);
            }
        }
    }
}
