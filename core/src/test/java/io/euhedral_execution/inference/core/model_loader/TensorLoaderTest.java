package io.euhedral_execution.inference.core.model_loader;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.gpu.GpuMemory;
import io.euhedral_execution.inference.core.model_loader.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorHandle;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TensorLoaderTest {

    @TempDir
    Path tempDirectory;

    @Test
    void uploadsExactPayloadAndReturnsMetadata() throws Exception {
        byte[] payload = {(byte) 0x00, (byte) 0x7F, (byte) 0x80, (byte) 0xFF};
        TensorDescriptor descriptor = descriptor(6, payload.length);
        Path path = writePayload(descriptor.dataOffset(), payload);
        FakeGpuMemory gpu = new FakeGpuMemory();

        TensorHandle handle = TensorLoader.load(path, descriptor, gpu);

        assertEquals(1, gpu.allocateCalls);
        assertEquals(payload.length, gpu.allocatedByteSize);
        assertEquals(gpu.deviceAddress, gpu.copyDestination);
        assertEquals(payload.length, gpu.copyByteSize);
        assertArrayEquals(payload, gpu.copiedPayload);
        assertEquals(descriptor.name(), handle.name());
        assertArrayEquals(descriptor.shape(), handle.shape());
        assertEquals(descriptor.dataType(), handle.dataType());
        assertEquals(descriptor.format(), handle.format());
        assertEquals(gpu.deviceAddress, handle.deviceAddress());
        assertEquals(descriptor.byteSize(), handle.byteSize());
        assertEquals(0, gpu.freeCalls);
    }

    @Test
    void freesDeviceAllocationWhenUploadFails() throws Exception {
        byte[] payload = {1, 2, 3, 4};
        TensorDescriptor descriptor = descriptor(0, payload.length);
        Path path = writePayload(descriptor.dataOffset(), payload);
        FakeGpuMemory gpu = new FakeGpuMemory();
        gpu.failUpload = true;

        assertThrows(UploadFailure.class, () -> TensorLoader.load(path, descriptor, gpu));

        assertEquals(1, gpu.allocateCalls);
        assertEquals(1, gpu.freeCalls);
        assertEquals(gpu.deviceAddress, gpu.freedAddress);
    }

    @Test
    void doesNotAllocateWhenPayloadReadFails() throws Exception {
        TensorDescriptor descriptor = descriptor(3, 4);
        Path path = tempDirectory.resolve("truncated.edrl");
        Files.write(path, new byte[] {9, 8, 7});
        FakeGpuMemory gpu = new FakeGpuMemory();

        assertThrows(java.io.IOException.class, () -> TensorLoader.load(path, descriptor, gpu));

        assertEquals(0, gpu.allocateCalls);
        assertEquals(0, gpu.freeCalls);
    }

    private Path writePayload(long offset, byte[] payload) throws Exception {
        byte[] file = new byte[Math.toIntExact(offset) + payload.length];
        System.arraycopy(payload, 0, file, Math.toIntExact(offset), payload.length);
        Path path = tempDirectory.resolve("weights.edrl");
        Files.write(path, file);
        return path;
    }

    private static TensorDescriptor descriptor(long offset, long byteSize) {
        return new TensorDescriptor(
                "layer.weight",
                new long[] {2, 2},
                TensorDataType.FP16,
                WeightFormat.NVFP4,
                WeightLayout.CONTIGUOUS_LE_V1,
                offset,
                byteSize);
    }

    private static final class FakeGpuMemory implements GpuMemory {

        private final long deviceAddress = 0x1234L;
        private int allocateCalls;
        private int freeCalls;
        private long allocatedByteSize;
        private long copyDestination;
        private long copyByteSize;
        private long freedAddress;
        private byte[] copiedPayload;
        private boolean failUpload;

        @Override
        public long allocate(long byteSize) {
            allocateCalls++;
            allocatedByteSize = byteSize;
            return deviceAddress;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            copyDestination = destination;
            copyByteSize = byteSize;
            copiedPayload = source.toArray(ValueLayout.JAVA_BYTE);
            if (failUpload) {
                throw new UploadFailure();
            }
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void free(long address) {
            freeCalls++;
            freedAddress = address;
        }
    }

    private static final class UploadFailure extends RuntimeException {}
}
