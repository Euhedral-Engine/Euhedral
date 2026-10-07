package io.euhedral_execution.inference.core.gpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory.DeviceMemoryInfo;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.WeightLoader;
import io.euhedral_execution.inference.core.model.qwen38.loader.Weights;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

class CompactCudaResidencyIntegrationTest {

    private static final long EXPECTED_OBJECT_COUNT = 771;
    private static final Path DEFAULT_ARTIFACT = Path.of("/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl");

    @Test
    void reportsCudaDeviceMemory() throws Exception {
        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            CudaGpuMemory.DeviceMemoryInfo info = gpu.deviceMemoryInfo();

            assertTrue(info.freeBytes() > 0, "CUDA reported no free device memory");
            assertTrue(info.totalBytes() >= info.freeBytes(), "CUDA reported more free than total device memory");
        }
    }

    @Test
    void loadsAllCompactRuntimeObjectsAndRestoresVramAfterTeardown() throws Throwable {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact", DEFAULT_ARTIFACT.toString()));
        assertTrue(Files.isRegularFile(artifactPath), "compact EDRL artifact is missing: " + artifactPath);

        Artifact artifact = ArtifactReader.read(artifactPath);
        assertEquals(ArtifactHeader.COMPACT_VERSION, artifact.header().version(), "artifact is not compact EDRL");
        // A base load places the text model on the device: no vision tower, and the MTP layer and draft head
        // only come with speculative decoding.
        TensorDescriptor[] descriptors = java.util.Arrays.stream(artifact.tensors())
                .filter(descriptor -> !descriptor.name().startsWith("vision/")
                        && !descriptor.name().startsWith("mtp/")
                        && !descriptor.name().startsWith("text/draft_head"))
                .toArray(TensorDescriptor[]::new);
        assertEquals(EXPECTED_OBJECT_COUNT, descriptors.length, "compact runtime object inventory changed");
        long expectedDeviceBytes = sumDescriptorBytes(descriptors);
        Set<String> descriptorNames = descriptorNames(descriptors);
        assertEquals(descriptors.length, descriptorNames.size(), "artifact contains duplicate runtime object names");

        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            CudaGpuMemory.DeviceMemoryInfo before = gpu.deviceMemoryInfo();
            long allocatedBefore = gpu.allocatedBytes();
            Qwen38Model model = null;
            CudaGpuMemory.DeviceMemoryInfo resident = null;
            Throwable failure = null;
            long allocatedDeviceBytes = 0;
            try {
                if (before.freeBytes() < expectedDeviceBytes) {
                    throw new IllegalStateException("insufficient free VRAM before model load: artifact requires "
                            + gibibytes(expectedDeviceBytes) + ", available " + gibibytes(before.freeBytes()));
                }

                model = Qwen38Model.load(artifactPath, artifact, gpu);
                Weights weights = model.weights();
                verifyCompleteAssembly(weights, descriptors, descriptorNames);
                allocatedDeviceBytes = sumHandleBytes(weights.runtimeObjects().values());
                assertEquals(expectedDeviceBytes, allocatedDeviceBytes, "device allocations do not cover every object");
                assertEquals(
                        expectedDeviceBytes,
                        gpu.allocatedBytes() - allocatedBefore,
                        "the model's device footprint is not exactly its runtime objects");
                resident = gpu.deviceMemoryInfo();
                assertTrue(resident.freeBytes() > 0, "GPU reported no VRAM headroom after model load");
            } catch (Throwable loadFailure) {
                failure = loadFailure;
            } finally {
                if (model != null) {
                    try {
                        model.close();
                    } catch (Throwable cleanupFailure) {
                        if (failure == null) failure = cleanupFailure;
                        else failure.addSuppressed(cleanupFailure);
                    }
                }
            }

            long allocatedAfter = gpu.allocatedBytes();
            if (allocatedAfter != allocatedBefore) {
                IllegalStateException restoreFailure = new IllegalStateException(
                        "model teardown did not free every device allocation: allocated before=" + allocatedBefore
                                + ", after=" + allocatedAfter);
                if (failure == null) {
                    failure = restoreFailure;
                } else {
                    failure.addSuppressed(restoreFailure);
                }
            }

            if (failure != null) {
                System.out.printf(
                        "Qwen CUDA residency validation failed: artifact=%s objects=%d requestedDeviceBytes=%d "
                                + "allocatedBefore=%d allocatedAfter=%d%n",
                        artifactPath, descriptors.length, allocatedDeviceBytes, allocatedBefore, allocatedAfter);
                throw failure;
            }

            assertNotNull(resident);
            System.out.printf(
                    "Qwen CUDA residency validation passed: artifact=%s objects=%d allocatedDeviceBytes=%d "
                            + "freeBefore=%d freeAfterLoad=%d%n",
                    artifactPath, descriptors.length, allocatedDeviceBytes, before.freeBytes(), resident.freeBytes());
        }
    }

    @Test
    void releasesPartialCudaAllocationsWhenAnUploadFails() throws Exception {
        Path artifactPath = Path.of(System.getProperty("euhedral.qwen.artifact", DEFAULT_ARTIFACT.toString()));
        Artifact artifact = ArtifactReader.read(artifactPath);
        TensorDescriptor[] descriptors = artifact.tensors();

        Path libraryPath = Path.of(System.getProperty("euhedral.cuda.library"));
        try (CudaGpuMemory gpu = new CudaGpuMemory(libraryPath)) {
            DeviceMemoryInfo before = gpu.deviceMemoryInfo();
            long allocatedBefore = gpu.allocatedBytes();
            int failingCopyIndex = InjectingUploadFailure.FAIL_AFTER_COPIES;
            long bytesBeforeInjectedFailure = 0;
            for (int index = 0; index <= failingCopyIndex; index++) {
                bytesBeforeInjectedFailure = Math.addExact(bytesBeforeInjectedFailure, descriptors[index].byteSize());
            }
            assertTrue(
                    before.freeBytes() >= bytesBeforeInjectedFailure,
                    "insufficient VRAM to verify partial-load rollback");

            InjectingUploadFailure failingGpu = new InjectingUploadFailure(gpu);
            GpuMemoryException failure =
                    assertThrows(GpuMemoryException.class, () -> WeightLoader.load(artifactPath, artifact, failingGpu));
            assertTrue(failure.getMessage().contains("injected CUDA upload failure"));
            assertEquals(InjectingUploadFailure.FAIL_AFTER_COPIES, failingGpu.successfulCopies());
            assertEquals(InjectingUploadFailure.FAIL_AFTER_COPIES + 1, failingGpu.allocations());

            assertEquals(allocatedBefore, gpu.allocatedBytes(), "partial-load failure leaked CUDA allocations");
            System.out.printf(
                    "Qwen CUDA partial-load rollback passed: uploads=%d allocations=%d bytesBeforeFailure=%d%n",
                    failingGpu.successfulCopies(), failingGpu.allocations(), bytesBeforeInjectedFailure);
        }
    }

    private static void verifyCompleteAssembly(
            Weights weights, TensorDescriptor[] descriptors, Set<String> descriptorNames) {
        assertNotNull(weights, "Weights assembly is missing");
        assertNotNull(weights.tokenEmbedding(), "token embedding is missing");
        assertNotNull(weights.finalNorm(), "final norm is missing");
        assertNotNull(weights.lmHead(), "output head is missing");
        assertEquals(64, weights.layers().length, "Qwen text layer assembly is incomplete");
        assertNull(weights.mtp(), "a base load carries no MTP layer");
        assertEquals(EXPECTED_OBJECT_COUNT, weights.runtimeObjects().size(), "not every object was loaded");
        assertEquals(descriptorNames, weights.runtimeObjects().keySet(), "loaded object names differ from EDRL");

        Set<Long> addresses = new HashSet<>();
        for (TensorDescriptor descriptor : descriptors) {
            TensorHandle handle = weights.runtimeObjects().get(descriptor.name());
            assertNotNull(handle, "runtime object is missing after assembly: " + descriptor.name());
            assertArrayEquals(descriptor.shape(), handle.shape(), "shape mismatch for " + descriptor.name());
            assertEquals(descriptor.dataType(), handle.dataType(), "source dtype mismatch for " + descriptor.name());
            assertEquals(descriptor.format(), handle.format(), "storage format mismatch for " + descriptor.name());
            assertEquals(descriptor.layout(), handle.layout(), "runtime layout mismatch for " + descriptor.name());
            assertEquals(descriptor.byteSize(), handle.byteSize(), "payload size mismatch for " + descriptor.name());
            assertTrue(addresses.add(handle.deviceAddress()), "duplicate GPU address for " + descriptor.name());
        }
    }

    private static Set<String> descriptorNames(TensorDescriptor[] descriptors) {
        Set<String> names = new LinkedHashSet<>();
        for (TensorDescriptor descriptor : descriptors) {
            names.add(descriptor.name());
        }
        return names;
    }

    private static long sumDescriptorBytes(TensorDescriptor[] descriptors) throws IOException {
        long total = 0;
        for (TensorDescriptor descriptor : descriptors) {
            total = Math.addExact(total, descriptor.byteSize());
        }
        return total;
    }

    private static long sumHandleBytes(Iterable<TensorHandle> handles) {
        long total = 0;
        for (TensorHandle handle : handles) {
            total = Math.addExact(total, handle.byteSize());
        }
        return total;
    }

    private static String gibibytes(long bytes) {
        return String.format(java.util.Locale.ROOT, "%.3f GiB", bytes / (1024.0 * 1024.0 * 1024.0));
    }

    private static final class InjectingUploadFailure implements GpuMemory {

        private static final int FAIL_AFTER_COPIES = 3;

        private final CudaGpuMemory delegate;
        private int allocations;
        private int successfulCopies;

        private InjectingUploadFailure(CudaGpuMemory delegate) {
            this.delegate = delegate;
        }

        @Override
        public long allocate(long byteSize) {
            long address = delegate.allocate(byteSize);
            allocations++;
            return address;
        }

        @Override
        public void copyHostToDevice(long destination, MemorySegment source, long byteSize) {
            if (successfulCopies == FAIL_AFTER_COPIES) {
                throw new GpuMemoryException("injected CUDA upload failure");
            }
            delegate.copyHostToDevice(destination, source, byteSize);
            successfulCopies++;
        }

        @Override
        public void copyDeviceToHost(MemorySegment destination, long source, long byteSize) {
            delegate.copyDeviceToHost(destination, source, byteSize);
        }

        @Override
        public void free(long address) {
            delegate.free(address);
        }

        private int allocations() {
            return allocations;
        }

        private int successfulCopies() {
            return successfulCopies;
        }
    }
}
