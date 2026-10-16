package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.GpuMemoryException;
import io.euhedral_execution.inference.core.gpu.KernelArguments;
import io.euhedral_execution.inference.core.gpu.KernelTable;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/// The shared linear kernel table on the built library: its names, a launch, and the launcher's argument checks.
class LinearKernelTableCudaIntegrationTest {

    private static CudaGpuMemory open() {
        return new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library")));
    }

    @Test
    void theLinearTableMatchesTheEnum() {
        try (CudaGpuMemory gpu = open()) {
            List<String> names = gpu.tableKernelNames(KernelTable.LINEAR);
            assertEquals(
                    Arrays.stream(LinearKernel.values())
                            .map(LinearKernel::symbol)
                            .toList(),
                    names);
        }
    }

    @Test
    void theRowExactBf16KernelsSitWhereTheFlashNextTableHasThem() {
        try (CudaGpuMemory gpu = open()) {
            List<String> names = gpu.tableKernelNames(KernelTable.QWEN4);
            for (RowExactBf16.Kernel kernel : RowExactBf16.Kernel.values())
                assertEquals(kernel.symbol(), names.get(kernel.index()));
        }
    }

    @Test
    void aPrepassLaunchTransformsARow() {
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            int k = 128;
            MemorySegment host = arena.allocate(2L * k, 2);
            for (int i = 0; i < k; i++) host.set(ValueLayout.JAVA_SHORT, 2L * i, (short) 0x3f80); // bf16 1.0
            long x = gpu.allocate(2L * k), suh = gpu.allocate(2L * k), out = gpu.allocate(2L * k);
            try {
                gpu.copyHostToDevice(x, host, 2L * k);
                for (int i = 0; i < k; i++) host.set(ValueLayout.JAVA_SHORT, 2L * i, (short) 0x3c00); // fp16 1.0
                gpu.copyHostToDevice(suh, host, 2L * k);
                gpu.launchTableKernel(
                        LinearKernel.TRELLIS_PREPASS,
                        1,
                        1,
                        1,
                        128,
                        1,
                        1,
                        0,
                        new KernelArguments()
                                .pointer(x)
                                .pointer(suh)
                                .pointer(out)
                                .int32(1)
                                .int32(k)
                                .int32(1)
                                .int32(0));
                gpu.copyDeviceToHost(host, out, 2L * k);
                // H128 of ones is 128 / sqrt(128) = 11.3137 in element 0 (fp16 0x49a8) and 0 elsewhere.
                assertEquals((short) 0x49a8, host.get(ValueLayout.JAVA_SHORT, 0));
                assertEquals((short) 0, host.get(ValueLayout.JAVA_SHORT, 2));
            } finally {
                gpu.free(x);
                gpu.free(suh);
                gpu.free(out);
            }
        }
    }

    @Test
    void theLauncherRejectsArgumentsThatDisagreeWithTheKernel() {
        try (CudaGpuMemory gpu = open()) {
            // euhedral_trellis_prepass takes three pointers and four ints: none, too few, and a 4-byte pointer fail.
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(
                            LinearKernel.TRELLIS_PREPASS, 1, 1, 1, 128, 1, 1, 0, new KernelArguments()));
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(
                            LinearKernel.TRELLIS_PREPASS,
                            1,
                            1,
                            1,
                            128,
                            1,
                            1,
                            0,
                            new KernelArguments().pointer(1).pointer(1).pointer(1)));
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(
                            LinearKernel.TRELLIS_PREPASS,
                            1,
                            1,
                            1,
                            128,
                            1,
                            1,
                            0,
                            new KernelArguments()
                                    .int32(1)
                                    .pointer(1)
                                    .pointer(1)
                                    .int32(1)
                                    .int32(1)
                                    .int32(1)
                                    .int32(0)));
        }
    }
}
