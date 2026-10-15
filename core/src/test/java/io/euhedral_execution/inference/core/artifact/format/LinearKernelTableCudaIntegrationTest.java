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
    void aProbeLaunchWritesItsValue() {
        try (CudaGpuMemory gpu = open();
                Arena arena = Arena.ofConfined()) {
            long out = gpu.allocate(16);
            try {
                gpu.launchTableKernel(LinearKernel.PROBE, 1, 1, 1, 32, 1, 1, 0, new KernelArguments().pointer(out));
                MemorySegment host = arena.allocate(4, 4);
                gpu.copyDeviceToHost(host, out, 4);
                assertEquals(0x7e11150, host.get(ValueLayout.JAVA_INT, 0));
            } finally {
                gpu.free(out);
            }
        }
    }

    @Test
    void theLauncherRejectsArgumentsThatDisagreeWithTheKernel() {
        try (CudaGpuMemory gpu = open()) {
            // euhedral_trellis_probe takes one pointer: none, two, and a 4-byte one all fail.
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(LinearKernel.PROBE, 1, 1, 1, 32, 1, 1, 0, new KernelArguments()));
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(
                            LinearKernel.PROBE,
                            1,
                            1,
                            1,
                            32,
                            1,
                            1,
                            0,
                            new KernelArguments().pointer(1).pointer(1)));
            assertThrows(
                    GpuMemoryException.class,
                    () -> gpu.launchTableKernel(
                            LinearKernel.PROBE, 1, 1, 1, 32, 1, 1, 0, new KernelArguments().int32(1)));
        }
    }
}
