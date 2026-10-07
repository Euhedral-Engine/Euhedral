package io.euhedral_execution.inference.core.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.gpu.Qwen4Kernel;
import io.euhedral_execution.inference.core.gpu.TableKernel;
import org.junit.jupiter.api.Test;

/// The kernel table is indexed by declaration order, which must be the native table's order.
class KernelIndexTest {

    @Test
    void eachKernelIsLaunchedByItsPositionInTheTable() {
        Qwen4Kernel[] kernels = Qwen4Kernel.values();
        for (int i = 0; i < kernels.length; i++) {
            TableKernel kernel = kernels[i];
            assertEquals(i, kernel.index(), kernel.symbol());
        }
    }
}
