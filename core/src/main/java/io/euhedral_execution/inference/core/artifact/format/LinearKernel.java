package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.gpu.KernelTable;
import io.euhedral_execution.inference.core.gpu.TableKernel;

/// The kernels of the shared linear table (`euhedral_cuda_linear_launch`, native/src/host/linear_kernels.c). The
/// declaration order is the native table's order, which is its ABI; a test compares the names.
public enum LinearKernel implements TableKernel {
    /// Writes a known value; it proves the table loads and launches.
    PROBE("euhedral_trellis_probe");

    private final String symbol;

    LinearKernel(String symbol) {
        this.symbol = symbol;
    }

    @Override
    public KernelTable table() {
        return KernelTable.LINEAR;
    }

    @Override
    public int index() {
        return ordinal();
    }

    @Override
    public String symbol() {
        return this.symbol;
    }
}
