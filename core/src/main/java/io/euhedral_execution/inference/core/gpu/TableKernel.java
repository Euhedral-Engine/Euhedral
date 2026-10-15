package io.euhedral_execution.inference.core.gpu;

/// A kernel of a native kernel table ([KernelTable]), launched by its index in that table. A kernel enum implements
/// it; its declaration order must be the table's order.
public interface TableKernel {

    /// The table the kernel belongs to.
    KernelTable table();

    /// The kernel's index in the native table.
    int index();

    /// The kernel's native symbol, which a test compares with the table's names.
    String symbol();
}
