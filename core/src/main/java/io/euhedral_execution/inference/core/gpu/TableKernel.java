package io.euhedral_execution.inference.core.gpu;

/// A kernel of the native kernel table (`euhedral_cuda_qwen4_launch`), launched by its index in that table.
/// A model's kernel enum implements it; its declaration order must be the table's order.
public interface TableKernel {

    /// The kernel's index in the native table.
    int index();

    /// The kernel's native symbol, which a test compares with the table's names.
    String symbol();
}
