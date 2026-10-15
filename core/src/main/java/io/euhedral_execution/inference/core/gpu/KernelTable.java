package io.euhedral_execution.inference.core.gpu;

/// The native kernel tables, each with its own launcher (`euhedral_cuda_<table>_launch`). A [TableKernel] names the
/// table it belongs to; its index is a position in that table.
public enum KernelTable {
    /// The Flash-Next (qwen4_exp) operators (native/src/host/qwen4_ops.c).
    QWEN4,
    /// The weight formats' own launches, shared by every model (native/src/host/linear_kernels.c).
    LINEAR
}
