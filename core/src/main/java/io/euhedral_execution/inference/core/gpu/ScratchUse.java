package io.euhedral_execution.inference.core.gpu;

/// A route that takes the expansion scratch: what a stage declares when it will use it, so the scratch can be one
/// workspace buffer whose uses edges order.
public enum ScratchUse {
    /// A P2E2 or row-split Q3 linear: the P2E2 expansion, or the FP8 route's activations, from nine rows.
    Q3_LINEAR,
    /// A Q3 gate/up region: its gate/up rows at any row count, and the P2E2 expansion from nine.
    Q3_GATE_UP,
    /// A Q4 or Q5 linear on the FP8 route, from nine rows.
    MX_LINEAR,
    /// An NVFP4 linear on the native route, from nine rows.
    NVFP4_LINEAR,
    /// An NVFP4 gate/up region: its gate/up rows below 64, the native route's activations from nine.
    NVFP4_GATE_UP
}
