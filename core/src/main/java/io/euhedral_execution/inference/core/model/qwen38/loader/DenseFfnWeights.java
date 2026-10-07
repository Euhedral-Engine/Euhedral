package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorHandle;

/// Fused gate/up and compact down projections used by the Q3 runtime inventory.
public record DenseFfnWeights(TensorHandle gateUp, TensorHandle down) implements FfnWeights {}
