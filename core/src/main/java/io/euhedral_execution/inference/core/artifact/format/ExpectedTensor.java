package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDataType;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;

/// What the runtime registers for an object: the shape, data type, format and layout a descriptor must stand for.
public record ExpectedTensor(long[] shape, TensorDataType dataType, WeightFormat format, WeightLayout layout) {}
