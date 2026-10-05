package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.model_loader.layer_weights.TensorDataType;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightFormat;
import io.euhedral_execution.inference.core.model_loader.layer_weights.WeightLayout;

/// One matrix inside every expert record of a bank: its place in the record (`recordOffset`) and its
/// stored layout. An expert's projection is a complete tensor of its layout (for NVFP4
/// [io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout]), so a kernel can read it at
/// `slotAddress + recordOffset` with `byteSize` as the weight size.
public record ExpertProjection(
        String name,
        long[] shape,
        TensorDataType dataType,
        WeightFormat format,
        WeightLayout layout,
        long recordOffset,
        long byteSize) {}
