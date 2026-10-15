package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import java.util.List;

/// One linear launch: `weights` are the object first, then its companions (staged slot addresses already
/// substituted); `scratch` is the workspace scratch bound for the stage, or 0; `rowExact` marks a VERIFY quantum.
public record LinearLaunch(
        ExecutionPlan.Kind kind,
        List<TensorHandle> weights,
        long input,
        long output,
        int rows,
        int inFeatures,
        int outFeatures,
        long scratch,
        long scratchBytes,
        boolean rowExact,
        boolean hostMapped) {}
