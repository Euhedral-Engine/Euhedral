package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.gpu.ScratchUse;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;

/// The scratch route each linear kind declares: the gate/up region at any row count, every other kind from the
/// multi-row views on. NVFP4 weights take the native route's scratch; every other format takes the Q3 and FP8
/// routes'.
final class ScratchRules {

    private ScratchRules() {}

    static ScratchUse nonNvfp4(ExecutionPlan.Kind kind, boolean multiRow) {
        if (kind == ExecutionPlan.Kind.Q3_GATE_UP_SWIGLU) return ScratchUse.Q3_GATE_UP;
        if (!multiRow) return null;
        return switch (kind) {
            case Q3_LINEAR -> ScratchUse.Q3_LINEAR;
            case Q4_LINEAR, Q5_LINEAR -> ScratchUse.MX_LINEAR;
            case DFLASH_LM_HEAD -> ScratchUse.Q3_LINEAR;
            default -> null;
        };
    }

    static ScratchUse nvfp4(ExecutionPlan.Kind kind, boolean multiRow) {
        if (kind == ExecutionPlan.Kind.Q3_GATE_UP_SWIGLU) return ScratchUse.NVFP4_GATE_UP;
        if (!multiRow) return null;
        return switch (kind) {
            case Q3_LINEAR, Q4_LINEAR, Q5_LINEAR, DFLASH_LINEAR, DFLASH_LM_HEAD -> ScratchUse.NVFP4_LINEAR;
            default -> null;
        };
    }
}
