package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.loader.ResidencyPlanner;
import io.euhedral_execution.inference.quantization.Residency;
import java.util.OptionalInt;

/// Whether an allocation can be served: the residency plan of its composite artifact at the target context
/// keeps every object on the device.
public interface Feasibility {

    /// The verdict on one allocation: whether it fits, and the device bytes the plan needs.
    record Check(boolean fits, long deviceBytes) {}

    /// The device bytes the checks are made against.
    long freeBytes();

    Check check(Allocation allocation);

    /// The engine's planner ([ResidencyPlanner]) on the composite of `allocation`, for a device with
    /// `freeBytes` free and sequences of up to `context` tokens, the composite drafting `draftDepth` tokens per
    /// verification (a mixed artifact stores no depth until its finalist measurement). It fits when nothing is
    /// host-backed or disk-backed.
    static Feasibility planner(Sources sources, long freeBytes, int context, int draftDepth) {
        return new Feasibility() {
            @Override
            public long freeBytes() {
                return freeBytes;
            }

            @Override
            public Check check(Allocation allocation) {
                Artifact composite = Assemble.view(sources, allocation, OptionalInt.of(draftDepth));
                var plan = ResidencyPlanner.plan(
                        composite,
                        ArtifactProfile.of(composite),
                        freeBytes,
                        context,
                        Long.MAX_VALUE,
                        Residency.kvFormat(composite));
                return new Check(
                        plan.fits()
                                && plan.hostBacked().isEmpty()
                                && plan.diskBacked().isEmpty(),
                        plan.deviceBytes());
            }
        };
    }
}
