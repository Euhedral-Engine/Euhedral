package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.ArrayList;
import java.util.List;

/// Combining the fences of one slot. A slot is refilled behind every fence its leases were closed with; fences
/// that cover one another collapse, so the combination holds only fences that order different device work.
final class DeviceFences {
    private DeviceFences() {}

    /// The fence that orders behind both `current` (possibly null) and `incoming`. Fences that became redundant
    /// are added to `superseded`, for the caller to release.
    static DeviceFence merge(DeviceFence current, DeviceFence incoming, List<DeviceFence> superseded) {
        if (current == null) return incoming;
        List<DeviceFence> members = new ArrayList<>();
        if (current instanceof Composite composite) members.addAll(composite.members);
        else members.add(current);
        for (DeviceFence member : members) {
            if (member == incoming) return current;
            if (member.covers(incoming)) {
                superseded.add(incoming);
                return current;
            }
        }
        List<DeviceFence> kept = new ArrayList<>(members.size() + 1);
        for (DeviceFence member : members) {
            if (incoming.covers(member)) superseded.add(member);
            else kept.add(member);
        }
        kept.add(incoming);
        return kept.size() == 1 ? incoming : new Composite(List.copyOf(kept));
    }

    private record Composite(List<DeviceFence> members) implements DeviceFence {
        @Override
        public void awaitOn(GpuStream copyStream) {
            for (DeviceFence member : this.members) member.awaitOn(copyStream);
        }

        @Override
        public boolean covers(DeviceFence other) {
            for (DeviceFence member : this.members) if (member.covers(other)) return true;
            return false;
        }

        @Override
        public void release() {
            RuntimeException failure = null;
            for (DeviceFence member : this.members) {
                try {
                    member.release();
                } catch (RuntimeException releaseFailure) {
                    if (failure == null) failure = releaseFailure;
                    else failure.addSuppressed(releaseFailure);
                }
            }
            if (failure != null) throw failure;
        }
    }
}
