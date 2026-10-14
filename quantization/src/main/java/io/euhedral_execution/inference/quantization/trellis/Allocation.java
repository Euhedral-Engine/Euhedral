package io.euhedral_execution.inference.quantization.trellis;

import java.util.List;
import java.util.Map;

/// Bit allocation over the budgeted linears.
public final class Allocation {

    private Allocation() {}

    /// One budgeted linear. Linears with the same `group` are promoted together; `priority` is the largest
    /// `q_priority` among its enclosing modules.
    public record Linear(String name, int layer, long numel, int priority, String group) {}

    public static Map<String, Double> allocate(List<Linear> linears, int layers, double bpw) {
        throw new UnsupportedOperationException();
    }
}
