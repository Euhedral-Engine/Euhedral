package io.euhedral_execution.inference.quantization.trellis;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Bit allocation over the budgeted linears, ported from exllamav3 1.6.1, `conversion/allocation.py`
/// `create_q_strategy` without `--hq`, with half-integer steps (mul1
/// codebook): every linear starts at the largest supported rate not above the target, then groups of linears are
/// promoted one step at a time while the parameter-weighted budget allows.
///
/// Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).
public final class Allocation {

    private Allocation() {}

    /// One budgeted linear. Linears with the same `group` are promoted together; `priority` is the largest
    /// `q_priority` among its enclosing modules.
    public record Linear(String name, int layer, long numel, int priority, String group) {}

    /// Largest supported rate not above `bpw`: halves below 4, integers up to 8, at least 1.
    static double rateFloor(double bpw) {
        double r = bpw < 4 ? Math.floor(bpw * 2) / 2 : Math.floor(bpw);
        return Math.max(1, Math.min(8, r));
    }

    /// The next supported rate above `bpw` (8 stays 8).
    static double rateNext(double bpw) {
        return Math.min(8, bpw + (bpw < 4 ? 0.5 : 1));
    }

    private static final class Target {
        final Linear linear;
        double bits;

        Target(Linear linear, double bits) {
            this.linear = linear;
            this.bits = bits;
        }

        double delta() {
            return (rateNext(bits) - bits) * linear.numel();
        }
    }

    /// The bits per weight of every linear (keyed by name) for the average `bpw`. Groups are promoted in order of
    /// priority (highest first), distance to the nearer end of the `layers`-layer stack, layer and first
    /// appearance in `linears`; passes repeat until no group fits the budget `(long) (bpw * total weights)`.
    public static Map<String, Double> allocate(List<Linear> linears, int layers, double bpw) {
        double base = rateFloor(bpw);
        double totalNumel = 0;
        double sumBits = 0;
        Map<String, List<Target>> groups = new LinkedHashMap<>();
        Map<String, Target> targets = new LinkedHashMap<>();
        for (Linear linear : linears) {
            Target target = new Target(linear, base);
            totalNumel += linear.numel();
            sumBits += linear.numel() * base;
            targets.put(linear.name(), target);
            groups.computeIfAbsent(linear.group(), g -> new ArrayList<>()).add(target);
        }
        long maxBits = (long) (bpw * totalNumel);

        List<List<Target>> order = new ArrayList<>(groups.values());
        // List.sort is stable, so equal keys keep the groups' first-appearance order
        order.sort(Comparator.<List<Target>>comparingInt(g ->
                        -g.stream().mapToInt(t -> t.linear.priority()).max().orElse(0))
                .thenComparingInt(g -> Math.min(
                        g.get(0).linear.layer(), layers - 1 - g.get(0).linear.layer()))
                .thenComparingInt(g -> g.get(0).linear.layer()));

        while (sumBits < maxBits) {
            boolean updates = false;
            for (List<Target> group : order) {
                double cost = 0;
                for (Target t : group) cost += t.delta();
                if (cost > 0 && sumBits + cost <= maxBits) {
                    for (Target t : group) t.bits = rateNext(t.bits);
                    updates = true;
                    sumBits += cost;
                }
            }
            if (!updates) break;
        }

        Map<String, Double> bits = new LinkedHashMap<>();
        targets.forEach((name, t) -> bits.put(name, t.bits));
        return bits;
    }
}
