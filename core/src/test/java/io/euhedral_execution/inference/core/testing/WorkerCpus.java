package io.euhedral_execution.inference.core.testing;

import io.euhedral_execution.hardware_utils.topology.SystemInfo;
import java.util.BitSet;

/// The worker CPUs a test pins its engine to: `euhedral.test.worker-cpus` (-Peuhedral.test.worker-cpus, a list of
/// CPUs and ranges such as `2-5` or `2,3,8`), or else the first `count` performance CPUs of the machine.
public final class WorkerCpus {

    public static final String PROPERTY = "euhedral.test.worker-cpus";

    private WorkerCpus() {}

    /// The configured CPUs, or the first `count` performance CPUs.
    public static BitSet of(int count) {
        String configured = System.getProperty(PROPERTY);
        if (configured != null && !configured.isBlank()) return parse(configured);
        BitSet performance = SystemInfo.getPCpuSet();
        BitSet chosen = new BitSet();
        for (int cpu = performance.nextSetBit(0);
                cpu >= 0 && chosen.cardinality() < count;
                cpu = performance.nextSetBit(cpu + 1)) chosen.set(cpu);
        return chosen;
    }

    /// `2-5`, `2,3,8`, `0-1,4`: CPUs and inclusive ranges, separated by commas.
    public static BitSet parse(String list) {
        BitSet cpus = new BitSet();
        for (String part : list.split(",")) {
            String item = part.trim();
            if (item.isEmpty()) continue;
            int dash = item.indexOf('-');
            if (dash < 0) {
                cpus.set(Integer.parseInt(item));
                continue;
            }
            int first = Integer.parseInt(item.substring(0, dash).trim());
            int last = Integer.parseInt(item.substring(dash + 1).trim());
            if (last < first) throw new IllegalArgumentException("a CPU range runs backwards: " + item);
            cpus.set(first, last + 1);
        }
        if (cpus.isEmpty()) throw new IllegalArgumentException("no worker CPUs in " + list);
        return cpus;
    }
}
