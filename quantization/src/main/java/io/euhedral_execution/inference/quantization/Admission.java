package io.euhedral_execution.inference.quantization;

import java.util.Map;
import java.util.function.Predicate;

/// Whether another candidate may start while others are scored: the device bytes it would add against the device's
/// free memory. Candidates are admitted in the order they were submitted, so a large one is not starved by small ones
/// behind it.
final class Admission {

    /// What one candidate needs on the device: each object it uses (by key, with its bytes), and the bytes beyond its
    /// weights (its sequences' KV and state and its workspace).
    record Need(Map<WeightCache.Key, Long> objects, long otherBytes) {

        /// The device bytes starting it adds when `resident` says which objects are already on the device: its
        /// other bytes and the objects not yet there.
        long additional(Predicate<WeightCache.Key> resident) {
            long bytes = otherBytes;
            for (Map.Entry<WeightCache.Key, Long> object : objects.entrySet())
                if (!resident.test(object.getKey())) bytes += object.getValue();
            return bytes;
        }

        long weightBytes() {
            long bytes = 0;
            for (long object : objects.values()) bytes += object;
            return bytes;
        }
    }

    private Admission() {}

    /// The device bytes a candidate may add now: the lower of the free bytes the device reports and the free bytes
    /// when the scorer opened less what the cache holds and the running candidates' other bytes (an allocation
    /// that has been decided but not made does not show in the first), less `margin`.
    static long available(long freeNow, long freeAtOpen, long cacheBytes, long runningOtherBytes, long margin) {
        return Math.min(freeNow, freeAtOpen - cacheBytes - runningOtherBytes) - margin;
    }

    /// Whether the next candidate starts alongside `running` others: the limit `maxConcurrent` (0 for none) allows
    /// one more and `additional` bytes are within what is [#available]. With nothing running it always starts.
    static boolean admits(int running, int maxConcurrent, long additional, long available) {
        if (running == 0) return true;
        if (maxConcurrent > 0 && running >= maxConcurrent) return false;
        return additional <= available;
    }
}
