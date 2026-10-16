package io.euhedral_execution.inference.quantization.mixed;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/// The front of measured candidates over (KL, prefill microseconds per token, verification microseconds): the
/// candidates that fit the device and that no other fitting candidate beats in all three and in one strictly.
public final class Pareto {

    private Pareto() {}

    /// The front, by increasing KL; of candidates equal in all three, the first measured.
    public static List<Measured> front(List<Measured> measured) {
        List<Measured> front = new ArrayList<>();
        for (Measured candidate : measured) {
            if (!candidate.fits()) continue;
            boolean dominated = false;
            for (Measured other : measured) {
                if (other == candidate || !other.fits()) continue;
                if (dominates(other, candidate)
                        || (equal(other, candidate) && measured.indexOf(other) < measured.indexOf(candidate))) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) front.add(candidate);
        }
        front.sort(Comparator.comparingDouble(Measured::kl));
        return front;
    }

    private static boolean equal(Measured a, Measured b) {
        return a.kl() == b.kl() && a.prefill() == b.prefill() && a.decode() == b.decode();
    }

    private static boolean dominates(Measured a, Measured b) {
        boolean noWorse = a.kl() <= b.kl() && a.prefill() <= b.prefill() && a.decode() <= b.decode();
        return noWorse && !equal(a, b);
    }
}
