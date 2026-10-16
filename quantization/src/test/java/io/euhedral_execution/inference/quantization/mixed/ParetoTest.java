package io.euhedral_execution.inference.quantization.mixed;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.quantization.mixed.SearchState.Measured;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ParetoTest {

    private static Measured point(String tag, double kl, double prefill, double decode, boolean fits) {
        return new Measured(
                new Allocation(Map.of("text/layers/0/mlp/down", Option.NVFP4)),
                kl,
                0,
                0,
                prefill,
                decode,
                fits,
                1,
                tag);
    }

    @Test
    void keepsTheNonDominatedFeasiblePointsByKl() {
        List<Measured> data = List.of(
                point("a", 0.10, 10, 10, true),
                point("b", 0.05, 20, 20, true),
                point("c", 0.12, 12, 12, true), // dominated by a
                point("d", 0.02, 5, 5, false), // would dominate all but does not fit
                point("e", 0.05, 20, 25, true), // dominated by b
                point("f", 0.20, 4, 30, true)); // fastest prefill
        List<Measured> front = Pareto.front(data);
        assertEquals(
                List.of("b", "a", "f"), front.stream().map(Measured::origin).toList());
    }

    @Test
    void anEmptySetHasAnEmptyFront() {
        assertEquals(List.of(), Pareto.front(List.of()));
    }
}
