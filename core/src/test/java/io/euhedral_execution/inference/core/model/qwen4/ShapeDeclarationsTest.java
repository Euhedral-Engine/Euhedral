package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumSet;
import org.junit.jupiter.api.Test;

/// What a Flash-Next stage declares to the runtime: the one workspace buffer for every device stage, and the
/// expansion scratch for the stages whose NVFP4 linears take the native route.
class ShapeDeclarationsTest {

    @Test
    void onlyDeviceStagesDeclareTheWorkspace() {
        var host = EnumSet.of(
                Shape.Kind.PLEIDS,
                Shape.Kind.PLEGATHER,
                Shape.Kind.MID,
                Shape.Kind.OBSERVE,
                Shape.Kind.PREFETCH,
                Shape.Kind.FETCH);
        for (Shape.Kind kind : Shape.Kind.values())
            assertEquals(!host.contains(kind), Shape.declaresWorkspace(kind), kind.name());
    }

    @Test
    void theNvfp4LinearStagesTakeTheScratchFromNineRows() {
        var users = EnumSet.of(Shape.Kind.PLE, Shape.Kind.ATTENTION, Shape.Kind.BLOCK, Shape.Kind.SHARED);
        for (Shape.Kind kind : Shape.Kind.values()) {
            assertFalse(Shape.takesScratch(kind, 1), kind.name());
            assertFalse(Shape.takesScratch(kind, 8), kind.name());
            assertEquals(users.contains(kind), Shape.takesScratch(kind, 9), kind.name());
            assertEquals(users.contains(kind), Shape.takesScratch(kind, 512), kind.name());
        }
        assertTrue(Shape.takesScratch(Shape.Kind.SHARED, 16));
    }

    @Test
    void eachLayersAttentionStateAndThePleStateAreCarried() {
        int layers = 48;
        assertEquals(java.util.List.of(7), boxed(Shape.carriedKeys(Shape.Kind.BLOCK, 7, layers)));
        assertEquals(java.util.List.of(7), boxed(Shape.carriedKeys(Shape.Kind.ATTENTION, 7, layers)));
        assertEquals(java.util.List.of(layers), boxed(Shape.carriedKeys(Shape.Kind.PLEIDS, 2, layers)), "PLE context");
        assertEquals(java.util.List.of(layers + 1), boxed(Shape.carriedKeys(Shape.Kind.PLE, 2, layers)), "PLE history");
        for (Shape.Kind kind : Shape.Kind.values())
            if (kind != Shape.Kind.BLOCK
                    && kind != Shape.Kind.ATTENTION
                    && kind != Shape.Kind.PLEIDS
                    && kind != Shape.Kind.PLE) assertEquals(0, Shape.carriedKeys(kind, 3, layers).length, kind.name());
    }

    private static java.util.List<Integer> boxed(int[] keys) {
        return java.util.Arrays.stream(keys).boxed().toList();
    }
}
