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
}
