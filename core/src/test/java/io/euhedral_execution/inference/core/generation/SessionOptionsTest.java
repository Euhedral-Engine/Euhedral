package io.euhedral_execution.inference.core.generation;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.ModelDescription;
import org.junit.jupiter.api.Test;

class SessionOptionsTest {
    @Test
    void sessionsSpeculateByDefaultAndTheControlArmTurnsItOff() {
        assertTrue(SessionOptions.DEFAULT.speculation());
        assertFalse(new SessionOptions(false).speculation());
    }

    @Test
    void aModelDescriptionIsSpeculativeUnlessItsSpeculationIsNone() {
        assertFalse(ModelDescription.NONE.speculative());
        assertTrue(new ModelDescription("q3", "mtp", 3).speculative());
    }
}
