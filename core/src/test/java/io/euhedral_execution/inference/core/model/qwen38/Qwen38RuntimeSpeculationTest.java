package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Quantization;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile.Speculation;
import org.junit.jupiter.api.Test;

/// Which speculative strategy a session runs: the artifact's, unless the options turn speculation off (the
/// benchmark's control arm) or the plan cannot draft it.
class Qwen38RuntimeSpeculationTest {

    private static final ArtifactProfile MTP = new ArtifactProfile(Quantization.NVFP4, true, Speculation.MTP);
    private static final ArtifactProfile DFLASH2 = new ArtifactProfile(Quantization.Q3, false, Speculation.DFLASH2);
    private static final SessionOptions OFF = new SessionOptions(false);

    @Test
    void theDefaultOptionsRunTheArtifactsStrategy() {
        assertEquals(Speculation.MTP, Qwen38Runtime.speculation(SessionOptions.DEFAULT, MTP, true, true));
        assertEquals(Speculation.DFLASH2, Qwen38Runtime.speculation(SessionOptions.DEFAULT, DFLASH2, true, true));
    }

    @Test
    void optionsWithoutSpeculationDecodeOneTokenAtATime() {
        assertEquals(Speculation.NONE, Qwen38Runtime.speculation(OFF, MTP, true, true));
        assertEquals(Speculation.NONE, Qwen38Runtime.speculation(OFF, DFLASH2, true, true));
    }

    @Test
    void aPlanThatCannotDraftTheStrategyDoesNotSpeculate() {
        assertEquals(Speculation.NONE, Qwen38Runtime.speculation(SessionOptions.DEFAULT, MTP, false, true));
        assertEquals(Speculation.NONE, Qwen38Runtime.speculation(SessionOptions.DEFAULT, DFLASH2, true, false));
    }

    @Test
    void withoutAProfileNothingSpeculates() {
        assertEquals(Speculation.NONE, Qwen38Runtime.speculation(SessionOptions.DEFAULT, null, true, true));
    }
}
