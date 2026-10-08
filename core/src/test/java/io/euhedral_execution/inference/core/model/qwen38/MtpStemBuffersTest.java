package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

/// The MTP stem writes the normalized rows before it packs them: both are slots of the shared workspace, so both are
/// declared, and two sessions' drafts order their use of them.
class MtpStemBuffersTest {

    @Test
    void theStemDeclaresTheRowsItNormalizesInto() {
        var stem = ExecutionPlan.mtpStem(List.of(), 64, 4);
        assertTrue(stem.outputBuffers().contains(ExecutionPlan.Buffer.MTP_NORMED), "outputs " + stem.outputBuffers());
        assertTrue(stem.outputBuffers().contains(ExecutionPlan.Buffer.MTP_PACKED));
    }
}
