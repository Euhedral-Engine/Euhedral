package io.euhedral_execution.inference.core.model.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import io.euhedral_execution.inference.core.model.qwen4.loader.HostMemoryGpu;
import org.junit.jupiter.api.Test;

/// A prompt's chunks of one QSA layer are pending together: each starts where the previous one ended and reads the
/// tail it wrote; the prompt's retirement commits (or drops) all of them at once.
class QsaStateTest {

    @Test
    void twoPendingChunksCommitAsOne() {
        try (var state = new QsaState(new HostMemoryGpu(), 512, 4096)) {
            int first = state.beginChunk(512);
            long firstIn = state.tailIn(), firstOut = state.tailOut();
            state.submitted();
            int second = state.beginChunk(300);
            assertEquals(0, first);
            assertEquals(512, second, "the second chunk continues the first");
            assertEquals(firstOut, state.tailIn(), "it reads the tail the first chunk wrote");
            assertEquals(firstIn, state.tailOut());
            state.submitted();
            assertEquals(0, state.length(), "nothing is committed before retirement");
            state.commit();
            state.commit();
            assertEquals(812, state.length(), "both chunks, once");
            assertEquals(firstIn, state.tailIn(), "the committed tail is the second chunk's");
        }
    }

    @Test
    void aDroppedPromptDropsAllItsChunks() {
        try (var state = new QsaState(new HostMemoryGpu(), 512, 4096)) {
            state.beginChunk(256);
            state.submitted();
            state.beginChunk(256);
            state.submitted();
            state.discard();
            state.discard();
            assertEquals(0, state.length());
            assertEquals(0, state.beginChunk(256), "the next chunk starts at the committed length");
            assertNotEquals(0L, state.tailIn());
        }
    }
}
