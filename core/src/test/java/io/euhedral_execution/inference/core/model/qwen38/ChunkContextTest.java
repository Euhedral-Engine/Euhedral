package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

/// A prompt quantum's rows come in chunks: each chunk has its own rows, first position and place in the one input
/// record, which holds every chunk's token ids and then each chunk's position.
class ChunkContextTest {

    @Test
    void aPromptQuantumsChunksHaveTheirOwnRowsPositionsAndInput() {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var gpu = new MemoryGpu();
        var quantum = Quantum.prompt(
                plan, new Sequence(1), 0, new int[] {1, 2, 3, 4, 5}, 2, LogitsRequirement.LAST_TOKEN, null);
        assertEquals(3, quantum.chunkCount());
        assertEquals(new Quantum.Chunk(1, 2, 2, 2, false), quantum.chunk(1));
        assertEquals(new Quantum.Chunk(2, 4, 1, 4, true), quantum.chunk(2));
        assertEquals(0, quantum.chunk(0).logitsRows(LogitsRequirement.LAST_TOKEN));
        assertEquals(1, quantum.chunk(2).logitsRows(LogitsRequirement.LAST_TOKEN));
        quantum.begin(gpu, () -> {});
        Workspace workspace = quantum.workspace();
        long tokens = workspace.tokenIdsAddress();
        assertEquals(tokens + 8, workspace.tokenIdsAddress(quantum.chunk(1)));
        assertEquals(workspace.positionAddress() + 16, workspace.positionAddress(quantum.chunk(2)));
        ByteBuffer record = ByteBuffer.wrap(gpu.bytes(tokens, (int) workspace.inputByteSize()))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (int row = 0; row < 5; row++) assertEquals(1 + row, record.getInt(row * 4));
        int positions = (int) (workspace.positionAddress() - tokens);
        assertEquals(0, record.getLong(positions));
        assertEquals(2, record.getLong(positions + 8));
        assertEquals(4, record.getLong(positions + 16));
    }

    @Test
    void anOrdinaryQuantumIsOneChunkWithTheRecordItAlwaysHad() {
        var plan = new ExecutionPlan(EngineExecutionFixture.weights());
        var quantum = new Quantum(plan, new Sequence(2), Quantum.ExecutionKind.PREFILL, 0, new int[] {1, 2, 3});
        assertEquals(1, quantum.chunkCount());
        assertEquals(new Quantum.Chunk(0, 0, 3, 0, true), quantum.chunk(0));
        quantum.begin(new MemoryGpu(), () -> {});
        assertEquals(16 + 8, quantum.workspace().inputByteSize(), "three ids padded to 16 bytes, then one position");
        assertFalse(quantum.workspace().inputByteSize() > 24);
        assertTrue(quantum.workspace().positionAddress() > quantum.workspace().tokenIdsAddress());
    }
}
