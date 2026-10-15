package io.euhedral_execution.inference.core.model.qwen38.loader;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.model.qwen38.GdnState;
import io.euhedral_execution.inference.core.model.qwen38.LayerType;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.state.KvFormat;
import org.junit.jupiter.api.Test;

class SequenceFootprintTest {

    private static final class CountingGpu extends ExecutionFixtures.RecordingGpu {
        long bytes;

        @Override
        public void zeroDeviceMemory(long address, long byteSize) {}

        @Override
        public long allocate(long byteSize) {
            this.bytes += byteSize;
            return super.allocate(byteSize);
        }
    }

    @Test
    void gdnBytesEqualWhatTheStatesAllocate() {
        Qwen38Config config = ExecutionFixtures.statefulCompactWeights(8).config();
        int queryKeyWidth = 2 * config.linearNumKeyHeads() * config.linearKeyHeadDim();
        int valueWidth = config.linearNumValueHeads() * config.linearValueHeadDim();
        var gpu = new CountingGpu();
        int layers = 0;
        for (LayerType type : config.layerTypes()) {
            if (type != LayerType.GATED_DELTA_NET) continue;
            layers++;
            try (GdnState state = GdnState.allocate(
                    gpu,
                    config.linearNumKeyHeads(),
                    config.linearNumValueHeads(),
                    config.linearKeyHeadDim(),
                    config.linearValueHeadDim(),
                    config.linearConvKernelDim())) {
                state.speculative(
                        5,
                        queryKeyWidth,
                        2 * valueWidth,
                        config.linearNumValueHeads(),
                        queryKeyWidth + valueWidth,
                        valueWidth);
            }
        }
        assertTrue(layers > 0);
        assertEquals(SequenceFootprint.gdnBytes(config, 5), gpu.bytes);
        // Below eight rows the buffers still hold eight; above, they grow with the rows.
        assertEquals(SequenceFootprint.gdnBytes(config, 8), SequenceFootprint.gdnBytes(config, 1));
        assertTrue(SequenceFootprint.gdnBytes(config, 16) > SequenceFootprint.gdnBytes(config, 8));
    }

    @Test
    void anExtraSequenceCostsMoreWhenItDraftsAndWithMoreContext() {
        Qwen38Config config = ExecutionFixtures.statefulCompactWeights(8).config();
        long plain = SequenceFootprint.extraSequenceBytes(config, KvFormat.BF16, 4096, 5, false);
        long drafting = SequenceFootprint.extraSequenceBytes(config, KvFormat.BF16, 4096, 5, true);
        long longer = SequenceFootprint.extraSequenceBytes(config, KvFormat.BF16, 8192, 5, true);
        assertTrue(plain > SequenceFootprint.gdnBytes(config, 5));
        assertTrue(drafting > plain);
        assertTrue(longer > drafting);
        assertEquals(plain, SequenceFootprint.extraSequenceBytes(config, KvFormat.BF16, 8192, 5, false));
    }

    @Test
    void theDecodeScratchSeedRowsAndRecursionHiddenAreCounted() {
        Qwen38Config config = ExecutionFixtures.statefulCompactWeights(8).config();
        int rows = 5;
        var gpu = new CountingGpu();
        long scratchAndSeeds;
        try (AttentionStates states = AttentionStates.allocate(
                gpu,
                config.layerTypes(),
                config.numKeyValueHeads() * config.attentionHeadDim(),
                false,
                KvFormat.BF16)) {
            states.decodeScratch(config.numAttentionHeads(), Math.max(rows, AttentionStates.DRAFT_TWIN_ROWS));
            states.draftSeedRows(InferenceConfig.PREFILL_CHUNK_TOKENS, config.hiddenSize());
            states.draftRecursionHidden(config.hiddenSize());
            scratchAndSeeds = gpu.bytes;
        }
        long selection = 16 + rows * 8L;
        long mtpCache = ResidencyPlanner.kvBytes(config, 4096, KvFormat.BF16);
        assertEquals(
                SequenceFootprint.gdnBytes(config, rows) + scratchAndSeeds + selection + mtpCache,
                SequenceFootprint.extraSequenceBytes(config, KvFormat.BF16, 4096, rows, true));
    }
}
