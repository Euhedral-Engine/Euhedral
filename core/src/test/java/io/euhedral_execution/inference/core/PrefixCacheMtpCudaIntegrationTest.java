package io.euhedral_execution.inference.core;

import static io.euhedral_execution.inference.core.PrefixCacheEngines.CACHE_BYTES;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.counts;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.prompt;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.PrefixCacheEngines.Counts;
import io.euhedral_execution.inference.core.SharedEngines.Held;
import io.euhedral_execution.inference.core.model.qwen38.EngineExecutionFixture;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Runtime;
import io.euhedral_execution.inference.core.model.qwen38.SequenceStateProbe;
import io.euhedral_execution.inference.core.model.qwen38.Session;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;

/// The prefix cache of a speculative sequence also keeps the MTP layer's cache, and a restore rebuilds both. These
/// run on the NVFP4 artifact, whose MTP layer drafts. Checkpoints are 1024 tokens apart and chunks 512.
@ModelGroup.Nvfp4Engine
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class PrefixCacheMtpCudaIntegrationTest {

    @Test
    @Order(1)
    @Timeout(600)
    void aSpeculativeRestoreLeavesTheMtpCacheAsAColdRunDoes() throws Exception {
        var held = PrefixCacheEngines.nvfp4(CACHE_BYTES);
        var engine = held.engine();
        assumeTrue(engine.description().speculative(), "needs an artifact with the MTP layer");
        int[] prompt = prompt(engine, "Lima", 2700);
        Counts start = counts(engine);
        List<String> cold = mtpAndBaseDigests(held, prompt);
        List<String> warm = mtpAndBaseDigests(held, prompt);
        Counts change = counts(engine).since(start);
        assertEquals(1, change.hits());
        assertEquals(2560, change.reusedTokens());
        // The base state and every MTP page are the cold run's, except the page that holds MTP row 2559: a
        // restore at 2560 recomputes that row in a one-row quantum, where the cold run computed it inside a
        // chunk's catch-up. It differs in low bits, which can change a draft but never a token.
        String recomputed = "mtp page " + (2559 / 256) + " ";
        assertEquals(without(cold, recomputed), without(warm, recomputed));
        assertEquals(cold.size(), warm.size());
    }

    @Test
    @Order(2)
    @Timeout(1200)
    void aSpeculativePromptRestoresThroughMtpCheckpointsAndGeneratesTheSameTokens() throws Exception {
        var engine = PrefixCacheEngines.nvfp4(CACHE_BYTES).engine();
        assumeTrue(engine.description().speculative(), "needs an artifact with the MTP layer");
        // Greedy and unconstrained: the engine drafts with MTP. The cold run stores MTP checkpoints at 1024,
        // 2048 and 2560; the warm run restores 2560, re-pairs its last MTP row and prefills the rest.
        int[] prompt = prompt(engine, "Kilo", 2700);
        Counts start = counts(engine);
        List<Integer> cold = greedy(engine, prompt, 48);
        assertEquals(3, counts(engine).since(start).captured());
        List<Integer> warm = greedy(engine, prompt, 48);
        assertEquals(cold, warm);
        Counts change = counts(engine).since(start);
        assertEquals(1, change.hits());
        assertEquals(2560, change.reusedTokens());
        // A prompt that diverges after 2100 tokens restores the node at 2048.
        int[] diverging = Arrays.copyOf(prompt, 2700);
        for (int i = 2100; i < diverging.length; i++) diverging[i] = prompt[i] ^ 1;
        List<Integer> restored = greedy(engine, diverging, 48);
        assertEquals(2048, counts(engine).since(start).reusedTokens() - 2560);
        // The engine without a cache replaces the first one, which the device cannot hold beside it.
        assertEquals(greedy(PrefixCacheEngines.nvfp4(0).engine(), diverging, 48), restored);
    }

    private static List<String> without(List<String> digests, String prefix) {
        return digests.stream().filter(digest -> !digest.startsWith(prefix)).toList();
    }

    /// The digests of the base state and of the MTP cache after a one-token speculative generation.
    private static List<String> mtpAndBaseDigests(Held held, int[] prompt) throws Exception {
        try (Session session = held.engine().createSession(GenerationConfig.greedy(1))) {
            session.generate(prompt, 1, text -> {}, null);
            var sequence = EngineExecutionFixture.sequence(session);
            var layerTypes =
                    ((Qwen38Runtime) held.engine().modelRuntime()).config().layerTypes();
            List<String> digests =
                    new ArrayList<>(SequenceStateProbe.committedDigests(held.gpu(), sequence, layerTypes));
            List<String> mtp = SequenceStateProbe.mtpDigests(held.gpu(), sequence, layerTypes, prompt.length);
            assertTrue(!mtp.isEmpty(), "the speculative sequence has an MTP cache");
            digests.addAll(mtp);
            return digests;
        }
    }

    private static List<Integer> greedy(InferenceEngine engine, int[] prompt, int newTokens) throws Exception {
        try (Session session = engine.createSession(GenerationConfig.greedy(1))) {
            return session.generate(prompt, newTokens, text -> {}, null);
        }
    }
}
