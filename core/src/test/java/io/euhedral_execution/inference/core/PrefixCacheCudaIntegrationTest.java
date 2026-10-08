package io.euhedral_execution.inference.core;

import static io.euhedral_execution.inference.core.PrefixCacheEngines.CACHE_BYTES;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.counts;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.generate;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.prompt;
import static io.euhedral_execution.inference.core.PrefixCacheEngines.stateAfterPrefill;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.PrefixCacheEngines.Counts;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.Timeout;

/// A restored sequence continues exactly as a cold one: the state at a checkpoint is the state a cold prefill
/// holds there, and the rest is prefilled in the chunks a cold run uses.
///
/// Checkpoints are 1024 tokens apart and chunks 512. The tests run in the order of the engines they need (see
/// [PrefixCacheEngines]); the first four share one engine and use prompts that start differently, so the
/// entries of one never serve another.
@ModelGroup.CompactQ3Engine
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class PrefixCacheCudaIntegrationTest {

    @Test
    @Order(1)
    @Timeout(600)
    void aRestoredSessionHoldsTheColdStateBufferForBuffer() throws Exception {
        var held = PrefixCacheEngines.q3(CACHE_BYTES);
        var engine = held.engine();
        // 2304 tokens are 9 full pages. Chunks end at multiples of 512, so the cold run stores checkpoints at
        // 1024 and 2048 and prefills 256 tokens last; the warm run restores 2048 and prefills the same 256 in the
        // same single chunk.
        int[] prompt = prompt(engine, "Alpha", 2304);
        Counts start = counts(engine);
        List<String> cold = stateAfterPrefill(held, prompt);
        assertEquals(2, counts(engine).since(start).captured());
        List<String> warm = stateAfterPrefill(held, prompt);
        Counts change = counts(engine).since(start);
        assertEquals(1, change.hits());
        assertEquals(2048, change.reusedTokens());
        assertEquals(cold, warm);
    }

    @Test
    @Order(2)
    @Timeout(600)
    void sampledOutputAfterARestoreEqualsTheColdRun() throws Exception {
        var engine = PrefixCacheEngines.q3(CACHE_BYTES).engine();
        // 2700 tokens: checkpoints at 1024 and 2048, and at 2560, the last chunk boundary before the end.
        int[] prompt = prompt(engine, "Bravo", 2700);
        Counts start = counts(engine);
        List<Integer> cold = generate(engine, prompt, 24, 5);
        List<Integer> warm = generate(engine, prompt, 24, 5);
        assertEquals(cold, warm);
        Counts change = counts(engine).since(start);
        assertEquals(3, change.captured());
        assertEquals(1, change.hits());
        assertEquals(2560, change.reusedTokens());
    }

    @Test
    @Order(3)
    @Timeout(600)
    void aPromptThatEndsOnACheckpointStillPrefillsAChunk() throws Exception {
        var engine = PrefixCacheEngines.q3(CACHE_BYTES).engine();
        int[] prompt = prompt(engine, "Charlie", 2048);
        Counts start = counts(engine);
        List<Integer> cold = generate(engine, prompt, 16, 3);
        List<Integer> warm = generate(engine, prompt, 16, 3);
        assertEquals(cold, warm);
        assertEquals(1536, counts(engine).since(start).reusedTokens(), "the last boundary below the prompt's length");
    }

    @Test
    @Order(4)
    @Timeout(1200)
    void aFollowUpThatExtendsTheExchangeEqualsAColdRunOfIt() throws Exception {
        int[] followUp;
        List<Integer> warm;
        long reused;
        var engine = PrefixCacheEngines.q3(CACHE_BYTES).engine();
        int[] first = prompt(engine, "Delta", 2700);
        List<Integer> fed = new ArrayList<>(generate(engine, first, 24, 7));
        // A sampled terminator is returned but never fed to the model.
        if (!fed.isEmpty() && engine.tokenizer().isGenerationEosToken(fed.getLast())) fed.removeLast();
        int[] extra = engine.tokenizer().encodeText(" And then the second turn asks something else entirely.");
        followUp = new int[first.length + fed.size() + extra.length];
        System.arraycopy(first, 0, followUp, 0, first.length);
        for (int i = 0; i < fed.size(); i++) followUp[first.length + i] = fed.get(i);
        System.arraycopy(extra, 0, followUp, first.length + fed.size(), extra.length);
        long before = engine.prefixCacheStats().reusedTokens();
        warm = generate(engine, followUp, 24, 9);
        reused = engine.prefixCacheStats().reusedTokens() - before;
        assertEquals(2560, reused, "the follow-up restores the first prompt's last chunk boundary");
        // The engine without a cache replaces the first one, which the device cannot hold beside it.
        assertEquals(generate(PrefixCacheEngines.q3(0).engine(), followUp, 24, 9), warm);
    }

    @Test
    @Order(5)
    @Timeout(1200)
    void anEvictionRunStaysWithinTheBudgetAndKeepsHitting() throws Exception {
        var engine = PrefixCacheEngines.q3(EVICTION_BYTES).engine();
        int[] last = null;
        for (String lead : List.of("Echo", "Foxtrot", "Golf", "Hotel", "India")) {
            last = prompt(engine, lead, 2700);
            generate(engine, last, 4, 1);
        }
        PrefixCacheStats stats = engine.prefixCacheStats();
        System.out.println("EVICTION_STATS " + stats);
        assertTrue(stats.evictions() > 0, "five prompts do not fit " + (EVICTION_BYTES >> 20) + " MiB");
        assertTrue(stats.usedBytes() <= stats.totalBytes());
        long hits = stats.hits();
        generate(engine, last, 4, 1);
        assertEquals(hits + 1, engine.prefixCacheStats().hits(), "the newest prompt is still stored");
    }

    private static final long EVICTION_BYTES = 200L << 20;
}
