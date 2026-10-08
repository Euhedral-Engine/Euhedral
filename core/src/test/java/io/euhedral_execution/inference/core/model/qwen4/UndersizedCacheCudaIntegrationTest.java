package io.euhedral_execution.inference.core.model.qwen4;

import static io.euhedral_execution.inference.core.model.qwen4.GreedyAgreementCudaIntegrationTest.cases;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.testing.ModelGroup;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// What the expert cache holds is not part of the answer, even at its smallest: the smallest cache the planner allows
/// (5 GiB of device and a 262,144-token context leave about 20 slots, so experts are evicted all the time) gives the
/// bit-for-bit outputs of the roomy cache, for the greedy continuations of `GreedyAgreementCudaIntegrationTest` and
/// the logits of `InvarianceCudaIntegrationTest`. The roomy outputs are computed once per JVM by whichever class asks
/// first. This class sorts after the roomy-model classes, so the group swaps models once.
@ModelGroup.FlashNext
@Timeout(value = 5, unit = TimeUnit.MINUTES)
class UndersizedCacheCudaIntegrationTest {

    @Test
    void theGreedyTokensDoNotDependOnTheCacheCapacity() throws Exception {
        var all = cases();
        int[][] roomy = GreedyAgreementCudaIntegrationTest.roomyTokens(all);
        var loaded = SharedFlashNext.model(SharedFlashNext.SMALL_CACHE);
        var model = loaded.model();
        assertTrue(model.expertCache().slotCount() <= 64);
        long evictions = model.expertCache().stats().snapshot().evictions();
        // Three of the six prompts (a short one, a 97-token one and a chat-templated one): with 20 slots every step
        // evicts and re-reads experts whatever the prompt, so more prompts only repeat the expensive streaming.
        int[] picked = {0, 4, 5};
        int[][] minimal = GreedyAgreementCudaIntegrationTest.teacherForced(
                loaded, java.util.Arrays.stream(picked).mapToObj(all::get).toList());
        assertTrue(model.expertCache().stats().snapshot().evictions() > evictions);
        var tiers = model.hierarchyStats();
        System.out.println("minimal cache: " + model.expertCache().slotCount() + " device slots, host store "
                + model.plan().expertStore()
                + (tiers.ram() == null
                        ? ""
                        : ", " + tiers.ram().residentExperts() + " resident in RAM, "
                                + tiers.ram().totalEvictions() + " RAM evictions, "
                                + tiers.ram().totalHits()
                                + " RAM hits")
                + ", " + tiers.artifact().recordReads() + " artifact reads");
        for (int i = 0; i < picked.length; i++)
            assertArrayEquals(
                    roomy[picked[i]], minimal[i], "the cache capacity changed the tokens of prompt " + picked[i]);
    }

    @Test
    void thePrefillLogitsDoNotDependOnTheCacheCapacity() throws Exception {
        int[] tokens = InvarianceCudaIntegrationTest.prompt();
        short[] roomy = InvarianceCudaIntegrationTest.roomyLogits(tokens);
        var loaded = SharedFlashNext.model(SharedFlashNext.SMALL_CACHE);
        var gpu = loaded.gpu();
        var model = loaded.model();
        ExecutionPlan executor = loaded.plan();
        System.out.println("small cache: " + model.expertCache().slotCount() + " slots");
        assertTrue(model.expertCache().slotCount() <= 64, "expected a small cache");
        long evictions = model.expertCache().stats().snapshot().evictions();
        short[] small =
                InvarianceCudaIntegrationTest.prefill(gpu, executor, tokens, InvarianceCudaIntegrationTest.CUT, 0);
        assertArrayEquals(roomy, small, "a small cache changed the result");
        assertTrue(model.expertCache().stats().snapshot().evictions() > evictions);
    }
}
