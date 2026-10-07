package io.euhedral_execution.inference.core.model_loader.qwen4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/// The context/residency matrix: for a range of device budgets and contexts at the real model's object sizes,
/// the plan's invariants (not particular placements, which several layouts could satisfy).
class Qwen4ResidencyPlannerTest {

    static final long GIB = 1L << 30;
    static final int[] CONTEXTS = {4096, 16384, 32768, 65536, 131072, 262144};
    static final long[] DEVICES = {
        3 * GIB, 4 * GIB, 5 * GIB, 6 * GIB, 7 * GIB, 8 * GIB, 10 * GIB, 12 * GIB, 15 * GIB, 24 * GIB, 48 * GIB, 96 * GIB
    };
    static final HostBudget HOST = HostBudget.ofAvailable(48 * GIB);

    final Qwen4Artifact artifact = Qwen4TestArtifact.virtual(Qwen4TestArtifact.realConfig());

    Qwen4ResidencyPlan plan(long free, int context) {
        return Qwen4ResidencyPlanner.plan(this.artifact, Qwen4Mode.TEXT, free, HOST, context);
    }

    long hostBackedFixedBytes(Qwen4ResidencyPlan plan) {
        return plan.host().stagedBytes() + mappedFixed(plan);
    }

    /// Mapped bytes other than the token embedding.
    long mappedFixed(Qwen4ResidencyPlan plan) {
        long embedding =
                this.artifact.tensor("text/token_embedding").orElseThrow().byteSize();
        return plan.host().mappedBytes()
                - (plan.storageOf("text/token_embedding") == StorageClass.HOST_MAPPED ? embedding : 0);
    }

    @Test
    void everyFittingPlanStaysWithinTheDeviceAndMeetsTheMinimumCache() {
        for (long free : DEVICES) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (!plan.fits()) continue;
                assertTrue(plan.device().plannedBytes() <= free, report(plan, free, context));
                assertTrue(
                        plan.expertCache().slotCount() >= plan.expertCache().minimumSlots(),
                        report(plan, free, context));
                assertEquals(context, plan.maxContextTokens());
                // the context is reserved in full
                assertEquals(
                        Qwen4SequenceState.kvBytes(artifact.config(), context),
                        plan.device().kvBytes());
                assertTrue(plan.device().slackBytes() >= 0);
            }
        }
    }

    @Test
    void mandatoryObjectsAreResidentAndOtherComponentsHaveTheirStorage() {
        for (long free : DEVICES) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (!plan.fits()) continue;
                for (Qwen4Tensor tensor : artifact.tensors()) {
                    StorageClass storage = plan.storageOf(tensor.name());
                    switch (tensor.group()) {
                        case ROUTER, NORM_SMALL_STATE ->
                            assertEquals(StorageClass.DEVICE_RESIDENT, storage, tensor.name());
                        case MTP, VISION -> assertEquals(StorageClass.DEFERRED, storage, tensor.name());
                        case NGRAM -> assertEquals(StorageClass.HOST_STAGED, storage, tensor.name());
                        case TOKEN_EMBEDDING -> assertEquals(StorageClass.HOST_MAPPED, storage);
                        default -> {
                            if (tensor.byteSize() <= Qwen4Priority.SMALL_BYTES)
                                assertEquals(StorageClass.DEVICE_RESIDENT, storage, tensor.name());
                            assertTrue(
                                    storage != StorageClass.DEFERRED && storage != StorageClass.DEVICE_CACHED,
                                    tensor.name());
                        }
                    }
                }
                for (ExpertBank bank : artifact.banks())
                    assertEquals(
                            bank.group() == ComponentGroup.MTP ? StorageClass.DEFERRED : StorageClass.DEVICE_CACHED,
                            plan.storageOf(bank.name()));
            }
        }
    }

    @Test
    void theExpertCacheTakesTheRemainingBudgetAndShrinksAsContextGrows() {
        for (long free : DEVICES) {
            Qwen4ResidencyPlan previous = null;
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (!plan.fits()) {
                    previous = null;
                    continue;
                }
                var cache = plan.expertCache();
                if (cache.slotCount() < cache.totalExperts())
                    assertTrue(
                            plan.device().slackBytes() < cache.slotBytes(),
                            "unused " + plan.device().slackBytes() + " " + report(plan, free, context));
                // Host-backed bytes only grow with the context. While they and the prefill chunk do not change, a
                // longer
                // context leaves the cache less; when a fixed object moves to the host, or the chunk shrinks, the room
                // it
                // frees may enlarge the cache again.
                if (previous != null) {
                    assertTrue(hostBackedFixedBytes(plan) >= hostBackedFixedBytes(previous), "host-backed bytes fell");
                    if (hostBackedFixedBytes(plan) == hostBackedFixedBytes(previous)
                            && plan.prefillChunkTokens() == previous.prefillChunkTokens())
                        assertTrue(
                                cache.slotCount() <= previous.expertCache().slotCount(),
                                "cache grew with the context: " + free / (1 << 20) + " MiB, context " + context
                                        + ", chunk "
                                        + previous.prefillChunkTokens() + " to " + plan.prefillChunkTokens()
                                        + ", slots "
                                        + previous.expertCache().slotCount() + " to " + cache.slotCount());
                }
                previous = plan;
            }
        }
        // a larger context strictly shrinks an uncapped cache
        assertTrue(plan(24 * GIB, 262144).expertCache().slotCount()
                < plan(24 * GIB, 4096).expertCache().slotCount());
    }

    @Test
    void moreDeviceMemoryNeverAddsHostBackedBytesAndGrowsTheCacheWhileNothingMoves() {
        for (int context : CONTEXTS) {
            Qwen4ResidencyPlan previous = null;
            for (long free : DEVICES) {
                var plan = plan(free, context);
                if (!plan.fits()) continue;
                if (previous != null) {
                    assertTrue(
                            hostBackedFixedBytes(plan) <= hostBackedFixedBytes(previous),
                            "host-backed bytes rose with more memory");
                    if (hostBackedFixedBytes(plan) == hostBackedFixedBytes(previous))
                        assertTrue(
                                plan.expertCache().slotCount()
                                        >= previous.expertCache().slotCount(),
                                "cache shrank with more memory");
                }
                previous = plan;
            }
        }
    }

    @Test
    void fixedObjectsLeaveTheDeviceOnlyWhenTheMinimumCacheNeedsTheRoom() {
        long fixedResident = 0;
        for (Qwen4Tensor tensor : artifact.tensors())
            if (tensor.group() != ComponentGroup.NGRAM
                    && tensor.group() != ComponentGroup.MTP
                    && tensor.group() != ComponentGroup.VISION
                    && tensor.group() != ComponentGroup.TOKEN_EMBEDDING) fixedResident += tensor.byteSize();
        long minimumCache =
                Qwen4ResidencyPlanner.minimumSlots(artifact.config()) * artifact.banks()[0].maxRecordBytes();
        boolean sawOffload = false;
        for (long free : DEVICES) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (!plan.fits()) continue;
                long reserved = plan.device().contextBytes()
                        + plan.device().workspaceBytes()
                        + plan.device().runtimeReserveBytes();
                if (hostBackedFixedBytes(plan) > 0) {
                    sawOffload = true;
                    // with everything resident the minimum cache would not have fit
                    assertTrue(fixedResident + reserved + minimumCache > free, report(plan, free, context));
                } else {
                    // nothing moved: the cache is whatever the budget leaves, never below the minimum
                    assertTrue(
                            plan.expertCache().slotCount() >= plan.expertCache().minimumSlots());
                }
            }
        }
        assertTrue(sawOffload, "the matrix never pressed the fixed objects");
    }

    @Test
    void plansAreDeterministic() {
        for (long free : new long[] {5 * GIB, 15 * GIB})
            for (int context : CONTEXTS) assertEquals(plan(free, context), plan(free, context));
    }

    @Test
    void impossibleConfigurationsAreRefusedWithTheReason() {
        var tiny = plan(1 * GIB, 4096);
        assertFalse(tiny.fits());
        assertNotNull(tiny.explanation());
        assertTrue(tiny.explanation().contains("MiB free"), tiny.explanation());
        assertTrue(tiny.explanation().contains("smallest expert cache"), tiny.explanation());

        var beyondTrained = plan(96 * GIB, 262145);
        assertFalse(beyondTrained.fits());
        assertTrue(beyondTrained.explanation().contains("262144 positions"), beyondTrained.explanation());

        // the largest context fits where the matrix says the minimum does not
        var edge = plan(3 * GIB, 262144);
        assertFalse(edge.fits(), edge.report());
        assertTrue(edge.report().contains("does not fit"));
    }

    @Test
    void contextIsOnlyRefusedWhenNoPlacementCanServeIt() {
        long minimumCache =
                Qwen4ResidencyPlanner.minimumSlots(artifact.config()) * artifact.banks()[0].maxRecordBytes();
        boolean moved = false;
        boolean refused = false;
        for (long free : DEVICES) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (plan.fits()) {
                    moved |= hostBackedFixedBytes(plan) > 0;
                    continue;
                }
                refused = true;
                // refused: every movable object is already on the host and the minimum cache still does not fit
                for (Qwen4Tensor tensor : artifact.tensors())
                    if (Qwen4Priority.offloadRank(tensor) >= 0 && tensor.group() != ComponentGroup.NGRAM)
                        assertTrue(
                                plan.storageOf(tensor.name()) != StorageClass.DEVICE_RESIDENT
                                        || tensor.group() == ComponentGroup.TOKEN_EMBEDDING,
                                tensor.name() + " stayed on the device of a refused plan");
                assertTrue(plan.device().plannedBytes() + minimumCache > free, report(plan, free, context));
            }
        }
        assertTrue(moved, "no device in the matrix needed fixed objects on the host");
        assertTrue(refused, "no device in the matrix was too small for a context");
    }

    @Test
    void selectingMtpOrVisionLoadsThemAndEnlargesTheNeed() {
        var text = Qwen4ResidencyPlanner.plan(artifact, Qwen4Mode.TEXT, 15 * GIB, HOST, 32768);
        var withMtp = Qwen4ResidencyPlanner.plan(artifact, new Qwen4Mode(true, false), 15 * GIB, HOST, 32768);
        var withVision = Qwen4ResidencyPlanner.plan(artifact, new Qwen4Mode(false, true), 15 * GIB, HOST, 32768);
        assertEquals(StorageClass.DEFERRED, text.storageOf("mtp/fc_embedding"));
        assertEquals(StorageClass.DEVICE_RESIDENT, withMtp.storageOf("mtp/fc_embedding"));
        assertEquals(StorageClass.DEVICE_CACHED, withMtp.storageOf("mtp/layers/0/moe/experts"));
        assertEquals(StorageClass.DEVICE_RESIDENT, withVision.storageOf("vision/pos_embed/weight"));
        assertTrue(withMtp.device().fixedResidentBytes() > text.device().fixedResidentBytes());
        assertTrue(withVision.device().fixedResidentBytes() > text.device().fixedResidentBytes());
        assertTrue(text.host().deferredMtpBytes() > 0 && withMtp.host().deferredMtpBytes() == 0);
        assertTrue(text.host().deferredVisionBytes() > 0 && withVision.host().deferredVisionBytes() == 0);
    }

    @Test
    void theCacheGivesUpAtMostATenthOfItsMemoryToALongerPrefillChunk() {
        var config = artifact.config();
        long padding = Qwen4ResidencyPlanner.sharedDownPaddingBytes(config);
        long base = Qwen4SequenceState.workspaceBytes(config) + padding;
        for (long free : DEVICES) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                if (!plan.fits()) continue;
                int chunk = plan.prefillChunkTokens();
                assertTrue(chunk >= Qwen4SequenceState.PREFILL_CHUNK_TOKENS, report(plan, free, context));
                assertTrue(chunk <= Qwen4ResidencyPlanner.LARGEST_PREFILL_CHUNK_TOKENS);
                assertEquals(Integer.bitCount(chunk), 1, "a power of two: " + chunk);
                // The workspace the plan reserves is every workspace the execution plan allocates, and the shared
                // expert's padded down projections.
                assertEquals(
                        Qwen4SequenceState.workspaceBytes(config, chunk) + padding,
                        plan.device().workspaceBytes());
                if (chunk > Qwen4SequenceState.PREFILL_CHUNK_TOKENS) {
                    long extra = plan.device().workspaceBytes() - base;
                    long budget =
                            plan.device().expertCacheBytes() + plan.device().slackBytes() + extra;
                    assertTrue(extra <= budget / 10, report(plan, free, context));
                    assertTrue(chunk / 2 < context, "no chunk longer than the context needs");
                }
            }
        }
        assertEquals(
                Qwen4ResidencyPlanner.LARGEST_PREFILL_CHUNK_TOKENS,
                plan(96 * GIB, 32768).prefillChunkTokens(),
                "a roomy device takes the largest chunk");
        assertEquals(
                Qwen4SequenceState.PREFILL_CHUNK_TOKENS,
                plan(5 * GIB, 32768).prefillChunkTokens(),
                "a device with little to spare keeps its slots");
    }

    @Test
    void hostPlacementFollowsTheHostBudget() {
        var roomy = Qwen4ResidencyPlanner.plan(
                artifact, Qwen4Mode.TEXT, 15 * GIB, HostBudget.ofAvailable(256 * GIB), 32768);
        assertEquals(Qwen4ResidencyPlan.ExpertStoreMode.RAM_RESIDENT, roomy.expertStore());
        assertEquals(Qwen4ResidencyPlan.NgramMode.PINNED_ARENA, roomy.ngram());
        // Pinned memory stays a small tier however much memory there is: staging, never the experts.
        assertTrue(roomy.host().expertStagingPinnedBytes() < roomy.host().expertRamBytes() / 100);
        var cached =
                Qwen4ResidencyPlanner.plan(artifact, Qwen4Mode.TEXT, 15 * GIB, HostBudget.ofAvailable(40 * GIB), 32768);
        assertEquals(Qwen4ResidencyPlan.ExpertStoreMode.RAM_CACHED, cached.expertStore());
        assertEquals(Qwen4ResidencyPlan.NgramMode.MAPPED_FILE, cached.ngram());
        assertTrue(cached.host().expertRamSlots() > cached.expertCache().slotCount());
        assertTrue(cached.host().expertRamSlots() < cached.expertCache().totalExperts());
        assertTrue(cached.host().expertStagingPinnedBytes() < (1L << 30));
        assertTrue(cached.host().residentBytes()
                <= HostBudget.ofAvailable(40 * GIB).residentBytes());
        var tight =
                Qwen4ResidencyPlanner.plan(artifact, Qwen4Mode.TEXT, 15 * GIB, HostBudget.ofAvailable(8 * GIB), 32768);
        assertEquals(Qwen4ResidencyPlan.ExpertStoreMode.FILE_BACKED, tight.expertStore());
        assertEquals(0, tight.host().expertRamSlots());
    }

    @Test
    void printsTheMatrix() {
        List<String> rows = new ArrayList<>();
        for (long free : new long[] {3 * GIB, 4 * GIB, 5 * GIB, 6 * GIB, 7 * GIB, 8 * GIB, 15 * GIB, 24 * GIB}) {
            for (int context : CONTEXTS) {
                var plan = plan(free, context);
                rows.add(String.format(
                        "free %2d GiB ctx %6d: %s slots %5d cache %5d MiB resident %5d MiB host-backed %5d MiB",
                        free >> 30,
                        context,
                        plan.fits() ? "fits" : "NO  ",
                        plan.expertCache().slotCount(),
                        plan.device().expertCacheBytes() >> 20,
                        plan.device().fixedResidentBytes() >> 20,
                        hostBackedFixedBytes(plan) >> 20));
            }
        }
        System.out.println(String.join("\n", rows));
    }

    private String report(Qwen4ResidencyPlan plan, long free, int context) {
        return "free " + (free >> 20) + " MiB context " + context + "\n" + plan.report();
    }
}
