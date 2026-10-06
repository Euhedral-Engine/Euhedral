package io.euhedral_execution.inference.core.qwen4;

import static io.euhedral_execution.inference.core.qwen4.Qwen4Reference.bf;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.downloadBf16;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.error;
import static io.euhedral_execution.inference.core.qwen4.Qwen4TestSupport.upload;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertBank;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCache;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.FileExpertStore;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.GpuExpertTransfer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/// The whole MoE block of layer 0 (router, shared expert, routed experts through the expert cache,
/// combination) against the upstream fixtures, and the cache's independence: the same block with
/// the minimum cache (20 slots, so every chunk evicts) and with a cache that holds every expert
/// must give the same bits.
class Qwen4MoeFixtureCudaIntegrationTest {

    private static short[] runBlock(
            CudaGpuMemory gpu,
            Arena arena,
            Qwen4TestSupport.Weights weights,
            ExpertBank bank,
            int slots,
            ReferenceFixtures fixture,
            int layerIndex,
            int chunks,
            Qwen4TestSupport.Report report,
            double bound)
            throws Exception {
        short[] all = new short[0];
        var store = new FileExpertStore(gpu, weights.path(), new ExpertBank[] {bank}, 16);
        var transfer = new GpuExpertTransfer(gpu, 4);
        try (ExpertCache cache =
                new ExpertCache(store, transfer, gpu, slots, ExpertCache.slotBytesFor(new ExpertBank[] {bank}), 1)) {
            GpuStream stream = gpu.openStream();
            var geometry = Qwen4ExpertOps.Geometry.of(bank);
            var layer = new Qwen4MoeLayer(gpu, geometry, 512, 10, 640, 64, new Qwen4MoeLayer.Metrics());
            var moeWeights = weights_for(weights, layerIndex);
            try {
                for (int chunk = 0; chunk < chunks; chunk++) {
                    String at = "c" + chunk + "/L" + layerIndex + "/moe/";
                    short[] input = fixture.bf16(at + "in");
                    int rows = input.length / 2560;
                    long inputAddress = upload(gpu, arena, input);
                    long outputAddress = gpu.allocate((long) rows * 2560 * 2);
                    long scratchAddress = gpu.allocate(layer.scratchBytes(rows));
                    try {
                        var scratch = layer.scratch(scratchAddress, rows);
                        stream.submit(() -> {}, false);
                        Qwen4Blocking.runMoe(
                                layer, stream, cache, moeWeights, 0, inputAddress, rows, scratch, outputAddress);
                        Qwen4Streams.awaitCompletion(stream);
                        checkRouting(gpu, arena, fixture, at, scratch, rows);
                        report.check(
                                at + "shared_out",
                                error(
                                        fixture.bf16(at + "shared_out"),
                                        downloadBf16(gpu, arena, scratch.shared(), rows * 2560)),
                                bound);
                        short[] gate = downloadBf16(gpu, arena, scratch.gateRaw(), rows);
                        short[] expectedGate = fixture.bf16(at + "shared_gate");
                        short[] sigmoid = new short[rows];
                        for (int i = 0; i < rows; i++)
                            sigmoid[i] = Qwen4Reference.bits(Qwen4Reference.sigmoid(bf(gate[i])));
                        report.check(at + "shared_gate", error(expectedGate, sigmoid), bound);
                        report.check(
                                at + "routed_sum",
                                error(
                                        fixture.bf16(at + "routed_sum"),
                                        downloadBf16(gpu, arena, scratch.routed(), rows * 2560)),
                                bound);
                        short[] out = downloadBf16(gpu, arena, outputAddress, rows * 2560);
                        report.check(at + "out", error(fixture.bf16(at + "out"), out), bound);
                        short[] merged = Arrays.copyOf(all, all.length + out.length);
                        System.arraycopy(out, 0, merged, all.length, out.length);
                        all = merged;
                    } finally {
                        gpu.free(scratchAddress);
                        gpu.free(outputAddress);
                        gpu.free(inputAddress);
                    }
                }
            } finally {
                Qwen4Streams.awaitCompletion(stream);
                layer.close();
                stream.close();
            }
            System.out.println("slots " + slots + ": " + cache.stats().snapshot());
            if (slots < 100)
                assertTrue(cache.stats().snapshot().evictions() > 0, "the minimum cache must have evicted");
        }
        return all;
    }

    private static Qwen4MoeLayer.Weights weights_for(Qwen4TestSupport.Weights weights, int layerIndex)
            throws Exception {
        String p = "text/layers/" + layerIndex + "/moe/";
        return new Qwen4MoeLayer.Weights(
                weights.weight(p + "router"),
                weights.weight(p + "shared_expert/gate_proj"),
                weights.weight(p + "shared_expert/up_proj"),
                weights.weight(p + "shared_expert/down_proj"),
                weights.weight(p + "shared_expert_gate"));
    }

    /// The router's choice as a set per row, with the reference's weights per expert. A row whose
    /// 10th and 11th probabilities tie in the reference may legitimately differ in the tied expert.
    private static void checkRouting(
            CudaGpuMemory gpu,
            Arena arena,
            ReferenceFixtures fixture,
            String at,
            Qwen4MoeLayer.Scratch scratch,
            int rows)
            throws Exception {
        MemorySegment host = arena.allocate(rows * 40L, 16);
        gpu.copyDeviceToHost(host, scratch.ids(), rows * 40L);
        int[] ids = new int[rows * 10];
        MemorySegment.copy(host, ValueLayout.JAVA_INT, 0, ids, 0, ids.length);
        short[] weights = downloadBf16(gpu, arena, scratch.routeWeights(), rows * 10);
        int[] expected = fixture.i32(at + "topk_ids");
        short[] expectedWeights = fixture.bf16(at + "topk_weights");
        byte[] tie = fixture.u8(at + "boundary_tie");
        for (int row = 0; row < rows; row++) {
            Map<Integer, Float> want = new HashMap<>();
            for (int t = 0; t < 10; t++) want.put(expected[row * 10 + t], bf(expectedWeights[row * 10 + t]));
            int missing = 0;
            for (int t = 0; t < 10; t++) {
                Float w = want.get(ids[row * 10 + t]);
                if (w == null) {
                    missing++;
                    continue;
                }
                assertEquals(
                        w,
                        bf(weights[row * 10 + t]),
                        Math.abs(w) * 0.02 + 1e-6,
                        at + " row " + row + " weight of " + ids[row * 10 + t]);
            }
            assertTrue(
                    missing == 0 || tie[row] != 0,
                    at + " row " + row + " selects " + missing + " experts the reference does not");
        }
    }

    @Test
    void theMoeBlockFollowsTheReferenceWhateverTheCacheHolds() throws Exception {
        Path directory = Qwen4TestSupport.fixtureRoot().resolve("layer_moe");
        assumeTrue(Qwen4TestSupport.hasArtifact() && ReferenceFixtures.exists(directory), "no artifact or fixtures");
        ReferenceFixtures fixture = new ReferenceFixtures(directory);
        int chunks = fixture.metadata().get("chunks") == null
                ? 3
                : fixture.metadata().get("chunks").size();
        Qwen4TestSupport.Report report = new Qwen4TestSupport.Report("layer 0 MoE block");
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined();
                Qwen4TestSupport.Weights weights = new Qwen4TestSupport.Weights(gpu)) {
            ExpertBank bank =
                    weights.artifact().bank("text/layers/0/moe/experts").orElseThrow();
            for (boolean exact : new boolean[] {true, false}) {
                gpu.selectExactNumerics(exact);
                double bound = exact ? 3e-3 : 2e-2;
                short[] minimal = runBlock(gpu, arena, weights, bank, 20, fixture, 0, chunks, report, bound);
                short[] roomy = runBlock(gpu, arena, weights, bank, 520, fixture, 0, chunks, report, bound);
                assertArrayEquals(minimal, roomy, "the cache's contents changed the result");
            }
            gpu.selectExactNumerics(false);
        }
        report.finish();
    }

    /// A QSA layer's MoE block (layer 3 of the model case `short`) on the reference's own input:
    /// the block alone agrees with the reference far better than the whole layer does when the
    /// attention block's small error reaches the router (the reason deep layers of the model
    /// comparison show a few percent).
    @Test
    void aMoeBlockOfAnAttentionLayerOnTheReferenceInput() throws Exception {
        Path directory = Qwen4TestSupport.modelFixtureRoot().resolve("short");
        assumeTrue(Qwen4TestSupport.hasArtifact() && ReferenceFixtures.exists(directory), "no artifact or fixtures");
        ReferenceFixtures fixture = new ReferenceFixtures(directory);
        Qwen4TestSupport.Report report = new Qwen4TestSupport.Report("layer 3 MoE block, case short");
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined();
                Qwen4TestSupport.Weights weights = new Qwen4TestSupport.Weights(gpu)) {
            ExpertBank bank =
                    weights.artifact().bank("text/layers/3/moe/experts").orElseThrow();
            gpu.selectExactNumerics(true);
            runBlock(gpu, arena, weights, bank, 520, fixture, 3, 4, report, 3e-3);
            gpu.selectExactNumerics(false);
        }
        report.finish();
    }

    @Test
    void theMoeBlockOfLayerZeroOnTheEosCase() throws Exception {
        Path directory = Qwen4TestSupport.modelFixtureRoot().resolve("eos");
        assumeTrue(Qwen4TestSupport.hasArtifact() && ReferenceFixtures.exists(directory), "no artifact or fixtures");
        ReferenceFixtures fixture = new ReferenceFixtures(directory);
        Qwen4TestSupport.Report report = new Qwen4TestSupport.Report("layer 0 MoE block, case eos");
        try (CudaGpuMemory gpu = Qwen4TestSupport.openGpu();
                Arena arena = Arena.ofConfined();
                Qwen4TestSupport.Weights weights = new Qwen4TestSupport.Weights(gpu)) {
            ExpertBank bank =
                    weights.artifact().bank("text/layers/0/moe/experts").orElseThrow();
            runBlock(gpu, arena, weights, bank, 520, fixture, 0, 3, report, 3e-2);
        }
        report.finish();
    }
}
