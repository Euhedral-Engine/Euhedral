package io.euhedral_execution.inference.core.model.qwen38.prefix;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionFixtures;
import io.euhedral_execution.inference.core.model.qwen38.GdnState;
import io.euhedral_execution.inference.core.model.qwen38.GdnStates;
import io.euhedral_execution.inference.core.model.qwen38.HeldWork;
import io.euhedral_execution.inference.core.model.qwen38.MemoryGpu;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.speculative.MtpCheckpoint;
import io.euhedral_execution.inference.core.prefix.HostExtents;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.prefix.PrefixTree;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PrefixLayoutTest {
    private static final Qwen38Config CONFIG =
            ExecutionFixtures.statefulCompactWeights(8).config();

    private final MemoryGpu gpu = new MemoryGpu();
    private final PrefixLayout layout = PrefixLayout.of(CONFIG);

    private GdnStates gdn() {
        return GdnStates.allocate(
                this.gpu,
                CONFIG.layerTypes(),
                CONFIG.linearNumKeyHeads(),
                CONFIG.linearNumValueHeads(),
                CONFIG.linearKeyHeadDim(),
                CONFIG.linearValueHeadDim(),
                CONFIG.linearConvKernelDim());
    }

    private AttentionStates attention() {
        return AttentionStates.allocate(
                this.gpu, CONFIG.layerTypes(), CONFIG.numKeyValueHeads() * CONFIG.attentionHeadDim());
    }

    /// Reserves and commits `rows` rows of KV in the attention layer, as a prefill would.
    private static void commitRows(AttentionStates states, int rows) {
        AttentionKvState state = states.forLayer(1);
        state.prepareAppend(0, rows);
        state.appendSubmitted(rows);
        state.commitSubmitted();
    }

    private void fillState(GdnStates gdn, AttentionStates attention, int seed) {
        GdnState g = gdn.forLayer(0);
        this.gpu.fill(g.convolutionStateAddress(), (int) g.convolutionBytes(), seed);
        this.gpu.fill(g.recurrentStateAddress(), (int) g.recurrentBytes(), seed + 1);
        List<Long> pages = attention.forLayer(1).pageAddresses();
        for (int i = 0; i < pages.size(); i++)
            this.gpu.fill(pages.get(i), (int) (2 * attention.forLayer(1).planePageBytes()), seed + 10 + i);
    }

    private void capture(MemorySegment host, PrefixNode node, GdnStates gdn, AttentionStates attention) {
        for (var copy : this.layout.captureCopies(node, gdn, attention))
            this.gpu.copyDeviceToHost(
                    host.asSlice(copy.hostOffset(), copy.bytes()), copy.deviceAddress(), copy.bytes());
    }

    private void restore(MemorySegment host, List<PrefixNode> chain, GdnStates gdn, AttentionStates attention) {
        for (var copy : this.layout.restoreCopies(chain, gdn, attention))
            this.gpu.copyHostToDevice(
                    copy.deviceAddress(), host.asSlice(copy.hostOffset(), copy.bytes()), copy.bytes());
    }

    @Test
    void sizesAMatchingGeometryToTheStateClasses() {
        try (var gdn = gdn();
                var attention = attention()) {
            GdnState g = gdn.forLayer(0);
            long page = 2 * attention.forLayer(1).planePageBytes();
            assertEquals(g.convolutionBytes() + g.recurrentBytes() + 3 * page, this.layout.extentBytes(0, 768));
            assertEquals(
                    g.convolutionBytes() + g.recurrentBytes() + 2 * page,
                    this.layout.extentBytes(512, 1024),
                    "pages 2 and 3 of a span that starts at 512");
        }
    }

    @Test
    void capturingThenRestoringReproducesTheStateByteForByte() {
        var tree = new PrefixTree(new HostExtents(8L << 20));
        int[] tokens = IntStream.range(0, 768).toArray();
        try (Arena arena = Arena.ofConfined();
                var gdn = gdn();
                var attention = attention()) {
            MemorySegment host = arena.allocate(8L << 20);
            commitRows(attention, 768);
            fillState(gdn, attention, 3);
            PrefixNode node = tree.reserve(tree.root(), tokens, 768, null, this.layout.extentBytes(0, 768));
            capture(host, node, gdn, attention);
            tree.publish(node);

            try (var gdn2 = gdn();
                    var attention2 = attention()) {
                commitRows(attention2, 768);
                restore(host, List.of(node), gdn2, attention2);
                GdnState a = gdn.forLayer(0);
                GdnState b = gdn2.forLayer(0);
                assertArrayEquals(
                        this.gpu.bytes(a.convolutionStateAddress(), (int) a.convolutionBytes()),
                        this.gpu.bytes(b.convolutionStateAddress(), (int) b.convolutionBytes()));
                assertArrayEquals(
                        this.gpu.bytes(a.recurrentStateAddress(), (int) a.recurrentBytes()),
                        this.gpu.bytes(b.recurrentStateAddress(), (int) b.recurrentBytes()));
                int pageBytes = (int) (2 * attention.forLayer(1).planePageBytes());
                for (int page = 0; page < 3; page++)
                    assertArrayEquals(
                            this.gpu.bytes(attention.forLayer(1).pageAddresses().get(page), pageBytes),
                            this.gpu.bytes(
                                    attention2.forLayer(1).pageAddresses().get(page), pageBytes),
                            "page " + page);
            }
        }
    }

    @Test
    void aChainRestoresEachNodesPagesAndTheLeafsGdnState() {
        var tree = new PrefixTree(new HostExtents(16L << 20));
        int[] tokens = IntStream.range(0, 1280).toArray();
        try (Arena arena = Arena.ofConfined();
                var gdn = gdn();
                var attention = attention()) {
            MemorySegment host = arena.allocate(16L << 20);
            commitRows(attention, 1280);
            int pageBytes = (int) (2 * attention.forLayer(1).planePageBytes());
            fillState(gdn, attention, 5);
            PrefixNode first = tree.reserve(tree.root(), tokens, 512, null, this.layout.extentBytes(0, 512));
            capture(host, first, gdn, attention);
            tree.publish(first);
            // Pages 0 and 1 (rows 0-511) are the first node's, as filled now.
            byte[][] firstPages = new byte[2][];
            for (int page = 0; page < 2; page++)
                firstPages[page] =
                        this.gpu.bytes(attention.forLayer(1).pageAddresses().get(page), pageBytes);
            // The sequence moves on: every buffer is rewritten, then the second node (pages 2-4) is captured.
            fillState(gdn, attention, 9);
            PrefixNode second = tree.reserve(first, tokens, 1280, null, this.layout.extentBytes(512, 1280));
            capture(host, second, gdn, attention);
            tree.publish(second);

            try (var gdn2 = gdn();
                    var attention2 = attention()) {
                commitRows(attention2, 1280);
                restore(host, List.of(first, second), gdn2, attention2);
                for (int page = 0; page < 2; page++)
                    assertArrayEquals(
                            firstPages[page],
                            this.gpu.bytes(
                                    attention2.forLayer(1).pageAddresses().get(page), pageBytes),
                            "page " + page + " comes from the first node");
                for (int page = 2; page < 5; page++)
                    assertArrayEquals(
                            this.gpu.bytes(attention.forLayer(1).pageAddresses().get(page), pageBytes),
                            this.gpu.bytes(
                                    attention2.forLayer(1).pageAddresses().get(page), pageBytes),
                            "page " + page + " comes from the second node");
                GdnState a = gdn.forLayer(0);
                GdnState b = gdn2.forLayer(0);
                assertArrayEquals(
                        this.gpu.bytes(a.recurrentStateAddress(), (int) a.recurrentBytes()),
                        this.gpu.bytes(b.recurrentStateAddress(), (int) b.recurrentBytes()),
                        "the leaf's GDN state, not the first node's");
            }
        }
    }

    // --- MTP state: the MTP cache's rows [0, position - 1) and the base hidden row of position - 1.

    private final MtpCheckpoint mtp = new MtpCheckpoint(CONFIG);

    private static Sequence holding(AttentionStates attention) {
        var sequence = new Sequence(1);
        var held = HeldWork.admit(sequence);
        sequence.setKvCacheState(attention);
        held.commit(0);
        return sequence;
    }

    private AttentionStates attentionWithMtp() {
        return AttentionStates.allocate(
                this.gpu, CONFIG.layerTypes(), CONFIG.numKeyValueHeads() * CONFIG.attentionHeadDim(), true);
    }

    private static void commitMtpRows(AttentionStates states, int rows) {
        AttentionKvState state = states.forLayer(CONFIG.numHiddenLayers());
        state.prepareAppend(0, rows);
        state.appendSubmitted(rows);
        state.commitSubmitted();
    }

    private void fillMtpPages(AttentionStates attention, int seed) {
        AttentionKvState mtp = attention.forLayer(CONFIG.numHiddenLayers());
        List<Long> pages = mtp.pageAddresses();
        for (int i = 0; i < pages.size(); i++) this.gpu.fill(pages.get(i), (int) (2 * mtp.planePageBytes()), seed + i);
    }

    private void captureMtp(MemorySegment host, PrefixNode node, GdnStates gdn, AttentionStates attention, long seed) {
        this.mtp.seedRow(seed);
        var copies = new java.util.ArrayList<>(this.layout.captureCopies(node, gdn, attention));
        copies.addAll(this.mtp.captureCopies(node, this.layout.speculativeOffset(node), holding(attention)));
        for (var copy : copies)
            this.gpu.copyDeviceToHost(
                    host.asSlice(copy.hostOffset(), copy.bytes()), copy.deviceAddress(), copy.bytes());
    }

    private void restoreMtp(MemorySegment host, List<PrefixNode> chain, GdnStates gdn, AttentionStates attention) {
        Sequence sequence = holding(attention);
        var copies = new java.util.ArrayList<>(this.layout.restoreCopies(chain, gdn, attention));
        copies.addAll(this.mtp.restoreCopies(chain, this.layout::speculativeOffset, sequence));
        for (var copy : copies)
            this.gpu.copyHostToDevice(
                    copy.deviceAddress(), host.asSlice(copy.hostOffset(), copy.bytes()), copy.bytes());
        this.mtp.restored(sequence, chain.getLast().position());
    }

    private long restoredSeed(AttentionStates attention) {
        return attention.draftSeedRows(1, CONFIG.hiddenSize());
    }

    @Test
    void mtpExtentsAddTheSeedRowAndOnlyTheMtpPagesBelowPositionMinusOne() {
        try (var attention = attentionWithMtp()) {
            long page = 2 * attention.forLayer(CONFIG.numHiddenLayers()).planePageBytes();
            long seedRow = (long) CONFIG.hiddenSize() * Short.BYTES;
            // Rows [0, 767) fill 3 pages.
            assertEquals(seedRow + 3 * page, this.mtp.extentBytes(0, 768));
            // A span from 512 starts one MTP page early (the page of row 511): pages 1, 2 and 3 for rows below 1023.
            assertEquals(seedRow + 3 * page, this.mtp.extentBytes(512, 1024));
        }
    }

    @Test
    void capturingThenRestoringReproducesTheMtpCacheAndTheSeedRow() {
        var tree = new PrefixTree(new HostExtents(8L << 20));
        int[] tokens = IntStream.range(0, 768).toArray();
        try (Arena arena = Arena.ofConfined();
                var gdn = gdn();
                var attention = attentionWithMtp();
                var gdn2 = gdn();
                var attention2 = attentionWithMtp()) {
            MemorySegment host = arena.allocate(8L << 20);
            commitRows(attention, 768);
            commitMtpRows(attention, 767);
            fillState(gdn, attention, 3);
            fillMtpPages(attention, 40);
            long seed = this.gpu.allocate(256);
            this.gpu.fill(seed, 256, 77);
            PrefixNode node = tree.reserve(
                    tree.root(),
                    tokens,
                    768,
                    MtpCheckpoint.KIND,
                    this.layout.extentBytes(0, 768) + this.mtp.extentBytes(0, 768));
            captureMtp(host, node, gdn, attention, seed);
            tree.publish(node);

            commitRows(attention2, 768);
            restoreMtp(host, List.of(node), gdn2, attention2);
            assertEquals(767, attention2.forLayer(CONFIG.numHiddenLayers()).length());
            long seed2 = restoredSeed(attention2);
            AttentionKvState a = attention.forLayer(CONFIG.numHiddenLayers());
            AttentionKvState b = attention2.forLayer(CONFIG.numHiddenLayers());
            int pageBytes = (int) (2 * a.planePageBytes());
            for (int page = 0; page < 3; page++)
                assertArrayEquals(
                        this.gpu.bytes(a.pageAddresses().get(page), pageBytes),
                        this.gpu.bytes(b.pageAddresses().get(page), pageBytes),
                        "MTP page " + page);
            assertArrayEquals(this.gpu.bytes(seed, 256), this.gpu.bytes(seed2, 256));
        }
    }

    @Test
    void aChainRestoresTheOverlappingMtpPageFromTheLaterNode() {
        var tree = new PrefixTree(new HostExtents(16L << 20));
        int[] tokens = IntStream.range(0, 1280).toArray();
        try (Arena arena = Arena.ofConfined();
                var gdn = gdn();
                var attention = attentionWithMtp();
                var gdn2 = gdn();
                var attention2 = attentionWithMtp()) {
            MemorySegment host = arena.allocate(16L << 20);
            commitRows(attention, 1280);
            commitMtpRows(attention, 1279);
            AttentionKvState mtp = attention.forLayer(CONFIG.numHiddenLayers());
            int pageBytes = (int) (2 * mtp.planePageBytes());
            long seed = this.gpu.allocate(256);
            fillState(gdn, attention, 5);
            fillMtpPages(attention, 100);
            this.gpu.fill(seed, 256, 1);
            PrefixNode first = tree.reserve(
                    tree.root(),
                    tokens,
                    512,
                    MtpCheckpoint.KIND,
                    this.layout.extentBytes(0, 512) + this.mtp.extentBytes(0, 512));
            captureMtp(host, first, gdn, attention, seed);
            tree.publish(first);
            byte[] firstPage0 = this.gpu.bytes(mtp.pageAddresses().get(0), pageBytes);
            // The sequence moves on: the MTP page that holds row 511 changes (row 511 is re-paired).
            fillMtpPages(attention, 200);
            this.gpu.fill(seed, 256, 2);
            PrefixNode second = tree.reserve(
                    first,
                    tokens,
                    1280,
                    MtpCheckpoint.KIND,
                    this.layout.extentBytes(512, 1280) + this.mtp.extentBytes(512, 1280));
            captureMtp(host, second, gdn, attention, seed);
            tree.publish(second);

            commitRows(attention2, 1280);
            restoreMtp(host, List.of(first, second), gdn2, attention2);
            long seed2 = restoredSeed(attention2);
            AttentionKvState restored = attention2.forLayer(CONFIG.numHiddenLayers());
            assertArrayEquals(
                    firstPage0,
                    this.gpu.bytes(restored.pageAddresses().get(0), pageBytes),
                    "MTP page 0 comes from the first node");
            for (int page = 1; page < 5; page++)
                assertArrayEquals(
                        this.gpu.bytes(mtp.pageAddresses().get(page), pageBytes),
                        this.gpu.bytes(restored.pageAddresses().get(page), pageBytes),
                        "MTP page " + page + " comes from the second node (page 1 overlaps the first)");
            assertArrayEquals(this.gpu.bytes(seed, 256), this.gpu.bytes(seed2, 256), "the last node's seed row");
        }
    }
}
