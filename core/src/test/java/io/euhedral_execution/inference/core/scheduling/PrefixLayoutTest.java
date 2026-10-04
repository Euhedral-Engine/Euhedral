package io.euhedral_execution.inference.core.scheduling;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.prefix.HostExtents;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.prefix.PrefixTree;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PrefixLayoutTest {
    private static final QwenConfig CONFIG =
            QwenExecutionFixtures.statefulCompactWeights(8).config();

    private final MemoryGpu gpu = new MemoryGpu();
    private final PrefixLayout layout = PrefixLayout.of(CONFIG);

    private GdnSequenceStates gdn() {
        return GdnSequenceStates.allocate(
                this.gpu,
                CONFIG.layerTypes(),
                CONFIG.linearNumKeyHeads(),
                CONFIG.linearNumValueHeads(),
                CONFIG.linearKeyHeadDim(),
                CONFIG.linearValueHeadDim(),
                CONFIG.linearConvKernelDim());
    }

    private AttentionSequenceStates attention() {
        return AttentionSequenceStates.allocate(
                this.gpu, CONFIG.layerTypes(), CONFIG.numKeyValueHeads() * CONFIG.attentionHeadDim());
    }

    /// Reserves and commits `rows` rows of KV in the attention layer, as a prefill would.
    private static void commitRows(AttentionSequenceStates states, int rows) {
        AttentionKvState state = states.forLayer(1);
        state.prepareAppend(0, rows);
        state.appendSubmitted(rows);
        state.commitSubmitted();
    }

    private void fillState(GdnSequenceStates gdn, AttentionSequenceStates attention, int seed) {
        QwenGdnSequenceState g = gdn.forLayer(0);
        this.gpu.fill(g.convolutionStateAddress(), (int) g.convolutionBytes(), seed);
        this.gpu.fill(g.recurrentStateAddress(), (int) g.recurrentBytes(), seed + 1);
        List<Long> pages = attention.forLayer(1).pageAddresses();
        for (int i = 0; i < pages.size(); i++)
            this.gpu.fill(pages.get(i), (int) (2 * attention.forLayer(1).planePageBytes()), seed + 10 + i);
    }

    private void capture(
            MemorySegment host, PrefixNode node, GdnSequenceStates gdn, AttentionSequenceStates attention) {
        for (var copy : this.layout.captureCopies(node, gdn, attention))
            this.gpu.copyDeviceToHost(
                    host.asSlice(copy.hostOffset(), copy.bytes()), copy.deviceAddress(), copy.bytes());
    }

    private void restore(
            MemorySegment host, List<PrefixNode> chain, GdnSequenceStates gdn, AttentionSequenceStates attention) {
        for (var copy : this.layout.restoreCopies(chain, gdn, attention))
            this.gpu.copyHostToDevice(
                    copy.deviceAddress(), host.asSlice(copy.hostOffset(), copy.bytes()), copy.bytes());
    }

    @Test
    void sizesAMatchingGeometryToTheStateClasses() {
        try (var gdn = gdn();
                var attention = attention()) {
            QwenGdnSequenceState g = gdn.forLayer(0);
            long page = 2 * attention.forLayer(1).planePageBytes();
            assertEquals(g.convolutionBytes() + g.recurrentBytes() + 3 * page, this.layout.extentBytes(0, 768, false));
            assertEquals(
                    g.convolutionBytes() + g.recurrentBytes() + 2 * page,
                    this.layout.extentBytes(512, 1024, false),
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
            PrefixNode node = tree.reserve(tree.root(), tokens, 768, false, this.layout.extentBytes(0, 768, false));
            capture(host, node, gdn, attention);
            tree.publish(node);

            try (var gdn2 = gdn();
                    var attention2 = attention()) {
                commitRows(attention2, 768);
                restore(host, List.of(node), gdn2, attention2);
                QwenGdnSequenceState a = gdn.forLayer(0);
                QwenGdnSequenceState b = gdn2.forLayer(0);
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
            PrefixNode first = tree.reserve(tree.root(), tokens, 512, false, this.layout.extentBytes(0, 512, false));
            capture(host, first, gdn, attention);
            tree.publish(first);
            // Pages 0 and 1 (rows 0-511) are the first node's, as filled now.
            byte[][] firstPages = new byte[2][];
            for (int page = 0; page < 2; page++)
                firstPages[page] =
                        this.gpu.bytes(attention.forLayer(1).pageAddresses().get(page), pageBytes);
            // The sequence moves on: every buffer is rewritten, then the second node (pages 2-4) is captured.
            fillState(gdn, attention, 9);
            PrefixNode second = tree.reserve(first, tokens, 1280, false, this.layout.extentBytes(512, 1280, false));
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
                QwenGdnSequenceState a = gdn.forLayer(0);
                QwenGdnSequenceState b = gdn2.forLayer(0);
                assertArrayEquals(
                        this.gpu.bytes(a.recurrentStateAddress(), (int) a.recurrentBytes()),
                        this.gpu.bytes(b.recurrentStateAddress(), (int) b.recurrentBytes()),
                        "the leaf's GDN state, not the first node's");
            }
        }
    }
}
