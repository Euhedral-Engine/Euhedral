package io.euhedral_execution.inference.core.prefix;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class PrefixTreeTest {
    private static final long BLOCK = 4096;

    private static int[] tokens(int from, int count) {
        return IntStream.range(from, from + count).toArray();
    }

    private static PrefixNode add(PrefixTree tree, PrefixNode parent, int[] tokens, int position, long bytes) {
        PrefixNode node = tree.reserve(parent, tokens, position, false, bytes);
        assertNotNull(node);
        tree.publish(node);
        return node;
    }

    @Test
    void anEmptyTreeMisses() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        assertNull(tree.lookup(tokens(0, 100), false));
    }

    @Test
    void matchesTheDeepestChain() {
        var tree = new PrefixTree(new HostExtents(64 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        PrefixNode b = add(tree, a, stored, 4096, BLOCK);
        var match = tree.lookup(tokens(0, 5000), false);
        assertSame(b, match.leaf());
        assertEquals(4096, match.position());
        assertEquals(2, match.chain().size());
    }

    @Test
    void stopsAtTheFirstSpanThatDiffers() {
        var tree = new PrefixTree(new HostExtents(64 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        add(tree, a, stored, 4096, BLOCK);
        int[] prompt = tokens(0, 5000);
        prompt[3000] = -1;
        assertSame(a, tree.lookup(prompt, false).leaf());
    }

    @Test
    void aPromptThatSharesOnlyTheFirstTokenMisses() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        add(tree, tree.root(), tokens(0, 3000), 2048, BLOCK);
        int[] prompt = tokens(0, 3000);
        prompt[1] = -1;
        assertNull(tree.lookup(prompt, false));
    }

    @Test
    void aPromptNeverMatchesAtOrBeyondItsOwnLength() {
        var tree = new PrefixTree(new HostExtents(64 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        add(tree, a, stored, 4096, BLOCK);
        assertSame(a, tree.lookup(tokens(0, 4096), false).leaf(), "a prompt that ends on a node uses the one before");
        assertNull(tree.lookup(tokens(0, 2048), false), "and a prompt that ends on the first node has none");
        assertSame(a, tree.lookup(tokens(0, 2049), false).leaf());
    }

    @Test
    void prefersTheChildThatReachesFurther() {
        var tree = new PrefixTree(new HostExtents(64 * BLOCK));
        int[] stored = tokens(0, 6000);
        add(tree, tree.root(), stored, 2048, BLOCK);
        PrefixNode far = add(tree, tree.root(), stored, 3072, BLOCK);
        assertSame(far, tree.lookup(tokens(0, 5000), false).leaf());
        assertEquals(1, tree.lookup(tokens(0, 5000), false).chain().size());
    }

    @Test
    void mtpNeedsAnUnbrokenMtpChain() {
        var tree = new PrefixTree(new HostExtents(64 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = tree.reserve(tree.root(), stored, 2048, true, BLOCK);
        tree.publish(a);
        PrefixNode b = tree.reserve(a, stored, 4096, false, BLOCK);
        tree.publish(b);
        PrefixNode c = tree.reserve(b, stored, 5120, true, BLOCK);
        assertEquals(false, c.hasMtp(), "an MTP node under a node without MTP state is unusable");
        tree.abort(c);
        assertSame(a, tree.lookup(tokens(0, 5000), true).leaf());
        assertSame(b, tree.lookup(tokens(0, 5000), false).leaf());
    }

    @Test
    void findLocatesAPublishedChildBySpan() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        int[] stored = tokens(0, 3000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        assertSame(a, tree.find(tree.root(), stored, 2048, false));
        assertNull(tree.find(tree.root(), stored, 1024, false));
        int[] other = tokens(0, 3000);
        other[10] = -1;
        assertNull(tree.find(tree.root(), other, 2048, false));
    }

    @Test
    void aNodeWithAndWithoutMtpStateCanCoexistForTheSameSpan() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        int[] stored = tokens(0, 3000);
        PrefixNode plain = add(tree, tree.root(), stored, 2048, BLOCK);
        PrefixNode withMtp = tree.reserve(tree.root(), stored, 2048, true, BLOCK);
        tree.publish(withMtp);
        assertSame(plain, tree.find(tree.root(), stored, 2048, false));
        assertSame(withMtp, tree.find(tree.root(), stored, 2048, true));
        assertSame(withMtp, tree.lookup(tokens(0, 3000), true).leaf());
    }

    @Test
    void anUnpublishedNodeIsInvisibleUntilPublished() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        PrefixNode node = tree.reserve(tree.root(), tokens(0, 3000), 2048, false, BLOCK);
        assertNull(tree.lookup(tokens(0, 3000), false));
        tree.publish(node);
        assertSame(node, tree.lookup(tokens(0, 3000), false).leaf());
    }

    @Test
    void abortFreesTheExtent() {
        var extents = new HostExtents(2 * BLOCK);
        var tree = new PrefixTree(extents);
        PrefixNode node = tree.reserve(tree.root(), tokens(0, 3000), 2048, false, 2 * BLOCK);
        assertEquals(0, extents.freeBytes());
        tree.abort(node);
        assertEquals(2 * BLOCK, extents.freeBytes());
        assertEquals(0, tree.size());
        assertNull(tree.lookup(tokens(0, 3000), false));
    }

    @Test
    void evictsTheLeastRecentlyUsedLeafForRoom() {
        var tree = new PrefixTree(new HostExtents(2 * BLOCK));
        PrefixNode first = add(tree, tree.root(), tokens(0, 3000), 2048, BLOCK);
        add(tree, tree.root(), tokens(10_000, 3000), 2048, BLOCK);
        tree.release(tree.lookup(tokens(0, 3000), false));
        add(tree, tree.root(), tokens(20_000, 3000), 2048, BLOCK);
        assertNull(tree.lookup(tokens(10_000, 3000), false), "the older entry was evicted");
        assertSame(first, tree.lookup(tokens(0, 3000), false).leaf());
        assertEquals(1, tree.evictions());
    }

    @Test
    void neverEvictsANodeWithChildren() {
        var tree = new PrefixTree(new HostExtents(3 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        add(tree, a, stored, 4096, BLOCK);
        add(tree, tree.root(), tokens(10_000, 3000), 2048, BLOCK);
        PrefixNode more = tree.reserve(tree.root(), tokens(20_000, 3000), 2048, false, BLOCK);
        assertNotNull(more, "the childless node made room");
        assertNull(tree.find(a, stored, 4096, false), "the child went, not the parent it hung from");
        assertSame(a, tree.find(tree.root(), stored, 2048, false));
    }

    @Test
    void neverEvictsAPinnedMatch() {
        var tree = new PrefixTree(new HostExtents(BLOCK));
        int[] stored = tokens(0, 3000);
        add(tree, tree.root(), stored, 2048, BLOCK);
        var match = tree.lookup(tokens(0, 3000), false);
        assertNull(tree.reserve(tree.root(), tokens(10_000, 3000), 2048, false, BLOCK), "nothing evictable");
        assertSame(match.leaf(), tree.find(tree.root(), stored, 2048, false));
        tree.release(match);
        assertNotNull(tree.reserve(tree.root(), tokens(10_000, 3000), 2048, false, BLOCK));
    }

    @Test
    void theParentOfAReservationIsNotEvictedForIt() {
        var tree = new PrefixTree(new HostExtents(2 * BLOCK));
        int[] stored = tokens(0, 6000);
        PrefixNode a = add(tree, tree.root(), stored, 2048, BLOCK);
        PrefixNode b = tree.reserve(a, stored, 4096, false, BLOCK);
        assertNotNull(b);
        assertNull(tree.reserve(a, stored, 4352, false, BLOCK), "the only evictable node is the pinned parent");
        assertSame(a, tree.find(tree.root(), stored, 2048, false));
        tree.abort(b);
        assertNotNull(tree.reserve(a, stored, 4352, false, BLOCK), "its pin was released with the abort");
    }

    @Test
    void refusesAReservationLargerThanTheArena() {
        var tree = new PrefixTree(new HostExtents(BLOCK));
        assertNull(tree.reserve(tree.root(), tokens(0, 3000), 2048, false, 2 * BLOCK));
    }

    @Test
    void rejectsSpansThatDoNotAdvanceOverrunTheTokensOrLeaveThePageGrid() {
        var tree = new PrefixTree(new HostExtents(16 * BLOCK));
        int[] stored = tokens(0, 3000);
        assertThrows(IllegalArgumentException.class, () -> tree.reserve(tree.root(), stored, 0, false, BLOCK));
        assertThrows(IllegalArgumentException.class, () -> tree.reserve(tree.root(), stored, 3072, false, BLOCK));
        assertThrows(IllegalArgumentException.class, () -> tree.reserve(tree.root(), stored, 2000, false, BLOCK));
    }
}
