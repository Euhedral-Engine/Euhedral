package io.euhedral_execution.inference.core.prefix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/// A radix tree of checkpoints over token ids. A node covers the tokens between its parent's position and
/// its own; a lookup walks down the nodes whose tokens equal the prompt's, comparing the tokens themselves,
/// so a match is exact. Nodes become visible when published, and are evicted least recently used first,
/// childless nodes only, never while pinned. Positions lie on the KV page grid, so a node's pages start where
/// its parent's end. Not thread-safe: confined to its owner (the prefix cache's owner frames); [#size] and
/// [#evictions] may be read from anywhere.
public final class PrefixTree {
    /// Every node position is a multiple of this: the number of tokens in a KV page.
    public static final int POSITION_GRANULE = 256;

    private final HostExtents extents;
    private final PrefixNode root = new PrefixNode(null, 0, 0, new int[0], null);
    private final Set<PrefixNode> published = new LinkedHashSet<>();
    private long clock;
    /// Written by the owner only; read by statistics from any thread.
    private volatile long evictions;
    private volatile int size;

    public PrefixTree(HostExtents extents) {
        this.extents = extents;
    }

    public PrefixNode root() {
        return this.root;
    }

    /// The nodes a lookup matched, from the first under the root to the deepest. Pinned until released.
    public record Match(List<PrefixNode> chain) {
        public PrefixNode leaf() {
            return this.chain.getLast();
        }

        public int position() {
            return leaf().position();
        }
    }

    /// The longest match of `prompt` that leaves at least one token to prefill, or null. With a `speculation`
    /// kind only nodes that hold that kind of speculative state match, so the chain is unbroken; with null any
    /// node matches.
    public Match lookup(int[] prompt, String speculation) {
        List<PrefixNode> chain = new ArrayList<>();
        PrefixNode node = this.root;
        while (true) {
            PrefixNode best = null;
            for (PrefixNode child : node.children) {
                if (child.position() >= prompt.length) continue;
                if (speculation != null && !speculation.equals(child.speculation())) continue;
                if (best != null && child.position() <= best.position()) continue;
                if (matches(child, prompt)) best = child;
            }
            if (best == null) break;
            chain.add(best);
            node = best;
        }
        if (chain.isEmpty()) return null;
        for (PrefixNode matched : chain) {
            matched.pins++;
            matched.lastUse = ++this.clock;
        }
        return new Match(List.copyOf(chain));
    }

    public void release(Match match) {
        for (PrefixNode node : match.chain()) node.pins--;
    }

    /// The published child of `parent` whose span `[parent.position, position)` equals `tokens` and whose
    /// speculative state is of the kind `speculation` (null: none), or null. A span can be stored once per kind.
    public PrefixNode find(PrefixNode parent, int[] tokens, int position, String speculation) {
        for (PrefixNode child : parent.children)
            if (child.position() == position
                    && java.util.Objects.equals(child.speculation(), speculation)
                    && position <= tokens.length
                    && matches(child, tokens)) {
                child.lastUse = ++this.clock;
                return child;
            }
        return null;
    }

    /// The published child of `parent` whose span `[parent.position, position)` equals `tokens`, whatever
    /// speculative state it holds, or null: every node holds the base state a plain capture would store.
    public PrefixNode findAny(PrefixNode parent, int[] tokens, int position) {
        for (PrefixNode child : parent.children)
            if (child.position() == position && position <= tokens.length && matches(child, tokens)) {
                child.lastUse = ++this.clock;
                return child;
            }
        return null;
    }

    /// Reserves `bytes` for a node covering `[parent.position, position)` of `tokens`, evicting least recently
    /// used nodes for room. Returns null when nothing evictable remains. The node is invisible until
    /// [#publish], and `parent` stays pinned until [#publish] or [#abort].
    public PrefixNode reserve(PrefixNode parent, int[] tokens, int position, String speculation, long bytes) {
        if (position <= parent.position() || position > tokens.length || position % POSITION_GRANULE != 0)
            throw new IllegalArgumentException("a node must advance, on the page grid, within the tokens");
        parent.pins++;
        long offset = this.extents.allocate(bytes);
        while (offset < 0) {
            PrefixNode victim = victim();
            if (victim == null) {
                parent.pins--;
                return null;
            }
            evict(victim);
            offset = this.extents.allocate(bytes);
        }
        PrefixNode node = new PrefixNode(
                parent,
                parent.position(),
                position,
                Arrays.copyOfRange(tokens, parent.position(), position),
                parent == this.root || java.util.Objects.equals(parent.speculation(), speculation)
                        ? speculation
                        : null);
        node.extentOffset = offset;
        node.extentBytes = bytes;
        return node;
    }

    /// Makes a reserved node visible to lookups.
    public void publish(PrefixNode node) {
        node.parent().children.add(node);
        this.published.add(node);
        this.size = this.published.size();
        node.lastUse = ++this.clock;
        node.parent().pins--;
    }

    /// Gives a reserved node's bytes back; it never became visible.
    public void abort(PrefixNode node) {
        this.extents.free(node.extentOffset, node.extentBytes);
        node.parent().pins--;
    }

    public int size() {
        return this.size;
    }

    public long evictions() {
        return this.evictions;
    }

    private PrefixNode victim() {
        PrefixNode oldest = null;
        for (PrefixNode node : this.published)
            if (node.children.isEmpty() && node.pins == 0 && (oldest == null || node.lastUse < oldest.lastUse))
                oldest = node;
        return oldest;
    }

    private void evict(PrefixNode node) {
        node.parent().children.remove(node);
        this.published.remove(node);
        this.extents.free(node.extentOffset, node.extentBytes);
        this.evictions = this.evictions + 1;
        this.size = this.published.size();
    }

    private static boolean matches(PrefixNode node, int[] prompt) {
        return Arrays.equals(prompt, node.startPosition(), node.position(), node.tokens(), 0, node.tokens().length);
    }
}
