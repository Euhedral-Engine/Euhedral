package io.euhedral_execution.inference.core.prefix;

import java.util.ArrayList;
import java.util.List;

/// One stored checkpoint: the sequence state at `position`, whose parent holds the state at `startPosition`.
/// The node owns the extent that holds its bytes; what the bytes mean is the layout's business. Its mutable
/// fields belong to the owning [PrefixTree] and are guarded by its monitor.
public final class PrefixNode {
    private final PrefixNode parent;
    private final int startPosition;
    private final int position;
    private final int[] tokens;
    /// The kind of speculative state the node also holds, or null.
    private final String speculation;
    long extentOffset = -1;
    long extentBytes;
    final List<PrefixNode> children = new ArrayList<>();
    int pins;
    long lastUse;

    PrefixNode(PrefixNode parent, int startPosition, int position, int[] tokens, String speculation) {
        this.parent = parent;
        this.startPosition = startPosition;
        this.position = position;
        this.tokens = tokens;
        this.speculation = speculation;
    }

    public PrefixNode parent() {
        return this.parent;
    }

    public int startPosition() {
        return this.startPosition;
    }

    public int position() {
        return this.position;
    }

    /// The tokens at positions `[startPosition, position)`.
    public int[] tokens() {
        return this.tokens;
    }

    /// The kind of speculative decoding state the node holds beside the base state ([#speculation]'s owner
    /// defines it), or null. It is usable only with an unbroken chain of ancestors of the same kind.
    public String speculation() {
        return this.speculation;
    }

    public long extentOffset() {
        return this.extentOffset;
    }

    public long extentBytes() {
        return this.extentBytes;
    }
}
