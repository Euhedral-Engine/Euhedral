package io.euhedral_execution.inference.core.guidance;

import io.euhedral_execution.inference.core.tokenizer.TokenConstraint;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/// One generation's position in an llguidance grammar.
///
/// The token mask for the next step is computed once, when the step first asks, and every query reads it; a
/// commit invalidates it. Like every constraint it is used by one generation step after another, never
/// concurrently; [#close] frees the native matcher and must follow the generation's end.
public final class GrammarConstraint implements TokenConstraint {
    private final Llguidance library;
    private final int vocabularySize;
    private final int[] endIds;
    private final Arena arena = Arena.ofShared();
    private final MemorySegment mask;
    private MemorySegment matcher;
    private boolean maskReady;

    GrammarConstraint(Llguidance library, MemorySegment matcher, int vocabularySize, int[] endIds) {
        this.library = library;
        this.endIds = endIds;
        this.matcher = matcher;
        this.vocabularySize = vocabularySize;
        long bytes;
        try {
            bytes = (long) library.maskByteSize.invokeExact(matcher);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance mask size failed", failure);
        }
        this.mask = this.arena.allocate(bytes, 4);
    }

    @Override
    public boolean allows(int tokenId) {
        if (tokenId < 0 || tokenId >= this.vocabularySize) return false;
        ensureMask();
        return (this.mask.getAtIndex(ValueLayout.JAVA_INT, tokenId >>> 5) & (1 << (tokenId & 31))) != 0;
    }

    @Override
    public void accept(int tokenId) {
        if (!allows(tokenId)) throw new IllegalArgumentException("token " + tokenId + " violates the grammar");
        // A terminator ends the generation; the finished grammar has nothing to consume.
        for (int end : this.endIds) if (end == tokenId) return;
        int result;
        try {
            result = (int) this.library.consumeToken.invokeExact(open(), tokenId);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance token commit failed", failure);
        }
        this.maskReady = false;
        if (result != 0) throw new IllegalStateException("llguidance refused a committed token: " + error());
    }

    /// Overwrites the logits of every token the grammar does not allow with negative infinity.
    @Override
    public void maskDisallowed(float[] logits) {
        ensureMask();
        int words = (Math.min(logits.length, this.vocabularySize) + 31) >>> 5;
        for (int word = 0; word < words; word++) {
            int bits = this.mask.getAtIndex(ValueLayout.JAVA_INT, word);
            if (bits == -1) continue;
            int base = word << 5;
            for (int bit = 0; bit < 32 && base + bit < logits.length; bit++)
                if ((bits & (1 << bit)) == 0) logits[base + bit] = Float.NEGATIVE_INFINITY;
        }
        for (int id = this.vocabularySize; id < logits.length; id++) logits[id] = Float.NEGATIVE_INFINITY;
    }

    /// True when the text so far is a whole sentence of the grammar.
    public boolean complete() {
        try {
            return (boolean) this.library.isAccepting.invokeExact(open());
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance accept check failed", failure);
        }
    }

    @Override
    public synchronized void close() {
        if (this.matcher == null) return;
        try {
            this.library.freeMatcher.invokeExact(this.matcher);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance matcher free failed", failure);
        } finally {
            this.matcher = null;
            this.arena.close();
        }
    }

    private void ensureMask() {
        if (this.maskReady) return;
        int result;
        try {
            result = (int) this.library.computeMaskInto.invokeExact(open(), this.mask, this.mask.byteSize());
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance mask computation failed", failure);
        }
        if (result != 0) throw new IllegalStateException("llguidance could not compute a token mask: " + error());
        this.maskReady = true;
    }

    private MemorySegment open() {
        if (this.matcher == null) throw new IllegalStateException("the grammar constraint is closed");
        return this.matcher;
    }

    private String error() {
        try {
            String message = Llguidance.string((MemorySegment) this.library.matcherError.invokeExact(this.matcher));
            return message == null ? "unknown error" : message;
        } catch (Throwable failure) {
            return "unknown error";
        }
    }
}
