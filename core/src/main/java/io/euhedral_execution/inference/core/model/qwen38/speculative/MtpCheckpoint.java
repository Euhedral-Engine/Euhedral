package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixLayout;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import io.euhedral_execution.inference.core.state.AttentionKvState;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

/// The MTP layer's state in a prefix checkpoint: the base hidden row of the node's last position, then the MTP
/// cache's pages.
///
/// The MTP cache of a checkpoint at `position` holds rows `[0, position - 1)`: row `position - 1` pairs the
/// base hidden of that position with the token at `position`, which belongs to whichever prompt resumes here,
/// so a restore recomputes it from the stored hidden row. Its pages run from the page of row `startPosition - 1`
/// (which a restore overwrites from the later node, as that row is the parent's last) to
/// `ceil((position - 1) / 256)`. A restore leaves the hidden row in the sequence's draft seed buffer.
public final class MtpCheckpoint implements SpeculativeCheckpoint {
    public static final String KIND = "mtp";

    private final int mtpLayer;
    private final int hidden;
    private final long pageBytes;
    private final long hiddenRowBytes;
    /// The device address of the base hidden row of the position the next capture stores; set by the decoder.
    private volatile long seedRowAddress;

    public MtpCheckpoint(Qwen38Config config) {
        this.mtpLayer = config.numHiddenLayers();
        this.hidden = config.hiddenSize();
        this.pageBytes = PrefixLayout.pageBytes(config);
        this.hiddenRowBytes = (long) config.hiddenSize() * Short.BYTES;
    }

    /// The hidden row the next capture saves: the base post-final-norm hidden of the checkpoint's last position.
    public void seedRow(long address) {
        this.seedRowAddress = address;
    }

    @Override
    public String kind() {
        return KIND;
    }

    private static int firstPage(int startPosition) {
        return startPosition == 0 ? 0 : startPosition / PrefixLayout.PAGE - 1;
    }

    private static int endPage(int position) {
        return PrefixLayout.pages(position - 1);
    }

    @Override
    public long extentBytes(int startPosition, int position) {
        int pages = Math.max(0, endPage(position) - firstPage(startPosition));
        return this.hiddenRowBytes + (long) pages * this.pageBytes;
    }

    @Override
    public boolean holds(Sequence sequence, int position) {
        return this.seedRowAddress != 0
                && sequence.kvCacheState() instanceof AttentionStates attention
                && attention.layerCount() > this.mtpLayer
                && attention.forLayer(this.mtpLayer).length() >= position - 1;
    }

    @Override
    public List<PrefixLayout.Copy> captureCopies(PrefixNode node, long at, Sequence sequence) {
        var attention = (AttentionStates) sequence.kvCacheState();
        List<PrefixLayout.Copy> copies = new ArrayList<>();
        copies.add(new PrefixLayout.Copy(at, this.seedRowAddress, this.hiddenRowBytes));
        addPages(copies, at + this.hiddenRowBytes, node, attention);
        return copies;
    }

    @Override
    public List<PrefixLayout.Copy> restoreCopies(
            List<PrefixNode> chain, ToLongFunction<PrefixNode> at, Sequence sequence) {
        var attention = (AttentionStates) sequence.kvCacheState();
        PrefixNode leaf = chain.getLast();
        attention.forLayer(this.mtpLayer).prepareAppend(0, leaf.position() - 1);
        long seedRow = attention.draftSeedRows(1, this.hidden);
        List<PrefixLayout.Copy> copies = new ArrayList<>();
        // In chain order, so a later node overwrites the page it shares with its parent.
        for (PrefixNode node : chain) addPages(copies, at.applyAsLong(node) + this.hiddenRowBytes, node, attention);
        copies.add(new PrefixLayout.Copy(at.applyAsLong(leaf), seedRow, this.hiddenRowBytes));
        return copies;
    }

    @Override
    public void restored(Sequence sequence, int position) {
        AttentionKvState state = ((AttentionStates) sequence.kvCacheState()).forLayer(this.mtpLayer);
        state.appendSubmitted(position - 1);
        state.commitSubmitted();
    }

    private void addPages(List<PrefixLayout.Copy> copies, long at, PrefixNode node, AttentionStates attention) {
        List<Long> pages = attention.forLayer(this.mtpLayer).pageAddresses();
        for (int page = firstPage(node.startPosition()); page < endPage(node.position()); page++) {
            copies.add(new PrefixLayout.Copy(at, pages.get(page), this.pageBytes));
            at += this.pageBytes;
        }
    }
}
