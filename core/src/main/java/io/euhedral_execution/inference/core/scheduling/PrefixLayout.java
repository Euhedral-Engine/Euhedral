package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/// Where each piece of a sequence's state sits in a checkpoint's extent, and the copies that move it.
///
/// A node's extent holds, in this order: for every GDN layer its convolution then recurrent buffer; then the
/// KV pages of its span, page by page, each page for every full-attention layer in layer order; then, for a node
/// with MTP state, the base hidden row of its last position and the MTP cache's pages. A span's pages run from
/// `startPosition / 256` to `ceil(position / 256)`; a span starts on a page boundary because node positions
/// are multiples of the page. A page is the K plane then the V plane.
///
/// The MTP cache of a checkpoint at `position` holds rows `[0, position - 1)`: row `position - 1` pairs the
/// base hidden of that position with the token at `position`, which belongs to whichever prompt resumes here,
/// so a restore recomputes it from the stored hidden row. Its pages run from the page of row `startPosition - 1`
/// (which a restore overwrites from the later node, as that row is the parent's last) to
/// `ceil((position - 1) / 256)`.
final class PrefixLayout {
    static final int PAGE = AttentionKvState.PAGE_TOKENS;

    /// One copy between a checkpoint's extent (at `hostOffset` in the arena) and a device buffer.
    record Copy(long hostOffset, long deviceAddress, long bytes) {}

    private final int[] gdnLayers;
    private final int[] kvLayers;
    private final int mtpLayer;
    private final long convolutionBytes;
    private final long recurrentBytes;
    private final long pageBytes;
    private final long hiddenRowBytes;

    private PrefixLayout(
            int[] gdnLayers,
            int[] kvLayers,
            int mtpLayer,
            long convolutionBytes,
            long recurrentBytes,
            long pageBytes,
            long hiddenRowBytes) {
        this.gdnLayers = gdnLayers;
        this.kvLayers = kvLayers;
        this.mtpLayer = mtpLayer;
        this.convolutionBytes = convolutionBytes;
        this.recurrentBytes = recurrentBytes;
        this.pageBytes = pageBytes;
        this.hiddenRowBytes = hiddenRowBytes;
    }

    static PrefixLayout of(QwenConfig config) {
        return of(config, false);
    }

    /// With `mtp`, the sequence's attention states carry the MTP layer's cache after the base layers'.
    static PrefixLayout of(QwenConfig config, boolean mtp) {
        QwenLayerType[] types = config.layerTypes();
        int[] gdn = IntStream.range(0, types.length)
                .filter(i -> types[i] == QwenLayerType.GATED_DELTA_NET)
                .toArray();
        int[] attention = IntStream.range(0, types.length)
                .filter(i -> types[i] == QwenLayerType.FULL_ATTENTION)
                .toArray();
        long channels = 2L * config.linearNumKeyHeads() * config.linearKeyHeadDim()
                + (long) config.linearNumValueHeads() * config.linearValueHeadDim();
        long convolution = channels * (config.linearConvKernelDim() - 1L) * Short.BYTES;
        long recurrent = (long) config.linearNumValueHeads()
                * config.linearKeyHeadDim()
                * config.linearValueHeadDim()
                * Float.BYTES;
        long width = (long) config.numKeyValueHeads() * config.attentionHeadDim();
        long page = 2L * PAGE * AttentionKvState.HEAD_ROW_BYTES * (width / 256);
        return new PrefixLayout(
                gdn,
                attention,
                mtp ? types.length : -1,
                convolution,
                recurrent,
                page,
                (long) config.hiddenSize() * Short.BYTES);
    }

    int[] kvLayers() {
        return this.kvLayers.clone();
    }

    /// The index of the MTP cache in the attention states, or -1 when the layout has none.
    int mtpLayer() {
        return this.mtpLayer;
    }

    /// Bytes of the BF16 hidden row a node with MTP state keeps.
    long hiddenRowBytes() {
        return this.hiddenRowBytes;
    }

    private long gdnBytes() {
        return (long) this.gdnLayers.length * (this.convolutionBytes + this.recurrentBytes);
    }

    private static int pages(int position) {
        return (position + PAGE - 1) / PAGE;
    }

    private static int mtpFirstPage(int startPosition) {
        return startPosition == 0 ? 0 : startPosition / PAGE - 1;
    }

    private static int mtpEndPage(int position) {
        return pages(position - 1);
    }

    private long kvBytes(int startPosition, int position) {
        return (long) (pages(position) - startPosition / PAGE) * this.kvLayers.length * this.pageBytes;
    }

    /// Bytes of a node's extent for the span `[startPosition, position)`; `mtp` adds the MTP state.
    long extentBytes(int startPosition, int position, boolean mtp) {
        long bytes = gdnBytes() + kvBytes(startPosition, position);
        if (mtp) {
            int mtpPages = Math.max(0, mtpEndPage(position) - mtpFirstPage(startPosition));
            bytes += this.hiddenRowBytes + (long) mtpPages * this.pageBytes;
        }
        return bytes;
    }

    List<Copy> captureCopies(PrefixNode node, GdnSequenceStates gdn, AttentionSequenceStates attention) {
        return captureCopies(node, gdn, attention, 0);
    }

    /// The copies that save `node`'s state from the sequence's buffers into its extent. A node with MTP state
    /// also saves the hidden row at `seedRowAddress`, the base hidden of its last position.
    List<Copy> captureCopies(
            PrefixNode node, GdnSequenceStates gdn, AttentionSequenceStates attention, long seedRowAddress) {
        List<Copy> copies = new ArrayList<>();
        long at = node.extentOffset();
        for (int layer : this.gdnLayers) {
            QwenGdnSequenceState state = gdn.forLayer(layer);
            copies.add(new Copy(at, state.convolutionStateAddress(), this.convolutionBytes));
            at += this.convolutionBytes;
            copies.add(new Copy(at, state.recurrentStateAddress(), this.recurrentBytes));
            at += this.recurrentBytes;
        }
        addPages(copies, at, node, attention);
        if (node.hasMtp()) {
            long mtpAt = node.extentOffset() + gdnBytes() + kvBytes(node.startPosition(), node.position());
            copies.add(new Copy(mtpAt, seedRowAddress, this.hiddenRowBytes));
            addMtpPages(copies, mtpAt + this.hiddenRowBytes, node, attention);
        }
        return copies;
    }

    List<Copy> restoreCopies(List<PrefixNode> chain, GdnSequenceStates gdn, AttentionSequenceStates attention) {
        return restoreCopies(chain, gdn, attention, 0);
    }

    /// The copies that load a matched chain into a sequence: the KV pages of every node, and the GDN state
    /// of the last (the state at the matched position). With a non-zero `seedRowAddress` the chain holds MTP
    /// state too: the MTP pages of every node, and the last node's hidden row into that address.
    List<Copy> restoreCopies(
            List<PrefixNode> chain, GdnSequenceStates gdn, AttentionSequenceStates attention, long seedRowAddress) {
        List<Copy> copies = new ArrayList<>();
        PrefixNode leaf = chain.getLast();
        long at = leaf.extentOffset();
        for (int layer : this.gdnLayers) {
            QwenGdnSequenceState state = gdn.forLayer(layer);
            copies.add(new Copy(at, state.convolutionStateAddress(), this.convolutionBytes));
            at += this.convolutionBytes;
            copies.add(new Copy(at, state.recurrentStateAddress(), this.recurrentBytes));
            at += this.recurrentBytes;
        }
        for (PrefixNode node : chain) addPages(copies, node.extentOffset() + gdnBytes(), node, attention);
        if (seedRowAddress != 0) {
            for (PrefixNode node : chain) {
                long mtpAt = node.extentOffset() + gdnBytes() + kvBytes(node.startPosition(), node.position());
                addMtpPages(copies, mtpAt + this.hiddenRowBytes, node, attention);
            }
            long leafAt = leaf.extentOffset() + gdnBytes() + kvBytes(leaf.startPosition(), leaf.position());
            copies.add(new Copy(leafAt, seedRowAddress, this.hiddenRowBytes));
        }
        return copies;
    }

    private void addPages(List<Copy> copies, long at, PrefixNode node, AttentionSequenceStates attention) {
        int first = node.startPosition() / PAGE;
        int end = pages(node.position());
        List<List<Long>> layerPages = new ArrayList<>();
        for (int layer : this.kvLayers) layerPages.add(attention.forLayer(layer).pageAddresses());
        for (int page = first; page < end; page++) {
            for (List<Long> pages : layerPages) {
                copies.add(new Copy(at, pages.get(page), this.pageBytes));
                at += this.pageBytes;
            }
        }
    }

    private void addMtpPages(List<Copy> copies, long at, PrefixNode node, AttentionSequenceStates attention) {
        List<Long> pages = attention.forLayer(this.mtpLayer).pageAddresses();
        for (int page = mtpFirstPage(node.startPosition()); page < mtpEndPage(node.position()); page++) {
            copies.add(new Copy(at, pages.get(page), this.pageBytes));
            at += this.pageBytes;
        }
    }
}
