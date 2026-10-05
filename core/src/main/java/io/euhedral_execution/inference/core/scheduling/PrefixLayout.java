package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.inference.core.model_loader.config.QwenConfig;
import io.euhedral_execution.inference.core.model_loader.config.QwenLayerType;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

/// Where each piece of a sequence's base state sits in a checkpoint's extent, and the copies that move it.
///
/// A node's extent holds, in this order: for every GDN layer its convolution then recurrent buffer; then the
/// KV pages of its span, page by page, each page for every full-attention layer in layer order; then, for a node
/// that holds speculative state, that state ([SpeculativeCheckpoint]) from [#speculativeOffset]. A span's pages
/// run from `startPosition / 256` to `ceil(position / 256)`; a span starts on a page boundary because node
/// positions are multiples of the page. A page is the K plane then the V plane.
final class PrefixLayout {
    static final int PAGE = AttentionKvState.PAGE_TOKENS;

    /// One copy between a checkpoint's extent (at `hostOffset` in the arena) and a device buffer.
    record Copy(long hostOffset, long deviceAddress, long bytes) {}

    private final int[] gdnLayers;
    private final int[] kvLayers;
    private final long convolutionBytes;
    private final long recurrentBytes;
    private final long pageBytes;

    private PrefixLayout(int[] gdnLayers, int[] kvLayers, long convolutionBytes, long recurrentBytes, long pageBytes) {
        this.gdnLayers = gdnLayers;
        this.kvLayers = kvLayers;
        this.convolutionBytes = convolutionBytes;
        this.recurrentBytes = recurrentBytes;
        this.pageBytes = pageBytes;
    }

    static PrefixLayout of(QwenConfig config) {
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
        return new PrefixLayout(gdn, attention, convolution, recurrent, pageBytes(config));
    }

    /// Bytes of one 256-token KV page (K and V planes) of one full-attention layer.
    static long pageBytes(QwenConfig config) {
        long width = (long) config.numKeyValueHeads() * config.attentionHeadDim();
        return 2L * PAGE * AttentionKvState.HEAD_ROW_BYTES * (width / 256);
    }

    int[] kvLayers() {
        return this.kvLayers.clone();
    }

    private long gdnBytes() {
        return (long) this.gdnLayers.length * (this.convolutionBytes + this.recurrentBytes);
    }

    static int pages(int position) {
        return (position + PAGE - 1) / PAGE;
    }

    private long kvBytes(int startPosition, int position) {
        return (long) (pages(position) - startPosition / PAGE) * this.kvLayers.length * this.pageBytes;
    }

    /// Bytes of a node's base state for the span `[startPosition, position)`.
    long extentBytes(int startPosition, int position) {
        return gdnBytes() + kvBytes(startPosition, position);
    }

    /// Where `node`'s speculative state starts in the arena: after its base state.
    long speculativeOffset(PrefixNode node) {
        return node.extentOffset() + extentBytes(node.startPosition(), node.position());
    }

    /// The copies that save `node`'s base state from the sequence's buffers into its extent.
    List<Copy> captureCopies(PrefixNode node, GdnSequenceStates gdn, AttentionSequenceStates attention) {
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
        return copies;
    }

    /// The copies that load a matched chain's base state into a sequence: the KV pages of every node, and the
    /// GDN state of the last (the state at the matched position).
    List<Copy> restoreCopies(List<PrefixNode> chain, GdnSequenceStates gdn, AttentionSequenceStates attention) {
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
}
