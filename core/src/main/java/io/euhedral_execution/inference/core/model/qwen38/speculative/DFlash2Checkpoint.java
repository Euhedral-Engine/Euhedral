package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.inference.core.model.qwen38.AttentionStates;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import io.euhedral_execution.inference.core.model.qwen38.loader.DFlash2Config;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixLayout;
import io.euhedral_execution.inference.core.prefix.PrefixNode;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToLongFunction;

/// The DFlash2 drafter's state in a prefix checkpoint: each draft layer's context ring (keys, then values), which
/// holds the keys and values of the last `window` positions before the checkpoint. That is all the drafter keeps
/// of a sequence, so a checkpoint stores it whole and a restore needs nothing recomputed: the ring of a sequence
/// restored at `p` is the ring the cold run had at `p`. Each node holds its own ring; only the leaf's is restored.
final class DFlash2Checkpoint implements SpeculativeCheckpoint {
    static final String KIND = "dflash2";

    private final DFlash2Config config;
    private final long ringBytes;

    DFlash2Checkpoint(DFlash2Config config) {
        this.config = config;
        this.ringBytes = (long) config.slidingWindow() * config.keyValueWidth() * Short.BYTES;
    }

    @Override
    public String kind() {
        return KIND;
    }

    @Override
    public long extentBytes(int startPosition, int position) {
        return 2L * this.config.layers() * this.ringBytes;
    }

    @Override
    public boolean holds(Sequence sequence, int position) {
        return sequence.kvCacheState() instanceof AttentionStates attention
                && attention.dflash2() != null
                && attention.dflash2().contextLength() == position;
    }

    @Override
    public List<PrefixLayout.Copy> captureCopies(PrefixNode node, long at, Sequence sequence) {
        return copies(at, ((AttentionStates) sequence.kvCacheState()).dflash2());
    }

    @Override
    public List<PrefixLayout.Copy> restoreCopies(
            List<PrefixNode> chain, ToLongFunction<PrefixNode> at, Sequence sequence) {
        DFlash2State state = ((AttentionStates) sequence.kvCacheState()).dflash2(this.config);
        return copies(at.applyAsLong(chain.getLast()), state);
    }

    @Override
    public void restored(Sequence sequence, int position) {
        ((AttentionStates) sequence.kvCacheState()).dflash2().commitContext(position);
    }

    private List<PrefixLayout.Copy> copies(long at, DFlash2State state) {
        List<PrefixLayout.Copy> copies = new ArrayList<>();
        for (int layer = 0; layer < this.config.layers(); layer++) {
            copies.add(new PrefixLayout.Copy(at, state.ringKeys(layer), this.ringBytes));
            at += this.ringBytes;
            copies.add(new PrefixLayout.Copy(at, state.ringValues(layer), this.ringBytes));
            at += this.ringBytes;
        }
        return copies;
    }
}
