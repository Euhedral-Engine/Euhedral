package io.euhedral_execution.inference.core.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.RamTierShard;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.TierDirective;
import io.euhedral_execution.inference.core.scheduling.graph.AsyncReads;

/// A prefetch: the artifact read of a record straight into the host tier slot the owner planned for it, ahead of any
/// request, with no device copy. The owner's prefetch frame starts it; the read's completion is a frame on whichever
/// worker reaps it, which gives the disk's read back and publishes the owner's frame that makes the slot ready (or
/// returns it when the read failed). A request that finds the slot filling publishes itself again until it is
/// ready, then copies from it.
final class TierFill {
    private final ExpertCacheOwner owner;
    private final RamTierShard tier;
    private final TierDirective directive;
    private final int bank;
    private final int expert;
    private final Done done;

    TierFill(ExpertCacheOwner owner, RamTierShard tier, TierDirective directive, int bank, int expert, long seed) {
        this.owner = owner;
        this.tier = tier;
        this.directive = directive;
        this.bank = bank;
        this.expert = expert;
        this.done = new Done(seed);
    }

    /// Submits the read. Called by the owner's prefetch frame; returns false, having changed nothing the owner must
    /// undo but the slot, when the read cannot be made asynchronously.
    boolean start() {
        return this.owner.cache.store().readPartAsync(this.bank, this.expert, -1, this.directive, 0, 1, this.done);
    }

    /// The read's completion: gives the disk's read back and hands the slot to the owner.
    private final class Done extends AsyncReads.Read {
        Done(long seed) {
            super(seed);
        }

        @Override
        public void execute() {
            TierFill fill = TierFill.this;
            fill.owner.cache.store().completePart(fill.bank, fill.expert, -1, fill.directive, 0, 1, readNanos());
            fill.owner.readEnded();
            fill.owner.lake.publish(new Settle(failure() == null));
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            TierFill.this.owner.readEnded();
            TierFill.this.owner.lake.publish(new Settle(false));
        }
    }

    /// On the owner: the slot holds the record, or goes back.
    private final class Settle extends AbstractFrame {
        private final boolean read;

        Settle(boolean read) {
            super(ExpertCacheOwner.HASH);
            this.read = read;
        }

        @Override
        public void execute() {
            TierFill fill = TierFill.this;
            if (this.read) fill.tier.filled(fill.directive);
            else fill.tier.abandoned(fill.directive);
            fill.owner.prefetchEnded(this.read);
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }
}
