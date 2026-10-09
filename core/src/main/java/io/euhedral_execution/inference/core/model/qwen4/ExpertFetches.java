package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import java.util.concurrent.atomic.AtomicInteger;

/// A decode block's requests for its experts, made together: the cache's owner answers them in one record, so the
/// experts the device holds are leased at once and the block's grouped launch can take them in one run, while those
/// it must load arrive one by one and are launched as they come. Each expert has a target that the owner tells, and
/// that the fetch stage of the expert (which waits for a missing one) and the group stage (which takes the leased
/// ones) read. One per graph's block, reused by every quantum of the graph.
final class ExpertFetches implements ExpertCacheOwner.Batch {

    /// Nothing heard of the expert yet.
    private static final int PENDING = 0;
    /// The owner handed over the lease: the group takes the expert, or its fetch stage ends at once.
    private static final int ARRIVED = 1;
    /// The expert's fetch stage waits for the owner.
    private static final int WAITING = 2;

    /// What a fetch stage that waits for its expert is: it is told when the expert arrived.
    interface Waiter {
        void arrived();
    }

    /// One expert's request and what became of it.
    final class Target implements ExpertCacheOwner.Fetch {
        private final int index;
        private final AtomicInteger state = new AtomicInteger();
        private volatile Waiter waiter;
        final ExpertCacheOwner.Request request = new ExpertCacheOwner.Request(this);

        Target(int index) {
            this.index = index;
        }

        @Override
        public boolean stopped() {
            return ExpertFetches.this.quantum.stopRequested();
        }

        @Override
        public void abandoned() {
            settle();
        }

        @Override
        public void arrived(ExpertLease lease) {
            if (stopped()) lease.close();
            else ExpertFetches.this.moe.hold(this.index, lease);
            settle();
        }

        @Override
        public void failed(Throwable failure) {
            ExpertFetches.this.quantum.fail(failure);
            settle();
        }

        /// The owner is done with the expert: tells the fetch stage that waits, or leaves word for the one to come.
        private void settle() {
            if (this.state.compareAndSet(PENDING, ARRIVED)) return;
            this.waiter.arrived();
        }

        /// Whether the owner handed the expert over.
        boolean arrived() {
            return this.state.get() == ARRIVED;
        }

        /// The fetch stage of the expert asks to be told: false when the owner already did.
        boolean await(Waiter stage) {
            this.waiter = stage;
            return this.state.compareAndSet(PENDING, WAITING);
        }

        @Override
        public boolean scan() {
            return ExpertFetches.this.rows > 1;
        }
    }

    private final Target[] targets;
    private final boolean[] grouped;
    private final MoeLayer moe;
    private Quantum quantum;
    private int rows;
    private int count;
    private Runnable asked;

    ExpertFetches(MoeLayer moe, int capacity) {
        this.moe = moe;
        this.targets = new Target[capacity];
        this.grouped = new boolean[capacity];
        for (int index = 0; index < capacity; index++) this.targets[index] = new Target(index);
    }

    /// Prepares the requests of the block's `count` experts of bank `bank` for `quantum`; `asked` runs once the
    /// owner has answered them all (a missing expert is still on its way).
    void begin(Quantum quantum, int rows, int bank, int count, Runnable asked) {
        this.quantum = quantum;
        this.rows = rows;
        this.count = count;
        this.asked = asked;
        for (int index = 0; index < count; index++) {
            Target target = this.targets[index];
            target.state.set(PENDING);
            target.waiter = null;
            this.grouped[index] = false;
            target.request.set(bank, this.moe.activeExpert(index));
        }
    }

    Target target(int index) {
        return this.targets[index];
    }

    /// Whether the group stage launched the expert.
    boolean grouped(int index) {
        return this.grouped[index];
    }

    void grouped(int index, boolean value) {
        this.grouped[index] = value;
    }

    @Override
    public int size() {
        return this.count;
    }

    @Override
    public ExpertCacheOwner.Request request(int index) {
        return this.targets[index].request;
    }

    @Override
    public void asked() {
        this.asked.run();
    }
}
