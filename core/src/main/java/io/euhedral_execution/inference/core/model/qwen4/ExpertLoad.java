package io.euhedral_execution.inference.core.model.qwen4;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.gpu.GpuStream;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertCacheShard;
import io.euhedral_execution.inference.core.model.qwen4.expert.ExpertLease;
import io.euhedral_execution.inference.core.model.qwen4.expert.HostExpertStore;
import io.euhedral_execution.inference.core.model.qwen4.expert.HostRecord;
import io.euhedral_execution.inference.core.model.qwen4.expert.RamTierShard;
import io.euhedral_execution.inference.core.model.qwen4.expert.TierDirective;
import io.euhedral_execution.inference.core.runtime.graph.AsyncReads;
import io.euhedral_execution.inference.core.runtime.graph.FrameSeeds;
import io.euhedral_execution.inference.core.runtime.graph.Join;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// One miss of the device cache, from the fetch that reserved its slot to the retirement of its copy, as
/// frames and the dependencies between them:
///
/// ```
/// fetch (owner) ─┬─> read parts 0..n ─ join ─> submit ─┐
///                └─> copy out of RAM, submit ──────────┴─> (the fetch has its lease) ─ ─ (device) ─ ─> retire (owner)
/// ```
///
/// The fetch reserved the slot and made the lease (the slot's address and the marker the copy will record). A
/// record read from the artifact is read in page-aligned parts: each part's frame submits its read and ends, and
/// the read's completion is a frame of its own (a fill's part then copies its range into the staging buffer);
/// the parts join into the frame that submits the device copy; a record the host tier holds is one frame that copies it
/// out of RAM and submits. Whichever frame
/// submits the copy hands the lease to the fetch: from there the expert's kernels wait for the marker on the
/// device. The copy's retirement is a driver callback that publishes the retire frame, the owner's, which makes
/// the slot resident, settles the host tier and gives the staging buffer back. A load that fails publishes the
/// owner's failure frame instead. Nothing but the fetch, the retirement and a failure touches the cache's
/// bookkeeping; the reads, the copy and the submission carry hashes of their own and spread over the workers.
final class ExpertLoad {

    private static final Logger LOG = LoggerFactory.getLogger(ExpertLoad.class);

    final ExpertCacheOwner owner;
    final ExpertCacheOwner.Fetch target;
    final ExpertCacheShard.Load load;
    final ExpertLease lease;
    final RamTierShard tier;
    final TierDirective directive;
    /// Whether the load reads the artifact (one of the disk's reads in flight until its record is in).
    final boolean reads;
    /// Whether the load gave its read back.
    private final java.util.concurrent.atomic.AtomicBoolean readGiven = new java.util.concurrent.atomic.AtomicBoolean();
    private final long seed;
    private final Part[] parts;
    private final Join join;
    private final Copy copy;
    private final Submit submit;
    private final Retire retire;
    private final Fail fail;
    /// The staging buffer the owner reserved for the load.
    private final int buffer;

    // Written by the frames; each is read by a frame that follows the writer across a dependency.
    private int stream;
    private HostRecord record;
    private boolean read;
    private boolean touched;
    private Throwable failure;
    private long ticket;
    // The load's timeline: fetched, started (first frame), read (record ready), submitting, submitted, retired
    // (callback), confirmed (retire frame).
    final long fetchedAt;
    volatile long startedAt;
    private final long[] partEnd;
    long readAt;
    long submittingAt;
    long submittedAt;
    volatile long retiredAt;
    long confirmedAt;

    ExpertLoad(
            ExpertCacheOwner owner,
            ExpertCacheOwner.Fetch target,
            ExpertCacheShard.Load load,
            int buffer,
            RamTierShard tier,
            TierDirective directive,
            boolean reads,
            long seed) {
        this.owner = owner;
        this.target = target;
        this.load = load;
        this.buffer = buffer;
        this.lease = load.lease();
        this.tier = tier;
        this.seed = seed;
        this.fetchedAt = System.nanoTime();
        this.directive = directive;
        this.reads = reads;
        int count = this.directive.mode() != TierDirective.Mode.HIT
                        && owner.cache.store().rangedReads()
                ? Math.max(1, owner.readParts)
                : 0;
        this.parts = new Part[count];
        for (int part = 0; part < count; part++) this.parts[part] = new Part(part);
        this.partEnd = new long[count];
        this.submit = new Submit();
        this.retire = new Retire();
        this.fail = new Fail();
        this.join = count == 0 ? null : new Join(owner.lake, this.submit);
        this.copy = count == 0 ? new Copy() : null;
    }

    private HostExpertStore store() {
        return this.owner.cache.store();
    }

    /// Publishes the load's first frames. Called by the owner's fetch. A record in a pinned tier needs no frame:
    /// its copy is submitted here, from its slot.
    void start() {
        if (this.join == null && !store().stagesThrough(this.directive)) {
            this.startedAt = System.nanoTime();
            try {
                this.record = store().open(this.load.bank(), this.load.expert(), this.buffer, this.directive);
                readDone();
                this.read = true;
                this.readAt = System.nanoTime();
                this.submittingAt = this.readAt;
                submitCopy();
            } catch (Throwable thrown) {
                failed(thrown);
                return;
            }
            this.target.arrived(this.lease);
            return;
        }
        if (this.join != null) {
            this.join.expect(this.parts.length);
            for (Part part : this.parts) this.owner.lake.publish(part);
        } else this.owner.lake.publish(this.copy);
    }

    /// Gives the load's read back to the owner, once: when its record is in, or when the load ends without it.
    void readDone() {
        if (this.reads && this.readGiven.compareAndSet(false, true)) this.owner.readEnded();
    }

    /// The staging buffer; the first frame that writes the record marks the load started.
    private int staging() {
        if (this.startedAt == 0) this.startedAt = System.nanoTime();
        return this.buffer;
    }

    /// Submits the device copy of the ready record. Runs on the frame that has the record; touches none of the
    /// owner's state. The caller then hands the lease to the fetch.
    private void submitCopy() throws Throwable {
        this.touched = true;
        this.stream = this.owner.cache.transfer().nextStream();
        this.owner.cache.transfer().stream(
                this.stream,
                this.record,
                this.load.deviceAddress(),
                this.load.fence(),
                this.load.readyMarker(),
                this.retire);
        // The copy and its ready marker are queued: another block's claim of the expert may join the load now.
        this.load.copySubmitted();
        this.submittedAt = System.nanoTime();
    }

    /// The load cannot complete: the owner's failure frame ends it.
    private void failed(Throwable thrown) {
        this.failure = thrown;
        this.owner.lake.publish(this.fail);
    }

    /// One part of the record's artifact read: a page-aligned range read straight into where the record goes.
    /// The frame submits the read and ends; the read's completion is a frame of its own, emitted when the bytes
    /// arrived, which finishes the part and arrives at the join. Where reads cannot be asynchronous, the part is
    /// read here.
    private final class Part extends AbstractFrame {
        private final int index;
        private final PartRead read;

        Part(int index) {
            super(FrameSeeds.ID_HASH);
            this.index = index;
            randomizeHash(ExpertLoad.this.seed + 1 + index);
            this.read = new PartRead(this);
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            try {
                int buffer = load.staging();
                if (load.store()
                        .readPartAsync(
                                load.load.bank(),
                                load.load.expert(),
                                buffer,
                                load.directive,
                                this.index,
                                load.parts.length,
                                this.read)) return;
                load.store()
                        .readPart(
                                load.load.bank(),
                                load.load.expert(),
                                buffer,
                                load.directive,
                                this.index,
                                load.parts.length);
            } catch (Throwable thrown) {
                load.partEnd[this.index] = System.nanoTime();
                load.join.fail(thrown);
                return;
            }
            load.partEnd[this.index] = System.nanoTime();
            load.join.arrive();
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            ExpertLoad.this.join.fail(new IllegalStateException("the lattice rejected an expert read", rejection));
        }
    }

    /// A part's bytes arrived (or its read failed): finish the part and arrive at the join.
    private final class PartRead extends AsyncReads.Read {
        private final Part part;

        PartRead(Part part) {
            super(ExpertLoad.this.seed + 33 + part.index);
            this.part = part;
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            Throwable thrown = failure();
            if (thrown == null) {
                try {
                    load.store()
                            .completePart(
                                    load.load.bank(),
                                    load.load.expert(),
                                    load.buffer,
                                    load.directive,
                                    this.part.index,
                                    load.parts.length,
                                    readNanos());
                } catch (Throwable copy) {
                    thrown = copy;
                }
            }
            load.partEnd[this.part.index] = System.nanoTime();
            if (thrown != null) load.join.fail(thrown);
            else load.join.arrive();
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            ExpertLoad.this.join.fail(new IllegalStateException("the lattice rejected a read's completion", rejection));
        }
    }

    /// The parts are read: make the record addressable and submit its copy.
    private final class Submit extends AbstractFrame {
        Submit() {
            super(FrameSeeds.ID_HASH);
            randomizeHash(ExpertLoad.this.seed);
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            load.submittingAt = System.nanoTime();
            long last = 0;
            for (long end : load.partEnd) last = Math.max(last, end);
            load.readAt = last;
            load.readDone();
            try {
                Throwable thrown = load.join.failure();
                if (thrown != null) throw thrown;
                load.record =
                        load.store().completeOpen(load.load.bank(), load.load.expert(), load.buffer, load.directive);
                load.read = true;
                load.submitCopy();
            } catch (Throwable thrown) {
                load.failed(thrown);
                return;
            }
            load.target.arrived(load.lease);
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            ExpertLoad.this.failed(new IllegalStateException("the lattice rejected an expert submission", rejection));
        }
    }

    /// A record the host tier holds: one copy of it out of RAM into a staging buffer, then its device copy.
    private final class Copy extends AbstractFrame {
        Copy() {
            super(FrameSeeds.ID_HASH);
            randomizeHash(ExpertLoad.this.seed);
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            try {
                load.record = load.store().open(load.load.bank(), load.load.expert(), load.staging(), load.directive);
                load.readDone();
                load.read = true;
                load.readAt = System.nanoTime();
                load.submittingAt = load.readAt;
                load.submitCopy();
            } catch (Throwable thrown) {
                load.failed(thrown);
                return;
            }
            load.target.arrived(load.lease);
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            ExpertLoad.this.failed(new IllegalStateException("the lattice rejected an expert copy", rejection));
        }
    }

    /// Gives back what [RamTierShard#plan] reserved for a load that will not start. Owner frames only.
    static void giveBack(RamTierShard tier, TierDirective directive) {
        if (tier == null) return;
        switch (directive.mode()) {
            case HIT -> tier.used(directive);
            case FILL -> tier.abandoned(directive);
            case NONE, BYPASS -> {}
        }
        directive.clear();
    }

    /// The load is done with its tier slot: a fill whose record was read (`read`) keeps it for the next miss;
    /// one that was not gives the slot back. Owner frames only.
    private void settleTier(boolean read) {
        if (this.tier == null) return;
        switch (this.directive.mode()) {
            case HIT -> this.tier.used(this.directive);
            case FILL -> {
                if (read) this.tier.filled(this.directive);
                else this.tier.abandoned(this.directive);
            }
            case NONE, BYPASS -> {}
        }
        this.directive.clear();
    }

    /// The load failed before its copy was submitted: on the owner, the slot returns to the cache with the lease
    /// that was never handed on, and the fetch hears of it.
    private final class Fail extends AbstractFrame {
        Fail() {
            super(ExpertCacheOwner.HASH);
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            Throwable thrown = load.failure;
            load.settleTier(load.read);
            if (load.touched) {
                try {
                    load.owner.cache.transfer().recover(load.stream, thrown);
                } catch (RuntimeException | Error recovery) {
                    thrown.addSuppressed(recovery);
                }
            }
            load.load.failed(load.lease, thrown);
            if (load.buffer >= 0) load.store().releaseStaging(load.buffer);
            load.target.failed(thrown);
            load.owner.ended(load, false);
        }

        @Override
        public void doFinally() {}

        /// The lattice rejected the frame without running it; the load still ends once, and the rejecting
        /// thread is never a driver callback.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }

    /// The copy retired: on the owner, the slot holds the expert, the tier is settled and the staging buffer is
    /// free again.
    private final class Retire extends AbstractFrame implements GpuStream.RetirementListener {
        Retire() {
            super(ExpertCacheOwner.HASH);
        }

        /// A driver thread: it only publishes.
        @Override
        public void retired(long ticket, boolean driverThread) {
            ExpertLoad load = ExpertLoad.this;
            load.ticket = ticket;
            load.retiredAt = System.nanoTime();
            if (driverThread) load.owner.lake.publishFromCallback(this);
            else load.owner.lake.publish(this);
        }

        @Override
        public void execute() {
            ExpertLoad load = ExpertLoad.this;
            load.confirmedAt = System.nanoTime();
            Throwable device;
            try {
                device = load.owner.cache.transfer().confirm(load.stream, load.ticket);
            } catch (Throwable thrown) {
                device = thrown;
            }
            load.load.retired(device);
            // The fetch's quantum may be over: a copy that failed on the device poisons the device, and the
            // quantum that read it fails at its own retirement.
            if (device != null) LOG.error("an expert copy failed on the device", device);
            load.settleTier(true);
            if (load.buffer >= 0) load.store().releaseStaging(load.buffer);
            load.owner.ended(load, true);
        }

        @Override
        public void doFinally() {}

        /// The lattice rejected the frame without running it; the copy still retired once, and the rejecting
        /// thread is never a driver callback.
        @Override
        public void doFinallyWithError(Throwable rejection) {
            execute();
        }
    }
}
