package io.euhedral_execution.inference.core.scheduling.graph;

import io.euhedral_execution.hashing.HasherApi;
import java.util.concurrent.atomic.AtomicLong;

/// Routing hashes for Inference frames. Every frame shares one id hash and mixes it with its own seed:
/// consecutive seeds from a base drawn per frame group (xxHash64 mixing makes neighbouring seeds route
/// independently), as Euhedral's `NoOpFrame.generate` does. Frames that may run in parallel therefore route to
/// different workers.
public final class FrameSeeds {

    /// The id hash every Inference frame is built with.
    public static final long ID_HASH = HasherApi.mix(HasherApi.BASE_SEED);

    private static final AtomicLong GENERATION = new AtomicLong(1);
    /// Host jobs (tokenization, host tasks) draw from their own sequence, so the stage graphs' routing, and
    /// with it their lane placement, does not depend on how much host work ran before a graph was built.
    private static final AtomicLong HOST_GENERATION = new AtomicLong(1);

    private long next;

    /// Seeds for one stage graph.
    public FrameSeeds() {
        this.next = HasherApi.mix(HasherApi.BASE_SEED + GENERATION.getAndIncrement());
    }

    private FrameSeeds(long base) {
        this.next = base;
    }

    /// Seeds for one host job, from a sequence disjoint from the graphs'.
    public static FrameSeeds forHostWork() {
        return new FrameSeeds(HasherApi.mix(HasherApi.BASE_SEED - HOST_GENERATION.getAndIncrement()));
    }

    /// The seed of the group's next frame. Not thread-safe: one builder hands out a group's seeds.
    public long next() {
        return this.next++;
    }
}
