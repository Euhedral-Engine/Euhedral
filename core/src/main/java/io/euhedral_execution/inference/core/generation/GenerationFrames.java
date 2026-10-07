package io.euhedral_execution.inference.core.generation;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.impl.FrameFactory;
import io.euhedral_execution.core.impl.FrameManager;
import io.euhedral_execution.hashing.HasherApi;
import io.euhedral_execution.inference.core.runtime.graph.FrameLake;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/// A session's recycled generation frames: one [FrameManager] per frame type, and the lake they are thrown into.
/// Each manager keeps its single checkout owner: a frame type is checked out only by the frame that throws it
/// (an Admit by the generation's start or a Select, a Select by an Admit, a Finish by whichever ends the chain),
/// and those hand-offs order the checkouts. The frames start unordered, so the factories draw a new routing seed
/// on create and on every reuse.
public final class GenerationFrames {

    private static final int RECYCLED = 4;

    private final FrameLake lake;
    private final long password = HasherApi.mix(ThreadLocalRandom.current().nextLong());
    private final FrameManager<Generation, Admit> admits = new FrameManager<>(RECYCLED, this.password);
    private final FrameManager<Generation, Select> selects = new FrameManager<>(RECYCLED, this.password);
    private final FrameManager<Generation, Finish> finishes = new FrameManager<>(RECYCLED, this.password);
    private int created;

    public GenerationFrames(FrameLake lake) {
        this.lake = Objects.requireNonNull(lake, "lake");
        this.admits.setFactory(new FrameFactory<>(
                (id, generation) -> counted(new Admit(id, this.admits, this, generation)),
                (generation, frame) -> frame.generation = generation));
        this.selects.setFactory(new FrameFactory<>(
                (id, generation) -> counted(new Select(id, this.selects, this, generation)),
                (generation, frame) -> frame.generation = generation));
        this.finishes.setFactory(new FrameFactory<>(
                (id, generation) -> counted(new Finish(id, this.finishes, generation)),
                (generation, frame) -> frame.generation = generation));
    }

    public FrameLake lake() {
        return this.lake;
    }

    /// Frames built so far (tests: recycling keeps this at one per type).
    public int createdFrames() {
        return this.created;
    }

    Admit admit(Generation generation, StepPort port) {
        Admit frame = this.admits.getOrCreate(generation, this.password);
        frame.port = port;
        return frame;
    }

    Select select(Generation generation, StepPort port) {
        Select frame = this.selects.getOrCreate(generation, this.password);
        frame.port = port;
        return frame;
    }

    Finish finish(Generation generation) {
        return this.finishes.getOrCreate(generation, this.password);
    }

    /// Throws `frame` into the lake; a lake that refuses it runs it here, so a generation always reaches its Finish.
    void publish(AbstractFrame frame) {
        this.lake.publishOrRun(frame);
    }

    private <F extends AbstractFrame> F counted(F frame) {
        this.created++;
        return frame;
    }
}
