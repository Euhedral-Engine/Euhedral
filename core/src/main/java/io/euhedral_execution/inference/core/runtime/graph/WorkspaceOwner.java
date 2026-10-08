package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.hashing.HasherApi;
import java.util.Arrays;

/// The workspace's owner: an owner-confined record of each buffer's last accessors. Every admission runs on
/// frames ordered on [#HASH] (their `idHash`, which they keep as their routing hash), one at a time, so each graph's
/// first accessors of a buffer follow the last ones of the graph bound before it: concurrent quanta interleave
/// buffer by buffer. One at a time is all the routing gives: frames published from different threads cross the
/// lake's queues of several partitions in any order, so graphs bind in the order their admissions ran, which need not be the
/// order they were published. Nothing depends on that order: quanta that share sequence state are never in flight
/// together, and a decision that reads the records ([#last]) checks who wrote them.
public final class WorkspaceOwner {

    /// The workspace owner's `idHash`. Frames built with it and left ordered run one at a time, on the owner.
    public static final long HASH = HasherApi.mix(0x0b_0ff3_45L);

    /// The graph that last used a buffer: its shape, its quantum, and the frames of its exits of the buffer.
    public record Last(GraphShape shape, StageQuantum quantum, StageFrame[] exits) {}

    private final Last[] last;
    private ExternalEdge[] edges = new ExternalEdge[64];

    public WorkspaceOwner(int buffers) {
        this.last = new Last[buffers];
    }

    /// The graph that last used `buffer`, or null. Owner frames only.
    public Last last(int buffer) {
        return this.last[buffer];
    }

    /// Starts `graph` for `quantum` behind the last accessors of every buffer `shape` touches, then records the
    /// graph's own exits. Owner frames only.
    public void bind(StageGraph graph, GraphShape shape, WorkspaceUse use, StageQuantum quantum) {
        int count = 0;
        for (int b = 0; b < this.last.length; b++) {
            Last previous = this.last[b];
            if (previous == null) continue;
            for (int entry : use.entries(b)) {
                StageFrame target = graph.stage(entry);
                for (StageFrame exit : previous.exits()) {
                    // The graph's own previous binding: it was recycled only after its device work retired.
                    if (exit.graph() == graph) continue;
                    if (count == this.edges.length) this.edges = Arrays.copyOf(this.edges, count * 2);
                    this.edges[count++] = new ExternalEdge(exit, target);
                }
            }
        }
        try {
            graph.start(quantum, this.edges, count);
        } finally {
            Arrays.fill(this.edges, 0, count, null);
        }
        for (int b = 0; b < this.last.length; b++) {
            int[] exits = use.exits(b);
            if (exits.length == 0) continue;
            StageFrame[] frames = new StageFrame[exits.length];
            for (int i = 0; i < exits.length; i++) frames[i] = graph.stage(exits[i]);
            this.last[b] = new Last(shape, quantum, frames);
        }
    }
}
