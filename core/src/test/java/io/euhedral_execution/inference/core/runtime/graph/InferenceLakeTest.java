package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.core.generics.LatticeReceiver;
import io.euhedral_execution.core.generics.LatticeSource;
import io.euhedral_execution.core.generics.LatticeTerminal;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/// The lake is a set of ingest sinks that are each an upstream source of the lattice: frames go
/// into the sink their routing hash selects, workers pull from every sink, and the lake completes
/// once every admitted unit terminated.
class InferenceLakeTest {

    private static final class Frame extends AbstractFrame {
        Frame(long routing) {
            super(1);
            randomizeHash(routing);
        }
    }

    /// A lattice that only records its upstreams and gives each a downstream, so tests can pull
    /// from them.
    private static final class Recording implements LatticeTerminal {
        final List<LatticeSource> sources = new ArrayList<>();
        final AtomicInteger completions = new AtomicInteger();

        @Override
        public void addUpstream(LatticeSource upstream) {
            this.sources.add(upstream);
            upstream.addDownstream(new LatticeReceiver() {
                @Override
                public void addUpstream(LatticeSource source) {}

                @Override
                public void push(AbstractFrame frame) {}

                @Override
                public void onComplete() {
                    Recording.this.completions.incrementAndGet();
                }

                @Override
                public void onError(Throwable error) {
                    throw new AssertionError(error);
                }
            });
        }

        long pull(LatticeSource source, List<AbstractFrame> into) {
            return source.pull(into::add, frame -> false, Long.MAX_VALUE);
        }
    }

    @Test
    void everySinkIsAnUpstreamSourceOfTheLattice() {
        var lattice = new Recording();
        var lake = new InferenceLake(lattice, 4, 2);
        assertEquals(0, lattice.sources.size(), "nothing attaches before the first unit");
        assertFalse(lake.isAttached());
        lake.admit();
        assertEquals(4, lattice.sources.size());
        assertEquals(4, lake.sinks());
        assertTrue(lake.isAttached());
        lake.admit();
        assertEquals(4, lattice.sources.size(), "attached once");
    }

    @Test
    void framesOfOneRoutingHashShareASinkAndOthersSpread() {
        var lattice = new Recording();
        var lake = new InferenceLake(lattice, 4, 1);
        for (int i = 0; i < 10; i++) lake.publish(new Frame(7));
        int sinksHoldingThem = 0;
        for (LatticeSource source : lattice.sources) {
            List<AbstractFrame> pulled = new ArrayList<>();
            lattice.pull(source, pulled);
            if (!pulled.isEmpty()) {
                sinksHoldingThem++;
                assertEquals(10, pulled.size(), "a lane keeps to one sink");
            }
        }
        assertEquals(1, sinksHoldingThem);

        for (int i = 0; i < 200; i++) lake.publish(new Frame(1_000 + i));
        int used = 0;
        for (LatticeSource source : lattice.sources) {
            List<AbstractFrame> pulled = new ArrayList<>();
            lattice.pull(source, pulled);
            if (!pulled.isEmpty()) used++;
        }
        assertTrue(used > 1, "frames that may run in parallel spread over the sinks");
    }

    @Test
    void aFramePublishedFromACallbackIsEnqueuedLikeAnyOther() {
        var lattice = new Recording();
        var lake = new InferenceLake(lattice, 2, 1);
        lake.publishFromCallback(new Frame(3));
        List<AbstractFrame> pulled = new ArrayList<>();
        for (LatticeSource source : lattice.sources) lattice.pull(source, pulled);
        assertEquals(1, pulled.size());
    }

    @Test
    void theLakeCompletesOnlyAfterEveryAdmittedUnitTerminated() {
        var lattice = new Recording();
        var lake = new InferenceLake(lattice, 3, 1);
        lake.admit();
        lake.admit();
        assertEquals(2, lake.active());
        lake.completeGracefully();
        assertFalse(lake.isComplete(), "a unit is still running");
        assertThrows(IllegalStateException.class, lake::admit, "admission closed");
        lake.admitDuringDrain();
        lake.terminated();
        lake.terminated();
        assertFalse(lake.isComplete(), "the continuation is still running");
        lake.terminated();
        assertTrue(lake.isComplete());
        lake.awaitTermination();
        assertEquals(3, lattice.completions.get(), "every sink was detached");
        assertFalse(lake.isAttached());
        assertThrows(IllegalStateException.class, () -> lake.publish(new Frame(1)));
    }

    @Test
    void aLakeWithNothingAdmittedCompletesAtOnce() {
        var lattice = new Recording();
        var lake = new InferenceLake(lattice, 2, 1);
        lake.completeGracefully();
        assertTrue(lake.isComplete());
        assertEquals(0, lattice.sources.size(), "a lake that was never used never attaches");
    }

    @Test
    void terminatingMoreThanAdmittedIsRejected() {
        var lake = new InferenceLake(new Recording(), 1, 1);
        assertThrows(IllegalStateException.class, lake::terminated);
    }

    /// A frame of `owner`, stamped with its place in publication order, ordered on its owner.
    private static final class Owned extends AbstractFrame {
        final long owner;
        final int stamp;

        Owned(long owner, int stamp) {
            super(owner);
            this.owner = owner;
            this.stamp = stamp;
        }
    }

    private static List<Owned> drainOwned(Recording lattice) {
        List<AbstractFrame> pulled = new ArrayList<>();
        for (LatticeSource source : lattice.sources) lattice.pull(source, pulled);
        List<Owned> owned = new ArrayList<>();
        for (AbstractFrame frame : pulled) owned.add((Owned) frame);
        return owned;
    }

    @Test
    void framesOfOneOwnerLeaveTheLakeInPublicationOrder() {
        var lattice = new Recording();
        InferenceLake lake = io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime.newLake(lattice);
        for (int i = 0; i < 2000; i++) lake.publish(new Owned(0x5eed_0001L, i));
        List<Owned> owned = drainOwned(lattice);
        assertEquals(2000, owned.size());
        for (int i = 0; i < owned.size(); i++) assertEquals(i, owned.get(i).stamp, "publication order at " + i);
    }

    @Test
    void chainsOfTwoOwnersKeepTheirOwnOrder() {
        var lattice = new Recording();
        InferenceLake lake = io.euhedral_execution.inference.core.runtime.EuhedralInferenceRuntime.newLake(lattice);
        for (int i = 0; i < 2000; i++) lake.publish(new Owned(i % 2 == 0 ? 0xa11cL : 0xb0bL, i));
        int lastA = -1, lastB = -1;
        for (Owned frame : drainOwned(lattice)) {
            if (frame.owner == 0xa11cL) {
                assertTrue(frame.stamp > lastA, "owner A out of order");
                lastA = frame.stamp;
            } else {
                assertTrue(frame.stamp > lastB, "owner B out of order");
                lastB = frame.stamp;
            }
        }
    }
}
