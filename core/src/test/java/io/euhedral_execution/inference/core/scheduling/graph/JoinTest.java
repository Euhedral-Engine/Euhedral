package io.euhedral_execution.inference.core.scheduling.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.core.frames.AbstractFrame;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/// The fan-in of frames spawned at run time: the last arrival publishes the continuation, once.
class JoinTest {

    @Test
    void theLastArrivalPublishesTheContinuationOnceAndCarriesTheFirstFailure() {
        List<AbstractFrame> published = new ArrayList<>();
        FrameLake lake = new FrameLake() {
            @Override
            public void publish(AbstractFrame frame) {
                published.add(frame);
            }

            @Override
            public void publishFromCallback(AbstractFrame frame) {
                published.add(frame);
            }

            @Override
            public void admit() {}

            @Override
            public void admitDuringDrain() {}

            @Override
            public void terminated() {}
        };
        AbstractFrame continuation = new AbstractFrame(FrameSeeds.ID_HASH) {};
        Join join = new Join(lake, continuation);
        join.expect(3);
        join.arrive();
        join.fail(new IllegalStateException("first"));
        assertTrue(published.isEmpty(), "the join waits for every arrival");
        join.fail(new IllegalStateException("second"));
        assertEquals(List.of(continuation), published);
        assertEquals("first", join.failure().getMessage());
        join.expect(1);
        assertNull(join.failure(), "the next round starts clean");
        join.arrive();
        assertEquals(2, published.size());
    }
}
