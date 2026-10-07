package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import org.junit.jupiter.api.Test;

class EmissionTest {

    @Test
    void handsTextToTheCallerInOrderThenTheTokens() throws Exception {
        var emission = new Emission();
        Thread.ofPlatform().start(() -> {
            emission.text("a");
            emission.text("");
            emission.text("b");
            emission.end(List.of(1, 2), null);
        });
        var seen = new ArrayList<String>();
        assertEquals(List.of(1, 2), emission.drain(seen::add, () -> {}));
        assertEquals(List.of("a", "b"), seen);
    }

    @Test
    void aThrowingOutputCancelsOnceAndRethrowsAfterTheEnd() {
        var emission = new Emission();
        var cancels = new java.util.concurrent.atomic.AtomicInteger();
        Thread.ofPlatform().start(() -> {
            emission.text("a");
            emission.text("b");
            emission.end(List.of(), null);
        });
        var thrown = assertThrows(
                IllegalStateException.class,
                () -> emission.drain(
                        text -> {
                            throw new IllegalStateException(text);
                        },
                        cancels::incrementAndGet));
        assertEquals("a", thrown.getMessage());
        assertEquals(1, cancels.get());
    }

    @Test
    void anInterruptedCallerCancelsWaitsForTheEndAndThrows() throws Exception {
        var emission = new Emission();
        var cancelled = new CountDownLatch(1);
        Thread caller = Thread.currentThread();
        Thread.ofPlatform().start(() -> {
            caller.interrupt();
            try {
                cancelled.await();
            } catch (InterruptedException ignored) {
                return;
            }
            emission.end(List.of(), null);
        });
        assertThrows(InterruptedException.class, () -> emission.drain(text -> {}, cancelled::countDown));
        assertTrue(Thread.interrupted(), "the interrupt is restored");
    }
}
