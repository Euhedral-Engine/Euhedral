package io.euhedral_execution.inference.core.runtime.graph;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(30)
class SequencerTest {

    private static final class Item implements Sequencer.Entry {
        final int index;
        volatile boolean ready;

        Item(int index) {
            this.index = index;
        }

        @Override
        public boolean ready() {
            return this.ready;
        }
    }

    private static final class Recording extends Sequencer<Item> {
        final List<Integer> drained = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger draining = new AtomicInteger();
        final AtomicInteger overlaps = new AtomicInteger();
        Consumer<Item> onDrained = item -> {};

        @Override
        protected void drained(Item item) {
            if (this.draining.incrementAndGet() != 1) this.overlaps.incrementAndGet();
            try {
                this.drained.add(item.index);
                this.onDrained.accept(item);
            } finally {
                this.draining.decrementAndGet();
            }
        }

        void complete(Item item) {
            item.ready = true;
            drain();
        }
    }

    @Test
    void aReadyEntryWaitsForEveryEntryOfferedBeforeIt() {
        var sequencer = new Recording();
        var first = new Item(0);
        var second = new Item(1);
        var third = new Item(2);
        sequencer.offer(first);
        sequencer.offer(second);
        sequencer.offer(third);

        sequencer.complete(third);
        sequencer.complete(second);
        assertEquals(List.of(), sequencer.drained, "the head is not ready");
        sequencer.complete(first);
        assertEquals(List.of(0, 1, 2), sequencer.drained);
        assertTrue(sequencer.isEmpty());
    }

    @Test
    void entriesCompletingOnManyThreadsDrainOnceEachInOfferOrderOnOneThreadAtATime() throws Exception {
        for (int round = 0; round < 20; round++) {
            var sequencer = new Recording();
            int count = 2_000;
            List<Item> items = new ArrayList<>();
            for (int index = 0; index < count; index++) {
                var item = new Item(index);
                items.add(item);
                sequencer.offer(item);
            }
            var shuffled = new ArrayList<>(items);
            Collections.shuffle(shuffled);
            var work = new ConcurrentLinkedQueue<>(shuffled);
            var start = new CountDownLatch(1);
            try (var executor = Executors.newFixedThreadPool(8)) {
                for (int thread = 0; thread < 8; thread++) {
                    executor.submit(() -> {
                        start.await();
                        for (Item item; (item = work.poll()) != null; ) sequencer.complete(item);
                        return null;
                    });
                }
                start.countDown();
                executor.shutdown();
                assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
            }
            List<Integer> expected = new ArrayList<>();
            for (int index = 0; index < count; index++) expected.add(index);
            assertEquals(expected, sequencer.drained);
            assertEquals(0, sequencer.overlaps.get(), "two threads drained at once");
        }
    }

    @Test
    void anEntryOfferedAndCompletedWhileDrainingIsDrainedByTheSameLoop() {
        var sequencer = new Recording();
        var first = new Item(0);
        var second = new Item(1);
        sequencer.offer(first);
        sequencer.onDrained = item -> {
            if (item == first) {
                sequencer.offer(second);
                sequencer.complete(second);
                assertEquals(List.of(0), sequencer.drained, "a drain never recurses into itself");
            }
        };
        sequencer.complete(first);
        assertEquals(List.of(0, 1), sequencer.drained);
    }

    @Test
    void aFailingEntryDoesNotStopTheDrain() {
        var sequencer = new Recording();
        var first = new Item(0);
        var second = new Item(1);
        sequencer.offer(first);
        sequencer.offer(second);
        sequencer.onDrained = item -> {
            if (item == first) throw new IllegalStateException("injected");
        };
        second.ready = true;
        sequencer.complete(first);
        assertEquals(List.of(0, 1), sequencer.drained);

        var third = new Item(2);
        sequencer.offer(third);
        sequencer.complete(third);
        assertEquals(List.of(0, 1, 2), sequencer.drained, "the drain is not left claimed");
    }
}
