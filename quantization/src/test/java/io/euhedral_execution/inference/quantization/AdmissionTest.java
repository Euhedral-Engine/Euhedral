package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/// How many candidates start for given free bytes, cache contents and candidate sizes, and which wait.
class AdmissionTest {

    private static final long MIB = 1L << 20;
    private static final Path FILE = Path.of("f.edrl");

    private static WeightCache.Key key(String name) {
        return new WeightCache.Key(FILE, name, 0);
    }

    private static Admission.Need need(long other, Object... objects) {
        Map<WeightCache.Key, Long> map = new LinkedHashMap<>();
        for (int i = 0; i < objects.length; i += 2) map.put(key((String) objects[i]), (Long) objects[i + 1] * MIB);
        return new Admission.Need(map, other * MIB);
    }

    @Test
    void aCandidateAddsItsOtherBytesAndTheObjectsNotYetOnTheDevice() {
        Admission.Need need = need(300, "a", 100L, "b", 200L, "c", 400L);
        assertEquals(1000 * MIB, need.additional(k -> false));
        assertEquals(
                300 * MIB + 200 * MIB,
                need.additional(k -> Set.of(key("a"), key("c")).contains(k)));
        assertEquals(300 * MIB, need.additional(k -> true));
        assertEquals(700 * MIB, need.weightBytes());
    }

    @Test
    void availableIsTheLowerOfWhatTheDeviceReportsAndWhatIsLeftOfTheOpeningFreeBytesLessTheMargin() {
        // The device reports 6000 free, but the cache holds 5000 and a running candidate has 1500 of other bytes
        // not yet allocated: of the 10000 free at open, 3500 are left.
        assertEquals(
                3500 * MIB - 512 * MIB,
                Admission.available(6000 * MIB, 10000 * MIB, 5000 * MIB, 1500 * MIB, 512 * MIB));
        // The device reports less than the books.
        assertEquals(
                2000 * MIB - 512 * MIB,
                Admission.available(2000 * MIB, 10000 * MIB, 5000 * MIB, 1500 * MIB, 512 * MIB));
    }

    @Test
    void candidatesStartInOrderWhileTheyFitAndTheRestWait() {
        long freeAtOpen = 10000 * MIB, margin = 512 * MIB;
        // Three candidates share a, b (3000 MiB of weights) and differ in one 1000 MiB object each; each has 800 MiB
        // of other bytes. The device's report is taken as the books say (nothing allocated outside the cache).
        List<Admission.Need> waiting = List.of(
                need(800, "a", 1500L, "b", 1500L, "x", 1000L),
                need(800, "a", 1500L, "b", 1500L, "y", 1000L),
                need(800, "a", 1500L, "b", 1500L, "z", 1000L),
                need(800, "a", 1500L, "b", 1500L, "w", 1000L));
        java.util.Set<WeightCache.Key> cache = new java.util.HashSet<>();
        long cacheBytes = 0, otherBytes = 0;
        int running = 0;
        int started = 0;
        for (Admission.Need need : waiting) {
            long additional = need.additional(cache::contains);
            long available = Admission.available(
                    freeAtOpen - cacheBytes - otherBytes, freeAtOpen, cacheBytes, otherBytes, margin);
            if (!Admission.admits(running, 0, additional, available)) break;
            for (WeightCache.Key object : need.objects().keySet())
                if (cache.add(object)) cacheBytes += need.objects().get(object);
            otherBytes += need.otherBytes();
            running++;
            started++;
        }
        // First: 4000 (3000 + 1000) + 800. Second adds 1000 + 800 (a, b shared). Third adds 1800. Fourth 1800 does
        // not fit: 10000 - 512 - (3000 + 3000) - 2400 = 88 left.
        assertEquals(3, started);
        assertEquals(3, running);
    }

    @Test
    void aCandidateTooBigForTheDeviceStillRunsAloneAndTheLimitHolds() {
        assertTrue(Admission.admits(0, 0, Long.MAX_VALUE, -1));
        assertFalse(Admission.admits(1, 0, 10, 9));
        assertTrue(Admission.admits(1, 0, 9, 9));
        // At the limit nothing more starts, however much room there is.
        assertFalse(Admission.admits(2, 2, 1, 1000));
        assertTrue(Admission.admits(1, 2, 1, 1000));
    }
}
