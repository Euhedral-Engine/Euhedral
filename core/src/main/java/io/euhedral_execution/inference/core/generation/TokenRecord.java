package io.euhedral_execution.inference.core.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/// The tokens a session generated across its prompts: appended by one generation frame at a time, read by any
/// thread. The array only grows, with its earlier entries copied before it is published, so a reader that takes the
/// count and then the array sees that many tokens.
public final class TokenRecord {
    private volatile int[] ids = new int[64];
    private volatile int count;

    public void add(int id) {
        int count = this.count;
        int[] ids = this.ids;
        if (count == ids.length) {
            ids = Arrays.copyOf(ids, count * 2);
            this.ids = ids;
        }
        ids[count] = id;
        this.count = count + 1;
    }

    public List<Integer> snapshot() {
        int count = this.count;
        int[] ids = this.ids;
        List<Integer> copy = new ArrayList<>(count);
        for (int i = 0; i < count; i++) copy.add(ids[i]);
        return List.copyOf(copy);
    }
}
