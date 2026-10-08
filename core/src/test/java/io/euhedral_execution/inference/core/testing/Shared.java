package io.euhedral_execution.inference.core.testing;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/// Immutable state that tests of one JVM share: a model on the device, an engine around it. Each value is created on
/// first use and closed when the test run ends ([SharedClose]), or sooner when a test of another scope needs the
/// device. A scope is a model group (see [ModelGroup]): two models do not fit the device together, so asking for
/// the first value of a new scope closes every value of the previous one, newest first.
///
/// Whatever a test takes from here, it leaves as it found it: allocations it makes are its own to free, and a toggled
/// setting is restored in a `finally`.
public final class Shared {
    /// The scope of values that outlive every model group, such as the device handle.
    public static final String GLOBAL = "";

    /// Creates a shared value.
    @FunctionalInterface
    public interface Factory<T> {
        T create() throws Exception;
    }

    private record Entry(String scope, AutoCloseable value) {}

    private static final Map<Object, Entry> OPEN = new LinkedHashMap<>();
    private static String current = GLOBAL;

    private Shared() {}

    /// The value kept under `key`, created by `factory` on first use.
    @SuppressWarnings("unchecked")
    public static synchronized <T extends AutoCloseable> T get(String scope, Object key, Factory<T> factory) {
        Entry held = OPEN.get(key);
        if (held != null) return (T) held.value();
        try {
            if (!scope.equals(GLOBAL) && !scope.equals(current)) {
                closeScoped();
                current = scope;
            }
            T created = factory.create();
            OPEN.put(key, new Entry(scope, created));
            return created;
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Exception failure) {
            throw new IllegalStateException("could not create the shared value " + key, failure);
        }
    }

    /// Closes the values of every model scope, newest first, and leaves the global ones open.
    private static void closeScoped() {
        List<Object> keys = new ArrayList<>(OPEN.keySet());
        Throwable failure = null;
        for (int index = keys.size() - 1; index >= 0; index--) {
            Entry entry = OPEN.get(keys.get(index));
            if (entry.scope().equals(GLOBAL)) continue;
            OPEN.remove(keys.get(index));
            try {
                entry.value().close();
            } catch (Exception | Error closing) {
                if (failure == null) failure = closing;
                else failure.addSuppressed(closing);
            }
        }
        if (failure instanceof Error error) throw error;
        if (failure != null) throw new IllegalStateException("could not close the shared values", failure);
    }

    /// Closes everything, newest first. The test run calls this once, when it ends.
    static synchronized void closeAll() {
        Throwable failure = null;
        List<Object> keys = new ArrayList<>(OPEN.keySet());
        for (int index = keys.size() - 1; index >= 0; index--) {
            Entry entry = OPEN.remove(keys.get(index));
            try {
                entry.value().close();
            } catch (Exception | Error closing) {
                if (failure == null) failure = closing;
                else failure.addSuppressed(closing);
            }
        }
        current = GLOBAL;
        if (failure instanceof Error error) throw error;
        if (failure != null) throw new IllegalStateException("could not close the shared values", failure);
    }
}
