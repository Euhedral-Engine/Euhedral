package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;

/// Device work that read an expert slot, which the next refill of that slot must not overtake.
///
/// A lease closed with a fence hands the cache a marker for the kernels that read the slot, instead of the
/// host waiting for them. The cache keeps the fence with the slot and passes it to the transfer that refills
/// it, which orders its copy behind the fence on the device ([#awaitOn]); no host thread ever blocks on it. A
/// slot whose leases were closed with several fences is refilled behind all of them.
///
/// The cache owns a fence from the moment it is handed over: it calls [#release] exactly once, after a
/// refill that waited for it has completed, or when the cache closes with the fence still pending. A fence
/// that was handed over must not be used by the caller again.
public interface DeviceFence {

    /// Orders work submitted to `copyStream` from now on behind the device work this fence stands for. The
    /// host does not wait.
    void awaitOn(GpuStream copyStream);

    /// Whether waiting for this fence implies waiting for `other`, so the slot need keep only this one.
    /// Conservative by default.
    default boolean covers(DeviceFence other) {
        return this == other;
    }

    /// Releases what the fence holds (for a stream marker, the marker). Called once by the cache.
    default void release() {}
}
