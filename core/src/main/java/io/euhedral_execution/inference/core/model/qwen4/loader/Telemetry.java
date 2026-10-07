package io.euhedral_execution.inference.core.model.qwen4.loader;

/// What a loaded model reports about its storage, for the engine that executes it: where the bytes are, and what the
/// expert cache and the n-gram store have done since the load.
public record Telemetry(
        long fixedDeviceBytes,
        long fixedHostBackedBytes,
        long hostMappedBytes,
        long contextReserveBytes,
        long expertCacheBytes,
        int expertSlots,
        long expertCacheHits,
        long expertCacheMisses,
        long expertEvictions,
        long expertTransferBytes,
        long expertTransferNanos,
        long expertWaitNanos,
        long ngramTransferBytes,
        long ngramRowsGathered,
        long ngramHostBytes) {

    /// Host-to-device rate of the expert transfers over their own durations, in bytes per second; 0 before any.
    public double expertTransferBytesPerSecond() {
        return this.expertTransferNanos == 0 ? 0 : this.expertTransferBytes * 1e9 / this.expertTransferNanos;
    }

    public double expertHitRate() {
        long requests = this.expertCacheHits + this.expertCacheMisses;
        return requests == 0 ? 0 : (double) this.expertCacheHits / requests;
    }
}
