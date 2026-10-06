package io.euhedral_execution.inference.core.model_loader.qwen4;

import io.euhedral_execution.inference.core.model_loader.qwen4.expert.ExpertCacheStats;
import io.euhedral_execution.inference.core.model_loader.qwen4.expert.RamTier;

/// One reading of every tier an expert passes through on its way to the device, kept apart so that each
/// tier's work is visible by itself:
///
/// ```
/// gpu cache <- h2d <- pinned staging <- ram tier <- artifact
/// ```
///
/// @param gpu the device cache's counters
/// @param ram the RAM tier's counters by layer, or null when there is no RAM tier
/// @param artifact the artifact reads the hierarchy caused
/// @param staging the pinned staging slots
/// @param h2d host-to-device copies
public record ExpertHierarchyStats(
        ExpertCacheStats.Snapshot gpu, RamTier.Stats ram, Artifact artifact, Staging staging, H2d h2d) {

    /// @param recordReads positional reads of one record
    /// @param bytesRead bytes they read
    /// @param readNanos time in those reads, summed (parallel reads add up)
    /// @param concurrentReadsHighWater the most reads in flight at once
    public record Artifact(long recordReads, long bytesRead, long readNanos, int concurrentReadsHighWater) {
        public double bytesPerSecond() {
            return this.readNanos == 0 ? 0 : this.bytesRead * 1e9 / this.readNanos;
        }
    }

    /// @param lanes pinned staging slots, one per copy lane
    /// @param recordsStaged records that reached a staging slot
    /// @param ramCopyBytes bytes copied from the RAM tier into staging slots
    /// @param ramCopyNanos time in those copies, summed
    public record Staging(int lanes, long recordsStaged, long ramCopyBytes, long ramCopyNanos) {
        public double ramCopyBytesPerSecond() {
            return this.ramCopyNanos == 0 ? 0 : this.ramCopyBytes * 1e9 / this.ramCopyNanos;
        }
    }

    /// @param bytes bytes copied to the device
    /// @param nanos time the copies held the copy streams, summed over copies
    /// @param copies copies submitted
    public record H2d(long bytes, long nanos, long copies) {
        /// Effective rate over the copies' own durations.
        public double bytesPerSecond() {
            return this.nanos == 0 ? 0 : this.bytes * 1e9 / this.nanos;
        }
    }
}
