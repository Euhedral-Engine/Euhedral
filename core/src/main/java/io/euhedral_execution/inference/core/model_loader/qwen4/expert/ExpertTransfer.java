package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;

/// Moves expert records from pinned host memory into device memory on copy lanes: each lane has a
/// copy stream of its own, so lanes overlap and need no lock between them.
///
/// A copy is submitted on a lane, records a device marker behind itself, and reports its retirement
/// through a listener that a CUDA driver thread calls (and that may only enqueue). Work that reads
/// the destination waits for the marker on its own stream; the host never waits for the bytes.
public interface ExpertTransfer extends AutoCloseable {

    /// The copy lanes: the most copies that are in flight at once.
    int lanes();

    /// Opens a device marker that [#stream] records behind a copy.
    long openMarker();

    void closeMarker(long marker);

    /// Submits the copy of `record` to `deviceAddress` on the copy stream of `lane`, behind
    /// `waitFor` (null for nothing) on the device, records `marker` behind it, and arms `retired`
    /// on that stream: it is called (on a driver thread) once the copy retired, and its ticket is
    /// then given to [#confirm].
    ///
    /// A lane is used by one caller at a time, from `stream` until its `confirm`, which is what
    /// makes a lock unnecessary: the lane's stream is the order of its copies. When this method
    /// throws nothing is armed and the copy may or may not have been queued: the caller must
    /// [#recover] the lane before reusing the record's staging.
    void stream(
            int lane,
            HostRecord record,
            long deviceAddress,
            DeviceFence waitFor,
            long marker,
            GpuStream.RetirementListener retired);

    /// The device failure of the copy whose boundary retired as `ticket`, or null.
    Throwable confirm(int lane, long ticket);

    /// Proves the lane's copy stream idle after a failed [#stream].
    void recover(int lane, Throwable failure);

    /// Waits for the copies in flight (on the device) and releases what the transfer holds.
    @Override
    void close();
}
