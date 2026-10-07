package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;

/// Moves expert records from pinned host memory into device memory on copy streams. A stream is
/// only an order of copies on the device: any thread may submit to any stream, and one stream may
/// hold many copies from different staging buffers at once. How many streams there are is
/// independent of the staging buffers and of the loads in flight.
///
/// A copy is submitted on a stream, records a device marker behind itself, and reports its
/// retirement through a listener that a CUDA driver thread calls (and that may only enqueue). Work
/// that reads the destination waits for the marker on its own stream; the host never waits for the
/// bytes.
public interface ExpertTransfer extends AutoCloseable {

    /// The copy streams.
    int streams();

    /// The stream the next copy goes on: the transfer's own choice, in turn.
    default int nextStream() {
        return 0;
    }

    /// Opens a device marker that [#stream] records behind a copy.
    long openMarker();

    void closeMarker(long marker);

    /// Submits the copy of `record` to `deviceAddress` on copy stream `stream`, behind `waitFor`
    /// (null for nothing) on the device, records `marker` behind it, and arms `retired` on that
    /// stream: it is called (on a driver thread) once the copy retired, and its ticket is then given
    /// to [#confirm]. Any thread, any stream: copies submitted to one stream by different threads
    /// are ordered as the stream receives them, and a marker or retirement recorded behind one copy
    /// may also stand behind another's, which only orders later than needed. A fence the copy waits
    /// for holds the stream's later copies too.
    ///
    /// When this method throws nothing is armed and the copy may or may not have been queued: the
    /// caller must [#recover] the stream before reusing the record's staging buffer.
    void stream(
            int stream,
            HostRecord record,
            long deviceAddress,
            DeviceFence waitFor,
            long marker,
            GpuStream.RetirementListener retired);

    /// The device failure of the copy whose boundary retired as `ticket` on `stream`, or null.
    Throwable confirm(int stream, long ticket);

    /// Proves copy stream `stream` idle after a failed [#stream].
    void recover(int stream, Throwable failure);

    /// Waits for the copies in flight (on the device) and releases what the transfer holds.
    @Override
    void close();
}
