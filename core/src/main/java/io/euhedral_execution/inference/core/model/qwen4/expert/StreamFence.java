package io.euhedral_execution.inference.core.model.qwen4.expert;

import io.euhedral_execution.inference.core.gpu.GpuStream;
import java.util.Objects;

/// A [DeviceFence] that is a marker of a compute stream: the owner recorded `marker` with
/// [GpuStream#mark] after the kernels that read the slot, and the copy stream waits for that recording on the
/// device ([GpuStream#await]).
///
/// Two flavours differ in who owns the marker. A shared fence (`new StreamFence(stream, marker)`) leaves it to
/// the caller, which typically re-records one marker per stream after every layer: waiting for the marker waits
/// for its latest recording, which covers every earlier fence on the same marker, so a slot never accumulates
/// one fence per lease. An owning fence ([#owning]) hands the marker to the cache, which closes it
/// ([GpuStream#closeMarker]) after the refill that waited for it has completed; such a marker is used by one
/// fence only.
///
/// @param stream the stream the marker was recorded on, which opened it
/// @param marker the marker from [GpuStream#openMarker]
/// @param closeOnRelease whether releasing the fence closes the marker
public record StreamFence(GpuStream stream, long marker, boolean closeOnRelease) implements DeviceFence {

    public StreamFence {
        Objects.requireNonNull(stream, "stream");
    }

    /// A fence on a marker the caller keeps.
    public StreamFence(GpuStream stream, long marker) {
        this(stream, marker, false);
    }

    /// A fence on a marker that the cache closes when it is done with it.
    public static StreamFence owning(GpuStream stream, long marker) {
        return new StreamFence(stream, marker, true);
    }

    @Override
    public void awaitOn(GpuStream copyStream) {
        copyStream.await(this.marker);
    }

    @Override
    public boolean covers(DeviceFence other) {
        return other instanceof StreamFence fence && fence.stream == this.stream && fence.marker == this.marker;
    }

    @Override
    public void release() {
        if (this.closeOnRelease) this.stream.closeMarker(this.marker);
    }
}
