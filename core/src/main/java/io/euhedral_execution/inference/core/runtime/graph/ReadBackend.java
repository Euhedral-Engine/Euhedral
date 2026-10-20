package io.euhedral_execution.inference.core.runtime.graph;

import io.euhedral_execution.hardware_utils.util.OSName;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.BooleanSupplier;

/// The operating system's half of [AsyncReads]: how a file is opened, how a read is handed to the kernel and how its
/// completion comes back. [AsyncReads] keeps everything else (the pending reads, short reads, the frames, the
/// accounting), the same on every system.
///
/// A file is an `int` on every system: Linux's descriptor, or a Windows handle, whose significant bits are its low
/// 32 (Windows hands 64-bit processes 32-bit handles so they can be shared with 32-bit ones; a handle may be truncated
/// and sign-extended back).
interface ReadBackend extends AutoCloseable {

    /// The backends, in the order [#choose] prefers them on their system.
    enum Kind {
        /// Linux's io_uring, through liburing.
        IO_URING("io_uring"),
        /// Windows' I/O rings (Windows 11 and Server 2022 on): io_uring's counterpart, a ring per submitting thread.
        IORING("Windows I/O rings"),
        /// Windows' overlapped reads and one I/O completion port: every Windows that has no I/O rings.
        COMPLETION_PORT("Windows I/O completion ports");

        private final String label;

        Kind(String label) {
            this.label = label;
        }

        @Override
        public String toString() {
            return this.label;
        }
    }

    /// The backend for the system named `osName` (the `os.name` property), or null when it has none. `ioRings` says
    /// whether Windows' I/O rings can read; it is asked only on Windows, so nothing of Windows' is looked up
    /// elsewhere.
    static Kind choose(OSName os, BooleanSupplier ioRings) {
        return switch (os) {
            case LINUX -> Kind.IO_URING;
            case WINDOWS -> ioRings.getAsBoolean() ? Kind.IORING : Kind.COMPLETION_PORT;
            default -> null;
        };
    }

    /// Why `os` has no asynchronous reads.
    static String unsupported(OSName os) {
        return "asynchronous reads need Linux (io_uring) or Windows (I/O rings or I/O completion ports); this machine"
                + " runs " + os;
    }

    /// What the reads of one submitting thread go through. The thread is its only submitter; [#reap] is called by
    /// one thread at a time, which may be another.
    interface Queue {
        /// Hands the read of `length` bytes at `offset` of `file` into `address` to the kernel, tagged
        /// `tag`. True when the kernel took it: exactly one completion of `tag` follows. False when it did
        /// not: no completion of `tag` ever follows, and nothing is ever read into `address` for it. May
        /// throw only when nothing was taken.
        boolean submit(int file, long address, int length, long offset, long tag);

        /// Takes one completion into `into`; false when there is none now. Never blocks.
        boolean reap(Completion into);

        /// Whether the queue takes no more reads: its thread needs a new one. It is still reaped.
        boolean retired();

        /// Releases the queue. No read of it may be in flight.
        void close();
    }

    /// A completed read: `tag` read `bytes` bytes (0: the file ended first), or failed with `error`, the system's
    /// code (non-zero).
    final class Completion {
        long tag;
        long bytes;
        int error;
    }

    /// Whether every thread submits to one queue ([#queue] is called once), instead of one queue each.
    boolean sharedQueue();

    /// A new queue, whose tags are below `tags`.
    Queue queue(int tags);

    /// `file`, open for reading; past the page cache when `direct`, where every read must be aligned (offset, length
    /// and address) to [AsyncReads#DIRECT_ALIGNMENT]. Throws when the file system cannot read `file` so.
    int open(Path file, boolean direct) throws IOException;

    void close(int file);

    /// Reads `length` bytes at `offset` of `file` into `address`, blocking the calling thread until they are in.
    void readBlocking(int file, long address, long length, long offset) throws IOException;

    /// The failure that `error`, a completion's code, names.
    IOException failure(int error);

    /// Releases what the backend holds beyond its queues.
    @Override
    void close();
}
