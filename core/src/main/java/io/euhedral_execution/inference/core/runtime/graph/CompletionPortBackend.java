package io.euhedral_execution.inference.core.runtime.graph;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicIntegerArray;

/// Windows' reads where it has no I/O rings: overlapped `ReadFile`s, whose completions an I/O completion port
/// collects. Every thread submits to the one port (a file can be bound to only one), and the sink takes the
/// completions off it without waiting (`GetQueuedCompletionStatusEx`, no timeout), one poll at a time. Its functions
/// are
/// looked up when the class initializes, which only Windows does.
///
/// Each tag has an `OVERLAPPED` of its own, which the kernel writes until the read completes: a tag is a pending
/// read's slot, held by the read until it is reaped, so the structure is never reused while its read runs.
final class CompletionPortBackend implements ReadBackend {

    /// `OVERLAPPED_ENTRY`: `ULONG_PTR lpCompletionKey`, `LPOVERLAPPED lpOverlapped`, `ULONG_PTR Internal`,
    /// `DWORD dwNumberOfBytesTransferred`, padded to 8 bytes.
    private static final long ENTRY_BYTES = 32;
    private static final long ENTRY_OVERLAPPED = 8;
    private static final long ENTRY_TRANSFERRED = 24;
    private static final int WAIT_TIMEOUT = 258;

    private static final MethodHandle CREATE_PORT;
    private static final MethodHandle GET_COMPLETIONS;
    private static final MethodHandle POST_COMPLETION;

    static {
        CREATE_PORT = WindowsFiles.downcall(
                "CreateIoCompletionPort",
                FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT));
        GET_COMPLETIONS = WindowsFiles.downcall(
                "GetQueuedCompletionStatusEx",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT),
                WindowsFiles.CAPTURES_ERROR);
        POST_COMPLETION = WindowsFiles.downcall(
                "PostQueuedCompletionStatus",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.ADDRESS));
    }

    /// Null: the class initialized, so its functions are there (a missing one fails the initialization).
    static Throwable unavailable() {
        return null;
    }

    private final Arena arena = Arena.ofShared();
    private final MemorySegment port;

    CompletionPortBackend() {
        MemorySegment created;
        try {
            created = (MemorySegment) CREATE_PORT.invokeExact(MemorySegment.ofAddress(-1L), MemorySegment.NULL, 0L, 0);
        } catch (Throwable failure) {
            this.arena.close();
            throw new IllegalStateException("creating an I/O completion port failed", failure);
        }
        if (created.address() == 0) {
            this.arena.close();
            throw new IllegalStateException("creating an I/O completion port failed");
        }
        this.port = created;
    }

    @Override
    public boolean sharedQueue() {
        return true;
    }

    @Override
    public Queue queue(int tags) {
        return new Port(tags);
    }

    /// `file`, bound to the port: its overlapped reads complete there.
    @Override
    public int open(Path file, boolean direct) throws IOException {
        int opened = WindowsFiles.open(file, direct);
        MemorySegment bound;
        try {
            bound = (MemorySegment) CREATE_PORT.invokeExact(WindowsFiles.handle(opened), this.port, 0L, 0);
        } catch (Throwable failure) {
            WindowsFiles.close(opened);
            throw new IOException("binding " + file + " to the completion port failed", failure);
        }
        if (bound.address() == 0) {
            WindowsFiles.close(opened);
            throw new IOException("binding " + file + " to the completion port failed");
        }
        return opened;
    }

    @Override
    public void close(int file) {
        WindowsFiles.close(file);
    }

    @Override
    public void readBlocking(int file, long address, long length, long offset) throws IOException {
        WindowsFiles.readBlocking(file, address, length, offset);
    }

    @Override
    public IOException failure(int error) {
        return WindowsFiles.failure(error);
    }

    @Override
    public void close() {
        WindowsFiles.closeHandle(this.port);
        this.arena.close();
    }

    /// The port, as every thread's queue.
    private final class Port implements Queue {
        private final int tags;
        /// One `OVERLAPPED` per tag.
        private final MemorySegment overlapped;
        /// Each tag's file, to ask a failed read's error of.
        private final AtomicIntegerArray files;
        /// Each tag's error when its read failed to start, posted to the port in its stead.
        private final AtomicIntegerArray refused;
        private final MemorySegment entries;
        private final MemorySegment removed;
        private final MemorySegment transferred;
        /// The entries the last take removed, and the next to hand out.
        private int taken;
        private int next;

        Port(int tags) {
            Arena arena = CompletionPortBackend.this.arena;
            this.tags = tags;
            this.overlapped = arena.allocate(WindowsFiles.OVERLAPPED_BYTES * tags, 8);
            this.files = new AtomicIntegerArray(tags);
            this.refused = new AtomicIntegerArray(tags);
            this.entries = arena.allocate(ENTRY_BYTES * AsyncReads.RING_ENTRIES, 8);
            this.removed = arena.allocate(ValueLayout.JAVA_INT);
            this.transferred = arena.allocate(ValueLayout.JAVA_INT);
        }

        private MemorySegment overlapped(long tag) {
            return this.overlapped.asSlice(WindowsFiles.OVERLAPPED_BYTES * tag, WindowsFiles.OVERLAPPED_BYTES);
        }

        /// A read that fails to start still completes: its error is posted to the port as its completion, so the
        /// read is not handed back, unless the post itself fails.
        @Override
        public boolean submit(int file, long address, int length, long offset, long tag) {
            if (tag < 0 || tag >= this.tags) throw new IllegalArgumentException("tag " + tag + " is out of range");
            int slot = (int) tag;
            MemorySegment overlapped = overlapped(tag);
            this.files.set(slot, file);
            this.refused.set(slot, 0);
            overlapped.set(ValueLayout.JAVA_LONG, WindowsFiles.OVERLAPPED_EVENT, 0L);
            int error = WindowsFiles.startRead(file, address, length, offset, overlapped);
            if (error == 0) return true;
            this.refused.set(slot, error);
            int posted;
            try {
                posted = (int) POST_COMPLETION.invokeExact(CompletionPortBackend.this.port, 0, 0L, overlapped);
            } catch (Throwable failure) {
                posted = 0;
            }
            return posted != 0;
        }

        @Override
        public boolean reap(Completion into) {
            if (this.next == this.taken && !take()) return false;
            MemorySegment entry = this.entries.asSlice(ENTRY_BYTES * this.next++, ENTRY_BYTES);
            long tag = (entry.get(ValueLayout.ADDRESS, ENTRY_OVERLAPPED).address() - this.overlapped.address())
                    / WindowsFiles.OVERLAPPED_BYTES;
            int slot = (int) tag;
            into.tag = tag;
            into.bytes = 0;
            into.error = 0;
            int refused = this.refused.get(slot);
            if (refused != 0) {
                if (refused != WindowsFiles.ERROR_HANDLE_EOF) into.error = refused;
                return true;
            }
            if (overlapped(tag).get(ValueLayout.JAVA_LONG, WindowsFiles.OVERLAPPED_INTERNAL) == 0) {
                into.bytes = Integer.toUnsignedLong(entry.get(ValueLayout.JAVA_INT, ENTRY_TRANSFERRED));
                return true;
            }
            // A failed read: its Win32 error, from the finished request (end of file is 0 bytes).
            long result = WindowsFiles.result(this.files.get(slot), overlapped(tag), this.transferred, false);
            if (result >= 0) into.bytes = result;
            else if (result != -WindowsFiles.ERROR_HANDLE_EOF) into.error = (int) -result;
            return true;
        }

        /// Takes the completions the port holds, without waiting. False when it holds none.
        private boolean take() {
            this.taken = 0;
            this.next = 0;
            int ok;
            try {
                ok = (int) GET_COMPLETIONS.invokeExact(
                        WindowsFiles.state(),
                        CompletionPortBackend.this.port,
                        this.entries,
                        AsyncReads.RING_ENTRIES,
                        this.removed,
                        0,
                        0);
            } catch (Throwable failure) {
                throw new IllegalStateException("GetQueuedCompletionStatusEx failed", failure);
            }
            if (ok == 0) {
                int error = WindowsFiles.lastError();
                if (error == WAIT_TIMEOUT) return false;
                throw new IllegalStateException("GetQueuedCompletionStatusEx failed with error " + error);
            }
            this.taken = this.removed.get(ValueLayout.JAVA_INT, 0);
            return this.taken > 0;
        }

        @Override
        public boolean retired() {
            return false;
        }

        @Override
        public void close() {}
    }
}
