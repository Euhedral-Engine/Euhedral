package io.euhedral_execution.inference.core.runtime.graph;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

/// Windows' I/O rings (ioringapi.h, KernelBase; Windows 11 and Server 2022 on): io_uring's counterpart, a ring per
/// submitting thread, which the sink reaps. Files and buffers are passed raw (`IORING_REF_RAW`): a read's file and
/// destination are any of the backend's, given per read, so there is nothing fixed to register. Its functions are
/// looked up when the class initializes, which only Windows does.
///
/// A submission Windows refuses leaves every entry it was given in the ring's submission queue, and an entry built
/// cannot be withdrawn: the ring then retires (it is never submitted again, so the entry is never sent) and its thread
/// makes a new one.
final class IoRingBackend implements ReadBackend {

    private static final Linker LINKER = Linker.nativeLinker();

    private static final int IORING_VERSION_1 = 1;
    private static final int IORING_OP_READ = 1;
    private static final int IORING_REF_RAW = 0;
    private static final int IOSQE_FLAGS_NONE = 0;
    private static final int S_OK = 0;
    private static final int S_FALSE = 1;
    /// `HRESULT_FROM_WIN32(ERROR_HANDLE_EOF)`: the read began past the end of the file.
    private static final int END_OF_FILE = 0x8007_0000 | WindowsFiles.ERROR_HANDLE_EOF;

    /// `IORING_CREATE_FLAGS`: the required and the advisory flags, two 32-bit enumerations, passed by value.
    private static final StructLayout CREATE_FLAGS = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("Required"), ValueLayout.JAVA_INT.withName("Advisory"));
    /// `IORING_HANDLE_REF` and `IORING_BUFFER_REF`: an `IORING_REF_KIND` and an 8-byte union (a raw handle or
    /// address; a registered index, with an offset for a buffer), passed by value.
    private static final StructLayout REF = MemoryLayout.structLayout(
            ValueLayout.JAVA_INT.withName("Kind"), MemoryLayout.paddingLayout(4), ValueLayout.ADDRESS.withName("Raw"));
    /// `IORING_CQE`: `UINT_PTR UserData`, `HRESULT ResultCode`, `ULONG_PTR Information` (the bytes read).
    private static final long CQE_BYTES = 24;
    private static final long CQE_RESULT = 8;
    private static final long CQE_INFORMATION = 16;
    /// `IORING_CAPABILITIES`: `IORING_VERSION MaxVersion`, `UINT32 MaxSubmissionQueueSize`,
    /// `UINT32 MaxCompletionQueueSize`, `IORING_FEATURE_FLAGS FeatureFlags`.
    private static final long CAPABILITIES_BYTES = 16;

    private static final MethodHandle QUERY_CAPABILITIES;
    private static final MethodHandle CREATE;
    private static final MethodHandle CLOSE;
    private static final MethodHandle OP_SUPPORTED;
    private static final MethodHandle BUILD_READ;
    private static final MethodHandle SUBMIT;
    private static final MethodHandle POP_COMPLETION;

    static {
        SymbolLookup kernelBase = SymbolLookup.libraryLookup("KernelBase.dll", Arena.global());
        QUERY_CAPABILITIES = handle(
                kernelBase,
                "QueryIoRingCapabilities",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        CREATE = handle(
                kernelBase,
                "CreateIoRing",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        CREATE_FLAGS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS));
        CLOSE = handle(kernelBase, "CloseIoRing", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        OP_SUPPORTED = handle(
                kernelBase,
                "IsIoRingOpSupported",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        BUILD_READ = handle(
                kernelBase,
                "BuildIoRingReadFile",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        REF,
                        REF,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_LONG,
                        ValueLayout.JAVA_INT));
        SUBMIT = handle(
                kernelBase,
                "SubmitIoRing",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS));
        POP_COMPLETION = handle(
                kernelBase,
                "PopIoRingCompletion",
                FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.ADDRESS));
    }

    private static MethodHandle handle(SymbolLookup library, String name, FunctionDescriptor descriptor) {
        return LINKER.downcallHandle(library.find(name).orElseThrow(), descriptor);
    }

    /// Whether this Windows reads through I/O rings: it has them, at version 1 or later, and a ring can read. False
    /// for any failure, a missing function included.
    static boolean usable() {
        try (Arena arena = Arena.ofConfined()) {
            int entries = submissionEntries(arena);
            if (entries <= 0) return false;
            MemorySegment ring = create(arena, entries);
            try {
                return (int) OP_SUPPORTED.invokeExact(ring, IORING_OP_READ) != 0;
            } finally {
                int ignored = (int) CLOSE.invokeExact(ring);
            }
        } catch (Throwable unavailable) {
            return false;
        }
    }

    /// The entries of a ring's submission queue: [AsyncReads#RING_ENTRIES], or the system's most when that is less;
    /// 0 when the system has no version-1 rings.
    private static int submissionEntries(Arena arena) throws Throwable {
        MemorySegment capabilities = arena.allocate(CAPABILITIES_BYTES, 4);
        int status = (int) QUERY_CAPABILITIES.invokeExact(capabilities);
        if (status < 0 || capabilities.get(ValueLayout.JAVA_INT, 0) < IORING_VERSION_1) return 0;
        long most = Integer.toUnsignedLong(capabilities.get(ValueLayout.JAVA_INT, 4));
        return (int) Math.min(AsyncReads.RING_ENTRIES, most);
    }

    /// A version-1 ring of `entries` submissions and twice as many completions.
    private static MemorySegment create(Arena arena, int entries) throws Throwable {
        MemorySegment flags = arena.allocate(CREATE_FLAGS);
        MemorySegment out = arena.allocate(ValueLayout.ADDRESS);
        int status = (int) CREATE.invokeExact(IORING_VERSION_1, flags, entries, 2 * entries, out);
        if (status < 0) throw new IllegalStateException(String.format("CreateIoRing failed with 0x%08x", status));
        return out.get(ValueLayout.ADDRESS, 0);
    }

    private final Arena arena = Arena.ofShared();
    private final int entries;

    IoRingBackend() {
        try {
            this.entries = submissionEntries(this.arena);
        } catch (Throwable failure) {
            this.arena.close();
            throw new IllegalStateException("Windows I/O rings are unavailable", failure);
        }
        if (this.entries <= 0) {
            this.arena.close();
            throw new IllegalStateException("Windows I/O rings are unavailable");
        }
    }

    @Override
    public boolean sharedQueue() {
        return false;
    }

    @Override
    public Queue queue(int tags) {
        return new Ring();
    }

    @Override
    public int open(Path file, boolean direct) throws IOException {
        return WindowsFiles.open(file, direct);
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
        return new IOException(String.format("read failed with HRESULT 0x%08x", error));
    }

    @Override
    public void close() {
        this.arena.close();
    }

    /// One thread's ring: that thread is its only submitter; the sink reaps it.
    private final class Ring implements Queue {
        private final MemorySegment ring;
        private final MemorySegment cqe;
        private final MemorySegment handleRef;
        private final MemorySegment bufferRef;
        /// Reads submitted and not yet reaped: at most the submission queue's entries, so the completion queue (twice
        /// as long) never fills.
        private final AtomicInteger inFlight = new AtomicInteger();
        private volatile boolean retired;
        private boolean closed;

        Ring() {
            Arena arena = IoRingBackend.this.arena;
            try {
                this.ring = create(arena, IoRingBackend.this.entries);
            } catch (Throwable failure) {
                throw new IllegalStateException("Windows I/O ring setup failed", failure);
            }
            this.cqe = arena.allocate(CQE_BYTES, 8);
            this.handleRef = arena.allocate(REF);
            this.bufferRef = arena.allocate(REF);
            this.handleRef.set(ValueLayout.JAVA_INT, 0, IORING_REF_RAW);
            this.bufferRef.set(ValueLayout.JAVA_INT, 0, IORING_REF_RAW);
        }

        @Override
        public boolean submit(int file, long address, int length, long offset, long tag) {
            if (this.retired) return false;
            if (this.inFlight.incrementAndGet() > IoRingBackend.this.entries) {
                this.inFlight.decrementAndGet();
                return false;
            }
            int built;
            try {
                this.handleRef.set(ValueLayout.ADDRESS, 8, WindowsFiles.handle(file));
                this.bufferRef.set(ValueLayout.ADDRESS, 8, MemorySegment.ofAddress(address));
                built = (int) BUILD_READ.invokeExact(
                        this.ring, this.handleRef, this.bufferRef, length, offset, tag, IOSQE_FLAGS_NONE);
            } catch (Throwable failure) {
                this.inFlight.decrementAndGet();
                throw new IllegalStateException("Windows I/O ring submission failed", failure);
            }
            // Nothing was built (the queue is full): nothing is sent.
            if (built < 0) {
                this.inFlight.decrementAndGet();
                return false;
            }
            int submitted;
            try {
                submitted = (int) SUBMIT.invokeExact(this.ring, 0, 0, MemorySegment.NULL);
            } catch (Throwable failure) {
                submitted = -1;
            }
            if (submitted >= 0) return true;
            // The entry stays in the submission queue: the ring is never submitted again.
            this.retired = true;
            this.inFlight.decrementAndGet();
            return false;
        }

        @Override
        public boolean reap(Completion into) {
            if (this.closed) return false;
            int status;
            try {
                status = (int) POP_COMPLETION.invokeExact(this.ring, this.cqe);
            } catch (Throwable failure) {
                throw new IllegalStateException("Windows I/O ring completion failed", failure);
            }
            if (status == S_FALSE) return false;
            if (status != S_OK)
                throw new IllegalStateException(String.format("PopIoRingCompletion failed with 0x%08x", status));
            this.inFlight.decrementAndGet();
            into.tag = this.cqe.get(ValueLayout.JAVA_LONG, 0);
            int result = this.cqe.get(ValueLayout.JAVA_INT, CQE_RESULT);
            if (result >= 0) {
                into.bytes = this.cqe.get(ValueLayout.JAVA_LONG, CQE_INFORMATION);
                into.error = 0;
            } else if (result == END_OF_FILE) {
                into.bytes = 0;
                into.error = 0;
            } else {
                into.bytes = 0;
                into.error = result;
            }
            return true;
        }

        @Override
        public boolean retired() {
            return this.retired;
        }

        @Override
        public void close() {
            if (this.closed) return;
            this.closed = true;
            int status;
            try {
                status = (int) CLOSE.invokeExact(this.ring);
            } catch (Throwable failure) {
                throw new IllegalStateException("Windows I/O ring teardown failed", failure);
            }
            if (status < 0) throw new IllegalStateException(String.format("CloseIoRing failed with 0x%08x", status));
        }
    }
}
