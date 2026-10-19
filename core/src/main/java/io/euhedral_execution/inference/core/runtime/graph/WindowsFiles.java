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
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/// The Windows file calls both Windows backends share (kernel32): opening a file for overlapped reads, past the
/// cache when asked, a blocking read of such a file, and the calling thread's last error. Its functions are looked
/// up when the class initializes, which only Windows does.
///
/// Every file is opened for overlapped reads (`FILE_FLAG_OVERLAPPED`): the reads then run asynchronously whichever
/// backend submits them, and a blocking read waits for its own.
final class WindowsFiles {

    private WindowsFiles() {}

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup KERNEL32 = SymbolLookup.libraryLookup("kernel32.dll", Arena.global());

    /// The state a call that sets the last error saves for its caller: [#lastError] reads it.
    private static final StructLayout CAPTURE = Linker.Option.captureStateLayout();
    private static final VarHandle LAST_ERROR =
            CAPTURE.varHandle(MemoryLayout.PathElement.groupElement("GetLastError"));
    static final Linker.Option CAPTURES_ERROR = Linker.Option.captureCallState("GetLastError");

    /// `OVERLAPPED`: `ULONG_PTR Internal` (the request's status), `ULONG_PTR InternalHigh` (its bytes), the offset
    /// as `DWORD Offset` and `DWORD OffsetHigh`, `HANDLE hEvent`.
    static final long OVERLAPPED_BYTES = 32;

    static final long OVERLAPPED_INTERNAL = 0;
    static final long OVERLAPPED_OFFSET = 16;
    static final long OVERLAPPED_OFFSET_HIGH = 20;
    static final long OVERLAPPED_EVENT = 24;

    private static final int GENERIC_READ = 0x8000_0000;
    private static final int FILE_SHARE_READ = 0x1;
    private static final int FILE_SHARE_WRITE = 0x2;
    private static final int OPEN_EXISTING = 3;
    private static final int FILE_ATTRIBUTE_NORMAL = 0x80;
    private static final int FILE_FLAG_OVERLAPPED = 0x4000_0000;
    private static final int FILE_FLAG_NO_BUFFERING = 0x2000_0000;
    /// `FILE_INFO_BY_HANDLE_CLASS.FileStorageInfo`.
    private static final int FILE_STORAGE_INFO = 16;
    /// `FILE_STORAGE_INFO`: seven `ULONG`s, the first `LogicalBytesPerSector`, the smallest unit an unbuffered read
    /// may move.
    private static final long FILE_STORAGE_INFO_BYTES = 7 * 4;

    static final int ERROR_HANDLE_EOF = 38;
    static final int ERROR_IO_INCOMPLETE = 996;
    static final int ERROR_IO_PENDING = 997;

    private static final MethodHandle CREATE_FILE;
    private static final MethodHandle CLOSE_HANDLE;
    private static final MethodHandle READ_FILE;
    private static final MethodHandle GET_OVERLAPPED_RESULT;
    private static final MethodHandle CREATE_EVENT;
    private static final MethodHandle GET_FILE_INFORMATION;

    static {
        CREATE_FILE = downcall(
                "CreateFileW",
                FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS),
                CAPTURES_ERROR);
        CLOSE_HANDLE = downcall("CloseHandle", FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS));
        READ_FILE = downcall(
                "ReadFile",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS),
                CAPTURES_ERROR);
        GET_OVERLAPPED_RESULT = downcall(
                "GetOverlappedResult",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT),
                CAPTURES_ERROR);
        CREATE_EVENT = downcall(
                "CreateEventW",
                FunctionDescriptor.of(
                        ValueLayout.ADDRESS,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS),
                CAPTURES_ERROR);
        GET_FILE_INFORMATION = downcall(
                "GetFileInformationByHandleEx",
                FunctionDescriptor.of(
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT,
                        ValueLayout.ADDRESS,
                        ValueLayout.JAVA_INT),
                CAPTURES_ERROR);
    }

    static MethodHandle downcall(String name, FunctionDescriptor descriptor, Linker.Option... options) {
        return LINKER.downcallHandle(KERNEL32.find(name).orElseThrow(), descriptor, options);
    }

    /// Each thread's saved call state, for the calls that capture the last error.
    private static final ThreadLocal<MemorySegment> STATE =
            ThreadLocal.withInitial(() -> Arena.ofAuto().allocate(CAPTURE));

    /// The calling thread's state segment, passed first to a call made with [#CAPTURES_ERROR].
    static MemorySegment state() {
        return STATE.get();
    }

    /// The last error the calling thread's last capturing call saved.
    static int lastError() {
        return (int) LAST_ERROR.get(STATE.get(), 0L);
    }

    /// A file as the backends pass it, and back: a handle's significant bits are its low 32.
    static int file(MemorySegment handle) {
        return (int) handle.address();
    }

    static MemorySegment handle(int file) {
        return MemorySegment.ofAddress(file);
    }

    /// `file`, open for overlapped reads, past the cache when `direct`. Throws when the file cannot be opened, or when
    /// its volume's sectors are coarser than [AsyncReads#DIRECT_ALIGNMENT] (the unbuffered reads' callers align to
    /// that, and Windows needs whole sectors).
    static int open(Path file, boolean direct) throws IOException {
        MemorySegment handle;
        try (Arena path = Arena.ofConfined()) {
            int flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
            handle = (MemorySegment) CREATE_FILE.invokeExact(
                    state(),
                    path.allocateFrom(file.toString(), StandardCharsets.UTF_16LE),
                    GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE,
                    MemorySegment.NULL,
                    OPEN_EXISTING,
                    flags,
                    MemorySegment.NULL);
        } catch (Throwable failure) {
            throw new IOException("opening " + file + " failed", failure);
        }
        // INVALID_HANDLE_VALUE.
        if (handle.address() == -1L) throw new IOException("opening " + file + " failed with error " + lastError());
        if (direct) {
            long sector;
            try {
                sector = logicalSector(handle);
            } catch (IOException failure) {
                closeHandle(handle);
                throw failure;
            }
            if (sector <= 0 || sector > AsyncReads.DIRECT_ALIGNMENT || AsyncReads.DIRECT_ALIGNMENT % sector != 0) {
                closeHandle(handle);
                throw new IOException("the volume of " + file + " has " + sector + "-byte sectors; unbuffered reads"
                        + " are aligned to " + AsyncReads.DIRECT_ALIGNMENT);
            }
        }
        return file(handle);
    }

    /// The logical sector of `handle`'s volume: the smallest unit an unbuffered read may move.
    private static long logicalSector(MemorySegment handle) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment info = arena.allocate(FILE_STORAGE_INFO_BYTES, 4);
            int ok = (int) GET_FILE_INFORMATION.invokeExact(
                    state(), handle, FILE_STORAGE_INFO, info, (int) FILE_STORAGE_INFO_BYTES);
            if (ok == 0) throw new IOException("the volume's sector size is unknown: error " + lastError());
            return Integer.toUnsignedLong(info.get(ValueLayout.JAVA_INT, 0));
        } catch (IOException failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IOException("the volume's sector size is unknown", failure);
        }
    }

    static void close(int file) {
        closeHandle(handle(file));
    }

    static void closeHandle(MemorySegment handle) {
        int ok;
        try {
            ok = (int) CLOSE_HANDLE.invokeExact(handle);
        } catch (Throwable failure) {
            throw new IllegalStateException("closing a handle failed", failure);
        }
        if (ok == 0) throw new IllegalStateException("closing a handle failed");
    }

    /// Starts an overlapped read of `length` bytes at `offset` of `file` into `address`, described by `overlapped`
    /// (zeroed but for its event). 0 when the read started or finished; otherwise the error that stopped it, which
    /// leaves nothing in flight.
    static int startRead(int file, long address, int length, long offset, MemorySegment overlapped) {
        overlapped.set(ValueLayout.JAVA_LONG, OVERLAPPED_INTERNAL, 0L);
        overlapped.set(ValueLayout.JAVA_LONG, OVERLAPPED_INTERNAL + 8, 0L);
        overlapped.set(ValueLayout.JAVA_INT, OVERLAPPED_OFFSET, (int) offset);
        overlapped.set(ValueLayout.JAVA_INT, OVERLAPPED_OFFSET_HIGH, (int) (offset >>> 32));
        int ok;
        try {
            ok = (int) READ_FILE.invokeExact(
                    state(), handle(file), MemorySegment.ofAddress(address), length, MemorySegment.NULL, overlapped);
        } catch (Throwable failure) {
            throw new IllegalStateException("ReadFile failed", failure);
        }
        if (ok != 0) return 0;
        int error = lastError();
        return error == ERROR_IO_PENDING ? 0 : error;
    }

    /// The bytes `overlapped`'s read moved, waiting for it when `wait`; a negated error when it failed (end of file
    /// included), or [Integer#MIN_VALUE] when it is still running and `wait` is false.
    static long result(int file, MemorySegment overlapped, MemorySegment transferred, boolean wait) {
        int ok;
        try {
            ok = (int) GET_OVERLAPPED_RESULT.invokeExact(state(), handle(file), overlapped, transferred, wait ? 1 : 0);
        } catch (Throwable failure) {
            throw new IllegalStateException("GetOverlappedResult failed", failure);
        }
        if (ok != 0) return Integer.toUnsignedLong(transferred.get(ValueLayout.JAVA_INT, 0));
        int error = lastError();
        if (error == ERROR_IO_INCOMPLETE) return Integer.MIN_VALUE;
        return -(long) error;
    }

    /// Reads `length` bytes at `offset` of `file` into `address`, blocking the calling thread until they are in. The
    /// read's event has its low bit set, so a file bound to a completion port posts no completion for it.
    static void readBlocking(int file, long address, long length, long offset) throws IOException {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment event;
            try {
                event = (MemorySegment) CREATE_EVENT.invokeExact(state(), MemorySegment.NULL, 1, 0, MemorySegment.NULL);
            } catch (Throwable failure) {
                throw new IOException("CreateEventW failed", failure);
            }
            if (event.address() == 0) throw new IOException("CreateEventW failed with error " + lastError());
            try {
                MemorySegment overlapped = arena.allocate(OVERLAPPED_BYTES, 8);
                MemorySegment transferred = arena.allocate(ValueLayout.JAVA_INT);
                overlapped.set(ValueLayout.JAVA_LONG, OVERLAPPED_EVENT, event.address() | 1L);
                long done = 0;
                while (done < length) {
                    int chunk = (int) Math.min(length - done, AsyncReads.MAX_READ);
                    int error = startRead(file, address + done, chunk, offset + done, overlapped);
                    long read = error != 0 ? -(long) error : result(file, overlapped, transferred, true);
                    if (read == -ERROR_HANDLE_EOF || read == 0)
                        throw new IOException("the file ends before byte " + (offset + length));
                    if (read < 0)
                        throw new IOException(
                                "reading " + length + " bytes at " + offset + " failed with error " + -read);
                    done += read;
                }
            } finally {
                closeHandle(event);
            }
        }
    }

    static IOException failure(int error) {
        return error == ERROR_HANDLE_EOF
                ? new IOException("the read ran past the end of the file")
                : new IOException("read failed with Windows error " + error);
    }
}
