package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/// One block of ordinary (pageable) off-heap memory with an explicit lifetime.
///
/// Where the platform allows it the block is an anonymous mapping that the kernel commits as pages are first
/// touched: reserving a large tier costs nothing until records arrive, and the memory the engine holds is the
/// memory it has filled. Elsewhere it is an arena allocation, which commits (and zeroes) the whole block at
/// once.
final class PageableMemory implements AutoCloseable {

    private static final int PROT_READ_WRITE = 3;
    private static final int MAP_PRIVATE = 0x02;
    private static final int MAP_ANONYMOUS = 0x20;
    private static final int MAP_NORESERVE = 0x4000;

    private static final MethodHandle MMAP;
    private static final MethodHandle MUNMAP;

    static {
        MethodHandle mmap = null;
        MethodHandle munmap = null;
        if (System.getProperty("os.name", "").toLowerCase().contains("linux")) {
            try {
                Linker linker = Linker.nativeLinker();
                var lookup = linker.defaultLookup();
                mmap = linker.downcallHandle(
                        lookup.find("mmap").orElseThrow(),
                        FunctionDescriptor.of(
                                ValueLayout.ADDRESS,
                                ValueLayout.ADDRESS,
                                ValueLayout.JAVA_LONG,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_INT,
                                ValueLayout.JAVA_LONG));
                munmap = linker.downcallHandle(
                        lookup.find("munmap").orElseThrow(),
                        FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            } catch (RuntimeException | Error unavailable) {
                mmap = null;
                munmap = null;
            }
        }
        MMAP = mmap;
        MUNMAP = munmap;
    }

    private final MemorySegment segment;
    private final Arena arena;
    private final AtomicBoolean closed = new AtomicBoolean();

    private PageableMemory(MemorySegment segment, Arena arena) {
        this.segment = segment;
        this.arena = arena;
    }

    /// A block of `bytes` bytes, page aligned.
    static PageableMemory allocate(long bytes) {
        if (bytes <= 0) throw new IllegalArgumentException("bytes must be positive");
        if (MMAP != null) {
            try {
                MemorySegment address = (MemorySegment) MMAP.invokeExact(
                        MemorySegment.NULL,
                        bytes,
                        PROT_READ_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                        -1,
                        0L);
                if (address.address() != -1L && address.address() != 0L)
                    return new PageableMemory(address.reinterpret(bytes), null);
            } catch (Throwable failure) {
                // fall through to the arena
                if (failure instanceof Error error && !(failure instanceof LinkageError)) throw error;
            }
        }
        Arena arena = Arena.ofShared();
        try {
            return new PageableMemory(arena.allocate(bytes, 4096), arena);
        } catch (RuntimeException | Error failure) {
            arena.close();
            throw failure;
        }
    }

    MemorySegment segment() {
        return this.segment;
    }

    /// Whether the block is committed lazily.
    boolean lazy() {
        return this.arena == null;
    }

    @Override
    public void close() {
        if (!this.closed.compareAndSet(false, true)) return;
        if (this.arena != null) {
            this.arena.close();
            return;
        }
        try {
            int ignored = (int) Objects.requireNonNull(MUNMAP)
                    .invokeExact(MemorySegment.ofAddress(this.segment.address()), this.segment.byteSize());
        } catch (Throwable failure) {
            if (failure instanceof Error error) throw error;
            throw new IllegalStateException("releasing the pageable block failed", failure);
        }
    }
}
