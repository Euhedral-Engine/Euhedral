package io.euhedral_execution.inference.core.guidance;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/// The llguidance C API (`llguidance.h`, built from the pinned crate without its thread pool), bound with FFM.
///
/// Only what grammar-constrained sampling needs is bound: tokenizers from token bytes, matchers from Lark grammars,
/// token masks, and token commits. Struct offsets are those of the x86-64 C ABI on Linux and Windows.
public final class Llguidance {
    private static final Map<Path, Llguidance> LOADED = new ConcurrentHashMap<>();
    private static final Linker LINKER = Linker.nativeLinker();
    private static final ValueLayout ADDRESS = ValueLayout.ADDRESS;
    private static final ValueLayout.OfLong SIZE = ValueLayout.JAVA_LONG;
    private static final ValueLayout.OfInt U32 = ValueLayout.JAVA_INT;
    private static final ValueLayout.OfBoolean BOOL = ValueLayout.JAVA_BOOLEAN;

    /// `LlgTokenizerInitV2`.
    static final long TOKENIZER_INIT_SIZE = 96;
    static final long TOKENIZER_INIT_VOCAB_SIZE = 8;
    static final long TOKENIZER_INIT_EOS = 12;
    static final long TOKENIZER_INIT_TOKEN_LENS = 16;
    static final long TOKENIZER_INIT_TOKEN_BYTES = 24;
    static final long TOKENIZER_INIT_APPROXIMATE_TOKENIZE = 56;
    static final long TOKENIZER_INIT_EOS_EXTRA = 80;
    static final long TOKENIZER_INIT_EOS_EXTRA_COUNT = 88;
    /// `LlgConstraintInit`.
    static final long CONSTRAINT_INIT_SIZE = 80;
    static final long CONSTRAINT_INIT_LOG_BUFFER = 8;
    static final long CONSTRAINT_INIT_LOG_STDERR = 12;

    final MethodHandle newTokenizer;
    final MethodHandle freeTokenizer;
    final MethodHandle constraintInitDefaults;
    final MethodHandle newMatcher;
    final MethodHandle matcherError;
    final MethodHandle matcherIsError;
    final MethodHandle computeMaskInto;
    final MethodHandle maskByteSize;
    final MethodHandle consumeToken;
    final MethodHandle isAccepting;
    final MethodHandle cloneMatcher;
    final MethodHandle freeMatcher;

    private Llguidance(Path library) {
        SymbolLookup symbols = SymbolLookup.libraryLookup(library, Arena.global());
        this.newTokenizer =
                bind(symbols, "llg_new_tokenizer_v2", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, SIZE));
        this.freeTokenizer = bind(symbols, "llg_free_tokenizer", FunctionDescriptor.ofVoid(ADDRESS));
        this.constraintInitDefaults =
                bind(symbols, "llg_constraint_init_set_defaults", FunctionDescriptor.ofVoid(ADDRESS, ADDRESS));
        this.newMatcher = bind(symbols, "llg_new_matcher", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, ADDRESS));
        this.matcherError = bind(symbols, "llg_matcher_get_error", FunctionDescriptor.of(ADDRESS, ADDRESS));
        this.matcherIsError = bind(symbols, "llg_matcher_is_error", FunctionDescriptor.of(BOOL, ADDRESS));
        this.computeMaskInto =
                bind(symbols, "llg_matcher_compute_mask_into", FunctionDescriptor.of(U32, ADDRESS, ADDRESS, SIZE));
        this.maskByteSize = bind(symbols, "llg_matcher_get_mask_byte_size", FunctionDescriptor.of(SIZE, ADDRESS));
        this.consumeToken = bind(symbols, "llg_matcher_consume_token", FunctionDescriptor.of(U32, ADDRESS, U32));
        this.isAccepting = bind(symbols, "llg_matcher_is_accepting", FunctionDescriptor.of(BOOL, ADDRESS));
        this.cloneMatcher = bind(symbols, "llg_clone_matcher", FunctionDescriptor.of(ADDRESS, ADDRESS));
        this.freeMatcher = bind(symbols, "llg_free_matcher", FunctionDescriptor.ofVoid(ADDRESS));
    }

    /// Loads the library at `library` once per path.
    public static Llguidance load(Path library) {
        Objects.requireNonNull(library, "library");
        Path resolved = library.toAbsolutePath().normalize();
        if (!Files.isRegularFile(resolved))
            throw new IllegalStateException("the constrained-decoding library is missing: " + resolved);
        return LOADED.computeIfAbsent(resolved, Llguidance::new);
    }

    /// Where the library sits in an installation: beside the CUDA library.
    public static Path besideLibrary(Path cudaLibrary) {
        return cudaLibrary.toAbsolutePath().resolveSibling(System.mapLibraryName("llguidance"));
    }

    private static MethodHandle bind(SymbolLookup symbols, String name, FunctionDescriptor descriptor) {
        MemorySegment symbol =
                symbols.find(name).orElseThrow(() -> new IllegalStateException("the llguidance library lacks " + name));
        return LINKER.downcallHandle(symbol, descriptor);
    }

    /// Reads a NUL-terminated UTF-8 string the library owns; null for a null pointer.
    static String string(MemorySegment pointer) {
        if (pointer.equals(MemorySegment.NULL)) return null;
        return pointer.reinterpret(Long.MAX_VALUE).getString(0);
    }
}
