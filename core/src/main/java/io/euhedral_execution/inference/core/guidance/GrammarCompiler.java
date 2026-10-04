package io.euhedral_execution.inference.core.guidance;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.IntFunction;

/// Compiles llguidance Lark grammars against one vocabulary and hands out a fresh [GrammarConstraint] per
/// generation.
///
/// A compiled grammar is kept, least recently used first out, so a client that repeats its tools or response
/// schema every turn compiles them once; each generation then starts from a clone of it. Compiling runs on the
/// caller's thread (a worker frame) and outside the cache's lock.
public final class GrammarCompiler implements AutoCloseable {
    /// Compiled grammars kept for reuse.
    static final int CACHED_GRAMMARS = 32;

    private final Llguidance library;
    private final int vocabularySize;
    private final MemorySegment tokenizer;
    private final int[] endIds;
    private final Map<String, MemorySegment> compiled = new LinkedHashMap<>(16, 0.75f, true);
    private boolean closed;

    /// `tokenBytes` returns each token's bytes: its text for ordinary tokens, `0xFF` and its name for tokens a
    /// grammar may only name explicitly (control tokens); every ID below `vocabularySize` needs bytes. A
    /// grammar ends with any of `endIds`.
    public GrammarCompiler(Llguidance library, int vocabularySize, IntFunction<byte[]> tokenBytes, int[] endIds) {
        this.library = Objects.requireNonNull(library, "library");
        if (vocabularySize <= 0) throw new IllegalArgumentException("vocabularySize must be positive");
        if (endIds.length == 0) throw new IllegalArgumentException("a grammar needs an end token");
        this.vocabularySize = vocabularySize;
        this.endIds = endIds.clone();
        try (Arena arena = Arena.ofConfined()) {
            byte[][] tokens = new byte[vocabularySize][];
            long total = 0;
            for (int id = 0; id < vocabularySize; id++) {
                tokens[id] = Objects.requireNonNull(tokenBytes.apply(id), "token bytes");
                if (tokens[id].length == 0) throw new IllegalArgumentException("token " + id + " has no bytes");
                total += tokens[id].length;
            }
            MemorySegment lengths = arena.allocate(ValueLayout.JAVA_INT, vocabularySize);
            MemorySegment bytes = arena.allocate(total);
            long offset = 0;
            for (int id = 0; id < vocabularySize; id++) {
                lengths.setAtIndex(ValueLayout.JAVA_INT, id, tokens[id].length);
                MemorySegment.copy(tokens[id], 0, bytes, ValueLayout.JAVA_BYTE, offset, tokens[id].length);
                offset += tokens[id].length;
            }
            MemorySegment extra = arena.allocate(ValueLayout.JAVA_INT, Math.max(1, endIds.length - 1));
            for (int index = 1; index < endIds.length; index++)
                extra.setAtIndex(ValueLayout.JAVA_INT, index - 1, endIds[index]);
            MemorySegment init = arena.allocate(Llguidance.TOKENIZER_INIT_SIZE, 8);
            init.set(ValueLayout.JAVA_LONG, 0, Llguidance.TOKENIZER_INIT_SIZE);
            init.set(ValueLayout.JAVA_INT, Llguidance.TOKENIZER_INIT_VOCAB_SIZE, vocabularySize);
            init.set(ValueLayout.JAVA_INT, Llguidance.TOKENIZER_INIT_EOS, endIds[0]);
            init.set(ValueLayout.ADDRESS, Llguidance.TOKENIZER_INIT_TOKEN_LENS, lengths);
            init.set(ValueLayout.ADDRESS, Llguidance.TOKENIZER_INIT_TOKEN_BYTES, bytes);
            // Masks come from the token trie; the tokenize callback only serves features this server does not use
            // (forced tokens, prompt healing), so llguidance's own greedy tokenization stands in for it.
            init.set(ValueLayout.JAVA_BOOLEAN, Llguidance.TOKENIZER_INIT_APPROXIMATE_TOKENIZE, true);
            if (endIds.length > 1) {
                init.set(ValueLayout.ADDRESS, Llguidance.TOKENIZER_INIT_EOS_EXTRA, extra);
                init.set(ValueLayout.JAVA_INT, Llguidance.TOKENIZER_INIT_EOS_EXTRA_COUNT, endIds.length - 1);
            }
            MemorySegment error = arena.allocate(1024);
            MemorySegment created = (MemorySegment) library.newTokenizer.invokeExact(init, error, 1024L);
            if (created.equals(MemorySegment.NULL))
                throw new IllegalStateException("llguidance refused the vocabulary: " + error.getString(0));
            this.tokenizer = created;
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance tokenizer creation failed", failure);
        }
    }

    /// A compiler for the logits rows of a model with `vocabularySize` entries over `tokenizer`'s tokens; any
    /// generation terminator ends a grammar.
    public static GrammarCompiler forTokenizer(
            Llguidance library,
            io.euhedral_execution.inference.core.tokenizer.QwenTokenizer tokenizer,
            int vocabularySize) {
        int[] ends = java.util.stream.IntStream.concat(
                        java.util.stream.IntStream.of(tokenizer.eosTokenId()),
                        tokenizer.generationEosTokenIds().stream()
                                .mapToInt(Integer::intValue)
                                .sorted())
                .distinct()
                .toArray();
        return new GrammarCompiler(library, vocabularySize, tokenizer::grammarTokenBytes, ends);
    }

    public int vocabularySize() {
        return this.vocabularySize;
    }

    /// Compiles `lark` (reusing a cached compilation) and throws [GrammarException] when llguidance refuses it.
    public void check(String lark) {
        free(cloneOf(lark, "lark"));
    }

    /// Compiles a JSON Schema on its own, so that a refusal names the schema's problem rather than a grammar
    /// around it.
    public void checkJsonSchema(String schema) {
        free(cloneOf(schema, "json_schema"));
    }

    /// A constraint at the start of `lark`, owned by the caller, who must close it.
    public GrammarConstraint constraint(String lark) {
        return new GrammarConstraint(this.library, cloneOf(lark, "lark"), this.vocabularySize, this.endIds);
    }

    private MemorySegment cloneOf(String grammar, String type) {
        Objects.requireNonNull(grammar, "grammar");
        String key = type + "\u0000" + grammar;
        synchronized (this) {
            if (this.closed) throw new IllegalStateException("the grammar compiler is closed");
            MemorySegment template = this.compiled.get(key);
            if (template != null) return clone(template);
        }
        MemorySegment template = compile(grammar, type);
        synchronized (this) {
            if (this.closed) {
                free(template);
                throw new IllegalStateException("the grammar compiler is closed");
            }
            MemorySegment existing = this.compiled.putIfAbsent(key, template);
            if (existing != null) {
                free(template);
                template = existing;
            } else if (this.compiled.size() > CACHED_GRAMMARS) {
                var eldest = this.compiled.entrySet().iterator();
                free(eldest.next().getValue());
                eldest.remove();
            }
            return clone(template);
        }
    }

    private MemorySegment compile(String grammar, String grammarType) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment init = arena.allocate(Llguidance.CONSTRAINT_INIT_SIZE, 8);
            this.library.constraintInitDefaults.invokeExact(init, this.tokenizer);
            // Nothing is written to stderr or kept in a log buffer; errors are read from the matcher.
            init.set(ValueLayout.JAVA_INT, Llguidance.CONSTRAINT_INIT_LOG_BUFFER, 0);
            init.set(ValueLayout.JAVA_INT, Llguidance.CONSTRAINT_INIT_LOG_STDERR, 0);
            MemorySegment type = arena.allocateFrom(grammarType, StandardCharsets.UTF_8);
            MemorySegment data = arena.allocateFrom(grammar, StandardCharsets.UTF_8);
            MemorySegment matcher = (MemorySegment) this.library.newMatcher.invokeExact(init, type, data);
            if ((boolean) this.library.matcherIsError.invokeExact(matcher)) {
                String message = Llguidance.string((MemorySegment) this.library.matcherError.invokeExact(matcher));
                this.library.freeMatcher.invokeExact(matcher);
                throw new GrammarException(
                        message == null ? "the grammar is invalid" : GrammarException.summary(message));
            }
            return matcher;
        } catch (RuntimeException | Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance grammar compilation failed", failure);
        }
    }

    private MemorySegment clone(MemorySegment template) {
        try {
            return (MemorySegment) this.library.cloneMatcher.invokeExact(template);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance matcher clone failed", failure);
        }
    }

    private void free(MemorySegment matcher) {
        try {
            this.library.freeMatcher.invokeExact(matcher);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance matcher free failed", failure);
        }
    }

    /// Frees the cached grammars and the vocabulary; constraints already handed out stay valid until closed.
    @Override
    public synchronized void close() {
        if (this.closed) return;
        this.closed = true;
        for (MemorySegment template : this.compiled.values()) free(template);
        this.compiled.clear();
        try {
            this.library.freeTokenizer.invokeExact(this.tokenizer);
        } catch (Throwable failure) {
            throw new IllegalStateException("llguidance tokenizer free failed", failure);
        }
    }
}
