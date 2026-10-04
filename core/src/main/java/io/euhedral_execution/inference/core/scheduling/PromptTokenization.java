package io.euhedral_execution.inference.core.scheduling;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.scheduling.graph.FrameSeeds;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;

/// One prompt's tokenization as fine-grained lattice frames. A first frame splits the prompt into pre-tokens
/// (control tokens, NFC, the split expression: about 1% of the work), publishes one frame per further chunk
/// of [#CHUNK_PRETOKENS] pre-tokens and encodes the first chunk itself; each chunk's frame runs BPE over its
/// pieces, and the last to finish joins the chunks in order. The result equals
/// [QwenTokenizer#encodeText] (or
/// [QwenTokenizer#encodeWithModelSpecialTokens]), whose encoding is the same pre-tokenization and the same
/// per-piece BPE.
final class PromptTokenization {

    /// Pre-tokens per leaf frame: about 40 us of BPE. Measured in the model, 1 to 4 per frame tokenize a
    /// 16K-token prompt in 23-35 ms; 64 per frame took 140-250 ms.
    static final int CHUNK_PRETOKENS = 2;

    private final QwenTokenizer tokenizer;
    private final String text;
    private final boolean modelSpecialTokens;
    private final Consumer<AbstractFrame> publisher;
    private final Runnable terminated;
    private final CompletableFuture<int[]> result = new CompletableFuture<>();
    /// Routing seeds of the job's frames, consecutive from one base (FrameSeeds); frames are built on many workers.
    private final long seedBase = new FrameSeeds().next();
    private final AtomicLong nextSeed = new AtomicLong();
    private final AtomicBoolean finished = new AtomicBoolean();
    private QwenTokenizer.Pretokens pretokens;
    private int[][] chunks;
    private AtomicInteger remaining;

    /// `publisher` makes a frame available to the lattice; `terminated` runs once, after the result completed.
    private PromptTokenization(
            QwenTokenizer tokenizer,
            String text,
            boolean modelSpecialTokens,
            Consumer<AbstractFrame> publisher,
            Runnable terminated) {
        this.tokenizer = Objects.requireNonNull(tokenizer, "tokenizer");
        this.text = Objects.requireNonNull(text, "text");
        this.modelSpecialTokens = modelSpecialTokens;
        this.publisher = Objects.requireNonNull(publisher, "publisher");
        this.terminated = Objects.requireNonNull(terminated, "terminated");
    }

    /// Publishes the tokenization of `text` and returns its token IDs' future.
    static CompletableFuture<int[]> start(
            QwenTokenizer tokenizer,
            String text,
            boolean modelSpecialTokens,
            Consumer<AbstractFrame> publisher,
            Runnable terminated) {
        PromptTokenization job = new PromptTokenization(tokenizer, text, modelSpecialTokens, publisher, terminated);
        try {
            publisher.accept(job.new Split());
        } catch (RuntimeException | Error failure) {
            job.fail(failure);
        }
        return job.result.copy();
    }

    private void split() {
        this.pretokens = this.tokenizer.pretokenize(this.text);
        int count = Math.max(1, (this.pretokens.size() + CHUNK_PRETOKENS - 1) / CHUNK_PRETOKENS);
        this.chunks = new int[count][];
        this.remaining = new AtomicInteger(count);
        encode(0, count);
    }

    /// Encodes chunk `first`, after publishing a frame for each further chunk before `end`.
    private void encode(int first, int end) {
        for (int chunk = end - 1; chunk > first; chunk--) this.publisher.accept(new Encode(chunk, chunk + 1));
        int from = first * CHUNK_PRETOKENS;
        int to = Math.min(this.pretokens.size(), from + CHUNK_PRETOKENS);
        this.chunks[first] = this.tokenizer.encode(this.pretokens, from, to);
        // The decrement orders every chunk's write before the joining frame's reads.
        if (this.remaining.decrementAndGet() == 0) join();
    }

    private void join() {
        int length = 0;
        for (int[] chunk : this.chunks) length += chunk.length;
        int[] ids = new int[length];
        int offset = 0;
        for (int[] chunk : this.chunks) {
            System.arraycopy(chunk, 0, ids, offset, chunk.length);
            offset += chunk.length;
        }
        finish(this.modelSpecialTokens ? this.tokenizer.withModelSpecialTokens(ids) : ids, null);
    }

    private long nextSeed() {
        return this.seedBase + this.nextSeed.getAndIncrement();
    }

    private void fail(Throwable failure) {
        finish(null, failure);
    }

    private void finish(int[] ids, Throwable failure) {
        if (!this.finished.compareAndSet(false, true)) return;
        try {
            this.terminated.run();
        } finally {
            if (failure != null) this.result.completeExceptionally(failure);
            else this.result.complete(ids);
        }
    }

    /// Splits the prompt and fans its chunks out.
    private final class Split extends AbstractFrame {
        Split() {
            super(FrameSeeds.ID_HASH);
            randomizeHash(nextSeed());
        }

        @Override
        public void execute() {
            try {
                split();
            } catch (RuntimeException | Error failure) {
                fail(failure);
            }
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            fail(new IllegalStateException("the lattice rejected a tokenization frame", rejection));
        }
    }

    /// Encodes one chunk of pre-tokens.
    private final class Encode extends AbstractFrame {
        private final int first;
        private final int end;

        Encode(int first, int end) {
            super(FrameSeeds.ID_HASH);
            this.first = first;
            this.end = end;
            randomizeHash(nextSeed());
        }

        @Override
        public void execute() {
            try {
                encode(this.first, this.end);
            } catch (RuntimeException | Error failure) {
                fail(failure);
            }
        }

        @Override
        public void doFinally() {}

        @Override
        public void doFinallyWithError(Throwable rejection) {
            fail(new IllegalStateException("the lattice rejected a tokenization frame", rejection));
        }
    }
}
