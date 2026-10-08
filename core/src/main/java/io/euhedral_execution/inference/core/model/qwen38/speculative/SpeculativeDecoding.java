package io.euhedral_execution.inference.core.model.qwen38.speculative;

import io.euhedral_execution.core.frames.AbstractFrame;
import io.euhedral_execution.inference.core.generation.GenerationTimingListener;
import io.euhedral_execution.inference.core.generation.StepPort;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.qwen38.Execution;
import io.euhedral_execution.inference.core.model.qwen38.ExecutionPlan;
import io.euhedral_execution.inference.core.model.qwen38.Sequence;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.IntConsumer;
import java.util.function.IntPredicate;

/// A speculative decoding strategy for one sequence: it prefills a prompt and generates from it, drafting tokens
/// and committing only what the target's exact verification accepts, so every output token is the one ordinary
/// greedy decode produces. A session holds one, chosen from the loaded artifact when the session is created.
public interface SpeculativeDecoding extends AutoCloseable {

    /// Opens the strategy for one sequence. `runtime`, `plan`, `gpu` and `sequence` are borrowed.
    @FunctionalInterface
    interface Factory {
        SpeculativeDecoding open(
                Execution runtime,
                ExecutionPlan plan,
                ExecutionGpu gpu,
                Sequence sequence,
                IntPredicate endOfGeneration,
                int prefillChunk);
    }

    /// What the prefix cache needs from a speculative prompt: a call after each prefill chunk it [#wants] a
    /// checkpoint at, once the strategy's own state covers it and before the next chunk changes it. Once the cache
    /// is finished with the sequence's state it tells `failed` what stopped it (null: nothing) and then throws
    /// `next`. Chunks between those boundaries run ahead of the strategy's own steps.
    interface PrefixHooks {
        /// Whether the cache checkpoints the prompt after the chunk ending at `end`.
        boolean wants(int end);

        void afterChunk(int end, Consumer<Throwable> failed, AbstractFrame next);
    }

    /// The first step of a speculative generation: prefills `prompt` from `startPosition` (0 for a fresh
    /// sequence, or the position a prefix-cache restore left the sequence at, with this strategy's state restored
    /// too) and generates up to `maxNewTokens`, reporting each committed token to `onToken`; its last step hands
    /// every token to `ended` and names no next port. The steps run as generation frames: `onToken`, `timing` and
    /// `ended` are called on the workers that run them, one at a time, in order.
    StepPort start(
            int[] prompt,
            int maxNewTokens,
            IntConsumer onToken,
            GenerationTimingListener timing,
            PrefixHooks hooks,
            int startPosition,
            Consumer<List<Integer>> ended);

    /// The strategy's state as prefix checkpoints store it.
    SpeculativeCheckpoint checkpoint();

    @Override
    void close();
}
