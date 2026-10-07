package io.euhedral_execution.inference.core.model.qwen38.speculative;

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

    /// What the prefix cache needs from a speculative prompt: a call after each prefill chunk, once the
    /// strategy's own state covers it and before the next chunk changes it. `done` runs once the cache is
    /// finished with the sequence's state, with the failure that stopped it or null, on any thread; it must not
    /// block.
    @FunctionalInterface
    interface PrefixHooks {
        void afterChunk(int end, Consumer<Throwable> done);
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
