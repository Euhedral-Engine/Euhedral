package io.euhedral_execution.inference.core.model;

import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.ModelDescription;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import java.util.function.Supplier;

/// A loaded model as the engine sees it: sessions, host work on the lattice, and what the engine reports. The
/// engine chooses one at load (by the artifact's header) and delegates to it; nothing else names a model.
public interface ModelRuntime {

    /// A session over this model. `release` runs when the session closes.
    GenerationSession createSession(
            GenerationConfig config, SessionOptions options, Consumer<GenerationSession> release);

    /// Encodes `prompt` as a new session encodes its first prompt, on the lattice's workers.
    CompletableFuture<int[]> tokenizePrompt(String prompt);

    /// Runs `work` as one frame on the lattice's workers.
    <T> CompletableFuture<T> onWorker(Supplier<T> work);

    int vocabularySize();

    int maxPositionEmbeddings();

    /// Prompt tokens per prefill quantum.
    int prefillChunkTokens();

    ModelDescription description();

    InferenceRunSnapshot.Model identity();

    ExecutionGpu gpu();

    /// Bytes of weights kept in pinned host memory instead of on the device.
    long hostBackedWeightBytes();

    /// Device bytes the execution graphs keep between quanta (the GPU's scratch excluded).
    long retainedWorkspaceBytes();

    /// The prefix cache's counters, or null without one.
    PrefixCacheStats prefixCacheStats();

    /// Stops admission and waits for every accepted quantum and host task, then detaches from the lattice. The
    /// owner has closed every session first.
    void stop();

    /// Releases the memory the runtime owns. Runs after [#stop] and after the lattice stopped; the GPU an owner
    /// opened stays the owner's to close.
    void close();
}
