package io.euhedral_execution.inference.core.model.qwen38;

import io.euhedral_execution.core.generics.LatticeTerminal;
import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.InferenceRunSnapshot;
import io.euhedral_execution.inference.core.ModelDescription;
import io.euhedral_execution.inference.core.generation.GenerationSession;
import io.euhedral_execution.inference.core.generation.SessionOptions;
import io.euhedral_execution.inference.core.gpu.ExecutionGpu;
import io.euhedral_execution.inference.core.model.ModelRuntime;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.prefix.PrefixCache;
import io.euhedral_execution.inference.core.model.qwen38.speculative.DFlash2Decoder;
import io.euhedral_execution.inference.core.prefix.PrefixCacheStats;
import io.euhedral_execution.inference.core.sampling.GenerationConfig;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Consumer;
import java.util.function.Supplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/// The dense Qwen3.8 model as a [ModelRuntime]: its loaded weights, the plan that runs them, the execution that
/// admits its quanta, and the optional prefix cache. The engine loads the weights and the GPU (its bootstrap owns
/// both) and hands them over.
public final class Qwen38Runtime implements ModelRuntime {

    private static final Logger LOG = LoggerFactory.getLogger(Qwen38Runtime.class);
    private static final AtomicLong SEQUENCE_IDS = new AtomicLong();

    private final QwenTokenizer tokenizer;
    private final ExecutionGpu gpu;
    private final Qwen38Model model;
    private final ExecutionPlan plan;
    private final Execution execution;
    private final PrefixCache prefixCache;
    private final ArtifactProfile profile;
    private final InferenceRunSnapshot.Model identity;

    private Qwen38Runtime(
            QwenTokenizer tokenizer,
            ExecutionGpu gpu,
            Qwen38Model model,
            ExecutionPlan plan,
            Execution execution,
            PrefixCache prefixCache,
            ArtifactProfile profile,
            InferenceRunSnapshot.Model identity) {
        this.tokenizer = tokenizer;
        this.gpu = gpu;
        this.model = model;
        this.plan = plan;
        this.execution = execution;
        this.prefixCache = prefixCache;
        this.profile = profile;
        this.identity = identity;
    }

    /// Runs `model` on `lattice`, which must be started and outlive the runtime. `artifact` and `profile` are null
    /// when the engine was started without a profiled artifact (tests).
    public static Qwen38Runtime open(
            InferenceConfig config,
            QwenTokenizer tokenizer,
            ExecutionGpu gpu,
            Qwen38Model model,
            Artifact artifact,
            ArtifactProfile profile,
            LatticeTerminal lattice) {
        ExecutionPlan plan = new ExecutionPlan(model.weights(), model.staging());
        Execution execution = new Execution(lattice, plan, gpu);
        var identity = identity(config.artifactPath(), artifact, model);
        // Last, so nothing that can fail follows the pinned arena.
        PrefixCache prefixCache = openPrefixCache(config, gpu, plan);
        return new Qwen38Runtime(
                Objects.requireNonNull(tokenizer, "tokenizer"),
                gpu,
                model,
                plan,
                execution,
                prefixCache,
                profile,
                identity);
    }

    /// The cache is an optimisation: when it is off, the plan is not a full model, or its host memory cannot be
    /// pinned, the model serves without it.
    private static PrefixCache openPrefixCache(InferenceConfig config, ExecutionGpu gpu, ExecutionPlan plan) {
        if (config.prefixCacheBytes() == 0 || plan.weights().layers().length <= 1) return null;
        try {
            PrefixCache cache = PrefixCache.create(
                    gpu, plan.weights().config(), config.prefixCacheBytes(), config.prefixCacheCheckpointTokens());
            LOG.info(
                    "Prefix cache: {} MiB pinned, a checkpoint every {} tokens",
                    config.prefixCacheBytes() >> 20,
                    config.prefixCacheCheckpointTokens());
            return cache;
        } catch (RuntimeException | Error unavailable) {
            LOG.warn(
                    "Prefix cache disabled: {} bytes of host memory could not be pinned",
                    config.prefixCacheBytes(),
                    unavailable);
            return null;
        }
    }

    private static InferenceRunSnapshot.Model identity(Path path, Artifact artifact, Qwen38Model model) {
        Long bytes;
        try {
            bytes = Files.size(path);
        } catch (IOException | SecurityException unreadable) {
            bytes = null;
        }
        return new InferenceRunSnapshot.Model(
                path.toString(),
                bytes,
                artifact == null ? null : artifact.header().version(),
                dimensions(model.weights().config()));
    }

    /// The snapshot's scalar projection of `config`; layer kinds are counted rather than copying its array.
    public static InferenceRunSnapshot.Dimensions dimensions(Qwen38Config config) {
        int attention = 0;
        int gdn = 0;
        LayerType[] layers = config.layerTypes();
        if (layers != null) {
            for (LayerType layer : layers) {
                if (layer == LayerType.FULL_ATTENTION) attention++;
                else if (layer == LayerType.GATED_DELTA_NET) gdn++;
            }
        }
        return new InferenceRunSnapshot.Dimensions(
                config.vocabSize(),
                config.hiddenSize(),
                config.numHiddenLayers(),
                attention,
                gdn,
                config.numAttentionHeads(),
                config.numKeyValueHeads(),
                config.attentionHeadDim(),
                config.intermediateSize(),
                config.linearNumKeyHeads(),
                config.linearNumValueHeads(),
                config.linearKeyHeadDim(),
                config.linearValueHeadDim(),
                config.maxPositionEmbeddings());
    }

    @Override
    public GenerationSession createSession(
            GenerationConfig config, SessionOptions options, Consumer<GenerationSession> release) {
        return createSession(config, Session.DEFAULT_PREFILL_CHUNK_TOKENS, options, release);
    }

    /// As [#createSession(GenerationConfig, SessionOptions, Consumer)] with a smaller prefill chunk, so tests can
    /// exercise several prefill quanta on short prompts.
    public Session createSession(
            GenerationConfig config,
            int prefillChunkTokens,
            SessionOptions options,
            Consumer<GenerationSession> release) {
        Objects.requireNonNull(release, "release");
        if (prefillChunkTokens > this.execution.maxRows())
            throw new IllegalArgumentException("a prefill chunk of " + prefillChunkTokens
                    + " tokens exceeds the workspace, sized at load for " + this.execution.maxRows() + " rows");
        var session = new Session(
                this.tokenizer,
                this.plan,
                this.execution,
                this.gpu,
                SEQUENCE_IDS.getAndIncrement(),
                Objects.requireNonNull(config, "config"),
                prefillChunkTokens,
                release::accept);
        switch (speculation(options, this.profile, this.plan.drafts(), this.plan.draftsWithDFlash2())) {
            case MTP -> session.enableSpeculativeDecoding(this.profile.speculativeDepth());
            case DFLASH2 -> session.useSpeculativeDecoding(DFlash2Decoder.factory(this.profile.speculativeDepth()));
            case NONE -> {}
        }
        if (this.prefixCache != null) session.usePrefixCache(this.prefixCache);
        return session;
    }

    /// The speculative strategy a session runs: the profile's, when `options` allow speculation and the plan can draft
    /// it (`mtp`, `dflash2`); otherwise none.
    static ArtifactProfile.Speculation speculation(
            SessionOptions options, ArtifactProfile profile, boolean mtp, boolean dflash2) {
        if (!options.speculation() || profile == null) return ArtifactProfile.Speculation.NONE;
        return switch (profile.speculation()) {
            case MTP -> mtp ? ArtifactProfile.Speculation.MTP : ArtifactProfile.Speculation.NONE;
            case DFLASH2 -> dflash2 ? ArtifactProfile.Speculation.DFLASH2 : ArtifactProfile.Speculation.NONE;
            case NONE -> ArtifactProfile.Speculation.NONE;
        };
    }

    @Override
    public CompletableFuture<int[]> tokenizePrompt(String prompt) {
        return this.execution.tokenize(this.tokenizer, prompt, true);
    }

    @Override
    public <T> CompletableFuture<T> onWorker(Supplier<T> work) {
        return this.execution.onWorker(work);
    }

    @Override
    public int vocabularySize() {
        return this.model.weights().config().vocabSize();
    }

    @Override
    public int maxPositionEmbeddings() {
        return this.model.weights().config().maxPositionEmbeddings();
    }

    @Override
    public int prefillChunkTokens() {
        return Session.DEFAULT_PREFILL_CHUNK_TOKENS;
    }

    @Override
    public ModelDescription description() {
        if (this.profile == null) return ModelDescription.NONE;
        return new ModelDescription(
                this.profile.artifactName(),
                this.profile.speculation().name().toLowerCase(Locale.ROOT),
                this.profile.speculativeDepth());
    }

    @Override
    public InferenceRunSnapshot.Model identity() {
        return this.identity;
    }

    @Override
    public ExecutionGpu gpu() {
        return this.gpu;
    }

    @Override
    public long hostBackedWeightBytes() {
        long bytes = 0;
        for (var handle : this.model.weights().runtimeObjects().values())
            if (handle.hostBacked()) bytes += handle.byteSize();
        return bytes;
    }

    @Override
    public long retainedWorkspaceBytes() {
        return this.execution.retainedWorkspaceBytes();
    }

    @Override
    public PrefixCacheStats prefixCacheStats() {
        return this.prefixCache == null ? null : this.prefixCache.stats();
    }

    /// Closes the execution (every accepted quantum retires, the lake detaches), then the prefix cache.
    @Override
    public void stop() {
        this.execution.close();
        if (this.prefixCache != null) this.prefixCache.close();
    }

    /// Frees the model's device and host memory.
    @Override
    public void close() {
        this.model.close();
    }

    public ExecutionPlan plan() {
        return this.plan;
    }

    public ArtifactProfile profile() {
        return this.profile;
    }

    public Qwen38Config config() {
        return this.model.weights().config();
    }

    public Execution execution() {
        return this.execution;
    }
}
