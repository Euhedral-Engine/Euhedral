package io.euhedral_execution.inference.core.model.qwen38;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.InferenceConfig;
import io.euhedral_execution.inference.core.generation.DeviceLogits;
import io.euhedral_execution.inference.core.generation.LogitsRequirement;
import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import io.euhedral_execution.inference.core.runtime.PullingLattice;
import io.euhedral_execution.inference.core.state.KvFormat;
import io.euhedral_execution.inference.core.testing.ModelGroup;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

/// Measurement tool, not a regression test (skipped unless `euhedral.greedy.report` names an output file): greedy
/// generation of a chat prompt on a fresh sequence, written as the reference the Rust engine's greedy output is
/// compared with, byte for byte. The prompt is prefilled in production-sized chunks, the first token is selected by
/// the last prompt row, and each decode step feeds the token the step before selected (lowest id among equal maxima,
/// as the device selection does); a generation terminator is recorded and not fed back.
///
/// Report, text: a `prompt` line (the prompt's token ids, space separated), a `generated` line (the ids selected, up to
/// `euhedral.greedy.tokens`, ending with the terminator when one was selected) and a `stop` line (`eos` or `length`).
///
/// ```
/// ./gradlew :core:cudaOwnJvmTest --tests '*GreedyReferenceCudaIntegrationTest' --rerun \
///     -Peuhedral.greedy.report=$GOLDENS/greedy-rco.txt -Peuhedral.greedy.artifact=$RCO_ARTIFACT
/// ```
@ModelGroup.OwnJvm // loads the artifact named by euhedral.greedy.artifact, not one of the shared ones
class GreedyReferenceCudaIntegrationTest {

    /// The chat prompt: one user turn, then the assistant's turn open.
    static final String PROMPT = "<|im_start|>user\nExplain, in a few short paragraphs, how a language model "
            + "produces text one token at a time.<|im_end|>\n<|im_start|>assistant\n";

    @Test
    @Timeout(value = 3600, unit = TimeUnit.SECONDS)
    void recordGreedyGeneration() throws Throwable {
        String report = System.getProperty("euhedral.greedy.report", "");
        assumeTrue(!report.isBlank(), "measurement tool");
        Path library = Path.of(System.getProperty("euhedral.cuda.library"));
        Path artifact = Path.of(System.getProperty("euhedral.greedy.artifact"));
        long hostBytes = Long.getLong("euhedral.greedy.host-mib", 0L) << 20;
        int tokens = Integer.getInteger("euhedral.greedy.tokens", 128);
        QwenTokenizer tokenizer = QwenTokenizer.load(TokenizerFiles.fromCheckpoint(
                Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "unset-euhedral.qwen.tokenizer-dir"))));
        int[] prompt = tokenizer.encodeWithModelSpecialTokens(PROMPT);
        var data = ArtifactReader.read(artifact);
        try (CudaGpuMemory gpu = new CudaGpuMemory(library);
                Qwen38Model model = Qwen38Model.load(
                        artifact,
                        data,
                        gpu,
                        ArtifactProfile.Speculation.NONE,
                        HostWeightSelection.select(data, hostBytes));
                var lattice = new PullingLattice()) {
            var plan = new ExecutionPlan(model.weights(), model.staging());
            var runtime = new Execution(lattice, plan, gpu);
            try {
                var sequence = new Sequence(1);
                sequence.useKvFormat(KvFormat.valueOf(
                        System.getProperty("euhedral.greedy.kv-cache", "nvfp4").toUpperCase(Locale.ROOT)));
                List<Integer> generated = new ArrayList<>();
                boolean eos = false;
                try {
                    short[] logits = null;
                    for (int begin = 0; begin < prompt.length; begin += InferenceConfig.PREFILL_CHUNK_TOKENS) {
                        int end = Math.min(prompt.length, begin + InferenceConfig.PREFILL_CHUNK_TOKENS);
                        logits = run(
                                runtime,
                                gpu,
                                plan,
                                sequence,
                                Quantum.ExecutionKind.PREFILL,
                                Arrays.copyOfRange(prompt, begin, end));
                    }
                    while (true) {
                        int token = argmax(logits);
                        generated.add(token);
                        if (tokenizer.isGenerationEosToken(token)) {
                            eos = true;
                            break;
                        }
                        if (generated.size() >= tokens) break;
                        logits = run(runtime, gpu, plan, sequence, Quantum.ExecutionKind.DECODE, new int[] {token});
                    }
                } finally {
                    sequence.complete();
                }
                Files.writeString(
                        Path.of(report),
                        "prompt " + join(Arrays.stream(prompt).boxed().toList()) + "\n"
                                + "generated " + join(generated) + "\n"
                                + "stop " + (eos ? "eos" : "length") + "\n");
                System.out.printf(
                        Locale.ROOT,
                        "greedy %s: %d prompt tokens, %d generated, stop %s%n",
                        artifact.getFileName(),
                        prompt.length,
                        generated.size(),
                        eos ? "eos" : "length");
            } finally {
                runtime.close();
            }
        }
    }

    private static String join(List<Integer> ids) {
        return ids.stream().map(String::valueOf).collect(Collectors.joining(" "));
    }

    /// The lowest id among the largest logits (BF16 widened to float, as the device compares them).
    private static int argmax(short[] logits) {
        int best = 0;
        float bestValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < logits.length; i++) {
            float value = Float.intBitsToFloat(logits[i] << 16);
            if (value > bestValue) {
                bestValue = value;
                best = i;
            }
        }
        return best;
    }

    /// Runs one quantum and returns its last logits row.
    private static short[] run(
            Execution runtime,
            CudaGpuMemory gpu,
            ExecutionPlan plan,
            Sequence sequence,
            Quantum.ExecutionKind kind,
            int[] tokens)
            throws Exception {
        AtomicReference<short[]> captured = new AtomicReference<>();
        var context = new Quantum(
                plan, sequence, kind, sequence.currentTokenPosition(), tokens, LogitsRequirement.LAST_TOKEN);
        var outcome = runtime.submit(context, done -> {
                    try (Arena arena = Arena.ofConfined();
                            DeviceLogits logits = done.logitsOutput().orElseThrow()) {
                        int vocabulary = logits.vocabularySize();
                        int first = logits.tokenCount() - 1;
                        MemorySegment host = arena.allocate((long) vocabulary * Short.BYTES, Short.BYTES);
                        gpu.copyDeviceToHost(
                                host,
                                logits.deviceAddress() + (long) first * vocabulary * Short.BYTES,
                                host.byteSize());
                        short[] values = new short[vocabulary];
                        MemorySegment.copy(
                                host,
                                ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN),
                                0,
                                values,
                                0,
                                vocabulary);
                        captured.set(values);
                    }
                })
                .get(600, TimeUnit.SECONDS);
        if (outcome.status() != Quantum.Status.SUCCESS) throw new AssertionError(outcome.failure());
        return captured.get();
    }
}
