package io.euhedral_execution.inference.core.testing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.model.qwen38.loader.HostWeightSelection;
import java.nio.file.Files;
import java.nio.file.Path;

/// The Qwen3.8 artifacts loaded with their MTP drafter and, as the speculative tests run them, with part of the base
/// weights in pinned host memory. That is a different model object from [SharedQwen38]'s, so it has scopes of its own
/// ("mtp:q3", "mtp:nvfp4"): asking for it closes the plain model and the other way round, and the classes that use it
/// run next to the group of the artifact (they carry its tag) so that the switch happens once.
public final class SharedQwen38Mtp {
    private SharedQwen38Mtp() {}

    /// A model with its drafter. Plan it with `new ExecutionPlan(model.weights(), model.staging())`.
    public record Loaded(CudaGpuMemory gpu, Qwen38Model model, Path path, Artifact artifact) {}

    /// The compact Q3 artifact (`euhedral.speculative.artifact`, else `euhedral.qwen.artifact`).
    public static Loaded q3(long hostMiB) {
        return load(
                "mtp:q3",
                property("euhedral.speculative.artifact", "euhedral.qwen.artifact", SharedQwen38.Q3_DEFAULT),
                hostMiB);
    }

    /// The NVFP4 artifact (`euhedral.speculative.artifact`, else `euhedral.qwen.nvfp4-artifact`).
    public static Loaded nvfp4(long hostMiB) {
        return load(
                "mtp:nvfp4",
                property("euhedral.speculative.artifact", "euhedral.qwen.nvfp4-artifact", SharedQwen38.NVFP4_DEFAULT),
                hostMiB);
    }

    private static String property(String first, String second, String fallback) {
        return System.getProperty(first, System.getProperty(second, fallback));
    }

    private static Loaded load(String scope, String path, long hostMiB) {
        Path artifactPath = Path.of(path);
        assertTrue(Files.isRegularFile(artifactPath), "the Qwen artifact is missing: " + artifactPath);
        CudaGpuMemory gpu = SharedQwen38.gpu();
        return Shared.get(scope, "qwen38-mtp:" + artifactPath + ":" + hostMiB, () -> {
                    Artifact artifact = ArtifactReader.read(artifactPath);
                    return new Holder(new Loaded(
                            gpu,
                            Qwen38Model.load(
                                    artifactPath,
                                    artifact,
                                    gpu,
                                    ArtifactProfile.Speculation.MTP,
                                    HostWeightSelection.select(artifact, hostMiB << 20)),
                            artifactPath,
                            artifact));
                })
                .loaded();
    }

    private record Holder(Loaded loaded) implements AutoCloseable {
        @Override
        public void close() {
            this.loaded.model().close();
        }
    }
}
