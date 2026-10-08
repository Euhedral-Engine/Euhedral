package io.euhedral_execution.inference.core.testing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.euhedral_execution.inference.core.gpu.CudaGpuMemory;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Model;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.nio.file.Files;
import java.nio.file.Path;

/// The device handle and the Qwen3.8 artifacts that CUDA tests share. The model of a group is loaded by the first
/// class that asks and kept for the rest of the group ([Shared]).
public final class SharedQwen38 {
    static final String Q3_DEFAULT = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_q3.edrl";
    static final String NVFP4_DEFAULT = "/mnt/shared/qwen38-quant/artifacts/qwen3_8_27b_nvfp4.edrl";

    /// A model on the device, with the file and the parsed artifact it came from. Closing it is the registry's job.
    public record Loaded(CudaGpuMemory gpu, Qwen38Model model, Path path, Artifact artifact) {}

    private SharedQwen38() {}

    /// The process-wide device handle.
    public static CudaGpuMemory gpu() {
        return Shared.get(
                Shared.GLOBAL, "gpu", () -> new CudaGpuMemory(Path.of(System.getProperty("euhedral.cuda.library"))));
    }

    /// The compact Q3 artifact (`euhedral.qwen.artifact`).
    public static Loaded q3() {
        return load(ModelGroup.Q3, System.getProperty("euhedral.qwen.artifact", Q3_DEFAULT));
    }

    /// The NVFP4 artifact (`euhedral.qwen.nvfp4-artifact`).
    public static Loaded nvfp4() {
        return load(ModelGroup.NVFP4, System.getProperty("euhedral.qwen.nvfp4-artifact", NVFP4_DEFAULT));
    }

    /// The model of `path`, loaded once for the group `scope`.
    public static Loaded load(String scope, String path) {
        Path artifactPath = Path.of(path);
        assertTrue(Files.isRegularFile(artifactPath), "the Qwen artifact is missing: " + artifactPath);
        CudaGpuMemory gpu = gpu();
        return Shared.get(scope, "qwen38:" + artifactPath, () -> {
                    Artifact artifact = ArtifactReader.read(artifactPath);
                    return new Holder(
                            new Loaded(gpu, Qwen38Model.load(artifactPath, artifact, gpu), artifactPath, artifact));
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
