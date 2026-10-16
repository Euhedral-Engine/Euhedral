package io.euhedral_execution.inference.core.testing;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/// Composite artifacts built in memory from a trellis artifact and an NVFP4 one, as an assembled artifact would be:
/// descriptors with the source file each object is read from.
///
/// The files are `euhedral.test.trellis-artifact` (else `euhedral.qwen.trellis30-artifact`) and
/// `euhedral.test.nvfp4-artifact` (else `euhedral.qwen.nvfp4-artifact`); a test that needs an unset one skips.
public final class MixedArtifacts {

    private MixedArtifacts() {}

    /// The trellis artifact's file, or skips the test.
    public static Path trellis() {
        return file("euhedral.test.trellis-artifact", "euhedral.qwen.trellis30-artifact");
    }

    /// The NVFP4 artifact's file, or skips the test.
    public static Path nvfp4() {
        return file("euhedral.test.nvfp4-artifact", "euhedral.qwen.nvfp4-artifact");
    }

    private static Path file(String property, String fallback) {
        String value = System.getProperty(property, System.getProperty(fallback));
        assumeTrue(value != null && Files.isRegularFile(Path.of(value)), "no artifact for " + property);
        return Path.of(value);
    }

    /// The trellis artifact with every object's source named explicitly, the trellis file itself.
    public static Artifact identity(Path trellis) throws IOException {
        Artifact artifact = ArtifactReader.read(trellis);
        Map<String, Path> sources = new HashMap<>();
        for (TensorDescriptor tensor : artifact.tensors()) sources.put(tensor.name(), trellis);
        return artifact.withSources(sources);
    }

    /// Whether `name` is an FFN projection of a text layer or of the MTP layer.
    public static boolean ffn(String name) {
        boolean layer = name.startsWith("text/layers/") || name.startsWith("mtp/layer/");
        return layer && (name.endsWith("/mlp/gate_up") || name.endsWith("/mlp/down"));
    }

    /// The trellis artifact with every FFN projection (and its descriptor) taken from the NVFP4 artifact, which
    /// stores `draftDepth` as the MTP draft depth. The trellis companions of the replaced objects are dropped.
    public static Artifact mixed(Path trellis, Path nvfp4, int draftDepth) throws IOException {
        Artifact base = ArtifactReader.read(trellis);
        Artifact other = ArtifactReader.read(nvfp4);
        Map<String, TensorDescriptor> replacements = new HashMap<>();
        for (TensorDescriptor tensor : other.tensors()) if (ffn(tensor.name())) replacements.put(tensor.name(), tensor);
        Map<String, Path> sources = new HashMap<>();
        List<TensorDescriptor> tensors = new ArrayList<>();
        for (TensorDescriptor tensor : base.tensors()) {
            String name = tensor.name();
            if (ffn(name)) {
                tensors.add(replacements.get(name));
                sources.put(name, nvfp4);
            } else if (!companionOfFfn(name)) {
                tensors.add(tensor);
            }
        }
        Artifact composite = new Artifact(
                base.header(), base.config().withDraftDepth(draftDepth), tensors.toArray(TensorDescriptor[]::new));
        return composite.withSources(sources);
    }

    private static boolean companionOfFfn(String name) {
        int dot = name.lastIndexOf('.');
        return dot > 0 && ffn(name.substring(0, dot));
    }
}
