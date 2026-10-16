package io.euhedral_execution.inference.core.model.qwen38.artifact;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

/// A compact artifact: its header, configuration and tensor table. `sources` names the file an object's payload is
/// read from when it is not the artifact file itself (key: the object's name): an artifact assembled from several
/// files keeps each descriptor's offset relative to the file it names. Every payload read resolves its file with
/// [#pathOf].
public record Artifact(
        ArtifactHeader header, Qwen38Config config, TensorDescriptor[] tensors, Map<String, Path> sources) {

    public Artifact {
        sources = Map.copyOf(Objects.requireNonNull(sources, "sources"));
    }

    /// An artifact whose objects are all in the artifact file.
    public Artifact(ArtifactHeader header, Qwen38Config config, TensorDescriptor[] tensors) {
        this(header, config, tensors, Map.of());
    }

    /// The file `descriptor`'s payload is read from: its source when it has one, else `artifactPath`.
    public Path pathOf(Path artifactPath, TensorDescriptor descriptor) {
        return this.sources.getOrDefault(descriptor.name(), artifactPath);
    }

    /// This artifact with `sources` as its per-object files (replacing any it had).
    public Artifact withSources(Map<String, Path> sources) {
        return new Artifact(this.header, this.config, this.tensors, sources);
    }
}
