package io.euhedral_execution.inference.core.model.qwen38.artifact;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.model.qwen38.Qwen38Config;

public record Artifact(ArtifactHeader header, Qwen38Config config, TensorDescriptor[] tensors) {}
