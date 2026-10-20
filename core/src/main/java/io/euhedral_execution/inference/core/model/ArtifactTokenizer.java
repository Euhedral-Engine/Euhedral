package io.euhedral_execution.inference.core.model;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactReader;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// Adding the tokenizer section ([TokenizerSection]) to an artifact of either container version.
public final class ArtifactTokenizer {

    private ArtifactTokenizer() {}

    /// Writes `files` as the tokenizer of the artifact at `artifact`, in place: the artifact's content (header,
    /// tables and payloads) stays byte for byte, and any tokenizer it carried, or anything else after its content,
    /// is replaced.
    public static void add(Path artifact, TokenizerFiles files) throws IOException {
        long end = contentEnd(artifact);
        try (FileChannel channel = FileChannel.open(artifact, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            TokenizerSection.write(channel, end, files);
        }
    }

    /// Where the artifact's content ends, from its format alone, so a damaged tokenizer section does not hide it: for
    /// version 3 the size its header declares; for version 2 the end of its last payload (or of its tables when it
    /// has none).
    public static long contentEnd(Path artifact) throws IOException {
        return switch (ModelArchitecture.detect(artifact)) {
            case QWEN4_EXP ->
                io.euhedral_execution.inference.core.model.qwen4.loader.ArtifactReader.declaredSize(artifact);
            case QWEN38_DENSE -> {
                var dense = ArtifactReader.readIgnoringTokenizer(artifact);
                long end = dense.header().tensorDataOffset();
                for (TensorDescriptor tensor : dense.tensors())
                    end = Math.max(end, tensor.dataOffset() + tensor.byteSize());
                yield end;
            }
        };
    }
}
