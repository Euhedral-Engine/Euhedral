package io.euhedral_execution.inference.core.model;

import io.euhedral_execution.inference.core.artifact.ArtifactFileAccess;
import io.euhedral_execution.inference.core.artifact.ArtifactFormatException;
import io.euhedral_execution.inference.core.model.qwen38.artifact.ArtifactHeader;
import io.euhedral_execution.inference.core.model.qwen4.loader.Header;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/// The model family an artifact holds. The two are separate models with separate loaders, planners and storage
/// classes: a dense-model artifact is container version 2, a `qwen4_exp` artifact version 3 with the architecture in
/// its header.
public enum ModelArchitecture {
    /// Qwen3.8-27B: a dense hybrid of GDN and full-attention layers (EDRL version 2).
    QWEN38_DENSE,
    /// `qwen4_exp`, Qwen3.8-Flash-Next: a mixture of experts with hyper-connections, sparse attention and n-gram
    /// embeddings (EDRL version 3).
    QWEN4_EXP;

    /// Reads the architecture from the first bytes of the artifact.
    public static ModelArchitecture detect(Path path) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            ByteBuffer head = ByteBuffer.allocate(12).order(ByteOrder.BIG_ENDIAN);
            if (channel.size() < head.capacity())
                throw new ArtifactFormatException("artifact is truncated before the header");
            ArtifactFileAccess.readFully(channel, 0, head, "header");
            head.flip();
            int magic = head.getInt();
            int version = head.getInt();
            int architecture = head.getInt();
            if (magic != ArtifactHeader.MAGIC) throw new ArtifactFormatException("unsupported artifact magic");
            if (version == ArtifactHeader.COMPACT_VERSION) return QWEN38_DENSE;
            if (version == Header.VERSION) {
                if (architecture == Header.ARCHITECTURE_QWEN4_EXP) return QWEN4_EXP;
                throw new ArtifactFormatException("unsupported architecture " + architecture);
            }
            throw new ArtifactFormatException("unsupported artifact version " + version);
        }
    }
}
