package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.api.chat.QwenChatTemplate;
import io.euhedral_execution.inference.core.artifact.TokenizerSection;
import io.euhedral_execution.inference.core.model.ArtifactTokenizer;
import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Locale;

/// `package-tokenizer`: gives an artifact the tokenizer of a checkpoint directory ([TokenizerSection]). In place, the
/// artifact's content stays byte for byte and only the section after it is written; with `out`, the content is
/// copied there first and the artifact is left alone. The files are checked first: the tokenizer must load and the
/// chat template must be the one the server reproduces.
public final class PackageTokenizer {

    private PackageTokenizer() {}

    public static void run(Path artifact, Path checkpoint, Path out) throws IOException {
        TokenizerFiles files = TokenizerFiles.fromCheckpoint(checkpoint);
        QwenTokenizer.load(files);
        QwenChatTemplate.load(files);
        long contentEnd = ArtifactTokenizer.contentEnd(artifact);
        Path target = artifact;
        if (out != null) {
            target = out.resolveSibling(out.getFileName() + ".partial");
            try (FileChannel from = FileChannel.open(artifact, StandardOpenOption.READ);
                    FileChannel to = FileChannel.open(
                            target,
                            StandardOpenOption.CREATE,
                            StandardOpenOption.WRITE,
                            StandardOpenOption.TRUNCATE_EXISTING)) {
                for (long done = 0; done < contentEnd; ) {
                    long moved = from.transferTo(done, contentEnd - done, to);
                    if (moved <= 0) throw new IOException(artifact + " ended " + (contentEnd - done) + " bytes early");
                    done += moved;
                }
            }
        }
        try {
            ArtifactTokenizer.add(target, files);
        } catch (IOException | RuntimeException failure) {
            if (out != null) Files.deleteIfExists(target);
            throw failure;
        }
        if (out != null) Files.move(target, out, StandardCopyOption.REPLACE_EXISTING);
        System.out.printf(
                Locale.ROOT,
                "%s carries the tokenizer of %s: %d bytes after its %d bytes of content%n",
                out == null ? artifact : out,
                checkpoint,
                TokenizerSection.encode(files).length,
                contentEnd);
    }
}
