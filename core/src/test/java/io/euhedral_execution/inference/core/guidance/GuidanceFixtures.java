package io.euhedral_execution.inference.core.guidance;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.core.tokenizer.QwenTokenizer;
import java.nio.file.Files;
import java.nio.file.Path;

/// The checkpoint tokenizer and a grammar compiler over it, shared by the tests of one JVM.
final class GuidanceFixtures {
    static final int VOCABULARY = 248320;
    private static QwenTokenizer tokenizer;
    private static GrammarCompiler compiler;

    private GuidanceFixtures() {}

    static synchronized QwenTokenizer tokenizer() throws Exception {
        if (tokenizer == null) {
            Path checkpoint =
                    Path.of(System.getProperty("euhedral.qwen.tokenizer-dir", "/mnt/shared/qwen38-quant/source/qwen"));
            assumeTrue(Files.isRegularFile(checkpoint.resolve("tokenizer.json")));
            tokenizer = QwenTokenizer.load(checkpoint);
        }
        return tokenizer;
    }

    /// The build passes the host's llguidance as `euhedral.llguidance.library`.
    static synchronized GrammarCompiler compiler() throws Exception {
        if (compiler == null) {
            String library = System.getProperty("euhedral.llguidance.library");
            if (library == null) throw new IllegalStateException("euhedral.llguidance.library is not set");
            compiler = GrammarCompiler.forTokenizer(Llguidance.load(Path.of(library)), tokenizer(), VOCABULARY);
        }
        return compiler;
    }
}
