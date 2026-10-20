package io.euhedral_execution.inference.core.testing;

import io.euhedral_execution.inference.core.tokenizer.TokenizerFiles;

/// Tokenizer files for test artifacts whose tests never tokenize: the section is there, its contents are not a
/// tokenizer.
public final class TestTokenizer {

    public static final TokenizerFiles FILES = new TokenizerFiles("{}", "{}", "template", "{\"eos_token_id\":[2]}");

    private TestTokenizer() {}
}
