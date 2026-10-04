package io.euhedral_execution.inference.core.tokenizer;

/// A grammar over the UTF-8 bytes of generated text, independent of how tokens split them.
///
/// `allows` reports whether `bytes[from..]` may continue the committed text and must not change state;
/// `accept` commits them. `complete` is true when the committed text is a whole sentence of the grammar, after
/// which a generation terminator may end it.
public interface ByteGrammar {

    boolean allows(byte[] bytes, int from);

    void accept(byte[] bytes, int from);

    boolean complete();
}
