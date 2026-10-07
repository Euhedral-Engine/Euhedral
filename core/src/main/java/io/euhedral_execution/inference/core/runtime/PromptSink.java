package io.euhedral_execution.inference.core.runtime;

/// Receives a prompt's token IDs where its tokenization joins, on that worker: where a generation throws its first
/// frame. Called once, with the IDs or the failure. Must not block.
public interface PromptSink {
    void encoded(int[] ids);

    void failed(Throwable failure);
}
