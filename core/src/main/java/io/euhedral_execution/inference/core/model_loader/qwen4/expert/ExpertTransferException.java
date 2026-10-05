package io.euhedral_execution.inference.core.model_loader.qwen4.expert;

/// An expert could not be brought into the cache: its record could not be read, or the copy to the device
/// failed. Every acquirer that waited for the same transfer receives it; the cache is unchanged by the
/// failure and a later request for the same expert tries again.
public final class ExpertTransferException extends RuntimeException {

    public ExpertTransferException(String message, Throwable cause) {
        super(message, cause);
    }
}
