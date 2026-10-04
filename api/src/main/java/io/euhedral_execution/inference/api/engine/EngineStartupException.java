package io.euhedral_execution.inference.api.engine;

/// The engine, or what it reads from the checkpoint, failed to load. [StartupFailureAnalyzer] reports it as its
/// causes' messages instead of the bean-creation chain around them.
final class EngineStartupException extends RuntimeException {
    EngineStartupException(String step, Throwable cause) {
        super(step + " failed: " + cause.getMessage(), cause);
    }
}
