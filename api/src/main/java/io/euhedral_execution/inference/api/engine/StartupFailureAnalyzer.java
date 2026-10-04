package io.euhedral_execution.inference.api.engine;

import io.euhedral_execution.inference.core.gpu.GpuMemoryException;
import java.util.ArrayList;
import java.util.List;
import org.springframework.boot.diagnostics.FailureAnalysis;
import org.springframework.boot.diagnostics.FailureAnalyzer;

/// Turns a failed engine startup into Boot's short `APPLICATION FAILED TO START` report: what failed, in the
/// messages of its causes, and what to change.
final class StartupFailureAnalyzer implements FailureAnalyzer {
    private static final String SETTINGS =
            "The settings and their environment variables are listed in docs/OPERATIONS.md.";

    @Override
    public FailureAnalysis analyze(Throwable failure) {
        for (Throwable cause = failure; cause != null; cause = cause.getCause()) {
            if (cause instanceof SetupCheck.SetupException setup)
                return new FailureAnalysis(
                        "The server's settings point at files that are missing:\n\n    "
                                + String.join("\n    ", setup.problems()),
                        "Correct the settings above. " + SETTINGS,
                        setup);
            if (cause instanceof EngineStartupException engine) return engine(engine);
        }
        return null;
    }

    private static FailureAnalysis engine(EngineStartupException failure) {
        List<String> messages = new ArrayList<>();
        boolean gpu = false;
        for (Throwable cause = failure.getCause(); cause != null; cause = cause.getCause()) {
            gpu |= cause instanceof GpuMemoryException;
            String message = cause.getMessage();
            if (message != null && messages.stream().noneMatch(seen -> seen.contains(message))) messages.add(message);
        }
        String step = failure.getMessage().substring(0, failure.getMessage().indexOf(" failed: "));
        String action = gpu
                ? "The engine needs the NVIDIA driver, the CUDA 13 runtime libraries, and a Blackwell GPU (compute "
                        + "capability 12.x) this process can see: check nvidia-smi, and in a container pass the GPU "
                        + "(docker run --gpus all). " + SETTINGS
                : "Correct what the message names. " + SETTINGS;
        return new FailureAnalysis(
                step + " failed:\n\n    " + String.join("\n    caused by: ", messages), action, failure);
    }
}
