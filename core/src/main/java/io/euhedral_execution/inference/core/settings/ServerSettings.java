package io.euhedral_execution.inference.core.settings;

import java.time.Duration;
import java.util.Objects;

/// The HTTP server.
///
/// @param port the port to listen on
/// @param shutdownTimeout how long a stop waits for running requests
/// @param requests limits on one request
public record ServerSettings(int port, Duration shutdownTimeout, RequestSettings requests) {

    public ServerSettings {
        if (port < 1 || port > 65535) throw new IllegalArgumentException("port must be 1-65535: " + port);
        Objects.requireNonNull(shutdownTimeout, "shutdownTimeout");
        if (shutdownTimeout.isNegative() || shutdownTimeout.isZero())
            throw new IllegalArgumentException("shutdown-timeout must be positive");
        Objects.requireNonNull(requests, "requests");
    }

    public static ServerSettings defaults() {
        return new ServerSettings(1738, Duration.ofSeconds(10), RequestSettings.defaults());
    }
}
