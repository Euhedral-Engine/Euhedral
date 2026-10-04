package io.euhedral_execution.inference.api;

import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.ConfigurableApplicationContext;
import sun.misc.Signal;

/// Shuts the server down in order on `SIGTERM` and `SIGINT`: the application context first, then the JVM.
///
/// The JVM's own handlers call `System.exit`, which starts every shutdown hook at once. The Euhedral lattice and
/// its pinned worker threads register hooks of their own, so they stopped while the web server was still waiting
/// for the running generation, which then never ended: the client got no response and the engine's close waited
/// for it forever. Closing the context first lets the web server drain, the generation service stop what is
/// left, and the engine close its sessions and then the lattice; the hooks that run at exit find nothing to do.
/// A second signal halts the JVM at once.
final class ShutdownSignals {
    private static final Logger LOG = LoggerFactory.getLogger(ShutdownSignals.class);

    private ShutdownSignals() {}

    static void install(ConfigurableApplicationContext context) {
        AtomicBoolean stopping = new AtomicBoolean();
        for (String name : new String[] {"TERM", "INT"}) {
            Signal signal = new Signal(name);
            Signal.handle(signal, received -> {
                int status = 128 + received.getNumber();
                if (!stopping.compareAndSet(false, true)) {
                    LOG.warn("SIG{} again: exiting without waiting for the shutdown", received.getName());
                    // System.exit would wait in Spring's own hook for the close in progress.
                    Runtime.getRuntime().halt(status);
                }
                LOG.info("SIG{}: shutting down", received.getName());
                try {
                    context.close();
                } finally {
                    System.exit(status);
                }
            });
        }
    }
}
