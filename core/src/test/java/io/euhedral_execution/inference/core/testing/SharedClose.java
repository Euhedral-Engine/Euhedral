package io.euhedral_execution.inference.core.testing;

import org.junit.platform.launcher.LauncherSession;
import org.junit.platform.launcher.LauncherSessionListener;

/// Closes the [Shared] values when the test run ends, after the last class. Registered through `ServiceLoader`.
public final class SharedClose implements LauncherSessionListener {
    @Override
    public void launcherSessionClosed(LauncherSession session) {
        Shared.closeAll();
    }
}
