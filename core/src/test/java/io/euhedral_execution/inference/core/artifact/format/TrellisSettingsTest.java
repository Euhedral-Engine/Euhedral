package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.euhedral_execution.inference.core.settings.TrellisKernelSettings;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class TrellisSettingsTest {

    @Test
    void automaticTargetBlocksAreEightPerMultiprocessor() {
        TrellisSettings settings = TrellisSettings.of(TrellisKernelSettings.defaults(), 70);
        assertEquals(560, settings.targetCtas());
        assertEquals(512, settings.mmaMaxRows());
        assertEquals(64, settings.wholeKRows());
        assertEquals(128, settings.wholeKWideRows());
    }

    @Test
    void statedTargetBlocksAreKept() {
        var stated = new TrellisKernelSettings(256, OptionalInt.of(300), 48, 96);
        TrellisSettings settings = TrellisSettings.of(stated, 70);
        assertEquals(new TrellisSettings(256, 300, 48, 96), settings);
    }
}
