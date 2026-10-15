package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.Calibrate;
import io.euhedral_execution.inference.quantization.CalibrationLattice;
import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import io.euhedral_execution.inference.quantization.gptq.Gptq;
import io.euhedral_execution.inference.quantization.gpu.Device;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/// Two decoder layers of Swift-1.5 in trellis format on 8 calibration sequences: both layer files are written with
/// every quantized linear, a run stopped after layer 0 resumes at layer 1, and the MTP linears are quantized
/// uncalibrated into `mtp.safetensors`. Needs the checkpoint and the calibration tokens (skips without).
class TrellisCalibrateCudaIntegrationTest {

    private static final Path TOKENS = Path.of(System.getProperty(
            "euhedral.quantization.calibration-tokens", "unset-euhedral.quantization.calibration-tokens"));

    @AfterEach
    void resetHooks() {
        Calibrate.checkpointEvery(8);
        Calibrate.stopAfterLayers(0);
    }

    @Test
    void writesTwoLayers(@TempDir Path out) throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        assumeTrue(Files.isRegularFile(TOKENS), "no calibration tokens");
        Calibrate.run(settings(checkpoint, out, 2), library());
        for (int i = 0; i < 2; i++) {
            String header = header(out.resolve("model.language_model.layers." + i + ".safetensors"));
            assertTrue(header.contains("model.language_model.layers." + i + ".mlp.down_proj.trellis"));
            assertTrue(header.contains("model.language_model.layers." + i + ".mlp.gate_proj.svh"));
        }
        assertFalse(Files.exists(out.resolve("config.json")), "a partial run writes no final files");
    }

    @Test
    void resumesAtNextLayer(@TempDir Path out) throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        assumeTrue(Files.isRegularFile(TOKENS), "no calibration tokens");
        Calibrate.checkpointEvery(1);
        Calibrate.stopAfterLayers(1);
        Calibrate.run(settings(checkpoint, out, 2), library());
        assertEquals(0, Calibrate.lastResumeLayer());
        assertTrue(Files.isRegularFile(out.resolve("model.language_model.layers.0.safetensors")));
        assertFalse(Files.exists(out.resolve("model.language_model.layers.1.safetensors")));
        Calibrate.stopAfterLayers(0);
        Calibrate.run(settings(checkpoint, out, 2), library());
        assertEquals(1, Calibrate.lastResumeLayer(), "resumed at layer 1");
        for (int i = 0; i < 2; i++)
            assertTrue(Files.isRegularFile(out.resolve("model.language_model.layers." + i + ".safetensors")));
    }

    @Test
    void quantizesMtpWithoutHessians(@TempDir Path out) throws Exception {
        Path checkpoint = LinearsTest.checkpoint();
        try (Safetensors store = Safetensors.open(checkpoint);
                Device device = new Device(library());
                CalibrationLattice lattice = CalibrationLattice.start(2);
                Exl3Writer writer = new Exl3Writer(checkpoint, out, 3.0, 8, 2048)) {
            var objects = new Qwen38Objects(store, checkpoint);
            try (var rounding = new TrellisRounding(
                    device, lattice, objects, writer, java.util.Map.of(), s -> {}, null, 256, 1024)) {
                rounding.mtp().get();
            }
        }
        String header = header(out.resolve("mtp.safetensors"));
        assertTrue(header.contains("mtp.layers.0.mlp.down_proj.trellis"));
        assertTrue(header.contains("mtp.fc.svh"));
    }

    /// The JSON header of a safetensors file.
    private static String header(Path file) throws java.io.IOException {
        try (var channel = java.nio.channels.FileChannel.open(file)) {
            var size = java.nio.ByteBuffer.allocate(8).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            channel.read(size, 0);
            var json = java.nio.ByteBuffer.allocate((int) size.getLong(0));
            channel.read(json, 8);
            return new String(json.array(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    static Calibrate.Settings settings(Path checkpoint, Path out, int layers) {
        return new Calibrate.Settings(
                checkpoint,
                null,
                TOKENS,
                null,
                out,
                false,
                0.01f,
                8,
                8,
                layers,
                false,
                new Gptq.Nvfp4Search(2, 448f),
                3.0,
                null,
                256,
                1024,
                16);
    }

    static Path library() {
        return Path.of(System.getProperty("euhedral.cuda.library"));
    }
}
