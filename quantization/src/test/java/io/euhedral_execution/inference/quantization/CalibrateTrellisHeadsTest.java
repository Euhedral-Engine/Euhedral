package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.euhedral_execution.inference.quantization.gptq.Gptq;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import org.junit.jupiter.api.Test;

/// The end of a trellis run: the heads are rounded after the MTP linears and before the final files and before the
/// resume point is dropped, and a whole-model run needs the draft ids. The rounding is stubbed.
class CalibrateTrellisHeadsTest {

    private static Calibrate.Settings settings(int layers, Path draftIds, Recipe recipe) {
        return new Calibrate.Settings(
                Path.of("model"),
                recipe,
                Path.of("tokens.npy"),
                draftIds,
                Path.of("out"),
                false,
                0.01f,
                8,
                0,
                layers,
                false,
                new Gptq.Nvfp4Search(2, 448f),
                3.0,
                null,
                256,
                1024,
                16);
    }

    private static final class Recording implements Calibrate.TrellisEnd {
        final List<String> events = new ArrayList<>();
        boolean failHeads;

        @Override
        public void roundHeads() throws IOException {
            events.add("heads");
            if (failHeads) throw new IOException("heads failed");
        }

        @Override
        public void finish() {
            events.add("finish");
        }

        @Override
        public void dropCheckpoint() {
            events.add("drop");
        }
    }

    @Test
    void headsAreRoundedBeforeTheFilesAreFinishedAndTheResumePointDropped() {
        var end = new Recording();
        var mtp = new CompletableFuture<Void>();
        var done = Calibrate.endTrellis(mtp, end);
        assertEquals(List.of(), end.events, "nothing runs before the MTP linears are written");
        mtp.complete(null);
        done.join();
        assertEquals(List.of("heads", "finish", "drop"), end.events);
    }

    @Test
    void aFailedHeadRoundingKeepsTheResumePoint() {
        var end = new Recording();
        end.failHeads = true;
        var done = Calibrate.endTrellis(CompletableFuture.completedFuture(null), end);
        assertThrows(CompletionException.class, done::join);
        assertEquals(List.of("heads"), end.events);
        assertFalse(end.events.contains("drop"));
    }

    @Test
    void aWholeModelRunNeedsTheDraftIdsAndTheTrellisRecipe() {
        var trellis = new Recipe("trellis", false);
        assertThrows(
                IllegalArgumentException.class, () -> Calibrate.requireHeadInputs(settings(64, null, trellis), 64));
        assertThrows(
                IllegalArgumentException.class,
                () -> Calibrate.requireHeadInputs(settings(64, Path.of("a.edrl"), null), 64));
        Calibrate.requireHeadInputs(settings(64, Path.of("a.edrl"), trellis), 64);
    }

    @Test
    void aPartialRunNeedsNeither() {
        Calibrate.requireHeadInputs(settings(2, null, null), 64);
    }

    @Test
    void theCommandLineRejectsATrellisRunWithoutDraftIds() {
        var failure = assertThrows(
                IllegalArgumentException.class,
                () -> QuantizationMain.main(new String[] {
                    "calibrate",
                    "--model",
                    "m",
                    "--format",
                    "trellis",
                    "--bits",
                    "3.0",
                    "--tokens",
                    "t.npy",
                    "--out",
                    "o"
                }));
        assertEquals("--draft-ids-from is required", failure.getMessage());
    }

    @Test
    void theQuantizationTrellisNeedsTheTrellisFormat() {
        var failure = assertThrows(
                IllegalArgumentException.class,
                () -> QuantizationMain.main(new String[] {
                    "calibrate", "--model", "m", "--quantization", "trellis", "--tokens", "t.npy", "--out", "o"
                }));
        assertEquals("--quantization trellis needs --format trellis", failure.getMessage());
    }
}
