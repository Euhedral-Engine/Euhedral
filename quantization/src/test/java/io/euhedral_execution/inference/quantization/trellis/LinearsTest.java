package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.euhedral_execution.inference.quantization.checkpoint.Qwen38Objects;
import io.euhedral_execution.inference.quantization.checkpoint.Safetensors;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

/// Every quantized linear's rows are placed exactly once in the layer's objects, the GDN a/b projections stay
/// unquantized (their 48 output rows are not a multiple of 128), and the built-in shapes are the checkpoint's.
/// Needs the checkpoint (skips without it).
class LinearsTest {

    static Path checkpoint() {
        String dir = System.getProperty("euhedral.quantization.checkpoint");
        assumeTrue(dir != null && Files.isDirectory(Path.of(dir)), "no checkpoint");
        return Path.of(dir);
    }

    @Test
    void placementsCoverEveryRowOnce() throws Exception {
        Path dir = checkpoint();
        try (Safetensors store = Safetensors.open(dir)) {
            var objects = new Qwen38Objects(store, dir);
            for (int index : new int[] {0, 3}) {
                for (Linears.Linear linear : Linears.layer(objects, index)) {
                    if (!linear.quantized()) continue;
                    long covered = 0;
                    boolean[] seen = new boolean[linear.rows()];
                    for (var p : Linears.placements(objects, index, linear.tensor())) {
                        covered += p.rows();
                        for (long r = p.linearRow(); r < p.linearRow() + p.rows(); r++) {
                            assertFalse(seen[(int) r], linear.tensor() + " row " + r);
                            seen[(int) r] = true;
                        }
                    }
                    assertEquals(linear.rows(), covered, linear.tensor());
                }
            }
        }
    }

    @Test
    void skipsLinearsNotMultipleOf128() throws Exception {
        Path dir = checkpoint();
        try (Safetensors store = Safetensors.open(dir)) {
            var objects = new Qwen38Objects(store, dir);
            List<Linears.Linear> gdn = Linears.layer(objects, 0);
            int skipped = 0;
            for (var linear : gdn) {
                boolean ab =
                        linear.tensor().endsWith("in_proj_a") || linear.tensor().endsWith("in_proj_b");
                assertEquals(!ab, linear.quantized(), linear.tensor());
                if (ab) {
                    skipped++;
                    assertTrue(Linears.placements(objects, 0, linear.tensor()).isEmpty());
                }
            }
            assertEquals(2, skipped);
        }
    }

    @Test
    void builtInShapesMatchTheCheckpoint() throws Exception {
        Path dir = checkpoint();
        try (Safetensors store = Safetensors.open(dir)) {
            var objects = new Qwen38Objects(store, dir);
            for (int index : new int[] {0, 3}) {
                for (Linears.Linear linear : Linears.layer(objects, index)) {
                    if (!linear.quantized()) continue;
                    String role = linear.tensor().substring(linear.tensor().lastIndexOf('.') + 1);
                    assertArrayEquals(
                            new long[] {linear.rows(), linear.columns()}, Linears.SWIFT_SHAPES.get(role), role);
                }
            }
        }
    }
}
