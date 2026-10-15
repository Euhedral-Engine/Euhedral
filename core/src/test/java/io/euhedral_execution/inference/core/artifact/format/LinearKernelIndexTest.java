package io.euhedral_execution.inference.core.artifact.format;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/// A kernel is launched by its index, so [LinearKernel]'s declaration order must be the native table's order. This
/// reads the table from its C source, without a GPU; the CUDA test checks the built library.
class LinearKernelIndexTest {

    private static final Path TABLE = Path.of("../native/src/host/linear_kernels.c");
    private static final Pattern ENTRY = Pattern.compile("\\{\\s*\"([A-Za-z0-9_]+)\"\\s*,");

    @Test
    void eachKernelSitsAtItsIndexInTheNativeTable() throws IOException {
        String source = Files.readString(TABLE);
        int start = source.indexOf("kernels[] = {");
        assertTrue(start >= 0, "no kernel table in " + TABLE);
        String table = source.substring(start, source.indexOf("};", start));
        List<String> names = new ArrayList<>();
        for (Matcher entry = ENTRY.matcher(table); entry.find(); ) names.add(entry.group(1));
        assertEquals(
                names,
                Arrays.stream(LinearKernel.values()).map(LinearKernel::symbol).toList());
    }
}
