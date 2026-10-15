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

    private static final Path QWEN4_TABLE = Path.of("../native/src/host/qwen4_ops.c");

    @Test
    void eachKernelSitsAtItsIndexInTheNativeTable() throws IOException {
        assertEquals(
                names(TABLE),
                Arrays.stream(LinearKernel.values()).map(LinearKernel::symbol).toList());
    }

    @Test
    void rowExactBf16KernelsNameTheirFlashNextTableEntries() throws IOException {
        List<String> names = names(QWEN4_TABLE);
        for (RowExactBf16.Kernel kernel : RowExactBf16.Kernel.values()) {
            assertTrue(kernel.index() < names.size(), kernel + " is past the end of " + QWEN4_TABLE);
            assertEquals(kernel.symbol(), names.get(kernel.index()), kernel + " at index " + kernel.index());
        }
    }

    /// The kernel names of a native table, in order, read from its C source.
    private static List<String> names(Path source) throws IOException {
        String text = Files.readString(source);
        int start = text.indexOf("kernels[] = {");
        assertTrue(start >= 0, "no kernel table in " + source);
        String table = text.substring(start, text.indexOf("};", start));
        List<String> names = new ArrayList<>();
        for (Matcher entry = ENTRY.matcher(table); entry.find(); ) names.add(entry.group(1));
        return names;
    }
}
