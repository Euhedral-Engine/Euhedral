package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorHandle;
import java.util.Objects;

/// One token embedding lookup ([LinearFormat#embed]): the BF16 rows of the `[vocabulary, hidden]` table `table` for
/// the `rows` int32 token ids at `tokenIds`, written to `output` (`rows x hidden` BF16). An id outside the vocabulary
/// leaves its output row as it was.
public record EmbeddingLaunch(long tokenIds, TensorHandle table, long output, int rows, int vocabulary, int hidden) {
    public EmbeddingLaunch {
        Objects.requireNonNull(table, "table");
        if (rows <= 0 || vocabulary <= 0 || hidden <= 0)
            throw new IllegalArgumentException("an embedding lookup needs positive rows, vocabulary and width");
    }
}
