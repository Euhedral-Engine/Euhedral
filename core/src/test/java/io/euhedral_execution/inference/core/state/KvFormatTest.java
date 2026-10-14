package io.euhedral_execution.inference.core.state;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class KvFormatTest {

    @Test
    void pagesHoldEachFormatsHeadRows() {
        assertEquals(256L * 144 * 4, AttentionKvState.planePageBytes(1024, KvFormat.NVFP4));
        assertEquals(256L * 512 * 4, AttentionKvState.planePageBytes(1024, KvFormat.BF16));
    }
}
