package io.euhedral_execution.inference.quantization.trellis;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.InputStream;
import java.util.Iterator;
import java.util.Map;
import org.junit.jupiter.api.Test;

/// The allocation of Qwen3.8's linears equals exllamav3's at 2.5, 2.75, 3.0, 3.25 and 3.5 bpw (frozen fixture;
/// 2.75 and 3.25 promote part of the linears off the floor rate).
class AllocationTest {

    @Test
    void matchesExllamav3() throws Exception {
        JsonNode fixture;
        try (InputStream in = getClass().getResourceAsStream("/trellis-fixture/allocation.json")) {
            fixture = new ObjectMapper().readTree(in);
        }
        for (String bpw : new String[] {"2.5", "2.75", "3.0", "3.25", "3.5"}) {
            Map<String, Double> bits =
                    Allocation.allocate(Linears.qwen38(64, Linears.SWIFT_SHAPES), 64, Double.parseDouble(bpw));
            JsonNode expected = fixture.get(bpw);
            assertEquals(expected.size(), bits.size(), bpw + ": linear count");
            Iterator<Map.Entry<String, JsonNode>> fields = expected.fields();
            while (fields.hasNext()) {
                Map.Entry<String, JsonNode> e = fields.next();
                assertEquals(e.getValue().asDouble(), bits.get(e.getKey()), 0.0, bpw + " " + e.getKey());
            }
        }
    }
}
