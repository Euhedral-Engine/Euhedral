package io.euhedral_execution.inference.quantization;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

/// Every quantized object's format, layout and lowest code in all four recipes equal the converter's
/// (recipe-formats.json, written from tools/euhedral_artifacts/inventory.build_plans).
class RecipeTest {

    @Test
    void formatsMatchTheConverter() throws Exception {
        JsonNode recipes = new ObjectMapper().readTree(getClass().getResourceAsStream("/recipe-formats.json"));
        for (var recipe : recipes.properties()) {
            String key = recipe.getKey();
            Recipe built = new Recipe(key.startsWith("q3") ? "q3" : "nvfp4", key.endsWith("compressed"));
            for (var object : recipe.getValue().properties()) {
                var format = built.format(object.getKey());
                JsonNode expected = object.getValue();
                String where = key + " " + object.getKey();
                assertEquals(expected.get(0).asText(), format.name(), where);
                assertEquals(expected.get(1).asText(), format.layout(), where);
                if (!format.nvfp4()) {
                    int qmin = expected.get(2).isNull()
                            ? defaultMin(format.name())
                            : Math.max(
                                    defaultMin(format.name()), expected.get(2).asInt());
                    assertEquals(qmin, format.qmin(), where);
                }
            }
        }
    }

    @Test
    void trellisRoundsTheHeadsAsNvfp4RowSplit() {
        var recipe = new Recipe("trellis", false);
        for (String head : new String[] {"text/output_head", "text/draft_head"}) {
            var format = recipe.format(head);
            assertEquals("NVFP4", format.name(), head);
            assertEquals(Recipe.ROW_SPLIT, format.layout(), head);
        }
    }

    @Test
    void trellisOutputHeadFormatOverridesTheLmHeadOnly() {
        var recipe = new Recipe("trellis", false, "Q4G64_F16S");
        assertEquals("Q4G64_F16S", recipe.format("text/output_head").name());
        assertEquals("NVFP4", recipe.format("text/draft_head").name());
    }

    @Test
    void trellisHasNoFormatForALayerObject() {
        var recipe = new Recipe("trellis", false);
        assertThrows(IllegalArgumentException.class, () -> recipe.format("text/layers/3/mlp/down"));
        assertThrows(IllegalArgumentException.class, () -> recipe.format("mtp/input_projection"));
        assertThrows(IllegalArgumentException.class, () -> recipe.format("text/token_embedding"));
    }

    @Test
    void trellisHasNoCompressedForm() {
        assertThrows(IllegalArgumentException.class, () -> new Recipe("trellis", true));
    }

    private static int defaultMin(String format) {
        return switch (format) {
            case "Q3G64_F16S" -> -4;
            case "Q4G64_F16S" -> -8;
            default -> -16;
        };
    }
}
