package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.quantization.gptq.Gptq;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// How each quantized object is stored in an artifact (tools/euhedral_artifacts/recipes.py and inventory.py):
/// the q3 artifact's format per object, mapped to the recipe being built, and compressed q3's code limit.
/// `outputHead`, when set, is the LM head's format (row-split) in place of the recipe's. The trellis recipe stores
/// only the two heads through this class (NVFP4 row-split); its other objects are exllamav3 checkpoint linears.
public record Recipe(String quantization, boolean compressed, String outputHead) {

    public static final String ROW_SPLIT = "row-split-k128-v1";
    public static final String SD4 = "row-split-k128-sd4-v1";
    private static final Pattern LAYER = Pattern.compile("text/layers/(\\d+)/(.+)");

    public Recipe {
        if (!quantization.equals("q3") && !quantization.equals("nvfp4") && !quantization.equals("trellis")) {
            throw new IllegalArgumentException("quantization is q3, nvfp4 or trellis, not " + quantization);
        }
        if (quantization.equals("trellis") && compressed)
            throw new IllegalArgumentException("the trellis recipe has no compressed form");
        if (outputHead != null
                && !java.util.List.of("NVFP4", "Q3G64_F16S", "Q4G64_F16S", "Q5G64_F16S")
                        .contains(outputHead))
            throw new IllegalArgumentException("unknown LM head format " + outputHead);
    }

    public Recipe(String quantization, boolean compressed) {
        this(quantization, compressed, null);
    }

    public Gptq.Format format(String name) {
        if (quantization.equals("trellis")) {
            if (name.equals("text/output_head") && outputHead != null)
                return Gptq.Format.of(outputHead, ROW_SPLIT, null);
            if (name.equals("text/output_head") || name.equals("text/draft_head"))
                return Gptq.Format.of("NVFP4", ROW_SPLIT, null);
            throw new IllegalArgumentException("the trellis recipe rounds only the LM and draft heads, not " + name);
        }
        if (name.equals("text/output_head") && outputHead != null) return Gptq.Format.of(outputHead, ROW_SPLIT, null);
        String q3 = q3Format(name);
        String format, layout;
        if (quantization.equals("q3")) {
            format = q3;
            layout = ROW_SPLIT;
        } else {
            format = "NVFP4";
            layout = compressed && !drafting(name) ? SD4 : ROW_SPLIT;
        }
        // Compressed q3 stores its Q3 tensors other than the MLP down projections in P2E2, which holds -3..3.
        Integer lowest =
                quantization.equals("q3") && compressed && format.equals("Q3G64_F16S") && !name.endsWith("/mlp/down")
                        ? -3
                        : null;
        return Gptq.Format.of(format, layout, lowest);
    }

    /// The MTP layer and the draft head: they propose tokens the base model verifies.
    public static boolean drafting(String name) {
        return name.startsWith("mtp/") || name.equals("text/draft_head");
    }

    /// The object's format in the q3 artifact (inventory.build_plans).
    static String q3Format(String name) {
        if (name.equals("text/output_head") || name.startsWith("mtp/")) return "NVFP4";
        if (name.equals("text/draft_head") || name.equals("text/token_embedding")) return "Q3G64_F16S";
        Matcher m = LAYER.matcher(name);
        if (!m.matches()) throw new IllegalArgumentException("unknown object " + name);
        int layer = Integer.parseInt(m.group(1));
        return switch (m.group(2)) {
            case "attention/query_key", "gdn/query_key", "gdn/value_z" -> "Q4G64_F16S";
            case "attention/gate_value" -> "Q5G64_F16S";
            case "attention/output", "gdn/output", "mlp/gate_up" -> "Q3G64_F16S";
            // The MLP down projections of the first and last six layers carry most of their family's error.
            case "mlp/down" -> layer < 6 || layer >= 58 ? "Q4G64_F16S" : "Q3G64_F16S";
            default -> throw new IllegalArgumentException("unknown object " + name);
        };
    }
}
