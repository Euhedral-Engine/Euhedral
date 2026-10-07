package io.euhedral_execution.inference.core.model.qwen38.loader;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.artifact.Artifact;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/// Chooses which weights live in pinned host memory when the device cannot hold them all.
///
/// The token embedding goes first: it is a gather, so kernels read the rows they need in place over the
/// bus (a few KiB per token) and it costs almost nothing. Then layer projections, which are staged:
/// every one is read once per decode token, so each host-backed byte costs the same transfer time
/// whichever tensor it belongs to. The choice therefore minimizes the staging ring instead: whole
/// families are taken smallest tensor first, and within the last family the layers are spread evenly,
/// so that transfers interleave with resident work. The LM head is never host-backed.
public final class HostWeightSelection {

    /// The token embedding, the one host-mapped object.
    public static final String EMBEDDING = "text/token_embedding";

    /// Families in selection order, smallest tensor first.
    private static final List<String> FAMILIES = List.of(
            "gdn/query_key",
            "gdn/output",
            "attention/output",
            "attention/query_key",
            "attention/gate_value",
            "gdn/value_z",
            "mlp/down",
            "mlp/gate_up");

    private static final Pattern LAYER_OBJECT = Pattern.compile("text/layers/(\\d+)/(.+)");

    private HostWeightSelection() {}

    /// Names of the objects, the embedding and then projections, together at least `bytes` (or every
    /// candidate when they hold fewer), that are loaded into host memory. Empty for a non-positive
    /// request.
    public static Set<String> select(Artifact artifact, long bytes) {
        Set<String> selected = new LinkedHashSet<>();
        if (bytes <= 0) return selected;
        long remaining = bytes;
        for (TensorDescriptor tensor : artifact.tensors()) {
            if (tensor.name().equals(EMBEDDING)) {
                selected.add(EMBEDDING);
                remaining -= tensor.byteSize();
            }
        }
        if (remaining <= 0) return selected;
        Map<String, List<TensorDescriptor>> families = new HashMap<>();
        for (TensorDescriptor tensor : artifact.tensors()) {
            Matcher matcher = LAYER_OBJECT.matcher(tensor.name());
            if (!matcher.matches()
                    || !(tensor.layout() == WeightLayout.ROW_SPLIT_K128_V1
                            || tensor.layout() == WeightLayout.ROW_SPLIT_K128_SD4_V1)) continue;
            if (FAMILIES.contains(matcher.group(2)))
                families.computeIfAbsent(matcher.group(2), ignored -> new ArrayList<>())
                        .add(tensor);
        }
        for (String family : FAMILIES) {
            List<TensorDescriptor> tensors = families.getOrDefault(family, List.of());
            if (tensors.isEmpty()) continue;
            tensors.sort((a, b) -> Integer.compare(layer(a), layer(b)));
            long familyBytes =
                    tensors.stream().mapToLong(TensorDescriptor::byteSize).sum();
            if (familyBytes <= remaining) {
                tensors.forEach(tensor -> selected.add(tensor.name()));
                remaining -= familyBytes;
            } else {
                long size = tensors.getFirst().byteSize();
                int count = (int) Math.min(tensors.size(), (remaining + size - 1) / size);
                for (int index = 0; index < count; index++)
                    selected.add(tensors.get((int) ((long) index * tensors.size() / count))
                            .name());
                remaining = 0;
            }
            if (remaining <= 0) break;
        }
        return selected;
    }

    private static int layer(TensorDescriptor tensor) {
        Matcher matcher = LAYER_OBJECT.matcher(tensor.name());
        if (!matcher.matches()) throw new IllegalArgumentException(tensor.name());
        return Integer.parseInt(matcher.group(1));
    }
}
