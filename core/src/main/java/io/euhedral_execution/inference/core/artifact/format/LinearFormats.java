package io.euhedral_execution.inference.core.artifact.format;

import io.euhedral_execution.inference.core.artifact.TensorDescriptor;
import io.euhedral_execution.inference.core.artifact.TensorHandle;
import io.euhedral_execution.inference.core.artifact.WeightFormat;
import io.euhedral_execution.inference.core.artifact.WeightLayout;
import io.euhedral_execution.inference.core.model.qwen38.ArtifactProfile;
import java.util.List;
import java.util.Optional;

/// The registry of linear weight formats, keyed by `(WeightFormat, WeightLayout)`.
public final class LinearFormats {

    private static final List<LinearFormat> ALL = List.of(
            new GroupedFormat(WeightFormat.Q3_G64_FP16),
            new GroupedFormat(WeightFormat.Q4_G64_FP16),
            new GroupedFormat(WeightFormat.Q5_G64_FP16),
            new P2e2Format(),
            new Nvfp4Format(WeightLayout.ROW_SPLIT_K128_V1),
            new Nvfp4Format(WeightLayout.ROW_SPLIT_K128_SD4_V1),
            new Bf16Format());

    private LinearFormats() {}

    /// The format registered for `format` and `layout`, if any.
    public static Optional<LinearFormat> find(WeightFormat format, WeightLayout layout) {
        for (LinearFormat registered : ALL)
            if (registered.format() == format && registered.layout() == layout) return Optional.of(registered);
        return Optional.empty();
    }

    /// The format that routes (and sizes the scratch of) an instruction whose weights are `format` and `layout`: the
    /// registered one, else the Q3 grouped format, which anything unregistered has always been routed as.
    public static LinearFormat routing(WeightFormat format, WeightLayout layout) {
        return find(format, layout).orElseGet(() -> ALL.getFirst());
    }

    public static LinearFormat of(TensorDescriptor descriptor) {
        return find(descriptor.format(), descriptor.layout())
                .orElseThrow(() -> unsupported(descriptor.format(), descriptor.layout(), descriptor.name()));
    }

    public static LinearFormat of(TensorHandle handle) {
        return find(handle.format(), handle.layout())
                .orElseThrow(() -> unsupported(handle.format(), handle.layout(), handle.name()));
    }

    public static List<LinearFormat> all() {
        return ALL;
    }

    /// Whether some registered format of `format` is quantized (every one but BF16).
    public static boolean quantized(WeightFormat format) {
        for (LinearFormat registered : ALL)
            if (registered.format() == format && registered.quantization() != ArtifactProfile.Quantization.BF16)
                return true;
        return false;
    }

    private static IllegalArgumentException unsupported(WeightFormat format, WeightLayout layout, String name) {
        return new IllegalArgumentException("unsupported linear weight format " + format + "/" + layout + ": " + name);
    }
}
