package io.euhedral_execution.inference.quantization.mixed;

import java.util.Arrays;

/// What a projection object can be stored as in a mixed artifact: NVFP4, or trellis at 2.5, 3.0, 3.5 or 4.0 bits per
/// weight.
public enum Option {
    NVFP4("nvfp4"),
    TRELLIS_2_5("trellis-2.5"),
    TRELLIS_3_0("trellis-3.0"),
    TRELLIS_3_5("trellis-3.5"),
    TRELLIS_4_0("trellis-4.0");

    private final String key;

    Option(String key) {
        this.key = key;
    }

    /// The name of the option in allocation files and on the command line.
    public String key() {
        return key;
    }

    /// The option named `key`.
    public static Option of(String key) {
        for (Option option : values()) if (option.key.equals(key)) return option;
        throw new IllegalArgumentException("unknown option '" + key + "', the options are "
                + Arrays.stream(values()).map(Option::key).toList());
    }
}
