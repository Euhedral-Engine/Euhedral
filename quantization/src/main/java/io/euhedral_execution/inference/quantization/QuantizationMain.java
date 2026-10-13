package io.euhedral_execution.inference.quantization;

/// Command line of the quantization tools.
public final class QuantizationMain {

    private QuantizationMain() {}

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("usage: calibrate ... (see quantization/README.md)");
            System.exit(2);
        }
        throw new IllegalArgumentException("unknown command " + args[0]);
    }
}
