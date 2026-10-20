package io.euhedral_execution.inference.quantization;

import io.euhedral_execution.inference.quantization.gpu.CudaLibraries;
import java.nio.file.Path;
import java.util.Optional;

/// The CUDA libraries of the CUDA tests: the engine's `library` and the module's cuBLAS and cuSOLVER, which the
/// build names in `euhedral.quantization.cuda-libraries`.
public final class TestLibraries {

    private TestLibraries() {}

    public static CudaLibraries of(Path library) {
        return new CudaLibraries(
                library,
                Optional.ofNullable(System.getProperty("euhedral.quantization.cuda-libraries"))
                        .filter(directory -> !directory.isBlank())
                        .map(Path::of));
    }
}
