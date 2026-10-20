package io.euhedral_execution.inference.quantization.gpu;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/// The native libraries the quantization tools run on: the engine's CUDA library (`--library`), which has the CUDA
/// runtime and NVRTC beside it, and the directory of cuBLAS and cuSOLVER (`--cuda-libraries`), or empty to find them
/// on the library search path.
public record CudaLibraries(Path engine, Optional<Path> math) {

    public CudaLibraries {
        Objects.requireNonNull(engine, "engine");
        Objects.requireNonNull(math, "math");
    }

    /// The engine's library at `engine`, with cuBLAS and cuSOLVER from the library search path.
    public static CudaLibraries of(Path engine) {
        return new CudaLibraries(engine, Optional.empty());
    }
}
