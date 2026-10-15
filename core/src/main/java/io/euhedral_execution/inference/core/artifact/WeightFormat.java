package io.euhedral_execution.inference.core.artifact;

public enum WeightFormat {
    BF16,
    FP16,
    FP32,
    I32,
    Q3_G64_FP16,
    Q4_G64_FP16,
    Q5_G64_FP16,
    NVFP4,
    /// Trellis-coded weights ([io.euhedral_execution.inference.core.artifact.format.TrellisFormat]) with their
    /// companion scale vectors.
    TRELLIS
}
