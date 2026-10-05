package io.euhedral_execution.inference.core.model_loader.layer_weights;

/// Persistent byte layout independent of the source tensor data type.
public enum WeightLayout {
    CONTIGUOUS_LE_V1,
    ROW_SPLIT_K128_V1,
    /// Q3G64_F16S values entropy coded as 2-bit primary symbols plus a payload stream
    /// ([io.euhedral_execution.inference.core.model_loader.artifact.P2e2Layout]); smaller than
    /// [#ROW_SPLIT_K128_V1] and lossless with respect to it.
    ROW_SPLIT_P2E2_V1,
    /// NVFP4 whose block scales are 4-bit indices into a per-tensor table of 16 E4M3 codes
    /// ([io.euhedral_execution.inference.core.model_loader.artifact.Nvfp4Layout]): 4.25 instead of 4.5 bits
    /// per weight. Not lossless with respect to [#ROW_SPLIT_K128_V1]: the converter chooses each block's
    /// scale from the table.
    ROW_SPLIT_K128_SD4_V1,
    /// NVFP4 rows that are each self-contained, for gathers: a row's E2M1 codes (K/2 bytes) are followed by
    /// its E4M3 block scales (K/16 bytes), with no padding between rows, and the FP32 global scale follows
    /// the last row at the next 256-byte boundary
    /// ([io.euhedral_execution.inference.core.model_loader.qwen4.NgramLayout]).
    ROW_INTERLEAVED_NVFP4_V1
}
