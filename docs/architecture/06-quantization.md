# Quantization and artifact creation

This diagram covers the offline dense Qwen3.8 pipeline. Calibration learns rounding from activations; conversion packs the selected representations into an EDRL artifact consumed by the engine.

```mermaid
flowchart TD
    Checkpoint[BF16 safetensors checkpoint] --> Layer[Read one decoder layer]
    Tokens[Calibration token sequences] --> Forward[Forward batches and collect input Hessians]
    Layer --> Forward
    Forward --> Recipe{Output recipe}
    Recipe -->|Grouped Q3/Q4/Q5 or NVFP4 / SD4| GPTQ[GPTQ rounding and error propagation]
    Recipe -->|Trellis| Allocation[Allocate per-linear rates under average bpw budget]
    Allocation --> Trellis[Trellis rounding using Hessians]
    GPTQ --> Replace[Replace layer weights with decoded rounded values]
    Trellis --> Replace
    Replace --> Next[Forward again for next layer inputs]
    Next --> Layer
    GPTQ --> Codes[Pre-rounded codes]
    Trellis --> EXL3[EXL3 layer checkpoint files]
    Forward --> Heads[Calibrate output and draft heads]
    Heads --> Codes
    Codes --> Convert[Python converter]
    EXL3 --> Convert
    Checkpoint --> Convert
    Importance[Importance matrix and draft shortlist] --> Convert
    Convert --> Artifact[EDRL metadata, layouts and tensor payloads]
    Artifact --> Load[ArtifactReader and WeightLoader validation]
    Artifact --> Score[Offline scoring against reference logits]
```

The loop is layerwise: collect statistics, round, substitute decoded weights and propagate the resulting activations. Hidden states live in pinned host storage and support resume checkpoints. Calibration work runs as lattice frames with CUDA streams; cuBLAS/cuSOLVER and quantization CUDA kernels supply the numerical work.

`Recipe` determines tensor roles and storage rules. GPTQ writes codes read by `--codes-from`. Trellis allocation assigns rates to budgeted linears, and `Exl3Writer` writes the checkpoint used by `--trellis-from`; output and draft heads have their own rounding path. The converter handles embeddings and packs metadata plus payloads, using importance weights where applicable. Check the [quantization guide](../../quantization/README.md) for format-specific head and MTP handling.

## Allocation and RCO terminology

The implemented bit allocation here is [trellis Allocation](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/trellis/Allocation.java), ported from exllamav3 and driven by an average bits-per-weight budget. The converter documentation also names external `GSQ-RCO-GGUF` repositories as sources of importance matrices. Those names do not establish a local RCO optimizer: no separate RCO allocation stage is shown because the inspected calibration and converter code does not expose one.

## Implementation anchors

- [QuantizationMain](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/QuantizationMain.java), [Calibrate](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/Calibrate.java), [CalibrationLattice](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/CalibrationLattice.java) and [Recipe](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/Recipe.java).
- [GPTQ](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/gptq/Gptq.java), [TrellisQuantizer](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/trellis/TrellisQuantizer.java) and [Exl3Writer](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/trellis/Exl3Writer.java).
- [Converter entry point](../../tools/convert_checkpoint.py), [converter package](../../tools/euhedral_artifacts), [tools guide](../../tools/README.md) and [Score](../../quantization/src/main/java/io/euhedral_execution/inference/quantization/Score.java).

This diagram omits training/rotation commands, individual quantizer kernels, format byte layouts and the separate Flash-Next converter path. It makes no quality or throughput claim.
