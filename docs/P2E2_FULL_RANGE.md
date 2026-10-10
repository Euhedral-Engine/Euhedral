# Open work: P2E2 for the full Q3 code range

Calibrated Q3 rounding (`tools/convert_checkpoint.py --imatrix`, see `tools/README.md`) uses all eight Q3 codes, -4 to 3.
The P2E2 layout of the `q3-compressed` artifact ([COMPRESSED_Q3.md](COMPRESSED_Q3.md)) stores only -3 to 3, because the
round-to-nearest converter never produced -4. Until P2E2 can store -4, `q3-compressed` stays a round-to-nearest artifact and
the converter refuses `--quantization q3 --compressed --imatrix`.

## What has to change

- **The layout.** Give P2E2 a way to store code -4, losslessly, and give the new layout its own name and EDRL layout ordinal,
  so that existing `row-split-p2e2-v1` artifacts keep loading.
- **The transcoder.** `tools/euhedral_artifacts/q3_p2e2.py` (which rejects -4 today) writes the new layout, and its
  decode-back check covers -4.
- **Every reader of the layout:**
  - native: `native/src/q3/p2e2.cuh`, `native/src/q3/kernels.cu`, `native/src/embedding/kernels.cu`,
    `native/src/host/q3_p2e2_geometry.h`, `native/src/host/q3_linear_bf16.c`, `native/src/host/q3_embedding.c`;
  - Java: `P2e2Layout`, `WeightLayout`, `CompactTensorLayout`, `TensorLoader`, `CompactWeightLoader`, `ArtifactProfile`,
    `ExecutionPlan`, `ExecutionGpu`, `CudaGpuMemory`, `ScratchUse`.
- **The pipeline.** Remove the refusal in `tools/euhedral_artifacts/pipeline.py` (`convert`).

## How to know it is done

- `q3-compressed` converted with `--imatrix` is bit for bit the calibrated `q3` in every output: `QwenP2e2CudaIntegrationTest`
  against the calibrated `q3`, and the teacher-forced report of both artifacts byte for byte equal (docs/QUALITY.md).
- The tool tests cover a tensor that uses -4 (`tools/test_q3_p2e2*.py` and the native P2E2 tests).
- The P2E2 decode and prefill numbers in COMPRESSED_Q3.md are measured again, and the size of the new layout is recorded.

## Context

- The current `q3-compressed` artifact is round-to-nearest, while `q3` is calibrated, so the two artifacts no longer give the
  same outputs. Tests that compare them byte for byte must use a round-to-nearest `q3` until this work lands.
- The fallback, if extending P2E2 does not pay, is calibrated Q3 restricted to -3..3 for both artifacts (the search's code
  range in `tools/euhedral_artifacts/grouped.py`), which keeps today's P2E2 and gives up part of the calibrated gain.
