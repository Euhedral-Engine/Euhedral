# Speculative decoding

This diagram follows dense Qwen3.8 MTP. Drafts propose tokens; row-exact target verification determines the outputs and the persistent target state. An artifact may instead select DFlash2 through the common speculative strategy interface.

```mermaid
flowchart TD
    Prompt[Chunked target prefill] --> Seed[Target hidden seed rows]
    Seed --> Catch[MTP catch-up: committed tokens and target hidden rows]
    Catch --> First[Draft head proposes first token]
    First --> Recursive[Recursive MTP draft rows up to chosen length]
    Recursive --> Verify[Target VERIFY over last committed token and drafts]
    Verify --> Select[Select one target token per verified row]
    Select --> Accept[Accept matching draft prefix then append target selection]
    Accept --> Limit[Cut outputs at EOS or remaining output budget]
    Limit --> Commit[Commit target KV frontier and GDN state for committed rows]
    Commit --> Emit[Emit committed outputs]
    Emit --> Done{Generation ended?}
    Done -->|No| Catch
    Done -->|Yes| Finish[Finish generation]
```

For inputs `[t0, d1, ..., dn]`, row `j` selects target token `g_j`. Accept drafts while `d_(j+1) == g_j`; then output the matching drafts and the next target selection. If all drafts match, the final verified row supplies a bonus token. `SpeculativeAcceptance` also truncates at EOS and the output budget.

Committed **input rows** equal the output count. For `k` outputs, the target has consumed `t0` and the first `k-1` outputs; the last output becomes the next anchor and has not yet been fed. The sequence frontier and GDN recurrent state must match precisely that prefix, and speculative KV beyond that frontier is not committed history.

MTP catch-up commits its cache rows using the verified target hidden states. Recursive proposal rows do not commit MTP history. Prompt chunks also receive catch-up; earlier chunks can run ahead subject to carried-state dependencies and prefix checkpoint boundaries.

`Qwen38Runtime` chooses MTP, DFlash2 or ordinary decoding from the artifact profile, available draft weights and session options. The MTP draft length is the depth the artifact stores when it stores one; otherwise the policy measured for its quantization. `ArtifactProfile` reads the quantization from every text-layer projection, and an artifact whose projections mix formats (`MIXED`) must store its depth: the engine refuses MTP for one that does not, rather than inferring a depth from the formats present. DFlash2 replaces MTP proposal construction with target taps and a block drafter; acceptance remains target verification. Selection supports greedy tokens and a position-aware sampled selector; exactness means agreement with the corresponding ordinary target execution.

## Implementation anchors

- [MtpDecoder](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/speculative/MtpDecoder.java): prefill, catch-up, recursion and verification step ports.
- [SpeculativeAcceptance](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/speculative/SpeculativeAcceptance.java): matching, truncation and committed row count.
- [Quantum](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/Quantum.java) and [Sequence](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/Sequence.java): retirement and state commit.
- [ArtifactProfile](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/ArtifactProfile.java): quantization, speculation and the stored draft depth.
- [Qwen38Runtime](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/Qwen38Runtime.java) and [DFlash2Decoder](../../core/src/main/java/io/euhedral_execution/inference/core/model/qwen38/speculative/DFlash2Decoder.java).
- [MTP contract](../speculation/MTP_CONTRACT.md), [verifier](../speculation/MTP_VERIFIER.md) and [DFlash2](../speculation/DFLASH2.md).

The row-exact contract covers logits and state, not merely final token agreement. Existing CUDA integration tests exercise it; this documentation change does not rerun those tests. Kernel variants and checkpoint byte layouts are outside this diagram.
