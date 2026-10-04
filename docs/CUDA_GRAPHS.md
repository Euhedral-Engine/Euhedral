# Captured decode, verification and draft quanta (CUDA graphs)

Decode, speculative verification (VERIFY) and MTP draft (DRAFT) quanta of at most 8 rows replay CUDA graphs captured from earlier
quanta with the same shape. A replayed quantum costs one graph launch on its home lane instead of one host submission per launch
(about 1260 launches for a 3-row Q3 verification, 26 for a one-row draft). `EUHEDRAL_CUDA_GRAPHS=0` turns capture off; every
quantum then submits stage by stage. Sources: `StageGraph` (record and replay), `QwenExecutionContext.captureKey`,
`native/src/host/submission.c`. Hardware for every measurement: RTX 5070 Ti (sm_120, 70 SMs), driver 615.71, desktop session
running.

## Design

**Positions from device memory.** At admission each quantum uploads one input record ahead of its first launch: its token IDs,
then its start position (64-bit, 8-byte aligned), into the workspace's input slot. RoPE, the NVFP4 KV append and every decode
attention kernel (one-row, GQA, row twins, merges) read the position from the record, so a decode quantum's launch parameters do
not change with its position. The host keeps its copy of the position to validate launches and to size decode attention's grids.

**Capture key.** A quantum's key holds what its submissions depend on apart from the input record:

- kind, rows, logits requirement, host logits, draft seed and committed rows;
- the split counts that size decode attention (48-key splits below 2048 keys, 32-key splits from 2048, at most 64 each): below
  2048 keys the key changes every 48 positions, from 2048 it is constant;
- a fingerprint of the workspace storage and of every sequence-owned device address (KV page tables, decode scratch, draft seed
  rows, GDN states and their speculative checkpoints, the host logits' buffers), plus the pending ReplaySSM rows.

A quantum whose KV reservation would allocate pages or upload a page table (every 256 positions) has no key and runs stage by
stage; so do prefill quanta and quanta under exact numerics. Views that stage host-backed weights are captured: their transfers
copy from fixed pinned weights into fixed ring slots, and the ring's ordering with other quanta (`stagingIdle`) is submitted
around the graph on the home lane, as for a quantum that runs stage by stage.

**Recording.** The first quantum with a key runs stage by stage, allocating whatever its stages allocate lazily. The second
records: every launch, copy and memset goes to its lane as usual and again to the lane's shadow, a stream under capture.
Marker waits and records are mirrored on the shadows, so the captured graph keeps the quantum's lane branches. Registered
kernels of a stage that waits on no other lane are captured with programmatic dependent launch at every position and in
every kind of quantum, drafts included (stream submission keeps it below 1024 positions for decode and verification only). Recording costs no device time: the quantum itself runs as usual.
The capture is instantiated when the quantum's last stage submitted, while its device work still runs. A recording that meets a
submission a graph cannot repeat (a synchronous copy, growing the shared scratch) is abandoned, and its key is never recorded
again.

**Shared scratch.** The P2E2 expansion and the FP8 and native FP4 activations live in one scratch that every stream shares; each
use waits on the scratch's marker and records it after itself. A recording chains its uses the same way on the shadows with a
marker of its own, so uses on different branches of the captured graph stay ordered, and marks the capture *ordered*. An ordered
graph runs only while the scratch it captured is still in place (otherwise the quantum runs stage by stage and the capture is
discarded), behind the scratch's marker, which it records again after itself. Drafts of 2 to 8 rows use the scratch (the MTP
input projection is NVFP4 and runs on native FP4 tensor cores from 2 rows).

The ordering is not visible to the submission hash. A capture that omitted it computed different drafts (verifications still
kept the output exact, so only acceptance dropped); `CapturedQuantaCudaIntegrationTest` therefore compares tokens and the
accepted drafts of every verification between a capturing runtime and one that submits stage by stage.

**Replay.** Every later quantum with the key is one lattice frame. It launches the graph on the home lane, behind the input
record's upload, then runs every stage in topological order with its submissions checked instead of run: each stage keeps its
host-side effects (KV reservation, logits readback bookkeeping, ReplaySSM state) and its retirement hook, and the hash of what it
would have submitted (function, launch geometry, dynamic shared memory and parameter bytes; copy and memset addresses and sizes)
must equal the recorded stage's. The checked stages select a sink stream under capture, so a submission that bypasses the hooks is
captured there and never runs. A divergence fails the quantum and discards the capture. Each stage graph keeps at most eight
captures, releasing the least recently used.

## Measurements

### Bound (captured replay, profiler-free)

A probe captured one quantum and replayed it 20 times back to back (Q3 artifact, depth-2 MTP, CUDA events around the replays):

| Quantum | 64-token context | 1K | 4K | 16K |
|---|---|---|---|---|
| VERIFY, 3 rows (1066-1258 nodes) | 17.87 ms | 19.03 ms | 19.02 ms | 19.82 ms |
| VERIFY, 3 rows, programmatic edges at every position | | | 18.93 ms | |
| DRAFT, 1 row (26 nodes) | 0.68 ms | 0.69 ms | 0.69 ms | |

### Host timeline of a speculative step (64-token context, no profiler)

Per quantum, microseconds: gap after the previous quantum's outcome, admission, host submission of every stage, last submission to
the retirement callback, callback to worker, retirement.

| Quantum | Gap | Admission | Submission | To callback | To worker | Retirement |
|---|---|---|---|---|---|---|
| VERIFY | 75 | 40 | 4500 | 13600 | 5 | 30 |
| DRAFT | 75 | 15 | 170 | 610 | 1 | 11 |

The host stays well ahead of the device inside a quantum; the boundaries between a step's three quanta cost about 0.5 ms.

### Coverage

At a 4K prompt with 128 generated tokens per request (chat corpus, a new sequence per request), about 85% of verifications
and one-row drafts replay; the rest are the first and recording quanta of each key (a new sequence fingerprints new addresses)
and quanta at a page boundary. Drafts of 2 to 4 rows replay ordered.

### End to end

Decode tok/s with capture, chat corpus, 128 generated tokens, one fork of four prompts:

| Artifact | 4K | 16K | 32K |
|---|---|---|---|
| `q3` (depth 2) | 120.1 | 116.8 | 108.8 |
| `nvfp4-compressed` (depth 3) | 81.5 | 81.0 | 76.8 |

Prefill is not captured; 2048-token prefill measured 1972 tok/s (`q3`) and 3852 tok/s (`nvfp4-compressed`).

## Rejected

- **Recording on first sight.** A key's first quantum allocates lazily inside its stages (decode scratch, speculative GDN
  buffers, logits selections); a capture of it would hold one-time work, and its key changes with the new addresses anyway.
- **Capture per token with a graph update before launch.** Submitting a 3-row verification takes about 4.5 ms of host time
  while the device runs 18 ms; capturing first and launching after would leave the device idle for that time every quantum.
