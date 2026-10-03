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

- kind, rows, logits requirement, host logits, draft seed and committed rows, programmatic dependent launch;
- the split counts that size decode attention (48-key splits below 2048 keys, 32-key splits from 2048, at most 64 each): below
  2048 keys the key changes every 48 positions, from 2048 it is constant;
- a fingerprint of the workspace storage and of every sequence-owned device address (KV page tables, decode scratch, draft seed
  rows, GDN states and their speculative checkpoints, the host logits' buffers), plus the pending ReplaySSM rows.

A quantum whose KV reservation would allocate pages or upload a page table (every 256 positions) has no key and runs stage by
stage; so do prefill quanta, quanta under exact numerics, and plans that stage host-backed weights.

**Recording.** The first quantum with a key runs stage by stage, allocating whatever its stages allocate lazily. The second
records: every launch, copy and memset goes to its lane as usual and again to the lane's shadow, a stream under capture.
Marker waits and records are mirrored on the shadows, so the captured graph keeps the quantum's lane branches, and the
quantum's programmatic launches become programmatic edges. Recording costs no device time: the quantum itself runs as usual.
The capture is instantiated when the quantum's last stage submitted, while its device work still runs. A recording that meets a
submission a graph cannot repeat (a synchronous copy, the shared P2E2/FP8 scratch with its device-wide event) is abandoned, and
its key is never recorded again.

**Replay.** Every later quantum with the key is one lattice frame. It launches the graph on the home lane, behind the input
record's upload, then runs every stage in topological order with its submissions checked instead of run: each stage keeps its
host-side effects (KV reservation, logits readback bookkeeping, ReplaySSM state) and its retirement hook, and the hash of what it
would have submitted (function, launch geometry, dynamic shared memory and parameter bytes; copy and memset addresses and sizes)
must equal the recorded stage's. The checked stages select a sink stream under capture, so a submission that bypasses the hooks is
captured there and never runs. A divergence fails the quantum and discards the capture. Each stage graph keeps at most eight
captures, releasing the least recently used.

**Not captured.** Draft catch-up quanta of 2 or 3 rows run their linears on the FP8 route through the shared scratch, which a
graph cannot repeat; they run stage by stage.

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
