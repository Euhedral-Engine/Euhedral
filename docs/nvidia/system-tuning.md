# System tuning around the GPU

These levers sit outside the kernels: power and clocks, memory overclocking, the display, the
IOMMU and host memory. Most need root or a reboot. **None were applied during this research.**
Measured context is from [microbench/RESULTS.md](microbench/RESULTS.md) and `nvidia-smi`.

## 1. Move the desktop to the iGPU (largest no-code lever)

**Today.** The GNOME/Xwayland desktop, Chrome's GPU process and RustDesk all run on the 5070 Ti.

| Cost | Size | Source |
|---|---|---|
| GPU memory taken by the desktop | 470–705 MiB, fluctuating | NVFP4_COMPRESSED.md |
| Host-backed NVFP4 at long context | each 128 MiB costs 2–2.5% | MTP_VERIFIER.md |
| GPU time taken by the desktop | about 5–11% | observed during earlier gates |
| Bimodal benchmark JVMs (a slow mode on the GPU side) | 13% apart | FRAME_MODEL.md |
| Display watchdog | `KERNEL_EXEC_TIMEOUT = 1` is set, so a long kernel can be killed | attributes |

**The fix.** The i9-14900K has an Intel UHD 770 iGPU, but `lspci` lists only the NVIDIA GPU, so the
iGPU is disabled in firmware. To use it:
1. In the BIOS, enable the iGPU (usually "iGPU Multi-Monitor", or primary display = IGFX).
2. Plug the monitor into the motherboard.
3. Let GNOME run on the Intel driver. Apps that need the NVIDIA GPU can still use PRIME render
   offload.

**What it buys:**
- 0.5–0.7 GiB of VRAM, which means less host-backed weight streaming at 32K–64K.
- An end to the desktop's share of GPU time and the slow mode it causes.
- No watchdog.
- Cleaner benchmarks.

**What it costs:** the iGPU's desktop performance is lower, and the cable has to move. Nothing in
Euhedral changes.

## 2. Memory overclocking (decode is DRAM-bound)

**Why it matters.** One-row decode streams weights at 720–800 GB/s, against an 855 GB/s measured
ceiling. Raising the memory clock raises that ceiling about one-for-one, and DRAM-bound decode time
scales with it.

**How to set it.**
- Use NVML: `nvmlDeviceSetClockOffsets` for the memory clock domain (older drivers: `...MemClkVfOffset`).
  It needs root. Tools such as LACT wrap it, and it works without X.
- `nvidia-smi -lmc` only locks the memory clock within the stock table; it doesn't add an offset.
- The offset resets on reboot or driver reload.

**Hazards specific to GDDR7.**
- The link has CRC with Error Detection and Replay, and the dies have always-on ECC.
- An overclock that is too far shows up as replays: bandwidth drops while nothing visibly fails.
  Silent corruption is possible past that point.

**Procedure.**
1. Raise the offset in steps.
2. At each step, run `nvbench mem bw`, which should rise.
3. Back off as soon as bandwidth stops rising or falls.
4. Then run Euhedral's teacher-forced NLL check (docs/NVFP4_COMPRESSED.md "Quality method") to rule
   out corruption.

**Not measured here.** This changes hardware settings and needs your go-ahead and root. Reports of
typical GDDR7 headroom vary by board; treat any number as something to measure.

## 3. Core clock, voltage and power

**Measured state:**
- Under every load the SM clock sat at 2.80–2.88 GHz, with "Reliability" as the active limit (the
  voltage cap). The V/F table maximum is 3105 MHz.
- Power was 130–232 W in register-only tensor loops, against a 350 W limit (already raised from the
  300 W default).
- The software power-cap counter shows 0.45 s of capping since driver load.

**Implications:**
- Prefill is tensor-bound and would gain from a core-clock offset; decode would not.
- An undervolt (curve offset) raises the clock reachable at the reliability voltage.
- Raising power further has no effect while the voltage cap is the active limit.

**For benchmarks:**
- Lock clocks (`nvidia-smi -lgc` / `-lmc`, root) when comparing kernels whose clock behaviour may
  differ.
- For end-to-end gates, keep stock behaviour and pair the runs.

## 4. IOMMU passthrough for host-backed weights

**Measured:**
- With the IOMMU translating, pinned buffers on 4 KiB pages copy H2D at 25 GB/s. A THP arena reaches
  43.5 GB/s.
- D2H reaches 57 GB/s, so the link is not what limits H2D.

**Experiment:**
1. Add `iommu=pt` (keeping `intel_iommu=on`) to the kernel command line in GRUB.
2. Reboot.
3. Rerun `nvbench pcie`.

**Expected:**
- 4 KiB-page pinned memory should approach the THP rate, which would remove the arena's
  huge-page dependence.
- The 43.5 GB/s H2D ceiling may rise toward the D2H rate.
- Either result moves the NVFP4 long-context host stream, which bounds every quantum there.

**Risks:** passthrough gives up DMA isolation for devices that use it, and some virtualization
setups need translation. Weigh both before applying.

## 5. Host memory

- Keep host-backed weights in the single 2 MiB THP arena (docs/NVFP4_RESIDENCY.md "Platform").
  Measured here: `cuMemHostAlloc` without THP is 40% slower H2D, and pageable memory gets 15 GB/s.
- THP is in `madvise` mode on this host, so only regions that ask for it get huge pages. Euhedral's
  arena asks.
- Persistence mode is already on, which avoids driver re-initialisation between processes.

## 6. Other host-side items

**MPS is not enabled** (`MPS_ENABLED = 0`). Running several Euhedral processes on one GPU would
need it for concurrent kernels across processes. A single engine doesn't.

**Containers that share the GPU:**
- **`euhedral-inference-serve`** holds about 13 GB while it is up, so any benchmark needs it
  stopped (see the deployment section of the repo README).
- **`ollama`** has a container on this host too. It held no GPU memory during these runs, but a
  request to it would load a model onto the same GPU.
