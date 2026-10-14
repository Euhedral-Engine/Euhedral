"""Golden fixtures for the module's trellis port, from exllamav3 1.6.1's own kernels (MIT, Turboderp).

Usage: PYTHONPATH=<exl3 ext>:<exl3 lib> python -m trellis.make_fixture OUT_DIR CHECKPOINT
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from exllamav3 import Config, Model
from exllamav3.conversion.allocation import create_q_strategy
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.modules.quant.exl3_lib import quantize as q


def tiles(out: Path) -> None:
    g = torch.Generator(device="cpu").manual_seed(0)
    x = torch.randn(64, 256, generator=g).cuda()
    np.save(out / "tiles-in.npy", x.cpu().numpy().astype(np.float32))
    for k in (2, 3, 4, 2.5, 3.5):
        quant_args = {"K": k, "mul1": True}
        rec, idx = q.quantize_tiles(x, quant_args)
        np.save(out / f"tiles-{k}-idx.npy", idx.cpu().numpy().astype(np.uint16))
        np.save(out / f"tiles-{k}-out.npy", rec.cpu().numpy().astype(np.float16))
        packed = q.pack_trellis(idx.view(1, 64, 256), quant_args)
        np.save(out / f"tiles-{k}-packed.npy", packed.view(64, -1).cpu().numpy().astype(np.int16))


def linear(out: Path) -> None:
    g = torch.Generator(device="cpu").manual_seed(1)
    k, n = 1024, 512
    w = (torch.randn(n, k, generator=g) * 0.02).half()
    a = torch.randn(4096, k, generator=g)
    a[:, :64] *= 8.0  # a few heavy input channels, as real activations have
    count = a.shape[0]
    h = (a.T @ a / count).float()
    np.save(out / "linear-w.npy", w.numpy())
    np.save(out / "linear-h.npy", h.numpy())
    # exllamav3 takes the summed Hessian plus a sample count and divides, regularizes in place
    quant_args = {"K": 3, "mul1": True, "seed": 0, "sigma_reg": 0.025, "devices": [0], "apply_out_scales": True}
    h_data = {"H": (h * count).cuda(), "count": count, "finalized": False, "device": torch.device("cuda:0")}
    weight_q, proxy_err, out_tensors = q.quantize_exl3(
        w.T.float().cuda().contiguous(), h_data, quant_args, return_weight_q=True
    )
    for name in ("trellis", "suh", "svh"):
        np.save(out / f"linear-{name}.npy", out_tensors[name].cpu().numpy())
    np.save(out / "linear-reconstructed.npy", weight_q.T.contiguous().half().cpu().numpy())
    # what the loader rebuilds from the stored fp16 trellis/suh/svh (differs slightly from weight_q's fp32 refit scales)
    loaded = LinearEXL3(
        None, k, n, suh=out_tensors["suh"], svh=out_tensors["svh"], trellis=out_tensors["trellis"],
        mul1=out_tensors["mul1"],
    ).get_weight_tensor()
    np.save(out / "linear-loaded.npy", loaded.T.contiguous().half().cpu().numpy())
    (out / "linear.json").write_text(json.dumps({
        "k": k, "n": n, "bits": 3, "sigma_reg": 0.025, "apply_out_scales": True, "codebook": "mul1", "proxy_err": float(proxy_err),
        "g_scale": float(quant_args["g_scale"]),
    }, indent=1))


def allocations(out: Path, checkpoint: Path) -> None:
    config = Config.from_directory(str(checkpoint))
    model = Model.from_config(config)
    result = {}
    for bpw in (2.5, 2.75, 3.0, 3.25, 3.5):
        # as convert_model.py calls it: mul1 codebook (half steps), no --hq, default 6-bit head
        strategy, _ = create_q_strategy(model, None, config, bpw, 6, 4, False, half_steps=True)
        result[str(bpw)] = {key: float(bits) for key, bits in sorted(strategy.items()) if key != "lm_head"}
    (out / "allocation.json").write_text(json.dumps(result, indent=1, sort_keys=True))


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tiles(out)
    linear(out)
    allocations(out, Path(sys.argv[2]))
