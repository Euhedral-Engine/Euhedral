"""Writes the fixture GptqCudaIntegrationTest compares against: tools/gptq rounding (the PyTorch column loop, on the
CPU) of one random BF16 object under a correlated Hessian, in every format, with and without act-order.

Run from the repository root: python3 quantization/src/test/python/make_gptq_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "tools")
from euhedral_artifacts import device  # noqa: E402
from gptq import gptq as G  # noqa: E402

N, K = 64, 256
OUT = Path("quantization/src/test/resources/gptq-fixture")
device.set_device("cpu")
rng = np.random.default_rng(11)
weight = (rng.standard_normal((N, K)) * np.exp(rng.standard_normal((N, 1)))).astype(np.float32)
weight = torch.from_numpy(weight).to(torch.bfloat16)  # the object as the checkpoint stores it
mixing = rng.standard_normal((K, K)).astype(np.float64) * 0.3 + np.eye(K)
x = rng.standard_normal((4096, K)) @ mixing
hessian = torch.from_numpy((x.T @ x).astype(np.float32))
hessian[:, 7] = 0
hessian[7, :] = 0  # one column no input reaches
OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "weight_bf16.npy", weight.view(torch.int16).numpy())
np.save(OUT / "hessian.npy", hessian.numpy())
cases = []
for fmt, layout in (("Q3G64_F16S", "row-split-k128-v1"), ("Q5G64_F16S", "row-split-k128-v1"),
                    ("NVFP4", "row-split-k128-v1"), ("NVFP4", "row-split-k128-sd4-v1")):
    for act in (False, True):
        order = G.column_order(hessian) if act else None
        state, decoded = G.gptq(weight.float(), hessian, fmt, layout, (N, K), order=order, fused=False)
        name = f"{fmt}-{layout}-{'act' if act else 'seq'}"
        np.save(OUT / f"{name}-codes.npy", state.codes)
        np.save(OUT / f"{name}-scales.npy", state.scales)
        np.save(OUT / f"{name}-decoded.npy", decoded.numpy().astype(np.float32))
        cases.append({"name": name, "format": fmt, "layout": layout, "act_order": act,
                      "global_scale": float(state.global_scale), "table": None if state.table is None else state.table.tolist()})
(OUT / "cases.json").write_text(json.dumps({"rows": N, "columns": K, "damp": 0.01, "cases": cases}, indent=1))
print(len(cases), "cases")
