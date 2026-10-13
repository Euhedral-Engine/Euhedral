"""Writes the fixture LayerForwardCudaIntegrationTest compares against (into quantization/build/forward-fixture):
tools/gptq/qwen38.py's BF16 forward of layers 0 (GDN) and 3 (attention) over the same input, two calibration
sequences of 256 tokens embedded, on the CPU; layer 3 also with its keys and values through the engine's NVFP4 cache
codec (tools/flash_next_reference/kvcodec.py), as the engine's attention reads them.

Run from the repository root: python3 quantization/src/test/python/make_forward_fixture.py CHECKPOINT TOKENS.npy
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "tools")
import torch.nn.functional as F  # noqa: E402
from flash_next_reference import kvcodec  # noqa: E402
from gptq import qwen38  # noqa: E402
from gptq.qwen38 import Checkpoint, Layer  # noqa: E402

checkpoint, tokens_path = Path(sys.argv[1]), sys.argv[2]
OUT = Path("quantization/build/forward-fixture")
OUT.mkdir(parents=True, exist_ok=True)
cp = Checkpoint(checkpoint, device="cpu")
tokens = torch.from_numpy(np.load(tokens_path)[:2, :256].astype(np.int64))
hidden = cp.tensor("model.language_model.embed_tokens.weight")[tokens]  # [2, 256, 5120] BF16
np.save(OUT / "input.npy", hidden.view(torch.int16).numpy())
with torch.no_grad():
    for index in (0, 3):
        out = Layer(cp, index).forward(hidden.clone())
        np.save(OUT / f"layer{index}.npy", out.view(torch.int16).numpy())
        print("layer", index, "done")
    plain = F.scaled_dot_product_attention

    def cached(q, k, v, **kwargs):
        k = kvcodec.roundtrip_cache(k.to(torch.bfloat16)).float()
        v = kvcodec.roundtrip_cache(v.to(torch.bfloat16)).float()
        return plain(q, k, v, **kwargs)

    qwen38.F.scaled_dot_product_attention = cached
    out = Layer(cp, 3).forward(hidden.clone())
    qwen38.F.scaled_dot_product_attention = plain
    np.save(OUT / "layer3-kv.npy", out.view(torch.int16).numpy())
    print("layer 3 with the NVFP4 cache done")
