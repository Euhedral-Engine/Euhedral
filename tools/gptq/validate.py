"""Score the PyTorch forward (qwen38.py) against a llama.cpp BF16 reference: T1 log-probabilities, KL and NLL.
Usage: python -m gptq.validate CHECKPOINT REFERENCE.kld   (run in tools/)"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

import compare_reference as cr
from gptq.qwen38 import LAYERS, Checkpoint, Layer, Model


def main() -> None:
    checkpoint = Checkpoint(Path(sys.argv[1]))
    reference = cr.Reference(sys.argv[2])
    tokens = torch.from_numpy(reference.tokens[0].astype(np.int64)).cuda()[None]
    model = Model(checkpoint)
    start = time.time()
    with torch.inference_mode():
        h = model.embed(tokens)
        for index in range(LAYERS):
            layer = Layer(checkpoint, index)
            h = layer.forward(h)
            del layer
            if index % 8 == 7:
                print(f"layer {index} done at {time.time() - start:.0f} s", flush=True)
        first = reference.first
        logits = model.logits(h[:, first:first + reference.records])[0].float()
    q = torch.log_softmax(logits, dim=-1).double().cpu().numpy()
    p = np.concatenate([reference.log_probabilities(0, b, min(reference.records, b + 128))
                        for b in range(0, reference.records, 128)])
    targets = reference.targets(0)
    index = np.arange(reference.records)
    kl = cr.kl_divergence(p, q)
    nll, ref_nll = -q[index, targets], -p[index, targets]
    print(f"positions {reference.records}: dNLL {nll.mean() - ref_nll.mean():+.5f}, KL {kl.mean():.6f} "
          f"(p99 {np.percentile(kl, 99):.4f}), top-1 {(p.argmax(1) == q.argmax(1)).mean():.4f}, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
