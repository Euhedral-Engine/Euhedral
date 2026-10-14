# Trellis golden fixtures

Generated from exllamav3 1.6.1 (MIT License, Copyright Turboderp) with its own CUDA kernels:

    cd tools && PYTHONPATH=<exl3 build>/exllamav3_ext:<exl3 lib> python -m trellis.make_fixture \
        ../quantization/src/test/resources/trellis-fixture <swift-1.5 checkpoint>

Files are little-endian `.npy`. Tiles use the mul1 codebook; `tiles-K-*` hold K = 2, 3, 4, 2.5, 3.5.
`linear-*` is one 1024x512 linear quantized at 3 bpw (seed 0, sigma_reg 0.025, output scales on);
`linear-h.npy` is the normalized, unregularized Hessian exllamav3 consumed. `allocation.json` is
`create_q_strategy` over the Swift-1.5 text model (no hq, mul1 half steps), excluding `lm_head`.

Notes:
- `tiles-in.npy` is unscaled N(0,1) and, like `tiles-K-out.npy`, in kernel (tensor-core) element order; `tiles-K-idx` and the packed words are in that same order.
- `linear-reconstructed.npy` is `quantize_exl3`'s own `weight_q` (built with its fp32 refit scales), in `nn.Linear` [n, k] layout.
- `linear-loaded.npy` is `LinearEXL3(...).get_weight_tensor()` rebuilt from the saved fp16 `trellis`/`suh`/`svh` (what a loader reconstructs), [n, k] float16. The two differ by up to 1.2e-4.
- `allocation.json` has rates 2.5, 2.75, 3.0, 3.25, 3.5; bits are floats. Off-grid rates (2.75, 3.25) promote some linears above the floor rate.
