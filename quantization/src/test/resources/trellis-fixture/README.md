# Trellis golden fixtures

Generated from exllamav3 1.6.1 (MIT License, Copyright Turboderp) with its own CUDA kernels:

    cd tools && PYTHONPATH=<exl3 build>/exllamav3_ext:<exl3 lib> python -m trellis.make_fixture \
        ../quantization/src/test/resources/trellis-fixture <swift-1.5 checkpoint>

Files are little-endian `.npy`. Tiles use the mul1 codebook; `tiles-K-*` hold K = 2, 3, 4, 2.5, 3.5.
`linear-*` is one 1024x512 linear quantized at 3 bpw (seed 0, sigma_reg 0.025, output scales on);
`linear-h.npy` is the normalized, unregularized Hessian exllamav3 consumed. `allocation.json` is
`create_q_strategy` over the Swift-1.5 text model (no hq, mul1 half steps), excluding `lm_head`.
