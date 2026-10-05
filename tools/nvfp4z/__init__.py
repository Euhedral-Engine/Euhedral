"""Lossless compression of NVFP4 checkpoints (docs/NVFP4_LOSSLESS.md).

    kernels.cu   the CUDA kernels (NVRTC): table counting, lane-interleaved rANS encode and decode
    codec        GPU codec: NVFP4 blocks and BF16 high bytes <-> rANS streams, tables, pinned arena
    container    the .nvfp4z file format and safetensors parsing
    pipeline     shard compression/decompression and the threaded read -> GPU -> write pipeline

Command line: tools/nvfp4z.py.
"""
