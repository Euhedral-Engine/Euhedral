"""Builders for the four Euhedral inference artifacts (see tools/README.md).

    edrl         container constants, tensor table codec, object plans
    sources      the Hugging Face safetensors checkpoint and row-matrix views of it
    device       the quantization device (CPU NumPy or CUDA PyTorch) and its lookup tables
    grouped      Q3/Q4/Q5 grouped integer quantization
    nvfp4        NVFP4 and NVFP4-SD4 quantization
    q3_p2e2      the lossless Q3 -> P2E2 transcode
    recipes      what each of the four artifacts stores, per object
    inventory    the object inventory of the Qwen3.8-27B checkpoint
    pipeline     checkpoint directory -> artifact file
"""
