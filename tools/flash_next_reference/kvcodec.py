"""The engine's NVFP4 KV cache codec in torch (`--kv-format nvfp4`).

Ported from native/src/attention/nvfp4_kv.cuh (the device codec) with the semantics of
core/.../Nvfp4KvReference.java (the mathematical reference):

  * a row is one head, 256 BF16 values;
  * the row is rotated by the normalized Sylvester H256 (entries +-1/16) in FP32 with the device's fixed butterfly:
    five xor-shuffle stages over the low 5 index bits (stride 1, 2, 4, 8, 16: `lane & stride == 0 ? v + peer : peer - v`),
    then three register stages over the high 3 index bits, then `* 0.0625`;
  * each group of 16 rotated values takes the scale `E4M3(min(448, max(2^-9, fp32(max|x|) / 6)))` (round to nearest even,
    saturating; an all-zero group has scale code 0 and all-zero codes) and codes
    `E2M1(x / fp32(scale))` with IEEE division and round-to-nearest-even ties (a tie goes to the even code:
    <= 0.25 -> 0, < 0.75 -> 1, <= 1.25 -> 2, < 1.75 -> 3, <= 2.5 -> 4, < 3.5 -> 5, <= 5 -> 6, else 7; the sign bit of the value is
    kept, so -0.0 keeps its sign);
  * the represented value is `e2m1(code) * e4m3(scale)` (exact in FP32);
  * attention sees the INVERSE rotation of the represented row (H256 is self-inverse), i.e. `H^T dequant(quant(H k))`,
    the mathematical value the engine's rotated-domain attention computes.

Deviations from the engine, documented in docs/FLASH_NEXT_REFERENCE.md: the inverse rotation is applied here, in FP32 with
the same butterfly, and the result is rounded to BF16 (the cache dtype upstream uses); the engine keeps the rotated
representation and rotates queries and outputs instead, with its own accumulation order.
"""

from __future__ import annotations

import torch

HEAD_DIM = 256
GROUP = 16
SCALE_MIN = 1.0 / 512.0
SCALE_MAX = 448.0
MAX_FINITE = 6.0
_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def hadamard256(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester H256 on the last axis of fp32 `x` [..., 256], in the device's operation order."""
    if x.dtype != torch.float32 or x.shape[-1] != HEAD_DIM:
        raise ValueError("hadamard256 takes fp32 rows of 256")
    lead = x.shape[:-1]
    # index d = lane + 32 * r: lane = low 5 bits, r = high 3 bits.
    v = x.reshape(-1, 8, 32)
    for stride in (1, 2, 4, 8, 16):
        pairs = v.reshape(v.shape[0], 8, 32 // (2 * stride), 2, stride)
        low, high = pairs[:, :, :, 0, :], pairs[:, :, :, 1, :]
        # lane bit clear: value + peer; lane bit set: peer - value (value is the "high" element there)
        v = torch.stack([low + high, low - high], dim=3).reshape(-1, 8, 32)
    for span in (1, 2, 4):
        groups = v.reshape(v.shape[0], 8 // (2 * span), 2, span, 32)
        low, high = groups[:, :, 0], groups[:, :, 1]
        v = torch.stack([low + high, low - high], dim=2).reshape(-1, 8, 32)
    return (v * 0.0625).reshape(*lead, HEAD_DIM)


def e2m1_encode(value: torch.Tensor) -> torch.Tensor:
    """Codes (uint8, bit 3 = sign) of fp32 values, round to nearest even with saturation."""
    a = value.abs()
    code = torch.zeros(value.shape, dtype=torch.uint8, device=value.device)
    code = torch.where(a > 0.25, 1, code)
    code = torch.where(a >= 0.75, 2, code)
    code = torch.where(a > 1.25, 3, code)
    code = torch.where(a >= 1.75, 4, code)
    code = torch.where(a > 2.5, 5, code)
    code = torch.where(a >= 3.5, 6, code)
    code = torch.where(a > 5.0, 7, code)
    return code.to(torch.uint8) | (torch.signbit(value).to(torch.uint8) << 3)


def e2m1_decode(code: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_MAGNITUDES, dtype=torch.float32, device=code.device)
    magnitude = table[(code & 7).long()]
    return torch.where((code & 8) != 0, -magnitude, magnitude)


def quantize_rows(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16/fp32 rows [n, 256] -> (codes uint8 [n, 256], scale codes uint8 [n, 16]) of the rotated rows."""
    rotated = hadamard256(rows.to(torch.float32).contiguous())
    groups = rotated.reshape(-1, HEAD_DIM // GROUP, GROUP)
    max_abs = groups.abs().amax(dim=-1)
    # IEEE division by a same-device tensor: dividing by a Python scalar may be lowered to a multiplication by 1/6.
    raw = max_abs / torch.full_like(max_abs, MAX_FINITE)
    bounded = raw.clamp(min=SCALE_MIN, max=SCALE_MAX)
    scale_bits = bounded.to(torch.float8_e4m3fn).view(torch.uint8)
    scale_bits = torch.where(max_abs == 0, torch.zeros_like(scale_bits), scale_bits)
    represented = scale_bits.view(torch.float8_e4m3fn).to(torch.float32)
    divisor = torch.where(represented == 0, torch.ones_like(represented), represented)
    codes = e2m1_encode(groups / divisor.unsqueeze(-1))
    codes = torch.where((max_abs == 0).unsqueeze(-1), torch.zeros_like(codes), codes)
    return codes.reshape(-1, HEAD_DIM), scale_bits


def dequantize_rows(codes: torch.Tensor, scale_bits: torch.Tensor) -> torch.Tensor:
    """The represented rows, still in the rotated domain, fp32 [n, 256]."""
    represented = scale_bits.view(torch.float8_e4m3fn).to(torch.float32)
    values = e2m1_decode(codes).reshape(-1, HEAD_DIM // GROUP, GROUP) * represented.unsqueeze(-1)
    return values.reshape(-1, HEAD_DIM)


def roundtrip(rows: torch.Tensor) -> torch.Tensor:
    """`H^T dequant(quant(H row))` in fp32 for rows [..., 256]."""
    lead = rows.shape[:-1]
    flat = rows.reshape(-1, HEAD_DIM)
    codes, scales = quantize_rows(flat)
    return hadamard256(dequantize_rows(codes, scales)).reshape(*lead, HEAD_DIM)


def roundtrip_cache(states: torch.Tensor) -> torch.Tensor:
    """Applies the codec to `key_states`/`value_states` [batch, heads, tokens, 256] (BF16) and returns BF16."""
    return roundtrip(states).to(states.dtype)


def pack_rows(rows: torch.Tensor) -> torch.Tensor:
    """The engine's 144-byte cache rows [n, 144]: 128 code bytes (even element in the low nibble), then 16 scale bytes."""
    codes, scales = quantize_rows(rows)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([packed, scales], dim=1)
