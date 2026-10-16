"""NumPy decoder and float64 linear of a trellis (EXL3) weight: a direct port of native/src/trellis/codebook.cuh and of
TrellisKernels.reconstruct. Positions are tensor-core order within a 16x16 tile; the payload is [k/16, n/16, w] int16."""

import numpy as np

INV_SQRT_128 = np.float32(0.08838834764831845)


def positions(w):
    """Ring start bit of the 16-bit window of every position of a tile of `w` words (S(p) - 16 modulo the ring)."""
    ka, mask = w // 16, 0xAAAA if w % 16 == 8 else 0
    out = np.zeros(256, dtype=np.int64)
    for p in range(256):
        s = (p >> 4) * w + sum(ka + ((mask >> (j & 15)) & 1) for j in range((p & 15) + 1))
        out[p] = (s - 16) % (16 * w)
    return out


def decode_tiles(words, w):
    """FP16 values [tiles, 256] (tensor-core position order) of int16 words [tiles, w]."""
    words = np.ascontiguousarray(words, dtype="<i2").reshape(-1, w)
    ring = words.view("<u4").astype(np.uint64)
    nw = w // 2
    start = positions(w)
    wi, offset = start >> 5, (start & 31).astype(np.uint64)
    v = (ring[:, wi] << np.uint64(32)) | ring[:, (wi + 1) % nw]
    state = ((v >> (np.uint64(48) - offset)) & np.uint64(0xFFFF)).astype(np.uint32)
    x = (state.astype(np.uint64) * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
    total = np.uint64(0x6400) + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24)
    a = total.astype(np.uint16).view(np.float16).astype(np.float64)
    b = np.float64(np.uint16(0x1EEE).view(np.float16))
    c = np.float64(np.uint16(0xC931).view(np.float16))
    return (a * b + c).astype(np.float16)  # one fp16 fma: exact in float64, one rounding


def tensor_core_element(pos):
    t, j = pos >> 3, pos & 7
    row = (t & 3) * 2 + ((j & 3) >> 1) * 8 + (j & 1)
    col = (t >> 2) + (j >> 2) * 8
    return row, col


def rotated_weight(trellis):
    """Q [k, n] float16 of a payload [k/16, n/16, w]; decoded one strip of k16 tiles at a time."""
    kt, nt, w = trellis.shape
    row, col = tensor_core_element(np.arange(256))
    q = np.zeros((kt, 16, nt, 16), dtype=np.float16)
    for i in range(kt):
        values = decode_tiles(trellis[i], w)  # [nt, 256]
        q[i][row, :, col] = values.T
    return q.reshape(kt * 16, nt * 16)


def hadamard128_f32(x):
    """The engine's butterfly (stages h = 1..64, then * 1/sqrt(128)) on the last axis in blocks of 128, in float32."""
    x = np.array(x, dtype=np.float32)
    shape = x.shape
    x = x.reshape(-1, 128)
    for h in (1, 2, 4, 8, 16, 32, 64):
        y = x.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        x = y.reshape(-1, 128)
    return (x * INV_SQRT_128).reshape(shape)


def hadamard128_f64(x):
    x = np.array(x, dtype=np.float64)
    shape = x.shape
    x = x.reshape(-1, 128)
    for h in (1, 2, 4, 8, 16, 32, 64):
        y = x.reshape(-1, 128 // (2 * h), 2, h)
        a, b = y[:, :, 0, :].copy(), y[:, :, 1, :].copy()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        x = y.reshape(-1, 128)
    return (x / np.sqrt(128.0)).reshape(shape)


def segment_of_column(ends, n):
    out = np.zeros(n, dtype=np.int64)
    start = 0
    for g, end in enumerate(ends):
        out[start:end] = g
        start = end
    return out


def linear_f64(x, trellis, suh, svh, ends, chunk=512):
    """y = svh * H_n(H_k(x * suh[seg]) @ Q) in float64 for x [rows, k]; suh [s, k] (or [k]), svh [n]."""
    x = np.asarray(x, dtype=np.float64)
    suh = np.asarray(suh, dtype=np.float64).reshape(len(ends), -1)
    q = rotated_weight(trellis)
    z = np.zeros((x.shape[0], q.shape[1]))
    start = 0
    for g, end in enumerate(ends):
        u = hadamard128_f64(x * suh[g])
        for c in range(start, end, chunk):
            z[:, c:min(end, c + chunk)] = u @ q[:, c:min(end, c + chunk)].astype(np.float64)
        start = end
    return np.asarray(svh, dtype=np.float64) * hadamard128_f64(z)


def reconstruct_f64(trellis, suh, svh):
    """The dense [n, k] weight of a one-segment linear, W = diag(svh) H_n Q^T H_k diag(suh)."""
    k = trellis.shape[0] * 16
    return linear_f64(np.eye(k), trellis, suh, svh, [trellis.shape[1] * 16]).T


def splits(k, n, target_ctas):
    return max(1, min(-(-target_ctas // (n // 128)), k // 256))


def bf16_bits(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF).astype("<u2")


def bf16_values(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)
