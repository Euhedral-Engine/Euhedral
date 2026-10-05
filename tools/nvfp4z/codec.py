"""GPU codec: NVFP4 blocks and BF16 high bytes <-> lane-interleaved rANS streams (kernels.cu, docs/NVFP4_LOSSLESS.md).

Needs PyTorch with CUDA. The kernels are compiled at first use with NVRTC through PyTorch and cached on disk.
Everything here is lossless by construction and checked by decoding on the GPU (`verify=True`).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

PROB_BITS = 16
PROB_ONE = 1 << PROB_BITS
SEGMENT_BLOCKS = 1 << 27      # blocks of 16 weights coded as one independent unit (9 bytes of NVFP4 per block)
MAX_LANES = 1 << 16
MIN_BLOCKS_PER_LANE = 1024    # keeps the per-lane length and state at about 0.1% of the stream
WORDS_PER_BLOCK = 5.5         # initial per-lane capacity (average use is about 4.1 for NVFP4); doubled on overflow
KERNEL_NAMES = ("count_blocks", "enc_blocks", "dec_blocks", "compact_lanes", "enc_bytes", "dec_bytes")
PIECE_DTYPES = {"u8": torch.uint8, "u16": torch.int16, "u32": torch.int32}   # u16/u32 hold bit patterns
SOURCE = Path(__file__).with_name("kernels.cu")

_kernels: dict | None = None


def fail(message: str):
    raise ValueError(message)


def _cuda_include_home() -> str:
    """torch's NVRTC helper wants CUDA_HOME for header paths; the kernels include nothing, so any directory works."""
    home = os.environ.get("CUDA_HOME")
    return home if home else tempfile.gettempdir()


def kernels() -> dict:
    global _kernels
    if _kernels is None:
        if not torch.cuda.is_available():
            fail("nvfp4z needs a CUDA device")
        from torch.cuda import _utils
        source = SOURCE.read_text()
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        key = hashlib.sha256(f"{source}|sm_{props.major}{props.minor}|{torch.version.cuda}".encode()).hexdigest()[:24]
        cache = Path(os.environ.get("NVFP4Z_CACHE", Path.home() / ".cache" / "nvfp4z")) / f"{key}.ptx"
        if cache.exists():
            ptx = cache.read_bytes()
        else:
            previous = os.environ.get("CUDA_HOME")
            os.environ["CUDA_HOME"] = _cuda_include_home()
            try:
                ptx, _ = _utils._nvrtc_compile(source, KERNEL_NAMES[0])
            finally:
                if previous is None:
                    del os.environ["CUDA_HOME"]
                else:
                    os.environ["CUDA_HOME"] = previous
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(ptx if isinstance(ptx, bytes) else ptx.encode())
            os.replace(tmp, cache)
        _kernels = _utils._cuda_load_module(ptx, list(KERNEL_NAMES))
    return _kernels


def launch(name: str, threads: int, args: list, smem: int = 0, block: int = 128, grid: int | None = None) -> None:
    grid = grid if grid is not None else (threads + block - 1) // block
    kernels()[name](grid=(grid, 1, 1), block=(block, 1, 1), args=args, shared_mem=smem)


# ---------------------------------------------------------------------------------------------- pinned arena
class Arena:
    """Page-locked host buffer filled by asynchronous device-to-host copies and consumed as one contiguous file image."""

    ALIGN = 64
    PAGE = 4096

    def __init__(self, nbytes: int):
        self.buffer = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        self.array = self.buffer.numpy()
        self.used = 0

    @property
    def capacity(self) -> int:
        return self.buffer.numel()

    def reset(self) -> None:
        self.used = 0

    def alloc(self, nbytes: int) -> int:
        offset = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
        if offset + nbytes > self.capacity:
            fail("pinned arena is full")
        self.used = offset + nbytes
        return offset

    def put(self, tensor: torch.Tensor, kind: str) -> list:
        """Copies a device tensor in (asynchronously) and returns its piece reference [offset, elements, kind]."""
        flat = tensor.reshape(-1)
        nbytes = flat.numel() * flat.element_size()
        offset = self.alloc(nbytes)
        if nbytes:
            self.buffer[offset:offset + nbytes].copy_(flat.view(torch.uint8), non_blocking=True)
        return [offset, flat.numel(), kind]

    def put_host(self, data, kind: str) -> list:
        raw = np.frombuffer(data, dtype=np.uint8)
        offset = self.alloc(raw.size)
        self.array[offset:offset + raw.size] = raw
        return [offset, raw.size // {"u8": 1, "u16": 2, "u32": 4}[kind], kind]

    def piece(self, ref: list) -> torch.Tensor:
        """A pinned view of a piece reference."""
        offset, count, kind = ref
        dtype = PIECE_DTYPES[kind]
        nbytes = count * torch.empty((), dtype=dtype).element_size()
        return self.buffer[offset:offset + nbytes].view(dtype)


# ---------------------------------------------------------------------------------------------- tables
def pow2_floor(n: int) -> int:
    return 1 << max(0, int(n).bit_length() - 1)


def lanes_for(blocks: int) -> int:
    return int(min(MAX_LANES, max(32, pow2_floor(blocks // MIN_BLOCKS_PER_LANE))))


def shift_for(elements: int) -> int:
    """The scale bucket shift of the magnitude contexts: more contexts only pay for big segments."""
    return 3 if elements >= (1 << 28) else 4 if elements >= (1 << 24) else 7


def mag_contexts(shift: int) -> int:
    return (128 >> shift) * 32


def normalize(counts: torch.Tensor) -> torch.Tensor:
    """counts [R, A] int64 -> frequencies [R, A] int64 summing to 2^16 per row, each below 2^16, a count > 0 keeping a
    frequency >= 1, and every row having at least two nonzero entries."""
    c = counts.double()
    total = c.sum(1, keepdim=True)
    f = torch.floor(c / total.clamp(min=1) * PROB_ONE).long()
    f = torch.where((counts > 0) & (f < 1), torch.ones_like(f), f)
    rows = torch.arange(counts.shape[0], device=counts.device)
    top = counts.argmax(1)
    f[rows, top] += PROB_ONE - f.sum(1)
    empty = total[:, 0] == 0
    single = ((f > 0).sum(1) == 1) | empty
    f[empty] = 0
    f[empty, 0] = PROB_ONE
    other = torch.where(top == 0, torch.ones_like(top), torch.zeros_like(top))
    other = torch.where(empty, torch.ones_like(top), other)
    selected = rows[single]
    main = torch.where(empty[single], torch.zeros_like(top[single]), top[single])
    f[selected, other[single]] = 1
    f[selected, main] = PROB_ONE - 1
    return f


def cumulative(freq: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(freq, 1) - freq


def slot_table(freq: torch.Tensor) -> torch.Tensor:
    """uint8 [R * 2^16]: slot -> symbol, for rows of `freq` [R, A]."""
    inclusive = torch.cumsum(freq, 1)
    slots = torch.arange(PROB_ONE, device=freq.device)
    out = torch.empty(freq.shape[0], PROB_ONE, dtype=torch.uint8, device=freq.device)
    for row in range(0, freq.shape[0], 32):
        block = inclusive[row:row + 32].contiguous()
        out[row:row + 32] = torch.searchsorted(block, slots.expand(block.shape[0], PROB_ONE).contiguous(), right=True).to(torch.uint8)
    return out.reshape(-1)


def as_u16(tensor: torch.Tensor) -> torch.Tensor:
    """Values in [0, 2^16) as the int16 bit patterns the kernels read as u16."""
    return tensor.to(torch.int32).to(torch.int16)


def from_u16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.int64) & 0xFFFF


def offsets(lens: torch.Tensor) -> torch.Tensor:
    wide = lens.to(torch.int64)
    return torch.cumsum(wide, 0) - wide


# ---------------------------------------------------------------------------------------------- shared encoder driver
def run_encoder(encode, lanes: int, words_per_lane: float):
    """encode(cap, words, lens, flag) launches the encoder. Retries with a larger per-lane capacity on overflow.
    Returns (lane-major compact words int16 [total], lens int32 [lanes], total words, longest lane)."""
    cap = int(words_per_lane) + 16
    while True:
        words = torch.empty(cap * lanes, dtype=torch.int16, device="cuda")
        lens = torch.zeros(lanes, dtype=torch.int32, device="cuda")
        flag = torch.zeros(1, dtype=torch.int32, device="cuda")
        encode(cap, words, lens, flag)
        total, longest, overflow = torch.stack([lens.sum(dtype=torch.int64), lens.max().to(torch.int64), flag[0].to(torch.int64)]).tolist()
        if not overflow:
            break
        cap *= 2
    compact = torch.empty(total, dtype=torch.int16, device="cuda")
    launch("compact_lanes", lanes, [words, lens, offsets(lens), lanes, compact])
    return compact, lens, total, longest


def lens_piece(arena: Arena, lens: torch.Tensor, longest: int) -> list:
    if longest < (1 << 16):
        return arena.put(as_u16(lens), "u16")
    return arena.put(lens, "u32")


def load_lens(arena: Arena, ref: list) -> torch.Tensor:
    lens = arena.piece(ref).to("cuda", non_blocking=True)
    return (lens.to(torch.int32) & 0xFFFF) if ref[2] == "u16" else lens


def check_stream(freqs: list, lens: torch.Tensor, words: torch.Tensor) -> None:
    """Rejects tables that do not sum to 2^16 and lane lengths that do not add up to the words stored."""
    sums_ok = torch.stack([(freq.sum(1) == PROB_ONE).all() for freq in freqs]).all()
    if not bool(sums_ok) or int(lens.sum()) != words.numel():
        fail("corrupt segment (tables or lane lengths)")


# ---------------------------------------------------------------------------------------------- NVFP4 blocks
def encode_blocks(packed: torch.Tensor, scale: torch.Tensor, b0: int, b1: int, total_blocks: int, groups: int,
                  arena: Arena, verify: bool = True) -> dict:
    """Encodes blocks [b0, b1) of a tensor whose packed codes (uint8 [8 * blocks]) and E4M3 scale bytes
    (uint8 [blocks]) are on the device. `groups` equal parts of the tensor (experts) get their own scale table.
    Returns the segment record (piece references into `arena`)."""
    blocks = b1 - b0
    pk, sc = packed[8 * b0:8 * b1], scale[b0:b1]
    raw_bytes = 9 * blocks
    bpe = max(1, total_blocks // groups)
    shift = shift_for(blocks * 16)
    cm = mag_contexts(shift)
    count_scale = torch.zeros(groups * 128, dtype=torch.int32, device="cuda")
    count_mag = torch.zeros(cm * 8, dtype=torch.int32, device="cuda")
    stats = torch.zeros(2, dtype=torch.int32, device="cuda")
    launch("count_blocks", 0, [pk, sc, blocks, b0, bpe, groups, shift, count_scale, count_mag, cm, stats],
           smem=cm * 8 * 4, block=256, grid=min(1024, (blocks + 255) // 256))
    max_scale, negative_zero = stats.tolist()
    record = {"b0": b0, "b1": b1}
    if max_scale >= 128:      # not an E4M3 scale of a positive weight block: store as is
        record.update(mode="raw", packed=arena.put(pk, "u8"), scale=arena.put(sc, "u8"))
        return record
    freq_scale = normalize(count_scale.view(groups, 128).long())
    freq_mag = normalize(count_mag.view(cm, 8).long())
    cum_scale, cum_mag = cumulative(freq_scale), cumulative(freq_mag)
    lanes = lanes_for(blocks)
    rows = (blocks + lanes - 1) // lanes
    tables = [as_u16(freq_scale.reshape(-1)), as_u16(cum_scale.reshape(-1)), as_u16(freq_mag.reshape(-1)), as_u16(cum_mag.reshape(-1))]
    sign_all = int(negative_zero)

    def encode(cap, words, lens, flag):
        launch("enc_blocks", lanes, [pk, sc, blocks, b0, lanes, rows, bpe, groups, shift, sign_all, cap, *tables, cm, words, lens, flag],
               smem=cm * 8 * 4)

    compact, lens, total, longest = run_encoder(encode, lanes, rows * WORDS_PER_BLOCK)
    stored = 2 * total + (2 if longest < (1 << 16) else 4) * lanes + 2 * (freq_scale.numel() + freq_mag.numel())
    if stored >= raw_bytes:
        record.update(mode="raw", packed=arena.put(pk, "u8"), scale=arena.put(sc, "u8"))
        return record
    if verify:
        out_pk = torch.empty_like(pk)
        out_sc = torch.empty_like(sc)
        decode_blocks_device(out_pk, out_sc, b0, bpe, groups, lanes, rows, shift, sign_all, compact, lens, freq_scale, freq_mag)
        if not (torch.equal(out_pk, pk) and torch.equal(out_sc, sc)):
            fail("round-trip verification failed")
    record.update(mode="rans", lanes=lanes, rows=rows, shift=shift, sign_all=sign_all,
                  freq_scale=arena.put(as_u16(freq_scale.reshape(-1)), "u16"), freq_mag=arena.put(as_u16(freq_mag.reshape(-1)), "u16"),
                  lens=lens_piece(arena, lens, longest), words=arena.put(compact, "u16"))
    return record


def decode_blocks_device(out_pk, out_sc, b0, bpe, groups, lanes, rows, shift, sign_all, words, lens, freq_scale, freq_mag):
    cm = mag_contexts(shift)
    cum_scale, cum_mag = cumulative(freq_scale), cumulative(freq_mag)
    launch("dec_blocks", lanes, [out_pk, out_sc, out_sc.numel(), b0, lanes, rows, bpe, groups, shift, sign_all, words, lens,
                                 offsets(lens), slot_table(freq_scale), as_u16(freq_scale.reshape(-1)), as_u16(cum_scale.reshape(-1)),
                                 as_u16(freq_mag.reshape(-1)), cum_mag.reshape(-1).to(torch.int32), cm], smem=cm * 8 * 6)


def decode_blocks(record: dict, arena: Arena, out_pk: torch.Tensor, out_sc: torch.Tensor, total_blocks: int, groups: int) -> None:
    """Decodes one segment record into the device slices out_pk (uint8 [8 * blocks]) and out_sc (uint8 [blocks])."""
    b0, b1 = record["b0"], record["b1"]
    if record["mode"] == "raw":
        out_pk.copy_(arena.piece(record["packed"]), non_blocking=True)
        out_sc.copy_(arena.piece(record["scale"]), non_blocking=True)
        return
    bpe = max(1, total_blocks // groups)
    freq_scale = from_u16(arena.piece(record["freq_scale"]).to("cuda", non_blocking=True)).view(groups, 128)
    freq_mag = from_u16(arena.piece(record["freq_mag"]).to("cuda", non_blocking=True)).view(mag_contexts(record["shift"]), 8)
    words = arena.piece(record["words"]).to("cuda", non_blocking=True)
    lens = load_lens(arena, record["lens"])
    check_stream([freq_scale, freq_mag], lens, words)
    decode_blocks_device(out_pk, out_sc, b0, bpe, groups, record["lanes"], record["rows"], record["shift"], record["sign_all"],
                         words, lens, freq_scale, freq_mag)


# ---------------------------------------------------------------------------------------------- bytes (BF16 high bytes)
def encode_bytes(data: torch.Tensor, arena: Arena, verify: bool = True) -> dict:
    """Order-0 static rANS of a uint8 device vector."""
    n = data.numel()
    if n >= (1 << 31):
        fail("byte stream too long")
    lanes = lanes_for(n)
    rows = (n + lanes - 1) // lanes
    freq = normalize(torch.bincount(data.long(), minlength=256).view(1, 256))
    cum = cumulative(freq)
    tables = [as_u16(freq.reshape(-1)), as_u16(cum.reshape(-1))]

    def encode(cap, words, lens, flag):
        launch("enc_bytes", lanes, [data, n, lanes, rows, cap, *tables, words, lens, flag])

    compact, lens, total, longest = run_encoder(encode, lanes, rows * 0.6)
    if 2 * total + 4 * lanes + 512 >= n:
        return {"mode": "raw", "data": arena.put(data, "u8")}
    if verify:
        out = torch.empty_like(data)
        decode_bytes_device(out, lanes, rows, compact, lens, freq)
        if not torch.equal(out, data):
            fail("round-trip verification failed")
    return {"mode": "rans", "lanes": lanes, "rows": rows, "freq": arena.put(as_u16(freq.reshape(-1)), "u16"),
            "lens": lens_piece(arena, lens, longest), "words": arena.put(compact, "u16")}


def decode_bytes_device(out, lanes, rows, words, lens, freq):
    cum = cumulative(freq)
    launch("dec_bytes", lanes, [out, out.numel(), lanes, rows, words, lens, offsets(lens), slot_table(freq),
                                as_u16(freq.reshape(-1)), as_u16(cum.reshape(-1))])


def decode_bytes(record: dict, arena: Arena, count: int) -> torch.Tensor:
    if record["mode"] == "raw":
        return arena.piece(record["data"]).to("cuda", non_blocking=True)
    freq = from_u16(arena.piece(record["freq"]).to("cuda", non_blocking=True)).view(1, 256)
    out = torch.empty(count, dtype=torch.uint8, device="cuda")
    words, lens = arena.piece(record["words"]).to("cuda", non_blocking=True), load_lens(arena, record["lens"])
    check_stream([freq], lens, words)
    decode_bytes_device(out, record["lanes"], record["rows"], words, lens, freq)
    return out
