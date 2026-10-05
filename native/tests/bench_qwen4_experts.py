"""Kernel times of the Flash-Next routed-expert kernels (native/src/qwen4/experts.cuh) for a chunk of rows.

    python bench_qwen4_experts.py

The kernels of every wave of a chunk are captured in one CUDA graph, so the time is the GPU's, without the host's
launch gaps. The experts' records live in 100 device slots; small chunks replicate their waves over distinct slots until
the weights read per replay exceed 250 MB, so they come from DRAM and not from the L2. Routing is uniform
(`topk` distinct experts per row) or skewed (probability falling as rank^-0.9). Uses the numpy venv of the native tests.
"""

import ctypes as C
import sys

import numpy as np

import gpu_harness as H
from gpu_harness import Gpu
import test_qwen4_experts as T

P = H.P
CU = H.CUDA
STORED_BYTES = 2764808  # gate_up 1,843,204 + down 921,604: the bytes of an expert that a kernel reads
SLOTS = 100
ITEM = 16

event_create = H._bind(CU, "cuEventCreate", [C.POINTER(P), C.c_uint])
event_record = H._bind(CU, "cuEventRecord", [P, P])
event_sync = H._bind(CU, "cuEventSynchronize", [P])
event_elapsed = H._bind(CU, "cuEventElapsedTime", [C.POINTER(C.c_float), P, P])
stream_create = H._bind(CU, "cuStreamCreate", [C.POINTER(P), C.c_uint])
begin_capture = H._bind(CU, "cuStreamBeginCapture_v2", [P, C.c_int])
end_capture = H._bind(CU, "cuStreamEndCapture", [P, C.POINTER(P)])
instantiate = H._bind(CU, "cuGraphInstantiateWithFlags", [C.POINTER(P), P, C.c_ulonglong])
graph_launch = H._bind(CU, "cuGraphLaunch", [P, P])


class Bench:
    def __init__(self):
        self.gpu = Gpu(b'#include "qwen4/kernels.cu"\n', architecture="sm_120")
        self.rng = np.random.default_rng(7)
        records = [T.record(self.rng) for _ in range(3)]
        self.slab = self.gpu.upload((b"".join(r[0] for r in records) * 34)[:SLOTS * T.RECORD])
        self.tokens = 512
        self.x = self.gpu.upload(
            T.bf16_bits(self.rng.standard_normal((self.tokens, T.HIDDEN)).astype(np.float32)).tobytes())
        self.out = self.gpu.zeros(self.tokens * T.HIDDEN * 2)
        self.act = self.gpu.zeros(5200 * T.INTER * 2)
        self.weighted = self.gpu.zeros(5200 * T.HIDDEN * 2)
        self.stream = P()
        stream_create(C.byref(self.stream), 1)
        self.functions = {}

    def function(self, name):
        if name not in self.functions:
            handle = P()
            self.gpu.function(C.byref(handle), self.gpu.module, name.encode())
            self.functions[name] = handle
        return self.functions[name]

    def launch(self, name, grid, args, block):
        params = (P * len(args))(*[C.cast(C.pointer(v), P) for v in args])
        self.gpu.launch_kernel(self.function(name), grid[0], grid[1], 1, block, 1, 1, 0, self.stream, params, None)

    def route(self, tokens, topk, skewed):
        ids = np.zeros((tokens, topk), np.int64)
        p = None
        if skewed:
            p = (np.arange(512) + 1.0) ** -0.9
            p /= p.sum()
        for t in range(tokens):
            ids[t] = self.rng.choice(512, topk, replace=False, p=p)
        return ids

    @staticmethod
    def waves(ids, max_experts, max_pairs):
        """Ascending experts cut into waves; per wave (experts, items, pairs, row offsets, row pair lists)."""
        tokens = ids.shape[0]
        per = {}
        for t in range(tokens):
            for e in ids[t]:
                per.setdefault(int(e), []).append(t)
        waves, current, pairs = [], [], 0
        for e in sorted(per):
            if current and (len(current) == max_experts or pairs + len(per[e]) > max_pairs):
                waves.append(current)
                current, pairs = [], 0
            current.append(e)
            pairs += len(per[e])
        waves.append(current)
        described = []
        for experts in waves:
            items, pair_list = [], []
            for i, e in enumerate(experts):
                rows = per[e]
                for g in range(0, len(rows), ITEM):
                    items.append((i, len(pair_list) + g, min(ITEM, len(rows) - g)))
                pair_list += [(t, 0x3E00) for t in rows]
            lists = [[] for _ in range(tokens)]
            for p, (t, _) in enumerate(pair_list):
                lists[t].append(p)
            offsets = np.cumsum([0] + [len(l) for l in lists]).astype("<u4")
            flat = np.array([p for l in lists for p in l], dtype="<u4")
            described.append((experts, items, pair_list, offsets, flat))
        return described, per

    def upload_wave(self, wave, first_slot):
        experts, items, pair_list, offsets, flat = wave
        slots = np.array([self.slab + T.RECORD * ((first_slot + i) % SLOTS) for i in range(len(experts))], dtype="<u8")
        up = self.gpu.upload
        return dict(slots=up(slots.tobytes()),
                    items=up(np.array([[s, b, c, 0] for s, b, c in items], dtype="<i4").tobytes()),
                    pairs=up(np.array(pair_list, dtype="<i4").tobytes()),
                    offsets=up(offsets.tobytes()), flat=up(flat.tobytes()), item_count=len(items))

    def kernels(self, wave, first, tokens):
        g = self.gpu
        u64, u32 = C.c_uint64, C.c_uint
        return [
            (T.NAMES[0], (T.INTER // 32, wave["item_count"]),
             [u64(wave["slots"]), u64(wave["items"]), u64(wave["pairs"]), u64(self.x), u64(self.act),
              u32(T.GATE_UP_OFFSET), u32(T.HIDDEN), u32(T.INTER)], 128),
            (T.NAMES[1], (T.HIDDEN // 128, wave["item_count"]),
             [u64(wave["slots"]), u64(wave["items"]), u64(wave["pairs"]), u64(self.act), u64(self.weighted),
              u32(T.DOWN_OFFSET), u32(T.INTER), u32(T.HIDDEN)], 128),
            (T.NAMES[2], ((T.HIDDEN // 8 + 255) // 256, tokens),
             [u64(self.weighted), u64(wave["offsets"]), u64(wave["flat"]), u64(self.out), u32(T.HIDDEN),
              u32(1 if first else 0)], 256),
        ]

    def graph_time(self, launches, replays):
        """Microseconds per execution of a graph of `launches`, between two events on the stream."""
        begin_capture(self.stream, 0)
        for launch in launches:
            self.launch(*launch)
        graph = P()
        end_capture(self.stream, C.byref(graph))
        executable = P()
        instantiate(C.byref(executable), graph, 0)
        graph_launch(executable, self.stream)
        self.gpu.sync()
        start, stop = P(), P()
        event_create(C.byref(start), 0)
        event_create(C.byref(stop), 0)
        event_record(start, self.stream)
        for _ in range(replays):
            graph_launch(executable, self.stream)
        event_record(stop, self.stream)
        event_sync(stop)
        elapsed = C.c_float()
        event_elapsed(C.byref(elapsed), start, stop)
        return elapsed.value / replays * 1000.0

    def scenario(self, name, tokens, topk, skewed, max_experts=64, max_pairs=1024, replays=20):
        ids = self.route(tokens, topk, skewed)
        waves, per = self.waves(ids, max_experts, max_pairs)
        copies = max(1, int(np.ceil(250e6 / (len(per) * STORED_BYTES))))
        slot = 0
        chunks = []
        for _ in range(copies):
            row = []
            for wave in waves:
                row.append(self.upload_wave(wave, slot))
                slot += len(wave[0])
            chunks.append(row)
        parts = [[], [], []]
        for row in chunks:
            for k, wave in enumerate(row):
                for i, launch in enumerate(self.kernels(wave, k == 0, tokens)):
                    parts[i].append(launch)
        times = [self.graph_time(p, replays) / copies for p in parts]
        everything = [l for row in chunks for k, wave in enumerate(row) for l in self.kernels(wave, k == 0, tokens)]
        total = self.graph_time(everything, replays) / copies
        stored = len(per) * STORED_BYTES
        print(f"{name}: {tokens} rows, {tokens * topk} pairs, {len(per)} experts, {len(waves)} waves, at most "
              f"{max(len(v) for v in per.values())} pairs per expert: {total:.0f} us "
              f"(gate_up {times[0]:.0f}, down {times[1]:.0f}, combine {times[2]:.0f}); "
              f"{stored / 1e6:.0f} MB of experts, {stored / total / 1e3:.0f} GB/s")


if __name__ == "__main__":
    bench = Bench()
    bench.scenario("decode", 1, 10, False, replays=200)
    bench.scenario("2 rows", 2, 10, False, replays=100)
    bench.scenario("4 rows", 4, 10, False, replays=50)
    bench.scenario("64 rows uniform", 64, 10, False)
    bench.scenario("64 rows skewed", 64, 10, True)
    bench.scenario("512 rows uniform", 512, 10, False, replays=10)
    bench.scenario("512 rows skewed", 512, 10, True, replays=10)
    bench.scenario("512 rows uniform, waves of 20", 512, 10, False, max_experts=20, max_pairs=512, replays=10)
