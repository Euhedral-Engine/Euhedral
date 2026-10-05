"""Shard-level compression and decompression, and the threaded file pipeline.

A shard is one safetensors file. The GPU thread works on whole shard images held in page-locked buffers, so the
storage reads (several threads), the kernels and the storage writes (several threads) overlap and no data is copied on
the host except raw tensors:

    reader threads -> pinned input arena -> GPU thread -> pinned output arena -> writer threads
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import mmap
import os
from pathlib import Path
import queue
import struct
import threading
import time
from typing import Callable

import numpy as np
import torch

from nvfp4z import codec, container

EXTENSION = ".nvfp4z"
IO_CHUNK = 64 << 20
FORMAT = 1


def fail(message: str):
    raise ValueError(message)


# ---------------------------------------------------------------------------------------------- shard <-> arena
def compress_image(image: np.ndarray, name: str, sha256: str, out: codec.Arena, verify: bool = True,
                   segment_blocks: int = codec.SEGMENT_BLOCKS) -> bytes:
    """Compresses a safetensors file image (uint8 array, ideally page-locked) into `out`; returns the header block
    that precedes `out`'s payload in the container file."""
    data_start, header_bytes, header = container.parse_safetensors(image)
    items = container.plan(header)
    position = 0
    for item in items:
        if item["start"] != position:
            fail("the tensors of the shard do not tile its data region")
        position = item["end"]
    if data_start + position != len(image):
        fail("the shard has bytes after its last tensor")
    out.reset()
    source = torch.from_numpy(image)
    tensors = []
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for item in items:
            start, end = data_start + item["start"], data_start + item["end"]
            record = {"name": item["name"], "kind": item["kind"]}
            if item["kind"] == "nvfp4":
                packed_item = item
                scale_entry = header[item["partner"]]
                if scale_entry["data_offsets"][1] - scale_entry["data_offsets"][0] != item["blocks"]:
                    fail(f"{item['name']}: scale tensor size does not match the code tensor")
                scale_start = data_start + scale_entry["data_offsets"][0]
                packed = source[start:end].to("cuda", non_blocking=True)
                scale = source[scale_start:scale_start + item["blocks"]].to("cuda", non_blocking=True)
                total = item["blocks"]
                record.update(partner=item["partner"], blocks=total, groups=item["groups"], segments=[
                    codec.encode_blocks(packed, scale, b0, min(total, b0 + segment_blocks), total, item["groups"], out, verify)
                    for b0 in range(0, total, segment_blocks)])
                del packed, scale
            elif item["kind"] == "bf16":
                words = source[start:end].to("cuda", non_blocking=True).view(-1, 2)
                record.update(count=item["count"], hi=codec.encode_bytes(words[:, 1].contiguous(), out, verify),
                              lo=out.put(words[:, 0].contiguous(), "u8"))
                del words
            elif item["kind"] == "raw":
                record["data"] = out.put_host(image[start:end], "u8")
            tensors.append(record)
        stream.synchronize()
    meta = {"format": FORMAT, "source": {"name": name, "size": len(image), "sha256": sha256,
                                           "safetensors_header": header_bytes.decode()}, "tensors": tensors,
            "payload_bytes": out.used}
    return container.pack_header(meta)


def decompress_image(meta: dict, payload: codec.Arena, out: np.ndarray, out_tensor: torch.Tensor) -> None:
    """Rebuilds the source file image into `out` (uint8 array backed by the pinned tensor `out_tensor`)."""
    source = meta["source"]
    header_bytes = source["safetensors_header"].encode()
    data_start = 8 + len(header_bytes)
    out[:8] = np.frombuffer(struct.pack("<Q", len(header_bytes)), dtype=np.uint8)
    out[8:data_start] = np.frombuffer(header_bytes, dtype=np.uint8)
    header = {k: v for k, v in json.loads(header_bytes).items() if k != "__metadata__"}
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for record in meta["tensors"]:
            entry = header[record["name"]]
            start, end = data_start + entry["data_offsets"][0], data_start + entry["data_offsets"][1]
            kind = record["kind"]
            if kind == "nvfp4":
                blocks, groups = record["blocks"], record["groups"]
                packed = torch.empty(8 * blocks, dtype=torch.uint8, device="cuda")
                scale = torch.empty(blocks, dtype=torch.uint8, device="cuda")
                for segment in record["segments"]:
                    b0, b1 = segment["b0"], segment["b1"]
                    codec.decode_blocks(segment, payload, packed[8 * b0:8 * b1], scale[b0:b1], blocks, groups)
                out_tensor[start:end].copy_(packed, non_blocking=True)
                scale_entry = header[record["partner"]]
                scale_start = data_start + scale_entry["data_offsets"][0]
                out_tensor[scale_start:scale_start + blocks].copy_(scale, non_blocking=True)
                del packed, scale
            elif kind == "bf16":
                count = record["count"]
                words = torch.empty(count, 2, dtype=torch.uint8, device="cuda")
                words[:, 1] = codec.decode_bytes(record["hi"], payload, count)
                words[:, 0] = payload.piece(record["lo"]).to("cuda", non_blocking=True)
                out_tensor[start:end].copy_(words.view(-1), non_blocking=True)
                del words
            elif kind == "raw":
                offset, count, _ = record["data"]
                out[start:end] = payload.array[offset:offset + count]
        stream.synchronize()


# ---------------------------------------------------------------------------------------------- file IO
def read_into(path: Path, array: np.ndarray, size: int, hasher=None) -> None:
    view = memoryview(array)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        done = 0
        while done < size:
            count = os.preadv(fd, [view[done:min(size, done + IO_CHUNK)]], done)
            if count <= 0:
                fail(f"{path}: file ended early")
            if hasher is not None:
                hasher.update(view[done:done + count])
            done += count
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def filesystem_type(path: Path) -> str:
    """The filesystem type of the longest mount point containing `path` ('' when unknown)."""
    try:
        target = str(path.resolve())
        best, kind = "", ""
        with open("/proc/self/mountinfo") as handle:
            for line in handle:
                fields = line.split()
                mount = fields[4].replace("\\040", " ")
                if (target == mount or target.startswith(mount.rstrip("/") + "/")) and len(mount) >= len(best):
                    best, kind = mount, fields[fields.index("-") + 2]
        return kind
    except OSError:
        return ""


def wants_direct(target_dir: Path, setting: str) -> bool:
    """--direct-io auto|on|off. exFAT writes are faster without the page cache; ext4 and others are not."""
    if setting == "auto":
        return filesystem_type(target_dir) == "exfat"
    return setting == "on"


def write_file(path: Path, header: bytes | None, array: np.ndarray, length: int, fsync: bool, hasher=None,
               direct: bool = False) -> None:
    """Writes `header` then array[:length] to path through a temporary file, renamed when complete. With `direct` the
    data bypasses the page cache: whole pages are written (from `array`'s capacity past `length`) and the file is
    then cut to its size; the header must be a multiple of 4096 bytes and the array page aligned."""
    tmp = path.with_name(path.name + ".tmp")
    direct = direct and array.ctypes.data % codec.Arena.PAGE == 0 and (header is None or len(header) % codec.Arena.PAGE == 0) \
        and length + codec.Arena.PAGE <= array.size
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | (os.O_DIRECT if direct else 0), 0o644)
    try:
        if header is not None:
            if direct:
                bounce = mmap.mmap(-1, len(header))
                bounce.write(header)
                os.write(fd, memoryview(bounce))
                bounce.close()
            else:
                os.write(fd, header)
        view = memoryview(array)
        padded = (length + codec.Arena.PAGE - 1) // codec.Arena.PAGE * codec.Arena.PAGE if direct else length
        done = 0
        while done < padded:
            end = min(padded, done + IO_CHUNK)
            if hasher is not None and done < length:
                hasher.update(view[done:min(end, length)])
            written = 0
            while written < end - done:
                written += os.write(fd, view[done + written:end])
            done = end
        if direct and padded != length:
            os.ftruncate(fd, (len(header) if header is not None else 0) + length)
        if fsync:
            os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------------- the runner
@dataclass
class Job:
    source: Path
    target: Path
    size: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class Result:
    name: str
    bytes_in: int
    bytes_out: int
    seconds: float


class Pool:
    """Page-locked buffers, allocated on first demand up to `count` (locking pages costs about 0.2 s per GB)."""

    def __init__(self, count: int, nbytes: int):
        self.items: queue.Queue = queue.Queue()
        self.count, self.nbytes, self.created = count, nbytes, 0
        self.lock = threading.Lock()

    def get(self, stop: threading.Event):
        while not stop.is_set():
            with self.lock:
                if self.items.empty() and self.created < self.count:
                    self.created += 1
                    return codec.Arena(self.nbytes)
            try:
                return self.items.get(timeout=0.05)
            except queue.Empty:
                continue
        raise InterruptedError

    def put(self, arena) -> None:
        self.items.put(arena)


def pinned_budget(gigabytes: float | None) -> int:
    """Page-locked memory for shard buffers: the given GiB, by default a quarter of RAM up to 16 GiB."""
    if gigabytes is not None:
        return int(gigabytes * 2 ** 30)
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return int(min(16 * 2 ** 30, ram // 4))


def arena_counts(budget: int, in_bytes: int, out_bytes: int, readers: int, writers: int) -> tuple[int, int]:
    """How many input and output arenas fit the pinned-memory budget, at least one each."""
    wanted_in, wanted_out = readers + 1, writers + 1
    while wanted_in + wanted_out > 2 and wanted_in * in_bytes + wanted_out * out_bytes > budget:
        if wanted_in >= wanted_out and wanted_in > 1:
            wanted_in -= 1
        elif wanted_out > 1:
            wanted_out -= 1
        else:
            break
    return wanted_in, wanted_out


def run_pipeline(jobs: list[Job], *, in_bytes: int, out_bytes: int, read: Callable, process: Callable, write: Callable,
                 readers: int, writers: int, gpus: int, pinned_budget: int, on_result: Callable[[Result], None]) -> None:
    """read(job, in_arena) -> context; process(job, context, in_arena, out_arena) -> header (GPU threads);
    write(job, header, out_arena) -> bytes written. Raises the first error from any stage."""
    if not jobs:
        return
    readers, writers, gpus = min(readers, len(jobs)), min(writers, len(jobs)), min(gpus, len(jobs))
    n_in, n_out = arena_counts(pinned_budget, in_bytes, out_bytes, readers + gpus - 1, writers + gpus - 1)
    in_pool, out_pool = Pool(n_in, in_bytes), Pool(n_out, out_bytes)
    stop, errors = threading.Event(), []
    todo: queue.Queue = queue.Queue()
    for job in jobs:
        todo.put(job)
    loaded: queue.Queue = queue.Queue()
    processed: queue.Queue = queue.Queue()
    started = {}
    taken, lock, finished = [0], threading.Lock(), threading.Event()

    def guarded(function):
        def run():
            try:
                function()
            except InterruptedError:
                pass
            except BaseException as error:  # noqa: BLE001 - reported by the caller
                errors.append(error)
                stop.set()
        return run

    def reader():
        while not stop.is_set():
            try:
                job = todo.get_nowait()
            except queue.Empty:
                return
            arena = in_pool.get(stop)
            started[job.source] = time.perf_counter()
            loaded.put((job, read(job, arena), arena))

    def gpu():
        while True:
            with lock:
                if taken[0] >= len(jobs):
                    return
                taken[0] += 1
            while True:
                if stop.is_set():
                    raise InterruptedError
                try:
                    job, context, in_arena = loaded.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            out_arena = out_pool.get(stop)
            header = process(job, context, in_arena, out_arena)
            in_pool.put(in_arena)
            processed.put((job, header, out_arena))

    def writer_thread():
        while not stop.is_set():
            try:
                job, header, out_arena = processed.get(timeout=0.1)
            except queue.Empty:
                if finished.is_set() and processed.empty():
                    return
                continue
            written = write(job, header, out_arena)
            out_pool.put(out_arena)
            on_result(Result(job.source.name, job.size, written, time.perf_counter() - started[job.source]))

    threads = [threading.Thread(target=guarded(reader), daemon=True) for _ in range(readers)]
    threads += [threading.Thread(target=guarded(writer_thread), daemon=True) for _ in range(writers)]
    gpu_threads = [threading.Thread(target=guarded(gpu), daemon=True) for _ in range(gpus)]
    for thread in threads + gpu_threads:
        thread.start()
    for thread in gpu_threads:
        thread.join()
    finished.set()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def compress_files(jobs: list[Job], *, verify: bool = True, segment_blocks: int = codec.SEGMENT_BLOCKS, readers: int = 3,
                   writers: int = 4, gpus: int = 2, pinned_gb: float | None = None, fsync: bool = True, direct_io: str = "auto",
                   on_result=lambda result: None) -> None:
    for job in jobs:
        job.size = job.source.stat().st_size
    biggest = max((job.size for job in jobs), default=0)

    def read(job, arena):
        hasher = hashlib.sha256()
        read_into(job.source, arena.array, job.size, hasher)
        return hasher.hexdigest()

    def process(job, sha256, in_arena, out_arena):
        return compress_image(in_arena.array[:job.size], job.source.name, sha256, out_arena, verify, segment_blocks)

    def write(job, header, out_arena):
        write_file(job.target, header, out_arena.array, out_arena.used, fsync, direct=wants_direct(job.target.parent, direct_io))
        return len(header) + out_arena.used

    run_pipeline(jobs, in_bytes=biggest, out_bytes=biggest + (4 << 20), read=read, process=process, write=write, readers=readers,
                 writers=writers, gpus=gpus, pinned_budget=pinned_budget(pinned_gb), on_result=on_result)


def read_container_meta(path: Path) -> tuple[dict, int]:
    """(meta, header block size) of a container file."""
    with open(path, "rb") as handle:
        block = container.header_block_size(handle.read(16))
        handle.seek(0)
        return container.read_header(handle.read(block))[0], block


def decompress_files(jobs: list[Job], *, readers: int = 3, writers: int = 4, gpus: int = 2, pinned_gb: float | None = None, fsync: bool = True,
                     direct_io: str = "auto", on_result=lambda result: None) -> None:
    metas = {}
    for job in jobs:
        meta, block = read_container_meta(job.source)
        metas[job.source] = (meta, block)
        job.size = job.source.stat().st_size
        job.extra["original"] = meta["source"]["size"]
    biggest_payload = max((job.size for job in jobs), default=0)
    biggest_original = max((job.extra["original"] for job in jobs), default=0)

    def read(job, arena):
        meta, block = metas[job.source]
        view = memoryview(arena.array)
        fd = os.open(job.source, os.O_RDONLY)
        try:
            done = 0
            size = job.size - block
            while done < size:
                count = os.preadv(fd, [view[done:min(size, done + IO_CHUNK)]], block + done)
                if count <= 0:
                    fail(f"{job.source}: file ended early")
                done += count
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return meta

    def process(job, meta, in_arena, out_arena):
        decompress_image(meta, in_arena, out_arena.array, out_arena.buffer)
        return meta["source"]

    def write(job, source, out_arena):
        hasher = hashlib.sha256()
        write_file(job.target, None, out_arena.array, source["size"], fsync, hasher, direct=wants_direct(job.target.parent, direct_io))
        if hasher.hexdigest() != source["sha256"]:
            job.target.unlink(missing_ok=True)
            fail(f"{job.source.name}: rebuilt file does not match the recorded SHA-256")
        return source["size"]

    run_pipeline(jobs, in_bytes=biggest_payload, out_bytes=biggest_original + codec.Arena.PAGE, read=read, process=process, write=write, readers=readers,
                 writers=writers, gpus=gpus, pinned_budget=pinned_budget(pinned_gb), on_result=on_result)
