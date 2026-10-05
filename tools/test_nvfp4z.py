from __future__ import annotations

from pathlib import Path
import json
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import nvfp4  # noqa: E402
from nvfp4z import container  # noqa: E402

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:  # the CPU tests need only NumPy
    torch = None
    CUDA = False

if CUDA:
    from nvfp4z import codec, pipeline  # noqa: E402


def safetensors_bytes(tensors: dict[str, tuple[str, list[int], bytes]]) -> bytes:
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"".join(data for _, _, data in tensors.values())


def nvfp4_tensors(rng: np.random.Generator, shape: tuple[int, ...], sigma: float = 0.02) -> tuple[bytes, bytes]:
    """Packed codes and E4M3 scale bytes of normal weights, quantized with the converter's NVFP4 quantizer."""
    values = rng.normal(0, sigma, size=(int(np.prod(shape[:-1])), shape[-1])).astype(np.float32)
    packed, scales = nvfp4.quantize_nvfp4_rows(values, nvfp4.nvfp4_global_scale(float(np.abs(values).max())))
    return packed.tobytes(), scales.tobytes()


class PlanTest(unittest.TestCase):
    def header(self):
        return {
            "a.weight_packed": {"dtype": "U8", "shape": [4, 8], "data_offsets": [0, 32]},
            "a.weight_scale": {"dtype": "F8_E4M3", "shape": [4, 1], "data_offsets": [32, 36]},
            "a.weight_scale_2": {"dtype": "F32", "shape": [1], "data_offsets": [36, 40]},
            "b.weight": {"dtype": "U8", "shape": [2, 3, 16], "data_offsets": [40, 136]},
            "b.weight_scale": {"dtype": "F8_E4M3", "shape": [2, 3, 2], "data_offsets": [136, 148]},
            "c.weight_packed": {"dtype": "U8", "shape": [4, 8], "data_offsets": [148, 180]},
            "c.weight_scale": {"dtype": "F8_E4M3", "shape": [4, 2], "data_offsets": [180, 188]},
            "big": {"dtype": "BF16", "shape": [1 << 16], "data_offsets": [188, 188 + (2 << 16)]},
            "small": {"dtype": "BF16", "shape": [8], "data_offsets": [188 + (2 << 16), 204 + (2 << 16)]},
        }

    def test_kinds_follow_the_common_layouts(self):
        kinds = {item["name"]: item for item in container.plan(self.header())}
        self.assertEqual(kinds["a.weight_packed"]["kind"], "nvfp4")
        self.assertEqual(kinds["a.weight_packed"]["partner"], "a.weight_scale")
        self.assertEqual(kinds["a.weight_scale"]["kind"], "paired")
        self.assertEqual(kinds["a.weight_scale_2"]["kind"], "raw")
        self.assertEqual(kinds["b.weight"]["kind"], "nvfp4")
        self.assertEqual(kinds["b.weight"]["groups"], 2)
        self.assertEqual(kinds["b.weight"]["blocks"], 12)
        self.assertEqual(kinds["big"]["kind"], "bf16")
        self.assertEqual(kinds["small"]["kind"], "raw")

    def test_a_scale_of_the_wrong_shape_is_not_a_partner(self):
        kinds = {item["name"]: item["kind"] for item in container.plan(self.header())}
        self.assertEqual(kinds["c.weight_packed"], "raw")
        self.assertEqual(kinds["c.weight_scale"], "raw")

    def test_plan_is_in_file_order(self):
        offsets = [item["start"] for item in container.plan(self.header())]
        self.assertEqual(offsets, sorted(offsets))


class ContainerTest(unittest.TestCase):
    def test_header_block_is_page_aligned_and_round_trips(self):
        block = container.pack_header({"format": 1, "tensors": [{"name": "x" * 1000}]})
        self.assertEqual(len(block) % container.ALIGN, 0)
        meta, end = container.read_header(block + b"payload")
        self.assertEqual(end, len(block))
        self.assertEqual(meta["tensors"][0]["name"], "x" * 1000)
        self.assertEqual(container.header_block_size(block[:16]), len(block))

    def test_bad_magic_and_truncation_are_rejected(self):
        with self.assertRaises(ValueError):
            container.read_header(b"NOTNVFP4" + bytes(64))
        block = container.pack_header({"format": 1})
        with self.assertRaises(ValueError):
            container.read_header(block[:20])

    def test_safetensors_parse(self):
        image = safetensors_bytes({"t": ("U8", [4], b"abcd")})
        start, raw, header = container.parse_safetensors(image)
        self.assertEqual(image[start:], b"abcd")
        self.assertEqual(header["t"]["data_offsets"], [0, 4])


@unittest.skipUnless(CUDA, "needs a CUDA device")
class CodecTest(unittest.TestCase):
    def roundtrip(self, packed: torch.Tensor, scale: torch.Tensor, groups: int, segment_blocks: int | None = None):
        blocks = scale.numel()
        arena = codec.Arena(4 * blocks * 9 + (1 << 20))
        pk, sc = packed.reshape(-1).cuda(), scale.reshape(-1).cuda()
        step = segment_blocks or blocks
        records = [codec.encode_blocks(pk, sc, b0, min(blocks, b0 + step), blocks, groups, arena) for b0 in range(0, blocks, step)]
        torch.cuda.synchronize()
        out_pk, out_sc = torch.empty_like(pk), torch.empty_like(sc)
        for record in records:
            b0, b1 = record["b0"], record["b1"]
            codec.decode_blocks(record, arena, out_pk[8 * b0:8 * b1], out_sc[b0:b1], blocks, groups)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out_pk, pk))
        self.assertTrue(torch.equal(out_sc, sc))
        return records, arena.used

    def random_blocks(self, blocks: int, seed: int, negative_zero: bool = False, max_scale: int = 127):
        generator = torch.Generator().manual_seed(seed)
        scale = torch.randint(0, max_scale, (blocks,), dtype=torch.uint8, generator=generator)
        magnitude = (torch.rand(blocks, 16, generator=generator) ** 3 * 8).long().clamp(max=7)
        magnitude[:, 0] = 7
        sign = (torch.rand(blocks, 16, generator=generator) < 0.5).long() * ((magnitude > 0) | negative_zero)
        codes = (magnitude | (sign << 3)).to(torch.uint8)
        return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous(), scale

    def test_sizes_from_one_block_to_several_lanes_rows(self):
        for blocks, groups in [(1, 1), (7, 1), (31, 1), (1000, 4), (5000, 3), (70000, 7)]:
            with self.subTest(blocks=blocks, groups=groups):
                self.roundtrip(*self.random_blocks(blocks, blocks), groups)

    def test_segments_are_independent(self):
        records, _ = self.roundtrip(*self.random_blocks(9000, 5), 3, segment_blocks=2048)
        self.assertEqual(len(records), 5)

    def test_negative_zero_codes_are_kept(self):
        records, _ = self.roundtrip(*self.random_blocks(3000, 6, negative_zero=True), 1)
        self.assertEqual(records[0]["sign_all"], 1)

    def test_scale_bytes_beyond_the_positive_range_are_stored_raw(self):
        packed, scale = self.random_blocks(500, 7)
        scale[3] = 200
        records, _ = self.roundtrip(packed, scale, 1)
        self.assertEqual(records[0]["mode"], "raw")

    def test_incompressible_data_is_never_stored_larger_than_raw(self):
        generator = torch.Generator().manual_seed(8)
        packed = torch.randint(0, 256, (4000, 8), dtype=torch.uint8, generator=generator)
        scale = torch.randint(0, 127, (4000,), dtype=torch.uint8, generator=generator)
        _, used = self.roundtrip(packed, scale, 1)
        self.assertLessEqual(used, 9 * 4000 + 64)

    def test_nvfp4_weights_compress(self):
        rng = np.random.default_rng(1)
        packed, scales = nvfp4_tensors(rng, (4096, 1024))
        records, used = self.roundtrip(torch.frombuffer(bytearray(packed), dtype=torch.uint8).clone(),
                                       torch.frombuffer(bytearray(scales), dtype=torch.uint8).clone(), 4)
        self.assertEqual(records[0]["mode"], "rans")
        self.assertLess(used, 0.95 * (len(packed) + len(scales)))

    def test_byte_streams(self):
        for data in (torch.zeros(1, dtype=torch.uint8), torch.full((100000,), 3, dtype=torch.uint8),
                     torch.randint(0, 256, (70001,), dtype=torch.uint8), (torch.rand(300000) ** 4 * 40).to(torch.uint8)):
            with self.subTest(n=data.numel()):
                arena = codec.Arena(4 * data.numel() + (1 << 20))
                record = codec.encode_bytes(data.cuda(), arena)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(codec.decode_bytes(record, arena, data.numel()).cpu(), data))

    def test_corrupt_tables_are_rejected(self):
        arena = codec.Arena(1 << 22)
        packed, scale = nvfp4_tensors(np.random.default_rng(2), (512, 256))
        pk = torch.frombuffer(bytearray(packed), dtype=torch.uint8).clone().cuda()
        sc = torch.frombuffer(bytearray(scales := scale), dtype=torch.uint8).clone().cuda()
        record = codec.encode_blocks(pk, sc, 0, sc.numel(), sc.numel(), 1, arena)
        torch.cuda.synchronize()
        table = arena.piece(record["freq_mag"])
        table[0] ^= 1
        with self.assertRaises(ValueError):
            codec.decode_blocks(record, arena, torch.empty_like(pk), torch.empty_like(sc), sc.numel(), 1)


@unittest.skipUnless(CUDA, "needs a CUDA device")
class PipelineTest(unittest.TestCase):
    def shard(self) -> bytes:
        rng = np.random.default_rng(3)
        packed, scales = nvfp4_tensors(rng, (6 * 64, 512))
        small_packed, small_scales = nvfp4_tensors(rng, (16, 256))
        bf16 = (rng.normal(0, 0.02, 100000).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        return safetensors_bytes({
            "experts.weight_packed": ("U8", [6, 64, 256], packed),
            "experts.weight_scale": ("F8_E4M3", [6, 64, 32], scales),
            "experts.weight_scale_2": ("F32", [6], np.arange(6, dtype=np.float32).tobytes()),
            "dense.weight_packed": ("U8", [16, 128], small_packed),
            "dense.weight_scale": ("F8_E4M3", [16, 16], small_scales),
            "norm.weight": ("BF16", [100000], bf16),
            "bias": ("BF16", [8], bytes(16)),
        })

    def run_directory(self, directory: Path, **options):
        source, packed, restored = directory / "src", directory / "z", directory / "out"
        source.mkdir()
        (source / "model-00001-of-00002.safetensors").write_bytes(self.shard())
        (source / "model-00002-of-00002.safetensors").write_bytes(self.shard()[:-16] + bytes(range(16)))
        jobs = [pipeline.Job(path, packed / (path.name + pipeline.EXTENSION)) for path in sorted(source.glob("*.safetensors"))]
        packed.mkdir()
        pipeline.compress_files(jobs, pinned_gb=0.25, **options)
        back = [pipeline.Job(job.target, restored / job.source.name) for job in jobs]
        restored.mkdir()
        pipeline.decompress_files(back, pinned_gb=0.25, **options)
        return source, packed, restored

    def test_shards_round_trip_byte_for_byte(self):
        for options in ({}, {"readers": 1, "writers": 1, "gpus": 1, "fsync": False, "direct_io": "off"}, {"direct_io": "on"}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as tmp:
                source, packed, restored = self.run_directory(Path(tmp), **options)
                for path in sorted(source.glob("*.safetensors")):
                    self.assertEqual((restored / path.name).read_bytes(), path.read_bytes())
                    self.assertLess((packed / (path.name + pipeline.EXTENSION)).stat().st_size, path.stat().st_size)

    def test_a_flipped_payload_bit_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, packed, restored = self.run_directory(Path(tmp))
            target = packed / "model-00001-of-00002.safetensors.nvfp4z"
            data = bytearray(target.read_bytes())
            meta, block = container.read_header(bytes(data))
            data[block + meta["payload_bytes"] // 2] ^= 0x10
            target.write_bytes(bytes(data))
            (restored / "model-00001-of-00002.safetensors").unlink()
            with self.assertRaises(ValueError):
                pipeline.decompress_files([pipeline.Job(target, restored / "model-00001-of-00002.safetensors")], pinned_gb=0.25)
            self.assertFalse((restored / "model-00001-of-00002.safetensors").exists())

    def test_shards_with_gaps_are_rejected(self):
        image = bytearray(self.shard() + b"\x00" * 8)
        arena = codec.Arena(len(image) * 2)
        with self.assertRaises(ValueError):
            pipeline.compress_image(np.frombuffer(image, dtype=np.uint8), "x", "0" * 64, arena)


if __name__ == "__main__":
    unittest.main()
