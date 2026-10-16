"""Tests of the trellis (EXL3) path of the converter: validation, packing, the command line and the manifest."""

import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import convert_checkpoint  # noqa: E402
from euhedral_artifacts import edrl, inventory, pipeline, recipes, trellis  # noqa: E402
from euhedral_artifacts.edrl import ObjectPlan  # noqa: E402
from euhedral_artifacts.sources import SourceRef  # noqa: E402
from test_convert_checkpoint import tiny_q3_artifact  # noqa: E402

# A model small enough to carry real payloads, with the Swift layer structure: layer 3 is full attention.
SMALL = trellis.Dims(hidden=256, intermediate=256, layers=4, heads=2, kv_heads=1, head_dim=128, key_heads=1,
                     key_dim=128, value_heads=2, value_dim=128, full_attention=frozenset({3}))
QUANTIZATION = {"quant_method": "exl3", "version": "1.6.1", "bits": 3.5, "head_bits": 16,
                "calibration": {"rows": 8, "cols": 2048}, "out_scales": "always", "codebook": "mul1",
                "mtp_bits": 4}
RNG = np.random.default_rng(7)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], np.ndarray | None]]) -> None:
    """A safetensors file; a tensor whose data is None is declared (at its offset) but not written, so it must be
    last: the file ends before it."""
    header, offset = {}, 0
    for name, (dtype, shape, data) in tensors.items():
        size = (4 if dtype == "I32" else 2) * int(np.prod(shape, dtype=np.int64))
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)) + raw)
        for name, (_, _, data) in tensors.items():
            if data is None:
                break
            handle.write(data.tobytes())


def linear_tensors(linear: str, dims: trellis.Dims, words: int, mul1: int = trellis.MUL1,
                   with_data: bool = True) -> dict:
    n, k = trellis.linear_shape(linear, dims)
    data = (lambda array: array if with_data else None)
    return {
        linear + ".mul1": ("I32", (), np.asarray(mul1 & 0xFFFFFFFF, dtype="<u4").view("<i4").reshape(())),
        linear + ".suh": ("F16", (k,), RNG.normal(size=k).astype("<f2")),
        linear + ".svh": ("F16", (n,), RNG.normal(size=n).astype("<f2")),
        linear + ".trellis": ("I16", (k // 16, n // 16, words),
                              data(RNG.integers(-32768, 32768, (k // 16, n // 16, words)).astype("<i2"))),
    }


def all_linears(dims: trellis.Dims, mtp: bool = True) -> list[str]:
    linears = []
    for layer in range(dims.layers):
        p = f"{trellis.TEXT}layers.{layer}."
        if layer in dims.full_attention:
            linears += [p + "self_attn." + role for role in ("q_proj", "k_proj", "v_proj", "o_proj")]
        else:
            linears += [p + "linear_attn." + role for role in ("in_proj_qkv", "in_proj_z", "out_proj")]
        linears += [p + "mlp." + role for role in ("gate_proj", "up_proj", "down_proj")]
    if mtp:
        linears += ["mtp.fc"] + [trellis.MTP + "self_attn." + role for role in ("q_proj", "k_proj", "v_proj",
                                                                                 "o_proj")]
        linears += [trellis.MTP + "mlp." + role for role in ("gate_proj", "up_proj", "down_proj")]
    return linears


def build_checkpoint(directory: Path, dims: trellis.Dims = SMALL, words: dict[str, int] | None = None,
                     config: dict | None = None, with_data: bool = True, mul1: dict[str, int] | None = None,
                     storage: dict | None = None) -> None:
    """An EXL3 checkpoint directory: one shard per linear; `words` overrides a linear's tile width (default 56,
    the 3.5 bit layout; the MTP uses 64)."""
    quantization = dict(QUANTIZATION if config is None else config)
    weight_map = {}
    for linear in all_linears(dims, "mtp_bits" in quantization):
        shard = linear.replace(".", "-") + ".safetensors"
        default = 64 if linear.startswith("mtp") else 56
        tensors = linear_tensors(linear, dims, (words or {}).get(linear, default), (mul1 or {}).get(linear,
                                                                                                   trellis.MUL1),
                                 with_data)
        # the trellis tensor goes last, so a header-only file can end before it
        write_safetensors(directory / shard, tensors)
        weight_map.update({name: shard for name in tensors})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (directory / "config.json").write_text(json.dumps({"quantization_config": quantization}))
    if storage is not None:
        (directory / "quantization_config.json").write_text(json.dumps({**quantization, "tensor_storage": storage}))


def run_writers(plans: list[ObjectPlan]) -> dict[str, bytes]:
    """Every plan's payload bytes, written through its writer into a buffer."""
    result = {}
    for plan in plans:
        buffer = io.BytesIO()
        plan.writer(buffer, 0)
        result[plan.name] = buffer.getvalue()
        assert len(result[plan.name]) == plan.byte_size, plan.name
    return result


def planned(source: trellis.TrellisSource, name: str) -> tuple[list[ObjectPlan], dict[str, bytes]]:
    plans: list[ObjectPlan] = []
    source.add_plans(plans, name)
    return plans, run_writers(plans)


def stored_array(directory: Path, linear: str, suffix: str, dtype: str) -> np.ndarray:
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    base, header = trellis.read_safetensors_header(directory / index[linear + suffix])
    entry = header[linear + suffix]
    raw = (directory / index[linear + suffix]).read_bytes()[base + entry["data_offsets"][0]:
                                                            base + entry["data_offsets"][1]]
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


class PackingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        build_checkpoint(self.dir)
        self.source = trellis.TrellisSource(self.dir, SMALL)

    def tearDown(self):
        self.tmp.cleanup()

    def expected_tile(self, name: str, ti: int, tj: int) -> np.ndarray:
        """Tile (ti, tj) of object `name` from the source linears, one object row range at a time."""
        row = tj * 16
        for linear, first, rows in trellis.object_pieces(name, SMALL):
            if row < rows:
                return stored_array(self.dir, linear, ".trellis", "<i2")[ti, (first + row) // 16]
            row -= rows
        raise AssertionError("row beyond the object")

    def test_every_object_is_gathered_by_tile_column(self):
        for name in self.source.object_names():
            plans, written = planned(self.source, name)
            layout = self.source.layouts[name]
            payload = np.frombuffer(written[name], dtype="<i2").reshape(layout.payload_shape)
            for ti in range(layout.k // 16):
                for tj in range(layout.n // 16):
                    np.testing.assert_array_equal(payload[ti, tj], self.expected_tile(name, ti, tj), f"{name} {ti},{tj}")

    def test_the_four_tensors_of_an_object(self):
        name = "text/layers/0/mlp/gate_up"
        plans, written = planned(self.source, name)
        self.assertEqual([plan.name for plan in plans], [name, name + ".suh", name + ".svh", name + ".segments"])
        payload, suh, svh, segments = plans
        self.assertEqual((payload.shape, payload.source_dtype, payload.format_name, payload.layout),
                         ((16, 32, 56), "BF16", "TRELLIS", "trellis-tile16-v1"))
        self.assertEqual(payload.byte_size, 16 * 32 * 56 * 2)
        self.assertEqual((suh.shape, suh.source_dtype, suh.format_name, suh.layout),
                         ((2, 256), "FP16", "FP16", "contiguous-le-v1"))
        self.assertEqual((svh.shape, svh.source_dtype, svh.format_name), ((512,), "FP16", "FP16"))
        self.assertEqual((segments.shape, segments.source_dtype, segments.format_name), ((2,), "INT32", "I32"))
        gate, up = (f"{trellis.TEXT}layers.0.mlp.{role}" for role in ("gate_proj", "up_proj"))
        self.assertEqual(np.frombuffer(written[name + ".segments"], "<i4").tolist(), [256, 512])
        np.testing.assert_array_equal(np.frombuffer(written[name + ".suh"], "<f2").reshape(2, 256),
                                      np.stack([stored_array(self.dir, gate, ".suh", "<f2"),
                                                stored_array(self.dir, up, ".suh", "<f2")]))
        np.testing.assert_array_equal(
            np.frombuffer(written[name + ".svh"], "<f2"),
            np.concatenate([stored_array(self.dir, gate, ".svh", "<f2"), stored_array(self.dir, up, ".svh", "<f2")]))

    def test_single_segment_objects_keep_a_suh_row(self):
        for name in ("text/layers/0/gdn/query_key", "text/layers/1/mlp/down", "mtp/input_projection"):
            plans, written = planned(self.source, name)
            layout = self.source.layouts[name]
            self.assertEqual(plans[1].shape, (1, layout.k), name)
            self.assertEqual(np.frombuffer(written[name + ".segments"], "<i4").tolist(), [layout.n], name)

    def test_attention_heads_interleave_and_the_query_signs_repeat_for_the_gate(self):
        # q_proj holds per head [query 128 rows; gate 128 rows]: query_key = 2 query chunks, then k_proj.
        name = "text/layers/3/attention/query_key"
        _, written = planned(self.source, name)
        p = f"{trellis.TEXT}layers.3.self_attn."
        self.assertEqual(np.frombuffer(written[name + ".segments"], "<i4").tolist(), [256, 384])
        q_svh = stored_array(self.dir, p + "q_proj", ".svh", "<f2")
        svh = np.frombuffer(written[name + ".svh"], "<f2")
        np.testing.assert_array_equal(svh[:128], q_svh[0:128])
        np.testing.assert_array_equal(svh[128:256], q_svh[256:384])
        np.testing.assert_array_equal(svh[256:], stored_array(self.dir, p + "k_proj", ".svh", "<f2"))
        gate_name = "text/layers/3/attention/gate_value"
        _, gate = planned(self.source, gate_name)
        np.testing.assert_array_equal(np.frombuffer(gate[gate_name + ".svh"], "<f2")[:128], q_svh[128:256])

    def test_the_mtp_query_key_gate_value_layout(self):
        name = "mtp/layer/attention/query_key_gate_value"
        plans, written = planned(self.source, name)
        p = trellis.MTP + "self_attn."
        self.assertEqual(np.frombuffer(written[name + ".segments"], "<i4").tolist(), [256, 384, 640, 768])
        self.assertEqual(plans[1].shape, (4, 256))
        suh = np.frombuffer(written[name + ".suh"], "<f2").reshape(4, 256)
        q, k, v = (stored_array(self.dir, p + role, ".suh", "<f2") for role in ("q_proj", "k_proj", "v_proj"))
        np.testing.assert_array_equal(suh, np.stack([q, k, q, v]))
        self.assertEqual(plans[0].shape, (16, 48, 64))
        self.assertEqual(plans[0].byte_size, 16 * 48 * 64 * 2)

    def test_a_written_artifact_has_the_companion_contract_the_loader_checks(self):
        plans: list[ObjectPlan] = []
        for name in self.source.object_names():
            self.source.add_plans(plans, name)
        metadata = b"metadata"
        data_base = edrl.HEADER_SIZE + len(metadata) + len(edrl.encode_table(plans))
        size = edrl.assign_offsets(plans, data_base)
        table = edrl.encode_table(plans)
        path = Path(self.tmp.name) / "artifact.edrl"
        with path.open("w+b") as output:
            output.truncate(size)
            output.write(struct.pack(edrl.HEADER_FORMAT, edrl.MAGIC, edrl.VERSION, edrl.HEADER_SIZE, len(metadata),
                                     edrl.HEADER_SIZE + len(metadata), len(plans), 0, data_base) + metadata + table)
            with contextlib.redirect_stdout(io.StringIO()):
                pipeline.write_objects(path, output, plans, 1)
        with path.open("rb") as handle:
            _, _, objects = edrl.read_table(handle)
            by_name = {obj["name"]: obj for obj in objects}
            self.assertEqual(len(objects), 25 * 4)
            for name in self.source.object_names():
                obj = by_name[name]
                k16, n16, w = obj["shape"]
                self.assertEqual((obj["dtype"], obj["format"], obj["layout"]), (0, 8, 5), name)
                self.assertEqual(obj["bytes"], k16 * n16 * w * 2, name)
                suh, svh, segments = (by_name[name + part] for part in (".suh", ".svh", ".segments"))
                for companion, ordinals in ((suh, (1, 1, 0)), (svh, (1, 1, 0)), (segments, (6, 3, 0))):
                    self.assertEqual((companion["dtype"], companion["format"], companion["layout"]), ordinals)
                self.assertEqual((suh["shape"][1], svh["shape"], segments["shape"]), (k16 * 16, (n16 * 16,),
                                                                                     (suh["shape"][0],)), name)
                handle.seek(segments["offset"])
                ends = struct.unpack(f"<{segments['shape'][0]}i", handle.read(segments["bytes"]))
                self.assertEqual(ends[-1], n16 * 16)
                self.assertTrue(all(end % 128 == 0 for end in ends))
                self.assertEqual(list(ends), sorted(set(ends)))
                self.assertLessEqual(len(ends), 8)
            handle.seek(by_name["mtp/input_projection"]["offset"])
            expected = self.source.payload(self.source.layouts["mtp/input_projection"]).tobytes()
            self.assertEqual(handle.read(len(expected)), expected)

    def test_the_inventory_lists_every_linear_object(self):
        self.assertEqual(len(self.source.object_names()), 4 * 5 + 5)
        self.assertTrue(self.source.has_mtp)
        self.assertEqual(self.source.bits, 3.5)

    def test_files_are_checked_for_their_size(self):
        self.source.check_files()
        shard = self.dir / "mtp-fc.safetensors"
        shard.write_bytes(shard.read_bytes()[:-10])
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.source.check_files()

    def test_the_source_shapes_are_compared(self):
        self.source.check_sources(lambda name: trellis.linear_shape(name[:-len(".weight")], SMALL))
        with self.assertRaisesRegex(ValueError, "mtp.fc.*BF16 source"):
            self.source.check_sources(lambda name: (256, 256) if name == "mtp.fc.weight"
                                      else trellis.linear_shape(name[:-len(".weight")], SMALL))

    def test_a_checkpoint_without_mtp_has_no_mtp_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {key: value for key, value in QUANTIZATION.items() if key != "mtp_bits"}
            build_checkpoint(Path(tmp), config=config)
            source = trellis.TrellisSource(Path(tmp), SMALL)
        self.assertFalse(source.has_mtp)
        self.assertEqual(len(source.object_names()), 20)
        self.assertFalse(source.carries("mtp/input_projection"))


class RefusalTest(unittest.TestCase):
    def refused(self, pattern: str, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            build_checkpoint(Path(tmp), **kwargs)
            with self.assertRaisesRegex(ValueError, pattern):
                trellis.TrellisSource(Path(tmp), kwargs.get("dims", SMALL))

    def test_each_quantization_config_field(self):
        for key, value in (("codebook", "mcg"), ("out_scales", "never"), ("version", "1.7.0"),
                           ("quant_method", "gptq")):
            self.refused(key, config={**QUANTIZATION, key: value})

    def test_a_missing_quantization_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_checkpoint(Path(tmp))
            (Path(tmp) / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "not an EXL3 checkpoint"):
                trellis.TrellisSource(Path(tmp), SMALL)

    def test_a_linear_with_another_multiplier(self):
        linear = f"{trellis.TEXT}layers.2.mlp.down_proj"
        self.refused(f"{linear}.mul1", mul1={linear: 0x83DCD12E})

    def test_bits_that_disagree_with_the_tensor(self):
        linear = f"{trellis.TEXT}layers.1.mlp.up_proj"
        storage = {linear: {"bits_per_weight": 3.0}}
        self.refused(f"{linear}.*3.0 bits per weight.*56 words", storage=storage)

    def test_an_unsupported_width(self):
        linear = f"{trellis.TEXT}layers.0.mlp.gate_proj"
        for w in (8, 12, 136, 144):
            self.refused("supported width", words={linear: w})

    def test_a_trellis_tensor_of_the_wrong_shape(self):
        # a layer-0 MLP linear stored with another n: the tensors disagree with the linear
        dims = trellis.Dims(**{**SMALL.__dict__, "intermediate": 384})
        with tempfile.TemporaryDirectory() as tmp:
            build_checkpoint(Path(tmp), dims=dims)
            with self.assertRaisesRegex(ValueError, "needs \\[16, "):
                trellis.TrellisSource(Path(tmp), SMALL)

    def test_a_missing_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_checkpoint(Path(tmp))
            index_path = Path(tmp) / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            del index["weight_map"][f"{trellis.TEXT}layers.0.mlp.up_proj.svh"]
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "layers.0.mlp.up_proj.svh"):
                trellis.TrellisSource(Path(tmp), SMALL)

    def test_pieces_of_one_object_with_different_bits(self):
        k_proj = f"{trellis.TEXT}layers.3.self_attn.k_proj"
        self.refused("text/layers/3/attention/query_key: its pieces have different bit widths .*3.5", words={k_proj: 64})
        up = f"{trellis.TEXT}layers.0.mlp.up_proj"
        self.refused("text/layers/0/mlp/gate_up.*different bit widths", words={up: 48})

    def test_a_piece_that_is_not_aligned_to_128_rows(self):
        dims = trellis.Dims(**{**SMALL.__dict__, "head_dim": 64, "heads": 2, "kv_heads": 2})
        self.refused("not aligned to 128 rows", dims=dims)

    def test_a_missing_head_codes_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "missing head codes in .*text.output_head.*text.draft_head"):
                trellis.require_head_codes(Path(tmp), None)
            (Path(tmp) / "text.output_head.npz").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "missing head codes.*text.draft_head"):
                trellis.require_head_codes(Path(tmp), None)
            (Path(tmp) / "text.output_head.npz").unlink()
            with self.assertRaisesRegex(ValueError, "missing head codes in [^:]*: text/draft_head"):
                trellis.require_head_codes(Path(tmp), "BF16")  # a BF16 LM head needs no codes, the draft head does
            (Path(tmp) / "text.draft_head.npz").write_bytes(b"")
            trellis.require_head_codes(Path(tmp), "BF16")
            with self.assertRaisesRegex(ValueError, "missing head codes: pass --codes-from"):
                trellis.require_head_codes(None, None)


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = convert_checkpoint.main(list(argv))
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


BASE = ["--model", "/m", "--out", "/o.edrl", "--draft-ids-from", "/a.edrl", "--device", "cpu"]


class CommandLineTest(unittest.TestCase):
    def test_trellis_from_needs_the_trellis_quantization(self):
        for quantization in ("q3", "nvfp4", "bf16"):
            code, _, err = run(*BASE, "--quantization", quantization, "--trellis-from", "/x")
            self.assertEqual(code, 2, quantization)
            self.assertIn("--trellis-from is required with, and only valid for, --quantization trellis", err)

    def test_the_trellis_quantization_needs_trellis_from(self):
        code, _, err = run(*BASE, "--quantization", "trellis")
        self.assertEqual(code, 2)
        self.assertIn("--trellis-from is required", err)

    def test_arguments_reach_the_pipeline(self):
        with mock.patch.object(convert_checkpoint, "convert") as convert:
            convert.return_value = {}
            code = run(*BASE, "--quantization", "trellis", "--trellis-from", "/x", "--codes-from", "/heads",
                       "--output-head", "BF16")[0]
        self.assertEqual(code, 0)
        self.assertEqual(convert.call_args.args[2], recipes.Recipe("trellis", False, "BF16"))
        self.assertEqual(convert.call_args.args[-1], Path("/heads"))
        self.assertEqual(convert.call_args.kwargs, {"trellis_from": Path("/x")})

    def test_the_trellis_recipe_has_no_compressed_form(self):
        with self.assertRaisesRegex(ValueError, "no compressed form"):
            recipes.Recipe("trellis", True)

    def test_a_bf16_lm_head_is_stored_contiguous(self):
        for quantization in ("q3", "nvfp4", "trellis"):
            self.assertEqual(recipes.storage(recipes.Recipe(quantization, False, "BF16"), "text/output_head", "NVFP4"),
                             ("BF16", "contiguous-le-v1"))

    def test_the_trellis_recipe_storage(self):
        recipe = recipes.Recipe("trellis")
        self.assertEqual(recipes.storage(recipe, "text/token_embedding", "Q3G64_F16S"),
                         ("Q3G64_F16S", "row-split-k128-v1"))
        self.assertEqual(recipes.storage(recipe, "text/output_head", "NVFP4"), ("NVFP4", "row-split-k128-v1"))
        self.assertEqual(recipes.storage(recipe, "text/draft_head", "Q3G64_F16S"), ("NVFP4", "row-split-k128-v1"))
        self.assertEqual(recipes.storage(recipe, "text/layers/0/mlp/down", "Q3G64_F16S"),
                         ("TRELLIS", "trellis-tile16-v1"))


class FakeStore:
    def __init__(self, *_):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def ref(self, name, shape=None):
        return SourceRef("none", 0, shape if shape is not None else ())


class InventoryTest(unittest.TestCase):
    """The Swift-1.5 inventory over a header-only EXL3 checkpoint (no payload bytes, never read)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        build_checkpoint(cls.dir, dims=trellis.Dims(), with_data=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def plans(self, mtp=True):
        source = trellis.TrellisSource(self.dir)
        if not mtp:
            source.has_mtp = False
            source.layouts = {name: layout for name, layout in source.layouts.items() if not name.startswith("mtp/")}

        def add_quant(plans, recipe, name, matrix, q3_format, *_):
            format_name, layout = recipes.storage(recipe, name, q3_format)
            plans.append(ObjectPlan(name, matrix.shape, "BF16", format_name, layout, 0, lambda *_: None))

        with tempfile.TemporaryDirectory() as heads, mock.patch.object(inventory, "add_quant", add_quant):
            for name in trellis.HEAD_NAMES:
                (Path(heads) / (name.replace("/", ".") + ".npz")).write_bytes(b"")
            return inventory.build_plans(FakeStore(), np.arange(inventory.DRAFT_ROWS, dtype=np.int64),
                                         recipes.Recipe("trellis"), codes_from=Path(heads), trellis=source)

    def test_every_linear_object_is_followed_by_its_companions(self):
        plans = self.plans()
        self.assertEqual(len(plans), inventory.OBJECT_COUNT + 3 * 325)
        by_name = {plan.name: plan for plan in plans}
        self.assertEqual(len(by_name), len(plans))
        formats = {}
        for plan in plans:
            formats[plan.format_name] = formats.get(plan.format_name, 0) + 1
        self.assertEqual(formats["TRELLIS"], 325)
        self.assertEqual(formats["FP16"] + formats["I32"] - 1, 3 * 325)  # the shortlist is the one other I32
        index = [plan.name for plan in plans].index("text/layers/5/gdn/value_z")
        self.assertEqual([plan.name for plan in plans[index:index + 4]],
                         ["text/layers/5/gdn/value_z", "text/layers/5/gdn/value_z.suh",
                          "text/layers/5/gdn/value_z.svh", "text/layers/5/gdn/value_z.segments"])
        self.assertEqual(by_name["text/layers/5/gdn/value_z"].shape, (320, 768, 56))
        self.assertEqual(by_name["text/layers/5/gdn/value_z.suh"].shape, (2, 5120))
        self.assertEqual(by_name["mtp/layer/attention/query_key_gate_value.suh"].shape, (4, 5120))
        self.assertEqual(by_name["mtp/layer/attention/query_key_gate_value"].shape, (320, 896, 64))
        self.assertEqual(by_name["text/layers/3/attention/query_key"].shape, (320, 448, 56))

    def test_what_the_checkpoint_does_not_hold_comes_from_the_source(self):
        by_name = {plan.name: plan for plan in self.plans()}
        self.assertEqual(by_name["text/token_embedding"].format_name, "Q3G64_F16S")
        self.assertEqual(by_name["text/output_head"].format_name, "NVFP4")
        self.assertEqual(by_name["text/draft_head"].format_name, "NVFP4")
        self.assertEqual(by_name["text/layers/0/gdn/a_projection"].format_name, "BF16")
        self.assertEqual(by_name["text/layers/0/gdn/b_projection"].format_name, "BF16")

    def test_the_mtp_layer_is_bf16_when_the_checkpoint_has_none(self):
        plans = self.plans(mtp=False)
        self.assertEqual(len(plans), inventory.OBJECT_COUNT + 3 * 320)
        by_name = {plan.name: plan for plan in plans}
        for name in trellis.MTP_OBJECTS:
            self.assertEqual((by_name[name].format_name, by_name[name].layout), ("BF16", "contiguous-le-v1"), name)

    def test_the_recipe_and_the_checkpoint_go_together(self):
        source = trellis.TrellisSource(self.dir)
        with self.assertRaisesRegex(ValueError, "needs an EXL3 checkpoint"):
            inventory.build_plans(FakeStore(), np.arange(inventory.DRAFT_ROWS), recipes.Recipe("trellis"))
        with self.assertRaisesRegex(ValueError, "needs an EXL3 checkpoint"):
            inventory.build_plans(FakeStore(), np.arange(inventory.DRAFT_ROWS), recipes.Recipe("q3"), trellis=source)

    def test_plans_fit_one_table(self):
        plans = self.plans()
        end = edrl.assign_offsets(plans, 4096)
        self.assertEqual(end, plans[-1].offset + plans[-1].byte_size)
        self.assertGreater(len(edrl.encode_table(plans)), 0)


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root / "exl3"
        self.checkpoint.mkdir()
        build_checkpoint(self.checkpoint)
        self.source = trellis.TrellisSource(self.checkpoint, SMALL)
        self.written = []

    def tearDown(self):
        self.tmp.cleanup()

    def convert(self, codes_from, **kwargs):
        def write(model, config, output_path, recipe, selected, jobs, draft=None, draft_projections="bf16",
                  importance=None, codes_from=None, trellis=None):
            self.written.append(output_path)
            tiny_q3_artifact(output_path)

        with mock.patch.object(pipeline, "write_artifact", write), \
                mock.patch.object(pipeline, "read_json", lambda path: {}), \
                mock.patch.object(pipeline, "check_checkpoint", lambda config: None), \
                mock.patch.object(pipeline, "draft_token_ids", lambda path: np.arange(4)), \
                mock.patch.object(pipeline, "SourceStore", FakeStore), \
                mock.patch.object(pipeline, "TrellisSource", lambda directory: self.source), \
                mock.patch.object(trellis.TrellisSource, "check_sources", lambda self, shape_of: None):
            return pipeline.convert(Path("/model"), self.root / "artifact.edrl", recipes.Recipe("trellis"),
                                    draft_ids_from=Path("/ids"), codes_from=codes_from,
                                    trellis_from=self.checkpoint, **kwargs)

    def test_the_manifest_records_the_checkpoint(self):
        heads = self.root / "heads"
        heads.mkdir()
        for name in trellis.HEAD_NAMES:
            (heads / (name.replace("/", ".") + ".npz")).write_bytes(b"")
        manifest = self.convert(heads)
        self.assertEqual(manifest["artifact"], "trellis")
        self.assertEqual(manifest["trellis_from"], str(self.checkpoint))
        self.assertEqual(manifest["trellis_bits"], 3.5)
        self.assertEqual(manifest["quantization_config"], QUANTIZATION)
        self.assertEqual(manifest["codes_from_objects"], 2)
        on_disk = json.loads(Path(f"{self.root / 'artifact.edrl'}.manifest.json").read_text())
        self.assertEqual(on_disk["quantization_config"], QUANTIZATION)

    def test_missing_head_codes_stop_the_conversion_before_anything_is_written(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "missing head codes in .*text.output_head"):
            self.convert(empty)
        with self.assertRaisesRegex(ValueError, "missing head codes: pass --codes-from"):
            self.convert(None)
        self.assertEqual(self.written, [])
        self.assertEqual(list(self.root.glob("artifact*")), [])

    def test_trellis_from_is_refused_by_other_recipes(self):
        with mock.patch.object(pipeline, "read_json", lambda path: {}):
            with self.assertRaisesRegex(ValueError, "only valid for"):
                pipeline.convert(Path("/model"), self.root / "x.edrl", recipes.Recipe("q3"),
                                 draft_ids_from=Path("/ids"), trellis_from=self.checkpoint)
            with self.assertRaisesRegex(ValueError, "required with"):
                pipeline.convert(Path("/model"), self.root / "x.edrl", recipes.Recipe("trellis"),
                                 draft_ids_from=Path("/ids"))


if __name__ == "__main__":
    unittest.main()
