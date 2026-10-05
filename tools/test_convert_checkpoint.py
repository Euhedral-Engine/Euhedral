"""The converter's entry point and the recipes behind its four artifacts. Needs neither a GPU nor a checkpoint."""

import contextlib
import io
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

import convert_checkpoint  # noqa: E402
from euhedral_artifacts import device, edrl, inventory, pipeline, q3_p2e2, recipes  # noqa: E402
from euhedral_artifacts.sources import SourceRef  # noqa: E402


class FakeStore:
    """Answers shape lookups only: build_plans reads no weights, it only plans their conversion."""

    def ref(self, name, shape=None):
        return SourceRef("none", 0, shape if shape is not None else ())


def plans_of(recipe):
    return inventory.build_plans(FakeStore(), np.arange(inventory.DRAFT_ROWS, dtype=np.int64), recipe)


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = convert_checkpoint.main(list(argv))
        except SystemExit as exit_:
            code = exit_.code
    return code, out.getvalue(), err.getvalue()


BASE = ["--model", "/m", "--out", "/o.edrl", "--quantization", "q3", "--draft-ids-from", "/a.edrl"]


class ArgumentHandlingTest(unittest.TestCase):
    def test_each_choice_selects_one_of_four_recipes(self):
        seen = []
        with mock.patch.object(convert_checkpoint, "convert", lambda *args: seen.append(args[2]) or {}):
            for quantization in ("q3", "nvfp4"):
                for extra in ([], ["--compressed"]):
                    args = ["--model", "/m", "--out", "/o", "--quantization", quantization, "--ranking", "/r", "--device", "cpu"]
                    self.assertEqual(run(*args, *extra)[0], 0)
        self.assertEqual([(r.quantization, r.compressed) for r in seen],
                         [("q3", False), ("q3", True), ("nvfp4", False), ("nvfp4", True)])
        self.assertEqual([r.name for r in seen], ["q3", "q3-compressed", "nvfp4", "nvfp4-compressed"])

    def test_arguments_reach_the_pipeline(self):
        with mock.patch.object(convert_checkpoint, "convert") as convert:
            convert.return_value = {}
            self.assertEqual(run(*BASE, "--device", "cpu", "--jobs", "3", "--force")[0], 0)
        model, out, recipe, ranking, draft_ids, jobs, force, draft, draft_projections = convert.call_args.args
        self.assertEqual((model, out, ranking, draft_ids, jobs, force),
                         (Path("/m"), Path("/o.edrl"), None, Path("/a.edrl"), 3, True))
        self.assertEqual(recipe, recipes.Recipe("q3", False))
        self.assertEqual((draft, draft_projections), (None, "bf16"))

    def test_extend_adds_the_drafter_to_an_existing_artifact(self):
        with mock.patch.object(convert_checkpoint, "extend") as extend:
            extend.return_value = {}
            code = run("--extend", "/a.edrl", "--dflash2", "/d", "--dflash2-projections", "nvfp4", "--out", "/o.edrl",
                       "--device", "cpu")[0]
        self.assertEqual(code, 0)
        self.assertEqual(extend.call_args.args, (Path("/a.edrl"), Path("/d"), Path("/o.edrl"), "nvfp4", False))
        self.assertEqual(run("--extend", "/a.edrl", "--out", "/o.edrl")[0], 2, "--extend needs --dflash2")

    def test_default_jobs_follow_the_device(self):
        with mock.patch.object(convert_checkpoint, "convert") as convert, \
                mock.patch.object(convert_checkpoint.os, "cpu_count", return_value=12):
            convert.return_value = {}
            run(*BASE, "--device", "cuda")
            self.assertEqual(convert.call_args.args[5], convert_checkpoint.CUDA_JOBS)
            run(*BASE, "--device", "cpu")
            self.assertEqual(convert.call_args.args[5], 12)
        device.set_device("cpu")

    def test_required_arguments(self):
        for missing in ("--model", "--out", "--quantization"):
            argv = list(BASE)
            index = argv.index(missing)
            del argv[index:index + 2]
            self.assertEqual(run(*argv)[0], 2, missing)

    def test_quantization_is_q3_or_nvfp4(self):
        argv = list(BASE)
        argv[argv.index("q3")] = "q4"
        code, _, err = run(*argv)
        self.assertEqual(code, 2)
        self.assertIn("invalid choice", err)

    def test_draft_shortlist_source_is_exactly_one(self):
        neither = BASE[:-2]
        both = BASE + ["--ranking", "/r"]
        for argv in (neither, both):
            code, _, err = run(*argv)
            self.assertEqual(code, 2)
            self.assertIn("exactly one of --ranking and --draft-ids-from", err)

    def test_device_and_jobs_are_validated(self):
        self.assertEqual(run(*BASE, "--device", "tpu")[0], 2)
        self.assertEqual(run(*BASE, "--device", "cpu", "--jobs", "0")[0], 2)

    def test_internal_options_are_not_accepted(self):
        for option in (["--profile", "nvfp4"], ["--mtp-format", "nvfp4"], ["--reference", "/r"]):
            code, _, err = run(*BASE, *option)
            self.assertEqual(code, 2)
            self.assertIn("unrecognized arguments", err)

    def test_pipeline_errors_exit_with_a_message(self):
        with mock.patch.object(convert_checkpoint, "convert", side_effect=ValueError("bad checkpoint")):
            code, _, err = run(*BASE, "--device", "cpu")
        self.assertEqual(code, 2)
        self.assertIn("error: bad checkpoint", err)

    def test_help_lists_only_the_user_facing_options(self):
        code, out, _ = run("--help")
        self.assertEqual(code, 0)
        options = set(re.findall(r"^\s+(--[a-z0-9-]+)", out, re.M)) | set(re.findall(r"(?<=\[)--[a-z0-9-]+", out))
        self.assertEqual(options, {"--model", "--out", "--quantization", "--compressed", "--draft-ids-from",
                                   "--ranking", "--device", "--jobs", "--force", "--dflash2",
                                   "--dflash2-projections", "--extend"})
        for word in ("p2e2", "sd4", "nvmtp", "profile", "mtp-format", "reference"):
            self.assertNotIn(word, out.lower())


class RecipeTest(unittest.TestCase):
    def test_there_are_four_recipes(self):
        self.assertEqual([r.name for r in recipes.RECIPES], ["q3", "q3-compressed", "nvfp4", "nvfp4-compressed"])
        with self.assertRaises(ValueError):
            recipes.Recipe("q4")

    def test_storage_by_object(self):
        q3, q3c, nv, nvc = recipes.RECIPES
        rs, sd4 = recipes.LAYOUT_ROW_SPLIT, recipes.LAYOUT_SD4
        # (object, its format in the q3 artifact) -> storage per recipe: q3, q3 compressed, nvfp4, nvfp4 compressed
        cases = {
            ("text/token_embedding", "Q3G64_F16S"): [("Q3G64_F16S", rs)] * 4,
            ("text/layers/0/mlp/down", "Q3G64_F16S"): [("Q3G64_F16S", rs)] * 2 + [("NVFP4", rs), ("NVFP4", sd4)],
            ("text/layers/3/attention/query_key", "Q4G64_F16S"): [("Q4G64_F16S", rs)] * 2 + [("NVFP4", rs), ("NVFP4", sd4)],
            ("text/layers/3/attention/gate_value", "Q5G64_F16S"): [("Q5G64_F16S", rs)] * 2 + [("NVFP4", rs), ("NVFP4", sd4)],
            ("text/output_head", "Q3G64_F16S"): [("Q3G64_F16S", rs)] * 2 + [("NVFP4", rs), ("NVFP4", sd4)],
            ("text/draft_head", "Q3G64_F16S"): [("Q3G64_F16S", rs)] * 2 + [("NVFP4", rs)] * 2,
            ("mtp/layer/mlp/down", "NVFP4"): [("NVFP4", rs)] * 4,
            ("mtp/input_projection", "NVFP4"): [("NVFP4", rs)] * 4,
        }
        for (name, q3_format), expected in cases.items():
            for recipe, want in zip((q3, q3c, nv, nvc), expected):
                self.assertEqual(recipes.storage(recipe, name, q3_format), want, f"{recipe.name} {name}")


class InventoryTest(unittest.TestCase):
    def formats(self, recipe):
        counts = {}
        for plan in plans_of(recipe):
            counts[(plan.format_name, plan.layout)] = counts.get((plan.format_name, plan.layout), 0) + 1
        return counts

    def test_every_artifact_has_the_same_text_only_inventory(self):
        names = [plan.name for plan in plans_of(recipes.RECIPES[0])]
        self.assertEqual(len(names), inventory.OBJECT_COUNT)
        self.assertFalse([name for name in names if name.startswith("vision/")])
        for recipe in recipes.RECIPES:
            self.assertEqual([plan.name for plan in plans_of(recipe)], names)

    def test_q3_artifact(self):
        plans = {plan.name: plan for plan in plans_of(recipes.Recipe("q3"))}
        self.assertEqual(self.formats(recipes.Recipe("q3")), {
            ("Q3G64_F16S", "row-split-k128-v1"): 1 + 64 * 3 + 2,  # embedding, layers, output and draft heads
            ("Q4G64_F16S", "row-split-k128-v1"): 64,
            ("Q5G64_F16S", "row-split-k128-v1"): 64,
            ("NVFP4", "row-split-k128-v1"): 5,  # the MTP layer
            ("BF16", "contiguous-le-v1"): 360,
            ("FP32", "contiguous-le-v1"): 96,
            ("I32", "contiguous-le-v1"): 1,
        })
        self.assertEqual(plans["text/token_embedding"].format_name, "Q3G64_F16S")
        self.assertEqual(plans["text/draft_head"].format_name, "Q3G64_F16S")
        self.assertEqual(plans["text/output_head"].format_name, "Q3G64_F16S")
        mtp = [plan for plan in plans.values() if plan.name.startswith("mtp/") and len(plan.shape) == 2]
        self.assertEqual({plan.format_name for plan in mtp}, {"NVFP4"})
        self.assertEqual(len(mtp), 5)

    def test_nvfp4_artifact(self):
        plans = {plan.name: plan for plan in plans_of(recipes.Recipe("nvfp4"))}
        quantized = {name: plan for name, plan in plans.items() if plan.layout == "row-split-k128-v1"}
        self.assertEqual({plan.format_name for name, plan in quantized.items() if name != "text/token_embedding"}, {"NVFP4"})
        self.assertEqual(plans["text/token_embedding"].format_name, "Q3G64_F16S")
        self.assertEqual(len(quantized), 64 * 5 + 1 + 1 + 1 + 5)  # layers, embedding, output head, draft head, MTP

    def test_nvfp4_compressed_artifact(self):
        plans = {plan.name: plan for plan in plans_of(recipes.Recipe("nvfp4", True))}
        sd4 = {name for name, plan in plans.items() if plan.layout == "row-split-k128-sd4-v1"}
        self.assertEqual(len(sd4), 64 * 5 + 1)  # layers and the output head
        self.assertFalse([name for name in sd4 if name.startswith("mtp/") or name == "text/draft_head"])
        self.assertEqual({plans[name].format_name for name in sd4}, {"NVFP4"})
        self.assertEqual(plans["text/draft_head"].layout, "row-split-k128-v1")
        self.assertEqual(plans["text/draft_head"].format_name, "NVFP4")
        self.assertEqual(plans["text/token_embedding"].format_name, "Q3G64_F16S")

    def test_compressed_q3_builds_the_q3_inventory(self):
        plain = [(p.name, p.format_name, p.layout, p.byte_size) for p in plans_of(recipes.Recipe("q3"))]
        packed = [(p.name, p.format_name, p.layout, p.byte_size) for p in plans_of(recipes.Recipe("q3", True))]
        self.assertEqual(plain, packed)

    def test_plans_fit_one_table(self):
        for recipe in recipes.RECIPES:
            plans = plans_of(recipe)
            end = edrl.assign_offsets(plans, 4096)
            self.assertEqual(end, plans[-1].offset + plans[-1].byte_size)
            self.assertTrue(all(plan.offset % 256 == 0 for plan in plans))
            self.assertGreater(len(edrl.encode_table(plans)), 0)


def tiny_q3_artifact(path: Path) -> None:
    """A valid artifact with one Q3 tensor (4 x 1024), one Q4 tensor and one BF16 vector."""
    rng = np.random.default_rng(3)
    codes = rng.choice(np.arange(-3, 4, dtype=np.int8), size=(4, 1024), p=[0.02, 0.08, 0.23, 0.34, 0.23, 0.08, 0.02])
    q3 = bytearray(q3_p2e2.row_split_q3_size(4, 1024))
    planes = q3_p2e2.pack_q3(codes)
    q3[:len(planes)] = planes
    scales = edrl.align_up(4 * 16 * 24, 256)
    q3[scales:scales + 4 * 16 * 2] = rng.integers(0, 1 << 15, 4 * 16, dtype=np.uint16).astype("<u2").tobytes()
    q4 = bytes(range(256)) * 8
    vector = np.arange(8, dtype="<u2").tobytes()
    plans = [
        edrl.ObjectPlan("a", (4, 1024), "BF16", "Q3G64_F16S", "row-split-k128-v1", len(q3), lambda *_: None),
        edrl.ObjectPlan("b", (8, 128), "BF16", "Q4G64_F16S", "row-split-k128-v1", len(q4), lambda *_: None),
        edrl.ObjectPlan("c", (8,), "BF16", "BF16", "contiguous-le-v1", len(vector), lambda *_: None),
    ]
    metadata = b"metadata"
    table = edrl.encode_table(plans)
    data_base = edrl.HEADER_SIZE + len(metadata) + len(table)
    size = edrl.assign_offsets(plans, data_base)
    table = edrl.encode_table(plans)
    header = struct.pack(edrl.HEADER_FORMAT, edrl.MAGIC, edrl.VERSION, edrl.HEADER_SIZE, len(metadata),
                         edrl.HEADER_SIZE + len(metadata), len(plans), 0, data_base)
    blob = bytearray(size)
    blob[:edrl.HEADER_SIZE] = header
    blob[edrl.HEADER_SIZE:edrl.HEADER_SIZE + len(metadata)] = metadata
    blob[edrl.HEADER_SIZE + len(metadata):data_base] = table
    for plan, payload in zip(plans, (bytes(q3), q4, vector)):
        blob[plan.offset:plan.offset + len(payload)] = payload
    path.write_bytes(bytes(blob))


class PipelineTest(unittest.TestCase):
    def convert(self, recipe, directory, **kwargs):
        def write(model, config, output_path, recipe_, selected, jobs, draft=None, draft_projections="bf16"):
            tiny_q3_artifact(output_path)

        with mock.patch.object(pipeline, "write_artifact", write), \
                mock.patch.object(pipeline, "read_json", lambda path: {}), \
                mock.patch.object(pipeline, "check_checkpoint", lambda config: None), \
                mock.patch.object(pipeline, "draft_token_ids", lambda path: np.arange(4)):
            return pipeline.convert(Path("/model"), directory / "artifact.edrl", recipe,
                                    draft_ids_from=Path("/ids"), **kwargs)

    def test_q3_is_written_as_built(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.convert(recipes.Recipe("q3"), Path(tmp))
            self.assertEqual(manifest["artifact"], "q3")
            self.assertEqual(manifest["layout_counts"], {"row-split-k128-v1": 2, "contiguous-le-v1": 1})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["artifact.edrl", "artifact.edrl.manifest.json"])

    def test_compressed_q3_is_the_transcode_and_leaves_no_intermediate(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.convert(recipes.Recipe("q3", True), Path(tmp))
            self.assertEqual(manifest["artifact"], "q3-compressed")
            self.assertEqual(manifest["layout_counts"], {"row-split-p2e2-v1": 1, "row-split-k128-v1": 1, "contiguous-le-v1": 1})
            self.assertEqual(manifest["p2e2_tensor_count"], 1)
            self.assertEqual(manifest["round_trip"], "exact")
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["artifact.edrl", "artifact.edrl.manifest.json"])
            with (Path(tmp) / "artifact.edrl").open("rb") as handle:
                objects = edrl.read_table(handle)[2]
            self.assertEqual([o["layout"] for o in objects], [edrl.LAYOUT_P2E2, edrl.LAYOUT_ROW_SPLIT, 0])

    def test_intermediate_is_removed_when_conversion_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(q3_p2e2, "transcode", side_effect=ValueError("boom")):
                with self.assertRaises(ValueError):
                    self.convert(recipes.Recipe("q3", True), Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_existing_output_needs_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.convert(recipes.Recipe("q3"), Path(tmp))
            with self.assertRaisesRegex(ValueError, "--force"):
                self.convert(recipes.Recipe("q3"), Path(tmp))
            self.convert(recipes.Recipe("q3"), Path(tmp), force=True)

    def test_exactly_one_shortlist_source(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pipeline.convert(Path("/model"), Path("/nonexistent/o.edrl"), recipes.Recipe("q3"))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            pipeline.convert(Path("/model"), Path("/nonexistent/o.edrl"), recipes.Recipe("q3"),
                             ranking_path=Path("/r"), draft_ids_from=Path("/a"))


if __name__ == "__main__":
    unittest.main()
