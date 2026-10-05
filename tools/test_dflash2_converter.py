import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from euhedral_artifacts import dflash2, edrl  # noqa: E402

HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM, VOCAB = 256, 512, 4, 2, 64, 300
CONFIG = {
    "architectures": ["DFlash2DraftModel"], "num_hidden_layers": 2, "hidden_size": HIDDEN,
    "intermediate_size": INTERMEDIATE, "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM, "vocab_size": VOCAB, "sliding_window": 64, "rms_norm_eps": 1e-6,
    "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
    "dflash_config": {"block_size": 8, "mask_token_id": 299, "conv_kernel_size": 2, "conv_group_size": 16,
                      "selector_top_k": 16, "selector_rank": 32, "target_layer_ids": [1, 3]},
}


def tensors(rng):
    out = {"fc.weight": (HIDDEN, 2 * HIDDEN), "hidden_norm.weight": (HIDDEN,), "norm.weight": (HIDDEN,),
           "candidate_selector.hidden_projection.weight": (32, HIDDEN),
           "candidate_selector.predecessor_codebook": (VOCAB, 32),
           "candidate_selector.successor_codebook": (VOCAB, 32)}
    for layer in range(2):
        p = f"layers.{layer}."
        out.update({p + "input_layernorm.weight": (HIDDEN,), p + "post_attention_layernorm.weight": (HIDDEN,),
                    p + "self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
                    p + "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
                    p + "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
                    p + "self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
                    p + "self_attn.q_norm.weight": (HEAD_DIM,), p + "self_attn.k_norm.weight": (HEAD_DIM,),
                    p + "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN), p + "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
                    p + "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE)})
        for conv in ("attention_conv", "mlp_conv"):
            out[p + conv + ".base_kernel"] = (2, 2, HIDDEN)
            out[p + conv + ".kernel_projection.weight"] = (2 * 2 * HIDDEN // 16, HIDDEN)
    return {name: rng.integers(0, 1 << 16, size=shape, dtype=np.uint16) for name, shape in out.items()}


def write_safetensors(path: Path, values) -> None:
    header, offset, blobs = {}, 0, []
    for name, words in values.items():
        data = words.astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(words.shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def write_artifact(path: Path, payloads) -> None:
    metadata = b"\x01\x02\x03"
    plans = [edrl.ObjectPlan(name, (len(data) // 2,), "BF16", "BF16", "contiguous-le-v1", len(data),
                             lambda output, offset, data=data: (output.seek(offset), output.write(data)))
             for name, data in payloads.items()]
    base = edrl.HEADER_SIZE + len(metadata) + len(edrl.encode_table(plans))
    size = edrl.assign_offsets(plans, base)
    with path.open("w+b") as output:
        output.truncate(size)
        output.write(struct.pack(edrl.HEADER_FORMAT, edrl.MAGIC, edrl.VERSION, edrl.HEADER_SIZE, len(metadata),
                                 edrl.HEADER_SIZE + len(metadata), len(plans), 0, base))
        output.write(metadata)
        output.write(edrl.encode_table(plans))
        for plan in plans:
            plan.writer(output, plan.offset)


class DFlash2ConverterTest(unittest.TestCase):
    def test_extension_keeps_every_object_and_adds_the_drafter(self):
        rng = np.random.default_rng(7)
        values = tensors(rng)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "draft").mkdir()
            (root / "draft" / "config.json").write_text(json.dumps(CONFIG))
            write_safetensors(root / "draft" / "model.safetensors", values)
            base = {"text/a": bytes(rng.integers(0, 256, 1000, dtype=np.uint8)),
                    "text/b": bytes(rng.integers(0, 256, 4098, dtype=np.uint8))}
            write_artifact(root / "base.edrl", base)
            dflash2.extend(root / "base.edrl", root / "draft", root / "out.edrl", check=False)
            with (root / "out.edrl").open("rb") as handle:
                _, metadata, objects = edrl.read_table(handle)
                data = {}
                for obj in objects:
                    handle.seek(obj["offset"])
                    data[obj["name"]] = handle.read(obj["bytes"])
            self.assertEqual(b"\x01\x02\x03", metadata)
            for name, payload in base.items():
                self.assertEqual(payload, data[name])
            drafter = [obj for obj in objects if obj["name"].startswith("dflash2/")]
            self.assertEqual(4 + 2 * 13 + 3, len(drafter))
            words = np.frombuffer(data["dflash2/config"], "<i4")
            self.assertEqual([1, 2, HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM, 8, 299, 2, 16, 16, 32, 64, VOCAB],
                             words[:15].tolist())
            self.assertEqual([2, 1, 3], words[17:].tolist())
            self.assertAlmostEqual(1e7, np.frombuffer(words[16:17].tobytes(), "<f4")[0])
            key_value = np.frombuffer(data["dflash2/layers/1/attention/key_value"], "<u2").reshape(-1, HIDDEN)
            self.assertTrue(np.array_equal(key_value[:KV_HEADS * HEAD_DIM], values["layers.1.self_attn.k_proj.weight"]))
            self.assertTrue(np.array_equal(key_value[KV_HEADS * HEAD_DIM:], values["layers.1.self_attn.v_proj.weight"]))
            gate_up = np.frombuffer(data["dflash2/layers/0/mlp/gate_up"], "<u2").reshape(-1, HIDDEN)
            self.assertTrue(np.array_equal(gate_up[INTERMEDIATE:], values["layers.0.mlp.up_proj.weight"]))
            self.assertEqual(values["candidate_selector.successor_codebook"].tobytes(),
                             data["dflash2/selector/successor"])

    def test_the_published_configuration_is_checked(self):
        with self.assertRaises(ValueError):
            dflash2.check_draft(CONFIG)


if __name__ == "__main__":
    unittest.main()
