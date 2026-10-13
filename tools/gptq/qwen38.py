"""A layer-streamed PyTorch forward pass of Qwen3.8-27B (text), for calibration.

Each decoder layer's BF16 weights are read from the Hugging Face checkpoint onto the GPU when the layer runs, so the
whole model never needs to fit. The arithmetic follows the engine's exact reference kernels (docs in the engine:
core/src/test/.../gpu/FirstLayerCpuReference.java, native/src/gdn, native/src/attention): every op computes in FP32 and
rounds its output to BF16, except the GDN a/b projections and the decay and beta derived from them, which stay FP32.
The KV cache is not quantized here (the engine stores it in NVFP4).

`Layer.forward` takes hidden states [batch, tokens, 5120] (BF16, CUDA) and returns the next layer's. When `observe` is
given, it is called with (object name, input rows [N, K] BF16) for every quantized linear, before the linear runs; the
names are the artifact objects (`text/layers/L/mlp/down`, ...), whose weights `weights[name]` hold in artifact layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from euhedral_artifacts.inventory import FULL_ATTENTION_LAYERS
from euhedral_artifacts.sources import SourceStore

HIDDEN, INTERMEDIATE, LAYERS = 5120, 17408, 64
HEADS, KV_HEADS, HEAD_DIM, ROTARY = 24, 4, 256, 64
KEY_HEADS, VALUE_HEADS, KEY_DIM, VALUE_DIM = 16, 48, 128, 128
CONV = 4

Observe = Callable[[str, torch.Tensor], None]


class Checkpoint:
    """BF16 tensors of the Hugging Face checkpoint on the GPU, read with positioned reads (SourceStore)."""

    def __init__(self, model: Path, device: str = "cuda"):
        self.store = SourceStore(model)
        config = json.loads((model / "config.json").read_text())["text_config"]
        self.eps = float(config["rms_norm_eps"])
        self.theta = float(config["rope_parameters"]["rope_theta"])
        self.device = device

    def close(self) -> None:
        self.store.close()

    def tensor(self, name: str) -> torch.Tensor:
        """A checkpoint tensor (every one is BF16; SourceStore checks) on the GPU."""
        shape = self.store.ref(name).shape
        words = torch.from_numpy(self.store.words(name).view("int16").copy())
        return words.view(torch.bfloat16).reshape(shape).to(self.device)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, offset: bool = True) -> torch.Tensor:
    """RMSNorm in FP32, rounded to BF16; the weight is applied as (1 + w) unless `offset` is False."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    w = weight.float() + 1.0 if offset else weight.float()
    return (xf * inv * w).to(torch.bfloat16)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x [..., K] BF16 @ weight [N, K]^T with FP32 accumulation, rounded to BF16."""
    return F.linear(x, weight.to(x.dtype))


class Layer:
    """One decoder layer: weights in artifact layout (fused objects as the converter builds them)."""

    def __init__(self, checkpoint: Checkpoint, index: int):
        self.index, self.eps, self.theta = index, checkpoint.eps, checkpoint.theta
        self.attention = index in FULL_ATTENTION_LAYERS
        p = f"model.language_model.layers.{index}."
        o = f"text/layers/{index}/"
        t = checkpoint.tensor
        self.input_norm, self.post_norm = t(p + "input_layernorm.weight"), t(p + "post_attention_layernorm.weight")
        w: dict[str, torch.Tensor] = {}
        if self.attention:
            q = t(p + "self_attn.q_proj.weight").view(HEADS, 2, HEAD_DIM, HIDDEN)
            query, gate = q[:, 0].reshape(-1, HIDDEN), q[:, 1].reshape(-1, HIDDEN)
            w[o + "attention/query_key"] = torch.cat([query, t(p + "self_attn.k_proj.weight")])
            w[o + "attention/gate_value"] = torch.cat([gate, t(p + "self_attn.v_proj.weight")])
            w[o + "attention/output"] = t(p + "self_attn.o_proj.weight")
            self.query_norm, self.key_norm = t(p + "self_attn.q_norm.weight"), t(p + "self_attn.k_norm.weight")
        else:
            qkv = t(p + "linear_attn.in_proj_qkv.weight")
            w[o + "gdn/query_key"] = qkv[:2 * KEY_HEADS * KEY_DIM]
            w[o + "gdn/value_z"] = torch.cat([qkv[2 * KEY_HEADS * KEY_DIM:], t(p + "linear_attn.in_proj_z.weight")])
            w[o + "gdn/output"] = t(p + "linear_attn.out_proj.weight")
            self.a_projection, self.b_projection = t(p + "linear_attn.in_proj_a.weight"), t(p + "linear_attn.in_proj_b.weight")
            self.convolution = t(p + "linear_attn.conv1d.weight").view(-1, CONV).float()  # [10240, taps]
            self.a_log, self.dt_bias = t(p + "linear_attn.A_log").float(), t(p + "linear_attn.dt_bias").float()
            self.gdn_norm = t(p + "linear_attn.norm.weight")
        w[o + "mlp/gate_up"] = torch.cat([t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")])
        w[o + "mlp/down"] = t(p + "mlp.down_proj.weight")
        self.weights = w
        self.prefix = o

    def _linear(self, name: str, x: torch.Tensor, observe: Observe | None) -> torch.Tensor:
        if observe is not None:
            observe(self.prefix + name, x.reshape(-1, x.shape[-1]))
        return linear(x, self.weights[self.prefix + name])

    def forward(self, h: torch.Tensor, observe: Observe | None = None) -> torch.Tensor:
        x = rms_norm(h, self.input_norm, self.eps)
        mixed = self._attention(x, observe) if self.attention else self._gdn(x, observe)
        m = (h.float() + mixed.float()).to(torch.bfloat16)
        x = rms_norm(m, self.post_norm, self.eps)
        gate_up = self._linear("mlp/gate_up", x, observe)
        gate, up = gate_up.float().chunk(2, dim=-1)
        swiglu = (F.silu(gate) * up).to(torch.bfloat16)
        down = self._linear("mlp/down", swiglu, observe)
        return (m.float() + down.float()).to(torch.bfloat16)

    def _attention(self, x: torch.Tensor, observe: Observe | None) -> torch.Tensor:
        qk = self._linear("attention/query_key", x, observe)
        gv = self._linear("attention/gate_value", x, observe)
        return self._attend(qk, gv, observe)

    def _attend(self, qk: torch.Tensor, gv: torch.Tensor, observe: Observe | None) -> torch.Tensor:
        b, n, _ = qk.shape
        q = qk[..., :HEADS * HEAD_DIM].view(b, n, HEADS, HEAD_DIM)
        k = qk[..., HEADS * HEAD_DIM:].view(b, n, KV_HEADS, HEAD_DIM)
        gate = gv[..., :HEADS * HEAD_DIM].view(b, n, HEADS, HEAD_DIM)
        v = gv[..., HEADS * HEAD_DIM:].view(b, n, KV_HEADS, HEAD_DIM)
        q, k = self._norm_rope(q, self.query_norm), self._norm_rope(k, self.key_norm)
        # GQA: query head h reads KV head h // 6.
        k = k.repeat_interleave(HEADS // KV_HEADS, dim=2)
        v = v.repeat_interleave(HEADS // KV_HEADS, dim=2)
        out = F.scaled_dot_product_attention(q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(),
                                             is_causal=True, scale=HEAD_DIM ** -0.5).transpose(1, 2)
        context = (out * torch.sigmoid(gate.float())).to(torch.bfloat16).reshape(b, n, HEADS * HEAD_DIM)
        return self._linear("attention/output", context, observe)

    def _norm_rope(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm (1 + w), then half-split RoPE on the first 64 dims of each head."""
        y = rms_norm(x, weight, self.eps).float()
        n = x.shape[1]
        half = ROTARY // 2
        f = torch.arange(half, dtype=torch.float64, device=x.device)
        angle = torch.arange(n, dtype=torch.float64, device=x.device)[:, None] * self.theta ** (-2.0 * f / ROTARY)
        cos, sin = angle.cos().float()[None, :, None, :], angle.sin().float()[None, :, None, :]
        lo, hi = y[..., :half], y[..., half:ROTARY]
        rotated = torch.cat([lo * cos - hi * sin, hi * cos + lo * sin, y[..., ROTARY:]], dim=-1)
        return rotated.to(torch.bfloat16)

    def _gdn(self, x: torch.Tensor, observe: Observe | None) -> torch.Tensor:
        b, n, _ = x.shape
        qk = self._linear("gdn/query_key", x, observe)
        vz = self._linear("gdn/value_z", x, observe)
        a = x.float() @ self.a_projection.float().T  # [b, n, 48], FP32
        beta_in = x.float() @ self.b_projection.float().T
        alpha = torch.exp(-torch.exp(self.a_log) * F.softplus(a + self.dt_bias))
        beta = torch.sigmoid(beta_in)
        value_width = VALUE_HEADS * VALUE_DIM
        channels = torch.cat([qk, vz[..., :value_width]], dim=-1).float()  # [b, n, 10240]
        padded = F.pad(channels.transpose(1, 2), (CONV - 1, 0))  # [b, 10240, n + 3]
        conv = F.conv1d(padded, self.convolution[:, None, :], groups=channels.shape[-1])  # tap 3 = current token
        conv = F.silu(conv).to(torch.bfloat16).transpose(1, 2).float()  # [b, n, 10240]
        q = conv[..., :KEY_HEADS * KEY_DIM].view(b, n, KEY_HEADS, KEY_DIM)
        k = conv[..., KEY_HEADS * KEY_DIM:2 * KEY_HEADS * KEY_DIM].view(b, n, KEY_HEADS, KEY_DIM)
        v = conv[..., 2 * KEY_HEADS * KEY_DIM:].view(b, n, VALUE_HEADS, VALUE_DIM)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim=True) + 1e-6)
        # Value head h reads key head h // 3.
        q = q.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2)
        k = k.repeat_interleave(VALUE_HEADS // KEY_HEADS, dim=2)
        out = recurrence(q, k, v, alpha, beta) * VALUE_DIM ** -0.5  # VALUE_DIM == KEY_DIM == 128
        out = out.to(torch.bfloat16).float()  # [b, n, 48, 128]
        z = vz[..., value_width:].float().view(b, n, VALUE_HEADS, VALUE_DIM)
        inv = torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        gated = (out * inv * self.gdn_norm.float() * F.silu(z)).to(torch.bfloat16).reshape(b, n, value_width)
        return self._linear("gdn/output", gated, observe)


class MtpLayer(Layer):
    """The MTP block (docs/MTP_CONTRACT.md): a full-attention layer whose q, k, gate and v are one object, rows
    [q 6144; k 1024; gate 6144; v 1024], under the `mtp/layer/` object names."""

    def __init__(self, checkpoint: Checkpoint):
        self.index, self.eps, self.theta = -1, checkpoint.eps, checkpoint.theta
        self.attention = True
        p, t = "mtp.layers.0.", checkpoint.tensor
        self.input_norm, self.post_norm = t(p + "input_layernorm.weight"), t(p + "post_attention_layernorm.weight")
        q = t(p + "self_attn.q_proj.weight").view(HEADS, 2, HEAD_DIM, HIDDEN)
        query, gate = q[:, 0].reshape(-1, HIDDEN), q[:, 1].reshape(-1, HIDDEN)
        self.weights = {
            "mtp/layer/attention/query_key_gate_value": torch.cat(
                [query, t(p + "self_attn.k_proj.weight"), gate, t(p + "self_attn.v_proj.weight")]),
            "mtp/layer/attention/output": t(p + "self_attn.o_proj.weight"),
            "mtp/layer/mlp/gate_up": torch.cat([t(p + "mlp.gate_proj.weight"), t(p + "mlp.up_proj.weight")]),
            "mtp/layer/mlp/down": t(p + "mlp.down_proj.weight"),
        }
        self.query_norm, self.key_norm = t(p + "self_attn.q_norm.weight"), t(p + "self_attn.k_norm.weight")
        self.prefix = "mtp/layer/"

    def _attention(self, x: torch.Tensor, observe: Observe | None) -> torch.Tensor:
        fused = self._linear("attention/query_key_gate_value", x, observe)
        half = HEADS * HEAD_DIM + KV_HEADS * HEAD_DIM
        return self._attend(fused[..., :half], fused[..., half:], observe)


class Mtp:
    """The MTP stem and output norm around MtpLayer: row p reads the base model's normed hidden at p and the
    embedding of token p + 1, at position p, and its normed output is what the draft head reads."""

    def __init__(self, checkpoint: Checkpoint):
        t = checkpoint.tensor
        self.eps = checkpoint.eps
        self.embedding_norm, self.hidden_norm = t("mtp.pre_fc_norm_embedding.weight"), t("mtp.pre_fc_norm_hidden.weight")
        self.final_norm = t("mtp.norm.weight")
        self.input_projection = t("mtp.fc.weight")
        self.layer = MtpLayer(checkpoint)

    def stem(self, embedded: torch.Tensor, normed_hidden: torch.Tensor) -> torch.Tensor:
        """[RMSNorm(embedding of token p + 1); RMSNorm(base hidden at p after the final norm)], [b, n, 10240]."""
        return torch.cat([rms_norm(embedded, self.embedding_norm, self.eps),
                          rms_norm(normed_hidden, self.hidden_norm, self.eps)], dim=-1)


def recurrence(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """The gated delta rule, token by token over the batch: q, k [b, n, H, 128] (L2-normalized), v [b, n, H, 128],
    alpha, beta [b, n, H]. State S [b, H, 128 value, 128 key] FP32 from zero; returns S . q per token [b, n, H, 128]."""
    b, n, heads, dk = k.shape
    state = torch.zeros(b, heads, v.shape[-1], dk, dtype=torch.float32, device=k.device)
    outputs = torch.empty(b, n, heads, v.shape[-1], dtype=torch.float32, device=k.device)
    for t in range(n):
        kt, qt = k[:, t], q[:, t]  # [b, H, dk]
        dot = torch.einsum("bhvk,bhk->bhv", state, kt)
        a = alpha[:, t][..., None]
        delta = beta[:, t][..., None] * (v[:, t] - a * dot)  # [b, H, dv]
        state = a[..., None] * state + delta[..., None] * kt[:, :, None, :]
        outputs[:, t] = torch.einsum("bhvk,bhk->bhv", state, qt)
    return outputs


class Model:
    """Embedding, final norm and LM head of the checkpoint (the decoder layers are loaded one at a time)."""

    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint
        self.embedding = checkpoint.tensor("model.language_model.embed_tokens.weight")
        self.final_norm = checkpoint.tensor("model.language_model.norm.weight")
        self.head = checkpoint.tensor("lm_head.weight")

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding[tokens]

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return linear(rms_norm(h, self.final_norm, self.checkpoint.eps), self.head)
