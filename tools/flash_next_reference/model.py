"""Building the upstream `Qwen4ExpTextModel` over an EDRL v3 artifact without giant allocations.

The model is constructed on the meta device with the unmodified upstream classes; only what a run needs is materialized:
BF16 tensors become parameters, NVFP4 linears become `ExactNvfp4Linear`, the 512 experts of a layer become `LazyExperts`
(each expert's record is expanded when the loop reaches it) and the 320M-row n-gram table becomes a gather over the
artifact's shards.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (Qwen4ExpTextExperts, Qwen4ExpTextModel,
                                                              Qwen4ExpTextNGramEmbedding)

from .artifact import DEFAULT_ARTIFACT, DEFAULT_HF_CONFIG, FORMAT_NVFP4, Artifact
from .nvfp4 import ExactNvfp4Linear, exact_linear


# --- configuration -------------------------------------------------------------------------------------------------


def load_hf_text_config(path: Path | str = DEFAULT_HF_CONFIG) -> dict:
    return json.loads(Path(path).read_text())["text_config"]


def make_config(text_config: dict) -> Qwen4ExpTextConfig:
    cfg = Qwen4ExpTextConfig(**text_config)
    cfg._attn_implementation = "eager"
    return cfg


def config_checks(cfg: Qwen4ExpTextConfig, md: dict) -> list[tuple[str, object, object]]:
    """(field, HF config value, artifact metadata value) for every key field; all pairs must be equal."""
    rope = cfg.rope_parameters
    eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
    layer_types = ["full_attention" if t == "indexed_attention" else t for t in cfg.layer_types]
    return [
        ("hidden_size", cfg.hidden_size, md["text.hidden_size"]),
        ("vocab_size", cfg.vocab_size, md["text.vocab_size"]),
        ("num_hidden_layers", cfg.num_hidden_layers, md["text.num_hidden_layers"]),
        ("layer_types", layer_types, list(md["text.layer_types"])),
        ("rms_norm_eps", float(cfg.rms_norm_eps), md["text.rms_norm_eps"]),
        ("hidden_act", cfg.hidden_act, md["text.hidden_act"]),
        ("eos_token_id", eos, md["text.eos_token_id"]),
        ("num_attention_heads", cfg.num_attention_heads, md["attention.num_heads"]),
        ("num_key_value_heads", cfg.num_key_value_heads, md["attention.num_kv_heads"]),
        ("head_dim", cfg.head_dim, md["attention.head_dim"]),
        ("rope_theta", float(rope["rope_theta"]), md["attention.rope_theta"]),
        ("partial_rotary_factor", float(rope["partial_rotary_factor"]), md["attention.partial_rotary_factor"]),
        ("mrope_section", list(rope["mrope_section"]), list(md["attention.mrope_section"])),
        ("output_gate_type", cfg.output_gate_type, md["attention.output_gate"]),
        ("linear_num_key_heads", cfg.linear_num_key_heads, md["gdn.num_key_heads"]),
        ("linear_num_value_heads", cfg.linear_num_value_heads, md["gdn.num_value_heads"]),
        ("linear_key_head_dim", cfg.linear_key_head_dim, md["gdn.key_head_dim"]),
        ("linear_value_head_dim", cfg.linear_value_head_dim, md["gdn.value_head_dim"]),
        ("linear_conv_kernel_dim", cfg.linear_conv_kernel_dim, md["gdn.conv_kernel_dim"]),
        ("indexer_n_heads", cfg.indexer_n_heads, md["qsa.num_heads"]),
        ("indexer_kv_heads", cfg.indexer_kv_heads, md["qsa.kv_heads"]),
        ("indexer_head_dim", cfg.indexer_head_dim, md["qsa.head_dim"]),
        ("indexer_budget", cfg.indexer_budget, md["qsa.indexer_budget"]),
        ("indexer_compress_ratio", cfg.indexer_compress_ratio, md["qsa.compress_ratio"]),
        ("num_experts", cfg.num_experts, md["moe.num_experts"]),
        ("num_experts_per_tok", cfg.num_experts_per_tok, md["moe.experts_per_token"]),
        ("moe_intermediate_size", cfg.moe_intermediate_size, md["moe.intermediate_size"]),
        ("shared_expert_intermediate_size", cfg.shared_expert_intermediate_size,
         md["moe.shared_expert_intermediate_size"]),
        ("norm_topk_prob", cfg.norm_topk_prob, True),
        ("hc_count", cfg.hc_count, md["hc.count"]),
        ("hc_lowrank", cfg.hc_lowrank, md["hc.lowrank"]),
        ("ple_layer_ids (1-based) vs ple.layers (0-based)", [i - 1 for i in cfg.ple_layer_ids], list(md["ple.layers"])),
        ("ple_embed_dim", cfg.ple_embed_dim, md["ple.embed_dim"]),
        ("ple_conv_kernel_size", cfg.ple_conv_kernel_size, md["ple.conv_kernel_size"]),
        ("ngram_size", cfg.ngram_size, md["ngram.size"]),
        ("heads_per_ngram", cfg.heads_per_ngram, md["ngram.heads_per_ngram"]),
        ("ngram_vocab_size_base", cfg.ngram_vocab_size_base, md["ngram.vocab_size_base"]),
        ("make_ngram_vocab_size_divisible_by", cfg.make_ngram_vocab_size_divisible_by, md["ngram.vocab_divisible_by"]),
        ("seed", cfg.seed, 1234),
    ]


def assert_config_matches(cfg: Qwen4ExpTextConfig, md: dict) -> list[tuple[str, object, object]]:
    checks = config_checks(cfg, md)
    bad = [c for c in checks if c[1] != c[2]]
    if bad:
        raise AssertionError("HF config and artifact metadata disagree: " + "; ".join(f"{n}: {a!r} != {b!r}" for n, a, b in bad))
    return checks


# --- n-gram table --------------------------------------------------------------------------------------------------


def ngram_tables_from_metadata(md: dict, cfg: Qwen4ExpTextConfig, ple_layer_index: int = 0):
    """(multipliers, heads_offsets, heads_vocab_sizes) from the artifact, asserted equal to what upstream's own
    constructor helpers compute."""
    multipliers = torch.tensor(md["ngram.layer_multipliers"], dtype=torch.long)
    offsets = torch.tensor(md["ngram.heads_offsets"], dtype=torch.long)
    sizes = torch.tensor(md["ngram.heads_vocab_sizes"], dtype=torch.long)
    expected_multipliers = upstream._build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_layer_index, cfg.seed)
    heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    expected_sizes, expected_offsets, total = [], [], 0
    for head in range(heads):
        size = upstream._find_nth_prime_after(cfg.ngram_vocab_size_base - 1, ple_layer_index * heads + head + 1)
        expected_sizes.append(size)
        expected_offsets.append(total)
        total += size
    if not torch.equal(multipliers, expected_multipliers):
        raise AssertionError(f"artifact multipliers {multipliers.tolist()} != upstream {expected_multipliers.tolist()}")
    if sizes.tolist() != expected_sizes or offsets.tolist() != expected_offsets:
        raise AssertionError("artifact n-gram head offsets/vocabulary sizes differ from upstream's constructor")
    return multipliers, offsets, sizes


class NgramShardEmbedding(nn.Module):
    """Replaces `nn.Embedding(320M, 160)`: gathers rows from the artifact's shards and dequantizes them to BF16."""

    def __init__(self, artifact: Artifact, layer: int, device):
        super().__init__()
        self.artifact, self.layer, self.device_ = artifact, layer, device
        self.register_buffer("weight", torch.empty(0, device=device), persistent=False)  # `.weight.device` is read upstream

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        rows = self.artifact.ngram_rows(self.layer, unique.cpu().numpy(), self.device_)
        return rows[inverse].reshape(*ids.shape, rows.shape[-1])


class IdsOnlyEmbedding(nn.Module):
    """Stand-in table that returns the ids themselves, to capture upstream's id construction alone."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.register_buffer("weight", torch.empty(0, device=device), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.unsqueeze(-1)


def make_ngram_module(cfg: Qwen4ExpTextConfig, md: dict, table: nn.Module, layer_idx: int = 1,
                      ple_layer_index: int = 0, device="cpu") -> Qwen4ExpTextNGramEmbedding:
    """An upstream n-gram embedding module (unmodified code) with the artifact's tables and `table` as its embedding."""
    with torch.device("meta"):
        module = Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx, ple_layer_index)
    multipliers, offsets, sizes = ngram_tables_from_metadata(md, cfg, ple_layer_index)
    if module.head_vocab_sizes != sizes.tolist() or module.head_offsets != offsets.tolist():
        raise AssertionError("module head tables differ from the artifact's")
    module._buffers["layer_multipliers"] = multipliers.to(device)
    module._buffers["ngram_heads_vocab_sizes"] = sizes.to(device)
    module._buffers["ngram_heads_offsets"] = offsets.to(device)
    module.ngram_embedding = table
    return module


# --- experts -------------------------------------------------------------------------------------------------------


class LazyExperts(Qwen4ExpTextExperts):
    """`Qwen4ExpTextExperts` whose weights come from the artifact: upstream's loop verbatim, except that
    `gate_up_proj[expert_idx]` / `down_proj[expert_idx]` are that expert's record expanded exactly (fp32 weights; the
    products are `exact_linear`: fp32 accumulate, one BF16 rounding)."""

    def __init__(self, config, artifact: Artifact, layer: int, device):
        super().__init__(config)
        self._artifact, self._layer, self._device = artifact, layer, device
        self.observer = None
        for name in ("gate_up_proj", "down_proj"):  # the meta placeholders are never used
            self._parameters.pop(name, None)

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        hit_ids = []
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up_weight, down_weight = self._artifact.expert(self._layer, int(expert_idx), self._device).expand()
            gate, up = exact_linear(current_state, gate_up_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = exact_linear(current_hidden_states, down_weight)
            expert_out = current_hidden_states
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            hit_ids.append(int(expert_idx))
            if self.observer is not None:
                self.observer(int(expert_idx), token_idx, expert_out, current_hidden_states.to(final_hidden_states.dtype))

        if self.observer is not None:
            self.observer(None, hit_ids, None, None)
        return final_hidden_states


# --- the model ---------------------------------------------------------------------------------------------------


class StubMixer(nn.Module):
    """Replaces the final hyper-connection mixer in layer-isolated runs."""

    def forward(self, hidden_states):
        return hidden_states


class FlashNext:
    """The upstream text model over an artifact. `layers=None` materializes all 48 layers plus embedding and head."""

    def __init__(self, artifact: Artifact, cfg: Qwen4ExpTextConfig, device, layers=None):
        self.artifact, self.cfg, self.device = artifact, cfg, torch.device(device)
        self.layers = list(range(cfg.num_hidden_layers)) if layers is None else sorted(layers)
        self.full = layers is None
        self.check_rows = assert_config_matches(cfg, artifact.metadata)
        with torch.device("meta"):
            self.model = Qwen4ExpTextModel(cfg)
        self.model.eval().requires_grad_(False)
        self.by_index = {n: self.model.layers[n] for n in self.layers}
        self.model.rotary_emb = upstream.Qwen4ExpTextRotaryEmbedding(cfg).to(self.device)
        self.lm_head: torch.Tensor | None = None
        self._materialize()

    # --- materialization -----------------------------------------------------------------------------------

    def _bf16(self, name: str) -> nn.Parameter:
        return nn.Parameter(self.artifact.bf16(name, self.device), requires_grad=False)

    def _param(self, module: nn.Module, attr: str, name: str) -> None:
        new = self._bf16(name)
        old = getattr(module, attr)
        if tuple(old.shape) != tuple(new.shape):
            raise ValueError(f"{name}: artifact shape {tuple(new.shape)} != module {tuple(old.shape)}")
        setattr(module, attr, new)

    def _linear(self, parent: nn.Module, attr: str, name: str) -> None:
        entry = self.artifact.entry(name)
        old = getattr(parent, attr)
        if tuple(entry["shape"]) != (old.out_features, old.in_features):
            raise ValueError(f"{name}: artifact shape {entry['shape']} != linear {(old.out_features, old.in_features)}")
        if entry["format"] == FORMAT_NVFP4:
            setattr(parent, attr, ExactNvfp4Linear(self.artifact.nvfp4(name, self.device)))
        else:
            self._param(old, "weight", name)

    def _gated_residual(self, module: nn.Module, prefix: str, inject: bool = True) -> None:
        self._param(module.hc_norm, "weight", f"{prefix}/hc_norm")
        self._linear(module, "input_mix_weight_down", f"{prefix}/input_mix_down")
        self._linear(module, "input_mix_weight_up", f"{prefix}/input_mix_up")
        if inject:
            self._linear(module, "block_inject_weight", f"{prefix}/block_inject")

    def _materialize(self) -> None:
        art = self.artifact
        if self.full:
            self._param(self.model.embed_tokens, "weight", "text/token_embedding")
            head = art.bf16("text/output_head", self.device)
            self.lm_head = head
            self._gated_residual(self.model.hyper_connection_mixer, "text/hyper_connection_mixer", inject=False)
        else:
            self.model.hyper_connection_mixer = StubMixer()
        for n in self.layers:
            self._materialize_layer(n)
        if not self.full:
            self.model.layers = nn.ModuleList([self.model.layers[n] for n in self.layers])

    def _materialize_layer(self, n: int) -> None:
        layer = self.model.layers[n]
        p = f"text/layers/{n}"
        self._gated_residual(layer.attn_hyper_connection, f"{p}/attn_hc")
        self._gated_residual(layer.mlp_hyper_connection, f"{p}/mlp_hc")
        if layer.layer_type == "linear_attention":
            gdn = layer.linear_attn
            self._param(gdn, "A_log", f"{p}/gdn/a_log")
            self._param(gdn, "dt_bias", f"{p}/gdn/dt_bias")
            self._param(gdn.conv1d, "weight", f"{p}/gdn/conv1d")
            self._param(gdn.norm, "weight", f"{p}/gdn/norm")
            for attr, name in (("in_proj_qkv", "in_proj_qkv"), ("in_proj_z", "in_proj_z"), ("in_proj_b", "in_proj_b"),
                               ("in_proj_a", "in_proj_a"), ("out_proj", "out_proj")):
                self._linear(gdn, attr, f"{p}/gdn/{name}")
        else:
            attn = layer.self_attn
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self._linear(attn, attr, f"{p}/attention/{attr}")
            self._param(attn.q_norm, "weight", f"{p}/attention/q_norm")
            self._param(attn.k_norm, "weight", f"{p}/attention/k_norm")
            self._linear(attn.indexer, "index_qk_proj", f"{p}/attention/indexer/index_qk_proj")
            self._param(attn.indexer.q_layernorm, "weight", f"{p}/attention/indexer/q_layernorm")
            self._param(attn.indexer.k_layernorm, "weight", f"{p}/attention/indexer/k_layernorm")
        moe = layer.mlp
        self._param(moe.gate, "weight", f"{p}/moe/router")
        for attr in ("gate_proj", "up_proj", "down_proj"):
            self._linear(moe.shared_expert, attr, f"{p}/moe/shared_expert/{attr}")
        self._linear(moe, "shared_expert_gate", f"{p}/moe/shared_expert_gate")
        with torch.device("meta"):
            moe.experts = LazyExperts(self.cfg, self.artifact, n, self.device)
        if layer.ple is not None:
            ple = layer.ple
            self._linear(ple, "key_proj", f"{p}/ple/key_proj")
            self._linear(ple, "value_proj", f"{p}/ple/value_proj")
            self._param(ple.conv1d, "weight", f"{p}/ple/conv1d")
            for attr in ("norm_conv", "norm_key", "norm_query"):
                self._param(getattr(ple, attr), "weight", f"{p}/ple/{attr}")
            ple_index = self.cfg.ple_layer_ids.index(n + 1)
            ple.ple_embedding = make_ngram_module(self.cfg, self.artifact.metadata, NgramShardEmbedding(self.artifact, n, self.device),
                                                  layer_idx=n, ple_layer_index=ple_index, device=self.device)

    # --- helpers ---------------------------------------------------------------------------------------------

    def layer(self, n: int) -> nn.Module:
        return self.by_index[n]

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden, self.lm_head)
