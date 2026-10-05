"""Recording the upstream model's intermediates through hooks, wrapped module-level functions and recording subclasses.

Upstream source is never edited. Where an intermediate is not a module boundary (the PLE gate, the QSA indexer's
internals) a subclass carries a copy of the upstream forward body with recording statements; every call also runs the
unmodified upstream forward on the same inputs (the cache states are put back in between) and fails loudly unless the
results are bit-identical.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transformers.cache_utils import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as upstream
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer, Qwen4ExpTextQSAIndexer

from . import kvcodec
from .record import Recorder


class ReplayMismatch(RuntimeError):
    pass


def stable_topk(probs: torch.Tensor, k: int):
    """The deterministic selection rule: descending value, ties by ASCENDING expert index (a stable sort).
    Returns (indices [rows, k], values [rows, k], boundary_tie [rows] = the k-th and (k+1)-th values are equal)."""
    values, index = torch.sort(probs, dim=-1, descending=True, stable=True)
    tie = values[:, k - 1] == values[:, k] if probs.shape[-1] > k else torch.zeros(probs.shape[0], dtype=torch.bool,
                                                                                     device=probs.device)
    return index[:, :k], values[:, :k], tie


def _clone(t):
    return None if t is None else t.clone()


def snapshot_linear_layer(layer) -> dict:
    return {
        "conv_states": {k: _clone(v) for k, v in layer.conv_states.items()},
        "recurrent_states": {k: _clone(v) for k, v in layer.recurrent_states.items()},
        "has_previous_state": dict(layer.has_previous_state),
        "is_conv_states_initialized": dict(layer.is_conv_states_initialized),
        "is_recurrent_states_initialized": dict(layer.is_recurrent_states_initialized),
        "conv_kernel_size": dict(layer.conv_kernel_size),
        "device": layer.device,
        "dtype": layer.dtype,
    }


def restore_linear_layer(layer, snap: dict) -> None:
    layer.conv_states = {k: _clone(v) for k, v in snap["conv_states"].items()}
    layer.recurrent_states = {k: _clone(v) for k, v in snap["recurrent_states"].items()}
    layer.has_previous_state = dict(snap["has_previous_state"])
    layer.is_conv_states_initialized = dict(snap["is_conv_states_initialized"])
    layer.is_recurrent_states_initialized = dict(snap["is_recurrent_states_initialized"])
    layer.conv_kernel_size = dict(snap["conv_kernel_size"])
    layer.device, layer.dtype = snap["device"], snap["dtype"]


def _states_equal(a: dict, b: dict) -> bool:
    for key in ("conv_states", "recurrent_states"):
        for index, tensor in a[key].items():
            other = b[key][index]
            if (tensor is None) != (other is None) or (tensor is not None and not torch.equal(tensor, other)):
                return False
    return a["has_previous_state"] == b["has_previous_state"]


def snapshot_indexer(layer) -> dict:
    return {name: getattr(layer, name, None) for name in
            ("indexer_keys", "is_indexer_initialized", "indexer_dtype", "indexer_device")}


def restore_indexer(layer, snap: dict) -> None:
    for name, value in snap.items():
        if value is None and name in ("indexer_dtype", "indexer_device") and not hasattr(layer, name):
            continue
        setattr(layer, name, value)


# --- recording subclasses ----------------------------------------------------------------------------------------------


class RecordingPLELayer(Qwen4ExpTextPLELayer):
    """`Qwen4ExpTextPLELayer` with the forward body copied and recording; replayed against the upstream forward."""

    capture: "Capture" = None

    def _recording_forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
        cap, n = self.capture, self.layer_idx
        rec = cap.rec
        embeddings = self.ple_embedding(input_ids, past_key_values)
        key_proj = self.key_proj(embeddings)
        key_normed = self.norm_key(key_proj).unflatten(-1, (self.hc_count, self.hidden_size))
        rec.add(f"L{n}/ple/key_normed", key_normed[0])
        value = self.value_proj(embeddings)
        rec.add(f"L{n}/ple/value", value[0])
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        rec.add(f"L{n}/ple/query_normed", query_normed[0])
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        rec.add(f"L{n}/ple/gate_raw", gate[0])
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        rec.add(f"L{n}/ple/gate", gate[0])
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        rec.add(f"L{n}/ple/gated_value", gated_value[0])
        rec.add(f"L{n}/ple/gated_normed", gated_value_normed[0])
        if conv_mask is not None:
            gated_value = upstream.apply_mask_to_padding_states(gated_value, conv_mask)
            gated_value_normed = upstream.apply_mask_to_padding_states(gated_value_normed, conv_mask)
        conv = self._short_conv(gated_value_normed, past_key_values)
        rec.add(f"L{n}/ple/conv", conv[0])
        output = gated_value + conv
        return output

    def forward(self, hidden_states, input_ids, past_key_values, conv_mask=None):
        cap, n = self.capture, self.layer_idx
        layer_cache = past_key_values.layers[n] if past_key_values is not None else None
        before = snapshot_linear_layer(layer_cache) if layer_cache is not None else None
        output = self._recording_forward(hidden_states, input_ids, past_key_values, conv_mask)
        if layer_cache is not None:
            after = snapshot_linear_layer(layer_cache)
            restore_linear_layer(layer_cache, before)
        with cap.rec.mute():
            reference = Qwen4ExpTextPLELayer.forward(self, hidden_states, input_ids, past_key_values, conv_mask)
        if not torch.equal(output, reference):
            raise ReplayMismatch(f"layer {n}: recording PLE forward differs from upstream "
                                 f"(max abs {(output.float() - reference.float()).abs().max().item():.3e})")
        if layer_cache is not None and not _states_equal(after, snapshot_linear_layer(layer_cache)):
            raise ReplayMismatch(f"layer {n}: PLE cache state after the recording forward differs from upstream's")
        cap.replays["ple"] += 1
        if layer_cache is not None:
            cap.rec.add(f"L{n}/ple/conv_state", layer_cache.conv_states[1])
            cap.rec.add(f"L{n}/ple/ngram_context", layer_cache.conv_states[2])
        cap.rec.add(f"L{n}/ple/out", reference[0])
        return reference


class RecordingQSAIndexer(Qwen4ExpTextQSAIndexer):
    """`Qwen4ExpTextQSAIndexer` with the forward body copied and recording; replayed against the upstream forward."""

    capture: "Capture" = None

    def _recording_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        cap, n = self.capture, self.layer_idx
        rec = cap.rec
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, -1, self.index_head_dim)
        full_cos, full_sin = position_embeddings
        current_cos, current_sin = full_cos[:, -seq_length:, :], full_sin[:, -seq_length:, :]
        qk = self.index_qk_proj(hidden_states)
        q, token_k = torch.split(
            qk,
            [self.index_n_heads * self.index_head_dim, self.index_kv_heads * self.index_head_dim],
            dim=-1,
        )
        q, raw_keys = q.reshape(*hidden_shape), token_k.reshape(*hidden_shape).squeeze(2)
        q = self.q_layernorm(q)
        rec.add(f"L{n}/attn/index_q", q[0])
        q = upstream.apply_rotary_pos_emb(q, cos=current_cos, sin=current_sin, unsqueeze_dim=2)
        rec.add(f"L{n}/attn/index_q_rope", q[0])
        rec.add(f"L{n}/attn/index_raw_k", raw_keys[0])

        if past_key_values is not None:
            raw_keys = past_key_values.update_indexer(raw_keys, self.layer_idx)

        visible_token_indices = attention_mask if attention_mask.dtype == torch.bool else attention_mask == 0

        selected_token_indices = torch.full(
            (batch_size, seq_length, self.token_budget + self.compress_ratio - 1),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        row_scores, row_blocks, last_block_keys = [], [], None
        for batch_idx in range(batch_size):
            for query_idx in range(seq_length):
                local_visible_indices = torch.nonzero(
                    visible_token_indices[batch_idx, 0, query_idx], as_tuple=False
                ).flatten()
                num_complete_blocks = local_visible_indices.shape[-1] // self.compress_ratio
                if num_complete_blocks > 0:
                    block_token_indices = local_visible_indices[: num_complete_blocks * self.compress_ratio].view(
                        num_complete_blocks, self.compress_ratio
                    )

                    key_groups = raw_keys[batch_idx].index_select(0, block_token_indices.flatten())
                    key_groups = key_groups.view(*block_token_indices.shape, self.index_head_dim)
                    pooled_keys = key_groups.float().mean(dim=1).to(raw_keys.dtype)
                    pooled_keys = self.k_layernorm(pooled_keys)
                    group_starts = block_token_indices[:, 0]
                    block_key_states = upstream.apply_rotary_pos_emb(
                        pooled_keys.unsqueeze(1),
                        cos=full_cos[batch_idx].index_select(0, group_starts),
                        sin=full_sin[batch_idx].index_select(0, group_starts),
                    ).squeeze(1)

                    scores = torch.matmul(
                        q[batch_idx, query_idx].float(), block_key_states.float().transpose(-1, -2)
                    ).transpose(-1, -2)
                    scores = torch.relu(scores).sum(dim=-1) / math.sqrt(self.index_head_dim)

                    selected_block_indices = scores.topk(min(self.block_topk, num_complete_blocks), dim=0).indices
                    selected_tokens = block_token_indices.index_select(0, selected_block_indices).flatten()
                    row_scores.append(scores)
                    row_blocks.append(selected_block_indices)
                    last_block_keys = block_key_states
                else:
                    selected_tokens = torch.tensor([], device=hidden_states.device)
                    row_scores.append(None)
                    row_blocks.append(None)
                tail = local_visible_indices[num_complete_blocks * self.compress_ratio :]
                selected_tokens = torch.cat([selected_tokens, tail]).to(torch.int32)
                selected_token_indices[batch_idx, query_idx, : selected_tokens.numel()] = selected_tokens

        if rec.wants(f"L{n}/attn/index_scores") or rec.wants(f"L{n}/attn/block_ids"):
            width = max((s.numel() for s in row_scores if s is not None), default=0)
            scores_matrix = torch.full((seq_length, max(width, 1)), float("-inf"), dtype=torch.float32,
                                       device=hidden_states.device)
            block_ids = torch.full((seq_length, self.block_topk), -1, dtype=torch.int32, device=hidden_states.device)
            block_sorted = block_ids.clone()
            for row, (s, b) in enumerate(zip(row_scores, row_blocks)):
                if s is None:
                    continue
                scores_matrix[row, : s.numel()] = s
                block_ids[row, : b.numel()] = b.to(torch.int32)
                block_sorted[row, : b.numel()] = b.sort().values.to(torch.int32)
            rec.add(f"L{n}/attn/index_scores", scores_matrix)
            rec.add(f"L{n}/attn/block_ids", block_ids)
            rec.add(f"L{n}/attn/block_ids_sorted", block_sorted)
        if last_block_keys is not None:
            rec.add(f"L{n}/attn/index_block_keys", last_block_keys)
        rec.add(f"L{n}/attn/token_ids", selected_token_indices[0])

        kv_length = attention_mask.shape[-1]
        selected_token_mask = torch.zeros(
            (*selected_token_indices.shape[:-1], kv_length + 1), device=attention_mask.device, dtype=torch.bool
        )
        scatter_indices = torch.where(selected_token_indices >= 0, selected_token_indices, kv_length)
        selected_token_mask = selected_token_mask.scatter(-1, scatter_indices, True)[..., :kv_length].unsqueeze(1)
        if attention_mask.is_floating_point():
            min_dtype = torch.finfo(attention_mask.dtype).min
            selected_token_mask = torch.where(selected_token_mask, attention_mask.new_zeros(()), min_dtype)

        return selected_token_mask

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        cap = self.capture
        layer_cache = past_key_values.layers[self.layer_idx] if past_key_values is not None else None
        before = snapshot_indexer(layer_cache) if layer_cache is not None else None
        result = self._recording_forward(hidden_states, position_embeddings, attention_mask, past_key_values)
        if layer_cache is not None:
            after_keys = layer_cache.indexer_keys
            restore_indexer(layer_cache, before)
        with cap.rec.mute():
            reference = Qwen4ExpTextQSAIndexer.forward(self, hidden_states, position_embeddings, attention_mask,
                                                       past_key_values)
        if not torch.equal(result, reference):
            raise ReplayMismatch(f"layer {self.layer_idx}: recording QSA indexer mask differs from upstream's")
        if layer_cache is not None and not torch.equal(after_keys, layer_cache.indexer_keys):
            raise ReplayMismatch(f"layer {self.layer_idx}: indexer key cache differs after the replay")
        cap.replays["indexer"] += 1
        return reference


# --- capture ---------------------------------------------------------------------------------------------------------


class Capture:
    """Installs the recording on a `FlashNext` instance; `begin_chunk` selects the chunk context before a forward."""

    GLOBALS = ("causal_conv1d_fn", "causal_conv1d_update", "torch_chunk_gated_delta_rule",
               "torch_recurrent_gated_delta_rule", "eager_attention_forward")

    def __init__(self, fn, rec: Recorder, capture_layers=(), kv_format: str = "bf16", model_level: bool = True):
        if kv_format not in ("bf16", "nvfp4"):
            raise ValueError("kv_format must be bf16 or nvfp4")
        self.fn, self.rec = fn, rec
        self.capture_layers = {n for n in capture_layers if n in fn.by_index}
        self.kv_format = kv_format
        self.model_level = model_level
        self.rows = 0
        self.handles: list = []
        self.originals: dict = {}
        self.current_gdn: int | None = None
        self.replays = {"ple": 0, "indexer": 0}
        self.tie_stats = {"router_calls": 0, "rows": 0, "boundary_tie_rows": 0, "order_differs_rows": 0,
                          "set_differs_rows": 0, "layers_with_order_diff": set(), "layers_with_set_diff": set(),
                          "layers_with_tie": set(), "topk_nondeterministic_rows": 0}
        self.expert_events: dict[int, dict] = {}
        self._install()

    # --- helpers ---------------------------------------------------------------------------------------------

    def hook(self, module, kind: str, callback, **kwargs):
        register = {"pre": module.register_forward_pre_hook, "post": module.register_forward_hook}[kind]

        def observe(*args, **kw):  # a hook that returns something would replace the module's input or output
            callback(*args, **kw)

        self.handles.append(register(observe, **kwargs))

    def begin_chunk(self, prefix: str, rows: int, detail: bool = True) -> None:
        self.rec.begin(prefix, detail)
        self.rows = rows

    def layer_is_captured(self, n: int) -> bool:
        return n in self.capture_layers

    # --- installation ----------------------------------------------------------------------------------------

    def _install(self) -> None:
        fn, rec = self.fn, self.rec
        model = fn.model
        if fn.full and self.model_level:
            self.hook(model.embed_tokens, "post", lambda m, a, out: rec.add("embedding", out[0]))
            mixer = model.hyper_connection_mixer
            self.hook(mixer.hc_norm, "post", lambda m, a, out: rec.add("final_hc/normed", out[0]))
            self.hook(mixer.input_mix_weight_down, "post", lambda m, a, out: rec.add("final_hc/down", out[0]))
            self.hook(mixer.input_mix_weight_up, "post", lambda m, a, out: rec.add("final_hc/up", out[0]))
            self.hook(mixer, "post", lambda m, a, out: rec.add("final_mix", out[0]))
        for n, layer in fn.by_index.items():
            self._layer_hooks(n, layer)
        for name in self.GLOBALS:
            original = getattr(upstream, name)
            self.originals[name] = original
            setattr(upstream, name, getattr(self, f"_wrap_{name}")(original))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for name, original in self.originals.items():
            setattr(upstream, name, original)
        self.originals.clear()

    def _layer_hooks(self, n: int, layer) -> None:
        rec = self.rec

        def layer_in(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            rec.add(f"L{n}/in", hidden[0])
            if n == 0 and self.model_level:
                rec.add("streams0", hidden[0])

        self.hook(layer, "pre", layer_in, with_kwargs=True)
        self.hook(layer, "post", lambda m, a, out: rec.add(f"L{n}/out", out[0]))
        self.hook(layer.mlp.gate, "post", lambda m, a, out, n=n: self._router_stats(n, out))
        if n in self.capture_layers:
            self._detail_hooks(n, layer)

    def _gated_residual_hooks(self, n: int, module, tag: str, before_name: str) -> None:
        rec = self.rec
        self.hook(module, "pre", lambda m, a: rec.add(f"L{n}/{before_name}", a[0][0]))
        self.hook(module.hc_norm, "post", lambda m, a, out: rec.add(f"L{n}/{tag}/normed", out[0]))
        self.hook(module.input_mix_weight_down, "post", lambda m, a, out: rec.add(f"L{n}/{tag}/down", out[0]))
        self.hook(module.input_mix_weight_up, "post", lambda m, a, out: rec.add(f"L{n}/{tag}/up", out[0]))

        def done(m, a, out):
            rec.add(f"L{n}/{tag}/mixed", out[0][0])
            rec.add(f"L{n}/{tag}/injection", out[2][0])

        self.hook(module, "post", done)

    def _detail_hooks(self, n: int, layer) -> None:
        rec = self.rec
        self._gated_residual_hooks(n, layer.attn_hyper_connection, "attn_hc", "after_ple")
        self._gated_residual_hooks(n, layer.mlp_hyper_connection, "mlp_hc", "mid")
        if layer.layer_type == "linear_attention":
            self._gdn_hooks(n, layer.linear_attn)
        else:
            self._attention_hooks(n, layer.self_attn)
        if layer.ple is not None:
            self._ple_hooks(n, layer.ple)
        self._moe_hooks(n, layer.mlp)

    # --- GDN ---------------------------------------------------------------------------------------------------

    def _gdn_hooks(self, n: int, gdn) -> None:
        rec = self.rec

        def pre(module, args, kwargs):
            self.current_gdn = n
            hidden = args[0] if args else kwargs["hidden_states"]
            rec.add(f"L{n}/gdn/in", hidden[0])

        def post(module, args, kwargs, out):
            self.current_gdn = None
            rec.add(f"L{n}/gdn/out", out[0])
            cache = kwargs.get("cache_params")
            if cache is not None:
                state = cache.layers[n]
                rec.add(f"L{n}/gdn/conv_state", state.conv_states[0])
                rec.add(f"L{n}/gdn/recurrent_state", state.recurrent_states[0])

        self.hook(gdn, "pre", pre, with_kwargs=True)
        self.hook(gdn, "post", post, with_kwargs=True)
        for attr, name in (("in_proj_qkv", "qkv_proj"), ("in_proj_z", "z_proj"), ("in_proj_b", "b_proj"),
                           ("in_proj_a", "a_proj")):
            self.hook(getattr(gdn, attr), "post", lambda m, a, out, name=name: rec.add(f"L{n}/gdn/{name}", out[0]))
        self.hook(gdn.norm, "post", lambda m, a, out: rec.add(f"L{n}/gdn/norm_out", out.reshape(self.rows, -1)))

    def _wrap_causal_conv1d_fn(self, original):
        def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
            out = original(hidden_states, weight, bias, activation=activation, **kwargs)
            if self.current_gdn is not None:
                self.rec.add(f"L{self.current_gdn}/gdn/conv_out", out[0, :, -self.rows:].transpose(0, 1))
            return out
        return causal_conv1d_fn

    def _wrap_causal_conv1d_update(self, original):
        def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
            out = original(hidden_states, conv_state, weight, bias, activation)
            if self.current_gdn is not None:
                self.rec.add(f"L{self.current_gdn}/gdn/conv_out", out[0, :, -self.rows:].transpose(0, 1))
            return out
        return causal_conv1d_update

    def _wrap_delta(self, original):
        def delta_rule(query, key, value, g, beta, **kwargs):
            out = original(query, key, value, g=g, beta=beta, **kwargs)
            if self.current_gdn is not None:
                n = self.current_gdn
                self.rec.add(f"L{n}/gdn/beta", beta[0])
                self.rec.add(f"L{n}/gdn/g", g[0])
                self.rec.add(f"L{n}/gdn/core_out", out[0][0])
            return out
        return delta_rule

    def _wrap_torch_chunk_gated_delta_rule(self, original):
        return self._wrap_delta(original)

    def _wrap_torch_recurrent_gated_delta_rule(self, original):
        return self._wrap_delta(original)

    # --- QSA ---------------------------------------------------------------------------------------------------

    def _attention_hooks(self, n: int, attn) -> None:
        rec = self.rec
        attn.indexer.__class__ = RecordingQSAIndexer
        attn.indexer.capture = self

        def pre(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            rec.add(f"L{n}/attn/in", hidden[0])

        def post(module, args, kwargs, out):
            cache = kwargs.get("past_key_values")
            rec.add(f"L{n}/attn/out", out[0][0])
            if cache is not None:
                layer_cache = cache.layers[n]
                rec.add(f"L{n}/attn/kv_k", layer_cache.keys)
                rec.add(f"L{n}/attn/kv_v", layer_cache.values)
                rec.add(f"L{n}/attn/indexer_keys", layer_cache.indexer_keys)

        def q_proj(module, args, out):
            rec.add(f"L{n}/attn/q_proj", out[0])
            if rec.wants(f"L{n}/attn/gate"):
                rows = out.shape[1]
                rec.add(f"L{n}/attn/gate", out.view(1, rows, -1, 2 * attn.head_dim)[..., attn.head_dim:].reshape(rows, -1))

        self.hook(attn, "pre", pre, with_kwargs=True)
        self.hook(attn, "post", post, with_kwargs=True)
        self.hook(attn.q_proj, "post", q_proj)
        self.hook(attn.o_proj, "pre", lambda m, a: rec.add(f"L{n}/attn/gated", a[0][0]))

    def _wrap_eager_attention_forward(self, original):
        def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
            out = original(module, query, key, value, attention_mask, scaling, dropout, **kwargs)
            n = getattr(module, "layer_idx", None)
            if n in self.capture_layers:
                self.rec.add(f"L{n}/attn/q", query[0].transpose(0, 1))
                self.rec.add(f"L{n}/attn/core", out[0][0].reshape(query.shape[2], -1))
            return out
        return eager_attention_forward

    def attach_cache(self, cache: DynamicCache) -> None:
        """Wraps each indexed-attention cache layer's `update`: records k and v (after k_norm and RoPE) and applies the
        KV codec when `kv_format` is nvfp4."""
        rec = self.rec
        for n in self.fn.layers:
            if self.fn.by_index[n].layer_type != "indexed_attention":
                continue
            layer_cache = cache.layers[n]
            original = layer_cache.update

            def update(key_states, value_states, *args, n=n, original=original, **kwargs):
                if n in self.capture_layers:
                    rec.add(f"L{n}/attn/k", key_states[0].transpose(0, 1))
                    rec.add(f"L{n}/attn/v", value_states[0].transpose(0, 1))
                if self.kv_format == "nvfp4":
                    key_states = kvcodec.roundtrip_cache(key_states)
                    value_states = kvcodec.roundtrip_cache(value_states)
                return original(key_states, value_states, *args, **kwargs)

            layer_cache.update = update

    # --- PLE ---------------------------------------------------------------------------------------------------

    def _ple_hooks(self, n: int, ple) -> None:
        rec = self.rec
        ple.__class__ = RecordingPLELayer
        ple.capture = self
        embedding = ple.ple_embedding
        self.hook(embedding.ngram_embedding, "pre", lambda m, a: rec.add(f"L{n}/ple/ids", a[0][0]))
        self.hook(embedding.ngram_embedding, "post", lambda m, a, out: rec.add(f"L{n}/ple/rows", out[0]))
        self.hook(embedding, "post", lambda m, a, out: rec.add(f"L{n}/ple/emb", out[0]))
        self.hook(ple.key_proj, "post", lambda m, a, out: rec.add(f"L{n}/ple/key_proj", out[0]))

    # --- MoE ---------------------------------------------------------------------------------------------------

    def _router_stats(self, n: int, out) -> None:
        logits, scores, indices = out
        k = indices.shape[-1]
        probs = torch.softmax(logits, dtype=torch.float, dim=-1)
        stable_index, _, tie = stable_topk(probs, k)
        top_values, top_index = torch.topk(probs, k, dim=-1)
        stats = self.tie_stats
        stats["router_calls"] += 1
        stats["rows"] += logits.shape[0]
        order = (stable_index != indices).any(dim=-1)
        sets = (stable_index.sort(dim=-1).values != indices.sort(dim=-1).values).any(dim=-1)
        stats["boundary_tie_rows"] += int(tie.sum())
        stats["order_differs_rows"] += int(order.sum())
        stats["set_differs_rows"] += int(sets.sum())
        stats["topk_nondeterministic_rows"] += int((top_index != indices).any(dim=-1).sum())
        if tie.any():
            stats["layers_with_tie"].add(n)
        if order.any():
            stats["layers_with_order_diff"].add(n)
        if sets.any():
            stats["layers_with_set_diff"].add(n)
        if n in self.capture_layers:
            rec = self.rec
            rec.add(f"L{n}/moe/router_logits", logits)
            rec.add(f"L{n}/moe/topk_ids", indices.to(torch.int32))
            rec.add(f"L{n}/moe/topk_probs_raw", top_values)
            rec.add(f"L{n}/moe/topk_weights", scores)
            rec.add(f"L{n}/moe/boundary_tie", tie.to(torch.uint8))
            rec.add(f"L{n}/moe/topk_stable_order", stable_index.to(torch.int32))

    def _moe_hooks(self, n: int, moe) -> None:
        rec = self.rec

        def pre(module, args):
            hidden = args[0]
            rec.add(f"L{n}/moe/in", hidden.reshape(-1, hidden.shape[-1]))

        def post(module, args, out):
            rec.add(f"L{n}/moe/out", out.reshape(-1, out.shape[-1]))

        self.hook(moe, "pre", pre)
        self.hook(moe, "post", post)
        self.hook(moe.shared_expert, "post", lambda m, a, out: rec.add(f"L{n}/moe/shared_out", out))
        self.hook(moe.shared_expert_gate, "post",
                  lambda m, a, out: rec.add(f"L{n}/moe/shared_gate", F.sigmoid(out)))
        self.hook(moe.experts, "post", lambda m, a, out: rec.add(f"L{n}/moe/routed_sum", out))

        def observer(expert, tokens, out, weighted):
            if expert is None:
                rec.add(f"L{n}/moe/expert_ids", torch.tensor(tokens, dtype=torch.int32))
                return
            rec.add(f"L{n}/moe/expert/{expert}/tokens", tokens.to(torch.int32))
            rec.add(f"L{n}/moe/expert/{expert}/out", out)
            rec.add(f"L{n}/moe/expert/{expert}/weighted", weighted)

        moe.experts.observer = observer

    def stats_metadata(self) -> dict:
        s = dict(self.tie_stats)
        for key in ("layers_with_order_diff", "layers_with_set_diff", "layers_with_tie"):
            s[key] = sorted(s[key])
        return {"router_tie_stats": s, "replays_checked": dict(self.replays)}
