ENGINE_TEMPLATE = r'''from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


STRATEGY = __STRATEGY_JSON__


def _load_state_dict(weight_path):
    try:
        return torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location="cpu")


if triton is not None:
    @triton.jit
    def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, hidden_size: tl.constexpr, eps: tl.constexpr, block: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < hidden_size
        row_offset = row * hidden_size + offsets
        x = tl.load(x_ptr + row_offset, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / hidden_size
        y = x * tl.rsqrt(variance + eps) * w
        tl.store(y_ptr + row_offset, y, mask=mask)


class RequestState:
    def __init__(self, *, length, tokens=None, k_cache=None, v_cache=None, capacity=0):
        self.length = int(length)
        self.tokens = tokens
        self.k_cache = k_cache or []
        self.v_cache = v_cache or []
        self.capacity = int(capacity)


class Engine:
    def __init__(self, model_config, weight_dir, device="cuda"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.config = dict(model_config)
        self.device = torch.device(device)
        self.dtype = self._select_dtype(self.config)
        self.num_layers = int(self.config["num_hidden_layers"])
        self.num_heads = int(self.config["num_attention_heads"])
        self.num_kv_heads = int(self.config["num_key_value_heads"])
        self.head_dim = int(self.config.get("head_dim", int(self.config["hidden_size"]) // self.num_heads))
        self.hidden_size = int(self.config["hidden_size"])
        self.intermediate_size = int(self.config["intermediate_size"])
        self.vocab_size = int(self.config["vocab_size"])
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.eps = float(self.config.get("rms_norm_eps", 1e-5))
        self.rope_theta = float(self.config.get("rope_theta", 10000.0))
        self.prefill_policy = STRATEGY["prefill_policy"]
        self.kv_policy = STRATEGY["kv_policy"]
        self.attention_policy = STRATEGY["attention_policy"]
        self.decode_attention_policy = STRATEGY["decode_attention_policy"]
        self.cache_growth_policy = STRATEGY["cache_growth_policy"]
        self.cache_layout_policy = STRATEGY["cache_layout_policy"]
        self.norm_policy = STRATEGY["norm_policy"]
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.kv_repeat = self.num_heads // self.num_kv_heads
        weight_path = os.path.join(weight_dir, "model.pt")
        state_dict = _load_state_dict(weight_path)
        self.w = {
            name: tensor.to(device=self.device, dtype=self.dtype)
            for name, tensor in state_dict.items()
        }
        self.layers = []
        for layer_idx in range(self.num_layers):
            prefix = "layers.%d" % layer_idx
            qkv_weight = torch.cat(
                [
                    self.w[prefix + ".self_attn.q_proj.weight"],
                    self.w[prefix + ".self_attn.k_proj.weight"],
                    self.w[prefix + ".self_attn.v_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            gate_up_weight = torch.cat(
                [
                    self.w[prefix + ".mlp.gate_proj.weight"],
                    self.w[prefix + ".mlp.up_proj.weight"],
                ],
                dim=0,
            ).contiguous()
            self.layers.append(
                (
                    self.w[prefix + ".input_layernorm.weight"],
                    qkv_weight,
                    self.w[prefix + ".self_attn.o_proj.weight"],
                    self.w[prefix + ".post_attention_layernorm.weight"],
                    gate_up_weight,
                    self.w[prefix + ".mlp.down_proj.weight"],
                )
            )
        self.requests = {}
        self._rope_cache_len = 0
        self._rope_cos = None
        self._rope_sin = None
        initial_rope = int(self.config.get("max_position_embeddings", 0))
        self._ensure_rope_cache(max(16, min(max(initial_rope, 16), 8192)))

    def _select_dtype(self, config):
        if self.device.type != "cuda":
            return torch.float32
        dtype = str(config.get("torch_dtype", "float16")).lower()
        if dtype in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float16

    def _ensure_rope_cache(self, needed_len):
        needed_len = int(needed_len)
        if self._rope_cache_len >= needed_len:
            return
        new_len = max(needed_len, max(16, self._rope_cache_len * 2))
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=self.device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(new_len, device=self.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self._rope_cos = freqs.cos().to(self.dtype)
        self._rope_sin = freqs.sin().to(self.dtype)
        self._rope_cache_len = new_len

    def _rmsnorm(self, x, weight):
        if self.norm_policy == "triton_rmsnorm" and x.is_cuda and triton is not None:
            return self._triton_rmsnorm(x, weight)
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.eps)
        return x_norm.to(x.dtype) * weight

    def _triton_rmsnorm(self, x, weight):
        original_shape = x.shape
        x_flat = x.contiguous().view(-1, self.hidden_size)
        y = torch.empty_like(x_flat)
        block = triton.next_power_of_2(self.hidden_size)
        _rmsnorm_kernel[(x_flat.shape[0],)](x_flat, weight, y, self.hidden_size, self.eps, block)
        return y.view(original_shape)

    def _apply_rope_batch_seq(self, q, k, seqlen):
        self._ensure_rope_cache(seqlen)
        cos = self._rope_cos[:seqlen][None, None, :, :]
        sin = self._rope_sin[:seqlen][None, None, :, :]

        def rotate(x):
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rotated = torch.stack(
                (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
                dim=-1,
            )
            return rotated.flatten(-2)

        return rotate(q), rotate(k)

    def _apply_rope_decode(self, q, k, positions):
        positions = positions.to(device=self.device, dtype=torch.long)
        self._ensure_rope_cache(int(positions.max().item()) + 1 if positions.numel() else 1)
        cos = self._rope_cos.index_select(0, positions)[:, None, None, :]
        sin = self._rope_sin.index_select(0, positions)[:, None, None, :]

        def rotate(x):
            x_even = x[..., 0::2]
            x_odd = x[..., 1::2]
            rotated = torch.stack(
                (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
                dim=-1,
            )
            return rotated.flatten(-2)

        return rotate(q), rotate(k)

    def _forward_batch(self, input_ids, *, collect_cache=False, last_only=False, last_indices=None):
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        x = self.w["embed_tokens.weight"][input_ids]
        batch, seqlen, _ = x.shape
        causal_mask = None
        if self.attention_policy != "sdpa_prefill_only":
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=self.device, dtype=torch.float32),
                diagonal=1,
            )[None, None, :, :]
        collected_k = []
        collected_v = []
        for layer_idx, layer in enumerate(self.layers):
            input_norm, qkv_weight, o_proj, post_norm, gate_up_proj, down_proj = layer
            residual = x
            x_norm = self._rmsnorm(x, input_norm)
            qkv = F.linear(x_norm, qkv_weight)
            q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
            q = q.view(batch, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope_batch_seq(q, k, seqlen)
            if collect_cache:
                collected_k.append(k.detach())
                collected_v.append(v.detach())
            if self.attention_policy == "sdpa_prefill_only":
                if self.kv_repeat != 1:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                else:
                    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                attn_k = k.repeat_interleave(self.kv_repeat, dim=1) if self.kv_repeat != 1 else k
                attn_v = v.repeat_interleave(self.kv_repeat, dim=1) if self.kv_repeat != 1 else v
                attn = torch.matmul(q.float(), attn_k.float().transpose(-1, -2))
                attn = attn / math.sqrt(self.head_dim)
                attn = attn + causal_mask
                attn = torch.softmax(attn, dim=-1).to(x.dtype)
                y = torch.matmul(attn, attn_v)
            y = y.transpose(1, 2).contiguous().view(batch, seqlen, self.hidden_size)
            y = F.linear(y, o_proj)
            x = residual + y
            residual = x
            x_norm = self._rmsnorm(x, post_norm)
            gate_up = F.linear(x_norm, gate_up_proj)
            gate, up = torch.split(gate_up, (self.intermediate_size, self.intermediate_size), dim=-1)
            hidden = F.silu(gate) * up
            mlp_out = F.linear(hidden, down_proj)
            x = residual + mlp_out
        if last_only:
            if last_indices is None:
                x = x[:, -1, :]
            else:
                gather_index = last_indices.to(device=self.device, dtype=torch.long).view(batch, 1, 1)
                gather_index = gather_index.expand(-1, 1, self.hidden_size)
                x = x.gather(dim=1, index=gather_index).squeeze(1)
            x = self._rmsnorm(x, self.w["norm.weight"])
        else:
            x = self._rmsnorm(x, self.w["norm.weight"])
        logits = F.linear(x, self.w["lm_head.weight"])
        return logits, collected_k, collected_v

    def _state_from_cache(self, tokens, layer_k, layer_v, batch_index):
        length = int(tokens.numel())
        capacity = self._initial_cache_capacity(length)
        k_cache = []
        v_cache = []
        for layer_idx in range(self.num_layers):
            k_src = layer_k[layer_idx][batch_index, :, :length, :].contiguous()
            v_src = layer_v[layer_idx][batch_index, :, :length, :].contiguous()
            if self.kv_policy == "per_request_prealloc":
                if self.cache_layout_policy == "transposed_k":
                    k_dst = torch.empty((self.num_kv_heads, self.head_dim, capacity), device=self.device, dtype=self.dtype)
                    k_dst[:, :, :length] = k_src.transpose(1, 2)
                else:
                    k_dst = torch.empty((self.num_kv_heads, capacity, self.head_dim), device=self.device, dtype=self.dtype)
                    k_dst[:, :length, :] = k_src
                v_dst = torch.empty((self.num_kv_heads, capacity, self.head_dim), device=self.device, dtype=self.dtype)
                v_dst[:, :length, :] = v_src
                k_cache.append(k_dst)
                v_cache.append(v_dst)
            else:
                k_cache.append(k_src.clone())
                v_cache.append(v_src.clone())
        return RequestState(length=length, tokens=None, k_cache=k_cache, v_cache=v_cache, capacity=capacity)

    def _initial_cache_capacity(self, length):
        if self.kv_policy != "per_request_prealloc":
            return length
        if self.cache_growth_policy.startswith("decode_slack_"):
            return max(1, length + int(self.cache_growth_policy.rsplit("_", 1)[1]))
        return max(1, 1 << (length).bit_length())

    def _grown_cache_capacity(self, current_capacity, needed):
        if self.cache_growth_policy.startswith("decode_slack_"):
            return max(needed, current_capacity + int(self.cache_growth_policy.rsplit("_", 1)[1]))
        return max(needed, max(1, current_capacity * 2))

    def _ensure_state_capacity(self, state, needed):
        if self.kv_policy != "per_request_prealloc" or state.capacity >= needed:
            return
        new_capacity = self._grown_cache_capacity(state.capacity, needed)
        new_k = []
        new_v = []
        for layer_idx in range(self.num_layers):
            k_old = state.k_cache[layer_idx]
            v_old = state.v_cache[layer_idx]
            if self.cache_layout_policy == "transposed_k":
                k_dst = torch.empty((self.num_kv_heads, self.head_dim, new_capacity), device=self.device, dtype=self.dtype)
            else:
                k_dst = torch.empty((self.num_kv_heads, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
            v_dst = torch.empty((self.num_kv_heads, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
            if state.length:
                if self.cache_layout_policy == "transposed_k":
                    k_dst[:, :, :state.length] = k_old[:, :, :state.length]
                else:
                    k_dst[:, :state.length, :] = k_old[:, :state.length, :]
                v_dst[:, :state.length, :] = v_old[:, :state.length, :]
            new_k.append(k_dst)
            new_v.append(v_dst)
        state.k_cache = new_k
        state.v_cache = new_v
        state.capacity = new_capacity

    def _append_layer_cache(self, state, layer_idx, k_new, v_new):
        pos = state.length
        if self.kv_policy == "per_request_prealloc":
            if self.cache_layout_policy == "transposed_k":
                state.k_cache[layer_idx][:, :, pos : pos + 1] = k_new.transpose(1, 2)
            else:
                state.k_cache[layer_idx][:, pos : pos + 1, :] = k_new
            state.v_cache[layer_idx][:, pos : pos + 1, :] = v_new
        else:
            state.k_cache[layer_idx] = torch.cat([state.k_cache[layer_idx], k_new], dim=1)
            state.v_cache[layer_idx] = torch.cat([state.v_cache[layer_idx], v_new], dim=1)

    def _store_prefill_states(self, request_ids, tokens, layer_k, layer_v, collect_cache):
        for index, rid in enumerate(request_ids):
            rid = int(rid)
            if collect_cache:
                self.requests[rid] = self._state_from_cache(tokens[index], layer_k, layer_v, index)
            else:
                self.requests[rid] = RequestState(length=int(tokens[index].numel()), tokens=tokens[index].detach().clone())

    def _prefill_same_length(self, request_ids, input_ids):
        if not request_ids:
            return torch.empty((0, self.vocab_size), device=self.device, dtype=self.dtype)
        tokens = [ids.to(device=self.device, dtype=torch.long) for ids in input_ids]
        batch_ids = torch.stack(tokens, dim=0)
        collect_cache = self.kv_policy != "none"
        out, layer_k, layer_v = self._forward_batch(batch_ids, collect_cache=collect_cache, last_only=True)
        self._store_prefill_states(request_ids, tokens, layer_k, layer_v, collect_cache)
        return out

    def _prefill_padded(self, request_ids, input_ids, lengths):
        if not request_ids:
            return torch.empty((0, self.vocab_size), device=self.device, dtype=self.dtype)
        tokens = [ids.to(device=self.device, dtype=torch.long) for ids in input_ids]
        batch = len(tokens)
        max_len = max(lengths)
        batch_ids = torch.zeros((batch, max_len), device=self.device, dtype=torch.long)
        for index, ids in enumerate(tokens):
            batch_ids[index, : lengths[index]] = ids
        collect_cache = self.kv_policy != "none"
        last_indices = torch.tensor([length - 1 for length in lengths], device=self.device, dtype=torch.long)
        out, layer_k, layer_v = self._forward_batch(
            batch_ids,
            collect_cache=collect_cache,
            last_only=True,
            last_indices=last_indices,
        )
        self._store_prefill_states(request_ids, tokens, layer_k, layer_v, collect_cache)
        return out

    @torch.inference_mode()
    def prefill(self, request_ids, input_ids):
        request_ids = [int(rid) for rid in request_ids]
        if len(request_ids) != len(input_ids):
            raise ValueError("request_ids and input_ids must have the same length")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a prefill call")
        if not request_ids:
            return torch.empty((0, self.vocab_size), device=self.device, dtype=self.dtype)
        lengths = [int(ids.numel()) for ids in input_ids]
        if any(length <= 0 for length in lengths):
            raise ValueError("prefill input_ids must be non-empty 1D tensors")
        if self.prefill_policy in ("pad_batch", "group_by_length") and len(set(lengths)) == 1:
            return self._prefill_same_length(request_ids, input_ids)
        if self.prefill_policy == "pad_batch":
            return self._prefill_padded(request_ids, input_ids, lengths)
        if self.prefill_policy == "group_by_length":
            outputs = [None] * len(request_ids)
            by_len = {}
            for index, length in enumerate(lengths):
                by_len.setdefault(length, []).append(index)
            for group in by_len.values():
                logits = self._prefill_same_length(
                    [request_ids[index] for index in group],
                    [input_ids[index] for index in group],
                )
                for local_index, original_index in enumerate(group):
                    outputs[original_index] = logits[local_index]
            return torch.stack(outputs, dim=0)
        outputs = []
        for rid, ids in zip(request_ids, input_ids):
            outputs.append(self._prefill_same_length([rid], [ids])[0])
        return torch.stack(outputs, dim=0)

    def _decode_recompute(self, request_ids, token_ids):
        outputs = []
        for index, rid in enumerate(request_ids):
            state = self.requests[int(rid)]
            token = token_ids[index].to(device=self.device, dtype=torch.long).reshape(1)
            tokens = torch.cat([state.tokens.to(device=self.device, dtype=torch.long), token], dim=0)
            logits, _, _ = self._forward_batch(tokens.view(1, -1), collect_cache=False, last_only=True)
            state.tokens = tokens.detach().clone()
            state.length = int(tokens.numel())
            outputs.append(logits[0])
        return torch.stack(outputs, dim=0)

    def _decode_with_cache(self, request_ids, token_ids):
        batch = len(request_ids)
        states = [self.requests[int(rid)] for rid in request_ids]
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        for state in states:
            self._ensure_state_capacity(state, state.length + 1)
        x = self.w["embed_tokens.weight"][token_ids].view(batch, 1, self.hidden_size)
        positions = torch.tensor([state.length for state in states], device=self.device, dtype=torch.long)
        length_groups = {}
        for batch_index, state in enumerate(states):
            length_groups.setdefault(state.length + 1, []).append(batch_index)
        if len(length_groups) == 1:
            next_length, group_indices = next(iter(length_groups.items()))
            decode_groups = [(next_length, None, group_indices)]
        else:
            decode_groups = [
                (
                    next_length,
                    torch.tensor(group_indices, device=self.device, dtype=torch.long),
                    group_indices,
                )
                for next_length, group_indices in length_groups.items()
            ]
        for layer_idx, layer in enumerate(self.layers):
            input_norm, qkv_weight, o_proj, post_norm, gate_up_proj, down_proj = layer
            residual = x
            x_norm = self._rmsnorm(x, input_norm)
            qkv = F.linear(x_norm, qkv_weight)
            q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
            q = q.view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q, k = self._apply_rope_decode(q, k, positions)
            for batch_index, state in enumerate(states):
                k_new = k[batch_index, :, :, :]
                v_new = v[batch_index, :, :, :]
                self._append_layer_cache(state, layer_idx, k_new, v_new)
            y = None if len(decode_groups) == 1 else torch.empty((batch, 1, self.hidden_size), device=self.device, dtype=x.dtype)
            for next_length, index_tensor, group_indices in decode_groups:
                q_group = q if index_tensor is None else q.index_select(0, index_tensor)
                if self.cache_layout_policy == "transposed_k":
                    k_hist = torch.stack(
                        [states[group_index].k_cache[layer_idx][:, :, :next_length] for group_index in group_indices],
                        dim=0,
                    )
                else:
                    k_hist = torch.stack(
                        [states[group_index].k_cache[layer_idx][:, :next_length, :] for group_index in group_indices],
                        dim=0,
                    )
                v_hist = torch.stack(
                    [states[group_index].v_cache[layer_idx][:, :next_length, :] for group_index in group_indices],
                    dim=0,
                )
                if self.kv_repeat != 1:
                    if self.decode_attention_policy != "sdpa_by_length":
                        k_hist = k_hist.repeat_interleave(self.kv_repeat, dim=1)
                        v_hist = v_hist.repeat_interleave(self.kv_repeat, dim=1)
                if self.decode_attention_policy == "sdpa_by_length":
                    if self.cache_layout_policy == "transposed_k":
                        k_hist = k_hist.transpose(-1, -2).contiguous()
                    if self.kv_repeat != 1:
                        y_group = F.scaled_dot_product_attention(q_group, k_hist, v_hist, is_causal=False, enable_gqa=True)
                    else:
                        y_group = F.scaled_dot_product_attention(q_group, k_hist, v_hist, is_causal=False)
                else:
                    if self.cache_layout_policy == "transposed_k":
                        attn = torch.matmul(q_group.float(), k_hist.float())
                    else:
                        attn = torch.matmul(q_group.float(), k_hist.float().transpose(-1, -2))
                    attn = attn / math.sqrt(self.head_dim)
                    attn = torch.softmax(attn, dim=-1).to(x.dtype)
                    y_group = torch.matmul(attn, v_hist)
                y_group = y_group.transpose(1, 2).contiguous().view(len(group_indices), 1, self.hidden_size)
                if index_tensor is None:
                    y = y_group
                else:
                    y.index_copy_(0, index_tensor, y_group)
            y = F.linear(y, o_proj)
            x = residual + y
            residual = x
            x_norm = self._rmsnorm(x, post_norm)
            gate_up = F.linear(x_norm, gate_up_proj)
            gate, up = torch.split(gate_up, (self.intermediate_size, self.intermediate_size), dim=-1)
            hidden = F.silu(gate) * up
            mlp_out = F.linear(hidden, down_proj)
            x = residual + mlp_out
        x = self._rmsnorm(x, self.w["norm.weight"])
        logits = F.linear(x, self.w["lm_head.weight"])[:, 0, :]
        for state in states:
            state.length += 1
        return logits

    @torch.inference_mode()
    def decode(self, request_ids, token_ids):
        request_ids = [int(rid) for rid in request_ids]
        if len(request_ids) != int(token_ids.numel()):
            raise ValueError("request_ids and token_ids must have the same length")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request_ids must be unique within a decode call")
        if not request_ids:
            return torch.empty((0, self.vocab_size), device=self.device, dtype=self.dtype)
        for rid in request_ids:
            if rid not in self.requests:
                raise KeyError("decode called for unknown request_id %r" % (rid,))
        if self.kv_policy == "none":
            return self._decode_recompute(request_ids, token_ids)
        return self._decode_with_cache(request_ids, token_ids)

    def remove(self, request_ids):
        for rid in request_ids:
            self.requests.pop(int(rid), None)


def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(model_config, weight_dir, device)
'''
