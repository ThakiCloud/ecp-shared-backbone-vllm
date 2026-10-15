"""ECP Shared-Backbone (SB1): one vLLM generate engine = original Qwen3.5 (all layers, LM head untouched)
+ the Q12 evidence-gate branch: layers G (Q12: 44-47) re-run from the layer-(G0-1) output with LoRA-merged MLPs,
final norm, fp32 action head.

How a request becomes a gate request: its prompt ends with the Q12 prompt tail (config.ecp_gate_signature, token ids).
For the token that completes the signature the engine replaces the LM logits with -inf everywhere except the three
action label tokens (config.ecp_gate_label_ids), which carry the head's fp32 logits. Clients send max_tokens=1,
logprobs>=3 and read log p(action) = log_softmax(head logits) from the label tokens. Generation requests never end with
the signature and get the original logits.

The branch's attention/GDN modules are separate layers (prefix ...gate_branch.layers.{64+j}) so vLLM gives them their
own KV/GDN state; nothing the branch computes is written into the generation path's state. SB1 loads the branch's
attention/GDN weights as a copy (4 layers); tying them to the main layers is a later memory optimisation.
SB1 runs the branch for every scheduled token (gate or not); restricting it to gate requests is SB3 work.
"""
from __future__ import annotations

import os
import re
import types

import torch
from torch import nn

from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5ForCausalLM, Qwen3_5Model
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

LORA_RE = re.compile(r"layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.lora_([AB])\.[^.]+\.weight$")
TEXT_LAYER_RE = re.compile(r"^(?:model\.language_model\.|model\.)layers\.(\d+)\.(.+)$")
EXTRA = 8  # bf16 slots that carry 4 fp32 values (3 gate logits + flag)


def read_q12(path):
    from safetensors import safe_open
    lora = {}
    with safe_open(path, "pt") as f:
        for k in f.keys():
            m = LORA_RE.search(k) if k.startswith("adapter.") else None
            if m:
                lora.setdefault((int(m.group(1)), m.group(2)), {})[m.group(3)] = f.get_tensor(k).float()
        hw, hb = f.get_tensor("head.weight").float(), f.get_tensor("head.bias").float()
    return lora, hw, hb


class GateBranch(nn.Module):
    def __init__(self, vllm_config, cfg, layers, prefix):
        super().__init__()
        self.idx = tuple(layers)
        self.layers = nn.ModuleDict({
            str(i): Qwen3_5DecoderLayer(vllm_config, layer_type=cfg.layer_types[i],
                                        prefix=f"{prefix}.layers.{cfg.num_hidden_layers + j}")
            for j, i in enumerate(self.idx)})

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=Qwen3_5Model.hf_to_vllm_mapper)


@torch.library.custom_op("ecp_sb::stash", mutates_args=("bh", "br"))
def stash(bh: torch.Tensor, br: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> None:
    """Opaque copy of the branch-point (hidden, residual) into persistent buffers. Nothing leaves the compiled graph as
    an output, so the generation path's graph keeps its outputs; only a read is added (SB1 capture fix candidate)."""
    bh[: h.shape[0]].copy_(h)
    br[: r.shape[0]].copy_(r)


@stash.register_fake
def _(bh, br, h, r):
    return None


def _capture_stash(self, aux, layer_idx, hidden_states, residual):
    if layer_idx in self.aux_hidden_state_layers:
        torch.ops.ecp_sb.stash(self.ecp_bh, self.ecp_br, hidden_states, residual)
    return aux


def _capture_clone(self, aux, layer_idx, hidden_states, residual):
    if layer_idx in self.aux_hidden_state_layers:
        aux.append(hidden_states.clone())
        aux.append(residual.clone())
    return aux


def _wrap_post_norm(norm, owner):
    """Layer G0's post_attention_layernorm: stash its inputs (token-mixer output, residual), then run it unchanged.
    Instance forward is replaced (module tree and weight names untouched). Both inputs are materialised in the stock
    graph too (out_proj output; residual crosses the piecewise split at the attention op)."""
    orig = norm.forward

    def forward(x, residual=None):
        torch.ops.ecp_sb.stash(owner.ecp_bh, owner.ecp_br, x, residual)
        return orig(x, residual)
    norm.forward = forward


def _capture_branch_input(self, aux, layer_idx, hidden_states, residual):
    """Replaces Qwen3NextModel._maybe_add_hidden_state on the instance: keep hidden AND residual (not their sum) at the
    branch point, so the branch's first layer sees exactly what the main layer sees (fused add + norm)."""
    if layer_idx in self.aux_hidden_state_layers:
        aux.append(hidden_states)
        aux.append(residual)
    return aux


SKIP = ["mtp.", "model.visual.", "visual.", "model.mtp."]


class ECPStockQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Unmodified Qwen3_5ForCausalLM; only skips vision/MTP tensors of a ConditionalGeneration checkpoint."""

    def load_weights(self, weights):
        return AutoWeightsLoader(self, skip_prefixes=SKIP).load_weights(weights, mapper=self.hf_to_vllm_mapper)


class ECPSharedQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        cfg = self.config
        self.gate_idx = tuple(int(i) for i in cfg.ecp_gate_layers)
        assert self.gate_idx == tuple(range(self.gate_idx[0], self.gate_idx[-1] + 1)), "gate layers must be contiguous"
        # ECP_SB_MODE (diagnostic, SB1): off = stock behaviour; aux = capture only; branch = run branch, drop it; full;
        # nostate = off without even constructing the branch layers (no branch KV/GDN state, no branch weights)
        self.sb_mode = os.environ.get("ECP_SB_MODE", "full")
        self.gate_branch = (None if self.sb_mode == "nostate" else
                            GateBranch(vllm_config, cfg, self.gate_idx, maybe_prefix(prefix, "gate_branch")))
        self.gate_head = nn.Linear(cfg.hidden_size, 3, bias=True, dtype=torch.float32)
        self.register_buffer("gate_sig", torch.tensor(list(cfg.ecp_gate_signature), dtype=torch.long), persistent=False)
        self.register_buffer("gate_labels", torch.tensor(list(cfg.ecp_gate_label_ids), dtype=torch.long), persistent=False)
        self.q12_file = cfg.ecp_q12_file
        self.lora_alpha = float(getattr(cfg, "ecp_lora_alpha", 16.0))  # scale = alpha / rank (rank read from the file)
        # ECP_SB_CAPTURE: post (default, validated) | list (graph outputs; SB1 FAIL) | clone | stash (diagnostic, FAIL on 27B)
        self.capture = os.environ.get("ECP_SB_CAPTURE", "post")  # post = the validated SB1 capture (attempt 3)
        if self.sb_mode not in ("off", "nostate"):
            if self.capture != "post":  # post stashes inside layer G0; no aux outputs at all
                self.model.aux_hidden_state_layers = (self.gate_idx[0],)
            fn = {"list": _capture_branch_input, "clone": _capture_clone, "stash": _capture_stash,
                  "post": None}[self.capture]
            if self.capture in ("stash", "post"):
                n = vllm_config.scheduler_config.max_num_batched_tokens
                dt = vllm_config.model_config.dtype
                self.model.register_buffer("ecp_bh", torch.zeros(n, cfg.hidden_size, dtype=dt), persistent=False)
                self.model.register_buffer("ecp_br", torch.zeros(n, cfg.hidden_size, dtype=dt), persistent=False)
            if fn is not None:
                self.model._maybe_add_hidden_state = types.MethodType(fn, self.model)

    # ---- forward -------------------------------------------------------------------------------
    def _flags(self, input_ids):
        T, K = input_ids.shape[0], self.gate_sig.shape[0]
        flag = torch.zeros(T, dtype=torch.float32, device=input_ids.device)
        if T >= K:
            m = (input_ids.unfold(0, K, 1) == self.gate_sig).all(1)
            flag[K - 1:] = m.float()
        return flag

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        if self.sb_mode in ("off", "nostate"):
            return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        first = 0
        if self.capture in ("stash", "post"):
            h = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
            T = h.shape[0]
            g, r = self.model.ecp_bh[:T].clone(), self.model.ecp_br[:T].clone()
            if self.capture == "post":  # branch layer G0 = main token mixer output -> branch post-norm + LoRA MLP
                L0 = self.gate_branch.layers[str(self.gate_idx[0])]
                if self.sb_mode != "aux":
                    g, r = L0.post_attention_layernorm(g, r)
                    g = L0.mlp(g)
                first = 1
        else:
            h, aux = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
            g, r = aux[0], aux[1]
        if self.sb_mode == "aux":
            return h
        for i in self.gate_idx[first:]:
            g, r = self.gate_branch.layers[str(i)](positions=positions, hidden_states=g, residual=r)
        g, _ = self.model.norm(g, r)
        logits = self.gate_head(g.float())
        if self.sb_mode == "branch":
            return h
        flag = self._flags(input_ids) if input_ids is not None else torch.zeros(h.shape[0], device=h.device)
        extra = torch.cat([logits, flag[:, None]], 1).contiguous().view(h.dtype)  # bit-carried, never computed on
        return torch.cat([h, extra], 1)

    def compute_logits(self, hidden_states):
        H = self.config.hidden_size
        if hidden_states.shape[-1] == H:  # runner dummy paths
            return super().compute_logits(hidden_states)
        main = hidden_states[:, :H].contiguous()  # same memory layout as the stock lm_head input
        extra = hidden_states[:, H:].contiguous().view(torch.float32)
        logits = super().compute_logits(main)
        if logits is None:
            return None
        logits = logits.float()
        gate = torch.full_like(logits, float("-inf"))
        gate[:, self.gate_labels] = extra[:, :3]
        return torch.where(extra[:, 3:4] > 0.5, gate, logits)

    # ---- weights -------------------------------------------------------------------------------
    def load_weights(self, weights):
        lora, hw, hb = read_q12(self.q12_file)
        G = set(self.gate_idx)
        ranks = {v["A"].shape[0] for v in lora.values()}
        if len(ranks) != 1:
            raise ValueError(f"inconsistent LoRA rank {ranks}")
        scale = self.lora_alpha / ranks.pop()
        if {i for i, _ in lora} != G:
            raise ValueError(f"Q12 LoRA layers {sorted({i for i, _ in lora})} != gate layers {sorted(G)}")
        branch = []

        def stream():
            for name, t in weights:
                yield name, t
                m = TEXT_LAYER_RE.match(name)
                if m and int(m.group(1)) in G:
                    i, rest = int(m.group(1)), m.group(2)
                    mm = re.match(r"mlp\.(gate_proj|up_proj|down_proj)\.weight$", rest)
                    if mm:
                        ab = lora[(i, mm.group(1))]
                        t = (t.float() + scale * (ab["B"] @ ab["A"])).to(t.dtype)  # == q12_vllm_convert merge
                    branch.append((f"layers.{i}.{rest}", t))
        loaded = AutoWeightsLoader(self, skip_prefixes=SKIP + ["gate_branch.", "gate_head."]
                                   ).load_weights(stream(), mapper=self.hf_to_vllm_mapper)
        if self.gate_branch is not None:
            loaded |= {f"gate_branch.{n}" for n in self.gate_branch.load_weights(branch)}
        if self.capture == "post" and self.sb_mode not in ("off", "nostate"):
            L = self.model.layers[self.gate_idx[0]]
            assert not getattr(L, "layer_scale", False), "post capture assumes no layer_scale"
            _wrap_post_norm(L.post_attention_layernorm, self.model)
        with torch.no_grad():
            self.gate_head.weight.copy_(hw); self.gate_head.bias.copy_(hb)
        loaded |= {"gate_head.weight", "gate_head.bias"}
        return loaded
