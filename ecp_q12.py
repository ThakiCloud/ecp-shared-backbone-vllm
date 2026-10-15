"""ECP-Q12-27B-Research-Preview loader: evidence-decision readout on a frozen Qwen3.8-27B (research preview, not production certified).

This is NOT a text generator. `ECPQ12.decide_gated()` returns one of ANSWER / RETRIEVE_MORE / ABSTAIN with probabilities, a margin and the seed's sealed tau,
computed from a LoRA adapter (r 8, alpha 16, MLP of decoders 44-47) and a linear action head read at the last
prompt position of the hidden state after 48 decoders (the stack is truncated after decoder 47). Use the
unmodified backbone separately if you also want to generate an answer.

Prompt strings below are a byte-for-byte copy of the training contract (checked by the repo's parity test).

Decision rule (Q12). margin m = log P(ANSWER) - max(log P(RETRIEVE_MORE), log P(ABSTAIN)). Each seed ships a sealed
threshold tau (MANIFEST.json, chosen on a tune split, never on the Final split). The gated action is ANSWER iff m >= tau;
otherwise it is whichever of RETRIEVE_MORE / ABSTAIN has the larger probability (the same non-ANSWER choice the raw
argmax makes when ANSWER is not the argmax). The raw argmax is returned too. Research preview: the gate is not a safety control.

v0.1 supports batch = 1 only: decide_inputs() runs one unpadded sequence per forward pass. Shape-dependent
numerical differences were observed (padded batches moved probabilities by up to .066); the root cause has not been
identified. Do not batch if you want the card's numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from dataclasses import dataclass

ACTIONS = ("ANSWER", "RETRIEVE_MORE", "ABSTAIN")
LABELS = {"ANSWER": " Answer", "RETRIEVE_MORE": " Retrieve", "ABSTAIN": " Stop"}
POLICY = ("Policy: answer only if the retrieved evidence above contains a complete support chain for the "
          "answer. Do not use prior knowledge when judging whether the evidence is sufficient. "
          "If the evidence is insufficient and retrieval is available, retrieve more. "
          "If the evidence is insufficient and retrieval is not available, stop without answering.")
PREFILL = "Final action:"
READOUT, LAYERS, MODULES, RANK, ALPHA = 48, (44, 45, 46, 47), ("gate_proj", "up_proj", "down_proj"), 8, 16
MAXLEN = 2048
BACKBONE = {"num_hidden_layers": 64, "hidden_size": 5120, "intermediate_size": 17408}
# Runtime in which the card's numbers were reproduced bit-for-bit. Other runtimes run, but probabilities differ.
VERIFIED_RUNTIME = {"torch": "2.9.0", "cuda": "13.0", "flash-linear-attention": "0.5.2", "transformers": "5.14.1"}


def runtime_mismatch() -> dict:
    """Return {component: (found, verified)} for every component that differs from VERIFIED_RUNTIME."""
    import importlib.metadata as md
    import torch
    found = {"torch": torch.__version__.split("+")[0], "cuda": torch.version.cuda}
    for pkg in ("flash-linear-attention", "transformers"):
        try:
            found[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            found[pkg] = None
    try:
        md.version("causal-conv1d"); found["causal-conv1d"] = "installed"
    except md.PackageNotFoundError:
        found["causal-conv1d"] = None
    bad = {k: (found[k], v) for k, v in VERIFIED_RUNTIME.items() if found[k] != v}
    if found["causal-conv1d"]:
        bad["causal-conv1d"] = ("installed", "absent")
    return bad


def render_input(question: str, passages: list[dict], retrieval_available: bool) -> str:
    """passages: [{"title": str, "text": str}, ...] in retrieval order."""
    ps = "\n\n".join(f"[{i + 1}] {p['title']}\n{p['text']}" if p.get("title") else f"[{i + 1}] {p['text']}"
                     for i, p in enumerate(passages))
    return f"Question: {question}\n\nRetrieved evidence:\n{ps}\n\nRetrieval available: {'YES' if retrieval_available else 'NO'}"


def prompt_text(item_input: str) -> str:
    lab = (f"Actions:{LABELS['ANSWER']} = answer now;{LABELS['RETRIEVE_MORE']} = retrieve more evidence;"
           f"{LABELS['ABSTAIN']} = stop without answering.")
    return (f"{item_input}\n\n{POLICY}\n{lab}\n"
            f"Reply with the action word only, on one line of the form 'Final action: <action word>'.")


def build_ids(tok, item_input: str) -> list[int]:
    msgs = [{"role": "user", "content": prompt_text(item_input)}]
    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(t + PREFILL, add_special_tokens=False)["input_ids"]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def margin(probs: dict) -> float:
    """log P(ANSWER) - max(log P(RETRIEVE_MORE), log P(ABSTAIN)); probs keyed by ACTIONS."""
    lg = lambda p: math.log(max(float(p), 1e-38))
    return lg(probs["ANSWER"]) - max(lg(probs["RETRIEVE_MORE"]), lg(probs["ABSTAIN"]))


def non_answer(probs: dict) -> str:
    """Larger of RETRIEVE_MORE / ABSTAIN (ties go to RETRIEVE_MORE, as the first max in ACTIONS order)."""
    return "RETRIEVE_MORE" if probs["RETRIEVE_MORE"] >= probs["ABSTAIN"] else "ABSTAIN"


def gate(probs: dict, tau: float) -> tuple[str, float]:
    """Return (gated_action, margin): ANSWER iff margin >= tau, else the larger non-ANSWER action."""
    m = margin(probs)
    return ("ANSWER" if m >= tau else non_answer(probs)), m


def load_tau(weights: str, manifest: str | None = None) -> float:
    """Sealed tau of the seed whose file name matches `weights`, read from MANIFEST.json (next to this file by default)."""
    manifest = manifest or os.path.join(os.path.dirname(os.path.abspath(__file__)), "MANIFEST.json")
    with open(manifest) as f:
        seeds = json.load(f)["seeds"]
    name = os.path.basename(weights)
    for v in seeds.values():
        if v["file"] == name:
            return float(v["tau"])
    raise ValueError(f"{name} is not listed in {manifest}; pass tau= explicitly")


@dataclass
class Decision:
    action: str            # gated action (ANSWER only if margin >= tau)
    probs: dict
    sufficiency_logit: float
    raw_action: str = ""   # plain argmax of probs
    margin: float = float("nan")
    tau: float = float("nan")


class ECPQ12:
    def __init__(self, model, tok, head, suf, meta, tau=float("nan")):
        self.model, self.tok, self.head, self.suf, self.meta, self.tau = model, tok, head, suf, meta, tau

    @classmethod
    def from_pretrained(cls, backbone_dir: str, weights: str, expected_sha256: str | None = None,
                        device: str = "cuda", tau: float | None = None, manifest: str | None = None):
        import torch
        tau = load_tau(weights, manifest) if tau is None else float(tau)
        from peft import LoraConfig, get_peft_model
        from safetensors import safe_open
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        if expected_sha256 is not None and sha256_file(weights) != expected_sha256:
            raise ValueError("weights sha256 mismatch")
        bad = runtime_mismatch()
        if bad:
            warnings.warn(f"runtime differs from the verified one {bad}; the model runs, but probabilities and decisions "
                          "are not guaranteed to match the model card (see requirements.txt)", RuntimeWarning, stacklevel=2)
        cfg = AutoConfig.from_pretrained(backbone_dir, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        bad = {k: getattr(tc, k, None) for k, v in BACKBONE.items() if getattr(tc, k, None) != v}
        if bad:
            raise ValueError(f"incompatible backbone (expected Qwen3.8-27B geometry): {bad}")
        tok = AutoTokenizer.from_pretrained(backbone_dir, trust_remote_code=True)
        for a, s in LABELS.items():
            if len(tok.encode(s, add_special_tokens=False)) != 1:
                raise ValueError(f"tokenizer: label {s!r} is not one token")
        with safe_open(weights, "pt") as f:
            meta = dict(f.metadata() or {})
            sd = {k: f.get_tensor(k) for k in f.keys()}
        model = AutoModelForCausalLM.from_pretrained(backbone_dir, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                                                 target_modules=list(MODULES), layers_to_transform=list(LAYERS),
                                                 task_type="CAUSAL_LM"))
        named = {n: p for n, p in model.named_parameters() if "lora_" in n}
        ad = {k[len("adapter."):]: v for k, v in sd.items() if k.startswith("adapter.")}
        if set(named) != set(ad):
            raise ValueError(f"adapter keys differ: {len(set(named) ^ set(ad))} mismatched")
        with torch.no_grad():
            for n, p in named.items():
                p.copy_(ad[n].to(p.device, p.dtype))
        for p in model.parameters():
            p.requires_grad = False
        inner = model.base_model.model
        inner = inner.model if hasattr(inner, "model") else inner
        stack = inner.language_model if hasattr(inner, "language_model") else inner
        stack.layers = torch.nn.ModuleList(list(stack.layers)[:READOUT])
        H = tc.hidden_size
        head = torch.nn.Linear(H, 3).to(device).float(); suf = torch.nn.Linear(H, 1).to(device).float()
        head.load_state_dict({"weight": sd["head.weight"], "bias": sd["head.bias"]})
        suf.load_state_dict({"weight": sd["sufficiency.weight"], "bias": sd["sufficiency.bias"]})
        model.eval(); head.eval(); suf.eval()
        return cls(model, tok, head, suf, meta, tau)

    def decide_inputs(self, item_inputs: list[str]) -> list[Decision]:
        import torch
        out = []
        dev = next(self.head.parameters()).device
        with torch.no_grad():
            for inp in item_inputs:
                ids = build_ids(self.tok, inp)
                if len(ids) > MAXLEN:
                    raise ValueError(f"input is {len(ids)} tokens; the adapter was trained up to {MAXLEN}")
                t = torch.tensor([ids], device=dev)
                hs = self.model(input_ids=t, attention_mask=torch.ones_like(t), output_hidden_states=True,
                                use_cache=False).hidden_states
                if len(hs) != READOUT + 1:
                    raise RuntimeError("stack truncation failed")
                r = hs[-1][0, -1].float()
                p = torch.softmax(self.head(r), -1).cpu().tolist()
                pr = dict(zip(ACTIONS, p))
                act, m = gate(pr, self.tau)
                out.append(Decision(act, pr, float(self.suf(r).item()),
                                    ACTIONS[max(range(3), key=lambda i: p[i])], m, self.tau))
        return out

    def decide_gated(self, question: str, passages: list[dict], retrieval_available: bool) -> Decision:
        return self.decide_inputs([render_input(question, passages, retrieval_available)])[0]

    decide = decide_gated  # alias; both return the gated action plus raw_action, margin and tau


if __name__ == "__main__":
    print(json.dumps({"actions": ACTIONS, "layers": LAYERS, "readout": READOUT}))
