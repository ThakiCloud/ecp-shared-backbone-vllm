"""Build the SB1 model dirs from an original Qwen3.5 checkpoint dir (weights are symlinked, never rewritten):
  <out>/sb     architectures=[ECPSharedQwen3_5ForCausalLM] + ecp_* fields (gate layers, Q12 file, signature, labels)
  <out>/stock  architectures=[Qwen3_5ForCausalLM], same weights -> the GEN_ONLY reference engine
Signature = the token tail every Q12 gate prompt ends with (longest common suffix of build_ids over distinct inputs).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "chat_template.jinja", "generation_config.json", "preprocessor_config.json")


def text_config(bdir):
    cfg = json.load(open(os.path.join(bdir, "config.json")))
    tc = dict(cfg.get("text_config") or cfg)
    tc.setdefault("tie_word_embeddings", cfg.get("tie_word_embeddings", False))
    if "mtp_num_hidden_layers" in tc:
        tc["mtp_num_hidden_layers"] = 0
    return tc


def signature(tok, build_ids, k_max=48, k_min=8):
    probes = ["Question: a?\n\nRetrieved evidence:\n[1] T\nx.\n\nRetrieval available: YES",
              "Question: Which bridge is older, and by how many years?\n\nRetrieved evidence:\n[1] B\nIt opened in 1901.\n\nRetrieval available: NO",
              "Question: 누가 썼나?\n\nRetrieved evidence:\n[1] 책\n저자는 김이다.\n\nRetrieval available: YES"]
    seqs = [build_ids(tok, p) for p in probes]
    k = 0
    while k < min(map(len, seqs)) and len({s[-1 - k] for s in seqs}) == 1:
        k += 1
    if k < k_min:
        raise ValueError(f"common prompt tail is only {k} tokens")
    return seqs[0][-min(k, k_max):]


def build(bdir, q12_file, out, gate_layers, alpha=16.0, tau=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ecp_q12 as L
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(bdir, trust_remote_code=True)
    labels = []
    for a in L.ACTIONS:
        ids = tok.encode(L.LABELS[a], add_special_tokens=False)
        assert len(ids) == 1, (a, ids)
        labels.append(ids[0])
    sig = signature(tok, L.build_ids)
    tc = text_config(bdir)
    for name, arch, extra in (("stock", "ECPStockQwen3_5ForCausalLM", {}),
                              ("sb", "ECPSharedQwen3_5ForCausalLM",
                               {"ecp_gate_layers": list(gate_layers), "ecp_q12_file": os.path.abspath(q12_file),
                                "ecp_gate_signature": sig, "ecp_gate_label_ids": labels, "ecp_lora_alpha": alpha,
                                "ecp_tau": tau})):
        d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
        c = {**tc, "architectures": [arch], **extra}
        json.dump(c, open(os.path.join(d, "config.json"), "w"), indent=1)
        for fn in os.listdir(bdir):
            if fn.endswith(".safetensors") or fn == "model.safetensors.index.json" or fn in TOKENIZER_FILES:
                dst = os.path.join(d, fn)
                if os.path.lexists(dst):
                    os.remove(dst)
                os.symlink(os.path.abspath(os.path.join(bdir, fn)), dst)
    meta = {"label_ids": labels, "signature_len": len(sig), "gate_layers": list(gate_layers), "tau": tau}
    json.dump(meta, open(os.path.join(out, "sb_meta.json"), "w"), indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone"); ap.add_argument("--q12"); ap.add_argument("--out")
    ap.add_argument("--gate-layers", default="44,45,46,47"); ap.add_argument("--tau", type=float)
    a = ap.parse_args()
    print(json.dumps(build(a.backbone, a.q12, a.out, [int(x) for x in a.gate_layers.split(",")], tau=a.tau)))
