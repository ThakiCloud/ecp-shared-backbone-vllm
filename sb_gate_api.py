"""Gate API over the Shared-Backbone vLLM server (one deployment serves generation and the evidence gate).

    vllm serve <sbm/sb> --served-model-name ecp-sb ...    (engine + generation API, :8000, OpenAI endpoints)
    uvicorn sb_gate_api:app --port 8100                    (this file: no weights, no GPU)

POST /v1/gate {"input": "<rendered item input>"} | {"inputs": [...]} | {"question", "passages", "retrieval_available"}
-> {"action", "raw_action", "margin", "tau", "probs", "n_tokens", "timing_ms"}

The SB engine answers a prompt that ends with the Q12 signature with -inf logits everywhere except the three action
label tokens, which carry the gate head's fp32 logits; so log p(action) is read from the label tokens' logprobs of a
max_tokens=1 completion (return_tokens_as_token_ids). Margin m = log p[A] - max(log p[R], log p[S]); ANSWER iff m >= tau.
env: MODEL_DIR (sbm/sb: config with ecp_gate_label_ids / ecp_tau, tokenizer), ENGINE_URL, SERVED_NAME.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACTIONS = ("ANSWER", "RETRIEVE_MORE", "ABSTAIN")


def gate_from_label_logprobs(lp, tau):
    """lp: log p of (ANSWER, RETRIEVE_MORE, ABSTAIN) label tokens. Pure; unit-tested."""
    m = lp[0] - max(lp[1], lp[2])
    p = [math.exp(v) for v in lp]
    raw = ACTIONS[max(range(3), key=lambda i: lp[i])]
    na = "RETRIEVE_MORE" if lp[1] >= lp[2] else "ABSTAIN"
    return {"action": "ANSWER" if m >= tau else na, "raw_action": raw, "margin": m, "tau": tau,
            "probs": dict(zip(ACTIONS, p))}


def label_logprobs(top, label_ids):
    """top: one completion step's top_logprobs dict keyed 'token_id:N'. Missing label -> error (not a gate prompt)."""
    d = {int(k.split(":", 1)[1]): float(v) for k, v in top.items()}
    missing = [l for l in label_ids if l not in d]
    if missing:
        raise ValueError(f"label tokens {missing} not in top logprobs: the request did not hit the gate path")
    return [d[l] for l in label_ids]


def make_app():
    import aiohttp
    from fastapi import FastAPI, HTTPException
    from transformers import AutoTokenizer
    import ecp_q12 as L

    mdir = os.environ["MODEL_DIR"]; engine = os.environ.get("ENGINE_URL", "http://127.0.0.1:8000")
    name = os.environ.get("SERVED_NAME", "ecp-sb")
    cfg = json.load(open(os.path.join(mdir, "config.json")))
    labels, tau = [int(x) for x in cfg["ecp_gate_label_ids"]], float(cfg["ecp_tau"])
    tok = AutoTokenizer.from_pretrained(mdir, trust_remote_code=True)
    app = FastAPI(title="ECP-SB gate")
    st = {"session": None}

    @app.on_event("startup")
    async def _start():
        st["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600), connector=aiohttp.TCPConnector(limit=0))

    @app.get("/health")
    async def health():
        return {"ok": True, "tau": tau, "label_ids": labels}

    async def one(item_input):
        t0 = time.perf_counter()
        ids = await asyncio.to_thread(L.build_ids, tok, item_input)
        if len(ids) > L.MAXLEN:
            raise HTTPException(413, f"input is {len(ids)} tokens; trained up to {L.MAXLEN}")
        body = {"model": name, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "logprobs": 5,
                "return_tokens_as_token_ids": True}
        t1 = time.perf_counter()
        async with st["session"].post(f"{engine}/v1/completions", json=body) as r:
            txt = await r.text()
        if r.status != 200:
            raise HTTPException(502, f"engine {r.status}: {txt[:300]}")
        t2 = time.perf_counter()
        try:
            lp = label_logprobs(json.loads(txt)["choices"][0]["logprobs"]["top_logprobs"][0], labels)
        except ValueError as e:
            raise HTTPException(502, str(e)) from e
        out = gate_from_label_logprobs(lp, tau)
        out["label_logprobs"] = lp
        out["n_tokens"] = len(ids)
        out["timing_ms"] = {"tokenize": 1e3 * (t1 - t0), "engine": 1e3 * (t2 - t1), "total": 1e3 * (time.perf_counter() - t0)}
        return out

    @app.post("/v1/gate")
    async def gate(req: dict):
        if "inputs" in req:
            return {"results": await asyncio.gather(*(one(x) for x in req["inputs"]))}
        if "input" in req:
            return await one(req["input"])
        if "question" in req:
            return await one(L.render_input(req["question"], req.get("passages", []), bool(req.get("retrieval_available"))))
        raise HTTPException(400, "need input | inputs | question+passages+retrieval_available")

    return app


app = make_app() if os.environ.get("MODEL_DIR") else None
