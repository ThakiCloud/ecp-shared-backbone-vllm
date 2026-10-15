# ecp-shared-backbone-vllm

Serve a 27B generative model and an evidence gate from **one GPU and one model load**. A vLLM plugin adds the
[ECP-Q12](https://huggingface.co/ThakiCloud/ECP-Q12-27B-Research-Preview) evidence-gate branch (a LoRA on decoders
44 to 47 plus a 3-way action head) to an unmodified [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
engine. Layers 0 to 43 are shared. Two APIs come out of the same process tree:

| API | Port | What it returns |
|---|---|---|
| Generation | `:8000` | vLLM OpenAI endpoints (`/v1/chat/completions`, `/v1/completions`), original model logits |
| Evidence gate | `:8100` | `POST /v1/gate` → `ANSWER` / `RETRIEVE_MORE` / `ABSTAIN`, margin, `tau`, probabilities |

Weights, thresholds and the manifest live on the Hub:
[ThakiCloud/ECP-SB-27B-Research](https://huggingface.co/ThakiCloud/ECP-SB-27B-Research) (revision `v0.1.0`).

> **Research Deployment, not production certified.** This release passed a single-GPU functional deployment check.
> The gate model behind it (ECP-Q12) has a pooled unsupported-answer rate (UAR) of **6.5%** on its independent
> Final read, above the **5%** target. Do not use the gate as a safety control.

## Quick start

Verified runtime: **vLLM 0.28.0** (`vllm/vllm-openai:v0.28.0`) on **one NVIDIA H200 (141 GB)**. Other vLLM versions
are not supported by v0.1.0 (see [Versions](#versions)).

```bash
git clone https://github.com/ThakiCloud/ecp-shared-backbone-vllm && cd ecp-shared-backbone-vllm
docker build -f docker/Dockerfile -t ecp-sb:0.1.0 .
docker run --gpus '"device=0"' -p 8000:8000 -p 8100:8100 \
  -v /path/Qwen3.8-27B:/models/base -v /path/ECP-SB-27B-Research:/models/ecp ecp-sb:0.1.0
```

Without Docker, inside an environment that has `vllm==0.28.0`:

```bash
pip install --no-deps .                      # registers the vLLM plugin (entry point vllm.general_plugins)
./serve_sb.sh                                # downloads both repos if BASE / ECP_DIR are unset
BASE=/path/Qwen3.8-27B ECP_DIR=/path/ECP-SB-27B-Research SEED=s1 ./serve_sb.sh
```

`serve_sb.sh` builds two config directories that symlink the base weights (nothing is rewritten), starts one
`vllm serve` with the exact flags of the deployment check, waits for `/health`, then starts the gate front on CPU.

## Using the two APIs

```bash
# generation (original model)
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "ecp-sb", "messages": [{"role": "user", "content": "Explain RAG in two sentences."}],
  "max_tokens": 128, "chat_template_kwargs": {"enable_thinking": false}}'

# evidence gate
curl -s localhost:8100/v1/gate -H 'Content-Type: application/json' -d '{
  "question": "Who wrote the novel?",
  "passages": [{"title": "Note", "text": "The novel was written by A. Writer."}],
  "retrieval_available": true}'
```

Gate request forms: `{"question", "passages", "retrieval_available"}`, `{"input": "<rendered input>"}` or
`{"inputs": [...]}`. Inputs longer than 2,048 prompt tokens are rejected with HTTP 413 (the adapter was trained up to
that length). Python examples using only the standard library: [`examples/`](examples/) (`generate.py`,
`evidence_gate.py`, and `gated_generate.py`, which gates first and generates only on `ANSWER`).

Decision rule: `m = log P(ANSWER) - max(log P(RETRIEVE_MORE), log P(ABSTAIN))`; the action is `ANSWER` iff `m >= tau`,
otherwise the more likely of `RETRIEVE_MORE` / `ABSTAIN`. Each of the four seeds ships its own sealed `tau` in
`MANIFEST.json`; `SEED=s1` (`tau = -0.236`) is the configuration that was checked.

## How it works

A gate request is a prompt that ends with the ECP-Q12 prompt tail (a fixed token signature written into the model
config by `sb_build.py`). For the token that completes the signature, the engine re-runs decoders 44 to 47 from the
layer-43 output with LoRA-merged MLPs, applies the final norm and the fp32 action head, and returns `-inf` everywhere
except the three action-label tokens, which carry the head's logits. The gate front sends `max_tokens=1` and reads the
label log-probabilities. Every other request keeps the original logits. The branch has its own attention/GDN state, so
nothing it computes is written into the generation path's state. The branch's 4 layers are loaded as a copy; in v0.1.0
it runs for every scheduled token, gate or not.

## What was verified (and what was not)

The check ran once, on the frozen criteria, on 1× H200-NVL with vLLM 0.28.0 and the engine flags in `serve_sb.sh`,
seed `s1`. It was compared against unmodified vLLM serving the same base model with identical flags over the same
HTTP path.

| Check | Result |
|---|---|
| One GPU, base weights loaded once, no second GPU process | PASS |
| Generation API and evidence-gate API both answer | PASS |
| Generation-only concurrent traffic vs stock vLLM (64 prompts) | PASS: 2/64 token-divergent vs 1/64 stock-vs-stock; p99 logprob gap .18749142 vs bound .18749428 (passed by **0.00000286**) |
| Alternating generation and gate requests vs stock vLLM (16 prompts) | PASS: 0/16 divergent, p99 0 |
| Gate decisions under mixed traffic (256 items) | PASS: 0/256 decisions changed vs the offline reference, 0 errors |
| Idle 600 s, then traffic | PASS: only the engine holds the GPU, memory flat |
| Restart with the same configuration | PASS: base loaded once again, both APIs recover |
| Generation under mixed generation + gate traffic is identical to stock | **Not shown.** 6/64 prompts produced different tokens than stock vLLM (p99 gap .25), 4/64 vs the same engine without gate traffic. The cause is not attributed: differences of the same size also appeared between repeated stock runs in smaller checks. |
| Production safety | **Not certified.** ECP-Q12 pooled UAR 6.5% vs 5% target. |
| Throughput, sustained load, cost per GPU | Not measured. |

The p99 criterion passed by a margin of the same order as the earlier failure, so read this as a borderline pass, not a
comfortable one. Two earlier runs under previous criteria versions (v2, v3) failed on the comparison baseline (an offline
batch rather than the same HTTP path); those failures stay on record and were not relabelled. The check above is a
reproduction on one fixed workload (64 generation prompts, 16 alternating prompts, 256 gate items), not evidence about
other request distributions or loads.

GPU memory: 119,632 MiB held by the engine after load, 123,537 MiB on the device under traffic
(`--gpu-memory-utilization 0.85`, H200 141 GB).

## Versions

| Release | Engine | Status |
|---|---|---|
| v0.1.0 (this) | vLLM 0.28.0 | Verified as above |
| next | vLLM 0.31 | Port and compatibility check planned; **not verified**, do not assume it works |

The plugin uses vLLM's model registry and internal `qwen3_5` modules, which change between releases. A fork of the
engine will only be considered if a plugin cannot do the job.

## Repository layout

| Path | Role |
|---|---|
| `ecp_sb/` | vLLM plugin: registers `ECPSharedQwen3_5ForCausalLM` (shared engine) and `ECPStockQwen3_5ForCausalLM` (reference) |
| `sb_build.py` | Builds the `sb` / `stock` config directories from a base checkpoint and one ECP-Q12 seed file |
| `sb_gate_api.py` | Gate front (FastAPI, CPU only) over the engine's completions endpoint |
| `ecp_q12.py` | Prompt contract of ECP-Q12 (byte-for-byte the training prompt); the gate front builds token ids with it |
| `serve_sb.sh`, `docker/` | One-command serving with the verified flags |
| `examples/`, `tests/` | Client examples (stdlib only), CPU unit tests for the gate front |

## License

Apache-2.0. The base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is Apache-2.0 and is not
redistributed here.
