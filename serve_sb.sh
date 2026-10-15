#!/usr/bin/env bash
# One GPU, one vLLM engine, two APIs:
#   generation    : vLLM OpenAI endpoints on :8000 (/v1/chat/completions, /v1/completions)
#   evidence gate : POST /v1/gate on :8100 (no weights, no GPU; it calls the same engine)
#
# Usage:  [BASE=/path/Qwen3.8-27B] [ECP_DIR=/path/ECP-SB-27B-Research] [SEED=s1] [WORK=/tmp/ecp-sb] ./serve_sb.sh
#   BASE    local copy of Qwen/Qwen3.8-27B. Downloaded from the Hub if unset.
#   ECP_DIR local copy of ThakiCloud/ECP-SB-27B-Research (revision v0.1.0). Downloaded if unset.
# The engine flags below are the ones the deployment check ran with. Changing them leaves the verified configuration.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SEED="${SEED:-s1}"; WORK="${WORK:-/tmp/ecp-sb}"; NAME="${SERVED_NAME:-ecp-sb}"
ECP_REPO="${ECP_REPO:-ThakiCloud/ECP-SB-27B-Research}"; ECP_REV="${ECP_REV:-v0.1.0}"
fetch() {  # repo revision dest
  python3 -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3]))" "$@"
}
if [ -z "${BASE:-}" ]; then BASE="$WORK/base"; fetch Qwen/Qwen3.8-27B main "$BASE" >/dev/null; fi
if [ -z "${ECP_DIR:-}" ]; then ECP_DIR="$WORK/ecp"; fetch "$ECP_REPO" "$ECP_REV" "$ECP_DIR" >/dev/null; fi
FILE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['seeds'][sys.argv[2]]['file'])" "$ECP_DIR/MANIFEST.json" "$SEED")"
TAU="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['seeds'][sys.argv[2]]['tau'])" "$ECP_DIR/MANIFEST.json" "$SEED")"
python3 "$HERE/sb_build.py" --backbone "$BASE" --q12 "$ECP_DIR/weights/$FILE" --out "$WORK/sbm" --gate-layers 44,45,46,47 --tau "$TAU"
export ECP_SB_MODE=full ECP_SB_CAPTURE=post VLLM_WORKER_MULTIPROC_METHOD=spawn
echo "SERVE_CONFIG vllm serve $WORK/sbm/sb seed=$SEED tau=$TAU ECP_SB_CAPTURE=post"
vllm serve "$WORK/sbm/sb" --served-model-name "$NAME" --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-batched-tokens 16384 --max-num-seqs 64 --seed 0 \
  --trust-remote-code --host 0.0.0.0 --port 8000 --max-logprobs 20 &
ENGINE_PID=$!
trap 'kill $ENGINE_PID 2>/dev/null' EXIT
python3 - <<'PY'
import time, urllib.request
while True:
    try:
        if urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).status == 200:
            break
    except Exception:
        time.sleep(5)
PY
cd "$HERE"
CUDA_VISIBLE_DEVICES="" MODEL_DIR="$WORK/sbm/sb" ENGINE_URL=http://127.0.0.1:8000 SERVED_NAME="$NAME" \
  python3 -m uvicorn sb_gate_api:app --host 0.0.0.0 --port 8100
