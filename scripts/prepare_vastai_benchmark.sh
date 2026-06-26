#!/usr/bin/env bash
# prepare_vastai_benchmark.sh — One-shot pre-flight for Vast.ai.
#
# Use this after `setup_vastai.sh` succeeds and before the first benchmark run:
#   - verifies vLLM is reachable and reports model ctx / size
#   - installs Python deps via `uv sync --extra wandb`
#   - downloads the CooperBench dataset if not present
#   - runs a 1-token connectivity test through the served model
#
# Environment variables:
#   VLLM_BASE_URL        — vllm endpoint (default: http://localhost:8000/v1)
#   VLLM_MODEL           — model id as served (default: qwen3.6-27b-awq-int4)
#   VLLM_API_KEY         — auth token (default: dummy, vllm doesn't enforce)
#   COOPERBENCH_DIR      — path to cooperbench source (default: /opt/cooperbench)
#   HF_TOKEN             — HuggingFace token for dataset download (optional)

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[prepare]${NC} $*"; }
warn() { echo -e "${YELLOW}[prepare]${NC} $*"; }
err()  { echo -e "${RED}[prepare]${NC} $*"; }

VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
VLLM_MODEL="${VLLM_MODEL:-qwen3.6-27b-awq-int4}"
COOPERBENCH_DIR="${COOPERBENCH_DIR:-/opt/cooperbench}"

cd "$COOPERBENCH_DIR"

# ── 1. Verify vLLM is reachable ──────────────────────────────────────
log "checking vLLM at $VLLM_BASE_URL ..."
if ! curl -s "$VLLM_BASE_URL/models" >/dev/null 2>&1; then
    err "vLLM not reachable at $VLLM_BASE_URL"
    err "Run scripts/setup_vastai.sh first (it starts the cb-vllm container)."
    exit 1
fi

SERVER_INFO=$(curl -s "$VLLM_BASE_URL/models")
SERVER_MODEL=$(echo "$SERVER_INFO" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "?")
SERVER_CTX=$(echo "$SERVER_INFO"   | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['meta'].get('n_ctx', '?'))" 2>/dev/null || echo "?")
log "vllm ready — model: $SERVER_MODEL, ctx=$SERVER_CTX"

# ── 2. Install Python dependencies ───────────────────────────────────
log "syncing Python dependencies (uv sync --extra wandb)..."
uv sync --extra wandb
log "dependencies up to date"

# ── 3. Download dataset ──────────────────────────────────────────────
if [ -d "$COOPERBENCH_DIR/dataset" ] && [ "$(ls -A "$COOPERBENCH_DIR/dataset" 2>/dev/null)" ]; then
    log "dataset already present at $COOPERBENCH_DIR/dataset"
else
    log "downloading CooperBench dataset..."
    uv run cooperbench prepare
    log "dataset download complete"
fi

# ── 4. Connectivity test (Anthropic /v1/messages) ───────────────────
log "testing model connectivity via /v1/messages ..."
if uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import os, json, urllib.request
req = urllib.request.Request(
    '${VLLM_BASE_URL%/v1}/v1/messages',
    data=json.dumps({
        'model': '${VLLM_MODEL}',
        'max_tokens': 32,
        'messages': [{'role': 'user', 'content': 'Reply with the single word PONG.'}],
    }).encode(),
    headers={
        'content-type': 'application/json',
        'x-api-key': '${VLLM_API_KEY}',
        'anthropic-version': '2023-06-01',
    },
    method='POST',
)
with urllib.request.urlopen(req, timeout=60) as r:
    body = json.loads(r.read())
    txt = ''.join(b.get('text','') for b in body.get('content', []) if b.get('type')=='text')
    print('response:', txt.strip()[:200])
    print('OK')
" 2>&1; then
    log "Anthropic /v1/messages connectivity: PASSED"
else
    err "Anthropic /v1/messages connectivity: FAILED"
    err "Check that vLLM is serving with --enable-auto-tool-choice + a --tool-call-parser"
    err "and that --served-model-name matches the model id we sent."
    exit 1
fi

echo ""
log "Setup complete!"
log "  vllm:     $VLLM_BASE_URL"
log "  model:    $VLLM_MODEL"
log "  dataset:  $COOPERBENCH_DIR/dataset/"
echo ""
log "Next: run the benchmark with ./scripts/launch_vastai_benchmark.sh"
log "  or smoke test with ./scripts/smoke_test_vastai.sh"
log ""
log "For Weights & Biases logging:"
log "  1. Set WANDB_API_KEY in your environment"
log "  2. Run: ./scripts/launch_vastai_benchmark.sh --wandb-project my-project --wandb-entity my-team"
