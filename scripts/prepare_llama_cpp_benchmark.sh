#!/usr/bin/env bash
# prepare_llama_cpp_benchmark.sh — Setup for llama.cpp + CooperBench benchmarking.
#
# Usage:
#   ./scripts/prepare_llama_cpp_benchmark.sh [--pre-pull] [--subset NAME]
#
# Options:
#   --pre-pull        Pre-pull all Docker images for the subset (avoids first-run latency)
#   --subset NAME     Subset to pre-pull images for (default: flash_25)
#
# What it does:
#   1. Verifies the llama.cpp server is running on port 8050
#   2. Installs Python dependencies via uv (including wandb)
#   3. Downloads the CooperBench dataset from HuggingFace
#   4. Optionally pre-pulls Docker images (+ redis:alpine)
#   5. Runs a quick connectivity smoke test
#
# Environment variables:
#   LLAMA_CPP_BASE_URL  — llama-server URL (default: http://localhost:8050/v1)
#   LLAMA_CPP_API_KEY   — API key for llama-server (default: local-llama-cpp)
#   LLAMA_CPP_MODEL     — Model name for litellm (default: openai/Qwen3.6-27B-Q4_K_M.gguf)
#   HF_TOKEN            — HuggingFace token for dataset download
#   WANDB_API_KEY       — Weights & Biases API key (optional, for logging)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://localhost:8050/v1}"
LLAMA_CPP_API_KEY="${LLAMA_CPP_API_KEY:-local-llama-cpp}"
LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-openai/Qwen3.6-27B-Q4_K_M.gguf}"
PRE_PULL=false
SUBSET="flash_25"

# ── Parse args ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --pre-pull) PRE_PULL=true; shift ;;
        --subset) SUBSET="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[prepare]${NC} $*"; }
warn() { echo -e "${YELLOW}[prepare]${NC} $*"; }
err()  { echo -e "${RED}[prepare]${NC} $*"; }

# ── 1. Verify llama-server is running ─────────────────────────────────

log "checking llama-server at $LLAMA_CPP_BASE_URL ..."
if curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models" >/dev/null 2>&1; then
    MODEL_INFO=$(curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models")
    MODEL_NAME=$(echo "$MODEL_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "unknown")
    CTX=$(echo "$MODEL_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta']['n_ctx'])" 2>/dev/null || echo "?")
    SIZE=$(echo "$MODEL_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta']['size'])" 2>/dev/null || echo "?")
    SIZE_GB=$(python3 -c "print(f'{int($SIZE) / 1e9:.1f} GB')" 2>/dev/null || echo "?")
    log "llama-server ready — model: $MODEL_NAME ($SIZE_GB, ctx=$CTX)"
else
    err "llama-server not reachable at $LLAMA_CPP_BASE_URL"
    err "Start it with: ./scripts/serve_llama_cpp.sh --bg"
    exit 1
fi

# ── 2. Install Python dependencies ───────────────────────────────────

log "syncing Python dependencies (uv sync --extra wandb)..."
cd "$PROJECT_DIR"
uv sync --extra wandb
log "dependencies up to date"

# Verify wandb is available (warn but continue if missing)
if ! uv run python -c "import wandb" 2>/dev/null; then
    warn "wandb not available — install with: uv sync --extra wandb"
    warn "Benchmark will still run but won't log to Weights & Biases."
fi

# ── 3. Download dataset ──────────────────────────────────────────────

if [ -d "$PROJECT_DIR/dataset" ] && [ "$(ls -A "$PROJECT_DIR/dataset" 2>/dev/null)" ]; then
    log "dataset already present at dataset/"
else
    log "downloading CooperBench dataset..."
    uv run cooperbench prepare
    log "dataset download complete"
fi

# ── 4. Pre-pull Docker images ────────────────────────────────────────

if $PRE_PULL; then
    SUBSET_FILE="$PROJECT_DIR/dataset/subsets/${SUBSET}.json"
    if [ ! -f "$SUBSET_FILE" ]; then
        warn "subset file not found: $SUBSET_FILE — skipping pre-pull"
    else
        log "pre-pulling Docker images for subset '$SUBSET'..."

        # Extract unique (repo, task_id) pairs and generate Docker Hub image names.
        # Image naming: akhatua/cooperbench-{repo_clean}:task{id}
        #   repo_clean = repo_name.replace("_task", "").replace("_", "-")
        IMAGES=$(uv run python3 -c "
import json, sys
with open('$SUBSET_FILE') as f:
    data = json.load(f)
seen = set()
images = []
for entry in data.get('tasks', []):
    repo = entry['repo']
    tid = entry['task_id']
    key = (repo, tid)
    if key in seen:
        continue
    seen.add(key)
    repo_clean = repo.replace('_task', '').replace('_', '-')
    images.append(f'akhatua/cooperbench-{repo_clean}:task{tid}')
print('\n'.join(images))
")

        # Add redis image (needed for coop mode messaging)
        IMAGES="$IMAGES"$'\n'"redis:alpine"

        TOTAL=$(echo "$IMAGES" | wc -l)
        COUNT=0
        FAILED=0

        while IFS= read -r img; do
            [ -z "$img" ] && continue
            COUNT=$((COUNT + 1))
            printf "  [%2d/%2d] pulling %s ... " "$COUNT" "$TOTAL" "$img"
            if docker pull "$img" >/dev/null 2>&1; then
                echo "ok"
            else
                echo "FAILED"
                FAILED=$((FAILED + 1))
            fi
        done <<< "$IMAGES"

        if [ "$FAILED" -gt 0 ]; then
            warn "$FAILED/$TOTAL images failed to pull (network issue?) — benchmark will auto-pull on first use"
        else
            log "all $TOTAL Docker images pulled successfully"
        fi
    fi
fi

# ── 5. Smoke test connectivity ───────────────────────────────────────

log "testing model connectivity..."
if uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import litellm, os
litellm.suppress_debug_info = True
os.environ['LITELLM_LOG'] = 'ERROR'
resp = litellm.completion(
    model='$LLAMA_CPP_MODEL',
    messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}],
    api_base='$LLAMA_CPP_BASE_URL',
    api_key='$LLAMA_CPP_API_KEY',
)
print('Response:', resp.choices[0].message.content)
print('OK')
" 2>&1; then
    log "connectivity test PASSED"
else
    warn "connectivity test FAILED — check server logs"
fi

echo ""
log "Setup complete!"
log "  server:  $LLAMA_CPP_BASE_URL"
log "  model:   $LLAMA_CPP_MODEL"
log "  dataset: $PROJECT_DIR/dataset/"
if $PRE_PULL; then
    log "  images:  pre-pulled ($COUNT Docker images)"
fi
echo ""
log "Next: run the benchmark with ./scripts/launch_llama_cpp_benchmark.sh"
log "  or smoke test with ./scripts/smoke_test_llama_cpp.sh"
log ""
log "For Weights & Biases logging:"
log "  1. Set WANDB_API_KEY in your environment"
log "  2. Run: ./scripts/launch_llama_cpp_benchmark.sh --wandb-project my-project --wandb-entity my-team"
