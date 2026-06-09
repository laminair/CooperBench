#!/usr/bin/env bash
# prepare_ollama_benchmark.sh — One-shot setup for Ollama + CooperBench benchmarking.
#
# Usage:
#   ./scripts/prepare_ollama_benchmark.sh
#
# What it does:
#   1. Installs ollama if not already present
#   2. Starts ollama serve in background (if not running)
#   3. Pulls the qwen3.5:27b model
#   4. Installs Python dependencies via uv
#   5. Downloads the CooperBench dataset from HuggingFace
#   6. Runs a quick connectivity smoke test
#
# Environment variables:
#   OLLAMA_BASE_URL    — Ollama server URL (default: http://localhost:11434)
#   OLLAMA_MODEL       — Model to pull (default: qwen3.5:27b)
#   HF_TOKEN           — HuggingFace token for dataset download (if needed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:27b}"
OLLAMA_HOST="${OLLAMA_HOST:-$(echo "$OLLAMA_BASE_URL" | sed 's|http://||;s|https://||')}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[prepare]${NC} $*"; }
warn() { echo -e "${YELLOW}[prepare]${NC} $*"; }
err()  { echo -e "${RED}[prepare]${NC} $*"; }

# ── 1. Install ollama ────────────────────────────────────────────────

if command -v ollama &>/dev/null; then
    log "ollama found: $(ollama --version 2>&1 || echo 'unknown version')"
else
    log "installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    log "ollama installed"
fi

# ── 2. Start ollama serve ────────────────────────────────────────────

OLLAMA_HOST="$OLLAMA_HOST" ollama serve &
OLLAMA_PID=$!
log "started ollama serve (pid=$OLLAMA_PID)"

# Wait for ollama to be ready
for i in $(seq 1 30); do
    if curl -s "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
        log "ollama server is ready"
        break
    fi
    if [ "$i" -eq 30 ]; then
        err "ollama server failed to start after 30s"
        exit 1
    fi
    sleep 1
done

# ── 3. Pull model ────────────────────────────────────────────────────

log "pulling model: $OLLAMA_MODEL"
if ollama list | grep -q "^$OLLAMA_MODEL"; then
    log "model $OLLAMA_MODEL already present"
else
    ollama pull "$OLLAMA_MODEL"
    log "model $OLLAMA_MODEL pulled"
fi

# ── 4. Install Python dependencies ───────────────────────────────────

log "syncing Python dependencies (uv sync)..."
cd "$PROJECT_DIR"
uv sync
log "dependencies up to date"

# ── 5. Download dataset ──────────────────────────────────────────────

if [ -d "$PROJECT_DIR/dataset" ] && [ "$(ls -A "$PROJECT_DIR/dataset" 2>/dev/null)" ]; then
    log "dataset already present at dataset/"
else
    log "downloading CooperBench dataset..."
    uv run cooperbench prepare
    log "dataset download complete"
fi

# ── 6. Smoke test connectivity ───────────────────────────────────────

log "testing model connectivity..."
if uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import litellm
resp = litellm.completion(
    model='ollama_chat/$OLLAMA_MODEL',
    messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}],
    api_base='$OLLAMA_BASE_URL',
)
print('Response:', resp.choices[0].message.content)
print('OK')
" 2>&1; then
    log "connectivity test PASSED"
else
    warn "connectivity test FAILED (ollama may need more time or VRAM)"
fi

echo ""
log "Setup complete!"
log "  ollama:  $OLLAMA_BASE_URL (model: $OLLAMA_MODEL)"
log "  dataset: $PROJECT_DIR/dataset/"
log ""
log "Next: run the benchmark with ./scripts/launch_ollama_benchmark.sh"
log "  or smoke test with ./scripts/smoke_test_ollama.sh"
