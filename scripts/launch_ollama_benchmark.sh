#!/usr/bin/env bash
# launch_ollama_benchmark.sh — Run CooperBench solo + coop benchmarks via Ollama.
#
# Usage:
#   ./scripts/launch_ollama_benchmark.sh [OPTIONS]
#
# Options:
#   --solo-only       Run only the solo benchmark
#   --coop-only       Run only the coop benchmark
#   --subset NAME     Subset to use (default: flash_25)
#   --concurrency N   Number of parallel tasks (default: 1)
#                     Override with COOPERBENCH_CONCURRENCY env var
#   --force           Re-run tasks even if results exist
#
# Environment variables:
#   OLLAMA_BASE_URL          — Ollama server URL (default: http://localhost:11434)
#   OLLAMA_MODEL             — Model name (default: qwen3.5:27b)
#   COOPERBENCH_CONCURRENCY  — Parallel task count (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5:27b}"
LITELLM_MODEL="ollama_chat/${OLLAMA_MODEL}"
CONCURRENCY="${COOPERBENCH_CONCURRENCY:-1}"
SUBSET="flash_25"
RUN_SOLO=true
RUN_COOP=true
FORCE=""

# ── Parse args ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --solo-only) RUN_COOP=false; shift ;;
        --coop-only) RUN_SOLO=false; shift ;;
        --subset) SUBSET="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[bench]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

# ── Pre-flight checks ─────────────────────────────────────────────────

cd "$PROJECT_DIR"

# Ensure ollama is running
if ! curl -s "$OLLAMA_BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "ollama not running at $OLLAMA_BASE_URL — starting..."
    OLLAMA_HOST="${OLLAMA_BASE_URL#http://}" OLLAMA_HOST="${OLLAMA_HOST#https://}" ollama serve &
    sleep 3
fi

# Verify model is available
if ! ollama list 2>/dev/null | grep -q "^${OLLAMA_MODEL}\b"; then
    echo "ERROR: model '$OLLAMA_MODEL' not found. Run ./scripts/prepare_ollama_benchmark.sh first."
    exit 1
fi

# Ensure dataset exists
if [ ! -d "$PROJECT_DIR/dataset" ] || [ -z "$(ls -A "$PROJECT_DIR/dataset" 2>/dev/null)" ]; then
    echo "ERROR: dataset not found. Run ./scripts/prepare_ollama_benchmark.sh first."
    exit 1
fi

# ── Run solo benchmark ────────────────────────────────────────────────

if $RUN_SOLO; then
    RUN_NAME="solo-ol-qwen3.5-27b-${SUBSET}"
    header "Solo benchmark: $RUN_NAME"
    log "model:    $LITELLM_MODEL"
    log "subset:   $SUBSET"
    log "concurrency: $CONCURRENCY"
    echo ""

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -s "$SUBSET" \
        -m "$LITELLM_MODEL" \
        -a ollama \
        -c "$CONCURRENCY" \
        --backend docker \
        $FORCE

    log "Solo benchmark complete → logs/$RUN_NAME/solo/"
fi

# ── Run coop benchmark ────────────────────────────────────────────────

if $RUN_COOP; then
    RUN_NAME="coop-ol-qwen3.5-27b-${SUBSET}"
    header "Coop benchmark: $RUN_NAME"
    log "model:    $LITELLM_MODEL"
    log "subset:   $SUBSET"
    log "concurrency: $CONCURRENCY"
    echo ""

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -s "$SUBSET" \
        -m "$LITELLM_MODEL" \
        -a ollama \
        -c "$CONCURRENCY" \
        --backend docker \
        $FORCE

    log "Coop benchmark complete → logs/$RUN_NAME/coop/"
fi

# ── Summary ───────────────────────────────────────────────────────────

header "Benchmark complete"

if $RUN_SOLO; then
    SOLO_LOG="logs/solo-ol-qwen3.5-27b-${SUBSET}/solo"
    if [ -f "$SOLO_LOG/summary.json" ]; then
        echo "[solo] $(python3 -c "
import json
with open('$SOLO_LOG/summary.json') as f:
    s = json.load(f)
print(f\"{s.get('completed',0)} completed, {s.get('failed',0)} failed, {s.get('skipped',0)} skipped\")
e = s.get('eval', {})
if e:
    print(f'  eval: {e.get(\"passed\",0)} passed, {e.get(\"failed\",0)} failed ({e.get(\"pass_rate\",0)*100:.0f}%)')
")"
    fi
fi

if $RUN_COOP; then
    COOP_LOG="logs/coop-ol-qwen3.5-27b-${SUBSET}/coop"
    if [ -f "$COOP_LOG/summary.json" ]; then
        echo "[coop] $(python3 -c "
import json
with open('$COOP_LOG/summary.json') as f:
    s = json.load(f)
print(f\"{s.get('completed',0)} completed, {s.get('failed',0)} failed, {s.get('skipped',0)} skipped\")
e = s.get('eval', {})
if e:
    print(f'  eval: {e.get(\"passed\",0)} passed, {e.get(\"failed\",0)} failed ({e.get(\"pass_rate\",0)*100:.0f}%)')
")"
    fi
fi

echo ""
