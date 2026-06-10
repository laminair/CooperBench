#!/usr/bin/env bash
# launch_llama_cpp_benchmark.sh — Run CooperBench solo + coop benchmarks via llama.cpp.
#
# Usage:
#   ./scripts/launch_llama_cpp_benchmark.sh [OPTIONS]
#
# Options:
#   --solo-only         Run only the solo benchmark
#   --coop-only         Run only the coop benchmark
#   --subset NAME       Subset to use (default: flash_25)
#   --concurrency N     Parallel tasks (default: auto from GPU count, fallback 1)
#                       Override with COOPERBENCH_CONCURRENCY env var
#   --force             Re-run tasks even if results exist
#   --wandb-project P   Weights & Biases project name
#   --wandb-entity E    Weights & Biases entity (team/user)
#
# Environment variables:
#   LLAMA_CPP_BASE_URL               — llama-server URL (default: http://localhost:8050/v1)
#   LLAMA_CPP_API_KEY                — API key (default: local-llama-cpp)
#   LLAMA_CPP_MODEL                  — Model name (default: openai/Qwen3.6-27B-Q4_K_M.gguf)
#   LLAMA_CPP_COMPACTION_TRIGGER     — Compaction token threshold (auto from ctx size)
#   COOPERBENCH_CONCURRENCY          — Parallel task count (default: auto)
#   COOPERBENCH_VRAM_HEADROOM_MB     — VRAM reserved for memory spikes (default: 15000)
#   WANDB_PROJECT                    — Weights & Biases project (or --wandb-project)
#   WANDB_ENTITY                     — Weights & Biases entity (or --wandb-entity)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────

LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://localhost:8050/v1}"
LLAMA_CPP_API_KEY="${LLAMA_CPP_API_KEY:-local-llama-cpp}"
LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-openai/Qwen3.6-27B-Q4_K_M.gguf}"
SUBSET="flash_25"
RUN_SOLO=true
RUN_COOP=true
FORCE=""
CONCURRENCY="${COOPERBENCH_CONCURRENCY:-}"
VRAM_HEADROOM_MB="${COOPERBENCH_VRAM_HEADROOM_MB:-15000}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# ── Parse args ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --solo-only) RUN_COOP=false; shift ;;
        --coop-only) RUN_SOLO=false; shift ;;
        --subset) SUBSET="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[bench]${NC} $*"; }
warn() { echo -e "${YELLOW}[bench]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

cd "$PROJECT_DIR"

# ── Pre-flight checks ─────────────────────────────────────────────────

# Verify llama-server is running
if ! curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models" >/dev/null 2>&1; then
    echo "ERROR: llama-server not reachable at $LLAMA_CPP_BASE_URL"
    exit 1
fi

# Gather server info
SERVER_INFO=$(curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models")
SERVER_MODEL=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
SERVER_CTX=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta']['n_ctx'])" 2>/dev/null || echo "?")
SERVER_SIZE=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta']['size'])" 2>/dev/null || echo "?")
SERVER_SIZE_GB=$(python3 -c "print(f'{int($SERVER_SIZE) / 1e9:.1f} GB')" 2>/dev/null || echo "?")

# Auto-detect GPU count and VRAM
GPU_COUNT=0
GPU_NAMES=""
GPU_TOTAL_VRAM_MB=0
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
        GPU_NAMES=$(echo "$GPU_INFO" | cut -d',' -f2 | tr '\n' '|' | sed 's/|$//')
        # Sum VRAM across all GPUs (strip " MiB" suffix and sum)
        GPU_TOTAL_VRAM_MB=$(echo "$GPU_INFO" | cut -d',' -f3 | sed 's/ MiB//' | paste -sd+ | bc 2>/dev/null || echo 0)
    fi
fi

# Auto-set concurrency based on available VRAM and context size.
# Per-task estimate: model (size from server metadata) plus KV cache
# (~0.09 GB per 1K ctx).  Both are split across GPUs by tensor-parallelism
# so the aggregate formula holds.
if [ -z "$CONCURRENCY" ]; then
    if [ "$GPU_TOTAL_VRAM_MB" -gt 0 ] && [ "$SERVER_CTX" != "?" ] && [ "$SERVER_SIZE" != "?" ]; then
        CONCURRENCY=$(python3 -c "
total_gb  = (${GPU_TOTAL_VRAM_MB} - ${VRAM_HEADROOM_MB}) / 1024.0
model_gb  = ${SERVER_SIZE} / 1e9
kv_gb     = float(${SERVER_CTX}) / 1000 * 0.09
tasks     = int(total_gb / (model_gb + kv_gb))
print(max(1, min(tasks, 8)))
")
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ${VRAM_HEADROOM_MB}MB headroom, ctx=$SERVER_CTX)"
    elif [ "$GPU_TOTAL_VRAM_MB" -gt 160000 ]; then
        CONCURRENCY=8
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ctx unknown)"
    elif [ "$GPU_TOTAL_VRAM_MB" -gt 80000 ]; then
        CONCURRENCY=6
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ctx unknown)"
    elif [ "$GPU_COUNT" -ge 2 ]; then
        CONCURRENCY=4
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ctx unknown)"
    elif [ "$GPU_TOTAL_VRAM_MB" -gt 40000 ]; then
        CONCURRENCY=2
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ctx unknown)"
    else
        CONCURRENCY=1
        log "auto-set concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB VRAM, ctx unknown)"
    fi
fi

# Auto-set compaction trigger if not already set
if [ -z "${LLAMA_CPP_COMPACTION_TRIGGER:-}" ] && [ "$SERVER_CTX" != "?" ]; then
    # Leave ~10K headroom from the server's ctx size
    LLAMA_CPP_COMPACTION_TRIGGER=$((SERVER_CTX - 10000))
    log "auto-set compaction_trigger=$LLAMA_CPP_COMPACTION_TRIGGER (server ctx=$SERVER_CTX)"
fi
export LLAMA_CPP_COMPACTION_TRIGGER

# Ensure dataset exists
if [ ! -d "$PROJECT_DIR/dataset" ] || [ -z "$(ls -A "$PROJECT_DIR/dataset" 2>/dev/null)" ]; then
    echo "ERROR: dataset not found. Run ./scripts/prepare_llama_cpp_benchmark.sh first."
    exit 1
fi

# ── Print configuration ───────────────────────────────────────────────

echo ""
log "benchmark configuration"
log "======================"
log "  server:      $LLAMA_CPP_BASE_URL"
log "  model:       $SERVER_MODEL ($SERVER_SIZE_GB)"
log "  ctx size:    $SERVER_CTX"
if [ "$GPU_COUNT" -gt 0 ]; then
    log "  GPUs:        $GPU_COUNT ($GPU_NAMES)"
fi
log "  subset:      $SUBSET"
log "  concurrency: $CONCURRENCY"
log "  compaction:  ${LLAMA_CPP_COMPACTION_TRIGGER:-from config}"
if [ -n "$WANDB_PROJECT" ]; then
    log "  wandb:       $WANDB_PROJECT${WANDB_ENTITY:+ / $WANDB_ENTITY}"
fi
echo ""

# ── Run solo benchmark ────────────────────────────────────────────────

if $RUN_SOLO; then
    RUN_NAME="solo-lc-qwen3.6-27b-${SUBSET}"
    header "Solo benchmark: $RUN_NAME"

    WANDB_ARGS=""
    if [ -n "$WANDB_PROJECT" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"
    fi

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -s "$SUBSET" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        -c "$CONCURRENCY" \
        --backend docker \
        $FORCE \
        $WANDB_ARGS

    log "Solo benchmark complete → logs/$RUN_NAME/solo/"
fi

# ── Run coop benchmark ────────────────────────────────────────────────

if $RUN_COOP; then
    RUN_NAME="coop-lc-qwen3.6-27b-${SUBSET}"
    header "Coop benchmark: $RUN_NAME"

    WANDB_ARGS=""
    if [ -n "$WANDB_PROJECT" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"
    fi
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"
    fi

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -s "$SUBSET" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        -c "$CONCURRENCY" \
        --backend docker \
        $FORCE \
        $WANDB_ARGS

    log "Coop benchmark complete → logs/$RUN_NAME/coop/"
fi

# ── Summary ───────────────────────────────────────────────────────────

header "Benchmark complete"

if $RUN_SOLO; then
    SOLO_LOG="logs/solo-lc-qwen3.6-27b-${SUBSET}/solo"
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
    COOP_LOG="logs/coop-lc-qwen3.6-27b-${SUBSET}/coop"
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
