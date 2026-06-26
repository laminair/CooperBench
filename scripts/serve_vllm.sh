#!/usr/bin/env bash
# serve_vllm.sh — Start a local vLLM server (Anthropic /v1/messages native).
#
# Designed for the `claude_code` agent to talk to via `--base-url`.  The
# benchmark launches a vLLM container (see scripts/setup_vastai.sh) bound to
# the host's loopback on $VLLM_PORT, and agent containers reach it through
# the host.docker.internal rewrite in the claude_code adapter.
#
# Usage:
#   ./scripts/serve_vllm.sh [--bg] [--kill]
#
# Environment variables (all optional; defaults are the Vast.ai defaults):
#   VLLM_MODEL                    — HF model id (default: cyankiwi/Qwen3.6-27B-AWQ-INT4)
#   VLLM_PORT                     — Server port (default: 8000)
#   VLLM_HOST                     — Bind address (default: 0.0.0.0)
#   VLLM_MAX_MODEL_LEN            — Context size (default: 65536)
#   VLLM_GPU_MEMORY_UTILIZATION   — Fraction of VRAM for KV+activations (default: 0.92)
#   VLLM_QUANTIZATION             — Quant format (default: awq; "" disables)
#   VLLM_TOOL_PARSER              — Tool-call parser (default: qwen3_coder)
#   VLLM_TENSOR_PARALLEL_SIZE     — TP size (default: auto from GPU count)
#   VLLM_SERVED_NAME              — Public model name (default: derived from $VLLM_MODEL)
#   VLLM_EXTRA_ARGS               — Extra vllm flags (whitespace-separated)
#   HF_HOME                       — HF cache dir (default: ~/.cache/huggingface)
#   HF_TOKEN                      — Optional HF token for gated models

set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq}"
VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-qwen3_coder}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
BG_MODE=false
KILL_MODE=false

while [ $# -gt 0 ]; do
    case "$1" in
        --bg)   BG_MODE=true; shift ;;
        --kill) KILL_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[vllm]${NC} $*"; }
warn() { echo -e "${YELLOW}[vllm]${NC} $*"; }
err()  { echo -e "${RED}[vllm]${NC} $*"; }

# ── Sanity: vllm binary present? ─────────────────────────────────────
if ! command -v vllm &>/dev/null; then
    err "vllm not found in PATH.  Activate the venv that has vllm installed,"
    err "or rebuild the Vast.ai image (Dockerfile.vastai installs it)."
    exit 1
fi

# ── Kill existing if requested ───────────────────────────────────────
if $KILL_MODE; then
    EXISTING=$(lsof -ti :"$VLLM_PORT" 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        log "killing existing vllm on port $VLLM_PORT (pid $EXISTING)"
        kill $EXISTING 2>/dev/null || true
        sleep 2
    fi
fi

if lsof -i :"$VLLM_PORT" &>/dev/null; then
    err "Port $VLLM_PORT is already in use."
    err "Use --kill to stop the existing process, or set VLLM_PORT."
    exit 1
fi

# ── GPU detection ────────────────────────────────────────────────────
GPU_COUNT=0
GPU_NAMES=""
GPU_TOTAL_VRAM_MB=0
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
        GPU_NAMES=$(echo "$GPU_INFO" | cut -d',' -f2 | tr '\n' '|' | sed 's/|$//')
        GPU_TOTAL_VRAM_MB=$(echo "$GPU_INFO" | cut -d',' -f3 | sed 's/ MiB//' | paste -sd+ | bc 2>/dev/null || echo 0)
        GPU_IDS=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    fi
fi

# ── Auto tensor-parallel size ────────────────────────────────────────
if [ -z "$VLLM_TENSOR_PARALLEL_SIZE" ]; then
    if [ "$GPU_COUNT" -le 0 ]; then
        VLLM_TENSOR_PARALLEL_SIZE=1
    else
        VLLM_TENSOR_PARALLEL_SIZE="$GPU_COUNT"
    fi
fi

# ── Default served-model name ────────────────────────────────────────
if [ -z "${VLLM_SERVED_NAME:-}" ]; then
    VLLM_SERVED_NAME=$(echo "$VLLM_MODEL" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]')
fi

# ── Print configuration ──────────────────────────────────────────────
echo ""
log "vLLM server configuration"
log "=========================="
log "  model:    $VLLM_MODEL"
log "  quant:    ${VLLM_QUANTIZATION:-<none>}"
log "  served as: $VLLM_SERVED_NAME"
log "  bind:     $VLLM_HOST:$VLLM_PORT"
log "  ctx:      $VLLM_MAX_MODEL_LEN"
log "  gpu-mem:  $VLLM_GPU_MEMORY_UTILIZATION"
log "  tp:       $VLLM_TENSOR_PARALLEL_SIZE"
if [ "$GPU_COUNT" -gt 0 ]; then
    log "  GPUs:     $GPU_COUNT ($GPU_NAMES)"
    log "  VRAM:     $((GPU_TOTAL_VRAM_MB / 1024)) GB"
    log "  CUDA_VIS: $CUDA_VISIBLE_DEVICES"
else
    warn "  No NVIDIA GPUs detected — will run on CPU (very slow)"
fi
echo ""

# ── Build vllm serve command ─────────────────────────────────────────
CMD=(vllm serve "$VLLM_MODEL"
    --host "$VLLM_HOST"
    --port "$VLLM_PORT"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
    --served-model-name "$VLLM_SERVED_NAME"
    --enable-auto-tool-choice
    --tool-call-parser "$VLLM_TOOL_PARSER"
)
if [ -n "$VLLM_QUANTIZATION" ]; then
    CMD+=(--quantization "$VLLM_QUANTIZATION")
fi
if [ -n "$VLLM_EXTRA_ARGS" ]; then
    IFS=' ' read -ra EXTRA <<< "$VLLM_EXTRA_ARGS"
    CMD+=("${EXTRA[@]}")
fi

# ── Launch ───────────────────────────────────────────────────────────
if $BG_MODE; then
    log "starting vllm in background..."
    nohup "${CMD[@]}" > "/tmp/vllm-${VLLM_PORT}.log" 2>&1 &
    VLLM_PID=$!
    echo "$VLLM_PID" > "/tmp/vllm-${VLLM_PORT}.pid"

    for i in $(seq 1 90); do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
            log "vllm ready (pid=$VLLM_PID, port=$VLLM_PORT)"
            log "  logs:    /tmp/vllm-${VLLM_PORT}.log"
            log "  models:  http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
            log "  messages: http://${VLLM_HOST}:${VLLM_PORT}/v1/messages  (Anthropic)"
            exit 0
        fi
        sleep 2
    done
    err "vllm failed to start within 180s — see /tmp/vllm-${VLLM_PORT}.log"
    tail -40 "/tmp/vllm-${VLLM_PORT}.log" 2>/dev/null || true
    exit 1
else
    log "starting vllm in foreground (Ctrl+C to stop)..."
    exec "${CMD[@]}"
fi
