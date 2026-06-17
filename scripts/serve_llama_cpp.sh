#!/usr/bin/env bash
# serve_llama_cpp.sh — Start llama.cpp server with automatic GPU detection.
#
# Usage:
#   ./scripts/serve_llama_cpp.sh [--bg] [--kill]
#
# Options:
#   --bg       Run server in background (default: foreground)
#   --kill     Kill any existing llama-server on the configured port
#
# Environment variables:
#   LLAMA_MODEL_PATH       — Path to .gguf model (auto-detected if unset)
#   LLAMA_PORT             — Server port (default: 8050)
#   LLAMA_HOST             — Bind address (default: 127.0.0.1)
#   LLAMA_CTX_SIZE         — Context size (default: 65536)
#                            Larger ctx = less concurrency.  65K is a good
#                            sweet spot for CooperBench tasks.  Use 131072
#                            only if tasks need extreme context; expect
#                            ~2x KV-cache VRAM per concurrent request.
#   LLAMA_N_GPU_LAYERS     — Layers on GPU (default: 99 = all)
#   LLAMA_API_KEY          — API key for auth (default: local-llama-cpp)
#   LLAMA_FLASH_ATTN       — Flash attention mode (default: auto)
#   LLAMA_GPU_SPLIT        — Tensor split, e.g. "0.5,0.5" (default: auto)
#   LLAMA_EXTRA_ARGS       — Additional llama-server flags

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────

LLAMA_PORT="${LLAMA_PORT:-8050}"
LLAMA_HOST="${LLAMA_HOST:-127.0.0.1}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-65536}"
LLAMA_N_GPU_LAYERS="${LLAMA_N_GPU_LAYERS:-99}"
LLAMA_API_KEY="${LLAMA_API_KEY:-local-llama-cpp}"
LLAMA_FLASH_ATTN="${LLAMA_FLASH_ATTN:-auto}"
LLAMA_GPU_SPLIT="${LLAMA_GPU_SPLIT:-}"
LLAMA_EXTRA_ARGS="${LLAMA_EXTRA_ARGS:-}"
BG_MODE=false
KILL_MODE=false

# ── Parse args ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --bg) BG_MODE=true; shift ;;
        --kill) KILL_MODE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[llama-serve]${NC} $*"; }
warn() { echo -e "${YELLOW}[llama-serve]${NC} $*"; }
err()  { echo -e "${RED}[llama-serve]${NC} $*"; }

# ── Find llama-server binary ──────────────────────────────────────────

LLAMA_SERVER=""

# Check common locations
for candidate in \
    "/workspace/llama.cpp/build/bin/llama-server"
    "$PROJECT_DIR/../local-llm-runtime/llama.cpp/build/bin/llama-server" \
    "$HOME/code/local-llm-runtime/llama.cpp/build/bin/llama-server" \
    "$(which llama-server 2>/dev/null)" \
    "/usr/local/bin/llama-server"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
        LLAMA_SERVER="$candidate"
        break
    fi
done

if [ -z "$LLAMA_SERVER" ]; then
    err "llama-server binary not found."
    err "Tried: $PROJECT_DIR/../local-llm-runtime/llama.cpp/build/bin/llama-server"
    err "       $HOME/code/local-llm-runtime/llama.cpp/build/bin/llama-server"
    err "       $(which llama-server 2>/dev/null || echo 'not in PATH')"
    err ""
    err "Set LLAMA_SERVER_PATH env var to the binary location."
    exit 1
fi

# ── Find model ────────────────────────────────────────────────────────

if [ -z "${LLAMA_MODEL_PATH:-}" ]; then
    # Auto-detect: look for Qwen3.6 gguf in common locations
    for candidate in \
        "/workspace/.cache/llama_cache/Qwen_Qwen3.5-27B-Q4_K_M.gguf" \
        "$HOME/.cache/huggingface/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf" \
        "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B*/**/*.gguf" \
        "$PROJECT_DIR/models/*.gguf"; do
        # Use find for globs that need expansion
        for f in $candidate; do
            if [ -f "$f" ]; then
                LLAMA_MODEL_PATH="$f"
                break 2
            fi
        done
    done
fi

if [ -z "${LLAMA_MODEL_PATH:-}" ] || [ ! -f "$LLAMA_MODEL_PATH" ]; then
    err "Model not found. Set LLAMA_MODEL_PATH to the .gguf file."
    err "Example: LLAMA_MODEL_PATH=$HOME/.cache/huggingface/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf"
    exit 1
fi

# ── Kill existing if requested ────────────────────────────────────────

if $KILL_MODE; then
    EXISTING=$(lsof -ti :"$LLAMA_PORT" 2>/dev/null || true)
    if [ -n "$EXISTING" ]; then
        log "killing existing llama-server on port $LLAMA_PORT (pid $EXISTING)"
        kill $EXISTING 2>/dev/null || true
        sleep 1
    fi
fi

# Check port is free
if lsof -i :"$LLAMA_PORT" &>/dev/null; then
    err "Port $LLAMA_PORT is already in use."
    err "Use --kill to stop the existing server, or set LLAMA_PORT to another port."
    exit 1
fi

# ── GPU detection ─────────────────────────────────────────────────────

GPU_COUNT=0
GPU_NAMES=""
GPU_TOTAL_VRAM_MB=0

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
        GPU_NAMES=$(echo "$GPU_INFO" | cut -d',' -f2 | tr '\n' '|' | sed 's/|$//')
        GPU_TOTAL_VRAM_MB=$(echo "$GPU_INFO" | cut -d',' -f3 | sed 's/ MiB//' | paste -sd+ | bc 2>/dev/null || echo 0)

        # Auto-set CUDA_VISIBLE_DEVICES to all GPUs
        GPU_IDS=$(seq -s, 0 $((GPU_COUNT - 1)))
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"

        # Default context: 65536 is a good balance of headroom vs concurrency.
        # Bump LLAMA_CTX_SIZE to 131072 only if you need extreme context and
        # are willing to trade concurrency.  Each doubling of ctx roughly
        # doubles KV-cache VRAM, halving the number of concurrent tasks.
    fi
fi

# ── Auto tensor split ─────────────────────────────────────────────────

if [ -z "$LLAMA_GPU_SPLIT" ] && [ "$GPU_COUNT" -gt 1 ]; then
    # Even split across all GPUs
    SPLIT_VAL=$(python3 -c "print(','.join([str(1.0/$GPU_COUNT)]*$GPU_COUNT))")
    LLAMA_GPU_SPLIT="$SPLIT_VAL"
fi

# ── Print configuration ───────────────────────────────────────────────

echo ""
log "llama.cpp server configuration"
log "=============================="
log "  binary:     $LLAMA_SERVER"
log "  model:      $LLAMA_MODEL_PATH"
log "  model size: $(du -h "$LLAMA_MODEL_PATH" | cut -f1)"
log "  host:port:  $LLAMA_HOST:$LLAMA_PORT"
log "  ctx size:   $LLAMA_CTX_SIZE"
log "  gpu layers: $LLAMA_N_GPU_LAYERS"
log "  api key:    $LLAMA_API_KEY"
log "  flash attn: $LLAMA_FLASH_ATTN"
if [ "$GPU_COUNT" -gt 0 ]; then
    log "  GPUs:        $GPU_COUNT ($GPU_NAMES)"
    log "  total VRAM:  $(python3 -c "print(f'${GPU_TOTAL_VRAM_MB}/1024:.0f')") GB"
    log "  CUDA_VIS:    $CUDA_VISIBLE_DEVICES"
    # Estimate max concurrency: model is split across GPUs, KV cache is per-request
    MODEL_GB=$(python3 -c "import os; print(f'{os.path.getsize(\"$LLAMA_MODEL_PATH\")/1e9:.1f}')" 2>/dev/null || echo "17")
    KV_PER_TASK_GB=$(python3 -c "print(f'{float($LLAMA_CTX_SIZE)/1000 * 0.09:.1f}')")
    MAX_TASKS=$(python3 -c "
        total_gb = ${GPU_TOTAL_VRAM_MB} / 1024.0
        model_gb = $MODEL_GB
        kv_gb = $KV_PER_TASK_GB
        tasks = int(total_gb / (model_gb + kv_gb))
        print(max(1, tasks))
    " 2>/dev/null || echo "?")
    log "  est concurrency: ~$MAX_TASKS tasks (model ${MODEL_GB}GB + ctx ${KV_PER_TASK_GB}GB/task)"
    if [ -n "$LLAMA_GPU_SPLIT" ]; then
        log "  gpu split:   $LLAMA_GPU_SPLIT"
    fi
else
    warn "  No NVIDIA GPUs detected — will run on CPU"
fi
echo ""

# ── Build command ─────────────────────────────────────────────────────

CMD=(
    "$LLAMA_SERVER"
    --model "$LLAMA_MODEL_PATH"
    --host "$LLAMA_HOST"
    --port "$LLAMA_PORT"
    --ctx-size "$LLAMA_CTX_SIZE"
    --n-gpu-layers "$LLAMA_N_GPU_LAYERS"
    --api-key "$LLAMA_API_KEY"
    --flash-attn "$LLAMA_FLASH_ATTN"
    --jinja
)

if [ -n "$LLAMA_GPU_SPLIT" ] && [ "$GPU_COUNT" -gt 1 ]; then
    CMD+=(--tensor-split "$LLAMA_GPU_SPLIT")
fi

if [ -n "$LLAMA_EXTRA_ARGS" ]; then
    # Word-split extra args (user is responsible for quoting)
    IFS=' ' read -ra EXTRA <<< "$LLAMA_EXTRA_ARGS"
    CMD+=("${EXTRA[@]}")
fi

# ── Launch ────────────────────────────────────────────────────────────

if $BG_MODE; then
    log "starting llama-server in background..."
    nohup "${CMD[@]}" > /tmp/llama-server-${LLAMA_PORT}.log 2>&1 &
    LLAMA_PID=$!
    echo $LLAMA_PID > /tmp/llama-server-${LLAMA_PORT}.pid

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s -H "Authorization: Bearer $LLAMA_API_KEY" "http://${LLAMA_HOST}:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
            log "server ready (pid=$LLAMA_PID, port=$LLAMA_PORT)"
            log "logs: /tmp/llama-server-${LLAMA_PORT}.log"
            log "API:  http://${LLAMA_HOST}:${LLAMA_PORT}/v1"
            exit 0
        fi
        sleep 2
    done
    err "server failed to start after 60s — check /tmp/llama-server-${LLAMA_PORT}.log"
    exit 1
else
    log "starting llama-server in foreground (Ctrl+C to stop)..."
    exec "${CMD[@]}"
fi
