#!/usr/bin/env bash
# launch_vastai_benchmark.sh — Run CooperBench solo + coop benchmarks on Vast.ai
# against a vLLM-served local model and the `claude_code` agent.
#
# Usage:
#   ./scripts/launch_vastai_benchmark.sh [OPTIONS]
#
# Options:
#   --solo-only         Run only the solo benchmark
#   --coop-only         Run only the coop benchmark
#   --subset NAME       Subset to use (default: flash_25)
#   --concurrency N     Parallel tasks (default: auto from VRAM/ctx)
#   --force             Re-run tasks even if results exist
#   --no-auto-compaction  Don't auto-set COOPERBENCH_COMPACTION_TRIGGER
#   --wandb-project P   Weights & Biases project name
#   --wandb-entity E    W&B entity
#   --base-url URL      Override vllm endpoint (default: $VLLM_BASE_URL - /v1)
#   --model NAME        Override served model name (default: from vllm /v1/models)
#   --max-turns N       Claude Code max-turns (default: unset; agent default)
#
# Environment variables:
#   VLLM_BASE_URL                   — vllm endpoint (default: http://localhost:8000/v1)
#   VLLM_API_KEY                    — vllm auth (default: dummy)
#   VLLM_MODEL                      — served model name (auto-discovered if unset)
#   COOPERBENCH_CONCURRENCY         — Override parallel task count
#   COOPERBENCH_VRAM_HEADROOM_MB    — VRAM reserved for spikes (default: 15000)
#   COOPERBENCH_COMPACTION_TRIGGER  — Override claude-code context-compaction trigger
#   WANDB_PROJECT / WANDB_ENTITY     — W&B logging

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()    { echo -e "${GREEN}[bench]${NC} $*"; }
warn()   { echo -e "${YELLOW}[bench]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
VLLM_MODEL="${VLLM_MODEL:-}"
SUBSET="flash_25"
RUN_SOLO=true
RUN_COOP=true
FORCE=""
CONCURRENCY="${COOPERBENCH_CONCURRENCY:-}"
VRAM_HEADROOM_MB="${COOPERBENCH_VRAM_HEADROOM_MB:-15000}"
AUTO_COMPACTION=true
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
BASE_URL_OVERRIDE=""
MODEL_OVERRIDE=""
MAX_TURNS=""

while [ $# -gt 0 ]; do
    case "$1" in
        --solo-only) RUN_COOP=false; shift ;;
        --coop-only) RUN_SOLO=false; shift ;;
        --subset) SUBSET="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --no-auto-compaction) AUTO_COMPACTION=false; shift ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        --base-url) BASE_URL_OVERRIDE="$2"; shift 2 ;;
        --model) MODEL_OVERRIDE="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Resolve the host-reachable base URL ──────────────────────────────
# The benchmark runs ON the Vast.ai VM.  Agent containers need to reach vllm
# through host.docker.internal.  Passing a bare `localhost:8000` triggers the
# claude_code adapter's existing rewrite, but we explicitly use
# host.docker.internal here so the flag's intent is obvious in the logs.
HOST_BASE_URL="${BASE_URL_OVERRIDE:-${VLLM_BASE_URL}}"
if [[ "$HOST_BASE_URL" == *"localhost"* || "$HOST_BASE_URL" == *"127.0.0.1"* ]]; then
    # Rewrite localhost:8000/v1  -> host.docker.internal:8000/v1
    HOST_BASE_URL=$(echo "$HOST_BASE_URL" | sed -E 's#(localhost|127\.0\.0\.1)#host.docker.internal#')
fi

# ── Verify vLLM is reachable ─────────────────────────────────────────
if ! curl -s "${HOST_BASE_URL%/v1}/v1/models" >/dev/null 2>&1; then
    err "vLLM not reachable at ${HOST_BASE_URL%/v1}/v1/models"
    err "Run scripts/setup_vastai.sh first."
    exit 1
fi

# ── Gather server info ───────────────────────────────────────────────
SERVER_INFO=$(curl -s "${HOST_BASE_URL%/v1}/v1/models")
if [ -n "$MODEL_OVERRIDE" ]; then
    SERVER_MODEL="$MODEL_OVERRIDE"
else
    if [ -n "$VLLM_MODEL" ]; then
        SERVER_MODEL="$VLLM_MODEL"
    else
        SERVER_MODEL=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
    fi
fi
# vLLM doesn't expose n_ctx in /v1/models; query /v1/messages doesn't help either.
# Use VLLM_MAX_MODEL_LEN env or 65536 fallback.
SERVER_CTX="${VLLM_MAX_MODEL_LEN:-65536}"
log "model:   $SERVER_MODEL"
log "ctx:     $SERVER_CTX"
log "server:  $HOST_BASE_URL"

# ── Auto-set concurrency from VRAM ───────────────────────────────────
GPU_COUNT=0
GPU_NAMES=""
GPU_TOTAL_VRAM_MB=0
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        GPU_COUNT=$(echo "$GPU_INFO" | wc -l)
        GPU_NAMES=$(echo "$GPU_INFO" | cut -d',' -f2 | tr '\n' '|' | sed 's/|$//')
        GPU_TOTAL_VRAM_MB=$(echo "$GPU_INFO" | cut -d',' -f3 | sed 's/ MiB//' | paste -sd+ | bc 2>/dev/null || echo 0)
    fi
fi

# Rough model size: AWQ 27B ~= 14 GB.  KV cache ~ 0.09 GB per 1K ctx per request.
MODEL_GB="${COOPERBENCH_MODEL_GB:-14}"
KV_PER_TASK_GB=$(python3 -c "print(f'{float($SERVER_CTX)/1000 * 0.09:.1f}')")

if [ -z "$CONCURRENCY" ]; then
    if [ "$GPU_TOTAL_VRAM_MB" -gt 0 ]; then
        CONCURRENCY=$(python3 -c "
total_gb  = (${GPU_TOTAL_VRAM_MB} - ${VRAM_HEADROOM_MB}) / 1024.0
model_gb  = $MODEL_GB
kv_gb     = $KV_PER_TASK_GB
kv_budget = total_gb - model_gb
tasks     = int(kv_budget / kv_gb) if kv_gb > 0 else 1
print(max(1, min(tasks, 8)))
")
        log "auto concurrency=$CONCURRENCY (${GPU_COUNT}x GPU, $((GPU_TOTAL_VRAM_MB/1024)) GB, ${VRAM_HEADROOM_MB}MB headroom, model~${MODEL_GB}GB, ctx=$SERVER_CTX)"
    else
        CONCURRENCY=1
        warn "no nvidia-smi; defaulting concurrency=1"
    fi
fi

# ── Auto-set compaction trigger ──────────────────────────────────────
# CooperBench's claude_code adapter reads COOPERBENCH_COMPACTION_TRIGGER and
# forwards it as CLAUDE_CODE_COMPACTION_TRIGGER into the task container.
# 60% of ctx is a safe headroom: leaves enough room for the summary operation.
if $AUTO_COMPACTION && [ -z "${COOPERBENCH_COMPACTION_TRIGGER:-}" ]; then
    COOPERBENCH_COMPACTION_TRIGGER=$(python3 -c "print(int(float($SERVER_CTX) * 0.6))")
    log "auto compaction trigger: $COOPERBENCH_COMPACTION_TRIGGER (60% of $SERVER_CTX)"
fi
export COOPERBENCH_COMPACTION_TRIGGER

# ── Verify dataset present ───────────────────────────────────────────
if [ ! -d "dataset" ] || [ -z "$(ls -A dataset 2>/dev/null)" ]; then
    err "dataset not found.  Run scripts/prepare_vastai_benchmark.sh first."
    exit 1
fi

# ── Print configuration ──────────────────────────────────────────────
echo ""
log "benchmark configuration"
log "======================"
log "  server:      $HOST_BASE_URL"
log "  model:       $SERVER_MODEL"
log "  ctx size:    $SERVER_CTX"
if [ "$GPU_COUNT" -gt 0 ]; then
    log "  GPUs:        $GPU_COUNT ($GPU_NAMES)"
fi
log "  subset:      $SUBSET"
log "  concurrency: $CONCURRENCY"
if $AUTO_COMPACTION; then
    log "  compaction:  ${COOPERBENCH_COMPACTION_TRIGGER:-from config} (auto)"
else
    log "  compaction:  from config (auto disabled)"
fi
if [ -n "$WANDB_PROJECT" ]; then
    log "  wandb:       $WANDB_PROJECT${WANDB_ENTITY:+ / $WANDB_ENTITY}"
fi
echo ""

EXTRA_FLAGS=""
if [ -n "$MAX_TURNS" ]; then
    EXTRA_FLAGS="--agent-config <(echo 'agent: { max_turns: $MAX_TURNS }')"
    warn "--max-turns uses a temporary agent-config override"
fi

# ── Run solo ─────────────────────────────────────────────────────────
if $RUN_SOLO; then
    RUN_NAME="solo-cc-${SERVER_MODEL}-${SUBSET}"
    header "Solo benchmark: $RUN_NAME"

    WANDB_ARGS=""
    [ -n "$WANDB_PROJECT" ] && WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ]  && WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -s "$SUBSET" \
        -m "$SERVER_MODEL" \
        -a claude_code \
        -c "$CONCURRENCY" \
        --backend docker \
        --base-url "$HOST_BASE_URL" \
        --auth-token "$VLLM_API_KEY" \
        $FORCE \
        $WANDB_ARGS \
        $EXTRA_FLAGS

    log "Solo benchmark complete → logs/$RUN_NAME/solo/"
fi

# ── Run coop ─────────────────────────────────────────────────────────
if $RUN_COOP; then
    RUN_NAME="coop-cc-${SERVER_MODEL}-${SUBSET}"
    header "Coop benchmark: $RUN_NAME"

    WANDB_ARGS=""
    [ -n "$WANDB_PROJECT" ] && WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"
    [ -n "$WANDB_ENTITY" ]  && WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -s "$SUBSET" \
        -m "$SERVER_MODEL" \
        -a claude_code \
        -c "$CONCURRENCY" \
        --backend docker \
        --base-url "$HOST_BASE_URL" \
        --auth-token "$VLLM_API_KEY" \
        $FORCE \
        $WANDB_ARGS \
        $EXTRA_FLAGS

    log "Coop benchmark complete → logs/$RUN_NAME/coop/"
fi

# ── Summary ──────────────────────────────────────────────────────────
header "Benchmark complete"

if $RUN_SOLO; then
    SOLO_LOG="logs/solo-cc-${SERVER_MODEL}-${SUBSET}/solo"
    if [ -f "$SOLO_LOG/summary.json" ]; then
        echo "[solo] $(python3 -c "
import json
with open('$SOLO_LOG/summary.json') as f:
    s = json.load(f)
print(f\"{s.get('completed',0)} completed, {s.get('failed',0)} failed, {s.get('skipped',0)} skipped\")
e = s.get('eval', {})
if e:
    print(f\"  eval: {e.get('passed',0)} passed, {e.get('failed',0)} failed ({e.get('pass_rate',0)*100:.0f}%)\")
")"
    fi
fi
if $RUN_COOP; then
    COOP_LOG="logs/coop-cc-${SERVER_MODEL}-${SUBSET}/coop"
    if [ -f "$COOP_LOG/summary.json" ]; then
        echo "[coop] $(python3 -c "
import json
with open('$COOP_LOG/summary.json') as f:
    s = json.load(f)
print(f\"{s.get('completed',0)} completed, {s.get('failed',0)} failed, {s.get('skipped',0)} skipped\")
e = s.get('eval', {})
if e:
    print(f\"  eval: {e.get('passed',0)} passed, {e.get('failed',0)} failed ({e.get('pass_rate',0)*100:.0f}%)\")
")"
    fi
fi
echo ""
