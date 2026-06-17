#!/usr/bin/env bash
# slurm_enroot_benchmark.sh — SLURM batch job for CooperBench via enroot container.
#
# Submit with:
#   sbatch scripts/slurm_enroot_benchmark.sh
#
# Override defaults via --export:
#   sbatch --export=SUBSET=flash_25,CONCURRENCY=4 scripts/slurm_enroot_benchmark.sh
#
# Overridable environment variables:
#   SUBSET               — Benchmark subset (default: flash_25)
#   CONCURRENCY           — Parallel tasks (default: auto from VRAM)
#   SOLO_ONLY             — Run only solo benchmark (default: false)
#   COOP_ONLY             — Run only coop benchmark (default: false)
#   SKIP_PREPARE          — Skip dataset download and dep sync (default: false)
#   FORCE                 — Re-run completed tasks (default: false)
#   BACKEND               — Sandbox backend (default: docker)
#   WANDB_PROJECT         — Weights & Biases project name
#   WANDB_ENTITY          — Weights & Biases entity
#   VRAM_HEADROOM_MB      — VRAM headroom in MiB (default: 15000)
#   NO_AUTO_COMPACTION    — Disable auto compaction trigger (default: false)
#
# Container / paths (override if your setup differs):
#   CONTAINER_IMAGE       — enroot .sqsh image (default: cooperbench_v0.sqsh)
#   CONTAINER_NAME        — enroot container name (default: cooperbench_v0)
#   WORKSPACE_SRC         — Host path to mount as /workspace
#   MODEL_PATH            — Path to .gguf file inside container
#   LLAMA_SERVER_PATH     — Path to llama-server binary inside container
#   LLAMA_SERVER_PORT     — Port for llama-server (default: 8050)
#   LLAMA_SERVER_CTX      — Context size (default: 65536)
#   COOPERBENCH_DIR       — Path to CooperBench repo inside container
#
# One-time setup (run once on the cluster):
#   enroot create cooperbench_v0.sqsh
#   mkdir -p /dss/.../ge56heh2/.cache/llama_cache
#   mkdir -p /dss/.../ge56heh2/.cache/hf
#   mkdir -p /dss/.../ge56heh2/CooperBench
#   cd /dss/.../ge56heh2/CooperBench && git clone ... .

# ── SLURM directives ────────────────────────────────────────────────────
#SBATCH --job-name=cooperbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x.out
#SBATCH --error=logs/slurm/%j_%x.err

set -euo pipefail

# ── Logging ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${GREEN}[slurm]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[slurm]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[slurm]${NC} $(date '+%H:%M:%S') $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

# ── Configuration defaults ─────────────────────────────────────────────

SUBSET="${SUBSET:-flash_25}"
CONCURRENCY="${CONCURRENCY:-}"
SOLO_ONLY="${SOLO_ONLY:-false}"
COOP_ONLY="${COOP_ONLY:-false}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"
FORCE="${FORCE:-false}"
BACKEND="${BACKEND:-docker}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VRAM_HEADROOM_MB="${VRAM_HEADROOM_MB:-15000}"
NO_AUTO_COMPACTION="${NO_AUTO_COMPACTION:-false}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-cooperbench_v0.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-cooperbench_v0}"
WORKSPACE_SRC="${WORKSPACE_SRC:-/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2}"
MODEL_PATH="${MODEL_PATH:-/workspace/.cache/llama_cache/Qwen_Qwen3.5-27B-Q4_K_M.gguf}"
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8050}"
LLAMA_SERVER_CTX="${LLAMA_SERVER_CTX:-65536}"
COOPERBENCH_DIR="${COOPERBENCH_DIR:-/workspace/CooperBench}"

LLAMA_CPP_BASE_URL="http://localhost:${LLAMA_SERVER_PORT}/v1"
LLAMA_CPP_API_KEY="local-llama-cpp"
LLAMA_CPP_MODEL="openai/Qwen3.6-27B-Q4_K_M.gguf"

FORCE_FLAG=""
if [ "$FORCE" = "true" ]; then FORCE_FLAG="--force"; fi

NO_AUTO_COMPACTION_FLAG=""
if [ "$NO_AUTO_COMPACTION" = "true" ]; then NO_AUTO_COMPACTION_FLAG="--no-auto-compaction"; fi

WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"; fi
if [ -n "$WANDB_ENTITY" ];  then WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"; fi

# ── Pre-flight ─────────────────────────────────────────────────────────

header "SLURM enroot CooperBench job"
log "job id:     ${SLURM_JOB_ID:-?}"
log "node:       ${SLURMD_NODENAME:-?}"
log "gpus:       ${SLURM_GPUS:-1}"
log "subset:     $SUBSET"
log "concurrency: ${CONCURRENCY:-auto}"
log "backend:    $BACKEND"
log "workspace:  $WORKSPACE_SRC"
log "model:      $MODEL_PATH"

mkdir -p logs/slurm

if ! command -v enroot &>/dev/null; then
    err "enroot not found in PATH — load the enroot module: module load enroot"
    exit 1
fi

# ── Prepare enroot container image ─────────────────────────────────────

if [ -f "$CONTAINER_IMAGE" ]; then
    if ! enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}\b"; then
        log "importing container image '$CONTAINER_IMAGE' ..."
        enroot create --name "$CONTAINER_NAME" "$CONTAINER_IMAGE"
        log "container '$CONTAINER_NAME' created"
    else
        log "container '$CONTAINER_NAME' already exists"
    fi
elif enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}\b"; then
    log "found existing enroot container '$CONTAINER_NAME'"
else
    err "container image '$CONTAINER_IMAGE' not found and"
    err "enroot container '$CONTAINER_NAME' not in 'enroot list'."
    err ""
    err "Create it first:   enroot create cooperbench_v0.sqsh"
    err "Or from registry:  enroot import docker://your-registry/cooperbench:v0"
    exit 1
fi

if [ ! -d "$WORKSPACE_SRC" ]; then
    err "workspace source '$WORKSPACE_SRC' does not exist"
    exit 1
fi
log "workspace mount: $WORKSPACE_SRC → /workspace"

# ── Build the inner container script ───────────────────────────────────
# This entire script runs inside ONE enroot start invocation so that the
# llama-server stays alive for the whole benchmark.  We use a quoted
# heredoc to prevent outer-shell variable expansion — all values are
# passed via environment variables (-e flags on enroot start).

INNER_SCRIPT=$(cat <<'ENDOFSCRIPT'
set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${GREEN}[inner]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[inner]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[inner]${NC} $(date '+%H:%M:%S') $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

# ── Cleanup trap ───────────────────────────────────────────────────────

LLAMA_PID=""
cleanup() {
    local ec=$?
    if [ -n "$LLAMA_PID" ]; then
        kill "$LLAMA_PID" 2>/dev/null || true
        log "stopped llama-server (pid=$LLAMA_PID)"
    fi
    exit $ec
}
trap cleanup EXIT

# ── Step 1: start llama-server ─────────────────────────────────────────

header "Starting llama-server"

if [ -z "${LLAMA_SERVER_PATH:-}" ]; then
    for candidate in \
        /workspace/llama.cpp/build/bin/llama-server \
        /usr/local/bin/llama-server \
        /usr/bin/llama-server; do
        if [ -x "$candidate" ]; then
            LLAMA_SERVER_PATH="$candidate"
            break
        fi
    done
    if [ -z "${LLAMA_SERVER_PATH:-}" ]; then
        LLAMA_SERVER_PATH="$(which llama-server 2>/dev/null || true)"
    fi
fi

if [ -z "${LLAMA_SERVER_PATH:-}" ] || [ ! -x "$LLAMA_SERVER_PATH" ]; then
    err "llama-server binary not found inside container"
    err "Set LLAMA_SERVER_PATH in the sbatch script or export it."
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    err "model not found: $MODEL_PATH"
    exit 1
fi

log "binary: $LLAMA_SERVER_PATH"
log "model:  $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
log "port:   $LLAMA_SERVER_PORT"
log "ctx:    $LLAMA_SERVER_CTX"

nohup "$LLAMA_SERVER_PATH" \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "$LLAMA_SERVER_PORT" \
    --ctx-size "$LLAMA_SERVER_CTX" \
    --n-gpu-layers 99 \
    --api-key "$LLAMA_CPP_API_KEY" \
    --flash-attn auto \
    --jinja \
    > /tmp/llama-server-${LLAMA_SERVER_PORT}.log 2>&1 &

LLAMA_PID=$!
echo "$LLAMA_PID" > /tmp/llama-server-${LLAMA_SERVER_PORT}.pid
log "llama-server pid: $LLAMA_PID"

# Wait for server to be ready
READY=false
for i in $(seq 1 60); do
    if curl -sf -H "Authorization: Bearer $LLAMA_CPP_API_KEY" \
       "http://localhost:${LLAMA_SERVER_PORT}/v1/models" >/dev/null 2>&1; then
        READY=true
        log "llama-server ready after ${i}s"
        break
    fi
    sleep 2
done

if ! $READY; then
    err "llama-server failed to start after 120s"
    err "last 20 lines of server log:"
    tail -20 "/tmp/llama-server-${LLAMA_SERVER_PORT}.log" 2>/dev/null || true
    exit 1
fi

# ── Step 2: prepare benchmark (optional) ───────────────────────────────

if [ "${SKIP_PREPARE:-false}" != "true" ]; then
    header "Preparing benchmark"

    cd "$COOPERBENCH_DIR"

    log "syncing Python dependencies..."
    uv sync --extra wandb 2>&1 | tail -3

    if [ -d "dataset" ] && [ "$(ls -A dataset 2>/dev/null)" ]; then
        log "dataset already present"
    else
        log "downloading dataset..."
        uv run cooperbench prepare
        log "dataset download complete"
    fi

    log "testing connectivity to llama-server..."
    uv run python -c "
import litellm, os
os.environ['LITELLM_LOG'] = 'ERROR'
resp = litellm.completion(
    model='$LLAMA_CPP_MODEL',
    messages=[{'role': 'user', 'content': 'Say hi'}],
    api_base='$LLAMA_CPP_BASE_URL',
    api_key='$LLAMA_CPP_API_KEY',
)
print('Response:', resp.choices[0].message.content[:80])
print('connectivity OK')
"
    log "prepare complete"
else
    log "skipping prepare (SKIP_PREPARE=true)"
    cd "$COOPERBENCH_DIR"
fi

# ── Step 3: gather server info + auto-configure ────────────────────────

header "Gathering server info"

SERVER_INFO=$(curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models")
SERVER_CTX=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta'].get('n_ctx','?'))" 2>/dev/null || echo "?")
SERVER_SIZE=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['meta'].get('size','?'))" 2>/dev/null || echo "?")
SERVER_ID=$(echo "$SERVER_INFO" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
log "model id:   $SERVER_ID"
log "ctx size:   $SERVER_CTX"
log "model size: $(python3 -c "print(f'{int($SERVER_SIZE)/1e9:.1f} GB')" 2>/dev/null || echo "?")"

# Auto-set concurrency
CONCURRENCY_VAL="$CONCURRENCY"
if [ -z "$CONCURRENCY_VAL" ]; then
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | sed 's/ MiB//' || echo 0)
    if [ "$GPU_VRAM_MB" -gt 0 ] && [ "$SERVER_CTX" != "?" ] && [ "$SERVER_SIZE" != "?" ]; then
        CONCURRENCY_VAL=$(python3 -c "
total_gb    = (${GPU_VRAM_MB} - ${VRAM_HEADROOM_MB}) / 1024.0
model_gb    = ${SERVER_SIZE} / 1e9
kv_gb       = float(${SERVER_CTX}) / 1000 * 0.09
kv_budget   = total_gb - model_gb
tasks       = int(kv_budget / kv_gb) if kv_gb > 0 else 1
print(max(1, min(tasks, 8)))
")
        log "auto concurrency: $CONCURRENCY_VAL  (GPU ${GPU_VRAM_MB} MiB, model $(python3 -c "print(f'${SERVER_SIZE}/1e9:.1f')") GB)"
    elif [ "$GPU_VRAM_MB" -gt 160000 ]; then
        CONCURRENCY_VAL=8
    elif [ "$GPU_VRAM_MB" -gt 80000 ]; then
        CONCURRENCY_VAL=6
    elif [ "$GPU_VRAM_MB" -gt 40000 ]; then
        CONCURRENCY_VAL=2
    else
        CONCURRENCY_VAL=1
    fi
fi

# Auto-set compaction trigger
if [ "${NO_AUTO_COMPACTION:-false}" != "true" ] && [ -z "${LLAMA_CPP_COMPACTION_TRIGGER:-}" ] && [ "$SERVER_CTX" != "?" ]; then
    export LLAMA_CPP_COMPACTION_TRIGGER=$(python3 -c "print(int(float(${SERVER_CTX}) * 0.6))")
    log "auto compaction trigger: $LLAMA_CPP_COMPACTION_TRIGGER (60% of $SERVER_CTX)"
fi

# Check dataset
if [ ! -d "dataset" ] || [ -z "$(ls -A dataset 2>/dev/null)" ]; then
    warn "dataset not found — run without SKIP_PREPARE=true first"
fi

# ── Step 4: run benchmarks ─────────────────────────────────────────────

echo ""
log "Benchmark configuration"
log "  server:      $LLAMA_CPP_BASE_URL"
log "  model:       $LLAMA_CPP_MODEL"
log "  subset:      $SUBSET"
log "  concurrency: $CONCURRENCY_VAL"
log "  backend:     $BACKEND"
log "  force:       ${FORCE:-false}"
echo ""

# ── Solo benchmark ──

if [ "${COOP_ONLY:-false}" != "true" ]; then
    RUN_NAME="solo-lc-qwen3.6-27b-${SUBSET}"
    header "Solo benchmark: $RUN_NAME"

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -s "$SUBSET" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        -c "$CONCURRENCY_VAL" \
        --backend "$BACKEND" \
        ${FORCE_FLAG:-} \
        ${WANDB_ARGS:-}

    log "Solo benchmark complete → logs/$RUN_NAME/solo/"
fi

# ── Coop benchmark ──

if [ "${SOLO_ONLY:-false}" != "true" ]; then
    RUN_NAME="coop-lc-qwen3.6-27b-${SUBSET}"
    header "Coop benchmark: $RUN_NAME"

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -s "$SUBSET" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        -c "$CONCURRENCY_VAL" \
        --backend "$BACKEND" \
        ${FORCE_FLAG:-} \
        ${WANDB_ARGS:-}

    log "Coop benchmark complete → logs/$RUN_NAME/coop/"
fi

# ── Summary ──

header "Benchmark complete"

if [ "${COOP_ONLY:-false}" != "true" ]; then
    SOLO_LOG="logs/solo-lc-qwen3.6-27b-${SUBSET}/solo"
    if [ -f "$SOLO_LOG/summary.json" ]; then
        python3 -c "
import json
with open('$SOLO_LOG/summary.json') as f:
    s = json.load(f)
print(f'[solo] {s.get(\"completed\",0)} completed, {s.get(\"failed\",0)} failed, {s.get(\"skipped\",0)} skipped')
e = s.get('eval', {})
if e:
    print(f'  eval: {e.get(\"passed\",0)} passed, {e.get(\"failed\",0)} failed ({e.get(\"pass_rate\",0)*100:.0f}%)')
" || true
    fi
fi

if [ "${SOLO_ONLY:-false}" != "true" ]; then
    COOP_LOG="logs/coop-lc-qwen3.6-27b-${SUBSET}/coop"
    if [ -f "$COOP_LOG/summary.json" ]; then
        python3 -c "
import json
with open('$COOP_LOG/summary.json') as f:
    s = json.load(f)
print(f'[coop] {s.get(\"completed\",0)} completed, {s.get(\"failed\",0)} failed, {s.get(\"skipped\",0)} skipped')
e = s.get('eval', {})
if e:
    print(f'  eval: {e.get(\"passed\",0)} passed, {e.get(\"failed\",0)} failed ({e.get(\"pass_rate\",0)*100:.0f}%)')
" || true
    fi
fi
ENDOFSCRIPT
)

# ── Launch: single enroot invocation for the full workflow ─────────────
# Everything runs inside one enroot container so the llama-server
# background process survives for the entire benchmark.

header "Launching enroot container"

# shellcheck disable=SC2086
enroot start --root --rw \
    -e HF_HOME=/workspace/.cache/hf \
    -e LLAMA_CACHE=/workspace/.cache/llama_cache \
    -e LLAMA_SERVER_PATH="$LLAMA_SERVER_PATH" \
    -e LLAMA_SERVER_PORT="$LLAMA_SERVER_PORT" \
    -e LLAMA_SERVER_CTX="$LLAMA_SERVER_CTX" \
    -e LLAMA_CPP_BASE_URL="$LLAMA_CPP_BASE_URL" \
    -e LLAMA_CPP_API_KEY="$LLAMA_CPP_API_KEY" \
    -e LLAMA_CPP_MODEL="$LLAMA_CPP_MODEL" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e COOPERBENCH_DIR="$COOPERBENCH_DIR" \
    -e SUBSET="$SUBSET" \
    -e CONCURRENCY="$CONCURRENCY" \
    -e SOLO_ONLY="$SOLO_ONLY" \
    -e COOP_ONLY="$COOP_ONLY" \
    -e SKIP_PREPARE="$SKIP_PREPARE" \
    -e FORCE="$FORCE" \
    -e FORCE_FLAG="$FORCE_FLAG" \
    -e BACKEND="$BACKEND" \
    -e VRAM_HEADROOM_MB="$VRAM_HEADROOM_MB" \
    -e NO_AUTO_COMPACTION="$NO_AUTO_COMPACTION" \
    -e WANDB_ARGS="$WANDB_ARGS" \
    -e WANDB_PROJECT="$WANDB_PROJECT" \
    -e WANDB_ENTITY="$WANDB_ENTITY" \
    --mount "${WORKSPACE_SRC}:/workspace" \
    "$CONTAINER_NAME" \
    bash -c "$INNER_SCRIPT"

header "Job ${SLURM_JOB_ID:-?} complete"
