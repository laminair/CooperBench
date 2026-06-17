#!/usr/bin/env bash
# slurm_enroot_benchmark.sh — SLURM batch job for CooperBench via enroot + rootful Podman.
#
# One job per setting. Submit separately for solo and coop:
#
#   sbatch --time=10:00:00 --export=SETTING=solo scripts/slurm_enroot_benchmark.sh
#   sbatch --time=16:00:00 --export=SETTING=coop scripts/slurm_enroot_benchmark.sh
#
# For the full 652-task benchmark (not flash_25 subset):
#   sbatch --time=24:00:00 --export=SETTING=solo,SUBSET=full ...
#   sbatch --time=36:00:00 --export=SETTING=coop,SUBSET=full ...
#
# Overridable environment variables (via --export on sbatch):
#   SETTING              — Benchmark setting: solo or coop (default: solo)
#   SUBSET               — Benchmark subset (default: flash_25)
#   CONCURRENCY          — Parallel tasks (default: auto from VRAM)
#   SKIP_PREPARE         — Skip dataset download and dep sync (default: false)
#   FORCE                — Re-run completed tasks (default: false)
#   BACKEND              — Sandbox backend (default: docker)
#   WANDB_PROJECT        — Weights & Biases project name
#   WANDB_ENTITY         — Weights & Biases entity
#   VRAM_HEADROOM_MB     — VRAM headroom in MiB (default: 15000)
#   NO_AUTO_COMPACTION   — Disable auto compaction trigger (default: false)
#
# Container / paths (override if your setup differs):
#   CONTAINER_IMAGE      — enroot .sqsh image (default: cooperbench_v1.sqsh)
#   CONTAINER_NAME       — enroot container name (default: cooperbench_v1)
#   WORKSPACE_SRC        — Host path mounted as /workspace
#   MODEL_PATH           — Path to .gguf file inside container
#   LLAMA_SERVER_PATH    — Path to llama-server binary inside container
#   LLAMA_SERVER_PORT    — Port for llama-server (default: 8050)
#   LLAMA_SERVER_CTX     — Context size (default: 65536)
#   COOPERBENCH_DIR      — Path to CooperBench repo inside container
#   AGENT_CONFIG         — Path to agent YAML config inside container
#                          (default: $COOPERBENCH_DIR/src/cooperbench/agents/llama_cpp/config/enroot_hpc.yaml)
#
# One-time setup (run once on the cluster, not inside a job):
#   enroot create --name cooperbench_v1 cooperbench_v1.sqsh
#   mkdir -p /dss/.../ge56heh2/.cache/{llama_cache,hf}
#   mkdir -p /dss/.../ge56heh2/CooperBench
#   cd /dss/.../ge56heh2/CooperBench && git clone ... .

# ── SLURM directives ────────────────────────────────────────────────────
#SBATCH --job-name=cooperbench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j_%x_%a.out
#SBATCH --error=logs/slurm/%j_%x_%a.err

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

SETTING="${SETTING:-solo}"
SUBSET="${SUBSET:-flash_25}"
CONCURRENCY="${CONCURRENCY:-}"
SKIP_PREPARE="${SKIP_PREPARE:-false}"
FORCE="${FORCE:-false}"
BACKEND="${BACKEND:-docker}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VRAM_HEADROOM_MB="${VRAM_HEADROOM_MB:-15000}"
NO_AUTO_COMPACTION="${NO_AUTO_COMPACTION:-false}"

CONTAINER_IMAGE="${CONTAINER_IMAGE:-cooperbench_v1.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-cooperbench_v1}"
WORKSPACE_SRC="${WORKSPACE_SRC:-/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2}"
MODEL_PATH="${MODEL_PATH:-/workspace/.cache/llama_cache/Qwen_Qwen3.5-27B-Q4_K_M.gguf}"
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8050}"
LLAMA_SERVER_CTX="${LLAMA_SERVER_CTX:-65536}"
COOPERBENCH_DIR="${COOPERBENCH_DIR:-/workspace/CooperBench}"

# AGENT_CONFIG is resolved inside the container (needs COOPERBENCH_DIR to be available),
# so pass it as a pattern string and resolve it in the inner script.
AGENT_CONFIG="${AGENT_CONFIG:-}"

LLAMA_CPP_BASE_URL="http://localhost:${LLAMA_SERVER_PORT}/v1"
LLAMA_CPP_API_KEY="local-llama-cpp"
LLAMA_CPP_MODEL="openai/Qwen3.6-27B-Q4_K_M.gguf"

WANDB_ARGS=""
if [ -n "$WANDB_PROJECT" ]; then WANDB_ARGS="$WANDB_ARGS --wandb-project $WANDB_PROJECT"; fi
if [ -n "$WANDB_ENTITY" ];  then WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"; fi

# ── Pre-flight ─────────────────────────────────────────────────────────

header "SLURM enroot CooperBench job"
log "job id:     ${SLURM_JOB_ID:-?}"
log "node:       ${SLURMD_NODENAME:-?}"
log "gpus:       ${SLURM_GPUS:-1}"
log "setting:    $SETTING"
log "subset:     $SUBSET"
log "concurrency: ${CONCURRENCY:-auto}"
log "backend:    $BACKEND"
log "workspace:  $WORKSPACE_SRC"
log "model:      $MODEL_PATH"

mkdir -p logs/slurm

if [[ "$SETTING" != "solo" && "$SETTING" != "coop" ]]; then
    err "SETTING must be 'solo' or 'coop', got: $SETTING"
    exit 1
fi

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
        log "container '$CONTAINER_NAME' already exists, skipping import"
    fi
elif enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}\b"; then
    log "found existing enroot container '$CONTAINER_NAME'"
else
    err "container image '$CONTAINER_IMAGE' not found and"
    err "enroot container '$CONTAINER_NAME' not in 'enroot list'."
    err ""
    err "Create it first:   enroot create --name cooperbench_v1 cooperbench_v1.sqsh"
    err "Or from registry:  enroot import docker://your-registry/cooperbench:v1"
    exit 1
fi

if [ ! -d "$WORKSPACE_SRC" ]; then
    err "workspace source '$WORKSPACE_SRC' does not exist"
    exit 1
fi
log "workspace mount: $WORKSPACE_SRC → /workspace"

# ── Build the inner container script ───────────────────────────────────
# Runs inside ONE enroot invocation so llama-server survives the full
# benchmark.  Quoted heredoc prevents outer-shell expansion — all values
# arrive as environment variables via enroot -e flags.

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

# Resolve AGENT_CONFIG now that COOPERBENCH_DIR is set
if [ -z "${AGENT_CONFIG:-}" ]; then
    AGENT_CONFIG="${COOPERBENCH_DIR}/src/cooperbench/agents/llama_cpp/config/enroot_hpc.yaml"
fi

# ── Cleanup trap ───────────────────────────────────────────────────────

LLAMA_PID=""
PODMAN_PID=""
cleanup() {
    local ec=$?
    if [ -n "$LLAMA_PID" ]; then
        kill "$LLAMA_PID" 2>/dev/null || true
        log "stopped llama-server (pid=$LLAMA_PID)"
    fi
    if [ -n "$PODMAN_PID" ]; then
        kill "$PODMAN_PID" 2>/dev/null || true
        log "stopped Podman socket service (pid=$PODMAN_PID)"
    fi
    exit $ec
}
trap cleanup EXIT

# ── Step 1: start Podman socket ────────────────────────────────────────
# Rootful Podman nested inside Enroot.  DOCKER_HOST is already set via
# enroot -e, but start the service here so it's definitely alive.

header "Starting rootful Podman socket"

mkdir -p /run/podman
nohup podman system service -t 0 unix:///run/podman/podman.sock \
    > /tmp/podman-service.log 2>&1 &
PODMAN_PID=$!
export DOCKER_HOST=unix:///run/podman/podman.sock
export MSWEA_DOCKER_EXECUTABLE=podman

log "Podman service pid: $PODMAN_PID"

# Wait for socket to appear
for i in $(seq 1 20); do
    if [ -S /run/podman/podman.sock ]; then
        log "Podman socket ready after ${i}s"
        break
    fi
    sleep 1
done
if [ ! -S /run/podman/podman.sock ]; then
    err "Podman socket not available after 20s"
    err "Podman service log:"
    cat /tmp/podman-service.log 2>/dev/null || true
    exit 1
fi

# Smoke-test the socket
if ! podman info --log-level=error >/dev/null 2>&1; then
    err "podman info failed — rootful Podman may not be configured correctly"
    cat /tmp/podman-service.log 2>/dev/null || true
    exit 1
fi
log "Podman smoke-test passed"

# ── Step 2: start llama-server ─────────────────────────────────────────

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
    err "Set LLAMA_SERVER_PATH in the sbatch --export or as a default in this script."
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
        log "llama-server ready after $((i * 2))s"
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

# ── Step 3: prepare benchmark (optional) ───────────────────────────────

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

# ── Step 4: gather server info + auto-configure ────────────────────────

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
        log "auto concurrency: $CONCURRENCY_VAL  (GPU ${GPU_VRAM_MB} MiB)"
    elif [ "${GPU_VRAM_MB:-0}" -gt 160000 ]; then
        CONCURRENCY_VAL=8
    elif [ "${GPU_VRAM_MB:-0}" -gt 80000 ]; then
        CONCURRENCY_VAL=6
    elif [ "${GPU_VRAM_MB:-0}" -gt 40000 ]; then
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

if [ ! -d "dataset" ] || [ -z "$(ls -A dataset 2>/dev/null)" ]; then
    warn "dataset not found — run without SKIP_PREPARE=true first"
fi

# ── Step 5: run benchmark ──────────────────────────────────────────────

FORCE_FLAG=""
if [ "${FORCE:-false}" = "true" ]; then FORCE_FLAG="--force"; fi

RUN_NAME="${SETTING}-lc-qwen3.6-27b-${SUBSET}"
header "${SETTING} benchmark: $RUN_NAME"

echo ""
log "Benchmark configuration"
log "  setting:     $SETTING"
log "  server:      $LLAMA_CPP_BASE_URL"
log "  model:       $LLAMA_CPP_MODEL"
log "  subset:      $SUBSET"
log "  concurrency: $CONCURRENCY_VAL"
log "  backend:     $BACKEND"
log "  agent-config: $AGENT_CONFIG"
log "  force:       ${FORCE:-false}"
echo ""

uv run cooperbench run \
    -n "$RUN_NAME" \
    --setting "$SETTING" \
    -s "$SUBSET" \
    -m "$LLAMA_CPP_MODEL" \
    -a llama_cpp \
    -c "$CONCURRENCY_VAL" \
    --backend "$BACKEND" \
    --agent-config "$AGENT_CONFIG" \
    ${FORCE_FLAG:-} \
    ${WANDB_ARGS:-}

log "${SETTING} benchmark complete → logs/$RUN_NAME/${SETTING}/"

# ── Summary ──

header "Benchmark complete"

LOG_DIR="logs/${RUN_NAME}/${SETTING}"
if [ -f "$LOG_DIR/summary.json" ]; then
    python3 -c "
import json
with open('$LOG_DIR/summary.json') as f:
    s = json.load(f)
print(f'[${SETTING}] {s.get(\"completed\",0)} completed, {s.get(\"failed\",0)} failed, {s.get(\"skipped\",0)} skipped')
e = s.get('eval', {})
if e:
    print(f'  eval: {e.get(\"passed\",0)} passed, {e.get(\"failed\",0)} failed ({e.get(\"pass_rate\",0)*100:.0f}%)')
" || true
fi
ENDOFSCRIPT
)

# ── Launch: single enroot invocation for the full workflow ─────────────

header "Launching enroot container"

enroot start --root --rw \
    -e HF_HOME=/workspace/.cache/hf \
    -e LLAMA_CACHE=/workspace/.cache/llama_cache \
    -e DOCKER_HOST=unix:///run/podman/podman.sock \
    -e MSWEA_DOCKER_EXECUTABLE=podman \
    -e LLAMA_SERVER_PATH="$LLAMA_SERVER_PATH" \
    -e LLAMA_SERVER_PORT="$LLAMA_SERVER_PORT" \
    -e LLAMA_SERVER_CTX="$LLAMA_SERVER_CTX" \
    -e LLAMA_CPP_BASE_URL="$LLAMA_CPP_BASE_URL" \
    -e LLAMA_CPP_API_KEY="$LLAMA_CPP_API_KEY" \
    -e LLAMA_CPP_MODEL="$LLAMA_CPP_MODEL" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e COOPERBENCH_DIR="$COOPERBENCH_DIR" \
    -e AGENT_CONFIG="$AGENT_CONFIG" \
    -e SETTING="$SETTING" \
    -e SUBSET="$SUBSET" \
    -e CONCURRENCY="$CONCURRENCY" \
    -e SKIP_PREPARE="$SKIP_PREPARE" \
    -e FORCE="$FORCE" \
    -e BACKEND="$BACKEND" \
    -e VRAM_HEADROOM_MB="$VRAM_HEADROOM_MB" \
    -e NO_AUTO_COMPACTION="$NO_AUTO_COMPACTION" \
    -e WANDB_ARGS="$WANDB_ARGS" \
    -e WANDB_PROJECT="$WANDB_PROJECT" \
    -e WANDB_ENTITY="$WANDB_ENTITY" \
    --mount "${WORKSPACE_SRC}:/workspace" \
    "$CONTAINER_NAME" \
    bash -c "$INNER_SCRIPT"

header "Job ${SLURM_JOB_ID:-?} ($SETTING) complete"
