#!/usr/bin/env bash
# slurm_smoke_test.sh — 30-minute SLURM smoke test for the Enroot+Podman+CooperBench stack.
#
# Runs exactly ONE task (go_chi_task / task 26, solo, 8 steps max).
# Expected wall time: 10-20 min.
#
# Submit with:
#   sbatch scripts/slurm_smoke_test.sh
#
# Override container/paths if your setup differs:
#   sbatch --export=CONTAINER_NAME=cooperbench_v1,MODEL_PATH=/workspace/... scripts/slurm_smoke_test.sh

# ── SLURM directives ────────────────────────────────────────────────────
#SBATCH --job-name=cb-smoke
#SBATCH --gpus=1
#SBATCH --time=01:15:00
#SBATCH --output=logs/slurm/%j_smoke.out

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()    { echo -e "${GREEN}[slurm]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[slurm]${NC} $(date '+%H:%M:%S') $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

CONTAINER_IMAGE="${CONTAINER_IMAGE:-cooperbench_v1.sqsh}"
CONTAINER_NAME="${CONTAINER_NAME:-cooperbench_v1}"
WORKSPACE_SRC="${WORKSPACE_SRC:-/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2}"
MODEL_PATH="${MODEL_PATH:-/workspace/.cache/llama_cache/Qwen_Qwen3.5-27B-Q4_K_M.gguf}"
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8050}"
LLAMA_SERVER_CTX="${LLAMA_SERVER_CTX:-65536}"
COOPERBENCH_DIR="${COOPERBENCH_DIR:-/workspace/CooperBench}"
AGENT_CONFIG="${AGENT_CONFIG:-}"

header "CooperBench SLURM smoke test"
log "job id:  ${SLURM_JOB_ID:-?}"
log "node:    ${SLURMD_NODENAME:-?}"
log "image:   $CONTAINER_NAME"

mkdir -p logs/slurm

if ! command -v enroot &>/dev/null; then
    err "enroot not found — module load enroot"
    exit 1
fi

# ── Prepare container image ────────────────────────────────────────────

if [ -f "$CONTAINER_IMAGE" ]; then
    if ! enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}\b"; then
        log "importing $CONTAINER_IMAGE ..."
        enroot create --name "$CONTAINER_NAME" "$CONTAINER_IMAGE"
    fi
elif ! enroot list 2>/dev/null | grep -q "^${CONTAINER_NAME}\b"; then
    err "container '$CONTAINER_NAME' not found and '$CONTAINER_IMAGE' does not exist"
    exit 1
fi

[ -d "$WORKSPACE_SRC" ] || { err "workspace '$WORKSPACE_SRC' does not exist"; exit 1; }

# ── Inner script (runs inside the container) ───────────────────────────

INNER=$(cat <<'ENDOFSCRIPT'
set -euo pipefail
# Resolve AGENT_CONFIG inside the container
[ -z "${AGENT_CONFIG:-}" ] && \
    AGENT_CONFIG="${COOPERBENCH_DIR}/src/cooperbench/agents/llama_cpp/config/enroot_hpc.yaml"
export AGENT_CONFIG

cd "$COOPERBENCH_DIR"
bash scripts/smoke_test.sh
ENDOFSCRIPT
)

# ── Launch ─────────────────────────────────────────────────────────────

header "Launching enroot container"

enroot start --root --rw \
    -e DOCKER_HOST=unix:///run/podman/podman.sock \
    -e MSWEA_DOCKER_EXECUTABLE=podman \
    -e HF_HOME=/workspace/.cache/hf \
    -e LLAMA_CACHE=/workspace/.cache/llama_cache \
    -e LLAMA_SERVER_PATH="$LLAMA_SERVER_PATH" \
    -e LLAMA_SERVER_PORT="$LLAMA_SERVER_PORT" \
    -e LLAMA_SERVER_CTX="$LLAMA_SERVER_CTX" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e COOPERBENCH_DIR="$COOPERBENCH_DIR" \
    -e AGENT_CONFIG="$AGENT_CONFIG" \
    -e LLAMA_CPP_API_KEY=local-llama-cpp \
    -e LLAMA_CPP_MODEL=openai/Qwen3.6-27B-Q4_K_M.gguf \
    --mount "${WORKSPACE_SRC}:/workspace" \
    "$CONTAINER_NAME" \
    bash -c "$INNER"

header "Smoke test job ${SLURM_JOB_ID:-?} complete"
