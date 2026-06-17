#!/usr/bin/env bash
# smoke_test.sh — quick end-to-end sanity check for the Enroot+Podman+CooperBench stack.
#
# Runs exactly ONE task (go_chi_task / task 26, solo, 8 agent steps max).
# Expected wall time: ~10-20 min (dominated by llama-server cold start).
#
# Usage (inside an enroot container):
#   bash $COOPERBENCH_DIR/scripts/smoke_test.sh
#
# Environment variables (all optional):
#   COOPERBENCH_DIR     — path to the CooperBench repo (default: /workspace/CooperBench)
#   LLAMA_SERVER_PATH   — path to llama-server binary
#   MODEL_PATH          — path to .gguf model file
#   LLAMA_SERVER_PORT   — port (default: 8050)
#   LLAMA_SERVER_CTX    — context size (default: 65536)
#   SKIP_LLAMA_SERVER   — set to "true" if llama-server is already running
#   SKIP_PODMAN         — set to "true" if Podman socket is already up

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo -e "${GREEN}[smoke]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[smoke]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[smoke]${NC} $(date '+%H:%M:%S') $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }
pass()   { echo -e "\n${BOLD}${GREEN}✔  SMOKE TEST PASSED${NC}\n"; }
fail()   { echo -e "\n${BOLD}${RED}✘  SMOKE TEST FAILED${NC}\n"; }

COOPERBENCH_DIR="${COOPERBENCH_DIR:-/workspace/CooperBench}"
MODEL_PATH="${MODEL_PATH:-/workspace/.cache/llama_cache/Qwen_Qwen3.5-27B-Q4_K_M.gguf}"
LLAMA_SERVER_PATH="${LLAMA_SERVER_PATH:-}"
LLAMA_SERVER_PORT="${LLAMA_SERVER_PORT:-8050}"
LLAMA_SERVER_CTX="${LLAMA_SERVER_CTX:-65536}"
LLAMA_CPP_API_KEY="${LLAMA_CPP_API_KEY:-local-llama-cpp}"
LLAMA_CPP_BASE_URL="http://localhost:${LLAMA_SERVER_PORT}/v1"
LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-openai/Qwen3.6-27B-Q4_K_M.gguf}"
SKIP_LLAMA_SERVER="${SKIP_LLAMA_SERVER:-false}"
SKIP_PODMAN="${SKIP_PODMAN:-false}"

SMOKE_REPO="go_chi_task"
SMOKE_TASK_ID="26"
SMOKE_RUN_NAME="smoke-test-solo"
AGENT_CONFIG="${AGENT_CONFIG:-${COOPERBENCH_DIR}/src/cooperbench/agents/llama_cpp/config/enroot_hpc.yaml}"

LLAMA_PID=""
PODMAN_PID=""
OVERALL_EXIT=0

cleanup() {
    local ec=$?
    [ -n "$LLAMA_PID" ] && { kill "$LLAMA_PID" 2>/dev/null || true; log "stopped llama-server"; }
    [ -n "$PODMAN_PID" ] && { kill "$PODMAN_PID" 2>/dev/null || true; log "stopped Podman service"; }
    if [ $ec -ne 0 ] || [ $OVERALL_EXIT -ne 0 ]; then fail; else pass; fi
    exit $ec
}
trap cleanup EXIT

cd "$COOPERBENCH_DIR"

header "CooperBench smoke test"
log "repo:    $SMOKE_REPO"
log "task:    $SMOKE_TASK_ID"
log "setting: solo (8 steps max)"
log "agent-config: $AGENT_CONFIG"

# ── Step 1: Podman socket ──────────────────────────────────────────────

if [ "${SKIP_PODMAN:-false}" != "true" ]; then
    header "Starting rootful Podman socket"

    mkdir -p /run/podman
    nohup podman system service -t 0 unix:///run/podman/podman.sock \
        > /tmp/podman-service.log 2>&1 &
    PODMAN_PID=$!
    export DOCKER_HOST=unix:///run/podman/podman.sock
    export MSWEA_DOCKER_EXECUTABLE=podman
    log "Podman service pid: $PODMAN_PID"

    for i in $(seq 1 20); do
        [ -S /run/podman/podman.sock ] && { log "socket ready after ${i}s"; break; }
        sleep 1
    done
    [ -S /run/podman/podman.sock ] || { err "Podman socket not available after 20s"; cat /tmp/podman-service.log; exit 1; }

    if ! podman info --log-level=error >/dev/null 2>&1; then
        err "podman info failed — check /tmp/podman-service.log"
        cat /tmp/podman-service.log
        exit 1
    fi
    log "Podman OK"
else
    log "skipping Podman startup (SKIP_PODMAN=true)"
    export DOCKER_HOST="${DOCKER_HOST:-unix:///run/podman/podman.sock}"
    export MSWEA_DOCKER_EXECUTABLE="${MSWEA_DOCKER_EXECUTABLE:-podman}"
fi

# Quick Docker SDK check
log "checking Docker SDK connectivity..."
python3 -c "
import docker, sys
try:
    c = docker.from_env()
    c.ping()
    print('Docker SDK: connected')
except Exception as e:
    print(f'Docker SDK: FAILED — {e}', file=sys.stderr)
    sys.exit(1)
"

# ── Step 2: llama-server ───────────────────────────────────────────────

if [ "${SKIP_LLAMA_SERVER:-false}" != "true" ]; then
    header "Starting llama-server"

    if [ -z "${LLAMA_SERVER_PATH:-}" ]; then
        for candidate in \
            /workspace/llama.cpp/build/bin/llama-server \
            /usr/local/bin/llama-server \
            /usr/bin/llama-server; do
            [ -x "$candidate" ] && { LLAMA_SERVER_PATH="$candidate"; break; }
        done
        [ -z "${LLAMA_SERVER_PATH:-}" ] && LLAMA_SERVER_PATH="$(which llama-server 2>/dev/null || true)"
    fi

    [ -z "${LLAMA_SERVER_PATH:-}" ] || [ ! -x "$LLAMA_SERVER_PATH" ] && {
        err "llama-server not found — set LLAMA_SERVER_PATH"
        exit 1
    }
    [ -f "$MODEL_PATH" ] || { err "model not found: $MODEL_PATH"; exit 1; }

    log "binary: $LLAMA_SERVER_PATH"
    log "model:  $MODEL_PATH"

    nohup "$LLAMA_SERVER_PATH" \
        --model "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$LLAMA_SERVER_PORT" \
        --ctx-size "$LLAMA_SERVER_CTX" \
        --n-gpu-layers 99 \
        --api-key "$LLAMA_CPP_API_KEY" \
        --flash-attn auto \
        --jinja \
        > /tmp/llama-server-smoke.log 2>&1 &
    LLAMA_PID=$!
    log "llama-server pid: $LLAMA_PID"

    READY=false
    for i in $(seq 1 60); do
        curl -sf -H "Authorization: Bearer $LLAMA_CPP_API_KEY" \
            "http://localhost:${LLAMA_SERVER_PORT}/v1/models" >/dev/null 2>&1 && {
            READY=true; log "llama-server ready after $((i * 2))s"; break
        }
        sleep 2
    done
    $READY || {
        err "llama-server failed to start after 120s"
        tail -20 /tmp/llama-server-smoke.log
        exit 1
    }

    # Quick inference check
    log "testing inference..."
    uv run python -c "
import litellm, os
os.environ['LITELLM_LOG'] = 'ERROR'
resp = litellm.completion(
    model='$LLAMA_CPP_MODEL',
    messages=[{'role': 'user', 'content': 'Reply with just the word PONG'}],
    api_base='$LLAMA_CPP_BASE_URL',
    api_key='$LLAMA_CPP_API_KEY',
    max_tokens=10,
)
print('inference:', resp.choices[0].message.content.strip())
"
else
    log "skipping llama-server startup (SKIP_LLAMA_SERVER=true)"
fi

# ── Step 3: write smoke override config ───────────────────────────────

header "Preparing smoke test config"

SMOKE_CONFIG=$(mktemp /tmp/smoke_override_XXXX.yaml)
cat > "$SMOKE_CONFIG" <<'YAML'
# Smoke test overrides: cap at 8 steps so the test finishes quickly.
# The agent won't solve the task fully, but we confirm the full chain works.
agent:
  step_limit: 8
  cost_limit: 999
YAML

# Merge enroot_hpc overrides first, then smoke overrides on top
# (enroot_hpc sets run_args; smoke sets step_limit)
MERGED_CONFIG=$(mktemp /tmp/smoke_merged_XXXX.yaml)
python3 - "$AGENT_CONFIG" "$SMOKE_CONFIG" "$MERGED_CONFIG" <<'PY'
import sys, yaml
base = {}
for src in sys.argv[1:3]:
    try:
        with open(src) as f:
            d = yaml.safe_load(f) or {}
        def merge(a, b):
            r = dict(a)
            for k, v in b.items():
                r[k] = merge(a[k], v) if k in a and isinstance(a[k], dict) and isinstance(v, dict) else v
            return r
        base = merge(base, d)
    except FileNotFoundError:
        pass
with open(sys.argv[3], 'w') as f:
    yaml.dump(base, f)
PY

log "smoke config: $MERGED_CONFIG"
log "contents:"
cat "$MERGED_CONFIG"

# ── Step 4: run one task ───────────────────────────────────────────────

header "Running smoke task ($SMOKE_REPO / task $SMOKE_TASK_ID)"

START_TS=$(date +%s)

uv run cooperbench run \
    -n "$SMOKE_RUN_NAME" \
    --setting solo \
    -r "$SMOKE_REPO" \
    -t "$SMOKE_TASK_ID" \
    -m "$LLAMA_CPP_MODEL" \
    -a llama_cpp \
    -c 1 \
    --backend docker \
    --agent-config "$MERGED_CONFIG" \
    --no-auto-eval \
    --force

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))

# ── Step 5: check result ───────────────────────────────────────────────

header "Checking result"

LOG_DIR="logs/${SMOKE_RUN_NAME}/solo"
RESULT_FILE=$(find "$LOG_DIR" -name "result.json" 2>/dev/null | head -1)

if [ -z "$RESULT_FILE" ]; then
    err "No result.json found under $LOG_DIR — cooperbench may have crashed"
    OVERALL_EXIT=1
else
    STATUS=$(python3 -c "import json; d=json.load(open('$RESULT_FILE')); print(d.get('status','?'))" 2>/dev/null || echo "?")
    log "Task status: $STATUS  (wall time: ${ELAPSED}s)"

    if [ "$STATUS" = "Error" ]; then
        err "Task ended with status=Error — check $LOG_DIR for details"
        python3 -c "import json; d=json.load(open('$RESULT_FILE')); print('error:', d.get('error',''))" 2>/dev/null || true
        OVERALL_EXIT=1
    else
        log "Task completed without Error status (status=$STATUS) — pipeline is functional"
        log "The agent only had 8 steps so it likely didn't solve the task — that's expected."
    fi
fi

log "Smoke test wall time: ${ELAPSED}s"
rm -f "$SMOKE_CONFIG" "$MERGED_CONFIG"
