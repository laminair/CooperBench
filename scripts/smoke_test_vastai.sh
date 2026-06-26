#!/usr/bin/env bash
# smoke_test_vastai.sh — Single-task end-to-end sanity check on Vast.ai.
#
# Runs one solo + one coop task (default: dottxt_ai_outlines_task / 1655 / f7_f10)
# against the vLLM-served local model and the `claude_code` agent.  Caps
# claude-code at 8 turns so the test finishes in ~10-20 minutes.
#
# Usage:
#   ./scripts/smoke_test_vastai.sh [--repo REPO] [--task ID] [--features F1,F2]
#                                  [--solo-only] [--coop-only]
#                                  [--base-url URL]
#
# Environment variables:
#   VLLM_BASE_URL   — vllm endpoint (default: http://localhost:8000/v1)
#   VLLM_API_KEY    — vllm auth token (default: dummy)
#   VLLM_MODEL      — served model name (auto-discovered if unset)

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[smoke]${NC} $*"; }
warn() { echo -e "${YELLOW}[smoke]${NC} $*"; }
err()  { echo -e "${RED}[smoke]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-dummy}"
VLLM_MODEL="${VLLM_MODEL:-}"
REPO="dottxt_ai_outlines_task"
TASK_ID="1655"
FEATURES="7,10"
RUN_SOLO=true
RUN_COOP=true
BASE_URL_OVERRIDE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --task) TASK_ID="$2"; shift 2 ;;
        --features) FEATURES="$2"; shift 2 ;;
        --solo-only) RUN_COOP=false; shift ;;
        --coop-only) RUN_SOLO=false; shift ;;
        --base-url) BASE_URL_OVERRIDE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

HOST_BASE_URL="${BASE_URL_OVERRIDE:-${VLLM_BASE_URL}}"
if [[ "$HOST_BASE_URL" == *"localhost"* || "$HOST_BASE_URL" == *"127.0.0.1"* ]]; then
    HOST_BASE_URL=$(echo "$HOST_BASE_URL" | sed -E 's#(localhost|127\.0\.0\.1)#host.docker.internal#')
fi

# ── Sanity checks ───────────────────────────────────────────────────
if ! curl -s "${HOST_BASE_URL%/v1}/v1/models" >/dev/null 2>&1; then
    err "vLLM not reachable at ${HOST_BASE_URL%/v1}/v1/models"
    err "Run scripts/setup_vastai.sh first."
    exit 1
fi
if [ -z "$VLLM_MODEL" ]; then
    VLLM_MODEL=$(curl -s "${HOST_BASE_URL%/v1}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null || echo "?")
fi
if [ ! -d "dataset" ] || [ -z "$(ls -A dataset 2>/dev/null)" ]; then
    err "dataset not found.  Run scripts/prepare_vastai_benchmark.sh first."
    exit 1
fi

SMOKE_TURNS="${SMOKE_TURNS:-8}"

# Inline agent-config override that caps max_turns so the test finishes quickly.
# The agent may not solve the task in 8 turns — that's expected; we just want
# to confirm the whole pipeline works end-to-end.
SMOKE_CONFIG="$(mktemp /tmp/cb_smoke_XXXX.yaml)"
cat > "$SMOKE_CONFIG" <<YAML
agent:
  max_turns: $SMOKE_TURNS
YAML

header "CooperBench Vast.ai smoke test"
log "model:    $VLLM_MODEL"
log "repo:     $REPO"
log "task:     $TASK_ID"
log "features: $FEATURES"
log "server:   $HOST_BASE_URL"
log "turns:    $SMOKE_TURNS (cap, smoke only)"

# ── Solo ─────────────────────────────────────────────────────────────
if $RUN_SOLO; then
    header "Solo: $REPO/$TASK_ID [features $FEATURES]"
    RUN_NAME="smoke-solo-cc-$REPO-$TASK_ID"
    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -r "$REPO" \
        -t "$TASK_ID" \
        -f "$FEATURES" \
        -m "$VLLM_MODEL" \
        -a claude_code \
        --backend docker \
        --base-url "$HOST_BASE_URL" \
        --auth-token "$VLLM_API_KEY" \
        --agent-config "$SMOKE_CONFIG" \
        --no-auto-eval \
        --force

    FEAT_DIR=$(echo "$FEATURES" | tr ',' '_' | sed 's/\([0-9]*\)_\([0-9]*\)/f\1_f\2/')
    SOLO_DIR="logs/$RUN_NAME/solo/$REPO/$TASK_ID/$FEAT_DIR"
    if [ -f "$SOLO_DIR/result.json" ]; then
        echo ""
        log "Solo result:"
        uv run python -c "
import json
with open('$SOLO_DIR/result.json') as f:
    r = json.load(f)
agent = r.get('result', r.get('agent', {}))
print(f\"  status: {agent.get('status', '?')}\")
print(f\"  steps:  {agent.get('steps', 0)}\")
print(f\"  lines:  {len((agent.get('patch','') or '').splitlines())}\")
"
    else
        warn "No result.json under $SOLO_DIR"
    fi
fi

# ── Coop ─────────────────────────────────────────────────────────────
if $RUN_COOP; then
    header "Coop: $REPO/$TASK_ID [features $FEATURES]"
    RUN_NAME="smoke-coop-cc-$REPO-$TASK_ID"
    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -r "$REPO" \
        -t "$TASK_ID" \
        -f "$FEATURES" \
        -m "$VLLM_MODEL" \
        -a claude_code \
        --backend docker \
        --base-url "$HOST_BASE_URL" \
        --auth-token "$VLLM_API_KEY" \
        --agent-config "$SMOKE_CONFIG" \
        --no-auto-eval \
        --force

    FEAT_DIR=$(echo "$FEATURES" | tr ',' '_' | sed 's/\([0-9]*\)_\([0-9]*\)/f\1_f\2/')
    COOP_DIR="logs/$RUN_NAME/coop/$REPO/$TASK_ID/$FEAT_DIR"
    if [ -f "$COOP_DIR/result.json" ]; then
        echo ""
        log "Coop result:"
        uv run python -c "
import json
with open('$COOP_DIR/result.json') as f:
    r = json.load(f)
agents = r.get('results', r.get('agent_states', {}))
for aid, a in agents.items():
    print(f'  [{aid}] status: {a.get(\"status\", \"?\")}  steps: {a.get(\"steps\", 0)}')
"
    else
        warn "No result.json under $COOP_DIR"
    fi
fi

rm -f "$SMOKE_CONFIG"

echo ""
log "Smoke test complete. Check logs/ for detailed output."
