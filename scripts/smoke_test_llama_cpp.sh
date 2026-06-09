#!/usr/bin/env bash
# smoke_test_llama_cpp.sh — Single-task smoke test for llama.cpp + CooperBench.
#
# Usage:
#   ./scripts/smoke_test_llama_cpp.sh [OPTIONS]
#
# Options:
#   --repo REPO        Repository to test (default: dottxt_ai_outlines_task)
#   --task ID          Task ID to test (default: 1655)
#   --features F1,F2   Feature pair (default: 7,10)
#   --solo-only        Run only solo mode
#   --coop-only        Run only coop mode
#
# Environment variables:
#   LLAMA_CPP_BASE_URL  — llama-server URL (default: http://localhost:8050/v1)
#   LLAMA_CPP_API_KEY   — API key (default: local-llama-cpp)
#   LLAMA_CPP_MODEL     — Model name (default: openai/Qwen3.6-27B-Q4_K_M.gguf)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# ── Defaults ──────────────────────────────────────────────────────────

LLAMA_CPP_BASE_URL="${LLAMA_CPP_BASE_URL:-http://localhost:8050/v1}"
LLAMA_CPP_API_KEY="${LLAMA_CPP_API_KEY:-local-llama-cpp}"
LLAMA_CPP_MODEL="${LLAMA_CPP_MODEL:-openai/Qwen3.6-27B-Q4_K_M.gguf}"
REPO="dottxt_ai_outlines_task"
TASK_ID="1655"
FEATURES="7,10"
RUN_SOLO=true
RUN_COOP=true

# ── Parse args ────────────────────────────────────────────────────────

while [ $# -gt 0 ]; do
    case "$1" in
        --repo) REPO="$2"; shift 2 ;;
        --task) TASK_ID="$2"; shift 2 ;;
        --features) FEATURES="$2"; shift 2 ;;
        --solo-only) RUN_COOP=false; shift ;;
        --coop-only) RUN_SOLO=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Colors ────────────────────────────────────────────────────────────

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[smoke]${NC} $*"; }
warn() { echo -e "${YELLOW}[smoke]${NC} $*"; }
err()  { echo -e "${RED}[smoke]${NC} $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

cd "$PROJECT_DIR"

# ── Pre-flight checks ─────────────────────────────────────────────────

if ! curl -s -H "Authorization: Bearer $LLAMA_CPP_API_KEY" "$LLAMA_CPP_BASE_URL/models" >/dev/null 2>&1; then
    err "llama-server not reachable at $LLAMA_CPP_BASE_URL"
    exit 1
fi

if [ ! -d "$PROJECT_DIR/dataset" ] || [ -z "$(ls -A "$PROJECT_DIR/dataset" 2>/dev/null)" ]; then
    err "dataset not found. Run ./scripts/prepare_llama_cpp_benchmark.sh first."
    exit 1
fi

# ── Step 1: Raw litellm connectivity test ─────────────────────────────

header "1. LiteLLM connectivity test"

log "testing model: $LLAMA_CPP_MODEL at $LLAMA_CPP_BASE_URL"
if uv run python -c "
from dotenv import load_dotenv; load_dotenv()
import litellm, os
litellm.suppress_debug_info = True
os.environ['LITELLM_LOG'] = 'ERROR'
resp = litellm.completion(
    model='$LLAMA_CPP_MODEL',
    messages=[{'role': 'user', 'content': 'Say hello in exactly one sentence.'}],
    api_base='$LLAMA_CPP_BASE_URL',
    api_key='$LLAMA_CPP_API_KEY',
)
content = resp.choices[0].message.content
print(f'Response: {content}')
print(f'Tokens:   {resp.usage.prompt_tokens} in / {resp.usage.completion_tokens} out')
" 2>&1; then
    log "LiteLLM connectivity: PASSED"
else
    err "LiteLLM connectivity: FAILED"
    exit 1
fi

# ── Step 2: Solo task ─────────────────────────────────────────────────

if $RUN_SOLO; then
    header "2. Solo task: $REPO/$TASK_ID [features $FEATURES]"

    RUN_NAME="smoke-solo-lc-$REPO-$TASK_ID"
    log "running solo task..."

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting solo \
        -r "$REPO" \
        -t "$TASK_ID" \
        -f "$FEATURES" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        --backend docker \
        --force

    # Check result
    FEAT_DIR=$(echo "$FEATURES" | tr ',' '_' | sed 's/\([0-9]*\)_\([0-9]*\)/f\1_f\2/')
    SOLO_DIR="logs/$RUN_NAME/solo/$REPO/$TASK_ID/$FEAT_DIR"
    if [ -f "$SOLO_DIR"/result.json ]; then
        echo ""
        log "Solo result:"
        uv run python -c "
import json
with open('$SOLO_DIR/result.json') as f:
    r = json.load(f)
agent = r.get('result', r.get('agent', {}))
print(f\"  status: {agent.get('status', '?')}\")
print(f\"  steps:  {agent.get('steps', 0)}\")
print(f\"  lines:  {len(agent.get('patch', '').splitlines())}\")
"
    fi
fi

# ── Step 3: Coop task ─────────────────────────────────────────────────

if $RUN_COOP; then
    header "3. Coop task: $REPO/$TASK_ID [features $FEATURES]"

    RUN_NAME="smoke-coop-lc-$REPO-$TASK_ID"
    log "running coop task..."

    uv run cooperbench run \
        -n "$RUN_NAME" \
        --setting coop \
        -r "$REPO" \
        -t "$TASK_ID" \
        -f "$FEATURES" \
        -m "$LLAMA_CPP_MODEL" \
        -a llama_cpp \
        --backend docker \
        --force

    # Check result
    FEAT_DIR=$(echo "$FEATURES" | tr ',' '_' | sed 's/\([0-9]*\)_\([0-9]*\)/f\1_f\2/')
    COOP_DIR="logs/$RUN_NAME/coop/$REPO/$TASK_ID/$FEAT_DIR"
    if [ -f "$COOP_DIR"/result.json ]; then
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
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────

echo ""
log "Smoke test complete. Check logs/ for detailed output."
log "Log directories:"
if $RUN_SOLO; then echo "  logs/$RUN_NAME/solo/"; fi
if $RUN_COOP; then echo "  logs/$RUN_NAME/coop/"; fi
echo ""
