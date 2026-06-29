#!/usr/bin/env bash
# diagnose_vastai.sh — Pull together the state of the cooperbench
# container + its sidecars so you can see why setup_vastai.sh is stuck
# or what just broke.
#
# Safe to run any time.  Read-only — does not modify state.
#
# Usage:
#   bash scripts/diagnose_vastai.sh

set -uo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
hdr()  { echo -e "\n${CYAN}━━━ $* ━━━${NC}"; }
log()  { echo -e "${GREEN}[diag]${NC} $*"; }
warn() { echo -e "${YELLOW}[diag]${NC} $*"; }
err()  { echo -e "${RED}[diag]${NC} $*"; }

# ── Docker daemon & socket ───────────────────────────────────────────
hdr "Host docker"
if ! command -v docker &>/dev/null; then
    err "docker CLI not in PATH"
    exit 1
fi
if [ ! -S /var/run/docker.sock ]; then
    err "/var/run/docker.sock not present — are we inside the cooperbench"
    err "container with the socket mounted?  See docs/VASTAI.md §4."
else
    log "/var/run/docker.sock present"
fi
docker info 2>&1 | grep -E "Server Version|Storage Driver|Runtimes|NVIDIA" | head -10 || true

# ── cb-vllm state ────────────────────────────────────────────────────
hdr "cb-vllm container"
if ! docker inspect cb-vllm &>/dev/null; then
    err "cb-vllm container does not exist — setup_vastai.sh has not"
    err "created it yet, or it was removed manually."
else
    docker inspect cb-vllm --format \
        '  name={{.Name}} status={{.State.Status}} running={{.State.Running}} exit={{.State.ExitCode}} pid={{.State.Pid}}' 2>/dev/null
    if [ "$(docker inspect -f '{{.State.Running}}' cb-vllm)" = "true" ]; then
        log "cb-vllm is RUNNING"
    else
        warn "cb-vllm is NOT running (status above)"
        err "Recent docker logs (last 80 lines):"
        docker logs cb-vllm 2>&1 | tail -80 || true
    fi
fi

# ── vllm log file inside cb-vllm ────────────────────────────────────
hdr "vllm log file (inside cb-vllm)"
if docker exec cb-vllm true 2>/dev/null; then
    docker exec cb-vllm bash -c 'ls -la /tmp/vllm-*.log 2>/dev/null; echo; for f in /tmp/vllm-*.log; do [ -f "$f" ] && echo "==> $f <==" && tail -80 "$f"; done' 2>&1 | head -200
else
    err "could not exec into cb-vllm (container not running?)"
fi

# ── Port + readiness checks ─────────────────────────────────────────
hdr "vLLM endpoint"
VLLM_PORT="${VLLM_PORT:-8000}"
if curl -sS "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>&1 | head -50; then
    log "vllm endpoint responding"
else
    err "vllm endpoint http://127.0.0.1:${VLLM_PORT}/v1/models not responding"
fi

# ── cb-redis state ───────────────────────────────────────────────────
hdr "cb-redis container"
if ! docker inspect cb-redis &>/dev/null; then
    warn "cb-redis not created (setup_vastai.sh may have stopped before redis step)"
else
    docker inspect cb-redis --format \
        '  name={{.Name}} status={{.State.Status}} running={{.State.Running}}' 2>/dev/null
    docker exec cb-redis redis-cli -p 6379 ping 2>/dev/null \
        && log "redis PONG" \
        || err "redis ping failed"
fi

# ── GPU + CUDA sanity ───────────────────────────────────────────────
hdr "GPU + CUDA (from inside the outer container)"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>&1 | head -8
else
    warn "nvidia-smi not in PATH — no GPU access from this shell"
fi
if command -v nvcc &>/dev/null; then
    nvcc --version 2>&1 | tail -2
fi

# ── Recent cooperbench-vastai image ─────────────────────────────────
hdr "Image state"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}  created={{.CreatedSince}}' \
    ghcr.io/laminair/cooperbench-vastai 2>&1 | head -5

# ── HF model cache on the cb-vllm volume ─────────────────────────────
hdr "HF model cache (inside cb-vllm)"
docker exec cb-vllm bash -c 'du -sh /root/.cache/huggingface 2>/dev/null; ls -la /root/.cache/huggingface/hub/ 2>/dev/null | head -10' 2>/dev/null \
    || warn "(cb-vllm not running, or no HF cache yet)"

# ── Disk space ──────────────────────────────────────────────────────
hdr "Disk space"
df -h / 2>&1 | head -3

echo
log "diagnose done"
