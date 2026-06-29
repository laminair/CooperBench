#!/usr/bin/env bash
# setup_vastai.sh — One-shot Vast.ai VM bootstrap for CooperBench.
#
# Assumes the Vast.ai template was provisioned with:
#   - Docker image:    ghcr.io/laminair/cooperbench-vastai:<tag>   (built from Dockerfile.vastai)
#   - Docker socket:   /var/run/docker.sock mounted from the host VM
#   - ≥ 80 GB disk
#   - GPU: any 24 GB+ NVIDIA (recommended: RTX PRO 6000 Blackwell for the 27B model)
#
# Steps:
#   1. Verify docker socket + NVIDIA driver + nvidia-container-toolkit
#   2. Pull our own image (idempotent — needed so vllm/redis can be re-launched)
#   3. Start Redis sidecar container (`cb-redis`, host network, port 6379)
#   4. Start vLLM sidecar container (`cb-vllm`, host network, --gpus all, port 8000)
#   5. Wait for vLLM to be reachable at http://localhost:8000/v1/models
#   6. Pre-pull all dataset Docker images for $SUBSET
#   7. Print next-step hints
#
# Idempotent — safe to re-run.  Each step no-ops if the resource already exists.
#
# Environment overrides:
#   IMAGE_TAG                 — Docker image to run (default: ghcr.io/laminair/cooperbench-vastai:0.0.21)
#   VLLM_MODEL                — vllm model id (default: cyankiwi/Qwen3.6-27B-AWQ-INT4)
#   VLLM_PORT                 — vllm port (default: 8000)
#   VLLM_MAX_MODEL_LEN        — context size (default: 65536)
#   SUBSET                    — dataset subset to pre-pull (default: flash_25)
#   SKIP_IMAGE_PULL           — skip pre-pull of dataset Docker images (default: false)
#   COOPERBENCH_DIR           — path to cooperbench source (default: /opt/cooperbench)

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
log()    { echo -e "${GREEN}[setup]${NC} $(date '+%H:%M:%S') $*"; }
warn()   { echo -e "${YELLOW}[setup]${NC} $(date '+%H:%M:%S') $*"; }
err()    { echo -e "${RED}[setup]${NC} $(date '+%H:%M:%S') $*"; }
header() { echo -e "\n${CYAN}━━━ $* ━━━${NC}\n"; }

IMAGE_TAG="${IMAGE_TAG:-ghcr.io/laminair/cooperbench-vastai:0.0.21}"
VLLM_MODEL="${VLLM_MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
SUBSET="${SUBSET:-flash_25}"
SKIP_IMAGE_PULL="${SKIP_IMAGE_PULL:-false}"
COOPERBENCH_DIR="${COOPERBENCH_DIR:-/opt/cooperbench}"

# ── Step 1: prerequisites ────────────────────────────────────────────

header "Verifying prerequisites"

if ! command -v docker &>/dev/null; then
    err "docker CLI not found in PATH.  The Vast.ai image should bundle it;"
    err "if you built your own from a different base, install docker.io."
    exit 1
fi
log "docker:        $(docker --version)"

if [ ! -S /var/run/docker.sock ]; then
    err "/var/run/docker.sock not present."
    err ""
    err "Vast.ai template must mount the host's docker socket into the outer"
    err "container (the 'Docker Socket' / 'Mount Host Docker Socket' option in"
    err "the Vast.ai launch dialog).  Without it CooperBench cannot launch"
    err "agent containers from inside the outer container."
    err ""
    err "Stop the instance, enable the docker-socket mount, and relaunch."
    exit 1
fi
log "docker socket: /var/run/docker.sock ✓"

if ! docker info --format '{{.Driver}}' 2>/dev/null | grep -qE 'overlay2|native'; then
    warn "docker info did not report a recognized storage driver — continuing anyway"
fi

if ! command -v nvidia-smi &>/dev/null; then
    err "nvidia-smi not found — the Vast.ai VM has no working NVIDIA driver."
    err "Make sure the template was created with GPU support enabled."
    exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true)
if [ -z "$GPU_INFO" ]; then
    err "nvidia-smi returned no GPUs."
    exit 1
fi
log "GPU:           $(echo "$GPU_INFO" | head -1 | tr -s ' ')"

if ! command -v nvidia-ctk &>/dev/null && ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    warn "nvidia-container-toolkit not detected — vllm container may fail with"
    warn "'could not select device capability'.  Rebake the image or install it."
fi

# ── Step 2: pull our own image ───────────────────────────────────────

header "Ensuring CooperBench Vast.ai image is present"

if ! docker image inspect "$IMAGE_TAG" &>/dev/null; then
    log "pulling $IMAGE_TAG ..."
    if ! docker pull "$IMAGE_TAG"; then
        err "failed to pull $IMAGE_TAG"
        err "  - is the tag published on ghcr.io/laminair/cooperbench-vastai?"
        err "  - override with IMAGE_TAG=ghcr.io/laminair/<other>:<tag> $0"
        exit 1
    fi
else
    log "$IMAGE_TAG already local, skipping pull"
fi

# ── Step 3: Redis sidecar ────────────────────────────────────────────

header "Starting Redis sidecar (cb-redis)"

if docker inspect cb-redis &>/dev/null; then
    if [ "$(docker inspect -f '{{.State.Running}}' cb-redis)" != "true" ]; then
        log "starting existing cb-redis container"
        docker start cb-redis
    else
        log "cb-redis already running"
    fi
else
    log "creating cb-redis container"
    docker run -d --name cb-redis \
        --network host \
        --restart unless-stopped \
        redis:7-alpine
fi

# Wait for redis
for i in $(seq 1 30); do
    if docker exec cb-redis redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
        log "cb-redis ready after ${i}s"
        break
    fi
    sleep 1
done
if ! docker exec cb-redis redis-cli -p 6379 ping 2>/dev/null | grep -q PONG; then
    err "cb-redis did not become ready in 30s"
    docker logs cb-redis 2>&1 | tail -20
    exit 1
fi

# ── Step 4: vLLM sidecar ─────────────────────────────────────────────

header "Starting vLLM sidecar (cb-vllm)"

if docker inspect cb-vllm &>/dev/null; then
    if [ "$(docker inspect -f '{{.State.Running}}' cb-redis)" != "true" ]; then
        log "starting existing cb-vllm container"
        docker start cb-vllm
    else
        log "cb-vllm already running"
    fi
else
    log "creating cb-vllm container (this downloads the model on first run)"
    docker run -d --name cb-vllm \
        --network host \
        --gpus all \
        --restart unless-stopped \
        -e HF_HOME=/root/.cache/huggingface \
        -e VLLM_PORT="$VLLM_PORT" \
        -e VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
        -e VLLM_MODEL="$VLLM_MODEL" \
        -v vllm-cache:/root/.cache/huggingface \
        "$IMAGE_TAG" \
        bash -c "source /etc/profile.d/claude.sh && /opt/cooperbench/scripts/serve_vllm.sh --bg"
fi

# ── Step 5: wait for vLLM ────────────────────────────────────────────

header "Waiting for vLLM (model: $VLLM_MODEL)"

for i in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        log "vllm ready after $((i * 2))s"
        break
    fi
    if [ "$i" -eq 180 ]; then
        err "vllm did not become ready in 6 minutes"
        err "tail of /tmp/vllm-${VLLM_PORT}.log from inside the container:"
        docker exec cb-vllm tail -40 "/tmp/vllm-${VLLM_PORT}.log" 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# ── Step 6: pre-pull dataset Docker images ───────────────────────────

if [ "${SKIP_IMAGE_PULL}" != "true" ]; then
    header "Pre-pulling Docker images for subset: $SUBSET"
    if [ ! -d "$COOPERBENCH_DIR/dataset" ] || [ -z "$(ls -A "$COOPERBENCH_DIR/dataset" 2>/dev/null)" ]; then
        warn "dataset not present at $COOPERBENCH_DIR/dataset — skipping pre-pull."
        warn "Run scripts/prepare_vastai_benchmark.sh first to download the dataset."
    elif [ ! -f "$COOPERBENCH_DIR/dataset/subsets/${SUBSET}.json" ]; then
        warn "subset file not found: $COOPERBENCH_DIR/dataset/subsets/${SUBSET}.json"
        warn "Skipping pre-pull.  Pass SUBSET= to choose a different one."
    else
        cd "$COOPERBENCH_DIR"
        IMAGES=$(python3 -c "
import json
with open('dataset/subsets/${SUBSET}.json') as f:
    data = json.load(f)
seen = set()
images = []
for entry in data.get('tasks', []):
    repo = entry['repo']
    tid  = entry['task_id']
    key = (repo, tid)
    if key in seen: continue
    seen.add(key)
    repo_clean = repo.replace('_task','').replace('_','-')
    images.append(f'akhatua/cooperbench-{repo_clean}:task{tid}')
print('\n'.join(images))
")
        IMAGES="$IMAGES"$'\n'"redis:alpine"
        TOTAL=$(echo "$IMAGES" | wc -l)
        COUNT=0
        FAILED=0
        while IFS= read -r img; do
            [ -z "$img" ] && continue
            COUNT=$((COUNT + 1))
            printf "  [%2d/%2d] %s ... " "$COUNT" "$TOTAL" "$img"
            if docker pull "$img" >/dev/null 2>&1; then
                echo "ok"
            else
                echo "FAILED"
                FAILED=$((FAILED + 1))
            fi
        done <<< "$IMAGES"
        if [ "$FAILED" -gt 0 ]; then
            warn "$FAILED/$TOTAL images failed to pull (network?) — benchmark will auto-pull on first use"
        else
            log "all $TOTAL Docker images pulled successfully"
        fi
    fi
fi

# ── Step 7: print next-step hints ────────────────────────────────────

header "Setup complete"
log "image:       $IMAGE_TAG"
log "vllm:        http://127.0.0.1:${VLLM_PORT}/v1/models"
log "vllm:        http://127.0.0.1:${VLLM_PORT}/v1/messages  (Anthropic)"
log "redis:       redis://localhost:6379  (cb-redis container)"
log "model id:    $VLLM_MODEL"
log "subset:      $SUBSET"
log ""
log "Next steps:"
log "  cd $COOPERBENCH_DIR"
log "  ./scripts/launch_vastai_benchmark.sh --solo-only   # or --coop-only"
log ""
log "Troubleshooting:"
log "  docker logs cb-vllm     # vllm stdout/stderr"
log "  docker logs cb-redis    # redis stdout/stderr"
log "  ./scripts/serve_vllm.sh --kill && docker start cb-vllm   # restart vllm"
