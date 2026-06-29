#!/usr/bin/env bash
# prewarm_vllm_model.sh — Pre-download the vLLM model into the
# `vllm-cache` Docker volume so cb-vllm starts instantly.  Run this on
# the Vast.ai VM host (not inside a container).
#
# Why: vLLM serve downloads the model on first launch.  On a slow
# transpacific link from a Vast.ai Japan host to huggingface.co, that
# can take 30+ min with no progress and no way to speed it up.  This
# pre-download uses huggingface_hub + hf_transfer (multi-connection
# parallel) into the same `vllm-cache` volume cb-vllm reads from, so
# vllm's startup will see the model already cached and skip the
# download entirely.
#
# Safe to re-run: huggingface_hub skips already-downloaded files.
#
# Usage (on the Vast.ai VM host, NOT inside the cooperbench container):
#
#   # Pull the script and run it with defaults (model + image tag
#   # baked into this repo):
#   curl -fsSL https://raw.githubusercontent.com/laminair/CooperBench/main/scripts/prewarm_vllm_model.sh | bash
#
#   # Or override the model / image tag:
#   VLLM_MODEL=cyankiwi/Qwen3.6-27B-AWQ-INT4 \
#   IMAGE_TAG=ghcr.io/laminair/cooperbench-vastai:0.0.23 \
#   bash prewarm_vllm_model.sh
#
#   # Set HF_TOKEN if the model is gated:
#   HF_TOKEN=hf_xxxxx bash prewarm_vllm_model.sh
#
# After this completes, re-run the setup inside the cooperbench
# container — vllm will start in under a minute.

set -euo pipefail

VLLM_MODEL="${VLLM_MODEL:-cyankiwi/Qwen3.6-27B-AWQ-INT4}"
IMAGE_TAG="${IMAGE_TAG:-ghcr.io/laminair/cooperbench-vastai:0.0.23}"

if ! command -v docker &>/dev/null; then
    echo "docker CLI not found — run this on the VM host, not inside a container" >&2
    exit 1
fi

# Make sure the volume exists (cb-vllm will create it on first run,
# but we want it now so the pre-download has somewhere to land).
docker volume create vllm-cache &>/dev/null || true

echo "==> pre-downloading $VLLM_MODEL into the vllm-cache volume"
echo "    image:      $IMAGE_TAG"
echo "    destination: vllm-cache:/root/.cache/huggingface/hub"
echo
echo "(hf_transfer is installed on the fly and uses 8 parallel"
echo " connections — typically 3-5x faster than vllm's default"
echo " single-connection downloader)"
echo

# If the user passed HF_TOKEN, forward it; otherwise this is a no-op.
HF_TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_ARGS=(-e "HF_TOKEN=$HF_TOKEN")
fi

docker run --rm \
    -v vllm-cache:/root/.cache/huggingface \
    -e "HF_HOME=/root/.cache/huggingface" \
    -e "HF_HUB_ENABLE_HF_TRANSFER=1" \
    "${HF_TOKEN_ARGS[@]}" \
    "$IMAGE_TAG" \
    python3 - <<PYEOF
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
import time

# Install hf_transfer for multi-connection parallel downloads.
try:
    import hf_transfer  # noqa: F401
    print("hf_transfer already installed")
except ImportError:
    print("Installing hf_transfer...")
    import subprocess
    subprocess.check_call(["pip", "install", "--quiet", "hf_transfer"])
    import hf_transfer  # noqa: F401

# Re-import after possible install.
from huggingface_hub import snapshot_download

print()
print(f"Downloading {os.environ['VLLM_MODEL']} ...")
print(f"  to      {os.environ['HF_HOME']}/hub")
print(f"  workers 8 (parallel connections via hf_transfer)")
print()

t0 = time.time()
path = snapshot_download(
    repo_id=os.environ["VLLM_MODEL"],
    cache_dir=os.path.join(os.environ["HF_HOME"], "hub"),
    max_workers=8,
)
dt = time.time() - t0

print()
print(f"Done in {dt/60:.1f} min.  Model cached at:")
print(f"  {path}")
PYEOF

echo
echo "==> vllm-cache is now primed.  Re-run setup_vastai.sh inside"
echo "    the cooperbench container — vllm will start in <60s."
