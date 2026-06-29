#!/usr/bin/env bash
# setup_vastai_vm.sh — One-time VM host setup for CooperBench on Vast.ai.
#
# Run this once on the Vast.ai Virtual Machine (Ubuntu 22.04 VM template)
# BEFORE launching the cooperbench-vastai container.  It installs and
# configures the NVIDIA container toolkit on the host so 'docker run
# --gpus all' works.
#
# What it does:
#   1. Installs nvidia-container-toolkit (the host-side nvidia runtime).
#   2. Runs 'nvidia-ctk runtime configure --runtime=docker' to register
#      the nvidia runtime in /etc/docker/daemon.json.
#   3. Restarts dockerd.
#   4. Verifies GPU passthrough with a quick nvidia-smi under docker.
#
# After this completes, launch the cooperbench container:
#   docker run -d --name cooperbench \
#       -v /var/run/docker.sock:/var/run/docker.sock \
#       -v /workspace:/workspace \
#       --network host --gpus all --restart unless-stopped \
#       --entrypoint bash \
#       ghcr.io/laminair/cooperbench-vastai:0.0.21 -c 'sleep infinity'
#   docker exec -it cooperbench bash /opt/cooperbench/scripts/setup_vastai.sh
#
# Idempotent — safe to re-run.  Auto-elevates with sudo if not already root.

set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

if ! command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi not found — is this a GPU-enabled VM?  Aborting." >&2
    exit 1
fi

# ── 1. Install nvidia-container-toolkit ──────────────────────────────
if ! command -v nvidia-ctk &>/dev/null; then
    echo "Installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update
    apt-get install -y nvidia-container-toolkit
else
    echo "nvidia-container-toolkit already installed."
fi

# ── 2. Register the nvidia runtime in /etc/docker/daemon.json ─────────
echo "Configuring dockerd nvidia runtime..."
nvidia-ctk runtime configure --runtime=docker

# ── 3. Restart dockerd ───────────────────────────────────────────────
echo "Restarting dockerd..."
systemctl restart docker
sleep 2

# ── 4. Verify GPU passthrough ────────────────────────────────────────
echo "Verifying GPU passthrough..."
if ! docker run --rm --gpus all nvidia/cuda:13.2.0-base-ubuntu24.04 nvidia-smi; then
    echo "GPU passthrough verification failed." >&2
    echo "  - 'docker info' should show 'Runtimes: ... nvidia ...'" >&2
    echo "  - /etc/docker/daemon.json should include the nvidia runtime entry" >&2
    exit 1
fi

cat <<'EOF'

OK — host dockerd can now see the GPU.

Next, launch the cooperbench container (from the VM shell):

  docker run -d --name cooperbench \
      -v /var/run/docker.sock:/var/run/docker.sock \
      -v /workspace:/workspace \
      --network host --gpus all --restart unless-stopped \
      --entrypoint bash \
      ghcr.io/laminair/cooperbench-vastai:0.0.21 -c 'sleep infinity'

  docker exec -it cooperbench bash
  cd /opt/cooperbench
  bash scripts/setup_vastai.sh
EOF
