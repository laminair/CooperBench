# CooperBench on Vast.ai

Run CooperBench on rented Vast.ai GPUs against a local model served by
[vLLM](https://docs.vllm.ai/).  The `claude_code` agent (the official
Anthropic CLI) talks to vLLM's native Anthropic `/v1/messages` endpoint
— no translation proxy, no LiteLLM shim, no per-task rewriting.

This document replaces the previous enroot / SLURM / rootful-Podman flow
that CooperBench used on HPC clusters.  The current path is:

```
Vast.ai VM  ─►  outer container (our image)  ─►  cb-vllm  (sidecar)
                                              └►  cb-redis (sidecar)
                                              └►  cooperbench (process)
                                                     └►  agent container N
                                                          └►  claude-code CLI
                                                                └►  host.docker.internal:8000/v1/messages
```

The cooperbench-vastai image runs as a container **inside a Vast.ai Virtual
Machine (full Linux VM, not a plain Docker instance)**, with the VM's
own `/var/run/docker.sock` bind-mounted into it.  That gives the
cooperbench process inside our image a Docker daemon it can drive to
launch the `cb-vllm` and `cb-redis` sidecars and the per-task agent
containers.

> **Why a VM, not a plain Vast.ai instance?**  The standard Vast.ai
> template editor's "Docker Options" field only accepts ports and
> environment variables — volume mounts are filtered out.  The VM
> template bypasses that and gives you a real Linux host with Docker
> pre-installed, so a normal `docker run -v /var/run/docker.sock:…`
> works.  Vast.ai calls this "nested containerization" and it's the
> only first-class way to get a host Docker socket on the platform.

---

## 1. Build the image

The image is built on your workstation (or any CUDA 13.x host) and
pushed to GitHub Container Registry (ghcr.io):

```bash
# From the repo root.
docker build -f Dockerfile.vastai -t ghcr.io/laminair/cooperbench-vastai:0.0.23 .
docker push  ghcr.io/laminair/cooperbench-vastai:0.0.23
```

The default baked-in model is `cyankiwi/Qwen3.6-27B-AWQ-INT4`
(AWQ INT4 quantization of the 27B Qwen — see `OPTIMIZATION_SUMMARY.md`
for the tuning decisions that drove that choice).  Override the default
at build time:

```bash
docker build -f Dockerfile.vastai \
  --build-arg VLLM_VERSION=0.17.1 \
  --build-arg CLAUDE_CODE_VERSION=2.1.0 \
  -t ghcr.io/laminair/cooperbench-vastai:0.0.23 .
```

If vLLM does not yet publish a cu132 wheel, edit `Dockerfile.vastai` to
base on `FROM vllm/vllm-openai:v0.17.1` instead of the nvidia/cuda base
and remove the `RUN pip install vllm` line.  The vLLM project's image
already bundles the right CUDA + PyTorch; the 13.2 host driver remains
forward-compatible.

---

## 2. Provision a Vast.ai Virtual Machine

CooperBench needs `docker.sock` inside the container, so we can't use a
plain Vast.ai Docker instance.  Use a **VM template** instead — the
recommended one is **Ubuntu 22.04 VM** (it ships with Docker
pre-installed).  Search → filter → rent.

| Field | Value |
|---|---|
| **Template** | `Ubuntu 22.04 VM` (or any `vastai/kvm:…` template) |
| **Disk** | ≥ 80 GB (model weights + dataset + pulled images) |
| **GPU** | Any 24 GB+ NVIDIA; RTX PRO 6000 Blackwell (96 GB), 2× RTX 5090 (64 GB), or 4× RTX 5090 (128 GB) recommended for the 27B model |
| **Extra Filters** | `vms_enabled=true` (already set by the VM templates) |

When the VM reports running, get its SSH URL from the instance panel
and `ssh` in.  The `ubuntu` user has passwordless `sudo`.

> If you don't see any VM-capable machines on the search page, click
> "Edit" on the Ubuntu 22.04 VM template and confirm the **Extra
> Filters** field contains `vms_enabled=true` (the official template
> does, but a saved copy may not).

---

## 3. Install the NVIDIA runtime on the VM host

Vast.ai's Ubuntu 22.04 VM image includes Docker, but **not** the
NVIDIA container toolkit on the host side.  Without it, `docker run
--gpus all` fails with `could not select device driver "" with
capabilities: [[gpu]]` — the host dockerd doesn't know which runtime
to use for GPU passthrough.

Run this once on the VM (it auto-elevates with sudo if needed):

```bash
curl -fsSL https://raw.githubusercontent.com/laminair/CooperBench/main/scripts/setup_vastai_vm.sh | bash
```

…or copy `scripts/setup_vastai_vm.sh` from this repo to the VM (the
file is in the cooperbench source on the VM host if you checked it
out, or download via the URL above) and `bash` it directly.  The
script is idempotent: it installs `nvidia-container-toolkit`, runs
`nvidia-ctk runtime configure --runtime=docker`, restarts dockerd,
and verifies with a `docker run --gpus all nvidia-smi`.

> If you skip this step you'll get the `could not select device
> driver` error below.  Our container image has the toolkit
> installed *inside* itself, but the **host** dockerd needs its own
> config.

---

## 4. Launch cooperbench inside the VM

The Ubuntu 22.04 VM is just a host — we still need to start the
cooperbench-vastai container on it, with the VM's docker socket
bind-mounted in.  One command from the VM's shell:

```bash
docker run -d --name cooperbench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /workspace:/workspace \
    --network host \
    --gpus all \
    --restart unless-stopped \
    --entrypoint bash \
    ghcr.io/laminair/cooperbench-vastai:0.0.23 \
    -c 'sleep infinity'
```

> The `sleep infinity` is a placeholder so the container stays up; the
> real work happens via `docker exec`.  `--network host` + `--gpus all`
> let `cb-vllm` see the GPU and let agent containers reach the host
> vLLM via `host.docker.internal`.

Open a shell in the container and run the setup:

```bash
docker exec -it cooperbench bash
cd /opt/cooperbench
bash scripts/setup_vastai.sh
```

`setup_vastai.sh` is **idempotent**.  It:

1. Verifies the docker socket is mounted and the NVIDIA driver + toolkit work.
2. Pulls our own image (so `cb-vllm` can be re-launched as a copy).
3. Starts `cb-redis` (Redis 7, host network, port 6379).
4. **Pre-downloads the vLLM model** into the `vllm-cache` Docker
   volume using `huggingface_hub` + `hf_transfer` (8 parallel
   connections).  vllm's own single-connection downloader can take
   30+ min on a transpacific link from Vast.ai Japan to HF Hub with
   no progress; hf_transfer typically finishes the same download in
   5–10 min and shows progress.  Skipped if the model is already
   cached.
5. Starts `cb-vllm` (vLLM serving the 27B AWQ model, host network, `--gpus all`, port 8000).
6. Waits for vLLM to respond at `http://localhost:8000/v1/models` —
   the model is already in `vllm-cache` so vllm skips its own
   download and just loads weights + warms CUDA.  Typically <2 min.
   (If you bypass the pre-download and let vllm do it, the wait is
   up to **15 minutes**.)
7. Pre-pulls the `akhatua/cooperbench-*` task images for `flash_25`.

> **Re-running just the pre-download** (e.g. on a different model or
> to retry after a network blip): from the VM host, run
> `bash scripts/prewarm_vllm_model.sh`.  It's safe to re-run;
> `huggingface_hub` skips already-downloaded files.

Then a one-time prep:

```bash
cd /opt/cooperbench
bash scripts/prepare_vastai_benchmark.sh
```

This installs the Python dependencies, downloads the CooperBench dataset
from HuggingFace, and sends a 32-token sanity ping through the
Anthropic `/v1/messages` endpoint.

---

## 5. Run the benchmark

```bash
cd /opt/cooperbench

# Auto-detect concurrency from VRAM, run solo + coop
bash scripts/launch_vastai_benchmark.sh

# Or pin subset / concurrency / mode
bash scripts/launch_vastai_benchmark.sh --coop-only --concurrency 4 --subset flash_25

# Or invoke cooperbench directly (same flow with finer control)
uv run cooperbench run \
    -n my-vastai-run \
    --setting coop \
    -s flash_25 \
    -m qwen3.6-27b-awq-int4 \
    -a claude_code \
    -c 4 \
    --backend docker \
    --base-url http://host.docker.internal:8000 \
    --auth-token dummy
```

The `claude_code` adapter (no extra config needed):

- Reads `--base-url` and `--auth-token` from the CLI flags.
- Forwards `ANTHROPIC_BASE_URL` / `ANTHROPIC_AUTH_TOKEN` into the task container, rewriting `localhost`/`127.0.0.1` → `host.docker.internal`.
- Adds `--add-host=host.docker.internal:host-gateway` to the Docker run args.
- Sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0` to keep vLLM's KV cache from being invalidated by the Anthropic attribution header (~90% slowdown otherwise).
- Applies the small-context profile (`max_output_tokens=4096`, `file_read_max_tokens=4000`, `mcp_max_output_tokens=2000`, plus a `disallowed_tools` list that strips schemas eating thousands of baseline tokens) when the model name contains "qwen" (case-insensitive).

### Per-GPU concurrency

`launch_vastai_benchmark.sh` estimates concurrency as
`(VRAM_total − headroom − model_size) / KV_per_request`, capped at 8.
For the AWQ 27B model (~14 GB) and a 65K context (~6 GB KV / request):

| Vast.ai GPU (≈VRAM) | Auto concurrency | Manual cap (larger ctx) |
|---|---|---|
| RTX 4090 (24 GB) | 1 | 1 |
| 2× RTX 4090 (48 GB) | 2 | 2 |
| 2× RTX 5090 (64 GB) | 5 | 6 |
| 4× RTX 5090 (128 GB) | 8 | 8 (capped) |
| A100 40 GB | 2 | 2 |
| A100 80 GB | 6 | 6 |
| RTX PRO 6000 Blackwell (96 GB) | 8 | 8 |
| 2× H100 (180 GB) | 8 | 8 |
| 4× H100 (360 GB) | 8 | 8 (capped) |

> **Multi-GPU works automatically.** `serve_vllm.sh` auto-detects the
> GPU count and sets `--tensor-parallel-size` to match, so N×5090
> shards the 27B model across all N cards with no extra config.
> vllm 0.23.0 supports Blackwell (sm_120, RTX 5090) via its CUDA 12.9
> base image.  Override with `VLLM_TENSOR_PARALLEL_SIZE=K` if you want
> to use fewer GPUs than the host exposes (e.g. for a partial rent).

Override with `--concurrency N` or `COOPERBENCH_CONCURRENCY=N`.  For
128K context, divide by 2 (KV cache doubles per request).

### Compaction trigger

Auto-set to 60% of the served model's context (e.g. 39321 for 65K
ctx).  Override with `COOPERBENCH_COMPACTION_TRIGGER=N` or pass
`--no-auto-compaction` to use the YAML default.

---

## 6. Fallback: `mini_swe_agent_v2` (no Claude Code CLI)

If you can't install the `@anthropic-ai/claude-code` npm package (sandboxed
networks, no npm registry, etc.) the vLLM endpoint still works against
CooperBench's `mini_swe_agent_v2` adapter, which speaks OpenAI-compatible
tool calling via LiteLLM:

```bash
uv run cooperbench run \
    -n my-vastai-fallback \
    --setting coop \
    -s flash_25 \
    -m openai/qwen3.6-27b-awq-int4 \
    -a mini_swe_agent_v2 \
    -c 2 \
    --backend docker \
    --agent-config /opt/cooperbench/agent_config_vastai.yaml
```

Where `agent_config_vastai.yaml` carries the OpenAI-compatible endpoint:

```yaml
model:
  model_kwargs:
    api_base: http://host.docker.internal:8000/v1
    api_key: dummy
    drop_params: true
agent:
  step_limit: 100
  compaction_token_trigger: 28000   # leave headroom on 32K-context models
```

This adapter does its own prompt templating, so the small-context
profile / `disallowed_tools` pruning from `claude_code` does not apply —
expect more token usage and a lower per-task cap.  Use it for quick
sanity checks only; the Claude Code adapter is the recommended path.

---

## 7. Troubleshooting

### vLLM OOMs at startup

The AWQ INT4 quant of the 27B model is ~14 GB.  The auto-calculated
`gpu-memory-utilization` of 0.92 is fine on ≥ 32 GB GPUs.  For a 24 GB
RTX 4090 you need to either:

- Drop `gpu-memory-utilization` to `0.85` and `max-model-len` to `32768`.
- Or switch to a smaller model.

Override via env vars in the `cb-vllm` run:

```bash
docker rm -f cb-vllm
VLLM_MAX_MODEL_LEN=32768 VLLM_GPU_MEMORY_UTILIZATION=0.85 \
docker run -d --name cb-vllm --network host --gpus all --restart unless-stopped \
    -e VLLM_MAX_MODEL_LEN=32768 -e VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    -v vllm-cache:/root/.cache/huggingface \
    ghcr.io/laminair/cooperbench-vastai:0.0.23 \
    bash -c "source /etc/profile.d/claude.sh && /opt/cooperbench/scripts/serve_vllm.sh --bg"
```

### `host.docker.internal` not resolving inside agent containers

The `claude_code` adapter adds `--add-host=host.docker.internal:host-gateway`
to every container it launches.  If your Docker daemon is too old to
support `host-gateway` (pre-20.10), the agent container falls back to
the literal hostname, which won't resolve.  Upgrade Docker or run the
benchmark on the host (no outer container) and use `localhost:8000`.

### `could not select device capability` from `cb-vllm`

`nvidia-container-toolkit` is missing or not configured inside the
outer container.  Run `nvidia-ctk runtime configure --runtime=docker`
on the host VM and restart Docker, or rebuild the image with
`nvidia-container-toolkit` baked in (the Dockerfile already does this).

### Port 8000 already in use

Either stop the conflicting process (`lsof -i :8000`) or run
`cb-vllm` on a different port:

```bash
docker rm -f cb-vllm
VLLM_PORT=8001 docker run -d --name cb-vllm --network host --gpus all \
    -e VLLM_PORT=8001 -v vllm-cache:/root/.cache/huggingface \
    ghcr.io/laminair/cooperbench-vastai:0.0.23 \
    bash -c "source /etc/profile.d/claude.sh && /opt/cooperbench/scripts/serve_vllm.sh --bg"

# And point cooperbench at the new port
bash scripts/launch_vastai_benchmark.sh --base-url http://host.docker.internal:8001
```

### Restarting from scratch

```bash
docker rm -f cb-vllm cb-redis
docker volume rm vllm-cache
bash /opt/cooperbench/scripts/setup_vastai.sh
```

`vllm-cache` is the named volume that holds the downloaded model
weights.  Removing it forces a re-download on the next vLLM start.

### Useful commands

```bash
docker logs -f cb-vllm          # vllm stdout/stderr

# Or run the all-in-one diagnostic collector (container state + log tail +
# GPU info + HF cache size + endpoint check):
bash scripts/diagnose_vastai.sh
docker logs -f cb-redis         # redis stdout/stderr
nvidia-smi                      # GPU utilization
ls -la /var/run/docker.sock     # verify socket mount
```

---

## 8. References

- `Dockerfile.vastai` — the image recipe.
- `scripts/setup_vastai.sh` — one-shot VM bootstrap.
- `scripts/prepare_vastai_benchmark.sh` — pre-flight + dataset download.
- `scripts/launch_vastai_benchmark.sh` — the production runner.
- `scripts/smoke_test_vastai.sh` — one-task sanity check.
- `scripts/serve_vllm.sh` — raw vLLM launcher (used inside `cb-vllm`).
- `docs/QWEN_LOCAL.md` — generic vLLM/`claude_code` recipe (covers
  vLLM flags, the attribution-header workaround, the small-context
  profile).
- `CHANGELOG.md` — what changed in this release.
