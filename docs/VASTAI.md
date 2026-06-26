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

The Vast.ai template mounts `/var/run/docker.sock` from the host VM into
the outer container, so the cooperbench process inside our image can
launch agent containers using the same Docker daemon that runs the
vLLM and Redis sidecars.

---

## 1. Build the image

The image is built on your workstation (or any CUDA 13.x host) and
pushed to GitHub Container Registry (ghcr.io):

```bash
# From the repo root.
docker build -f Dockerfile.vastai -t ghcr.io/laminair/cooperbench-vastai:0.0.20 .
docker push  ghcr.io/laminair/cooperbench-vastai:0.0.20
```

The default baked-in model is `cyankiwi/Qwen3.6-27B-AWQ-INT4`
(AWQ INT4 quantization of the 27B Qwen — see `OPTIMIZATION_SUMMARY.md`
for the tuning decisions that drove that choice).  Override the default
at build time:

```bash
docker build -f Dockerfile.vastai \
  --build-arg VLLM_VERSION=0.17.1 \
  --build-arg CLAUDE_CODE_VERSION=2.1.0 \
  -t ghcr.io/laminair/cooperbench-vastai:0.0.20 .
```

If vLLM does not yet publish a cu132 wheel, edit `Dockerfile.vastai` to
base on `FROM vllm/vllm-openai:v0.17.1` instead of the nvidia/cuda base
and remove the `RUN pip install vllm` line.  The vLLM project's image
already bundles the right CUDA + PyTorch; the 13.2 host driver remains
forward-compatible.

---

## 2. Provision the Vast.ai instance

In the Vast.ai launch dialog, set:

| Field | Value |
|---|---|
| **Docker Image** | `ghcr.io/laminair/cooperbench-vastai:0.0.20` |
| **Docker Socket** | **ON** (mounts `/var/run/docker.sock` from the host) |
| **Disk** | ≥ 80 GB (model weights + dataset + pulled images) |
| **GPU** | Any 24 GB+ NVIDIA; recommend RTX PRO 6000 Blackwell for the 27B model |
| **On-start script** | `bash /opt/cooperbench/scripts/setup_vastai.sh` |

If you don't have a custom-image template yet, save one with these
fields and reuse it for every run.

---

## 3. First boot

After Vast.ai reports the instance is up, SSH in (or use the in-browser
terminal):

```bash
# Already running if onstart was set; otherwise:
bash /opt/cooperbench/scripts/setup_vastai.sh
```

This script is **idempotent**.  It:

1. Verifies the docker socket is mounted and the NVIDIA driver + toolkit work.
2. Pulls our own image (so `cb-vllm` can be re-launched as a copy).
3. Starts `cb-redis` (Redis 7, host network, port 6379).
4. Starts `cb-vllm` (vLLM serving the 27B AWQ model, host network, `--gpus all`, port 8000).
5. Waits for vLLM to respond at `http://localhost:8000/v1/models` (up to 6 minutes — first boot also downloads the model).
6. Pre-pulls the `akhatua/cooperbench-*` task images for `flash_25`.

Then a one-time prep:

```bash
cd /opt/cooperbench
bash scripts/prepare_vastai_benchmark.sh
```

This installs the Python dependencies, downloads the CooperBench dataset
from HuggingFace, and sends a 32-token sanity ping through the
Anthropic `/v1/messages` endpoint.

---

## 4. Run the benchmark

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
| A100 40 GB | 2 | 2 |
| A100 80 GB | 6 | 6 |
| RTX PRO 6000 Blackwell (96 GB) | 8 | 8 |
| 2× H100 (180 GB) | 8 | 8 |
| 4× H100 (360 GB) | 8 | 8 (capped) |

Override with `--concurrency N` or `COOPERBENCH_CONCURRENCY=N`.  For
128K context, divide by 2 (KV cache doubles per request).

### Compaction trigger

Auto-set to 60% of the served model's context (e.g. 39321 for 65K
ctx).  Override with `COOPERBENCH_COMPACTION_TRIGGER=N` or pass
`--no-auto-compaction` to use the YAML default.

---

## 5. Fallback: `mini_swe_agent_v2` (no Claude Code CLI)

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

## 6. Troubleshooting

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
    ghcr.io/laminair/cooperbench-vastai:0.0.20 \
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
    ghcr.io/laminair/cooperbench-vastai:0.0.20 \
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
docker logs -f cb-redis         # redis stdout/stderr
nvidia-smi                      # GPU utilization
ls -la /var/run/docker.sock     # verify socket mount
```

---

## 7. References

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
