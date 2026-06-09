# CooperBench — Llama.cpp + Qwen3.6 27B Benchmark

Local benchmarking of [CooperBench](https://github.com/cooperbench/CooperBench) using
a self-hosted [llama.cpp](https://github.com/ggerganov/llama.cpp) server running
Qwen3.6-27B-Q4_K_M.

## Hardware requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1x 24 GB VRAM (RTX 4090) | 2x 90 GB VRAM (RTX 6000 Pro) |
| System RAM | 32 GB | 64 GB |
| Disk | 30 GB (model + dataset + Docker images) | 50 GB |
| Software | Docker, NVIDIA drivers, CUDA toolkit | — |

### VRAM breakdown

| Component | Size |
|-----------|------|
| Qwen3.6-27B-Q4_K_M | ~16.8 GB |
| KV cache (65K ctx, per concurrent request) | ~6 GB |
| Overhead | ~2 GB |

With tensor-parallelism across 2 GPUs, the model is split (~8.5 GB per GPU).
Remaining VRAM is available for KV caches. At `concurrency=8` in coop mode
(16 agents), total KV cache usage is ~48 GB — well within 180 GB total.

## Quick start

### 1. Install dependencies

```bash
# Clone the repo and install everything
cd CooperBench
uv sync --extra wandb
```

### 2. Start the llama.cpp server

```bash
# Auto-detects GPUs, sets tensor-split for multi-GPU
./scripts/serve_llama_cpp.sh --bg
```

Or manually:

```bash
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  --model ~/.cache/huggingface/qwen3.6-27b/Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8050 \
  --ctx-size 65536 --n-gpu-layers 99 \
  --flash-attn auto --jinja \
  --api-key local-llama-cpp
```

Environment overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_MODEL_PATH` | auto-detected | Path to .gguf model |
| `LLAMA_PORT` | `8050` | Server port |
| `LLAMA_CTX_SIZE` | `65536` | Context window (higher = less concurrency) |
| `LLAMA_API_KEY` | `local-llama-cpp` | API auth key |
| `LLAMA_GPU_SPLIT` | auto (even) | Tensor split ratio, e.g. `0.5,0.5` |

### 3. Prepare benchmark data

```bash
# Basic setup
./scripts/prepare_llama_cpp_benchmark.sh

# With Docker image pre-pull (avoids first-run latency)
./scripts/prepare_llama_cpp_benchmark.sh --pre-pull
```

This will:
- Verify the server is reachable
- Install Python dependencies (including `wandb`)
- Download the CooperBench dataset from HuggingFace
- **`--pre-pull`**: Pull all Docker Hub images for the subset plus `redis:alpine`
- Run a quick connectivity smoke test

### 4. Smoke test (single task)

```bash
./scripts/smoke_test_llama_cpp.sh
```

Runs one solo + one coop task to verify end-to-end pipeline.

Options:
- `--repo REPO --task ID --features F1,F2` — choose a specific task
- `--solo-only` / `--coop-only` — run only one mode

### 5. Run the full benchmark

```bash
# Auto-detects GPU count and sets concurrency:
#   2x RTX 6000 Pro (180 GB) → concurrency=8
#   1x RTX 6000 Pro (90 GB)  → concurrency=6
#   2x smaller GPUs         → concurrency=4
#   1x RTX 4090 (24 GB)     → concurrency=1

./scripts/launch_llama_cpp_benchmark.sh
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--solo-only` | — | Only run solo benchmark |
| `--coop-only` | — | Only run coop benchmark |
| `--subset NAME` | `flash_25` | Subset: `flash_25`, `lite`, or full |
| `--concurrency N` | auto | Override parallel task count |
| `--force` | off | Re-run completed tasks |
| `--wandb-project P` | — | Weights & Biases project name |
| `--wandb-entity E` | — | W&B entity (team/user) |

### 6. Weights & Biases logging (optional)

CooperBench supports logging task results to W&B. Set it up:

```bash
# One-time setup
export WANDB_API_KEY="your-api-key"

# Run with W&B logging
./scripts/launch_llama_cpp_benchmark.sh \
  --wandb-project cooperbench-qwen3.6-27b \
  --wandb-entity your-team

# Or use env vars
WANDB_PROJECT=cooperbench-qwen3.6-27b \
WANDB_ENTITY=your-team \
  ./scripts/launch_llama_cpp_benchmark.sh
```

The W&B logger captures:
- Per-task: repo, task_id, features, run status, steps, cost, patch lines
- Per-eval: pass/fail for each feature, merge status, errors
- Aggregate: overall pass rate, total cost, wall time

## Architecture

### Adapter: `src/cooperbench/agents/llama_cpp/`

The `llama_cpp` adapter wraps mini-swe-agent v2 infrastructure,
connecting to a local llama.cpp server via litellm with OpenAI-compatible
tool calling. It is registered as `"llama_cpp"` (CLI shorthand `"lc"`).

Key differences from the `opencode_zen` adapter:
- No credential resolution — uses `LLAMA_CPP_BASE_URL` + `LLAMA_CPP_API_KEY` env vars
- No `provider_specific_fields` stripping
- Higher default `compaction_token_trigger` (55000 at 65K ctx vs 28000 at 32K)
- `reasoning_content` field from Qwen's thinking mode is preserved

### Config files

| File | Purpose |
|------|---------|
| `config/solo.yaml` | Single-agent prompt, limits, model_kwargs |
| `config/coop.yaml` | Multi-agent prompt with messaging/git sections |

Key config values in `model_kwargs`:

```yaml
api_base: http://localhost:8050/v1
api_key: local-llama-cpp
drop_params: true
```

Override compaction trigger at runtime:

```bash
LLAMA_CPP_COMPACTION_TRIGGER=120000 ./scripts/launch_llama_cpp_benchmark.sh
```

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/serve_llama_cpp.sh` | Start llama-server with GPU auto-detection |
| `scripts/prepare_llama_cpp_benchmark.sh` | Verify server, install deps, download dataset |
| `scripts/smoke_test_llama_cpp.sh` | Single-task end-to-end test |
| `scripts/launch_llama_cpp_benchmark.sh` | Full solo + coop benchmark run |

## Tuning for different hardware

### RTX 4090 (24 GB)

```bash
# Default 65K ctx works. Concurrency=1 to stay within VRAM.
./scripts/serve_llama_cpp.sh --bg
./scripts/launch_llama_cpp_benchmark.sh --concurrency 1
```

### 2x RTX 4090 (48 GB total)

```bash
./scripts/serve_llama_cpp.sh --bg
./scripts/launch_llama_cpp_benchmark.sh --concurrency 2
```

### RTX 6000 Pro (1x 90 GB)

```bash
./scripts/serve_llama_cpp.sh --bg
# Auto-detects → concurrency=6. Or override:
./scripts/launch_llama_cpp_benchmark.sh --concurrency 8
```

### RTX 6000 Pro (2x 90 GB)

```bash
# Optional: increase context for longer agent trajectories
LLAMA_CTX_SIZE=131072 ./scripts/serve_llama_cpp.sh --bg
# Auto-detects → concurrency=8. Or override:
COOPERBENCH_CONCURRENCY=12 ./scripts/launch_llama_cpp_benchmark.sh
```

### CPU-only (no GPU)

```bash
LLAMA_N_GPU_LAYERS=0 ./scripts/serve_llama_cpp.sh --bg
# Expect 10-50x slower. Reduce concurrency to 1.
./scripts/launch_llama_cpp_benchmark.sh --concurrency 1
```

## Context window vs concurrency trade-off

Each doubling of `LLAMA_CTX_SIZE` roughly doubles the KV-cache VRAM per request,
halving the number of concurrent tasks you can run.

| Context | KV cache/task | 2x90GB max concurrent |
|---------|---------------|----------------------|
| 32K | ~3 GB | ~16 |
| 65K | ~6 GB | ~8 |
| 131K | ~12 GB | ~4 |

For CooperBench tasks, 65K is the recommended sweet spot.

## Troubleshooting

### Server not reachable

```bash
# Check if server is running
curl -s -H "Authorization: Bearer local-llama-cpp" http://localhost:8050/v1/models

# Kill stale server and restart
./scripts/serve_llama_cpp.sh --kill --bg
```

### Model won't fit in VRAM

Reduce context size or use a smaller quant:

```bash
LLAMA_CTX_SIZE=32768 ./scripts/serve_llama_cpp.sh --bg
```

### Docker image pull failures

```bash
# Pre-pull images to avoid first-run latency
./scripts/prepare_llama_cpp_benchmark.sh --pre-pull

# Or pull a specific image
docker pull akhatua/cooperbench-dottxt-ai-outlines:task1655
```

Images are hosted on Docker Hub as `akhatua/cooperbench-{repo}:task{id}`.
The benchmark auto-pulls missing images, but pre-pulling avoids staggered
delays during the first run.

### Out of memory during benchmark

Lower concurrency:

```bash
COOPERBENCH_CONCURRENCY=2 ./scripts/launch_llama_cpp_benchmark.sh
```

### W&B login

```bash
uv run wandb login
# Or set WANDB_API_KEY in your environment
```

## Expected results

Based on prior DeepSeek V4 Flash benchmarks on `flash_25` (22 task pairs):

| Mode | Pass rate (DeepSeek) | Est. time (Qwen3.6, conc=1) |
|------|---------------------|----------------------------|
| Solo | 50% | ~4 hours |
| Coop | 27% | ~8 hours |
| Combined | 38% | ~12 hours |

At concurrency=8 on 2x RTX 6000 Pro, divide times by ~8.
