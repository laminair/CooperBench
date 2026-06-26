# CooperBench Optimization Summary

> **Note**: this document describes tuning decisions made for the
> llama.cpp / HPC-enroot flow that CooperBench used up to v0.0.19.  The
> compaction triggers, step limits, and small-context profile values
> below were carried forward into the v0.0.20 Vast.ai + vLLM
> deployment (see [docs/VASTAI.md](docs/VASTAI.md)) — the `claude_code`
> adapter still honors `COOPERBENCH_COMPACTION_TRIGGER` and the
> small-context model profile.  Hardware sizing tables and
> `scripts/launch_llama_cpp_benchmark.sh` references are retained for
> historical context only.

## Changes Implemented

### Phase 1: Context Window Management (Targets 12 failures)

**Files Modified:**
- `src/cooperbench/agents/llama_cpp/config/solo.yaml`
- `src/cooperbench/agents/llama_cpp/config/coop.yaml`

**Changes:**
1. **Compaction trigger**: 55000 → 40000 tokens
   - Triggers summarization earlier to prevent context overflow
   - Leaves more headroom for the summary operation itself

2. **Recent turns kept**: 2 → 1
   - More aggressive compaction preserves less recent context
   - Frees up space for new information

3. **Tool output truncation**: 10000 → 5000 chars
   - Head/tail: 5000/5000 → 2500/2500
   - Large file reads were filling context rapidly (some tasks hit limits in 11-15 steps)

**Expected Impact:** Fixes 12 "context exceeded" failures

---

### Phase 2: Step Efficiency (Targets 19 "no output" failures)

**Files Modified:**
- `src/cooperbench/agents/mini_swe_agent_v2/agents/default.py`
- `src/cooperbench/agents/llama_cpp/config/solo.yaml`
- `src/cooperbench/agents/llama_cpp/config/coop.yaml`

**Changes:**
1. **Progress checkpoints** (new `AgentConfig` field: `checkpoint_steps`)
   - Injects reminders at steps 25, 50, 75
   - Message: "Step X/100 (Y remaining). If you've made code changes, consider submitting now."
   - Prevents model from spending all 100 steps exploring without committing

2. **Early submission detection** (new `AgentConfig` field: `early_submit_step`)
   - After step 70, checks `git status --porcelain`
   - If uncommitted changes exist, prompts submission
   - Prevents "explores but doesn't commit" failure mode

3. **Improved system prompts**
   - Solo: "Make incremental progress and submit working code early rather than trying to perfect everything. Aim to produce a working patch within 60-70 steps..."
   - Coop: Same message added to existing system prompt

**Expected Impact:** Fixes 19 "LimitsExceeded with no patch" failures

---

### Phase 3: Empty Patches (Targets 2 failures)

**Files Modified:**
- `src/cooperbench/agents/llama_cpp/adapter.py`
- `src/cooperbench/agents/llama_cpp/config/solo.yaml`
- `src/cooperbench/agents/llama_cpp/config/coop.yaml`

**Changes:**
1. **Empty patch warning**
   - Logs warning when harvested patch is < 50 chars
   - Helps identify adapter/tool bugs

2. **Improved submission instructions**
   - Added validation step: `if [ ! -s patch.txt ]; then echo "ERROR: patch.txt is empty!"; fi`
   - Added `wc -l patch.txt` to verify line count
   - Makes empty patches visible to the model before submission

**Expected Impact:** Fixes 2 "empty patch" failures

---

### Phase 4: Coop Improvements (Targets 5 merge conflicts)

**Files Modified:**
- `src/cooperbench/agents/mini_swe_agent_v2/agents/default.py`
- `src/cooperbench/eval/sandbox.py`

**Changes:**
1. **Message truncation** (new `AgentConfig` field: `message_truncate_chars`)
   - Coop only: truncates inter-agent messages to 2000 chars
   - Format: first 1000 + "...[truncated]..." + last 1000
   - Reduces context overhead from coordination

2. **Three-way merge strategy**
   - New `_merge_three_way()` function in eval
   - Applies patches sequentially using `git apply --3way`
   - Better conflict resolution than naive merge
   - Integrated into `test_merged()` between naive merge and solo fallback

**Expected Impact:** Fixes some of the 5 "merge conflicts" failures

---

### Phase 5: Launch Script Optimization

**Files Modified:**
- `scripts/launch_llama_cpp_benchmark.sh`

**Changes:**
1. **Fixed concurrency formula**
   - Old: `total_gb / (model_gb + kv_gb)` — treated model as per-task
   - New: `(total_gb - model_gb) / kv_gb` — accounts for shared model weights
   - For single 90GB H100: now correctly estimates 8 concurrent tasks (was 3)

2. **Compaction trigger auto-set**
   - Old: `SERVER_CTX - 10000` (55000 for 65K ctx) — overrode optimized config
   - New: `int(SERVER_CTX * 0.6)` (39321 for 65K ctx) — matches our optimization
   - Added `--no-auto-compaction` flag to skip auto-set entirely

**Expected Impact:** Better resource utilization on single GPU setups

---

## Usage for Single 90GB H100

### Basic Usage
```bash
./scripts/launch_llama_cpp_benchmark.sh
```

The script will auto-detect:
- GPU count: 1
- VRAM: ~90GB
- Concurrency: 8 (based on new formula)
- Compaction trigger: 39321 (60% of 65K ctx)

### Conservative Usage
If you encounter OOM errors or want to be more conservative:
```bash
./scripts/launch_llama_cpp_benchmark.sh --concurrency 4
```

### Disable Auto-Compaction
To use the config YAML value (40000) instead of auto-set:
```bash
./scripts/launch_llama_cpp_benchmark.sh --no-auto-compaction
```

### Expected Performance
For a single 90GB H100 with Qwen 3.6 27B Q4_K_M:
- Model size: ~17GB
- KV cache per request: ~5.9GB at 65K context
- Available for KV: 90 - 15 (headroom) - 17 (model) = 58GB
- Max concurrent requests: 58 / 5.9 ≈ 8-10

The script caps at 8 for safety. You can override with `--concurrency` if needed.

---

## Test Coverage

**New Tests Added:**
- `tests/agents/mini_swe_agent_v2/test_default_agent.py` (12 tests)
  - Checkpoint injection
  - Early submission detection
  - Message truncation

- `tests/eval/test_sandbox.py` (2 tests)
  - Three-way merge function existence
  - Integration into test_merged()

**All Tests Pass:**
```
399 passed, 63 skipped, 0 failed
```

---

## Expected Benchmark Improvement

Based on the failure analysis in `logs/BENCHMARK_QWEN36_27B.md`:

| Failure Mode | Count | Expected Fix |
|--------------|-------|--------------|
| Context exceeded | 12 | Phase 1 (compaction + truncation) |
| LimitsExceeded (no output) | 19 | Phase 2 (checkpoints + early submit) |
| Empty patches | 2 | Phase 3 (validation + warnings) |
| Merge conflicts | 5 | Phase 4 (three-way merge) |
| **Total fixable** | **38** | **Out of 50 tasks** |

**Potential improvement:** From 14% (7/50) to ~70-80% pass rate if all optimizations work as expected.

---

## Verification Checklist

- [x] Ruff lint passes
- [x] Ruff format check passes (4 pre-existing issues unrelated to changes)
- [x] MyPy type checking passes (95 source files)
- [x] All tests pass (399 passed, 63 skipped)
- [x] Bash script syntax valid
- [x] Concurrency formula verified for 90GB H100

---

## Next Steps

1. **Run a small benchmark subset** (e.g., 5 tasks) to validate improvements
2. **Monitor context usage** in logs to verify compaction triggers correctly
3. **Check step distribution** to see if checkpoints encourage earlier submission
4. **Analyze merge conflicts** to see if three-way merge helps
5. **Adjust concurrency** if OOM errors occur (try `--concurrency 4` or `6`)

---

## Configuration Reference

### llama_cpp/config/solo.yaml
```yaml
agent:
  step_limit: 100
  compaction_token_trigger: 40000
  compaction_keep_recent_turns: 1
  checkpoint_steps: [25, 50, 75]
  early_submit_step: 70
```

### llama_cpp/config/coop.yaml
```yaml
agent:
  step_limit: 100
  compaction_token_trigger: 40000
  compaction_keep_recent_turns: 1
  checkpoint_steps: [25, 50, 75]
  early_submit_step: 70
  message_truncate_chars: 2000
```

### Environment Variables
- `LLAMA_CPP_COMPACTION_TRIGGER`: Override compaction trigger (set by script)
- `COOPERBENCH_CONCURRENCY`: Override concurrency (set by script)
- `COOPERBENCH_VRAM_HEADROOM_MB`: VRAM headroom (default: 15000)
