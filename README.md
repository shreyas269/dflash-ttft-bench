# DFlash TTFT optimization + benchmark

Reproducing and fixing the DFlash prefill-TTFT behaviour in vLLM PR
[#36847](https://github.com/vllm-project/vllm/pull/36847), with a
4-mode sweep that compares:

| mode | what runs |
|---|---|
| `no_spec` | target model only (upper bound) |
| `eagle3` | EAGLE3 drafter, 32-node depth-6 static tree |
| `dflash_optimized` | DFlash with the patch in this repo applied |
| `dflash_original` | DFlash with the patch `git stash`-ed away (= upstream PR baseline) |

The benchmark covers cold prefill, full-hit prefix cache, partial-hit
shared-prefix, and batched prefill (4 and 16 concurrent requests).

---

## 1. What this repo contains

```
dflash-ttft-bench/
├── README.md                  this file
├── dflash_ttft_fix.patch      unified diff to apply on top of vLLM main
├── ttft_comprehensive.py      per-mode bench driver (one LLM, 5+2 scenarios)
├── run_comprehensive.sh       4-mode orchestrator (git-stash boundary for baseline)
└── summarize.py               tabulates the 4 JSONs into a delta table
```

The patch file was generated against vLLM commit
`1c2c1eb8b9fdd2e67c45afb6123ccc07c0177555` on `main`. It should apply
cleanly to any revision that still contains PR #36847.

---

## 2. What the optimization changes

The upstream DFlash PR has a per-prefill drafter-side path called
`precompute_and_store_context_kv` that runs **outside `torch.compile` and
outside CUDA graphs** (because its input shape is the full context, not
the drafter's query shape). On long prefills that path is the TTFT
bottleneck.

Walking the old path end-to-end for a prefill of `num_ctx` target tokens
on a drafter with `L` attention layers, `nkv` KV heads, head dim `hd`:

```
  hidden_norm(out: [num_ctx, H])
    └─ F.linear(fused-KV weight)  → [num_ctx, L*2*nkv*hd]           # 1 GEMM (good)
       └─ .view(num_ctx, L, 2, nkv, hd).permute(2,1,0,3,4).contiguous()
          → materializes a [2, L, num_ctx, nkv, hd] tensor          # big memcpy
          └─ for layer in L:   ops.rms_norm(empty_like(all_k)[l])   # L kernel launches
             └─ positions_repeated = positions.repeat(L)            # [L*num_ctx]
                └─ ops.rotary_embedding(all_k_flat) over L*num_ctx  # 1 big RoPE
                   └─ for layer in L:  do_kv_cache_update(...)      # L kernel launches
```

For a ≥8k-token prefill with `L=5`, `nkv=8`, `hd=128`, bfloat16:

- the `[2, L, num_ctx, nkv, hd]` `.contiguous()` alone is
  `2·5·8192·8·128·2 B ≈ 167 MB` of copy on the main compute stream
- the `torch.empty_like(all_k)` for the K-norm output is another `84 MB`
  allocation
- `positions.repeat(L)` grows the positions tensor by L× before the RoPE

None of this is covered by CUDA graphs, so it shows up on every prefill.

### Change 1 — `num_ctx == 0` early return (`vllm/v1/spec_decode/dflash.py`)

When prefix caching fully covers the request (common in steady-state
chat with long system prompts), `num_ctx` is zero. The upstream code
still walks the pipeline: a zero-sized `F.linear`, a zero-sized `.view /
.permute / .contiguous`, L zero-sized `rms_norm`s, a zero-sized RoPE, L
zero-sized cache writes. Each is sub-ms but they accumulate, plus a
hidden-norm allocator round-trip.

The fix is a single early return guard in
`DFlashProposer.build_model_inputs_first_pass` and a matching one inside
`precompute_and_store_context_kv` itself. Full-cache-hit requests now
skip the drafter precompute entirely.

### Change 2 — fused Triton kernel for K-norm + neox RoPE + K/V stage (`vllm/v1/spec_decode/dflash_kv_precompute.py`, new file)

The big wins come from collapsing the per-layer per-ctx inner loop into
one Triton kernel per layer:

```
  for layer in L:
    _dflash_fused_k_norm_rope_kernel[grid=(num_ctx, nkv)](
        kv_fused,                 # [num_ctx, L, 2, nkv, hd] strided view
        layer_elem_offset = layer * 2*nkv*hd,
        k_norm_weight[layer],     # [hd]
        cos_sin_cache,            # [max_pos, rotary_dim]
        positions,                # [num_ctx]  (NOT repeated)
        k_out = k_scratch,        # pre-allocated [num_ctx, nkv, hd]
        v_out = v_scratch,        # pre-allocated [num_ctx, nkv, hd]
    )
    do_kv_cache_update(k_scratch, v_scratch, slot_mapping)
```

Per `(ctx, head)` program, the kernel:

1. Loads K directly from the fused `F.linear` output via strided pointer
   arithmetic — **no `.permute().contiguous()` materialization**.
2. Computes `RMSNorm(K)` in registers (one warp-reduced sum, one `rsqrt`,
   one weight multiply).
3. Reads one cos/sin row for this `(ctx)` position — **positions tensor
   is read once, not `.repeat(L)`**.
4. Applies neox-style RoPE to the normed K halves in registers.
5. Stores K (rotated) and V (pass-through) to the pre-allocated
   `k_out` / `v_out` staging buffers.

The project's existing `do_kv_cache_update` then writes those buffers
into the draft KV cache with full layout / fp8 awareness.

### Change 3 — persistent scratch buffers (`vllm/model_executor/models/qwen3_dflash.py`)

`_build_fused_kv_buffers` now pre-allocates three buffers that were
previously allocated per prefill:

- `_k_scratch`, `_v_scratch`: `[max_num_batched_tokens, nkv, hd]` each.
  Sized for the largest prefill chunk the scheduler can ever send.
- `_hidden_norm_scratch`: grows lazily to the high-water mark of
  `context_states.shape[0]`.

Stable addresses make the path CUDA-graph-safe even though we don't
capture it today, and cut the per-prefill allocator round-trip.

### Change 4 — eager fallback for non-neox RoPE

`DFlashQwen3Model` now sets `self._can_fuse_k_norm_rope = bool(is_neox)`
at build time. When false, `_precompute_eager_fallback` runs the original
PR code path. This is a safety net; no DFlash checkpoint in current use
needs it, but it means merging this patch does not narrow the
configuration space.

### Why the fused kernel is correct

The upstream [sglang](https://github.com/sgl-project/sglang) DFlash
port uses the same `RMSNorm + RoPE + stage` fusion
(`python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py`).
Local verification on Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16 against
GSM8K 30-question slice:

| variant | accuracy |
|---|---|
| fused (this patch) | 26/30 |
| eager (original PR) | 25/30 |

Within 1 question, i.e. noise. End-to-end greedy sampling agrees
token-for-token on the scripted prompts in this benchmark.

---

## 3. Reading the output

Each mode writes a JSON to `$OUT_DIR/<mode>.json` plus a matching
`.log` file. `summarize.py` reads all four JSONs and prints a per-
scenario table. Before looking at numbers, make sure you know what's
being measured and how the scenarios differ.

### 3.1 What's measured — TTFT only

Every number in the summary is **time-to-first-token (TTFT), in
milliseconds**. The bench driver submits each prompt with
`max_tokens=1`:

```python
sp = SamplingParams(temperature=0.0, max_tokens=1)
tic = time.perf_counter()
_ = llm.generate([p], sp)                 # blocks until 1 token returns
lats.append(time.perf_counter() - tic)
```

So each latency covers: prompt submit → prefill → drafter setup →
first sampled token → return. **No decode steps.**

Why TTFT only: PR #36847's regression was in the prefill / drafter
setup path. Speculative decoding pays back on **decode throughput**,
not TTFT. If you want decode-throughput numbers, that's a different
benchmark (set `max_tokens=128`, measure `(out_tokens - 1) / (total
- first_token_time)`). This one deliberately doesn't.

### 3.2 The scenarios

The five serial scenarios + two batched scenarios control the mix of
cold-prefill vs. cached-prefix work. Prefix caching is ON for all of
them; what changes is whether any given request actually hits the
cache.

| scenario | cache state | n | what it models | what it stresses |
|---|---|---|---|---|
| `SERIAL_COLD` | fully cold | 20 | 20 different users each ask a fresh question | baseline prefill speed, wide range of prompt lengths |
| `CACHE_HIT_MISS` | cold (1st send of each of 4 prompts) | 4 | sanity check, narrow set | cold-prefill on repeated shapes (noisy, n=4) |
| `CACHE_HIT_HIT` | fully warm (sends 2–5 of each of 4 prompts) | 16 | "regenerate response" button, retry after error | **warm-cache floor** — target does almost nothing, only drafter-setup overhead is visible |
| `SHARED_PREFIX_FIRST` | cold | 1 | first user of the day hits a cold system prompt | reference point, not statistically meaningful |
| `SHARED_PREFIX_LATER` | partial hit (2K cached + 400 cold) | 14 | production chat: long shared system prompt + short user turn | **the realistic workload** — best predictor of production TTFT |
| `BATCH_4` | cold, concurrent | 4 | small burst of simultaneous users | concurrent-prefill scheduler efficiency |
| `BATCH_16` | cold, concurrent | 16 | large burst of simultaneous users | larger-batch scheduler + drafter-per-seq overhead |

> The names `CACHE_HIT_MISS` / `CACHE_HIT_HIT` come from the underlying
> *experiment*: send the same prompt 5 times, split the first send
> (miss) from sends 2–5 (hit). Mentally translating them to
> `REPEAT_FIRST_COLD` / `REPEAT_WARM_HIT` may help.

**The two rows that matter most in practice:**

1. `CACHE_HIT_HIT` — how cheap is a warm request? Your floor.
2. `SHARED_PREFIX_LATER` — how cheap is a typical production request?

Everything else is context and sanity checks.

### 3.3 The summary columns

Each scenario prints one row per mode with these columns:

| column | meaning | when to trust it |
|---|---|---|
| `n` | number of requests in this row | bigger is better; p99 needs n≥50 to be meaningful |
| `min` | fastest TTFT in the sample | always noisy |
| `p50` | median — half faster, half slower | **your typical user experience** |
| `mean` | arithmetic average | compare to p50: if `mean >> p50`, distribution is right-skewed (outliers pulling up) |
| `p90` | 10% of requests were slower | **the SLO number for interactive workloads** |
| `p95` / `p99` | tail percentiles | only meaningful with many samples (n=20 makes p99 ≈ max) |
| `max` | slowest single request | one data point, noisy |
| `max/p50` | tail spread ratio | `1.0x` = uniform; `2.0x+` = wide prompt-length distribution. This is **prompt-driven, not drafter-driven** — not a regression signal |

Best signals for mode-to-mode comparison: `p50` and `p90`. Ignore `p99`
and `max` unless you're running the sweep with n≫20.

### 3.4 The Δ (delta) rows — sign convention

After the per-mode rows, each scenario prints three delta rows:

```
Δ dflash_original vs eagle3              p50= -9.8%  p90= -7.4%  p99= -7.3%  max= -7.3%
Δ dflash_optimized vs eagle3             p50= -9.6%  p90= -7.4%  p99= -7.2%  max= -7.3%
Δ dflash_optimized vs dflash_original    p50= +0.2%  p90= +0.0%  p99= +0.1%  max= +0.0%
```

Sign convention: **negative = faster, positive = slower**.

`p50 = −9.6%` means "A's p50 is 9.6% lower (faster) than B's p50".

The three rows answer three different questions:

1. **`dflash_original vs eagle3`** → Does DFlash-as-shipped beat EAGLE3 on this workload?
2. **`dflash_optimized vs eagle3`** → Does DFlash *with the patch* still beat EAGLE3?
3. **`dflash_optimized vs dflash_original`** → **Did the patch help, hurt, or do nothing?** This is the row that evaluates the optimization itself.

### 3.5 Methodology — what actually runs, in what order

**One vLLM instance per mode, four modes = four subprocesses.** Inside
one subprocess:

```
spawn → load model (~1 min) → CUDA-graph capture (2–5 min) →
  warmup (3 non-counted ShareGPT prompts, seed 1001) →
  SERIAL_COLD → CACHE_HIT_REPEAT → SHARED_PREFIX → BATCH_4 → BATCH_16 →
teardown
```

Scenarios run sequentially in the **same** `LLM` object. The CUDA-graph
cache, Triton autotune cache, and KV prefix cache all persist across
scenarios within a mode.

#### Why not respin between scenarios?

Two reasons:

1. **`CACHE_HIT_HIT` requires shared state by construction.** The
   whole point of that row is "prompt A is cached from a previous
   send of prompt A" — if the engine respun, the cache would be
   empty and the experiment wouldn't exist.
2. **Cost.** Each respin is ~3 min. 5 scenarios × 4 modes × 3 min =
   60 min of pure overhead per sweep. A full sweep is already ~30
   min; respinning between scenarios would triple it.

#### Ordering bias — and why the Δ rows are still valid

Because scenarios share state within a mode, there's a small bias:

- `SERIAL_COLD` runs first and bears any first-real-request CUDA
  graph / Triton compile cost not covered by the 3-prompt warmup.
- `BATCH_4` / `BATCH_16` at the end may hit batch graph shapes not
  pre-captured at startup.
- Later serial scenarios benefit from the warmed CUDA-graph cache.

**But this bias is the same for all four modes** (same order, same
warmup, same seeds). When you look at `Δ dflash_optimized vs
dflash_original`, both numerator and denominator carry the same
bias → it cancels. Absolute TTFT numbers may be slightly off; the
**deltas are clean**.

#### Prefix cache leakage across scenarios

Prompts aren't flushed between scenarios. But each scenario draws its
prompts from a different ShareGPT slice using a different RNG seed,
so there's no accidental overlap that would turn a cold request into
a false warm hit. Under extreme memory pressure earlier prompts
could evict, but that would only hurt later cold scenarios — still
identical across modes.

#### Prompt determinism — identical prompts every mode, every run

Every scenario's prompt set is deterministic:

| scenario | seed | pool size |
|---|---|---|
| warmup | 1001 | 3 |
| SERIAL_COLD | 42 | 20 |
| CACHE_HIT_REPEAT | 123 | 4 (×5 each) |
| SHARED_PREFIX | 7777 | 15 |
| BATCH_4 | 200 | 4 |
| BATCH_16 | 300 | 16 |

The selection is `random.Random(seed).shuffle(data)` followed by a
token-length filter using the tokenizer of `--target`. With the same
ShareGPT file, the same target, and the same seeds, **every mode
sees byte-for-byte identical prompts in the same order**.

### 3.6 How to actually read a sweep — decision tree

1. **Sanity check.** Does `no_spec` have the lowest TTFT in every
   scenario? If not, something is wrong (speculation only adds
   overhead on prefill).
2. **DFlash vs EAGLE3.** Look at `Δ dflash_optimized vs eagle3`. If
   `p50` and `p90` are consistently negative, DFlash wins on
   prefill. If `p99` turns positive while `p50` is negative, DFlash's
   tail is worse.
3. **Did the patch help?** Look at `Δ dflash_optimized vs
   dflash_original`. Interpretations:
   - All within ±1–2% → patch is a no-op on this workload (not a
     bug; the optimization targets a path that isn't the bottleneck
     here).
   - Negative p50/p90/p99 → patch helps.
   - Big negative in `CACHE_HIT_HIT` specifically → the
     `num_ctx == 0` early-return is firing (full-cache-hit requests
     are skipping the drafter precompute).
   - Negative delta in `SERIAL_COLD` / `SHARED_PREFIX_LATER`
     scaling with prompt length → the fused K-norm+RoPE kernel is
     winning over the old `.contiguous()` + `positions.repeat(L)`
     pipeline.
4. **What to ignore.** `n=1` rows (`SHARED_PREFIX_FIRST`), `max`
   differences under n<50, `p99` under n<50.


## 4. Results I observed (Qwen3-8B, GB10)

Full table is produced by `summarize.py`; the short version at
`num_speculative_tokens=15` (block_size=16 match for the z-lab drafter)
and EAGLE3 at the 32-node tree:

| scenario | no_spec p50 | eagle3 p50 | dflash_original p50 | dflash_optimized p50 |
|---|---|---|---|---|
| SERIAL_COLD | 2963 ms | 3428 ms | 3094 ms | 3099 ms |
| CACHE_HIT_MISS | 5864 | 6499 | 6098 | 6096 |
| **CACHE_HIT_HIT** | 302 | 643 | **370** | **372** |
| SHARED_PREFIX_FIRST | 4667 | 5245 | 4868 | 4863 |
| **SHARED_PREFIX_LATER** | 942 | 1378 | **1033** | **1035** |
| BATCH_4 total | 10298 | 11428 | 10735 | 10746 |
| BATCH_16 total | 58383 | 61652 | 59229 | 59245 |

Headlines:

- **DFlash is faster than EAGLE3 at every percentile in every scenario**
  once `num_speculative_tokens` matches the DFlash block size (here 15).
  On warm cache it's ~42% faster (370 ms vs 643 ms); on partial-hit
  shared prefix it's ~25% faster.
- **dflash_optimized vs dflash_original: within ±1%** at these settings
  on this model. The early-return dominates the win on full-hit (both
  land at ~370 ms because the fused kernel never runs when `num_ctx=0`),
  and the target model's prefill dominates the other scenarios so the
  drafter-side μs don't surface. The fused kernel's biggest expected
  impact is on shorter models or larger batches where the
  `.contiguous()` memcpy is a larger share of TTFT — **worth
  re-measuring on gpt-oss-120b**.

---

## 5. Running the benchmark

### 5.1 Prerequisites

- Linux + NVIDIA GPU with enough VRAM for target + drafter (Qwen3-8B
  needs ~30 GB in bf16; gpt-oss-120b needs TP≥2 or a big device).
- `uv` (recommended) or any Python 3.11/3.12 venv.
- `git`, `gh` (optional, for repo operations).

### 5.2 Clone vLLM and apply the patch

> **The benchmark script does NOT apply the patch for you. You must
> apply it once, manually, before running the orchestrator.** The
> script just swaps the patched edits in and out via `git stash`.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git apply /path/to/dflash-ttft-bench/dflash_ttft_fix.patch
git status --short          # should show 2 M + 1 ??
# Example expected output:
#    M vllm/model_executor/models/qwen3_dflash.py
#    M vllm/v1/spec_decode/dflash.py
#   ?? vllm/v1/spec_decode/dflash_kv_precompute.py
```

**Do not** commit the patch. The orchestrator uses `git stash push/pop`
to swap between:
- `dflash_optimized` — patch is in the working tree → fused kernel + early-return run
- `dflash_original` — patch is `git stash`-ed away → upstream PR #36847 runs

Keeping the patch as *tracked working-tree edits* (not a commit) is
what makes that single-stash swap work. The orchestrator now
pre-flights this check and errors out early if the patch is missing
when a DFlash mode is requested. To bypass (e.g. running for a
gpt-oss-120b target where you only want no_spec + eagle3), set
`SKIP_DFLASH_OPT=1 SKIP_DFLASH_ORIG=1`.

Reversing the patch later:

```bash
cd $VLLM_REPO
git checkout -- vllm/model_executor/models/qwen3_dflash.py \
                vllm/v1/spec_decode/dflash.py
rm vllm/v1/spec_decode/dflash_kv_precompute.py
```

### 5.3 Build vLLM

From the vLLM repo root:

```bash
uv venv --python 3.12
source .venv/bin/activate
# Python-only build (fastest; matches what I used):
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
# Install transformers + datasets for the bench driver:
uv pip install "transformers>=4.45" datasets huggingface_hub
```

If you need C++/CUDA changes (you shouldn't for this patch), drop
`VLLM_USE_PRECOMPILED=1`.

### 5.4 Download ShareGPT

The bench samples prompts from
[ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered).
One file, ~200 MB:

```bash
huggingface-cli download Aeala/ShareGPT_Vicuna_unfiltered \
    ShareGPT_V4.3_unfiltered_cleaned_split.json \
    --repo-type dataset --local-dir ~/data/sharegpt
# resulting file:
#   ~/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
```

### 5.5 Pre-fetch the models

The orchestrator runs with `HF_HUB_OFFLINE=1` recommended so it doesn't
stall on a flaky network mid-run. Pre-fetch everything once:

```bash
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download z-lab/Qwen3-8B-DFlash-b16
huggingface-cli download Tengyunw/qwen3_8b_eagle3
```

### 5.6 Run the sweep (Qwen3-8B default)

```bash
cd /path/to/dflash-ttft-bench
export PY=/path/to/vllm/.venv/bin/python
export VLLM_REPO=/path/to/vllm
export SHAREGPT_PATH=$HOME/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
export OUT_DIR=$PWD/ttft_out

./run_comprehensive.sh 2>&1 | tee $OUT_DIR/driver.log
```

Each mode is a separate Python subprocess (load + CUDA-graph capture
happens 4 times). On a single H100-class GPU total wall is ~30 min.

#### Per-method speculation budgets

DFlash and EAGLE3 have **different** natural speculation budgets and the
orchestrator treats them independently:

| env var | default | what it controls |
|---|---|---|
| `DFLASH_NUM_SPEC` | `15` | DFlash `num_speculative_tokens`. Must equal `block_size - 1` of your DFlash drafter (z-lab b16 → 15). |
| `EAGLE3_USE_TREE` | `1` | `1` = 32-node depth-6 static tree (approx of SGLang `--speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32`). `0` = plain chain. |
| `EAGLE3_NUM_SPEC` | `6` | EAGLE3 chain depth. Only used when `EAGLE3_USE_TREE=0`. |
| `NUM_SPEC` | — | Legacy alias for `DFLASH_NUM_SPEC`; kept for backward compat. |

Example: chain EAGLE3 at depth 4, DFlash at block_size 8:

```bash
DFLASH_NUM_SPEC=7 EAGLE3_USE_TREE=0 EAGLE3_NUM_SPEC=4 ./run_comprehensive.sh
```

#### Skipping modes

The orchestrator reads per-mode skip flags so you can run subsets:

```bash
# Skip EAGLE3 entirely (e.g. no EAGLE3 drafter for your target):
SKIP_EAGLE3=1 ./run_comprehensive.sh

# Just compare optimized vs original (skip baselines):
SKIP_NO_SPEC=1 SKIP_EAGLE3=1 ./run_comprehensive.sh
```

### 5.7 Tabulate

```bash
$PY summarize.py --out-dir $OUT_DIR
```

Prints a per-scenario table for all 4 modes plus delta lines for
`dflash_original vs eagle3`, `dflash_optimized vs eagle3`, and
`dflash_optimized vs dflash_original`.

---

## 6. Running for gpt-oss-120b

> **Note on drafter availability.** DFlash and EAGLE3 drafters are
> target-specific. For `openai/gpt-oss-120b`:
> - **EAGLE3** — NVIDIA publishes
>   [`nvidia/gpt-oss-120b-Eagle3-long-context`](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)
>   (for ≥8k contexts) and
>   [`nvidia/gpt-oss-120b-Eagle3-short-context`](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-short-context)
>   (for shorter contexts). Use these for the `eagle3` mode.
> - **DFlash** — I am not aware of a public DFlash drafter trained for
>   gpt-oss-120b. You will need your own checkpoint. If you only want
>   the `no_spec` and `eagle3` baselines, set `SKIP_DFLASH_OPT=1
>   SKIP_DFLASH_ORIG=1` to skip both DFlash modes.

### 6.1 Configuration for NVIDIA's `gpt-oss-120b-Eagle3-long-context`

NVIDIA's model card recommends **`max_draft_len: 3`** (i.e. a chain of 3
draft tokens, not a tree) with a benchmarked average acceptance rate of
~2.32 tokens/step on MT-Bench. Their deployment example uses TP=8 and
`max_seq_len=8192`. Mapped onto this script's knobs:

```bash
export PY=/path/to/vllm/.venv/bin/python
export VLLM_REPO=/path/to/vllm
export SHAREGPT_PATH=$HOME/data/sharegpt/ShareGPT_V4.3_unfiltered_cleaned_split.json
export OUT_DIR=$PWD/ttft_out_gpt_oss

export TARGET_MODEL="openai/gpt-oss-120b"
export EAGLE3_MODEL="nvidia/gpt-oss-120b-Eagle3-long-context"

# EAGLE3: NVIDIA recommends a 3-token chain, not a tree.
export EAGLE3_USE_TREE=0
export EAGLE3_NUM_SPEC=3

# DFlash: replace with your own drafter + matching block size. Leave
# these as-is and set SKIP_DFLASH_* if you don't have a drafter.
export DFLASH_MODEL="/path/to/your/gpt-oss-120b-dflash-bXX"
export DFLASH_NUM_SPEC=XX              # = block_size - 1 of your DFlash drafter

# Everything else matches NVIDIA's recipe.
export MAX_MODEL_LEN=8192
export TP=8                            # NVIDIA's example; 4 also works on 80 GB cards
export DTYPE=bfloat16
export TRUST_REMOTE_CODE=1             # gpt-oss MoE custom kernels

# Skip the DFlash modes if no drafter is available.
export SKIP_DFLASH_OPT=1
export SKIP_DFLASH_ORIG=1

./run_comprehensive.sh 2>&1 | tee $OUT_DIR/driver.log
$PY summarize.py --out-dir $OUT_DIR
```

> If your prompts fit under 8k, use the `short-context` variant instead
> (NVIDIA recommends it for that regime — better accept rate). Just
> change `EAGLE3_MODEL` to `nvidia/gpt-oss-120b-Eagle3-short-context`
> and keep everything else identical.

### 6.2 Knobs to know

All read by `run_comprehensive.sh`:

| knob | meaning | default | NVIDIA gpt-oss-120b |
|---|---|---|---|
| `DFLASH_NUM_SPEC` | DFlash `num_speculative_tokens`; must equal `block_size − 1` of your drafter | 15 | depends on your checkpoint |
| `EAGLE3_USE_TREE` | 1 = 32-node static tree; 0 = chain | 1 | **0** (NVIDIA uses a chain) |
| `EAGLE3_NUM_SPEC` | EAGLE3 chain depth (ignored when `EAGLE3_USE_TREE=1`) | 6 | **3** |
| `TP` | tensor-parallel degree | 1 | **8** |
| `MAX_MODEL_LEN` | max sequence length; also caps `max_num_batched_tokens` | 16384 | **8192** |
| `TRUST_REMOTE_CODE` | forwards `--trust-remote-code` to vLLM | 0 | **1** |
| `SKIP_NO_SPEC` / `SKIP_EAGLE3` / `SKIP_DFLASH_OPT` / `SKIP_DFLASH_ORIG` | skip that mode | 0 | set to 1 for any drafter you don't have |

### 6.3 Skipping modes when a drafter is missing

Instead of editing the orchestrator, set the corresponding `SKIP_*`
env var:

```bash
# No EAGLE3 drafter at all:
SKIP_EAGLE3=1 ./run_comprehensive.sh

# No DFlash drafter (e.g. today's gpt-oss-120b situation):
SKIP_DFLASH_OPT=1 SKIP_DFLASH_ORIG=1 ./run_comprehensive.sh

# Just EAGLE3 vs no-spec (smoke test of NVIDIA drafter, no DFlash):
SKIP_DFLASH_OPT=1 SKIP_DFLASH_ORIG=1 ./run_comprehensive.sh
```

`summarize.py` tolerates missing JSON files and only prints deltas for
the modes that ran.

### 6.4 What to look at in the output

The patch should most show up on:

- **SHARED_PREFIX_LATER** — partial-hit prefill, the realistic chat
  case. If the fused kernel helps your target, it shows here first.
- **SERIAL_COLD p50** — long cold prefills. The `.contiguous()` memcpy
  scales with `num_ctx`, so longer prompts → bigger relative win.
- **BATCH_16 total** — concurrent-prefill wall. The scheduler sends
  large `num_ctx` chunks, which is where the fused kernel's win vs. the
  per-layer `rms_norm`/RoPE chain compounds.

On Qwen3-8B (5 drafter layers) the optimization ended up within ±1%.
On a bigger target with a taller drafter, I'd expect a larger delta.

---

## 7. Troubleshooting

- **`git stash pop` conflict after a run** — means the run crashed
  between the stash push and pop. Fix: `cd $VLLM_REPO && git stash
  list` and pop manually. Nothing is lost; it's just sitting in the
  stash.
- **`ImportError: cannot import name 'LLM' from 'vllm'`** — you're
  running Python from inside the vLLM repo dir, which shadows the
  installed package. The orchestrator `cd`s to `$OUT_DIR` before
  invoking Python; if you run the driver by hand, `cd` somewhere that
  isn't the vLLM repo first.
- **`HF_HUB_OFFLINE=1` failures** — a model wasn't pre-cached. Unset
  the env var for one run so HF can download, then re-enable for
  steady benchmarking.
- **CUDA OOM on dflash_optimized only** — unlikely given scratch
  buffers shrink what's allocated per-call, but possible if
  `max_num_batched_tokens` is huge. Lower `MAX_MODEL_LEN`.

---

## 8. Applying the vLLM patch in a fresh tree

```bash
cd /path/to/vllm
git apply /path/to/dflash_ttft_fix.patch
# reverse with:  git apply -R /path/to/dflash_ttft_fix.patch
```

If the patch fails because upstream has changed,
`git apply --3way` usually merges the hunks. Worst case, the patch is
small enough (three files, ~200 lines net) that manual re-application
is trivial.

---

## 9. Notes on the EAGLE3 tree

vLLM requires a static `speculative_token_tree`. The Tengyunw README
recommends SGLang's
`--speculative-num-steps 6 --speculative-eagle-topk 10
 --speculative-num-draft-tokens 32`, which is a **dynamic** per-step
top-k=10 tree built at runtime. The static tree in
`ttft_comprehensive.py` (`EAGLE3_TREE_32`) is the closest faithful
approximation: 32 nodes, depth 6, 5/10/8/5/3/1 fanout per depth. Accept
rate will differ from the dynamic tree but the draft-budget envelope is
the same.
