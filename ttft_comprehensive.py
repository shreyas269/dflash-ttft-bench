"""Comprehensive DFlash vs EAGLE3 vs no-spec TTFT benchmark.

Runs 5 serial scenarios + 2 batched scenarios against ONE LLM instance per
mode (amortises load + CUDA-graph capture). Each mode is a separate
subprocess (driven by the companion bash script) to avoid engine
cross-contamination.

Scenarios
---------
1. SERIAL_COLD          N unique ShareGPT prompts, one at a time, prefix
                        cache on but never hits. Pure cold-prefill TTFT.
2. CACHE_HIT_MISS/HIT   K unique prompts, each re-sent R times. First of
                        each group is cold (miss); subsequent are full
                        cache hits. Measures warm-cache TTFT.
3. SHARED_PREFIX_*      Shared ~2K-token prefix + short unique suffix.
                        First req is cold; subsequent are partial-hit.
                        This is the production-shaped hot path.
4. BATCH_4 / BATCH_16   N prompts submitted together; vLLM batches them
                        in-flight. Total wall / n as concurrent TTFT.

Usage
-----
Models default to Qwen3-8B + z-lab/Qwen3-8B-DFlash-b16 +
Tengyunw/qwen3_8b_eagle3. Override via --target / --dflash-model /
--eagle3-model to run against a different target + paired drafters.

See the orchestrator script for the 4-mode sweep.
"""

import argparse
import json
import os
import random
import sys
import time
from statistics import mean

# 32-node static tree of depth 6, approximating the SGLang EAGLE3 config
# "--speculative-num-steps 6 --speculative-eagle-topk 10 --speculative-num-draft-tokens 32"
# recommended in the Tengyunw/qwen3_8b_eagle3 README. SGLang builds the tree
# dynamically from the drafter's top-k probabilities each step; vLLM requires
# a STATIC tree, so this is the closest faithful approximation (5/10/8/5/3/1
# fanout per depth, totalling 32). Exact accept rate will differ from a true
# dynamic top-k=10 tree.
EAGLE3_TREE_32 = [
    (0,), (1,), (2,), (3,), (4,),
    (0, 0), (0, 1), (1, 0), (1, 1), (2, 0),
    (2, 1), (3, 0), (3, 1), (4, 0), (4, 1),
    (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0),
    (2, 0, 0), (2, 1, 0), (3, 0, 0), (3, 1, 0),
    (0, 0, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0),
    (1, 1, 0, 0), (2, 0, 0, 0),
    (0, 0, 0, 0, 0), (0, 1, 0, 0, 0), (1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0),
]
assert len(EAGLE3_TREE_32) == 32, len(EAGLE3_TREE_32)


MODE_KEYS = ("no_spec", "eagle3", "dflash")


def _mode_config(mode, num_spec, *, dflash_model, eagle3_model, use_eagle3_tree):
    if mode == "no_spec":
        return None
    if mode == "eagle3":
        cfg = {"method": "eagle3", "model": eagle3_model}
        if use_eagle3_tree:
            cfg["speculative_token_tree"] = str(EAGLE3_TREE_32)
            cfg["num_speculative_tokens"] = len(EAGLE3_TREE_32)
        else:
            cfg["num_speculative_tokens"] = num_spec
        return cfg
    if mode == "dflash":
        return {
            "method": "dflash",
            "model": dflash_model,
            "num_speculative_tokens": num_spec,
        }
    raise ValueError(f"unknown mode {mode}")


def pctile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def summarize(name, lats):
    if not lats:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": len(lats),
        "min_ms": min(lats) * 1000,
        "p50_ms": pctile(lats, 0.5) * 1000,
        "mean_ms": mean(lats) * 1000,
        "p90_ms": pctile(lats, 0.9) * 1000,
        "p95_ms": pctile(lats, 0.95) * 1000,
        "p99_ms": pctile(lats, 0.99) * 1000,
        "max_ms": max(lats) * 1000,
        "ratio_max_over_p50": max(lats) / pctile(lats, 0.5),
        "iters_ms": [l * 1000 for l in lats],
    }


def print_summary(mode, scenario, lats):
    if not lats:
        print(f"[ttft] SUMMARY mode={mode} scenario={scenario} no data", flush=True)
        return
    s = summarize(scenario, lats)
    print(
        f"[ttft] SUMMARY mode={mode:<14} scenario={scenario:<17} "
        f"n={s['n']:<3} min={s['min_ms']:>7.1f} p50={s['p50_ms']:>7.1f} "
        f"mean={s['mean_ms']:>7.1f} p90={s['p90_ms']:>7.1f} "
        f"p95={s['p95_ms']:>7.1f} p99={s['p99_ms']:>7.1f} "
        f"max={s['max_ms']:>7.1f} max/p50={s['ratio_max_over_p50']:.2f}x (ms)",
        flush=True,
    )


def load_sharegpt_prompts(sharegpt_path, tokenizer, *,
                          min_tokens, max_tokens, num_prompts, seed):
    rng = random.Random(seed)
    with open(sharegpt_path) as f:
        data = json.load(f)
    rng.shuffle(data)

    prompts = []
    for conv in data:
        turns = conv.get("conversations", [])
        if not turns:
            continue
        first = turns[0]
        if first.get("from") != "human":
            continue
        text = (first.get("value") or "").strip()
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(ids) < min_tokens or len(ids) > max_tokens:
            continue
        prompts.append(text)
        if len(prompts) >= num_prompts:
            break
    if len(prompts) < num_prompts:
        raise RuntimeError(
            f"ShareGPT at {sharegpt_path} yielded only {len(prompts)} prompts "
            f"in [{min_tokens},{max_tokens}] tokens (wanted {num_prompts})"
        )
    return prompts


def build_shared_prefix_prompts(sharegpt_path, tokenizer, *,
                                prefix_tokens, suffix_tokens,
                                num_prompts, seed):
    rng = random.Random(seed)
    with open(sharegpt_path) as f:
        data = json.load(f)
    rng.shuffle(data)
    prefix_text = None
    for conv in data:
        turns = conv.get("conversations", [])
        if not turns or turns[0].get("from") != "human":
            continue
        t = (turns[0].get("value") or "").strip()
        ids = tokenizer(t, add_special_tokens=False).input_ids
        if len(ids) >= prefix_tokens:
            prefix_text = tokenizer.decode(
                ids[:prefix_tokens], skip_special_tokens=True
            )
            break
    if prefix_text is None:
        raise RuntimeError(
            f"No ShareGPT turn long enough for prefix_tokens={prefix_tokens}"
        )

    prompts = []
    vocab = min(tokenizer.vocab_size, 120_000)
    for i in range(num_prompts):
        prng = random.Random(seed * 1000 + i)
        suffix_ids = [prng.randrange(256, vocab) for _ in range(suffix_tokens)]
        suffix = tokenizer.decode(suffix_ids, skip_special_tokens=True)
        prompts.append(prefix_text + "\n\n" + suffix)
    return prompts


def measure_serial(llm, prompts):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    lats = []
    for p in prompts:
        tic = time.perf_counter()
        _ = llm.generate([p], sp)
        lats.append(time.perf_counter() - tic)
    return lats


def measure_batch(llm, prompts):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    tic = time.perf_counter()
    _ = llm.generate(prompts, sp)
    total = time.perf_counter() - tic
    return total, total / len(prompts)


def run(args):
    spec_cfg = _mode_config(
        args.mode,
        args.num_spec,
        dflash_model=args.dflash_model,
        eagle3_model=args.eagle3_model,
        use_eagle3_tree=args.eagle3_tree,
    )

    from transformers import AutoTokenizer

    print(f"[ttft] ============ mode={args.mode} ============", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.target, trust_remote_code=args.trust_remote_code
    )

    cold_prompts = load_sharegpt_prompts(
        args.sharegpt_path, tokenizer,
        min_tokens=args.cold_min_tokens, max_tokens=args.cold_max_tokens,
        num_prompts=args.num_cold_prompts, seed=42,
    )
    warmup_prompts = load_sharegpt_prompts(
        args.sharegpt_path, tokenizer,
        min_tokens=1024, max_tokens=4096, num_prompts=3, seed=1001,
    )
    repeat_unique = load_sharegpt_prompts(
        args.sharegpt_path, tokenizer,
        min_tokens=2048, max_tokens=4096, num_prompts=4, seed=123,
    )
    REPEAT_PER_PROMPT = 5
    repeat_prompts = [p for p in repeat_unique for _ in range(REPEAT_PER_PROMPT)]

    shared_prefix_prompts = build_shared_prefix_prompts(
        args.sharegpt_path, tokenizer,
        prefix_tokens=args.shared_prefix_tokens,
        suffix_tokens=args.shared_suffix_tokens,
        num_prompts=15, seed=7777,
    )

    batch4_prompts = load_sharegpt_prompts(
        args.sharegpt_path, tokenizer,
        min_tokens=1024, max_tokens=4096, num_prompts=4, seed=200,
    )
    batch16_prompts = load_sharegpt_prompts(
        args.sharegpt_path, tokenizer,
        min_tokens=1024, max_tokens=4096, num_prompts=16, seed=300,
    )

    llm_kwargs = dict(
        model=args.target,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    if spec_cfg is not None:
        llm_kwargs["speculative_config"] = spec_cfg
    print(
        f"[ttft] mode={args.mode} num_speculative_tokens={args.num_spec} "
        f"speculative_config={spec_cfg}",
        flush=True,
    )

    from vllm import LLM
    llm = LLM(**llm_kwargs)
    results = {}

    try:
        print(f"[ttft] warmup with {len(warmup_prompts)} prompts", flush=True)
        measure_serial(llm, warmup_prompts)

        # SERIAL_COLD
        print(f"[ttft] scenario=SERIAL_COLD n={len(cold_prompts)}", flush=True)
        cold_lats = measure_serial(llm, cold_prompts)
        for i, lat in enumerate(cold_lats):
            print(
                f"[ttft] mode={args.mode:<14} scenario=SERIAL_COLD      "
                f"req={i:>3} ttft_ms={lat*1000:>8.1f}",
                flush=True,
            )
        print_summary(args.mode, "SERIAL_COLD", cold_lats)
        results["SERIAL_COLD"] = summarize("SERIAL_COLD", cold_lats)

        # CACHE_HIT_REPEAT
        print(
            f"[ttft] scenario=CACHE_HIT_REPEAT "
            f"n={len(repeat_prompts)} ({len(repeat_unique)} unique x "
            f"{REPEAT_PER_PROMPT})",
            flush=True,
        )
        hit_lats_all = measure_serial(llm, repeat_prompts)
        miss_lats, hit_lats = [], []
        for i, lat in enumerate(hit_lats_all):
            pos = i % REPEAT_PER_PROMPT
            tag = "miss" if pos == 0 else "hit"
            print(
                f"[ttft] mode={args.mode:<14} scenario=CACHE_HIT_REPEAT "
                f"req={i:>3} ({tag:<4}) ttft_ms={lat*1000:>8.1f}",
                flush=True,
            )
            (miss_lats if pos == 0 else hit_lats).append(lat)
        print_summary(args.mode, "CACHE_HIT_MISS", miss_lats)
        print_summary(args.mode, "CACHE_HIT_HIT", hit_lats)
        results["CACHE_HIT_MISS"] = summarize("CACHE_HIT_MISS", miss_lats)
        results["CACHE_HIT_HIT"] = summarize("CACHE_HIT_HIT", hit_lats)

        # SHARED_PREFIX
        print(
            f"[ttft] scenario=SHARED_PREFIX n={len(shared_prefix_prompts)}",
            flush=True,
        )
        shared_lats = measure_serial(llm, shared_prefix_prompts)
        for i, lat in enumerate(shared_lats):
            tag = "miss" if i == 0 else "hit"
            print(
                f"[ttft] mode={args.mode:<14} scenario=SHARED_PREFIX    "
                f"req={i:>3} ({tag:<4}) ttft_ms={lat*1000:>8.1f}",
                flush=True,
            )
        print_summary(args.mode, "SHARED_PREFIX_FIRST", shared_lats[:1])
        print_summary(args.mode, "SHARED_PREFIX_LATER", shared_lats[1:])
        results["SHARED_PREFIX_FIRST"] = summarize(
            "SHARED_PREFIX_FIRST", shared_lats[:1]
        )
        results["SHARED_PREFIX_LATER"] = summarize(
            "SHARED_PREFIX_LATER", shared_lats[1:]
        )

        # BATCH_4
        print(f"[ttft] scenario=BATCH_4 n={len(batch4_prompts)}", flush=True)
        total_b4, per_b4 = measure_batch(llm, batch4_prompts)
        print(
            f"[ttft] SUMMARY mode={args.mode:<14} scenario=BATCH_4           "
            f"total_ms={total_b4*1000:.1f} per_req_ms={per_b4*1000:.1f}",
            flush=True,
        )
        results["BATCH_4"] = {
            "name": "BATCH_4",
            "n": len(batch4_prompts),
            "total_ms": total_b4 * 1000,
            "per_req_ms": per_b4 * 1000,
        }

        # BATCH_16
        print(f"[ttft] scenario=BATCH_16 n={len(batch16_prompts)}", flush=True)
        total_b16, per_b16 = measure_batch(llm, batch16_prompts)
        print(
            f"[ttft] SUMMARY mode={args.mode:<14} scenario=BATCH_16          "
            f"total_ms={total_b16*1000:.1f} per_req_ms={per_b16*1000:.1f}",
            flush=True,
        )
        results["BATCH_16"] = {
            "name": "BATCH_16",
            "n": len(batch16_prompts),
            "total_ms": total_b16 * 1000,
            "per_req_ms": per_b16 * 1000,
        }

        return results
    finally:
        from vllm.distributed import cleanup_dist_env_and_memory
        del llm
        cleanup_dist_env_and_memory()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=list(MODE_KEYS), required=True)
    p.add_argument(
        "--target", default="Qwen/Qwen3-8B",
        help="Target model HF id or local path."
    )
    p.add_argument(
        "--dflash-model", default="z-lab/Qwen3-8B-DFlash-b16",
        help="DFlash drafter HF id or local path (paired with --target)."
    )
    p.add_argument(
        "--eagle3-model", default="Tengyunw/qwen3_8b_eagle3",
        help="EAGLE3 drafter HF id or local path (paired with --target)."
    )
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--num-spec", type=int, default=15,
        help=(
            "num_speculative_tokens. Default 15 pairs with z-lab Qwen3-8B-"
            "DFlash-b16 (block_size=16 = 1 bonus + 15 mask). Match to your "
            "DFlash drafter's training block_size. Ignored for EAGLE3 when "
            "--eagle3-tree is set."
        ),
    )
    p.add_argument(
        "--eagle3-tree", action="store_true",
        help=(
            "For --mode=eagle3, use a 32-node depth-6 static tree that "
            "approximates SGLang's recommended --speculative-num-steps 6 "
            "--speculative-eagle-topk 10 --speculative-num-draft-tokens 32."
        ),
    )
    p.add_argument(
        "--sharegpt-path", required=True,
        help="Path to ShareGPT_V4.3_unfiltered_cleaned_split.json"
    )
    p.add_argument("--num-cold-prompts", type=int, default=20)
    p.add_argument("--cold-min-tokens", type=int, default=1024)
    p.add_argument("--cold-max-tokens", type=int, default=8192)
    p.add_argument("--shared-prefix-tokens", type=int, default=2000)
    p.add_argument("--shared-suffix-tokens", type=int, default=400)
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    print(
        f"[ttft] comprehensive test mode={args.mode} target={args.target} "
        f"num_spec={args.num_spec}"
    )
    res = run(args)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                {
                    "mode": args.mode,
                    "target": args.target,
                    "dflash_model": args.dflash_model,
                    "eagle3_model": args.eagle3_model,
                    "num_spec": args.num_spec,
                    "eagle3_tree": args.eagle3_tree,
                    "scenarios": res,
                },
                f, indent=2, default=str,
            )
        print(f"[ttft] dumped -> {args.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
