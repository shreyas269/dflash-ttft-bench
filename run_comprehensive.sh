#!/usr/bin/env bash
# Orchestrator: runs the 4-mode TTFT sweep (no_spec, eagle3, dflash_optimized,
# dflash_original). Each mode is a separate subprocess to avoid vLLM engine
# cross-contamination.
#
# - eagle3 runs with a 32-node depth-6 static tree approximating the SGLang
#   recipe --speculative-num-steps 6 --speculative-eagle-topk 10
#   --speculative-num-draft-tokens 32 (vLLM requires static trees).
# - dflash_optimized runs with your optimized vLLM working tree (patch applied).
# - dflash_original runs with the optimization `git stash`-ed away — i.e. it
#   runs the upstream PR #36847 code. This is the apples-to-apples baseline.
# - no_spec is the target-only upper bound.
#
# Required env vars:
#   PY                Python binary in the venv that has your vLLM built
#                     (e.g. /path/to/.venv/bin/python)
#   VLLM_REPO         Path to the vLLM clone that has the optimization patch
#                     applied as tracked edits (not committed)
#   SHAREGPT_PATH     Path to ShareGPT_V4.3_unfiltered_cleaned_split.json
#
# Optional env vars (override for a different target):
#   TARGET_MODEL      default Qwen/Qwen3-8B
#   DFLASH_MODEL      default z-lab/Qwen3-8B-DFlash-b16
#   EAGLE3_MODEL      default Tengyunw/qwen3_8b_eagle3
#
#   DFLASH_NUM_SPEC   default 15 (matches DFlash block_size=16 for z-lab b16
#                     drafter: 1 bonus + 15 mask = 16 query slots). Set this
#                     to `block_size - 1` of whatever DFlash drafter you use.
#                     Legacy alias: NUM_SPEC (used if DFLASH_NUM_SPEC unset).
#
#   EAGLE3_USE_TREE   default 1. When 1, EAGLE3 runs with the 32-node depth-6
#                     static tree approximating SGLang's recommended config
#                     (--speculative-num-steps 6 --speculative-eagle-topk 10
#                     --speculative-num-draft-tokens 32). Set to 0 to run
#                     EAGLE3 as a plain chain instead.
#   EAGLE3_NUM_SPEC   default 6. Only used when EAGLE3_USE_TREE=0. Typical
#                     EAGLE3 chain depth is 4-6.
#
#   OUT_DIR           default ./ttft_out
#   MAX_MODEL_LEN     default 16384
#   TP                default 1
#   DTYPE             default bfloat16
#   TRUST_REMOTE_CODE default 0 (set to 1 to pass --trust-remote-code)
#
# The script is also skip-friendly:
#   SKIP_NO_SPEC=1      skip the no-spec baseline
#   SKIP_EAGLE3=1       skip the EAGLE3 mode (use when no EAGLE3 drafter
#                       for your target exists)
#   SKIP_DFLASH_OPT=1   skip dflash_optimized
#   SKIP_DFLASH_ORIG=1  skip dflash_original (and the git-stash dance)

set -uo pipefail

: "${PY:?PY must point to your vLLM venv python (e.g. /path/to/.venv/bin/python)}"
: "${VLLM_REPO:?VLLM_REPO must point to the vLLM clone with the patch applied}"
: "${SHAREGPT_PATH:?SHAREGPT_PATH must be the local path to ShareGPT_V4.3_unfiltered_cleaned_split.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$SCRIPT_DIR/ttft_comprehensive.py}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DFLASH_MODEL="${DFLASH_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
EAGLE3_MODEL="${EAGLE3_MODEL:-Tengyunw/qwen3_8b_eagle3}"

# Per-method speculation budgets. DFlash must match its drafter's trained
# block_size; EAGLE3's depth is independent and typically smaller.
DFLASH_NUM_SPEC="${DFLASH_NUM_SPEC:-${NUM_SPEC:-15}}"
EAGLE3_USE_TREE="${EAGLE3_USE_TREE:-1}"
EAGLE3_NUM_SPEC="${EAGLE3_NUM_SPEC:-6}"

OUT_DIR="${OUT_DIR:-$PWD/ttft_out}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

SKIP_NO_SPEC="${SKIP_NO_SPEC:-0}"
SKIP_EAGLE3="${SKIP_EAGLE3:-0}"
SKIP_DFLASH_OPT="${SKIP_DFLASH_OPT:-0}"
SKIP_DFLASH_ORIG="${SKIP_DFLASH_ORIG:-0}"

TRC_FLAG=""
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    TRC_FLAG="--trust-remote-code"
fi

mkdir -p "$OUT_DIR"

echo "=== config ==="
echo "  PY                $PY"
echo "  VLLM_REPO         $VLLM_REPO"
echo "  BENCH             $BENCH"
echo "  SHAREGPT_PATH     $SHAREGPT_PATH"
echo "  TARGET_MODEL      $TARGET_MODEL"
echo "  DFLASH_MODEL      $DFLASH_MODEL"
echo "  EAGLE3_MODEL      $EAGLE3_MODEL"
echo "  DFLASH_NUM_SPEC   $DFLASH_NUM_SPEC"
echo "  EAGLE3_USE_TREE   $EAGLE3_USE_TREE  (0 = chain, 1 = 32-node static tree)"
echo "  EAGLE3_NUM_SPEC   $EAGLE3_NUM_SPEC  (only used when EAGLE3_USE_TREE=0)"
echo "  OUT_DIR           $OUT_DIR"
echo "  MAX_MODEL_LEN     $MAX_MODEL_LEN"
echo "  TP                $TP"
echo "  DTYPE             $DTYPE"
echo "  TRUST_REMOTE_CODE $TRUST_REMOTE_CODE"
echo "  SKIP              no_spec=$SKIP_NO_SPEC eagle3=$SKIP_EAGLE3 dflash_opt=$SKIP_DFLASH_OPT dflash_orig=$SKIP_DFLASH_ORIG"
echo

run_mode() {
    local mode_tag="$1"
    local python_mode="$2"
    local num_spec="$3"
    local extra_flags="${4:-}"
    local log="$OUT_DIR/${mode_tag}.log"
    local json="$OUT_DIR/${mode_tag}.json"

    echo "=== RUNNING ${mode_tag} (python_mode=${python_mode}, num_spec=${num_spec}, extra=${extra_flags}) ==="
    date

    # Wipe stale __pycache__ so the right code path is loaded (important
    # across the git-stash boundary).
    find "$VLLM_REPO/vllm" -name __pycache__ -type d -prune -exec rm -rf {} + \
        2>/dev/null || true

    (
        cd "$OUT_DIR" && \
        VLLM_LOGGING_LEVEL=WARNING \
        "$PY" "$BENCH" \
            --mode "$python_mode" \
            --target "$TARGET_MODEL" \
            --dflash-model "$DFLASH_MODEL" \
            --eagle3-model "$EAGLE3_MODEL" \
            --sharegpt-path "$SHAREGPT_PATH" \
            --num-spec "$num_spec" \
            --max-model-len "$MAX_MODEL_LEN" \
            --tensor-parallel-size "$TP" \
            --dtype "$DTYPE" \
            $TRC_FLAG \
            $extra_flags \
            --json-out "$json"
    ) > "$log" 2>&1
    local rc=$?

    echo "--- tail ${mode_tag} rc=$rc ---"
    grep -E "SUMMARY|Traceback|Error|CUDA out|assert|Killed|num_spec|tree" \
        "$log" | tail -40
    echo
}

echo "--- git state before run ---"
(cd "$VLLM_REPO" && git status --short)
echo

# Build EAGLE3 invocation based on tree vs chain choice.
if [[ "$EAGLE3_USE_TREE" == "1" ]]; then
    # Tree mode: num_speculative_tokens is ignored by the Python driver
    # (tree size is the budget); pass the tree length (32) for consistency
    # with log output.
    EAGLE3_SPEC_BUDGET=32
    EAGLE3_EXTRA="--eagle3-tree"
else
    EAGLE3_SPEC_BUDGET="$EAGLE3_NUM_SPEC"
    EAGLE3_EXTRA=""
fi

# no_spec: num-spec is ignored (no drafter); pass any value.
if [[ "$SKIP_NO_SPEC" != "1" ]]; then
    run_mode no_spec          no_spec  6
fi
if [[ "$SKIP_EAGLE3" != "1" ]]; then
    run_mode eagle3           eagle3   "$EAGLE3_SPEC_BUDGET"  "$EAGLE3_EXTRA"
fi
if [[ "$SKIP_DFLASH_OPT" != "1" ]]; then
    run_mode dflash_optimized dflash   "$DFLASH_NUM_SPEC"
fi

if [[ "$SKIP_DFLASH_ORIG" != "1" ]]; then
    echo "=== STASHING tracked edits for dflash_original (upstream PR baseline) ==="
    (cd "$VLLM_REPO" && git stash push --quiet \
        --message "dflash-ttft-bench-stash-$(date +%s)")
    (cd "$VLLM_REPO" && git status --short)
    echo

    run_mode dflash_original  dflash   "$DFLASH_NUM_SPEC"

    echo "=== RESTORING tracked edits ==="
    (cd "$VLLM_REPO" && git stash pop --quiet)
    (cd "$VLLM_REPO" && git status --short)
    echo
fi

echo "=== ALL MODES DONE ==="
date
echo
echo "Tabulate with:"
echo "  $PY $SCRIPT_DIR/summarize.py --out-dir $OUT_DIR"
