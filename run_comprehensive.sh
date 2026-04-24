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
#   NUM_SPEC          default 15 (matches DFlash block_size=16)
#   OUT_DIR           default ./ttft_out
#   MAX_MODEL_LEN     default 16384
#   TP                default 1
#   DTYPE             default bfloat16
#   TRUST_REMOTE_CODE default 0 (set to 1 to pass --trust-remote-code)

set -uo pipefail

: "${PY:?PY must point to your vLLM venv python (e.g. /path/to/.venv/bin/python)}"
: "${VLLM_REPO:?VLLM_REPO must point to the vLLM clone with the patch applied}"
: "${SHAREGPT_PATH:?SHAREGPT_PATH must be the local path to ShareGPT_V4.3_unfiltered_cleaned_split.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="${BENCH:-$SCRIPT_DIR/ttft_comprehensive.py}"

TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-8B}"
DFLASH_MODEL="${DFLASH_MODEL:-z-lab/Qwen3-8B-DFlash-b16}"
EAGLE3_MODEL="${EAGLE3_MODEL:-Tengyunw/qwen3_8b_eagle3}"
NUM_SPEC="${NUM_SPEC:-15}"
OUT_DIR="${OUT_DIR:-$PWD/ttft_out}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
TP="${TP:-1}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

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
echo "  NUM_SPEC          $NUM_SPEC"
echo "  OUT_DIR           $OUT_DIR"
echo "  MAX_MODEL_LEN     $MAX_MODEL_LEN"
echo "  TP                $TP"
echo "  DTYPE             $DTYPE"
echo "  TRUST_REMOTE_CODE $TRUST_REMOTE_CODE"
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

# no_spec: num-spec is ignored (no drafter), pass any value.
run_mode no_spec          no_spec  6
# eagle3 at the Tengyunw/SGLang recipe (32-node static tree, depth 6).
run_mode eagle3           eagle3   32  "--eagle3-tree"
# dflash with your optimization patch applied (in tracked edits).
run_mode dflash_optimized dflash   "$NUM_SPEC"

echo "=== STASHING tracked edits for dflash_original (upstream PR baseline) ==="
(cd "$VLLM_REPO" && git stash push --quiet \
    --message "dflash-ttft-bench-stash-$(date +%s)")
(cd "$VLLM_REPO" && git status --short)
echo

run_mode dflash_original  dflash   "$NUM_SPEC"

echo "=== RESTORING tracked edits ==="
(cd "$VLLM_REPO" && git stash pop --quiet)
(cd "$VLLM_REPO" && git status --short)
echo

echo "=== ALL MODES DONE ==="
date
echo
echo "Tabulate with:"
echo "  $PY $SCRIPT_DIR/summarize.py --out-dir $OUT_DIR"
