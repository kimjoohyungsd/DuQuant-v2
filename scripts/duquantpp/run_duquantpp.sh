#!/bin/bash

# ============================================================
# DuQuant++ : MXFP4 W4A4 Quantization -- --quant_method duquant, block_size 32
# ============================================================
# Same shape as ../torq/run_torq.sh, minimal argument changes:
#   --quant_method torq  -> duquant           (greedy block rotation search,
#                                               README.md's DuQuant++ recipe,
#                                               not TORQ's calibrated rotation)
#   --torq_* flags       -> --smooth --alpha 0.6   (README.md's own example)
#   (--permutation_times 0 and --max_rotation_step 256 are already main.py's
#    defaults, so left implicit -- this is "duquantpp" in
#    scripts/diverse_rotation_ablation.py's own naming: block=32, perm=0,
#    as opposed to legacy "duquant" = block=128, perm=1.)
#
# Llama/dense models only -- for Qwen3, use run_duquantpp_Qwen3.sh instead
# (this repo's shared `flatquant` env's transformers doesn't line up with
# what Qwen3 needs; see that script's header and ../../requirements_qwen3.txt).
#
# Each model runs on its own SINGLE GPU (not --multigpu: see ../../run.sh's
# own history), in parallel, full stdout+stderr captured per run under
# log/duquantpp/, plus --results_json so results are comparable against
# log/mxfp4_ablation/ (legacy duquant), log/hadamard/ and log/torq/ without
# re-parsing logs.
#
#   bash scripts/duquantpp/run_duquantpp.sh
#   bash scripts/duquantpp/run_duquantpp.sh 2>&1 | tee /tmp/duquantpp_console.log

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/log/duquantpp"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility in
# this env; force the pure-python parser so it loads.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1

run_one() {
    local gpu="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_duquantpp_b32_w4a4_${TS}.log"
    local jsonf="$LOG_DIR/${tag}_duquantpp_b32_w4a4_${TS}.json"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpu")

    echo ">>> $model  (GPU $gpu)  ->  $logf"
    (cd "$REPO" && "${pfx[@]}" python main.py \
        --block_size 32 \
        --quant_method duquant \
        --wbits 4 \
        --abits 4 \
        --model "$model" \
        --alpha 0.6 \
        --smooth \
        --eval_ppl \
        --batch_size 1 \
        --results_json "$jsonf" \
        "$@" \
        > "$logf" 2>&1) &
}

# --- DuQuant++ (without GPTQ), one model per GPU, in parallel ---
run_one 0 meta-llama/Llama-2-7b-hf
run_one 1 meta-llama/Llama-3.1-8B
# Qwen3: bash scripts/duquantpp/run_duquantpp_Qwen3.sh
# For DuQuant++* (with GPTQ), add --gptq to the run_one call (see README.md).

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true
