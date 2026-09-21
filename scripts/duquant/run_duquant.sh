#!/bin/bash

# ============================================================
# DuQuant (rotate -> zigzag permute -> rotate, block_size 32) : W4A4 MXFP4
# ============================================================
# Same shape as ../duquantpp/run_duquantpp.sh, ONE argument added:
#   + --permutation_times 1
#
# --permutation_times N makes UniformAffineQuantizer.online_duquant_cali
# (quantize/quantizer.py) run N rounds of {greedy block rotation, zigzag
# permutation} followed by one FINAL greedy block rotation. So
# --permutation_times 1 at --block_size 32 is exactly
#   block_rotation(32) -> zigzag_permutation -> block_rotation(32)
# -- the legacy DuQuant (NeurIPS'24) "rot->zigzag perm->rot" recipe (the
# scripts/diverse_rotation_ablation.py config named "duquant_shared", there
# run at block=128; here at block=32, MXFP4-group-aligned, same as
# duquantpp). Everything else (--smooth --alpha 0.6 etc.) is unchanged from
# duquantpp.
#
# Llama/dense models only -- for Qwen3, use run_duquant_Qwen3.sh instead
# (this repo's shared `flatquant` env's transformers doesn't line up with
# what Qwen3 needs; see that script's header and ../../requirements_qwen3.txt).
#
# Each model runs on its own SINGLE GPU (not --multigpu: see ../../run.sh's
# own history), in parallel, full stdout+stderr captured per run under
# log/duquant/, plus --results_json so results are comparable against
# log/mxfp4_ablation/, log/hadamard/, log/torq/ and log/duquantpp/ without
# re-parsing logs.
#
#   bash scripts/duquant/run_duquant.sh
#   bash scripts/duquant/run_duquant.sh 2>&1 | tee /tmp/duquant_console.log

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/log/duquant"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility in
# this env; force the pure-python parser so it loads.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1

run_one() {
    local gpu="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_duquant_b32perm1_w4a4_${TS}.log"
    local jsonf="$LOG_DIR/${tag}_duquant_b32perm1_w4a4_${TS}.json"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpu")

    echo ">>> $model  (GPU $gpu)  ->  $logf"
    (cd "$REPO" && "${pfx[@]}" python main.py \
        --block_size 32 \
        --permutation_times 1 \
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

# --- DuQuant (rotate -> zigzag -> rotate, block=32), one model per GPU, in parallel ---
run_one 0 meta-llama/Llama-2-7b-hf
run_one 1 meta-llama/Llama-3.1-8B
# Qwen3: bash scripts/duquant/run_duquant_Qwen3.sh
# For the GPTQ variant, add --gptq to the run_one call (see README.md).

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true
