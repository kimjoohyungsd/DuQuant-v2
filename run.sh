#!/bin/bash

# ============================================================
# DuQuant++ : MXFP4 W4A4 Quantization  --  --quant_method hadamard
# ============================================================
# Each model runs on its own GPU pair, in parallel, with the full
# stdout+stderr stream captured to its own file under log/hadamard/.
# (main.py also writes its own logger file into --output_dir; this is
#  the raw console stream -- tqdm bars, warnings, tracebacks, the final
#  "wikitext2 : <ppl>" line.)

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$REPO/log/hadamard"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility in
# this env; force the pure-python parser so it loads.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1

run_one() {
    local gpus="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_hadamard_b32_w4a4_${TS}.log"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpus")
    # Qwen3 needs the vendored (newer) transformers on PYTHONPATH so the
    # site-packages one doesn't shadow it (see scripts/env_utils.py).
    if [[ "$model" == *[Qq]wen* && -d "$REPO/.vendor/transformers_qwen3" ]]; then
        pfx+=("PYTHONPATH=$REPO/.vendor/transformers_qwen3${PYTHONPATH:+:$PYTHONPATH}")
    fi

    echo ">>> $model  (GPU $gpus)  ->  $logf"
    "${pfx[@]}" python main.py \
        --block_size 32 \
        --quant_method hadamard \
        --wbits 4 \
        --abits 4 \
        --model "$model" \
        --alpha 0.6 \
        --eval_ppl \
        --batch_size 32 \
        --multigpu \
        "$@" \
        > "$logf" 2>&1 &
}

# --- DuQuant++ (without GPTQ), quant_method = hadamard ---
# run_one 0,1 meta-llama/Llama-3.1-8B
# run_one 2,3 meta-llama/Llama-2-7b-hf
run_one  0,1,2,3,4,5,6,7 Qwen/Qwen3-14B
# run_one  3,4,5 Qwen/Qwen3-8B
# run_one 4,5 Qwen/Qwen3-14B --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true


# --- DuQuant++* (with GPTQ) ---
# python main.py \
#     --block_size 32 \
#     --max_rotation_step 128 \
#     --wbits 4 \
#     --abits 4 \
#     --model meta-llama/Llama-3-8B \
#     --alpha 0.6 \
#     --gptq \
#     --smooth \
#     --eval_ppl \
#     --batch_size 32 \
#     --tasks arc_easy,arc_challenge,winogrande,hellaswag,openbookqa,lambada_openai,piqa
