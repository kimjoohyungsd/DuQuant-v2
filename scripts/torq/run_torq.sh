#!/bin/bash

# ============================================================
# TORQ (arXiv:2605.19561) : MXFP4 W4A4 Quantization -- --quant_method torq
# ============================================================
# Llama/dense models only -- for Qwen3, use run_torq_Qwen3.sh instead (this
# repo's shared `flatquant` env's transformers doesn't line up with what
# Qwen3 needs, so those runs need either a dedicated env or the vendored
# transformers on PYTHONPATH; see that script's header and
# ../../requirements_qwen3.txt).
#
# Same shape as ../../run.sh (hadamard), one directory level deeper. Each
# model runs on its own SINGLE GPU (not --multigpu: that path mismatches
# devices for the shared rotary embedding once decoder layers span more than
# one GPU under this env's transformers -- see run.sh's own history), in
# parallel, with the full stdout+stderr stream captured to its own file
# under log/torq/, plus a machine-readable --results_json per run (the other
# scripts/*.py ablation runners' pattern) so results are comparable against
# log/mxfp4_ablation/ (duquant) and log/hadamard/ (hadamard) without
# re-parsing logs.
#
#   bash scripts/torq/run_torq.sh
#   bash scripts/torq/run_torq.sh 2>&1 | tee /tmp/torq_console.log   # also mirror to terminal

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/log/torq"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility in
# this env; force the pure-python parser so it loads.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1

run_one() {
    local gpu="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_torq_b32_w4a4_${TS}.log"
    local jsonf="$LOG_DIR/${tag}_torq_b32_w4a4_${TS}.json"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpu")

    echo ">>> $model  (GPU $gpu)  ->  $logf"
    (cd "$REPO" && "${pfx[@]}" python main.py \
        --block_size 32 \
        --quant_method torq \
        --wbits 4 \
        --abits 4 \
        --model "$model" \
        --eval_ppl \
        --batch_size 1 \
        --results_json "$jsonf" \
        "$@" \
        > "$logf" 2>&1) &
}

# --- TORQ, one model per GPU, in parallel ---
run_one 0 meta-llama/Llama-2-7b-hf
run_one 1 meta-llama/Llama-3.1-8B
# Qwen3: bash scripts/torq/run_torq_Qwen3.sh
# TORQ-specific knobs (defaults shown; see main.py --help for the rest):
#   --torq_eps_inter 1e-4 --torq_max_iter_inter 2000
#   --torq_max_iter_intra 10 --torq_k_top_frac 0.5 --torq_num_pairs <K_top/2>
#   --torq_lambda 1.0 --torq_max_samples 8192

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true
