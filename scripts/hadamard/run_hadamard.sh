REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/log/hadamard"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

# Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility in
# this env; force the pure-python parser so it loads.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1


cleanup() {
    echo ""
    echo "!!! Keyboard Interrupt detected. Terminating python processes... !!!"
    pkill -P $$
    exit 1
}
trap cleanup SIGINT


run_one() {
    local gpu="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_hadamard_b32perm0_w4a4_${TS}.log"
    local jsonf="$LOG_DIR/${tag}_hadamard_b32perm0_w4a4_${TS}.json"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpu")

    echo ">>> $model  (GPU $gpu)  ->  $logf"
    (cd "$REPO" && "${pfx[@]}" python main.py \
        --block_size 32 \
        --permutation_times 0 \
        --quant_method hadamard \
        --wbits 4 \
        --abits 4 \
        --model "$model" \
        --eval_ppl \
        --batch_size 1 \
        --results_json "$jsonf" \
        "$@" \
        > "$logf" 2>&1) &
}

# --- DuQuant (hadamard rotate), one model per GPU, in parallel ---
run_one 5 meta-llama/Llama-2-7b-hf
run_one 6 meta-llama/Llama-3.1-8B
run_one 7 meta-llama/Llama-2-13b-hf
# Qwen3: bash scripts/duquant/run_duquant_Qwen3.sh
# For the GPTQ variant, add --gptq to the run_one call (see README.md).

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true