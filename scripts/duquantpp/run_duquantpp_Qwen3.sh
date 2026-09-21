#!/bin/bash

# ============================================================
# DuQuant++ : MXFP4 W4A4 Quantization -- Qwen3 models only
# ============================================================
# Same shape as ../torq/run_torq_Qwen3.sh, minimal argument changes:
#   --quant_method torq  -> duquant  (+ --smooth --alpha 0.6, README.md's
#                                      DuQuant++ recipe -- see
#                                      run_duquantpp.sh's own header for why)
#
# This repo's shared `flatquant` conda env pins a transformers that doesn't
# line up with what Qwen3 needs (native `transformers.models.qwen3`
# support), so Qwen runs need a python whose installed transformers actually
# resolves the Qwen3 architecture. Two ways to provide that, tried in order:
#
#   1) QWEN3_PYTHON env var pointing at a DEDICATED env's python, built from
#      ../../requirements_qwen3.txt:
#          conda create -n duquant-qwen3 python=3.10 -y
#          conda activate duquant-qwen3 && pip install -r requirements_qwen3.txt
#          QWEN3_PYTHON=$(conda run -n duquant-qwen3 which python) \
#              bash scripts/duquantpp/run_duquantpp_Qwen3.sh
#   2) fallback (QWEN3_PYTHON unset): the shared flatquant python, with
#      PYTHONPATH=.vendor/transformers_qwen3 prepended (see
#      ../../.vendor/README.md).
#
# Otherwise identical to run_duquantpp.sh: one GPU per model, full
# stdout+stderr captured per run under log/duquantpp/, plus --results_json.
#
#   bash scripts/duquantpp/run_duquantpp_Qwen3.sh
#   QWEN3_PYTHON=/path/to/duquant-qwen3/bin/python bash scripts/duquantpp/run_duquantpp_Qwen3.sh

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$REPO/log/duquantpp"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HF_HUB_OFFLINE=1

PYTHON_BIN="${QWEN3_PYTHON:-python}"
USE_VENDOR=0
if [[ -z "${QWEN3_PYTHON:-}" && -d "$REPO/.vendor/transformers_qwen3" ]]; then
    USE_VENDOR=1
fi
if [[ "$USE_VENDOR" == "1" ]]; then
    echo "QWEN3_PYTHON not set -- falling back to \$PYTHON_BIN + .vendor/transformers_qwen3 on PYTHONPATH."
    echo "For a real dedicated env instead, see the header of this script and requirements_qwen3.txt."
else
    echo "Using QWEN3_PYTHON=$PYTHON_BIN"
fi

run_one() {
    local gpu="$1"; local model="$2"; shift 2
    local tag="${model##*/}"
    local logf="$LOG_DIR/${tag}_duquantpp_b32_w4a4_${TS}.log"
    local jsonf="$LOG_DIR/${tag}_duquantpp_b32_w4a4_${TS}.json"

    local -a pfx=(env "CUDA_VISIBLE_DEVICES=$gpu")
    if [[ "$USE_VENDOR" == "1" ]]; then
        pfx+=("PYTHONPATH=$REPO/.vendor/transformers_qwen3${PYTHONPATH:+:$PYTHONPATH}")
    fi

    echo ">>> $model  (GPU $gpu)  ->  $logf"
    (cd "$REPO" && "${pfx[@]}" "$PYTHON_BIN" main.py \
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

# --- DuQuant++, Qwen3 models, one GPU each, in parallel ---
run_one 0 Qwen/Qwen3-8B
run_one 1 Qwen/Qwen3-14B
# For DuQuant++* (with GPTQ), add --gptq to the run_one call (see README.md).

wait
echo "all runs finished. logs: $LOG_DIR"
grep -H "wikitext2 :" "$LOG_DIR"/*_"${TS}".log 2>/dev/null || true
