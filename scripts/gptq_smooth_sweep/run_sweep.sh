#!/bin/bash
# ============================================================
# GPTQ + Hadamard (block=32), No-Smooth vs. alpha-searched Smooth
# 5 models: Llama-2-7b/13b, Llama-3.1-8B, Qwen3-8B/14B
# ============================================================
# Thin wrapper around run_sweep.py. Kept as its own script (repo convention:
# every scripts/<method>/ has both a .py runner and a .sh entry point) and to
# make the Qwen3 prerequisite ORDER explicit and impossible to skip by
# accident:
#
#   1) conda create -n duquant-qwen3 (if missing)
#   2) pip install -r requirements_qwen3.txt   INTO THAT ENV, not the shared
#      `flatquant` env -- it pins an older transformers that predates the
#      qwen3 architecture (see requirements_qwen3.txt's own header).
#   3) only then does any Qwen main.py subprocess get launched.
#
# run_sweep.py's --setup_qwen_env flag does exactly steps 1-2 itself
# (idempotently) before touching any GPU job; this script just always passes
# it so that ordering can never be forgotten from this entry point.

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

# Force the pure-python protobuf parser (Llama's slow tokenizer trips a
# protobuf>=4 incompatibility in this env otherwise).
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python scripts/gptq_smooth_sweep/run_sweep.py \
    --setup_qwen_env \
    "$@"
