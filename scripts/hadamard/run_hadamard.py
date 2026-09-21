# coding=utf-8
"""Randomized block Hadamard rotation under MXFP4 W4A4.

    python scripts/hadamard/run_hadamard.py --gpus 0 1 2
    python scripts/hadamard/run_hadamard.py --report_only

The python replacement for run_hadamard.sh / run_hadamard_Qwen3.sh, with
artifacts named per (model, config) instead of per wall-clock second so a
re-run overwrites its own files (see scripts/run_utils.py).

--quant_method hadamard applies a FIXED randomized block Hadamard of
--block_size online: no greedy search and no calibration, but the same wrap /
scale setup as duquant (utils.random_hadamard_matrix +
quantizer.init_duquant), so it is the data-independent rotation baseline the
duquant / duquantpp / torq numbers are read against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_utils  # noqa: E402

# --wbits/--abits come from run_utils' shared flags (--wbits/--abits).
BASE = ['--quant_method', 'hadamard', '--permutation_times', '0']

SPEC = run_utils.Method(
    name='hadamard',
    headline='Randomized block Hadamard (fixed, no calibration)',
    configs={
        'hadamard': (
            'block=32 (MXFP4 group), no SmoothQuant  [= run_hadamard.sh]',
            BASE + ['--block_size', '32']),
        'hadamard_smooth': (
            'block=32 + SmoothQuant a=0.6',
            BASE + ['--block_size', '32', '--smooth', '--alpha', '0.6']),
        'hadamard_b128': (
            'block=128 -- rotation spans four MXFP4 groups',
            BASE + ['--block_size', '128']),
    },
    order=['hadamard'],
    # run_hadamard.sh's own three models.
    models=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3.1-8B',
            'meta-llama/Llama-2-13b-hf'],
)

if __name__ == '__main__':
    run_utils.main(SPEC)
