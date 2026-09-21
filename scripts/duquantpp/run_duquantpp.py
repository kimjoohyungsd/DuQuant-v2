# coding=utf-8
"""DuQuant++ (single greedy block rotation, block = MXFP4 group) under MXFP4 W4A4.

    python scripts/duquantpp/run_duquantpp.py --gpus 0 1
    python scripts/duquantpp/run_duquantpp.py --gptq --gpus 0 1        # DuQuant++*
    python scripts/duquantpp/run_duquantpp.py --report_only

The python replacement for run_duquantpp.sh / run_duquantpp_Qwen3.sh, with
artifacts named per (model, config) instead of per wall-clock second so a
re-run overwrites its own files (see scripts/run_utils.py).

DuQuant++ is --quant_method duquant with permutation_times=0 and block_size=32:
ONE greedy block rotation whose block is exactly the MXFP4 group, so the
rotation flattens precisely the values that will share one E8M0 scale -- as
opposed to legacy duquant's block=128 + zigzag permutation
(scripts/duquant/run_duquant.py). perm=0 is already main.py's default; it is
passed explicitly so the log and the results JSON record it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_utils  # noqa: E402

# --wbits/--abits come from run_utils' shared flags (--wbits/--abits).
BASE = ['--quant_method', 'duquant', '--block_size', '32', '--permutation_times', '0']

SPEC = run_utils.Method(
    name='duquantpp',
    headline='DuQuant++ (single greedy rotation, block=32 = MXFP4 group)',
    configs={
        'duquantpp': (
            'block=32, perm=0, smooth a=0.6  [= run_duquantpp.sh]',
            BASE + ['--smooth', '--alpha', '0.6']),
        'duquantpp_diverse': (
            'as duquantpp + --diverse_rotation (an independent R per group)',
            BASE + ['--smooth', '--alpha', '0.6', '--diverse_rotation']),
        'duquantpp_nosmooth': (
            'block=32, perm=0, NO SmoothQuant -- isolates what --smooth buys',
            list(BASE)),
    },
    order=['duquantpp'],
)

if __name__ == '__main__':
    run_utils.main(SPEC)
