# coding=utf-8
"""DuQuant (NeurIPS'24 recipe: rotate -> zigzag permute -> rotate) under MXFP4 W4A4.

    python scripts/duquant/run_duquant.py --gpus 0 1
    python scripts/duquant/run_duquant.py --models Qwen/Qwen3-8B Qwen/Qwen3-14B --gpus 0 1
    python scripts/duquant/run_duquant.py --report_only

The python replacement for run_duquant.sh / run_duquant_Qwen3.sh. Same main.py
flags; what changes is that artifacts are named per (model, config) instead of
per wall-clock second, so a re-run overwrites its own files rather than leaving
another timestamped generation in log/duquant/ (see scripts/run_utils.py).

--permutation_times N makes UniformAffineQuantizer.online_duquant_cali
(quantize/quantizer.py) run N rounds of {greedy block rotation, zigzag
permutation} then one FINAL greedy block rotation, so perm=1 at block=32 is
exactly block_rotation(32) -> zigzag_permutation -> block_rotation(32) -- the
legacy DuQuant recipe at MXFP4-group-aligned width instead of the paper's 128.

Qwen3 needs a transformers that resolves the Qwen3 architecture: either point
--qwen_python (or $QWEN3_PYTHON) at an env built from requirements_qwen3.txt,
or leave it unset and the vendored .vendor/transformers_qwen3 goes on PYTHONPATH.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_utils  # noqa: E402

# --wbits/--abits come from run_utils' shared flags (--wbits/--abits).
BASE = ['--quant_method', 'duquant']

SPEC = run_utils.Method(
    name='duquant',
    headline='DuQuant (rot -> zigzag perm -> rot), --quant_method duquant',
    configs={
        'duquant': (
            'block=32 (MXFP4 group), perm=1, smooth a=0.6  [= run_duquant.sh]',
            BASE + ['--block_size', '32', '--permutation_times', '1',
                    '--smooth', '--alpha', '0.6']),
        'duquant_b128': (
            'block=128 (paper width, spans 4 MXFP4 groups), perm=1, smooth a=0.6',
            BASE + ['--block_size', '128', '--permutation_times', '1',
                    '--smooth', '--alpha', '0.6']),
        'duquant_nosmooth': (
            'block=32, perm=1, NO SmoothQuant -- isolates what --smooth buys',
            BASE + ['--block_size', '32', '--permutation_times', '1']),
    },
    # A bare run reproduces run_duquant.sh; the other cells are opt-in via
    # --configs duquant duquant_b128 ...
    order=['duquant'],
)

if __name__ == '__main__':
    run_utils.main(SPEC)
