# coding=utf-8
"""TORQ (arXiv:2605.19561) two-level calibrated rotation under MXFP4 W4A4.

    python scripts/torq/run_torq.py --gpus 0 1
    python scripts/torq/run_torq.py --configs torq -- --torq_lambda 2.0
    python scripts/torq/run_torq.py --report_only

The python replacement for run_torq.sh, run_torq_smooth.sh and their _Qwen3
variants -- the plain and +SmoothQuant arms are two cells of one table here
instead of two scripts, and artifacts are named per (model, config) instead of
per wall-clock second, so a re-run overwrites its own files (see
scripts/run_utils.py).

--quant_method torq CALIBRATES the rotation from each layer's activations:
R_inter (Macro-Equilibrium, equalizes per-block energy across the --block_size
blocks) then R_intra (Micro-Alignment, spreads values across the 8 MXFP4
codewords within a block); see quantize/torq.py. In the torq_smooth cell
--smooth composes on top: QuantLinear's per-channel SmoothQuant scale is
applied in UniformAffineQuantizer.forward BEFORE init_duquant's dispatch, so
TORQ calibrates on already-smoothed activations.

TORQ's own knobs keep main.py's defaults; override any of them by appending
them after a `--` separator, e.g.
    python scripts/torq/run_torq.py -- --torq_k_top_frac 0.25 --torq_max_samples 4096
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_utils  # noqa: E402

# --wbits/--abits come from run_utils' shared flags (--wbits/--abits).
BASE = ['--quant_method', 'torq', '--block_size', '32']

SPEC = run_utils.Method(
    name='torq',
    headline='TORQ two-level calibrated rotation (R_inter then R_intra)',
    configs={
        'torq': (
            'block=32 (MXFP4 group), no SmoothQuant  [= run_torq.sh]',
            list(BASE)),
        'torq_smooth': (
            'block=32 + SmoothQuant a=0.6  [= run_torq_smooth.sh]',
            BASE + ['--smooth', '--alpha', '0.6']),
        'torq_b128': (
            'block=128 -- R_inter equalizes across four MXFP4 groups',
            ['--quant_method', 'torq', '--block_size', '128']),
    },
    # Both arms the shell scripts ran; torq_b128 is opt-in via --configs.
    order=['torq', 'torq_smooth'],
)

if __name__ == '__main__':
    run_utils.main(SPEC)
