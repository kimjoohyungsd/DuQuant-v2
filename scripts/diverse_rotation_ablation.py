# coding=utf-8
"""Shared block rotation (legacy DuQuant) vs. per-group "diverse" block rotation.

    python scripts/diverse_rotation_ablation.py --models meta-llama/Llama-2-7b-hf
    python scripts/diverse_rotation_ablation.py --report_only

quantize/quantizer.py's greedy block rotation searches ONE [block_size, block_size]
orthogonal R and reuses it block-diagonally for every group along the hidden dim
(hidden_dim // block_size groups share the exact same matrix). `--diverse_rotation`
(added in quantize/quantizer.py) instead runs an independent greedy search per
group, so R becomes [num_blocks, block_size, block_size] -- a genuine block-diagonal
rotation instead of one block repeated num_blocks times. Both W and its matching
activation are rotated with the same R (see quantize/int_linear.py:
weight_quantizer.copy_duquant_params(act_quantizer)), so X@W^T is preserved
pre-quantization in both modes -- see /tmp/.../test_diverse_rotation.py.

Four cells per model, crossing {shared, diverse} x {duquantpp block=32 perm=0,
duquant block=128 perm=1}:

  duquantpp_shared    block=32,  perm=0  -- one greedy rotation per MXFP4 group,
  duquantpp_diverse   block=32,  perm=0     shared vs. one-per-group (the
                                             [128,32,32] example: hidden 4096 /
                                             block 32 = 128 groups).
  duquant_shared      block=128, perm=1  -- the NeurIPS'24 recipe (rot->zigzag
  duquant_diverse     block=128, perm=1     perm->rot), shared vs. one-per-group
                                             (hidden 4096 / block 128 = 32 groups).

`_shared` cells reuse scripts/mxfp4_ablation.py's already-computed duquantpp.json /
duquant.json when present (same config, so identical result) instead of re-running.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_utils import subprocess_env  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name: (description, block_size, permutation_times, diverse_rotation, legacy_name)
# legacy_name: matching config name in log/mxfp4_ablation/<model>/<legacy_name>.json
# that can be reused verbatim for `shared` cells (identical config).
CONFIGS = {
    'duquantpp_shared':  ('block=32 perm=0, ONE shared [32,32] R for all 128 groups',
                          32, 0, False, 'duquantpp'),
    'duquantpp_diverse': ('block=32 perm=0, INDEPENDENT R per group -> [128,32,32]',
                          32, 0, True, None),
    'duquant_shared':    ('block=128 perm=1, ONE shared [128,128] R for all 32 groups',
                          128, 1, False, 'duquant'),
    'duquant_diverse':   ('block=128 perm=1, INDEPENDENT R per group -> [32,128,128]',
                          128, 1, True, None),
}
DEFAULT_ORDER = ['duquantpp_shared', 'duquantpp_diverse', 'duquant_shared', 'duquant_diverse']


def short(model):
    return model.rstrip('/').split('/')[-1]


def result_path(out_root, model, name):
    return os.path.join(out_root, short(model), f'{name}.json')


def legacy_path(model, legacy_name):
    """Where an equivalent already-computed result might live.

    scripts/mxfp4_ablation.py writes log/mxfp4_ablation/<model>/<name>.json.
    The earlier Qwen sweep (run by hand, same config, before this script
    existed) instead wrote log/qwen/<model>_<name>.json. Only a config match
    should be reused, so callers must confirm block_size/permutation_times
    themselves -- this just returns candidate paths in preference order.
    """
    candidates = [
        os.path.join(HERE, 'log', 'mxfp4_ablation', short(model), f'{legacy_name}.json'),
        os.path.join(HERE, 'log', 'qwen', f'{short(model)}_{legacy_name}.json'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def run_one(args, model, name):
    desc, block, perm, diverse, legacy_name = CONFIGS[name]
    out_json = result_path(args.out_root, model, name)
    log_dir = os.path.join(args.out_root, short(model), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if not diverse and legacy_name is not None and args.reuse_existing:
        src = legacy_path(model, legacy_name)
        if os.path.exists(src):
            with open(src) as f:
                src_cfg = json.load(f)
            # Only reuse if the config actually matches -- e.g. the earlier
            # by-hand Qwen sweep's "duquant" used block_size=32, not the 128
            # this script's duquant_shared expects, so it must NOT be reused
            # under that name.
            if (src_cfg.get('block_size') == block
                    and src_cfg.get('permutation_times') == perm
                    and not src_cfg.get('diverse_rotation', False)):
                os.makedirs(os.path.dirname(out_json), exist_ok=True)
                shutil.copy(src, out_json)
                print(f'[{short(model)} / {name}] reused existing {src} (identical config, not re-run)')
                return
            else:
                print(f'[{short(model)} / {name}] found {src} but config differs '
                      f'(block_size={src_cfg.get("block_size")} vs {block}, '
                      f'permutation_times={src_cfg.get("permutation_times")} vs {perm}) '
                      f'-- running fresh instead of reusing it')

    cmd = [
        args.python, 'main.py',
        '--model', model,
        '--wbits', '4', '--abits', '4',
        '--block_size', str(block),
        '--permutation_times', str(perm),
        '--max_rotation_step', str(args.max_rotation_step),
        '--alpha', str(args.alpha),
        '--smooth',
        '--eval_ppl', '--eval_datasets', 'wikitext2',
        '--nsamples', str(args.nsamples),
        '--output_dir', os.path.join(log_dir, name),
        '--results_json', out_json,
    ]
    if diverse:
        cmd.append('--diverse_rotation')
    if args.gptq:
        cmd.append('--gptq')
    if args.multigpu:
        cmd.append('--multigpu')

    print(f'\n{"=" * 78}\n[{short(model)} / {name}] {desc}\n{" ".join(cmd)}\n{"=" * 78}',
          flush=True)
    if args.dry_run:
        return

    env = subprocess_env(model)
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    tick = time.time()
    log_path = os.path.join(log_dir, f'{name}.stdout')
    with open(log_path, 'w') as log:
        ret = subprocess.call(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    mins = (time.time() - tick) / 60
    print(f'[{short(model)} / {name}] '
          + (f'done in {mins:.1f} min' if ret == 0
             else f'FAILED (exit {ret}) after {mins:.1f} min -- see {log_path}'))


def load(out_root, model, name):
    p = result_path(out_root, model, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fp16_ppl(model):
    # reuse scripts/fp16_baseline.py's output from the mxfp4_ablation run
    p = legacy_path(model, 'fp16')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get('ppl', {}).get('wikitext2')


def build_summary(args):
    summary = {'datatype': 'MXFP4 (E2M1 + E8M0 shared scale, group=32)', 'models': {}}
    lines = []
    lines.append('# Shared vs. diverse (per-group) greedy block rotation\n')
    lines.append('WikiText2 perplexity, W4A4 MXFP4 (E2M1 + E8M0 shared scale, group=32).\n')
    lines.append('`shared` = legacy DuQuant: one greedy-searched R block reused for every '
                 'group. `diverse` = --diverse_rotation: one independently-searched R per '
                 'group (e.g. block=32 on hidden=4096 -> R shape [128, 32, 32] instead of '
                 'one [32, 32] matrix reused 128 times).\n')
    for model in args.models:
        base = fp16_ppl(model)
        m = short(model)
        summary['models'][m] = {'fp16_ppl_wikitext2': base, 'configs': {}}
        lines.append(f'\n## {m}' + (f'  (FP16 baseline: {base:.4f})' if base else ''))
        lines.append('')
        lines.append('| config | block | perm | diverse | wiki2 PPL | vs FP16 | vs shared |')
        lines.append('|---|---:|---:|:---:|---:|---:|---:|')
        pairs = [('duquantpp_shared', 'duquantpp_diverse'), ('duquant_shared', 'duquant_diverse')]
        for shared_name, diverse_name in pairs:
            shared_res = load(args.out_root, model, shared_name)
            diverse_res = load(args.out_root, model, diverse_name)
            shared_ppl = shared_res['ppl'].get('wikitext2') if shared_res else None
            for name, res, ppl in ((shared_name, shared_res, shared_ppl),
                                    (diverse_name, diverse_res,
                                     diverse_res['ppl'].get('wikitext2') if diverse_res else None)):
                desc, block, perm, diverse, _ = CONFIGS[name]
                summary['models'][m]['configs'][name] = {
                    'block_size': block, 'permutation_times': perm,
                    'diverse_rotation': diverse,
                    'ppl_wikitext2': ppl,
                }
                if ppl is None:
                    lines.append(f'| {name} | {block} | {perm} | {diverse} | (not run) | - | - |')
                    continue
                vs_fp16 = f'{ppl - base:+.4f}' if base else '-'
                vs_shared = f'{ppl - shared_ppl:+.4f}' if (diverse and shared_ppl) else '-'
                lines.append(f'| {name} | {block} | {perm} | {diverse} | {ppl:.4f} | {vs_fp16} | {vs_shared} |')
    return summary, '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen3-8B'])
    ap.add_argument('--configs', nargs='+', default=DEFAULT_ORDER, choices=list(CONFIGS))
    ap.add_argument('--max_rotation_step', type=int, default=128)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--gptq', action='store_true')
    ap.add_argument('--multigpu', action='store_true')
    ap.add_argument('--gpu', default=None, help='value for CUDA_VISIBLE_DEVICES')
    ap.add_argument('--out_root', default=os.path.join(HERE, 'log', 'diverse_rotation_ablation'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--reuse_existing', action='store_true', default=True,
                    help='reuse log/mxfp4_ablation results for *_shared cells instead of re-running (default on)')
    ap.add_argument('--no_reuse_existing', dest='reuse_existing', action='store_false')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--report_only', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if not args.report_only:
        for model in args.models:
            for name in args.configs:
                if args.skip_existing and os.path.exists(result_path(args.out_root, model, name)):
                    print(f'[{short(model)} / {name}] already has a result, skipping')
                    continue
                run_one(args, model, name)
    if not args.dry_run:
        summary, report_md = build_summary(args)
        with open(os.path.join(args.out_root, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(args.out_root, 'summary.md'), 'w') as f:
            f.write(report_md)
        print('\n' + report_md)
        print(f'wrote {os.path.join(args.out_root, "summary.json")} and summary.md')


if __name__ == '__main__':
    main()
