# coding=utf-8
"""DuQuant (NeurIPS 2024) vs DuQuant++ under the MXFP4 datatype.

    python scripts/mxfp4_ablation.py --models meta-llama/Llama-2-7b-hf
    python scripts/mxfp4_ablation.py --report_only

Both arms quantize W4A4 with the *same* MXFP4 format (E2M1 elements + a shared
E8M0 scale per group of 32 along the reduction axis -- see quantize/fp4_ops.py;
the group size is fixed at 32 in UniformAffineQuantizer.per_token_fp4). The only
thing that differs is the rotation transform applied before quantization:

  duquant      block_size=128, permutation_times=1
               = the NeurIPS'24 recipe: greedy block rotation -> zigzag
                 permutation -> greedy block rotation. The 128-wide rotation
                 spans four MXFP4 groups, so it redistributes outliers *across*
                 groups that do not share a scale.

  duquantpp    block_size=32, permutation_times=0
               = DuQuant++: one greedy block rotation whose block is exactly the
                 MXFP4 group, so the rotation flattens precisely the values that
                 will share one E8M0 scale.

Two cross cells isolate which of the two changes (block size vs. dropping the
permutation) actually carries the difference.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_utils import subprocess_env  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name: (description, block_size, permutation_times)
CONFIGS = {
    'duquant':     ('DuQuant (NeurIPS 24): rot -> zigzag perm -> rot, block=128', 128, 1),
    'duquantpp':   ('DuQuant++: single greedy rot, block=32 = MXFP4 group',        32, 0),
    'b32_perm1':   ('cross: block=32 WITH zigzag permutation',                      32, 1),
    'b128_perm0':  ('cross: block=128 WITHOUT permutation',                        128, 0),
}
DEFAULT_ORDER = ['duquant', 'duquantpp', 'b32_perm1', 'b128_perm0']


def short(model):
    return model.rstrip('/').split('/')[-1]


def result_path(out_root, model, name):
    return os.path.join(out_root, short(model), f'{name}.json')


def run_one(args, model, name):
    desc, block, perm = CONFIGS[name]
    out_json = result_path(args.out_root, model, name)
    log_dir = os.path.join(args.out_root, short(model), 'logs')
    os.makedirs(log_dir, exist_ok=True)

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


def fp16_ppl(out_root, model):
    p = os.path.join(out_root, short(model), 'fp16.json')
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f).get('ppl', {}).get('wikitext2')


def print_table(args):
    print('\n' + '=' * 78)
    print('WikiText2 perplexity, W4A4 MXFP4 (E2M1 + E8M0 shared scale, group=32)')
    print('=' * 78)
    for model in args.models:
        base = fp16_ppl(args.out_root, model)
        print(f'\n{short(model)}' + (f'   [FP16 baseline: {base:.4f}]' if base else ''))
        print(f"  {'config':<12} {'block':>5} {'perm':>4}  {'wiki2 PPL':>10}  {'vs FP16':>8}  desc")
        for name in args.configs:
            desc, block, perm = CONFIGS[name]
            res = load(args.out_root, model, name)
            if res is None:
                print(f'  {name:<12} {block:>5} {perm:>4}  {"(not run)":>10}  {"-":>8}  {desc}')
                continue
            ppl = res['ppl'].get('wikitext2')
            if ppl is None:
                print(f'  {name:<12} {block:>5} {perm:>4}  {"(no ppl)":>10}  {"-":>8}  {desc}')
                continue
            delta = f'{ppl - base:+.4f}' if base else '-'
            print(f'  {name:<12} {block:>5} {perm:>4}  {ppl:>10.4f}  {delta:>8}  {desc}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3.1-8B'])
    ap.add_argument('--configs', nargs='+', default=DEFAULT_ORDER, choices=list(CONFIGS))
    ap.add_argument('--max_rotation_step', type=int, default=128)
    ap.add_argument('--alpha', type=float, default=0.6)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--gptq', action='store_true')
    ap.add_argument('--multigpu', action='store_true')
    ap.add_argument('--gpu', default=None, help='value for CUDA_VISIBLE_DEVICES')
    ap.add_argument('--out_root', default=os.path.join(HERE, 'log', 'mxfp4_ablation'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--skip_existing', action='store_true')
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
        print_table(args)


if __name__ == '__main__':
    main()
