# coding=utf-8
"""Run --quant_method blockwise_flatquant (quantize/blockwise_flatquant.py)
across models and record WikiText2 PPL to log/blockwise_flatquant/.

    python scripts/blockwise_flatquant_run.py --models meta-llama/Llama-2-7b-hf
    python scripts/blockwise_flatquant_run.py --report_only

blockwise_flatquant trains one SVDGroupTransMatrix (quantize/svd_trans.py) per
Linear layer -- U @ diag(s) @ V^T, U/V orthogonal via Cayley parametrization,
sized to the real MXFP4 group (32) -- via per-DecoderLayer block-wise gradient
descent (quantize/blockwise_flatquant.py, ported from Rotate-Test's own
flatquant/train_utils.py::cali_flat_quant). It is an alternative to
quantize/duquant.py's greedy block rotation, not a variant of it, so results
are compared against the same FP16 / duquant baselines already recorded in
log/mxfp4_ablation/ and log/qwen/ rather than re-measured here.
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


def short(model):
    return model.rstrip('/').split('/')[-1]


def result_path(out_root, model):
    return os.path.join(out_root, f'{short(model)}.json')


def fp16_ppl(model):
    for cand in (
        os.path.join(HERE, 'log', 'mxfp4_ablation', short(model), 'fp16.json'),
        os.path.join(HERE, 'log', 'qwen', f'{short(model)}_fp16.json'),
    ):
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f).get('ppl', {}).get('wikitext2')
    return None


def duquant_ppl(model):
    """block=128,perm=1 shared-rotation PPL, for a same-page comparison."""
    for cand in (
        os.path.join(HERE, 'log', 'mxfp4_ablation', short(model), 'duquant.json'),
        os.path.join(HERE, 'log', 'diverse_rotation_ablation', short(model), 'duquant_shared.json'),
    ):
        if os.path.exists(cand):
            with open(cand) as f:
                return json.load(f).get('ppl', {}).get('wikitext2')
    return None


def run_one(args, model):
    out_json = result_path(args.out_root, model)
    log_dir = os.path.join(args.out_root, short(model), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        args.python, 'main.py',
        '--model', model,
        '--wbits', '4', '--abits', '4',
        '--quant_method', 'blockwise_flatquant',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.cali_bsz),
        '--flat_lr', str(args.flat_lr),
        '--nsamples', str(args.nsamples),
        '--eval_ppl', '--eval_datasets', 'wikitext2',
        '--output_dir', log_dir,
        '--results_json', out_json,
    ]
    if args.multigpu:
        cmd.append('--multigpu')

    print(f'\n{"=" * 78}\n[{short(model)}] {" ".join(cmd)}\n{"=" * 78}', flush=True)
    if args.dry_run:
        return

    env = subprocess_env(model)
    if args.gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
    tick = time.time()
    log_path = os.path.join(log_dir, 'stdout.log')
    with open(log_path, 'w') as log:
        ret = subprocess.call(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    mins = (time.time() - tick) / 60
    print(f'[{short(model)}] ' + (f'done in {mins:.1f} min' if ret == 0
                                  else f'FAILED (exit {ret}) after {mins:.1f} min -- see {log_path}'))


def print_table(args):
    print('\n' + '=' * 88)
    print('WikiText2 perplexity, W4A4 MXFP4 -- blockwise_flatquant (trained SVD affine, '
         f'group={32}) vs. duquant (greedy rotation, block=128/perm=1) vs. FP16')
    print('=' * 88)
    print(f"{'model':<20} {'FP16':>10} {'duquant':>10} {'blockwise_flatquant':>20} {'vs duquant':>12}")
    summary = {}
    for model in args.models:
        base = fp16_ppl(model)
        dq = duquant_ppl(model)
        res = None
        p = result_path(args.out_root, model)
        if os.path.exists(p):
            with open(p) as f:
                res = json.load(f).get('ppl', {}).get('wikitext2')
        summary[short(model)] = {'fp16': base, 'duquant': dq, 'blockwise_flatquant': res}
        delta = f'{res - dq:+.4f}' if (res is not None and dq is not None) else '-'
        row = [short(model),
              f'{base:.4f}' if base else '-',
              f'{dq:.4f}' if dq else '-',
              f'{res:.4f}' if res is not None else '(not run)',
              delta]
        print(f"{row[0]:<20} {row[1]:>10} {row[2]:>10} {row[3]:>20} {row[4]:>12}")
    with open(os.path.join(args.out_root, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {os.path.join(args.out_root, "summary.json")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+',
                    default=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3.1-8B', 'Qwen/Qwen3-8B'])
    ap.add_argument('--epochs', type=int, default=3,
                    help="Rotate-Test's own default is 15, calibrated for their (much "
                         "larger, hidden_size-wide Kronecker-decomposed) transform. This "
                         "port's transform is a single 32x32 matrix per Linear -- a far "
                         "smaller, easier optimization -- and training MSE was already "
                         "flat by epoch 2-3 in testing; 15 would take ~7-8h/model here "
                         "(eager attention, single 3090) for no measurable benefit.")
    ap.add_argument('--cali_bsz', type=int, default=2,
                    help="Rotate-Test's own default is 4, but this repo's LlamaAttention "
                         "uses eager (non-flash) attention at real seqlen=2048, and 4 OOMs "
                         "a 24GB GPU on an 8B model within the first layer; 2 is validated "
                         "to run all layers of a 7-8B model without OOM on a single 3090.")
    ap.add_argument('--flat_lr', type=float, default=5e-3)
    ap.add_argument('--nsamples', type=int, default=32,
                    help="Rotate-Test's own default is 128; reduced 4x alongside --epochs "
                         "for the same wall-clock reason (matches quantize/duquant.py's "
                         "own greedy-rotation calibration in spirit -- both are one "
                         "block-wise calibration pass, just gradient- vs. search-based).")
    ap.add_argument('--multigpu', action='store_true')
    ap.add_argument('--gpu', default=None, help='value for CUDA_VISIBLE_DEVICES')
    ap.add_argument('--out_root', default=os.path.join(HERE, 'log', 'blockwise_flatquant'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--report_only', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    if not args.report_only:
        for model in args.models:
            if args.skip_existing and os.path.exists(result_path(args.out_root, model)):
                print(f'[{short(model)}] already has a result, skipping')
                continue
            run_one(args, model)
    if not args.dry_run:
        print_table(args)


if __name__ == '__main__':
    main()
