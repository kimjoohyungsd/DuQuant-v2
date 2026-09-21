# coding=utf-8
"""Shared runner behind scripts/<method>/run_<method>.py.

Same shape as scripts/mxfp4_ablation.py -- a CONFIGS table of named cells, one
main.py subprocess per cell, a --results_json per cell, a summary table at the
end -- factored out so each rotation method (duquant, duquantpp, hadamard,
torq) only declares its own cells and defaults.

WHY THIS EXISTS ALONGSIDE THE run_*.sh SCRIPTS
----------------------------------------------
The shell runners stamp wall-clock time into every artifact they write:

    log/<method>/<Model>_<cell>_w4a4_<YYYYmmdd_HHMMSS>.log     (+ .json)

and main.py's own logger adds log_rank0_<epoch>.txt under
log/<Model>_w4a4/ on top of that. So a single re-run never replaced anything:
each one dropped a fresh generation of files next to the old ones, and finding
"the" numbers for a config meant sorting timestamps by hand.

Here every path a run writes is a pure function of (method, model, config):

    <out_root>/<Model>/<config>.json                     results (config + PPL)
    <out_root>/<Model>/logs/<config>.stdout              full stdout+stderr
    <out_root>/<Model>/logs/<config>/<Model>_w4a4/log_rank0.txt
                                                         main.py's own logger,
                                                         via --log_name

with <out_root> defaulting to log/<method>/. Re-running a cell overwrites that
cell and touches nothing else, so the directory holds exactly one file set per
(model, config) no matter how many times it is run. --skip_existing keeps what
is already there instead; --out_root parks a run somewhere separate.

GPUs: --gpus 0 1 assigns models round-robin to those GPUs and runs the GPUs in
parallel (one model per GPU at a time, that GPU's configs sequentially) -- the
run_*.sh behaviour. With no --gpus everything runs sequentially and
CUDA_VISIBLE_DEVICES is left alone.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_utils import subprocess_env  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
PRINT_LOCK = threading.Lock()


class Method:
    """One rotation method's runner definition.

    name      directory/tag, e.g. 'duquant' -- also the default log root
    headline  one line printed above the results table
    configs   {cell name: (description, [extra main.py flags])}
    order     cells a bare run executes (default: every cell)
    models    models a bare run executes
    """

    def __init__(self, name, headline, configs, order=None, models=None):
        self.name = name
        self.headline = headline
        self.configs = configs
        self.order = order or list(configs)
        self.models = models or ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-3.1-8B']


def short(model):
    return model.rstrip('/').split('/')[-1]


def model_dir(out_root, model):
    return os.path.join(out_root, short(model))


def result_path(out_root, model, name):
    return os.path.join(model_dir(out_root, model), f'{name}.json')


def stdout_path(out_root, model, name):
    return os.path.join(model_dir(out_root, model), 'logs', f'{name}.stdout')


def python_for(args, model):
    """Interpreter for this model: --qwen_python (or $QWEN3_PYTHON) for Qwen.

    Mirrors the run_*_Qwen3.sh scripts' QWEN3_PYTHON hook. When it is unset,
    env_utils.subprocess_env still puts the vendored Qwen3-capable transformers
    on PYTHONPATH, which is the fallback those scripts document.
    """
    if 'qwen' in model.lower() and args.qwen_python:
        return args.qwen_python
    return args.python


def build_env(args, model, gpu):
    env = subprocess_env(model)
    # Llama's slow (sentencepiece) tokenizer trips a protobuf>=4 incompatibility
    # in this env; force the pure-python parser so it loads. setdefault, not
    # assignment, so an explicit setting in the caller's environment wins.
    env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    env.setdefault('HF_HUB_OFFLINE', '1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return env


def build_cmd(spec, args, model, name):
    """main.py argv for one cell: shared flags + that cell's own + passthrough."""
    desc, flags = spec.configs[name]
    logs = os.path.join(model_dir(args.out_root, model), 'logs')
    cmd = [
        python_for(args, model), 'main.py',
        '--model', model,
        '--wbits', str(args.wbits), '--abits', str(args.abits),
        '--eval_ppl', '--eval_datasets', *args.eval_datasets,
        '--nsamples', str(args.nsamples),
        '--batch_size', str(args.batch_size),
        '--seed', str(args.seed),
        # Fixed per (model, config) -- main.py appends <Model>_w<W>a<A>/ itself.
        '--output_dir', os.path.join(logs, name),
        '--log_name', 'log_rank0.txt',
        '--results_json', result_path(args.out_root, model, name),
    ]
    cmd += list(flags)
    if args.gptq:
        cmd.append('--gptq')
    if args.multigpu:
        cmd.append('--multigpu')
    cmd += args.extra
    return cmd, desc


def run_one(spec, args, model, name, gpu=None):
    cmd, desc = build_cmd(spec, args, model, name)
    logf = stdout_path(args.out_root, model, name)
    os.makedirs(os.path.dirname(logf), exist_ok=True)

    where = f'GPU {gpu}' if gpu is not None else 'default GPU'
    with PRINT_LOCK:
        print(f'\n{"=" * 78}\n[{short(model)} / {name}] ({where}) {desc}\n'
              f'{" ".join(cmd)}\n  -> {logf}\n{"=" * 78}', flush=True)
    if args.dry_run:
        return 0

    tick = time.time()
    with open(logf, 'w') as log:
        ret = subprocess.call(cmd, cwd=HERE, env=build_env(args, model, gpu),
                              stdout=log, stderr=subprocess.STDOUT)
    mins = (time.time() - tick) / 60
    with PRINT_LOCK:
        print(f'[{short(model)} / {name}] '
              + (f'done in {mins:.1f} min' if ret == 0
                 else f'FAILED (exit {ret}) after {mins:.1f} min -- see {logf}'), flush=True)
    return ret


def run_model(spec, args, model, gpu=None):
    for name in args.configs:
        if args.skip_existing and os.path.exists(result_path(args.out_root, model, name)):
            with PRINT_LOCK:
                print(f'[{short(model)} / {name}] already has a result, skipping')
            continue
        run_one(spec, args, model, name, gpu)


def run_all(spec, args):
    if not args.gpus:
        for model in args.models:
            run_model(spec, args, model)
        return
    # One worker thread per GPU; models are dealt round-robin onto them, so a
    # GPU runs its models one after another while the GPUs run in parallel.
    queues = {gpu: [] for gpu in args.gpus}
    for i, model in enumerate(args.models):
        queues[args.gpus[i % len(args.gpus)]].append(model)
    threads = []
    for gpu, models in queues.items():
        if not models:
            continue
        t = threading.Thread(target=lambda g=gpu, ms=models: [
            run_model(spec, args, m, g) for m in ms])
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


def load(out_root, model, name):
    p = result_path(out_root, model, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def fp16_ppl(args, model, dataset):
    """FP16 reference for `model`, from whichever --fp16_root holds a fp16.json.

    Written by scripts/fp16_baseline.py; log/mxfp4_ablation/<Model>/fp16.json is
    where the existing ones live, so that is the default first stop.
    """
    for root in (args.out_root, args.fp16_root):
        p = os.path.join(root, short(model), 'fp16.json')
        if os.path.exists(p):
            with open(p) as f:
                ppl = json.load(f).get('ppl', {}).get(dataset)
            if ppl is not None:
                return ppl
    return None


def print_table(spec, args):
    datasets = args.eval_datasets
    print('\n' + '=' * 78)
    print(spec.headline)
    print(f'W{args.wbits}A{args.abits} MXFP4 (E2M1 + E8M0 shared scale, group=32)'
          + ('  [+GPTQ]' if args.gptq else ''))
    print(f'results: {args.out_root}')
    print('=' * 78)
    for model in args.models:
        base = {d: fp16_ppl(args, model, d) for d in datasets}
        shown = '  '.join(f'{d} {v:.4f}' for d, v in base.items() if v is not None)
        print(f'\n{short(model)}' + (f'   [FP16: {shown}]' if shown else ''))
        head = f"  {'config':<20}"
        for d in datasets:
            head += f'{d:>12}{"vs FP16":>10}'
        print(head + '  desc')
        for name in args.configs:
            desc, _ = spec.configs[name]
            res = load(args.out_root, model, name)
            row = f'  {name:<20}'
            for d in datasets:
                if res is None:
                    row += f'{"(not run)":>12}{"-":>10}'
                    continue
                ppl = res.get('ppl', {}).get(d)
                if ppl is None:
                    row += f'{"(no ppl)":>12}{"-":>10}'
                    continue
                delta = f'{ppl - base[d]:+.4f}' if base.get(d) else '-'
                row += f'{ppl:>12.4f}{delta:>10}'
            print(row + f'  {desc}')


def build_parser(spec):
    ap = argparse.ArgumentParser(
        description=spec.headline,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='cells:\n' + '\n'.join(
            f'  {n:<20} {d}' for n, (d, _) in spec.configs.items()))
    ap.add_argument('--models', nargs='+', default=spec.models,
                    help='HF model ids. Qwen3 works here too -- see --qwen_python.')
    ap.add_argument('--configs', nargs='+', default=spec.order, choices=list(spec.configs),
                    help='which cells to run/report (default: %(default)s)')
    ap.add_argument('--gpus', nargs='+', default=None,
                    help='CUDA_VISIBLE_DEVICES values, one model per GPU in '
                         'parallel (e.g. --gpus 0 1). Omit to run sequentially.')
    ap.add_argument('--wbits', type=int, default=4)
    ap.add_argument('--abits', type=int, default=4)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--eval_datasets', nargs='+', default=['wikitext2'],
                    choices=['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new'])
    ap.add_argument('--gptq', action='store_true', help='add --gptq to every cell')
    ap.add_argument('--multigpu', action='store_true',
                    help='main.py --multigpu (shards ONE model over the visible '
                         'GPUs). The run_*.sh scripts avoid it -- see run.sh.')
    ap.add_argument('--out_root', default=os.path.join(HERE, 'log', spec.name),
                    help='default: %(default)s')
    ap.add_argument('--fp16_root', default=os.path.join(HERE, 'log', 'mxfp4_ablation'),
                    help='where to look for <Model>/fp16.json baselines '
                         '(default: %(default)s)')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--qwen_python', default=os.environ.get('QWEN3_PYTHON'),
                    help='interpreter for Qwen models (default: $QWEN3_PYTHON). '
                         'Unset -> the vendored .vendor/transformers_qwen3 goes '
                         'on PYTHONPATH instead; see env_utils.py.')
    ap.add_argument('--skip_existing', action='store_true',
                    help='keep cells that already have a <config>.json instead of '
                         'overwriting them')
    ap.add_argument('--dry_run', action='store_true', help='print commands only')
    ap.add_argument('--report_only', action='store_true',
                    help='re-print the table from existing JSONs, run nothing')
    ap.add_argument('extra', nargs='*', default=[],
                    help='extra flags passed through to main.py verbatim, after '
                         'a `--` separator (e.g. -- --torq_lambda 2.0)')
    return ap


def main(spec, argv=None):
    args = build_parser(spec).parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)
    if not args.report_only:
        run_all(spec, args)
    if not args.dry_run:
        print_table(spec, args)
