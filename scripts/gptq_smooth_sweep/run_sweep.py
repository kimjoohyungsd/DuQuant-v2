# coding=utf-8
"""GPTQ + Hadamard, No-Smooth vs. alpha-searched Smooth, across N models.

    python scripts/gptq_smooth_sweep/run_sweep.py --dry_run
    python scripts/gptq_smooth_sweep/run_sweep.py --setup_qwen_env
    python scripts/gptq_smooth_sweep/run_sweep.py --report_only

Two arms per model, both --quant_method hadamard --gptq, W4A4:

  exp1_nosmooth   no --smooth at all.
  exp2_best       --smooth --alpha <best>, where <best> is chosen by running
                  the calibration+GPTQ+eval_ppl pipeline once per
                  --alphas candidate (PPL-only, no --tasks -- zero-shot is
                  expensive and would otherwise re-run once per candidate)
                  and keeping the lowest --select_dataset PPL. That winning
                  config is then re-run once more WITH --tasks so the report
                  also gets zero-shot numbers for it.

  NOTE on "validation" PPL: this repo's datautils.get_wikitext2 has no
  separate held-out validation split -- get_loaders returns the wikitext2
  TRAIN split (for calibration) and the TEST split (for --eval_ppl). There is
  therefore no leakage-free validation set to search --alpha against beyond
  the same test-set PPL --eval_ppl already reports; --select_dataset (default
  wikitext2) is that number. This matches how the rest of this repo's
  scripts/*.py already pick alpha (see mxfp4_ablation.py's fixed alpha=0.6).

GPU scheduling: every model, in parallel, pulled from one shared pool of
`n`-GPU allocations (n from --gpus_needed's fp16-size heuristic, or
--gpus_per_model MODEL=N to override). The pool only offers GPUs that
currently have >= --gpu_min_free_mb free (nvidia-smi), so a card someone else
on this shared machine is already using is left alone instead of risking an
OOM on their job or ours. Within one model, exp1 -> all alpha searches ->
exp2_best run sequentially (the alpha winner is a real data dependency); the
pool is what lets a *different* model's job start on whatever GPUs are free
while that happens, and is why 5 models with 1-2 GPUs apiece can keep 7 cards
busy without any manual round-robin.

Qwen3: this repo's shared env pins a transformers version that predates the
qwen3 architecture (requirements_qwen3.txt). --setup_qwen_env conda-creates
`duquant-qwen3` and pip-installs requirements_qwen3.txt into it BEFORE any
GPU job is dispatched (see ensure_qwen_env) -- run it once (or pass
--qwen_python to point at an interpreter you already prepared yourself).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)
import run_utils  # noqa: E402
from env_utils import subprocess_env  # noqa: E402

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
PRINT_LOCK = threading.Lock()

DEFAULT_MODELS = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Meta-Llama-3.1-8B',
    'Qwen/Qwen3-8B',
    'Qwen/Qwen3-14B',
]
DEFAULT_ALPHAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_TASKS = ('arc_easy,arc_challenge,winogrande,hellaswag,'
                 'openbookqa,lambada_openai,piqa')

EXP1_FLAGS = []            # no --smooth
EXP2_FLAGS = ['--smooth']  # + --alpha <candidate>, appended per job


# --------------------------------------------------------------------------
# GPU discovery / pool
# --------------------------------------------------------------------------

def query_gpus():
    """[(id, total_mb, used_mb, free_mb), ...] via nvidia-smi."""
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free',
         '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE, text=True, check=True).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, total, used, free = (int(x) for x in re.split(r',\s*', line.strip()))
        gpus.append((idx, total, used, free))
    return gpus


def detect_usable_gpus(min_free_mb):
    """GPU ids with >= min_free_mb free right now; also returns the skipped ones.

    Deliberately re-checked live (not just "torch.cuda.device_count()") so a
    card another process on this shared box is already sitting on -- as
    happens on this machine -- is excluded instead of assumed available.
    """
    gpus = query_gpus()
    usable = [idx for idx, _, _, free in gpus if free >= min_free_mb]
    skipped = [(idx, free) for idx, _, _, free in gpus if free < min_free_mb]
    return usable, skipped


def gpus_needed(model, gpu_mem_gb, overrides):
    """How many GPUs this model needs for the EVAL forward pass.

    main.py's non---multigpu eval path does `lm.model = lm.model.to(device)`:
    the whole fp16 model must fit on one card. Quantization itself is
    layer-wise (LMClass loads device_map='cpu'; quantize/gptq.py and
    quantize/duquant.py move one decoder layer at a time) and needs far less,
    so this is purely about sizing the eval/--tasks pass.
    """
    if model in overrides:
        return overrides[model]
    m = re.search(r'(\d+(?:\.\d+)?)\s*[bB](?:-|_|$)', model.split('/')[-1])
    params_b = float(m.group(1)) if m else 8.0
    # ~2 bytes/param (fp16) + ~10% slack for KV cache / activation buffers.
    needed = math.ceil((params_b * 2.2) / gpu_mem_gb)
    return max(1, needed)


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        model, n = p.split('=', 1)
        out[model] = int(n)
    return out


class GpuPool:
    """Bin-packs GPU ids: acquire(n) blocks until n are free, release(ids) returns them."""

    def __init__(self, gpu_ids):
        self._cond = threading.Condition()
        self._free = list(gpu_ids)

    def acquire(self, n):
        with self._cond:
            while len(self._free) < n:
                self._cond.wait()
            ids = sorted(self._free[:n])
            self._free = self._free[n:]
            return ids

    def release(self, ids):
        with self._cond:
            self._free.extend(ids)
            self._cond.notify_all()


# --------------------------------------------------------------------------
# Qwen3 env prep
# --------------------------------------------------------------------------

def ensure_qwen_env(args, env_name='duquant-qwen3'):
    """conda-create `env_name`, then pip install -r requirements_qwen3.txt into
    it -- in that order, once, before returning -- so every later Qwen
    main.py subprocess has an interpreter that actually resolves the qwen3
    architecture. Idempotent: skips the parts already satisfied.
    """
    conda = shutil.which('conda')
    if conda is None:
        print('[qwen-env] no `conda` on PATH -- pass --qwen_python pointing at an '
              'interpreter with transformers>=4.51 instead', file=sys.stderr)
        return
    envs = subprocess.run([conda, 'env', 'list'], stdout=subprocess.PIPE, text=True).stdout
    exists = any(line.split()[0] == env_name
                 for line in envs.splitlines() if line and not line.startswith('#'))
    if not exists:
        print(f'[qwen-env] creating conda env `{env_name}` (python=3.10)...', flush=True)
        subprocess.check_call([conda, 'create', '-n', env_name, 'python=3.10', '-y'])

    qwen_python = subprocess.run(
        [conda, 'run', '-n', env_name, 'python', '-c', 'import sys; print(sys.executable)'],
        stdout=subprocess.PIPE, text=True, check=True).stdout.strip()

    ver = subprocess.run([qwen_python, '-c', 'import transformers; print(transformers.__version__)'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True).stdout.strip()
    ver_tuple = tuple(int(x) for x in re.findall(r'\d+', ver)[:2]) if ver else (0, 0)
    if ver_tuple < (4, 51):
        req = os.path.join(REPO_ROOT, 'requirements_qwen3.txt')
        print(f'[qwen-env] pip install -r {req} into `{env_name}` '
              f'(current transformers: {ver or "none"})...', flush=True)
        subprocess.check_call([conda, 'run', '-n', env_name, 'pip', 'install', '-r', req])
    else:
        print(f'[qwen-env] `{env_name}` already has transformers {ver}, skipping install')

    if not args.qwen_python:
        args.qwen_python = qwen_python
        print(f'[qwen-env] using {qwen_python} for Qwen jobs')


def python_for(args, model):
    if 'qwen' in model.lower() and args.qwen_python:
        return args.qwen_python
    return args.python


# --------------------------------------------------------------------------
# One main.py job
# --------------------------------------------------------------------------

def model_dir(args, model):
    return os.path.join(args.out_root, run_utils.short(model))


def result_path(args, model, name):
    return os.path.join(model_dir(args, model), f'{name}.json')


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def build_cmd(args, model, name, extra_flags, tasks, out_json):
    logs = os.path.join(model_dir(args, model), 'logs')
    cmd = [
        python_for(args, model), os.path.join(REPO_ROOT, 'main.py'),
        '--model', model,
        '--quant_method', 'hadamard',
        '--block_size', str(args.block_size),
        '--wbits', str(args.wbits), '--abits', str(args.abits),
        '--gptq',
        '--eval_ppl', '--eval_datasets', *args.eval_datasets,
        '--nsamples', str(args.nsamples),
        '--seed', str(args.seed),
        '--batch_size', str(args.batch_size),
        # main.py appends <Model>_w<W>a<A>/ to this itself.
        '--output_dir', os.path.join(logs, name),
        '--log_name', 'log_rank0.txt',
        '--results_json', out_json,
    ]
    cmd += extra_flags
    if tasks:
        cmd += ['--tasks', tasks]
    return cmd


def run_job(args, model, name, extra_flags, tasks, gpu_ids):
    out_json = result_path(args, model, name)
    cmd = build_cmd(args, model, name, extra_flags, tasks, out_json)
    if len(gpu_ids) > 1:
        cmd.append('--multigpu')

    env = subprocess_env(model)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    if args.offline:
        env.setdefault('HF_HUB_OFFLINE', '1')

    logf = os.path.join(model_dir(args, model), 'logs', f'{name}.stdout')
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    with PRINT_LOCK:
        print(f'\n{"=" * 78}\n[{run_utils.short(model)}/{name}] GPUs {gpu_ids}\n'
              f'{" ".join(cmd)}\n  -> {logf}\n{"=" * 78}', flush=True)
    if args.dry_run:
        return 0

    tick = time.time()
    with open(logf, 'w') as f:
        ret = subprocess.call(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    mins = (time.time() - tick) / 60
    with PRINT_LOCK:
        status = 'done' if ret == 0 else f'FAILED (exit {ret})'
        print(f'[{run_utils.short(model)}/{name}] {status} in {mins:.1f} min '
              f'(log: {logf})', flush=True)
    return ret


# --------------------------------------------------------------------------
# Per-model orchestration
# --------------------------------------------------------------------------

def model_worker(args, model, pool, overrides):
    need = gpus_needed(model, args.gpu_mem_gb, overrides)
    mdir = model_dir(args, model)
    os.makedirs(mdir, exist_ok=True)

    # --- Exp 1: No-Smooth ---
    p1 = result_path(args, model, 'exp1_nosmooth')
    if not (args.skip_existing and os.path.exists(p1)):
        gpu_ids = pool.acquire(need)
        try:
            run_job(args, model, 'exp1_nosmooth', EXP1_FLAGS, args.tasks, gpu_ids)
        finally:
            pool.release(gpu_ids)

    # --- Exp 2: alpha grid search (PPL-only), then one full run @ best alpha ---
    search = {}
    for alpha in args.alphas:
        name = f'exp2_alpha{alpha}'
        p = result_path(args, model, name)
        if args.skip_existing and os.path.exists(p):
            res = load_json(p)
        else:
            gpu_ids = pool.acquire(need)
            try:
                run_job(args, model, name, EXP2_FLAGS + ['--alpha', str(alpha)],
                       None, gpu_ids)
            finally:
                pool.release(gpu_ids)
            res = load_json(p)
        ppl = (res or {}).get('ppl', {}).get(args.select_dataset)
        if ppl is not None:
            search[alpha] = ppl

    if args.dry_run:
        return
    if not search:
        with PRINT_LOCK:
            print(f'[{run_utils.short(model)}] exp2: no alpha produced a '
                  f'{args.select_dataset} PPL -- skipping best-alpha run')
        return

    best_alpha = min(search, key=search.get)
    with PRINT_LOCK:
        print(f'[{run_utils.short(model)}] exp2 best alpha = {best_alpha} '
              f'({args.select_dataset} PPL {search[best_alpha]:.4f}); '
              f'searched {search}')
    with open(os.path.join(mdir, 'exp2_best_alpha.json'), 'w') as f:
        json.dump({'best_alpha': best_alpha, 'search': search}, f, indent=2)

    p_best = result_path(args, model, 'exp2_best')
    if not (args.skip_existing and os.path.exists(p_best)):
        gpu_ids = pool.acquire(need)
        try:
            run_job(args, model, 'exp2_best', EXP2_FLAGS + ['--alpha', str(best_alpha)],
                   args.tasks, gpu_ids)
        finally:
            pool.release(gpu_ids)


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------

def fmt(v, nd=4):
    return f'{v:.{nd}f}' if isinstance(v, (int, float)) else '--'


def build_report(args):
    lines = []
    lines.append(f'## GPTQ + Hadamard (block_size={args.block_size}, '
                 f'W{args.wbits}A{args.abits}) -- No-Smooth vs. alpha-searched Smooth\n')
    lines.append(f'Alpha candidates: `{args.alphas}`, selected by lowest '
                 f'`{args.select_dataset}` PPL (see run_sweep.py module docstring '
                 f'on why this doubles as "validation" PPL for this repo).\n')

    lines.append('### Perplexity\n')
    ds_cols = args.eval_datasets
    lines.append('| Model | Condition | alpha | ' + ' | '.join(ds_cols) + ' |')
    lines.append('|---|---|---|' + '---|' * len(ds_cols))
    for model in args.models:
        r1 = load_json(result_path(args, model, 'exp1_nosmooth'))
        rb = load_json(result_path(args, model, 'exp2_best'))
        alpha_info = load_json(os.path.join(model_dir(args, model), 'exp2_best_alpha.json')) or {}
        best_alpha = alpha_info.get('best_alpha', '--')
        row1 = [run_utils.short(model), 'No-Smooth', '--'] + \
               [fmt((r1 or {}).get('ppl', {}).get(d)) for d in ds_cols]
        row2 = [run_utils.short(model), 'Smooth (best)', str(best_alpha)] + \
               [fmt((rb or {}).get('ppl', {}).get(d)) for d in ds_cols]
        lines.append('| ' + ' | '.join(row1) + ' |')
        lines.append('| ' + ' | '.join(row2) + ' |')
    lines.append('')

    if args.tasks:
        task_cols = args.tasks.split(',')
        lines.append('### Zero-shot accuracy (%)\n')
        lines.append('| Model | Condition | alpha | ' + ' | '.join(task_cols) + ' | avg |')
        lines.append('|---|---|---|' + '---|' * (len(task_cols) + 1))
        for model in args.models:
            r1 = load_json(result_path(args, model, 'exp1_nosmooth'))
            rb = load_json(result_path(args, model, 'exp2_best'))
            alpha_info = load_json(os.path.join(model_dir(args, model), 'exp2_best_alpha.json')) or {}
            best_alpha = alpha_info.get('best_alpha', '--')
            for label, alpha, res in (('No-Smooth', '--', r1),
                                      ('Smooth (best)', best_alpha, rb)):
                accs = (res or {}).get('tasks', {})
                vals = [accs.get(t) for t in task_cols]
                present = [v for v in vals if v is not None]
                avg = accs.get('acc_avg') if accs.get('acc_avg') is not None else \
                    (sum(present) / len(present) if present else None)
                row = [run_utils.short(model), label, str(alpha)] + \
                    [fmt(v, 2) for v in vals] + [fmt(avg, 2)]
                lines.append('| ' + ' | '.join(row) + ' |')
        lines.append('')

    return '\n'.join(lines)


def print_report(args):
    report = build_report(args)
    print('\n' + '=' * 78)
    print(report)
    out_path = os.path.join(args.out_root, 'SUMMARY.md')
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'(written to {out_path})')


# --------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    ap.add_argument('--alphas', nargs='+', type=float, default=DEFAULT_ALPHAS)
    ap.add_argument('--select_dataset', default='wikitext2',
                    help='PPL dataset used to pick the best alpha')
    ap.add_argument('--eval_datasets', nargs='+', default=['wikitext2'],
                    choices=['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new'])
    ap.add_argument('--tasks', default=DEFAULT_TASKS,
                    help='comma-separated lm_eval tasks; empty string skips zero-shot')
    ap.add_argument('--block_size', type=int, default=32)
    ap.add_argument('--wbits', type=int, default=4)
    ap.add_argument('--abits', type=int, default=4)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=8)

    ap.add_argument('--gpu_mem_gb', type=float, default=24.0,
                    help='per-card memory this cluster has, for the 1-vs-N-GPU heuristic')
    ap.add_argument('--gpu_min_free_mb', type=int, default=20000,
                    help='a GPU with less free than this (someone else\'s job) is skipped')
    ap.add_argument('--gpus', nargs='+', type=int, default=None,
                    help='use exactly these GPU ids instead of auto-detecting free ones')
    ap.add_argument('--gpus_per_model', nargs='+', default=None,
                    help='override the size heuristic, e.g. --gpus_per_model '
                         'Qwen/Qwen3-14B=2 meta-llama/Llama-2-13b-hf=2')

    ap.add_argument('--out_root', default=os.path.join(REPO_ROOT, 'log', 'gptq_smooth_sweep'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--qwen_python', default=os.environ.get('QWEN3_PYTHON'))
    ap.add_argument('--setup_qwen_env', action='store_true',
                    help='conda-create duquant-qwen3 + pip install requirements_qwen3.txt '
                         'first (idempotent) -- see ensure_qwen_env')
    ap.add_argument('--offline', action='store_true',
                    help='HF_HUB_OFFLINE=1 -- only if every --models entry is already cached')
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--dry_run', action='store_true', help='print commands only, run nothing')
    ap.add_argument('--report_only', action='store_true',
                    help='re-print the table from existing JSONs, run nothing')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out_root, exist_ok=True)
    overrides = parse_overrides(args.gpus_per_model)

    if not args.report_only:
        if args.setup_qwen_env and any('qwen' in m.lower() for m in args.models):
            ensure_qwen_env(args)

        if args.gpus:
            usable, skipped = list(args.gpus), []
        else:
            usable, skipped = detect_usable_gpus(args.gpu_min_free_mb)
        if skipped:
            print(f'skipping GPU(s) already busy (< {args.gpu_min_free_mb}MB free): {skipped}')
        if not usable and not args.dry_run:
            sys.exit('no usable GPU found (see --gpu_min_free_mb / --gpus)')
        print(f'usable GPUs: {usable}')
        pool = GpuPool(usable if usable else [0])  # dry_run: pool content is unused

        with ThreadPoolExecutor(max_workers=max(1, len(args.models))) as ex:
            futs = {ex.submit(model_worker, args, m, pool, overrides): m for m in args.models}
            for fut in as_completed(futs):
                m = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    with PRINT_LOCK:
                        print(f'[{run_utils.short(m)}] worker crashed: {e!r}')

    if not args.dry_run:
        print_report(args)


if __name__ == '__main__':
    main()
