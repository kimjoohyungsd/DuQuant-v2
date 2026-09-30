# coding=utf-8
"""DuQuant (no SmoothQuant), GPTQ off vs on, across every locally-cached model.

    python scripts/duquant_gptq_ablation/run_ablation.py
    python scripts/duquant_gptq_ablation/run_ablation.py --report_only

Two arms per model, both --quant_method duquant --block_size 32
--permutation_times 1 (this repo's "duquant_nosmooth" cell -- see
scripts/duquant/run_duquant.py's docstring for what --permutation_times 1
means), NO --smooth, --eval_ppl --eval_datasets wikitext2 only (no zero-shot
--tasks this round):

  gptq_off   plain fake-quant (no GPTQ pass)
  gptq_on    + --gptq

Reuses scripts/gptq_smooth_sweep/run_sweep.py's already-hardened machinery
(GpuPool bin-packing, live GPU recheck right before each subprocess launch,
automatic retry on OOM/SIGKILL/"memory not enough" -- all added after this
box's other users were observed grabbing a GPU between acquire and launch)
instead of re-deriving it, via a straight import of that module by path.

Qwen3 needs a transformers that resolves the qwen3 architecture, which the
`duquant-qwen3` conda env (built from requirements_qwen3.txt) already
provides -- pass --qwen_python (default $QWEN3_PYTHON) at that env's python,
or --setup_qwen_env to (re)build it if it does not exist yet.
"""
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, os.path.join(SCRIPTS_DIR, 'gptq_smooth_sweep'))
import run_utils  # noqa: E402
from env_utils import subprocess_env  # noqa: E402
import run_sweep as rs  # noqa: E402  (GpuPool, wait_for_gpus_live, gpus_needed, ...)

CACHE_MODEL_IDS = {
    # models--<org>--<name> (HF hub cache dirname) -> the model id main.py wants.
    'meta-llama--Llama-2-7b-hf': 'meta-llama/Llama-2-7b-hf',
    'meta-llama--Llama-2-13b-hf': 'meta-llama/Llama-2-13b-hf',
    'meta-llama--Llama-3.1-8B': 'meta-llama/Llama-3.1-8B',
    'Qwen--Qwen3-8B': 'Qwen/Qwen3-8B',
    'Qwen--Qwen3-14B': 'Qwen/Qwen3-14B',
}


def discover_cached_models(hub_dir):
    found = []
    for d in sorted(glob.glob(os.path.join(hub_dir, 'models--*'))):
        key = os.path.basename(d)[len('models--'):]
        model_id = CACHE_MODEL_IDS.get(key)
        if model_id is None:
            # Fall back to the generic models--ORG--NAME -> ORG/NAME translation for
            # anything not in the table above (new models added to the cache later).
            parts = key.split('--')
            model_id = parts[0] + '/' + '--'.join(parts[1:]) if len(parts) > 1 else key
        found.append(model_id)
    return found


CONDITIONS = {
    'gptq_off': [],
    'gptq_on': ['--gptq'],
}
BASE_FLAGS = ['--quant_method', 'duquant', '--block_size', '32', '--permutation_times', '1']


def model_dir(args, model):
    return os.path.join(args.out_root, run_utils.short(model))


def result_path(args, model, name):
    return os.path.join(model_dir(args, model), f'{name}.json')


def python_for(args, model):
    if 'qwen' in model.lower() and args.qwen_python:
        return args.qwen_python
    return args.python


def build_cmd(args, model, name, extra_flags, out_json):
    logs = os.path.join(model_dir(args, model), 'logs')
    cmd = [
        python_for(args, model), os.path.join(REPO_ROOT, 'main.py'),
        '--model', model,
        '--wbits', str(args.wbits), '--abits', str(args.abits),
        '--eval_ppl', '--eval_datasets', 'wikitext2',
        '--nsamples', str(args.nsamples),
        '--seed', str(args.seed),
        '--batch_size', str(args.batch_size),
        '--output_dir', os.path.join(logs, name),
        '--log_name', 'log_rank0.txt',
        '--results_json', out_json,
    ] + BASE_FLAGS + extra_flags
    return cmd


def run_job(args, model, name, extra_flags, gpu_ids):
    out_json = result_path(args, model, name)
    cmd = build_cmd(args, model, name, extra_flags, out_json)
    if len(gpu_ids) > 1:
        cmd.append('--multigpu')

    env = subprocess_env(model)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    env.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

    logf = os.path.join(model_dir(args, model), 'logs', f'{name}.stdout')
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    label = f'{run_utils.short(model)}/{name}'
    if args.dry_run:
        with rs.PRINT_LOCK:
            print(f'\n{"=" * 78}\n[{label}] GPUs {gpu_ids}\n'
                  f'{" ".join(cmd)}\n  -> {logf}\n{"=" * 78}', flush=True)
        return 0

    oom_markers = ('out of memory', 'outofmemoryerror', 'acceleratorerror',
                  'memory not enough')
    max_attempts = args.max_oom_retries + 1
    for attempt in range(1, max_attempts + 1):
        rs.wait_for_gpus_live(gpu_ids, args.gpu_min_free_mb, label)
        with rs.PRINT_LOCK:
            print(f'\n{"=" * 78}\n[{label}] attempt {attempt}/{max_attempts} GPUs {gpu_ids}\n'
                  f'{" ".join(cmd)}\n  -> {logf}\n{"=" * 78}', flush=True)
        tick = time.time()
        with open(logf, 'w') as f:
            ret = subprocess.call(cmd, cwd=REPO_ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        mins = (time.time() - tick) / 60
        if ret == 0:
            with rs.PRINT_LOCK:
                print(f'[{label}] done in {mins:.1f} min (log: {logf})', flush=True)
            return ret
        with open(logf) as f:
            tail = f.read()[-4000:].lower()
        is_oom = any(m in tail for m in oom_markers) or ret == -9
        with rs.PRINT_LOCK:
            reason = 'OOM (someone else grabbed the GPU mid-run)' if is_oom else 'error'
            print(f'[{label}] FAILED (exit {ret}, {reason}) in {mins:.1f} min '
                  f'(log: {logf})', flush=True)
        if not is_oom or attempt == max_attempts:
            return ret
        with rs.PRINT_LOCK:
            print(f'[{label}] retrying ({attempt}/{max_attempts - 1} OOM retries used)...',
                  flush=True)
        time.sleep(30)
    return ret


def model_worker(args, model, pool, overrides):
    need = rs.gpus_needed(model, args.gpu_mem_gb, overrides)
    os.makedirs(model_dir(args, model), exist_ok=True)
    for name, extra_flags in CONDITIONS.items():
        p = result_path(args, model, name)
        if args.skip_existing and os.path.exists(p):
            with rs.PRINT_LOCK:
                print(f'[{run_utils.short(model)}/{name}] already has a result, skipping')
            continue
        gpu_ids = pool.acquire(need)
        try:
            run_job(args, model, name, extra_flags, gpu_ids)
        finally:
            pool.release(gpu_ids)


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def build_report(args):
    lines = [f'## DuQuant (block=32, perm=1, NO SmoothQuant), GPTQ off vs on -- '
            f'W{args.wbits}A{args.abits}\n',
            '| Model | GPTQ | wikitext2 PPL |', '|---|---|---|']
    for model in args.models:
        for name in ('gptq_off', 'gptq_on'):
            res = load_json(result_path(args, model, name))
            ppl = (res or {}).get('ppl', {}).get('wikitext2')
            lines.append(f'| {run_utils.short(model)} | '
                         f'{"on" if name == "gptq_on" else "off"} | '
                         f'{f"{ppl:.4f}" if isinstance(ppl, (int, float)) else "--"} |')
    return '\n'.join(lines)


def print_report(args):
    report = build_report(args)
    print('\n' + '=' * 78)
    print(report)
    out_path = os.path.join(args.out_root, 'SUMMARY.md')
    with open(out_path, 'w') as f:
        f.write(report + '\n')
    print(f'(written to {out_path})')


def build_parser():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', nargs='+', default=None,
                    help='default: every model currently in the local HF hub cache')
    ap.add_argument('--hub_cache', default=os.path.expanduser('~/.cache/huggingface/hub'))
    ap.add_argument('--wbits', type=int, default=4)
    ap.add_argument('--abits', type=int, default=4)
    ap.add_argument('--nsamples', type=int, default=128)
    ap.add_argument('--seed', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--gpu_mem_gb', type=float, default=24.0)
    ap.add_argument('--gpu_min_free_mb', type=int, default=20000)
    ap.add_argument('--max_oom_retries', type=int, default=5)
    ap.add_argument('--gpus', nargs='+', type=int, default=None)
    ap.add_argument('--gpus_per_model', nargs='+', default=None)
    ap.add_argument('--out_root', default=os.path.join(REPO_ROOT, 'log', 'duquant_gptq_ablation'))
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--qwen_python', default=os.environ.get('QWEN3_PYTHON'))
    ap.add_argument('--setup_qwen_env', action='store_true')
    ap.add_argument('--skip_existing', action='store_true')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--report_only', action='store_true')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.models is None:
        args.models = discover_cached_models(args.hub_cache)
    os.makedirs(args.out_root, exist_ok=True)
    overrides = rs.parse_overrides(args.gpus_per_model)

    if not args.report_only:
        if args.setup_qwen_env and any('qwen' in m.lower() for m in args.models):
            rs.ensure_qwen_env(args)

        if args.gpus:
            usable, skipped = list(args.gpus), []
        else:
            usable, skipped = rs.detect_usable_gpus(args.gpu_min_free_mb)
        if skipped:
            print(f'skipping GPU(s) already busy (< {args.gpu_min_free_mb}MB free): {skipped}')
        if not usable and not args.dry_run:
            sys.exit('no usable GPU found (see --gpu_min_free_mb / --gpus)')
        print(f'models: {args.models}')
        print(f'usable GPUs: {usable}')
        pool = rs.GpuPool(usable if usable else [0])

        with ThreadPoolExecutor(max_workers=max(1, len(args.models))) as ex:
            futs = {ex.submit(model_worker, args, m, pool, overrides): m for m in args.models}
            for fut in as_completed(futs):
                m = futs[fut]
                try:
                    fut.result()
                except Exception as e:
                    with rs.PRINT_LOCK:
                        print(f'[{run_utils.short(m)}] worker crashed: {e!r}')

    if not args.dry_run:
        print_report(args)


if __name__ == '__main__':
    main()
