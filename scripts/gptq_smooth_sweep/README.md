# GPTQ + Hadamard: No-Smooth vs. alpha-searched Smooth

`run_sweep.py` (+ `run_sweep.sh` wrapper) runs, for each of `--models`:

- **exp1_nosmooth** -- `--quant_method hadamard --gptq`, no `--smooth`.
- **exp2_best** -- `--quant_method hadamard --gptq --smooth --alpha <best>`,
  where `<best>` is chosen from `--alphas` (default `0.3 0.4 0.5 0.6 0.7 0.8`)
  by lowest `--select_dataset` PPL. Each candidate is run PPL-only first
  (`exp2_alpha<a>.json`, no `--tasks` -- zero-shot is expensive and would
  otherwise re-run once per candidate); the winner is then re-run once more
  WITH `--tasks` to get its zero-shot numbers too.

Every `main.py` flag used matches the existing CLI (`--wbits 4`, `--abits 4`,
`--eval_ppl`, ...); nothing about `main.py`'s argument parsing changed. The
one `main.py` change this adds is threading zero-shot task accuracy into
`--results_json`'s `"tasks"` key -- it used to only reach the log text.

## "Validation" PPL, honestly

This repo's `datautils.get_wikitext2` has no held-out validation split: it
returns the wikitext2 **train** split (for calibration) and **test** split
(what `--eval_ppl` reports). So there is no leakage-free validation set to
search `--alpha` against beyond the same test PPL `--eval_ppl` already
prints -- `--select_dataset` (default `wikitext2`) *is* that number. This
matches how the rest of this repo already treats alpha (e.g.
`scripts/mxfp4_ablation.py` just fixes `alpha=0.6`); it is a real limitation
worth knowing about, not hidden by this script.

## GPU scheduling

GPUs are pulled from one shared pool, sized per model by a params-from-name
heuristic (`gpus_needed`; override with `--gpus_per_model MODEL=N`) against
`--gpu_mem_gb` (default 24, this machine's 3090s). Only cards with
`>= --gpu_min_free_mb` (default 20000) free *right now* go into the pool, so
a GPU someone else on a shared box is already using is left alone -- check
with `nvidia-smi` before a big run; `--gpus 0 1 2 ...` pins an exact set
instead of auto-detecting.

Within one model exp1 -> alpha search -> exp2_best run in that order (the
alpha winner is a real dependency); different models run in parallel threads,
each blocking on the pool until its GPU(s) are free, which is what keeps
several models' jobs interleaved across whatever cards are actually open
without manual round-robin.

## Qwen3

The shared `flatquant` env's transformers predates the qwen3 architecture.
`--setup_qwen_env` (what `run_sweep.sh` always passes) conda-creates
`duquant-qwen3` and does `pip install -r requirements_qwen3.txt` **into
that env** -- not the shared one -- before any Qwen `main.py` subprocess is
launched, and is a no-op on a re-run if that env already has
`transformers>=4.51`. `--qwen_python /path/to/python` (or `$QWEN3_PYTHON`)
skips this and uses an interpreter you prepared yourself instead.

## Usage

```bash
# see the exact commands without running anything
python scripts/gptq_smooth_sweep/run_sweep.py --dry_run

# Llama-only smoke test on 2 GPUs, skipping what's already there
bash scripts/gptq_smooth_sweep/run_sweep.sh \
    --models meta-llama/Llama-2-7b-hf meta-llama/Meta-Llama-3.1-8B \
    --gpus 0 1 --skip_existing

# the full 5-model sweep, auto-detecting free GPUs
bash scripts/gptq_smooth_sweep/run_sweep.sh --skip_existing

# re-print the Markdown table from whatever JSONs already exist
python scripts/gptq_smooth_sweep/run_sweep.py --report_only
```

Artifacts, one file set per (model, config), safe to re-run
(`--skip_existing` keeps what's there):

```
log/gptq_smooth_sweep/<Model>/exp1_nosmooth.json
log/gptq_smooth_sweep/<Model>/exp2_alpha<a>.json      # one per --alphas candidate
log/gptq_smooth_sweep/<Model>/exp2_best_alpha.json    # {best_alpha, search: {alpha: ppl}}
log/gptq_smooth_sweep/<Model>/exp2_best.json
log/gptq_smooth_sweep/<Model>/logs/<config>.stdout
log/gptq_smooth_sweep/<Model>/logs/<config>/<Model>_w4a4/log_rank0.txt
log/gptq_smooth_sweep/SUMMARY.md                      # the Markdown report
```
