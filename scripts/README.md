# scripts/ runners

Each rotation method has a python runner in the same shape as
`mxfp4_ablation.py`: a table of named **cells**, one `main.py` subprocess per
cell, a `--results_json` per cell, and a summary table at the end.

| runner | replaces | cells (default first) |
|---|---|---|
| `duquant/run_duquant.py` | `run_duquant.sh`, `run_duquant_Qwen3.sh` | `duquant`, `duquant_b128`, `duquant_nosmooth` |
| `duquantpp/run_duquantpp.py` | `run_duquantpp.sh`, `run_duquantpp_Qwen3.sh` | `duquantpp`, `duquantpp_diverse`, `duquantpp_nosmooth` |
| `hadamard/run_hadamard.py` | `run_hadamard.sh`, `run_hadamard_Qwen3.sh` | `hadamard`, `hadamard_smooth`, `hadamard_b128` |
| `torq/run_torq.py` | `run_torq.sh`, `run_torq_smooth.sh`, both `_Qwen3` | `torq`, `torq_smooth`, `torq_b128` |

`run_utils.py` holds everything they share; the `.sh` scripts are left in place
and still work.

## Why: one file set per (model, config)

The shell runners stamp wall-clock time into every artifact
(`<Model>_<cell>_w4a4_<YYYYmmdd_HHMMSS>.log`), and `main.py`'s own logger adds
`log_rank0_<epoch>.txt`, so **every** re-run left another generation of files
behind and finding "the" numbers for a config meant sorting timestamps by hand.

Here every path is a pure function of (method, model, config):

```
log/<method>/<Model>/<config>.json                     # results: config + PPL
log/<method>/<Model>/logs/<config>.stdout              # full stdout+stderr
log/<method>/<Model>/logs/<config>/<Model>_w4a4/log_rank0.txt   # main.py's logger
```

Re-running a cell overwrites that cell and nothing else, so the directory holds
exactly one file set per (model, config) however often it is run. The fixed
`log_rank0.txt` comes from `main.py --log_name`, added for this; without that
flag `main.py` keeps its old timestamped behaviour.

Pass `--skip_existing` to keep results that are already there, or `--out_root
<dir>` to park a run somewhere separate.

## Usage

```bash
# run_duquant.sh's two models, one per GPU, in parallel
python scripts/duquant/run_duquant.py --gpus 0 1

# Qwen3 -- no separate script; $QWEN3_PYTHON or --qwen_python picks the
# interpreter, otherwise .vendor/transformers_qwen3 goes on PYTHONPATH
python scripts/duquant/run_duquant.py --models Qwen/Qwen3-8B Qwen/Qwen3-14B --gpus 0 1

# more cells than the default, and the DuQuant++* (GPTQ) arm
python scripts/duquantpp/run_duquantpp.py --configs duquantpp duquantpp_diverse --gptq --gpus 0 1

# extra main.py flags pass through verbatim after `--`
python scripts/torq/run_torq.py -- --torq_k_top_frac 0.25 --torq_max_samples 4096

# print the commands without running them / re-print the table from the JSONs
python scripts/hadamard/run_hadamard.py --dry_run
python scripts/hadamard/run_hadamard.py --report_only
```

The table's FP16 column comes from `<Model>/fp16.json`, written by
`fp16_baseline.py`; `--fp16_root` says where to look (default
`log/mxfp4_ablation/`, where the existing baselines live).
