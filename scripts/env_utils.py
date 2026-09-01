# coding=utf-8
"""Shared subprocess-env helper for the scripts/ ablation runners.

The shared `flatquant` conda env this repo runs in pins an older transformers
(older than even requirements.txt's 4.43.1) that predates Qwen3, so
`AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-*")` can't resolve the
architecture. See .vendor/README.md for why this vendors a newer transformers
via `pip install --target` instead of upgrading the shared env in place.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
QWEN_TRANSFORMERS_DIR = os.path.join(REPO_ROOT, ".vendor", "transformers_qwen3")


def subprocess_env(model, base_env=None):
    """Env dict for launching main.py / fp16_baseline.py as a subprocess.

    Prepends the vendored Qwen3-capable transformers to PYTHONPATH whenever
    `model` looks like a Qwen model, so the installed (older) transformers in
    site-packages doesn't shadow it.
    """
    env = dict(base_env if base_env is not None else os.environ)
    if "qwen" in model.lower() and os.path.isdir(QWEN_TRANSFORMERS_DIR):
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (QWEN_TRANSFORMERS_DIR + os.pathsep + existing
                             if existing else QWEN_TRANSFORMERS_DIR)
    return env
