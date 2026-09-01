# coding=utf-8
"""Unquantized WikiText2 perplexity, measured with the exact same loop main.py uses.

    python scripts/fp16_baseline.py --model meta-llama/Llama-2-7b-hf

main.py cannot serve as its own FP16 baseline: UniformAffineQuantizer.forward runs
init_duquant (the rotation) *before* the `n_bits >= 16` early return, so a --wbits 16
--abits 16 run rotates activations without rotating the matching weights and reports
garbage. This loads the plain HF model instead and reuses main.py's testloader cache
and NLL loop so the numbers are directly comparable.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datautils import get_loaders  # noqa: E402


def model_family(net):
    n = net.lower()
    if "llama-3" in n or "llama3" in n:
        return "Llama3"
    if "llama-2" in n or "llama2" in n:
        return "Llama2"
    if "qwen" in n:
        return "Qwen3" if "qwen3" in n else "Qwen"
    return net.split("-")[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--results_json", default=None)
    args = ap.parse_args()

    net = args.model.rstrip("/").split("/")[-1]
    os.makedirs(args.cache_dir, exist_ok=True)
    cache = f"{args.cache_dir}/testloader_{model_family(net)}_wikitext2_all.cache"
    if os.path.exists(cache):
        testloader = torch.load(cache, weights_only=False)
        print(f"loaded testloader from {cache}")
    else:
        _, testloader = get_loaders("wikitext2", seed=args.seed, model=args.model,
                                    seqlen=args.seqlen)
        torch.save(testloader, cache)
    testenc = testloader.input_ids

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cpu")
    model.eval()
    model.config.use_cache = False
    model = model.cuda()

    nsamples = testenc.numel() // args.seqlen
    loss_fct = nn.CrossEntropyLoss()
    nlls = []
    for i in tqdm(range(nsamples)):
        batch = testenc[:, i * args.seqlen: (i + 1) * args.seqlen].cuda()
        logits = model(batch).logits
        shift_logits = logits[:, :-1, :]
        shift_labels = batch[:, 1:]
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1))
        nlls.append(loss.float() * args.seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen)).item()
    print(f"{net}  wikitext2 FP16 PPL = {ppl:.4f}")

    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump({"model": args.model, "datatype": "FP16",
                       "ppl": {"wikitext2": ppl}}, f, indent=2)


if __name__ == "__main__":
    main()
