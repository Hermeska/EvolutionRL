"""Correctness gate for the vLLM sampling backend.

Runs in the `spl` env against an already-running vLLM server. Samples a prompt
via VLLMSamplingClient, then teacher-forces the returned tokens through the HF
Qwen3-8B model and compares per-token logprobs. The training loss computes
new_logprobs = log_softmax(raw logits)[token] (temperature=1, no top_p), so for
the importance ratio exp(new-old) to be well-behaved, vLLM's returned (old)
logprobs must match that convention.

Two checks:
  1. temp=1.0, top_p=1.0  -> all logprob conventions collapse; vLLM must match HF.
     (validates token-id round-trip + alignment + basic correctness)
  2. temp=0.7             -> reveals whether vLLM scales logprobs by temperature.

Usage (on a GPU node, spl env, with vllm_serve.sh already running):
    python scripts/validate_vllm_logprobs.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-8B
"""
import argparse
import os

import torch
import torch.nn.functional as F

from tinker import types
from tinker_cookbook.local_backend.client import _SharedLocalModel
from tinker_cookbook.local_backend.vllm_client import VLLMSamplingClient


def hf_teacher_forced_logprobs(sm, prompt_ids, gen_ids):
    """Raw (temperature=1) log P(gen token | context) via one HF forward pass —
    exactly the convention _compute_datum_loss uses for new_logprobs."""
    full = torch.tensor(prompt_ids + gen_ids, dtype=torch.long, device=sm.device).unsqueeze(0)
    with torch.no_grad():
        logits = sm.model(input_ids=full).logits[0]  # [L, vocab]
    out = []
    P = len(prompt_ids)
    for t, tok in enumerate(gen_ids):
        lp = F.log_softmax(logits[P + t - 1].float(), dim=-1)[tok].item()
        out.append(lp)
    return out


def compare(tag, vllm_lp, hf_lp):
    n = min(len(vllm_lp), len(hf_lp))
    if n == 0:
        print(f"[{tag}] EMPTY sequence — cannot compare"); return False
    diffs = [abs(vllm_lp[i] - hf_lp[i]) for i in range(n)]
    mean_d = sum(diffs) / n
    max_d = max(diffs)
    # ratio check: if vLLM scales by temperature, vllm/hf would be ~constant != 1
    ratios = [vllm_lp[i] / hf_lp[i] for i in range(n) if abs(hf_lp[i]) > 1e-3]
    mean_r = sum(ratios) / len(ratios) if ratios else float("nan")
    print(f"[{tag}] n={n}  mean|Δ|={mean_d:.4f}  max|Δ|={max_d:.4f}  mean(vllm/hf)={mean_r:.3f}")
    print(f"        vllm[:5]={[round(x,3) for x in vllm_lp[:5]]}")
    print(f"        hf  [:5]={[round(x,3) for x in hf_lp[:5]]}")
    return mean_d < 0.10  # bf16 kernel noise is ~0.01-0.05; 0.10 is a generous PASS bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompt", default="Solve step by step: what is 17 times 23? Answer:")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/ibex/user/khangea/espl/hf_cache")
    print(f"Loading HF {args.model} (bf16) for teacher-forcing ...")
    sm = _SharedLocalModel(args.model, lora_rank=0, device="cuda")  # base = matches vLLM served base
    prompt_ids = sm.tokenizer.encode(args.prompt)
    model_input = types.ModelInput.from_ints(tokens=prompt_ids)
    client = VLLMSamplingClient(args.base_url, args.model)

    ok = True
    for tag, temp, top_p in [("temp=1.0,top_p=1.0", 1.0, 1.0), ("temp=0.7", 0.7, 0.95)]:
        sp = types.SamplingParams(max_tokens=args.max_tokens, temperature=temp, top_p=top_p, stop=[])
        seq = client.sample(prompt=model_input, num_samples=1, sampling_params=sp).result().sequences[0]
        hf_lp = hf_teacher_forced_logprobs(sm, prompt_ids, seq.tokens)
        passed = compare(tag, seq.logprobs, hf_lp)
        ok = ok and (passed or tag.startswith("temp=0.7"))  # temp=1 must pass; temp=0.7 is diagnostic
        print()

    print("=" * 60)
    if ok:
        print("PASS: vLLM logprobs match HF at temp=1 -> consistent with the training loss.")
        print("      (Check the temp=0.7 line: if mean|Δ| is also small, vLLM returns raw")
        print("       logprobs and we're fully safe; if it's off by a ~constant ratio,")
        print("       vLLM scales by temperature and we must correct before a real run.)")
    else:
        print("FAIL: vLLM logprobs disagree with HF at temp=1 — do NOT trust for training.")
        print("      Likely a token-id/alignment bug or a logprob-convention mismatch.")


if __name__ == "__main__":
    main()
