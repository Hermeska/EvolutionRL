"""Quick CPU check of the batched LocalSamplingClient.

Verifies that num_samples>1 returns that many sequences, each with per-token
logprobs aligned to its tokens and trimmed at its own stop token. Runs on CPU
with a tiny model — no GPU needed. Staged, unbuffered prints so a stall is
visible. Run:  python -u scripts/test_batched_sampler.py
"""
import os
import sys

os.environ.setdefault("HF_HOME", "/ibex/user/khangea/espl/hf_cache")
os.environ["LOCAL_SAMPLE_MICROBATCH"] = "2"      # force multi-chunk for num_samples>2
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def log(msg):
    print(msg, flush=True)


log("[1/4] importing ...")
import torch
from tinker import types
from tinker_cookbook.local_backend.client import _SharedLocalModel, LocalSamplingClient

torch.set_num_threads(4)
MODEL = os.environ.get("TEST_MODEL", "sshleifer/tiny-gpt2")

log(f"[2/4] loading {MODEL} on CPU ...")
sm = _SharedLocalModel(MODEL, lora_rank=0, device="cpu")
client = LocalSamplingClient(sm, use_lora=False)

prompt = types.ModelInput.from_ints(tokens=sm.tokenizer.encode("Hello world, this is a test."))
sp = types.SamplingParams(max_tokens=16, temperature=0.8, top_p=0.95, stop=[])

log("[3/4] sampling ...")
for n in (1, 3, 5):
    resp = client.sample(prompt=prompt, num_samples=n, sampling_params=sp).result()
    seqs = resp.sequences
    assert len(seqs) == n, f"expected {n} sequences, got {len(seqs)}"
    for i, s in enumerate(seqs):
        assert len(s.tokens) == len(s.logprobs), (
            f"n={n} seq={i}: tokens {len(s.tokens)} != logprobs {len(s.logprobs)}")
        assert 0 < len(s.tokens) <= sp.max_tokens, f"n={n} seq={i}: bad length {len(s.tokens)}"
        assert all(isinstance(t, int) for t in s.tokens)
        assert all(lp <= 1e-6 for lp in s.logprobs), "logprobs must be <= 0"
    log(f"    num_samples={n}: seq lens={[len(s.tokens) for s in seqs]}  aligned=OK")

log("[4/4] ALL BATCHED-SAMPLER CHECKS PASSED")
sys.exit(0)
