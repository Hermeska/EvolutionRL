"""
vLLM sampling backend for the local E-SPL setup.

Training stays on the HF PEFT path (LocalTrainingClient) — vLLM is inference
only. Only *sampling* (rollouts, test eval, evolution reflection) is offloaded
to a local vLLM OpenAI-compatible server, which gives continuous batching over
the many long, variable-length reasoning generations per RL step.

Architecture (one SLURM job, one A100):
    training proc (spl env, HF PEFT)  --localhost HTTP-->  `vllm serve` (spl_vllm env)

Each RL step the current-policy LoRA adapter is saved to disk and hot-loaded into
vLLM; current-policy sampling uses that adapter, reference sampling uses the base
model. The training env never imports vllm, so vLLM's torch pin can't clobber it.

Drop-in for the `client.py` classes:
    service = VLLMServiceClient(base_url=..., served_model_name="Qwen/Qwen3-8B")
    training_client = service.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)
    ref_sampler = service.create_sampling_client(base_model="Qwen/Qwen3-8B")   # base, no LoRA
"""

import concurrent.futures
import logging
import os
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

from tinker import types
from tinker_cookbook.local_backend.client import (
    LocalFuture,
    LocalServiceClient,
    LocalTrainingClient,
)

logger = logging.getLogger(__name__)

# Fire sampling requests concurrently so vLLM's continuous batcher sees the whole
# step's ~hundreds of sequences at once instead of one problem's group at a time.
# The recipe already builds all futures first and collects later, so async .sample()
# gives cross-request concurrency for free.
_CLIENT_WORKERS = int(os.environ.get("VLLM_CLIENT_WORKERS", "128"))
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=_CLIENT_WORKERS)
_SESSION = requests.Session()
_SESSION.mount("http://", HTTPAdapter(pool_maxsize=_CLIENT_WORKERS, max_retries=0))


class _AsyncFuture:
    """Future over an in-flight HTTP sampling request; .result() blocks on it."""

    def __init__(self, fut: "concurrent.futures.Future"):
        self._fut = fut

    def result(self):
        return self._fut.result()

    async def result_async(self):
        return self._fut.result()

# One rotating adapter name/dir: the current policy changes every step, so we
# overwrite rather than accumulate adapters in vLLM and on disk.
_POLICY_LORA_NAME = "policy_current"
_POLICY_ADAPTER_SUBDIR = "adapter_current"


# ---------------------------------------------------------------------------
# Sampling client (HTTP -> vLLM /v1/completions)
# ---------------------------------------------------------------------------

class VLLMSamplingClient:
    """Drop-in for LocalSamplingClient that samples from a vLLM server.

    `model` is either the served base-model name (reference policy, no LoRA) or a
    loaded LoRA adapter name (current policy).
    """

    def __init__(self, base_url: str, model: str, timeout: float = 3600.0):
        self._url = base_url.rstrip("/") + "/v1/completions"
        self._model = model
        self._timeout = timeout

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int = 1,
        sampling_params: Optional[types.SamplingParams] = None,
    ) -> _AsyncFuture:
        sp = sampling_params or types.SamplingParams(
            max_tokens=512, temperature=0.7, top_p=0.95, stop=[]
        )
        stop_strs = [s for s in (sp.stop or []) if isinstance(s, str)]
        stop_ids = [s for s in (sp.stop or []) if isinstance(s, int)]

        body = {
            "model": self._model,
            # Token-id prompt: bypass re-tokenization so ids match the trainer exactly.
            "prompt": prompt.to_ints(),
            "n": max(1, num_samples),
            "max_tokens": sp.max_tokens,
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            # logprobs=1 -> response carries the chosen token's logprob per position.
            "logprobs": 1,
            # tokens come back as "token_id:<int>" so we recover exact ids + logprobs.
            "return_tokens_as_token_ids": True,
        }
        if stop_strs:
            body["stop"] = stop_strs
        if stop_ids:
            body["stop_token_ids"] = stop_ids

        # Fire the request in a worker thread and return immediately; the recipe
        # collects .result() later, so all of a step's requests run concurrently.
        return _AsyncFuture(_EXECUTOR.submit(self._post, body))

    def _post(self, body) -> types.SampleResponse:
        resp = _SESSION.post(self._url, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        sequences = []
        for choice in data["choices"]:
            lp = choice.get("logprobs") or {}
            raw_tokens = lp.get("tokens", [])
            token_logprobs = lp.get("token_logprobs", [])
            tokens = [_parse_token_id(t) for t in raw_tokens]
            logprobs = [float(x) if x is not None else 0.0 for x in token_logprobs]
            sequences.append(
                types.SampledSequence(
                    tokens=tokens,
                    logprobs=logprobs,
                    stop_reason=choice.get("finish_reason", "stop") or "stop",
                )
            )
        return types.SampleResponse(sequences=sequences)


def _parse_token_id(tok) -> int:
    """Parse vLLM's return_tokens_as_token_ids format ('token_id:1234' -> 1234)."""
    if isinstance(tok, int):
        return tok
    return int(str(tok).split(":")[-1])


# ---------------------------------------------------------------------------
# Training client (HF training + push LoRA adapter to vLLM)
# ---------------------------------------------------------------------------

class VLLMTrainingClient(LocalTrainingClient):
    """HF PEFT training (unchanged) that, on save-for-sampler, writes the current
    LoRA adapter to disk and hot-loads it into the vLLM server."""

    def __init__(self, shared_model, checkpoint_dir: str, base_url: str):
        super().__init__(shared_model, checkpoint_dir=checkpoint_dir)
        self._base_url = base_url.rstrip("/")

    def save_weights_for_sampler(self, name: str = "") -> LocalFuture:
        adapter_dir = os.path.join(self._checkpoint_dir, _POLICY_ADAPTER_SUBDIR)
        self._sm.save_lora(adapter_dir)  # PEFT save_pretrained -> adapter_model.safetensors
        self._reload_policy_lora(adapter_dir)
        return LocalFuture(types.SaveWeightsForSamplerResponse(path=f"vllm:{_POLICY_LORA_NAME}"))

    async def save_weights_for_sampler_async(self, name: str) -> LocalFuture:
        return self.save_weights_for_sampler(name)

    def _reload_policy_lora(self, adapter_dir: str):
        # Unload the previous version (ignore if absent), then load the new one.
        try:
            requests.post(
                f"{self._base_url}/v1/unload_lora_adapter",
                json={"lora_name": _POLICY_LORA_NAME},
                timeout=120,
            )
        except requests.RequestException:
            pass
        resp = requests.post(
            f"{self._base_url}/v1/load_lora_adapter",
            json={"lora_name": _POLICY_LORA_NAME, "lora_path": os.path.abspath(adapter_dir)},
            timeout=600,
        )
        resp.raise_for_status()
        logger.info(f"vLLM loaded policy adapter '{_POLICY_LORA_NAME}' from {adapter_dir}")


# ---------------------------------------------------------------------------
# Service client (factory)
# ---------------------------------------------------------------------------

class VLLMServiceClient(LocalServiceClient):
    """Drop-in for LocalServiceClient: HF model for training, vLLM for sampling."""

    def __init__(
        self,
        base_url: str,
        served_model_name: str,
        device: Optional[str] = None,
        checkpoint_dir: str = "/tmp/espl_checkpoints",
        health_timeout: float = 1200.0,
        grad_checkpointing: bool = False,
    ):
        super().__init__(device=device, checkpoint_dir=checkpoint_dir,
                         grad_checkpointing=grad_checkpointing)
        self._base_url = base_url.rstrip("/")
        self._served = served_model_name
        self._wait_healthy(health_timeout)

    def _wait_healthy(self, timeout: float):
        deadline = time.monotonic() + timeout
        url = f"{self._base_url}/health"
        while time.monotonic() < deadline:
            try:
                if requests.get(url, timeout=10).status_code == 200:
                    logger.info(f"vLLM server healthy at {self._base_url}")
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise RuntimeError(f"vLLM server at {self._base_url} not healthy after {timeout}s")

    def create_lora_training_client(self, base_model: str, rank: int = 16) -> VLLMTrainingClient:
        sm = self._get_or_create_model(base_model, lora_rank=rank)
        return VLLMTrainingClient(sm, checkpoint_dir=self.checkpoint_dir, base_url=self._base_url)

    def create_training_client_from_state(self, state_path: str, rank: int = 16) -> VLLMTrainingClient:
        # On resume the HF training model may not exist yet: unlike the pure-HF
        # backend, our reference sampler goes to vLLM and never builds it. Create
        # it from the served base model, then load the saved LoRA/optimizer state.
        if self._shared_model is None:
            self._get_or_create_model(self._served, lora_rank=rank)
        elif self._shared_model.lora_rank == 0:
            self._get_or_create_model(self._shared_model.model_name, lora_rank=rank)
        client = VLLMTrainingClient(self._shared_model, checkpoint_dir=self.checkpoint_dir, base_url=self._base_url)
        client.load_state(state_path)
        return client

    def create_sampling_client(
        self,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> VLLMSamplingClient:
        if model_path is not None and model_path.startswith("vllm:"):
            return VLLMSamplingClient(self._base_url, model_path[len("vllm:"):])
        # base_model / reference sampling -> served base model, no LoRA
        return VLLMSamplingClient(self._base_url, self._served)