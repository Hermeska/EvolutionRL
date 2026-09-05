"""vLLM-backed sampling compatible with the small subset of Tinker used by E-SPL."""

from __future__ import annotations

import logging
import multiprocessing
import os
import queue
import threading
import time
import traceback
import atexit
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Optional

from tinker import types

from .future import LocalFuture

logger = logging.getLogger(__name__)


@dataclass
class _Request:
    prompt_ids: list[int]
    num_samples: int
    sampling_params: types.SamplingParams
    result: Future
    seed: Optional[int] = None


def _expand_multi_sample_requests(
    prompts: list[dict],
    raw_params: list[dict],
) -> tuple[list[dict], list[dict], list[int]]:
    """Turn vLLM ``n>1`` requests into independent ``n=1`` requests.

    vLLM 0.10.x can leak an extra parent ``RequestOutput`` from multi-sample
    requests into a later offline ``generate`` call.  Keeping every engine
    request at ``n=1`` avoids that parent/child path while preserving the
    public one-request/many-completions contract of this module.
    """
    expanded_prompts: list[dict] = []
    expanded_params: list[dict] = []
    owners: list[int] = []
    for request_index, (prompt, raw_param) in enumerate(zip(prompts, raw_params)):
        num_samples = max(int(raw_param.get("n", 1)), 1)
        for sample_index in range(num_samples):
            child_params = dict(raw_param)
            child_params["n"] = 1
            if child_params.get("seed") is not None:
                child_params["seed"] = int(child_params["seed"]) + sample_index
            expanded_prompts.append(prompt)
            expanded_params.append(child_params)
            owners.append(request_index)
    return expanded_prompts, expanded_params, owners


def _select_current_request_outputs(
    raw_outputs: list,
    first_request_id: int,
    request_count: int,
) -> tuple[list, list[str], list[str]]:
    """Select only outputs created by the current ``LLM.generate`` call."""
    expected_ids = [str(first_request_id + index) for index in range(request_count)]
    expected_set = set(expected_ids)
    current_outputs = {}
    ignored_ids = []
    duplicate_ids = []
    for output in raw_outputs:
        request_id = str(output.request_id)
        if request_id not in expected_set:
            ignored_ids.append(request_id)
            continue
        if request_id in current_outputs:
            duplicate_ids.append(request_id)
            previous = current_outputs[request_id]
            previous_tokens = sum(
                len(completion.token_ids) for completion in previous.outputs
            )
            current_tokens = sum(
                len(completion.token_ids) for completion in output.outputs
            )
            if current_tokens <= previous_tokens:
                continue
        current_outputs[request_id] = output

    missing_ids = [
        request_id
        for request_id in expected_ids
        if request_id not in current_outputs
    ]
    if missing_ids:
        raise RuntimeError(
            "vLLM did not return current request IDs: " + ", ".join(missing_ids)
        )
    return (
        [current_outputs[request_id] for request_id in expected_ids],
        ignored_ids,
        duplicate_ids,
    )


class VLLMEngine:
    """Own a vLLM subprocess restricted to the dedicated sampling GPU."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.9,
        max_lora_rank: int = 32,
        max_num_seqs: int = 64,
        speculative_model: Optional[str] = None,
        num_speculative_tokens: int = 15,
        attention_backend: Optional[str] = None,
    ):
        logger.info(
            "Loading vLLM sampler for %s on %s (max_model_len=%s, gpu_memory_utilization=%.2f)",
            model_name,
            device,
            max_model_len,
            gpu_memory_utilization,
        )
        device_index = device.split(":", 1)[1] if ":" in device else device
        context = multiprocessing.get_context("spawn")
        self._parent_connection, child_connection = context.Pipe()
        engine_kwargs = {
            "model": model_name,
            "dtype": dtype,
            "trust_remote_code": True,
            "enable_lora": True,
            "max_lora_rank": max_lora_rank,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if attention_backend:
            engine_kwargs["attention_backend"] = attention_backend
        if speculative_model:
            engine_kwargs["speculative_config"] = {
                "method": "dflash",
                "model": speculative_model,
                "num_speculative_tokens": max(int(num_speculative_tokens), 1),
                "attention_backend": attention_backend or "FLASH_ATTN",
            }
            logger.info(
                "Enabling DFlash draft %s (%s speculative tokens)",
                speculative_model,
                num_speculative_tokens,
            )

        self._process = context.Process(
            target=_engine_worker_main,
            args=(
                child_connection,
                device_index,
                engine_kwargs,
            ),
            name="espl-vllm-gpu-worker",
        )
        self._process.start()
        if not self._parent_connection.poll(900):
            self._process.terminate()
            self._process.join(timeout=5)
            raise TimeoutError("vLLM worker did not start within 15 minutes")
        status, payload = self._parent_connection.recv()
        if status != "ready":
            self._process.join(timeout=5)
            raise RuntimeError(f"vLLM worker failed during startup:\n{payload}")
        self._lock = threading.Lock()
        atexit.register(self.close)

    def generate(self, requests: list[_Request], adapter_path: Optional[str], adapter_id: int):
        prompts = [request.prompt_ids for request in requests]
        params = []
        for request in requests:
            source = request.sampling_params
            stop_strings = [item for item in (source.stop or []) if isinstance(item, str)]
            stop_token_ids = [item for item in (source.stop or []) if isinstance(item, int)]
            raw_params = {
                "n": max(int(request.num_samples), 1),
                "max_tokens": int(source.max_tokens),
                "temperature": float(source.temperature),
                "top_p": float(source.top_p),
                "stop": stop_strings or None,
                "stop_token_ids": stop_token_ids or None,
                "logprobs": 1,
            }
            if request.seed is not None:
                raw_params["seed"] = int(request.seed)
            params.append(raw_params)

        started = time.monotonic()
        with self._lock:
            self._parent_connection.send(
                ("generate", prompts, params, adapter_path, adapter_id)
            )
            status, outputs = self._parent_connection.recv()
            if status != "ok":
                raise RuntimeError(f"vLLM generation failed:\n{outputs}")
        logger.info(
            "vLLM completed a batch of %s prompts in %.2fs",
            len(requests),
            time.monotonic() - started,
        )
        return outputs

    def close(self):
        process = getattr(self, "_process", None)
        if process is None or not process.is_alive():
            return
        try:
            self._parent_connection.send(("close",))
        except (BrokenPipeError, EOFError, OSError):
            pass
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


class VLLMSamplingClient:
    """Collect nearby sample calls and submit them to vLLM as one batch."""

    def __init__(
        self,
        engine: VLLMEngine,
        adapter_path: Optional[str] = None,
        adapter_id: int = 0,
        batch_wait_ms: int = 25,
        max_batch_prompts: int = 128,
    ):
        self._engine = engine
        self._adapter_path = adapter_path
        self._adapter_id = adapter_id
        self._batch_wait_s = max(batch_wait_ms, 0) / 1000
        self._max_batch_prompts = max(int(max_batch_prompts), 1)
        self._queue: queue.Queue[_Request] = queue.Queue()
        self._worker = threading.Thread(
            target=self._batch_loop,
            name="espl-vllm-batcher",
            daemon=True,
        )
        self._worker.start()

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int = 1,
        sampling_params: Optional[types.SamplingParams] = None,
    ) -> LocalFuture:
        params = sampling_params or types.SamplingParams(
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            stop=[],
        )
        result = Future()
        self._queue.put(
            _Request(
                prompt_ids=prompt.to_ints(),
                num_samples=num_samples,
                sampling_params=params,
                result=result,
            )
        )
        return LocalFuture(result)

    def sample_seeded(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        seed: int,
    ) -> LocalFuture:
        """Sample deterministically for paired prompt comparisons."""
        result = Future()
        self._queue.put(
            _Request(
                prompt_ids=prompt.to_ints(),
                num_samples=num_samples,
                sampling_params=sampling_params,
                result=result,
                seed=int(seed),
            )
        )
        return LocalFuture(result)

    def _batch_loop(self):
        while True:
            first = self._queue.get()
            requests = [first]
            if self._batch_wait_s:
                time.sleep(self._batch_wait_s)
            while len(requests) < self._max_batch_prompts:
                try:
                    requests.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            try:
                outputs = self._engine.generate(
                    requests,
                    adapter_path=self._adapter_path,
                    adapter_id=self._adapter_id,
                )
                if len(outputs) != len(requests):
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} results for {len(requests)} prompts"
                    )
                for request, output in zip(requests, outputs):
                    request.result.set_result(_to_tinker_response(output))
            except BaseException as exc:
                for request in requests:
                    if not request.result.done():
                        request.result.set_exception(exc)


def _to_tinker_response(request_output) -> types.SampleResponse:
    sequences = []
    for completion in request_output:
        token_ids = completion["token_ids"]
        token_logprobs = completion["logprobs"]
        if len(token_logprobs) != len(token_ids):
            raise RuntimeError("vLLM token/logprob lengths do not match")
        sequences.append(
            types.SampledSequence(
                tokens=token_ids,
                logprobs=token_logprobs,
                stop_reason=str(completion["finish_reason"] or "stop"),
            )
        )
    return types.SampleResponse(sequences=sequences)


def _engine_worker_main(connection, device_index: str, engine_kwargs: dict):
    """Initialize vLLM after restricting this spawned process to one GPU."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_index)
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(**engine_kwargs)
        connection.send(("ready", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        return

    while True:
        try:
            message = connection.recv()
            if message[0] == "close":
                return
            _, prompt_ids, raw_params, adapter_path, adapter_id = message
            lora_request = None
            if adapter_path is not None:
                lora_request = LoRARequest(
                    lora_name=f"espl-step-{adapter_id}",
                    lora_int_id=adapter_id,
                    lora_path=adapter_path,
                )
            prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
            expanded_prompts, expanded_raw_params, owners = (
                _expand_multi_sample_requests(prompts, raw_params)
            )
            first_request_id = int(llm.request_counter.counter)
            raw_outputs = llm.generate(
                expanded_prompts,
                sampling_params=[
                    SamplingParams(**item) for item in expanded_raw_params
                ],
                lora_request=lora_request,
                use_tqdm=False,
            )
            expanded_outputs, ignored_ids, duplicate_ids = (
                _select_current_request_outputs(
                    raw_outputs,
                    first_request_id=first_request_id,
                    request_count=len(expanded_prompts),
                )
            )
            if ignored_ids:
                logger.warning(
                    "Ignored %s stale vLLM outputs from earlier calls: %s",
                    len(ignored_ids),
                    ", ".join(ignored_ids[:10]),
                )
            if duplicate_ids:
                logger.warning(
                    "Deduplicated %s repeated current vLLM outputs: %s",
                    len(duplicate_ids),
                    ", ".join(duplicate_ids[:10]),
                )

            outputs = [[] for _ in prompts]
            for owner, request_output in zip(owners, expanded_outputs):
                for completion in request_output.outputs:
                    token_ids = list(completion.token_ids)
                    logprobs = []
                    for token_id, candidates in zip(
                        token_ids, completion.logprobs or []
                    ):
                        selected = candidates.get(token_id)
                        if selected is None:
                            raise RuntimeError(
                                f"Missing vLLM logprob for sampled token {token_id}"
                            )
                        logprobs.append(float(selected.logprob))
                    outputs[owner].append(
                        {
                            "token_ids": token_ids,
                            "logprobs": logprobs,
                            "finish_reason": completion.finish_reason,
                        }
                    )
            for request_index, (request_output, raw_param) in enumerate(
                zip(outputs, raw_params)
            ):
                expected = max(int(raw_param.get("n", 1)), 1)
                if len(request_output) != expected:
                    raise RuntimeError(
                        f"vLLM produced {len(request_output)} completions for "
                        f"request {request_index}; expected {expected}"
                    )
            connection.send(("ok", outputs))
        except EOFError:
            return
        except BaseException:
            connection.send(("error", traceback.format_exc()))
