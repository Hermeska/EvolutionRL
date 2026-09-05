"""SGLang-backed sampling for the subset of the Tinker API used by E-SPL."""

from __future__ import annotations

import atexit
import json
import logging
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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


class SGLangEngine:
    """Run an SGLang HTTP server on a dedicated sampling GPU."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda:1",
        dtype: str = "bfloat16",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.88,
        max_running_requests: int = 64,
        startup_timeout_s: int = 900,
    ):
        device_index = device.split(":", 1)[1] if ":" in device else device
        self._port = _find_available_port()
        self._base_url = f"http://127.0.0.1:{self._port}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_index)
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--dtype",
            dtype,
            "--attention-backend",
            "triton",
            "--sampling-backend",
            "pytorch",
            "--context-length",
            str(max_model_len),
            "--mem-fraction-static",
            str(gpu_memory_utilization),
            "--max-running-requests",
            str(max_running_requests),
            "--trust-remote-code",
        ]
        logger.info(
            "Starting SGLang sampler for %s on %s "
            "(max_model_len=%s, gpu_memory_utilization=%.2f)",
            model_name,
            device,
            max_model_len,
            gpu_memory_utilization,
        )
        self._process = subprocess.Popen(command, env=env)
        try:
            self._wait_until_ready(startup_timeout_s)
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _wait_until_ready(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        health_urls = [
            f"{self._base_url}/health_generate",
            f"{self._base_url}/health",
        ]
        last_error: Optional[BaseException] = None
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"SGLang server exited during startup with code {return_code}"
                )
            for health_url in health_urls:
                try:
                    with urllib.request.urlopen(health_url, timeout=5) as response:
                        if 200 <= response.status < 300:
                            logger.info("SGLang sampler is ready at %s", self._base_url)
                            return
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
            time.sleep(2)
        raise TimeoutError(
            f"SGLang server did not start within {timeout_s}s: {last_error}"
        )

    def generate(self, requests: list[_Request]):
        flat_prompts: list[list[int]] = []
        flat_params: list[dict] = []
        sample_counts: list[int] = []
        for request in requests:
            source = request.sampling_params
            stop_strings = [item for item in (source.stop or []) if isinstance(item, str)]
            stop_token_ids = [item for item in (source.stop or []) if isinstance(item, int)]
            count = max(int(request.num_samples), 1)
            sample_counts.append(count)
            for sample_index in range(count):
                raw_params = {
                    "max_new_tokens": int(source.max_tokens),
                    "temperature": float(source.temperature),
                    "top_p": float(source.top_p),
                    "stop": stop_strings or None,
                    "stop_token_ids": stop_token_ids or None,
                }
                if request.seed is not None:
                    raw_params["seed"] = int(request.seed) + sample_index
                flat_prompts.append(request.prompt_ids)
                flat_params.append(raw_params)

        payload = {
            "input_ids": flat_prompts,
            "sampling_params": flat_params,
            "return_logprob": [True] * len(flat_prompts),
            "logprob_start_len": [-1] * len(flat_prompts),
            "top_logprobs_num": [0] * len(flat_prompts),
            "stream": False,
        }
        started = time.monotonic()
        raw_outputs = self._post_json("/generate", payload)
        if isinstance(raw_outputs, dict):
            raw_outputs = [raw_outputs]
        if len(raw_outputs) != len(flat_prompts):
            raise RuntimeError(
                f"SGLang returned {len(raw_outputs)} results for "
                f"{len(flat_prompts)} flattened samples"
            )

        outputs = []
        offset = 0
        for count in sample_counts:
            outputs.append([
                _normalize_sglang_completion(output)
                for output in raw_outputs[offset : offset + count]
            ])
            offset += count
        logger.info(
            "SGLang completed %s prompts / %s samples in %.2fs",
            len(requests),
            len(flat_prompts),
            time.monotonic() - started,
        )
        return outputs

    def _post_json(self, path: str, payload: dict):
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=7200) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang request failed ({exc.code}): {body}") from exc

    def close(self):
        process = getattr(self, "_process", None)
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


class SGLangSamplingClient:
    """Collect nearby Tinker sample calls into one SGLang continuous batch."""

    def __init__(
        self,
        engine: SGLangEngine,
        batch_wait_ms: int = 25,
        max_batch_prompts: int = 128,
    ):
        self._engine = engine
        self._batch_wait_s = max(batch_wait_ms, 0) / 1000
        self._max_batch_prompts = max(int(max_batch_prompts), 1)
        self._queue: queue.Queue[_Request] = queue.Queue()
        self._worker = threading.Thread(
            target=self._batch_loop,
            name="espl-sglang-batcher",
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
        self._queue.put(_Request(prompt.to_ints(), num_samples, params, result))
        return LocalFuture(result)

    def sample_seeded(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        seed: int,
    ) -> LocalFuture:
        result = Future()
        self._queue.put(
            _Request(
                prompt.to_ints(),
                num_samples,
                sampling_params,
                result,
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
                outputs = self._engine.generate(requests)
                for request, output in zip(requests, outputs):
                    request.result.set_result(_to_tinker_response(output))
            except BaseException as exc:
                for request in requests:
                    if not request.result.done():
                        request.result.set_exception(exc)


def _normalize_sglang_completion(output: dict) -> dict:
    meta = output.get("meta_info", {})
    logprob_entries = meta.get("output_token_logprobs") or []
    token_ids = []
    token_logprobs = []
    for entry in logprob_entries:
        if entry is None or len(entry) < 2:
            continue
        token_logprobs.append(float(entry[0]))
        token_ids.append(int(entry[1]))
    if not token_ids and output.get("output_ids"):
        raise RuntimeError("SGLang returned output token IDs without sampled-token logprobs")
    finish_reason = meta.get("finish_reason", "stop")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type", "stop")
    return {
        "token_ids": token_ids,
        "logprobs": token_logprobs,
        "finish_reason": str(finish_reason or "stop"),
    }


def _to_tinker_response(completions: list[dict]) -> types.SampleResponse:
    sequences = [
        types.SampledSequence(
            tokens=completion["token_ids"],
            logprobs=completion["logprobs"],
            stop_reason=completion["finish_reason"],
        )
        for completion in completions
    ]
    return types.SampleResponse(sequences=sequences)


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
