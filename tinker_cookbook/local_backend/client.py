"""
Local backend replacing the Tinker cloud API.

Uses HuggingFace transformers + PEFT LoRA for local training and inference.
Loads model in float16, trains only LoRA adapters.

Drop-in replacement for tinker.ServiceClient:
    service_client = LocalServiceClient()
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-8B", rank=32)
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-8B")
"""

import logging
import os
import asyncio
import threading
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F

from tinker import types
from tinker.types.tensor_data import TensorData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------

class LocalFuture:
    """Small APIFuture-compatible wrapper around a value or worker future."""

    def __init__(self, value):
        self._value = value

    def result(self):
        if isinstance(self._value, ConcurrentFuture):
            return self._value.result()
        return self._value

    async def result_async(self):
        if isinstance(self._value, ConcurrentFuture):
            return await asyncio.wrap_future(self._value)
        return self._value


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class _SharedLocalModel:
    """
    Holds the (base model + optional LoRA adapters) and an optimizer.
    Shared between LocalSamplingClient and LocalTrainingClient.
    """

    def __init__(
        self,
        model_name: str,
        lora_rank: int = 16,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attention_backend: str = "sdpa",
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        self.model_name = model_name
        self.device = device
        self.lora_rank = lora_rank

        requested_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
        if device.startswith("cuda") and requested_dtype == torch.bfloat16:
            device_index = int(device.split(":", 1)[1]) if ":" in device else 0
            if not torch.cuda.is_bf16_supported(device_index):
                logger.warning("BF16 is not supported on %s; falling back to float16", device)
                requested_dtype = torch.float16
        logger.info(
            "Loading %s in %s on %s (attention=%s) ...",
            model_name,
            requested_dtype,
            device,
            attention_backend,
        )
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=requested_dtype,
            device_map=device,
            attn_implementation=attention_backend,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if lora_rank > 0:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
            )
            self.model = get_peft_model(self.base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = self.base_model

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.pending_optimizer_state = None
        self._pending_loss: Optional[torch.Tensor] = None
        self._grad_accumulation_count = 0
        self._total_weight = 0.0
        self.lock = threading.RLock()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def disable_lora(self):
        if hasattr(self.model, "disable_adapter_layers"):
            self.model.disable_adapter_layers()

    def enable_lora(self):
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()

    def save_lora(self, path: str):
        os.makedirs(path, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)

    def load_lora(self, path: str):
        from peft import PeftModel
        if os.path.exists(path):
            self.model.load_adapter(path, adapter_name="default")


# ---------------------------------------------------------------------------
# Sampling client
# ---------------------------------------------------------------------------

class LocalSamplingClient:
    """
    Drop-in for tinker.SamplingClient.
    Uses a _SharedLocalModel for local inference.
    Set use_lora=False for reference-model sampling (base weights only).
    """

    def __init__(
        self,
        shared_models: list[_SharedLocalModel],
        executor: ThreadPoolExecutor,
        use_lora: bool = True,
    ):
        self._models = shared_models
        self._executor = executor
        self.use_lora = use_lora
        self._next_worker = 0
        self._worker_lock = threading.Lock()

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int = 1,
        sampling_params: Optional[types.SamplingParams] = None,
    ) -> LocalFuture:
        with self._worker_lock:
            worker_idx = self._next_worker
            self._next_worker = (self._next_worker + 1) % len(self._models)
        future = self._executor.submit(
            self._do_sample_with_fallback,
            self._models[worker_idx],
            prompt,
            num_samples,
            sampling_params,
        )
        return LocalFuture(future)

    def _do_sample_with_fallback(self, shared_model, prompt, num_samples, sampling_params):
        try:
            return self._do_sample(shared_model, prompt, num_samples, sampling_params)
        except torch.OutOfMemoryError:
            if not shared_model.device.startswith("cuda") or num_samples <= 1:
                raise
            logger.warning(
                "CUDA OOM on %s with num_samples=%s; retrying as smaller groups",
                shared_model.device,
                num_samples,
            )
            torch.cuda.empty_cache()
            left_size = num_samples // 2
            right_size = num_samples - left_size
            left = self._do_sample_with_fallback(
                shared_model, prompt, left_size, sampling_params
            )
            right = self._do_sample_with_fallback(
                shared_model, prompt, right_size, sampling_params
            )
            return types.SampleResponse(sequences=left.sequences + right.sequences)

    def _do_sample(
        self,
        shared_model: _SharedLocalModel,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: Optional[types.SamplingParams],
    ) -> types.SampleResponse:
        sp = sampling_params or types.SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95, stop=[])
        max_new_tokens = sp.max_tokens
        temperature = sp.temperature
        top_p = sp.top_p
        stop = sp.stop  # list of str or list of int

        input_ids = torch.tensor(prompt.to_ints(), dtype=torch.long).unsqueeze(0).to(shared_model.device)

        # Determine stop token IDs
        stop_token_ids = []
        stop_strings = []
        for s in (stop or []):
            if isinstance(s, int):
                stop_token_ids.append(s)
            elif isinstance(s, str):
                stop_strings.append(s)

        if shared_model.tokenizer.eos_token_id is not None:
            stop_token_ids.append(shared_model.tokenizer.eos_token_id)

        model = shared_model.model
        context_manager = _lora_context(model, self.use_lora)

        num_samples = max(int(num_samples), 1)
        with shared_model.lock, context_manager:
            model.eval()
            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    num_return_sequences=num_samples,
                    num_beams=num_samples if temperature <= 0 and num_samples > 1 else 1,
                    eos_token_id=stop_token_ids if stop_token_ids else None,
                    pad_token_id=shared_model.tokenizer.eos_token_id,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

        # Extract generated tokens (excluding input)
        input_len = input_ids.shape[1]
        sequences = []
        for sample_idx in range(outputs.sequences.shape[0]):
            generated_ids = outputs.sequences[sample_idx, input_len:].tolist()

            # Drop padding after EOS so rollout lengths and logprobs stay aligned.
            if shared_model.tokenizer.eos_token_id in generated_ids:
                eos_idx = generated_ids.index(shared_model.tokenizer.eos_token_id)
                generated_ids = generated_ids[: eos_idx + 1]

            logprobs = _compute_logprobs_from_scores(
                outputs.scores,
                generated_ids,
                sample_idx=sample_idx,
            )

            if stop_strings:
                decoded = shared_model.tokenizer.decode(generated_ids, skip_special_tokens=False)
                for stop_str in stop_strings:
                    idx = decoded.find(stop_str)
                    if idx >= 0:
                        truncated = decoded[:idx]
                        generated_ids = shared_model.tokenizer.encode(truncated, add_special_tokens=False)
                        logprobs = logprobs[:len(generated_ids)]
                        break

            sequences.append(types.SampledSequence(
                tokens=generated_ids,
                logprobs=logprobs,
                stop_reason="stop",
            ))
        return types.SampleResponse(sequences=sequences)


# ---------------------------------------------------------------------------
# Training client
# ---------------------------------------------------------------------------

class LocalTrainingClient:
    """
    Drop-in for tinker.TrainingClient.
    Implements forward_backward + optim_step using PEFT LoRA.
    """

    def __init__(
        self,
        shared_models: list[_SharedLocalModel],
        executor: ThreadPoolExecutor,
        checkpoint_dir: str = "/tmp/espl_checkpoints",
        microbatch_size: int = 1,
    ):
        self._models = shared_models
        self._sm = shared_models[0]  # compatibility alias and checkpoint authority
        self._executor = executor
        self._checkpoint_dir = checkpoint_dir
        self._microbatch_size = max(int(microbatch_size), 1)
        self._grad_scale = 0.0

    # --- Called every step ---

    def save_weights_for_sampler(self, name: str = "") -> LocalFuture:
        # Locally: the sampling client already references the same model.
        # Return a path sentinel that LocalServiceClient will recognise.
        response = types.SaveWeightsForSamplerResponse(path=f"local:{name}")
        return LocalFuture(response)

    def forward_backward(
        self,
        training_datums: list,
        loss_fn: str = "importance_sampling",
        loss_fn_config: Optional[dict] = None,
    ) -> LocalFuture:
        from tinker.types.tensor_data import TensorData
        cfg = loss_fn_config or {}
        total_loss = 0.0
        total_weight = 0.0
        per_datum_outputs = []

        indexed_partitions = [[] for _ in self._models]
        for datum_idx, datum in enumerate(training_datums):
            indexed_partitions[datum_idx % len(self._models)].append((datum_idx, datum))

        futures = []
        for sm, partition in zip(self._models, indexed_partitions):
            futures.append(self._executor.submit(
                self._backward_partition, sm, partition, loss_fn, cfg
            ))

        indexed_results = []
        for future in futures:
            partition_weight, partition_results = future.result()
            total_weight += partition_weight
            indexed_results.extend(partition_results)

        self._synchronize_gradients(total_weight)
        indexed_results.sort(key=lambda item: item[0])
        for _, datum_loss_val, weight in indexed_results:
            total_loss += datum_loss_val * weight
            per_datum_outputs.append({
                "loss": TensorData(data=[datum_loss_val], dtype="float32", shape=[1])
            })

        self._grad_scale = total_weight
        if total_weight > 0:
            for sm in self._models:
                torch.nn.utils.clip_grad_norm_(sm.trainable_parameters(), max_norm=1.0)

        result = types.ForwardBackwardOutput(
            loss_fn_output_type="scalar",
            loss_fn_outputs=per_datum_outputs,
            metrics={"mean_loss": total_loss / max(total_weight, 1e-8)},
        )
        return LocalFuture(result)

    def _backward_partition(self, sm, indexed_datums, loss_fn, cfg):
        with sm.lock:
            microbatch_size = self._microbatch_size
            while True:
                partition_weight = 0.0
                indexed_results = []
                sm.model.train()
                sm.enable_lora()
                sm.model.zero_grad(set_to_none=True)
                try:
                    for start in range(0, len(indexed_datums), microbatch_size):
                        microbatch_items = indexed_datums[start : start + microbatch_size]
                        microbatch = [datum for _, datum in microbatch_items]
                        batch_loss, datum_results = _compute_batch_loss(
                            sm.model, microbatch, loss_fn, cfg, sm.device
                        )
                        if batch_loss is not None:
                            batch_loss.backward()
                        for (datum_idx, _), (datum_loss, weight) in zip(
                            microbatch_items, datum_results
                        ):
                            partition_weight += weight
                            indexed_results.append((datum_idx, datum_loss, weight))
                    return partition_weight, indexed_results
                except torch.OutOfMemoryError:
                    sm.model.zero_grad(set_to_none=True)
                    if not sm.device.startswith("cuda") or microbatch_size <= 1:
                        raise
                    microbatch_size = max(microbatch_size // 2, 1)
                    logger.warning(
                        "CUDA OOM on %s; retrying training with microbatch_size=%s",
                        sm.device,
                        microbatch_size,
                    )
                    torch.cuda.empty_cache()

    def _synchronize_gradients(self, total_weight: float):
        if total_weight <= 0:
            return
        parameter_lists = [list(sm.trainable_parameters()) for sm in self._models]
        for replicas in zip(*parameter_lists):
            available = [p.grad for p in replicas if p.grad is not None]
            if not available:
                continue
            reduced = torch.zeros_like(replicas[0], device=replicas[0].device)
            for grad in available:
                reduced.add_(grad.detach().to(reduced.device))
            reduced.div_(total_weight)
            for parameter in replicas:
                parameter.grad = reduced.to(parameter.device)

    def optim_step(self, adam_params: types.AdamParams) -> LocalFuture:
        for sm in self._models:
            with sm.lock:
                if sm.optimizer is None:
                    sm.optimizer = torch.optim.AdamW(
                        sm.trainable_parameters(),
                        lr=adam_params.learning_rate,
                        betas=(adam_params.beta1, adam_params.beta2),
                        eps=adam_params.eps,
                        weight_decay=0.0,
                        fused=sm.device.startswith("cuda"),
                    )
                    if sm.pending_optimizer_state is not None:
                        sm.optimizer.load_state_dict(sm.pending_optimizer_state)
                        sm.pending_optimizer_state = None
                sm.optimizer.step()
                sm.optimizer.zero_grad(set_to_none=True)
        result = types.OptimStepResponse(metrics={})
        return LocalFuture(result)

    def save_state(self, name: str, path: Optional[str] = None) -> LocalFuture:
        save_path = os.path.join(path or self._checkpoint_dir, f"state_{name}")
        self._do_save(save_path)
        return LocalFuture(SimpleNamespace(path=save_path))

    async def save_state_async(self, name: str) -> LocalFuture:
        save_path = os.path.join(self._checkpoint_dir, f"state_{name}")
        self._do_save(save_path)
        return LocalFuture(SimpleNamespace(path=save_path))

    async def save_weights_for_sampler_async(self, name: str) -> LocalFuture:
        response = types.SaveWeightsForSamplerResponse(path=f"local:{name}")
        return LocalFuture(response)

    def _do_save(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        self._sm.save_lora(save_path)
        if self._sm.optimizer is not None:
            torch.save(self._sm.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
        logger.info(f"Saved checkpoint to {save_path}")

    def load_state(self, state_path: str) -> LocalFuture:
        self._sm.load_lora(state_path)
        opt_path = os.path.join(state_path, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer_state = torch.load(opt_path, map_location="cpu")
            for sm in self._models:
                sm.pending_optimizer_state = optimizer_state
        self._sync_trainable_parameters()
        return LocalFuture(None)

    def _sync_trainable_parameters(self):
        primary = dict(self._sm.model.named_parameters())
        for replica in self._models[1:]:
            replica_params = dict(replica.model.named_parameters())
            for name, source in primary.items():
                if source.requires_grad and name in replica_params:
                    replica_params[name].data.copy_(source.data.to(replica_params[name].device))


# ---------------------------------------------------------------------------
# Service client (factory)
# ---------------------------------------------------------------------------

class LocalServiceClient:
    """
    Drop-in for tinker.ServiceClient.
    Maintains a shared model instance across all clients.

    Usage:
        service = LocalServiceClient()
        training_client = service.create_lora_training_client(
            base_model="Qwen/Qwen3-8B", rank=32
        )
        ref_sampler = service.create_sampling_client(base_model="Qwen/Qwen3-8B")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "/tmp/espl_checkpoints",
        training_microbatch_size: int = 1,
        num_gpus: int = 1,
        dtype: str = "bfloat16",
        attention_backend: str = "sdpa",
    ):
        # base_url is ignored (compatibility shim)
        if device is None:
            available_gpus = torch.cuda.device_count()
            requested_gpus = max(int(num_gpus), 1)
            if available_gpus:
                self.devices = [f"cuda:{i}" for i in range(min(requested_gpus, available_gpus))]
            else:
                self.devices = ["cpu"]
        else:
            self.devices = [d.strip() for d in device.split(",")]
        self.device = self.devices[0]
        self.checkpoint_dir = checkpoint_dir
        self.training_microbatch_size = max(int(training_microbatch_size), 1)
        self.dtype = dtype
        self.attention_backend = attention_backend
        self._shared_models: list[_SharedLocalModel] = []
        self._executor = ThreadPoolExecutor(
            max_workers=len(self.devices), thread_name_prefix="espl-gpu"
        )

        if any(d.startswith("cuda") for d in self.devices):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        logger.info("Local backend devices: %s", self.devices)

    def _get_or_create_models(self, model_name: str, lora_rank: int = 0) -> list[_SharedLocalModel]:
        if not self._shared_models:
            self._shared_models = [
                _SharedLocalModel(
                    model_name,
                    lora_rank=lora_rank,
                    device=device,
                    dtype=self.dtype,
                    attention_backend=self.attention_backend,
                )
                for device in self.devices
            ]
            if lora_rank > 0:
                self._sync_trainable_parameters()
        elif lora_rank > 0 and self._shared_models[0].lora_rank == 0:
            # Model was created without LoRA (e.g. by ref sampler), apply LoRA now
            from peft import LoraConfig, get_peft_model, TaskType
            for sm in self._shared_models:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_rank * 2,
                    target_modules="all-linear",
                    lora_dropout=0.0,
                    bias="none",
                )
                sm.model = get_peft_model(sm.model, lora_config)
                sm.lora_rank = lora_rank
            self._sync_trainable_parameters()
            self._shared_models[0].model.print_trainable_parameters()
            logger.info("Applied LoRA rank=%s to %s replicas", lora_rank, len(self._shared_models))
        return self._shared_models

    def _sync_trainable_parameters(self):
        if len(self._shared_models) < 2:
            return
        primary = dict(self._shared_models[0].model.named_parameters())
        for replica in self._shared_models[1:]:
            replica_params = dict(replica.model.named_parameters())
            for name, source in primary.items():
                if source.requires_grad and name in replica_params:
                    replica_params[name].data.copy_(source.data.to(replica_params[name].device))

    def create_lora_training_client(self, base_model: str, rank: int = 16) -> LocalTrainingClient:
        models = self._get_or_create_models(base_model, lora_rank=rank)
        return LocalTrainingClient(
            models,
            self._executor,
            checkpoint_dir=self.checkpoint_dir,
            microbatch_size=self.training_microbatch_size,
        )

    def create_sampling_client(
        self,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> LocalSamplingClient:
        if not self._shared_models:
            assert base_model is not None, "Must call create_lora_training_client first or pass base_model"
            self._get_or_create_models(base_model, lora_rank=0)

        models = self._shared_models

        # "local:..." paths come from save_weights_for_sampler — use current LoRA weights
        if model_path is not None and model_path.startswith("local:"):
            return LocalSamplingClient(models, self._executor, use_lora=True)

        # base_model reference sampler — use base weights (LoRA disabled)
        if base_model is not None:
            return LocalSamplingClient(models, self._executor, use_lora=False)

        return LocalSamplingClient(models, self._executor, use_lora=True)

    def create_training_client_from_state(self, state_path: str, rank: int = 16) -> LocalTrainingClient:
        assert self._shared_models, "Must call create_sampling_client first to load the model"
        if self._shared_models[0].lora_rank == 0:
            # Apply LoRA before loading saved LoRA weights
            self._get_or_create_models(self._shared_models[0].model_name, lora_rank=rank)
        client = LocalTrainingClient(
            self._shared_models,
            self._executor,
            checkpoint_dir=self.checkpoint_dir,
            microbatch_size=self.training_microbatch_size,
        )
        client.load_state(state_path)
        return client

    def create_rest_client(self):
        raise NotImplementedError("create_rest_client not available in local mode")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _lora_context:
    """Context manager that temporarily enables or disables LoRA adapters."""

    def __init__(self, model, enable: bool):
        self.model = model
        self.enable = enable

    def __enter__(self):
        if self.enable:
            if hasattr(self.model, "enable_adapter_layers"):
                self.model.enable_adapter_layers()
        else:
            if hasattr(self.model, "disable_adapter_layers"):
                self.model.disable_adapter_layers()
        return self

    def __exit__(self, *args):
        # Always re-enable after (training client expects LoRA enabled)
        if hasattr(self.model, "enable_adapter_layers"):
            self.model.enable_adapter_layers()


def _compute_logprobs_from_scores(
    scores: tuple,
    generated_ids: list[int],
    sample_idx: int = 0,
) -> list[float]:
    """
    Compute log P(token | context) for each generated token.
    `scores` is the tuple returned by model.generate(output_scores=True).
    Each element is [batch, vocab_size] unnormalized logits.
    """
    logprobs = []
    for step_idx, (score_t, tok_id) in enumerate(zip(scores, generated_ids)):
        # score_t: [1, vocab_size] raw logits (before temperature/sampling)
        log_p = F.log_softmax(score_t[sample_idx].float(), dim=-1)
        lp = log_p[tok_id].item()
        logprobs.append(lp)
    return logprobs


def _compute_datum_loss(
    model,
    datum: types.Datum,
    loss_fn: str,
    cfg: dict,
    device: str,
) -> tuple[Optional[torch.Tensor], float]:
    """
    Compute the RL loss for a single training datum.
    Returns (loss_tensor, weight) or (None, 0) if datum should be skipped.

    Datum structure:
        model_input   : prompt + completion token IDs  (length N+1)
        target_tokens : shifted target (length N)       ids at each position
        logprobs      : log P under old policy (length N), 0 on prompt positions
        advantages    : advantage values (length N),     0 on prompt positions
    """
    input_ids = torch.tensor(datum.model_input.to_ints(), dtype=torch.long).to(device)
    target_ids = _td_to_tensor(datum.loss_fn_inputs["target_tokens"], dtype=torch.long).to(device)
    old_logprobs = _td_to_tensor(datum.loss_fn_inputs["logprobs"], dtype=torch.float32).to(device)
    advantages = _td_to_tensor(datum.loss_fn_inputs["advantages"], dtype=torch.float32).to(device)

    # Action mask: positions where we have a real advantage signal
    action_mask = (advantages != 0).float()
    weight = action_mask.sum().item()
    if weight == 0:
        return None, 0.0

    # Forward pass
    outputs = model(input_ids=input_ids.unsqueeze(0))
    logits = outputs.logits[0]  # [seq_len, vocab_size]  seq_len == N (shifted by model)

    # Align lengths: logits has seq_len, target_ids should be seq_len
    seq_len = min(logits.shape[0], target_ids.shape[0])
    logits = logits[:seq_len].float()
    target_ids = target_ids[:seq_len]
    old_logprobs = old_logprobs[:seq_len]
    advantages = advantages[:seq_len]
    action_mask = action_mask[:seq_len]

    # New log probabilities under current policy
    new_logprobs = F.log_softmax(logits, dim=-1).gather(1, target_ids.unsqueeze(1)).squeeze(1)

    if loss_fn == "importance_sampling":
        ratio = torch.exp(new_logprobs - old_logprobs)
        loss = -(ratio * advantages * action_mask).sum()

    elif loss_fn == "cispo":
        clip_low = cfg.get("clip_low_threshold", 0.0)
        clip_high = cfg.get("clip_high_threshold", 4.0)
        ratio = torch.exp(new_logprobs - old_logprobs).clamp(clip_low, clip_high)
        loss = -(ratio * advantages * action_mask).sum()

    elif loss_fn == "ppo":
        eps = cfg.get("clip_eps", 0.2)
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped = ratio.clamp(1 - eps, 1 + eps)
        loss = -(torch.min(ratio * advantages, clipped * advantages) * action_mask).sum()

    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}")

    return loss, weight


def _compute_batch_loss(
    model,
    datums: list[types.Datum],
    loss_fn: str,
    cfg: dict,
    device: str,
) -> tuple[Optional[torch.Tensor], list[tuple[float, float]]]:
    """Compute the same token-weighted RL loss for a padded microbatch."""
    if not datums:
        return None, []

    inputs = [torch.tensor(d.model_input.to_ints(), dtype=torch.long) for d in datums]
    targets = [_td_to_tensor(d.loss_fn_inputs["target_tokens"], dtype=torch.long) for d in datums]
    old_lps = [_td_to_tensor(d.loss_fn_inputs["logprobs"]) for d in datums]
    advantages = [_td_to_tensor(d.loss_fn_inputs["advantages"]) for d in datums]
    max_len = max(x.numel() for x in inputs)
    batch_size = len(datums)

    pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(getattr(model, "config", None), "eos_token_id", 0) or 0
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for idx, values in enumerate(inputs):
        length = values.numel()
        input_ids[idx, :length] = values.to(device)
        attention_mask[idx, :length] = 1

    # Keep the full tensor in BF16/FP16; only the active sequence slice is
    # promoted for numerically stable log-softmax.
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    total_objective = None
    results: list[tuple[float, float]] = []
    for idx, (target, old_lp, advantage) in enumerate(zip(targets, old_lps, advantages)):
        seq_len = min(inputs[idx].numel(), target.numel(), logits.shape[1])
        target = target[:seq_len].to(device)
        old_lp = old_lp[:seq_len].to(device)
        advantage = advantage[:seq_len].to(device)
        action_mask = (advantage != 0).float()
        weight = float(action_mask.sum().item())
        if weight == 0:
            results.append((0.0, 0.0))
            continue

        new_lp = F.log_softmax(logits[idx, :seq_len].float(), dim=-1).gather(
            1, target.unsqueeze(1)
        ).squeeze(1)
        ratio = torch.exp(new_lp - old_lp)
        if loss_fn == "importance_sampling":
            token_objective = ratio * advantage
        elif loss_fn == "cispo":
            token_objective = ratio.clamp(
                cfg.get("clip_low_threshold", 0.0), cfg.get("clip_high_threshold", 4.0)
            ) * advantage
        elif loss_fn == "ppo":
            eps = cfg.get("clip_eps", 0.2)
            token_objective = torch.min(
                ratio * advantage,
                ratio.clamp(1 - eps, 1 + eps) * advantage,
            )
        else:
            raise ValueError(f"Unknown loss_fn: {loss_fn}")

        datum_sum = -(token_objective * action_mask).sum()
        total_objective = datum_sum if total_objective is None else total_objective + datum_sum
        results.append((float((datum_sum / weight).detach().item()), weight))

    return total_objective, results


def _td_to_tensor(td: TensorData, dtype=torch.float32) -> torch.Tensor:
    """Convert TensorData to a 1D torch tensor."""
    data = td.data
    return torch.tensor(data, dtype=dtype)
