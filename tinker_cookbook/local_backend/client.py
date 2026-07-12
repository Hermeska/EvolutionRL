"""
Local backend replacing the Tinker cloud API.

Uses HuggingFace transformers + PEFT LoRA for local training and inference.
Loads model in bfloat16, trains only LoRA adapters.

Drop-in replacement for tinker.ServiceClient:
    service_client = LocalServiceClient()
    training_client = service_client.create_lora_training_client(base_model="Qwen/Qwen3-7B", rank=32)
    sampling_client = service_client.create_sampling_client(base_model="Qwen/Qwen3-7B")
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F

from tinker import types
from tinker.types.tensor_data import TensorData

logger = logging.getLogger(__name__)

# Max sequences decoded concurrently in one generate() call. Batching the
# group_size / test_group_size samples is the main throughput win; cap it so a
# large group can't blow the KV cache. Override via LOCAL_SAMPLE_MICROBATCH.
_SAMPLE_MICROBATCH = int(os.environ.get("LOCAL_SAMPLE_MICROBATCH", "16"))


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------

class LocalFuture:
    """Synchronous stand-in for tinker.APIFuture — result() and result_async() both return immediately."""

    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value

    async def result_async(self):
        return self._value


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class _SharedLocalModel:
    """
    Holds the (base model + optional LoRA adapters) and an optimizer.
    Shared between LocalSamplingClient and LocalTrainingClient.
    """

    def __init__(self, model_name: str, lora_rank: int = 16, device: str = "cuda",
                 grad_checkpointing: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        self.model_name = model_name
        self.device = device
        self.lora_rank = lora_rank

        # bfloat16, not float16: fp16's ~65504 max overflows as the LoRA drifts,
        # giving NaN logits and a CUDA device-side assert in multinomial sampling.
        # bf16 has fp32's range (A100-native) and eliminates that failure mode.
        logger.info(f"Loading model {model_name} in bfloat16 on {device} ...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
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

        # Gradient checkpointing: trades ~20% compute for a large activation-memory
        # saving in the backward pass. Only safe when this model does training only
        # (e.g. the vLLM path, where sampling is served by vLLM) — it disables the
        # KV cache, which the HF-sampling path needs. Opt-in via grad_checkpointing.
        if grad_checkpointing and lora_rank > 0:
            self.base_model.config.use_cache = False
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled on the training model.")

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._pending_loss: Optional[torch.Tensor] = None
        self._grad_accumulation_count = 0
        self._total_weight = 0.0

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

    def __init__(self, shared_model: _SharedLocalModel, use_lora: bool = True):
        self._sm = shared_model
        self.use_lora = use_lora

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int = 1,
        sampling_params: Optional[types.SamplingParams] = None,
    ) -> LocalFuture:
        result = self._do_sample(prompt, sampling_params, num_samples)
        return LocalFuture(result)

    def _do_sample(
        self,
        prompt: types.ModelInput,
        sampling_params: Optional[types.SamplingParams],
        num_samples: int = 1,
    ) -> types.SampleResponse:
        """Draw num_samples completions for a single prompt.

        The samples are decoded in one (micro-)batched generate() call via
        num_return_sequences, rather than one at a time — this is the throughput
        win over the original single-sequence path. Each returned sequence is
        trimmed at its own stop token (finished sequences are padded by generate)
        and carries its own per-token logprobs.
        """
        sp = sampling_params or types.SamplingParams(max_tokens=512, temperature=0.7, top_p=0.95, stop=[])
        max_new_tokens = sp.max_tokens
        temperature = sp.temperature
        top_p = sp.top_p
        stop = sp.stop  # list of str or list of int

        input_ids = torch.tensor(prompt.to_ints(), dtype=torch.long).unsqueeze(0).to(self._sm.device)
        attention_mask = torch.ones_like(input_ids)
        input_len = input_ids.shape[1]

        # Determine stop token IDs / strings
        stop_token_ids: list[int] = []
        stop_strings: list[str] = []
        for s in (stop or []):
            if isinstance(s, int):
                stop_token_ids.append(s)
            elif isinstance(s, str):
                stop_strings.append(s)
        eos_id = self._sm.tokenizer.eos_token_id
        if eos_id is not None:
            stop_token_ids.append(eos_id)
        stop_id_set = set(stop_token_ids)

        do_sample = temperature > 0
        model = self._sm.model
        context_manager = _lora_context(model, self.use_lora)

        sequences: list[types.SampledSequence] = []
        remaining = max(1, num_samples)
        with context_manager:
            with torch.no_grad():
                while remaining > 0:
                    k = min(remaining, _SAMPLE_MICROBATCH)
                    remaining -= k
                    # Greedy decoding can't return multiple distinct sequences;
                    # decode once and replicate below.
                    n_return = k if do_sample else 1
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_return_sequences=n_return,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature if do_sample else 1.0,
                        top_p=top_p,
                        do_sample=do_sample,
                        eos_token_id=stop_token_ids if stop_token_ids else None,
                        pad_token_id=eos_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    # outputs.sequences: [n_return, input_len + gen_len]
                    # outputs.scores: tuple(gen_len) of [n_return, vocab]
                    built = [
                        self._build_sequence(outputs, i, input_len, stop_id_set, stop_strings)
                        for i in range(n_return)
                    ]
                    if not do_sample and k > 1:
                        built = [built[0]] * k
                    sequences.extend(built)

        return types.SampleResponse(sequences=sequences)

    def _build_sequence(self, outputs, seq_idx, input_len, stop_id_set, stop_strings):
        """Extract tokens + per-token logprobs for one sequence of a batched generate."""
        gen_ids = outputs.sequences[seq_idx, input_len:].tolist()

        # Trim at the first stop token: finished sequences in a batch are padded
        # out to the batch max with pad_token_id (== eos), which we must drop.
        cut = len(gen_ids)
        for j, tok in enumerate(gen_ids):
            if tok in stop_id_set:
                cut = j + 1
                break
        gen_ids = gen_ids[:cut]

        logprobs = [
            F.log_softmax(outputs.scores[step][seq_idx].float(), dim=-1)[tok].item()
            for step, tok in enumerate(gen_ids)
        ]

        # Handle string stop sequences (truncate at the first match)
        if stop_strings:
            decoded = self._sm.tokenizer.decode(gen_ids, skip_special_tokens=False)
            for stop_str in stop_strings:
                idx = decoded.find(stop_str)
                if idx >= 0:
                    reencoded = self._sm.tokenizer.encode(decoded[:idx], add_special_tokens=False)
                    gen_ids = reencoded
                    logprobs = logprobs[:len(reencoded)]
                    break

        return types.SampledSequence(tokens=gen_ids, logprobs=logprobs, stop_reason="stop")


# ---------------------------------------------------------------------------
# Training client
# ---------------------------------------------------------------------------

class LocalTrainingClient:
    """
    Drop-in for tinker.TrainingClient.
    Implements forward_backward + optim_step using PEFT LoRA.
    """

    def __init__(self, shared_model: _SharedLocalModel, checkpoint_dir: str = "/tmp/espl_checkpoints"):
        self._sm = shared_model
        self._checkpoint_dir = checkpoint_dir
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

        self._sm.model.train()
        self._sm.enable_lora()

        for datum in training_datums:
            loss, weight = _compute_datum_loss(
                self._sm.model, datum, loss_fn, cfg, self._sm.device
            )
            if loss is None:
                per_datum_outputs.append({"loss": TensorData(data=[0.0], dtype="float32", shape=[1])})
                continue
            (loss * weight).backward()
            datum_loss_val = loss.item()
            total_loss += datum_loss_val * weight
            total_weight += weight
            per_datum_outputs.append({
                "loss": TensorData(data=[datum_loss_val], dtype="float32", shape=[1])
            })

        if total_weight > 0:
            for p in self._sm.trainable_parameters():
                if p.grad is not None:
                    p.grad.data.div_(total_weight)

        self._grad_scale = total_weight
        torch.nn.utils.clip_grad_norm_(self._sm.trainable_parameters(), max_norm=1.0)

        result = types.ForwardBackwardOutput(
            loss_fn_output_type="scalar",
            loss_fn_outputs=per_datum_outputs,
            metrics={"mean_loss": total_loss / max(total_weight, 1e-8)},
        )
        return LocalFuture(result)

    def optim_step(self, adam_params: types.AdamParams) -> LocalFuture:
        if self._sm.optimizer is None:
            self._sm.optimizer = torch.optim.AdamW(
                self._sm.trainable_parameters(),
                lr=adam_params.learning_rate,
                betas=(adam_params.beta1, adam_params.beta2),
                eps=adam_params.eps,
                weight_decay=0.0,
            )
        self._sm.optimizer.step()
        self._sm.optimizer.zero_grad()
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
        if os.path.exists(opt_path) and self._sm.optimizer is not None:
            self._sm.optimizer.load_state_dict(torch.load(opt_path, map_location=self._sm.device))
        return LocalFuture(None)


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
            base_model="Qwen/Qwen3-7B", rank=32
        )
        ref_sampler = service.create_sampling_client(base_model="Qwen/Qwen3-7B")
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "/tmp/espl_checkpoints",
        grad_checkpointing: bool = False,
    ):
        # base_url is ignored (compatibility shim)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.grad_checkpointing = grad_checkpointing
        self._shared_model: Optional[_SharedLocalModel] = None

    def _get_or_create_model(self, model_name: str, lora_rank: int = 0) -> _SharedLocalModel:
        if self._shared_model is None:
            self._shared_model = _SharedLocalModel(
                model_name, lora_rank=lora_rank, device=self.device,
                grad_checkpointing=self.grad_checkpointing,
            )
        elif lora_rank > 0 and self._shared_model.lora_rank == 0:
            # Model was created without LoRA (e.g. by ref sampler), apply LoRA now
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
            )
            self._shared_model.model = get_peft_model(self._shared_model.model, lora_config)
            self._shared_model.lora_rank = lora_rank
            self._shared_model.model.print_trainable_parameters()
            logger.info(f"Applied LoRA rank={lora_rank} to existing model")
        return self._shared_model

    def create_lora_training_client(self, base_model: str, rank: int = 16) -> LocalTrainingClient:
        sm = self._get_or_create_model(base_model, lora_rank=rank)
        return LocalTrainingClient(sm, checkpoint_dir=self.checkpoint_dir)

    def create_sampling_client(
        self,
        base_model: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> LocalSamplingClient:
        if self._shared_model is None:
            assert base_model is not None, "Must call create_lora_training_client first or pass base_model"
            self._get_or_create_model(base_model, lora_rank=0)

        sm = self._shared_model

        # "local:..." paths come from save_weights_for_sampler — use current LoRA weights
        if model_path is not None and model_path.startswith("local:"):
            return LocalSamplingClient(sm, use_lora=True)

        # base_model reference sampler — use base weights (LoRA disabled)
        if base_model is not None:
            return LocalSamplingClient(sm, use_lora=False)

        return LocalSamplingClient(sm, use_lora=True)

    def create_training_client_from_state(self, state_path: str, rank: int = 16) -> LocalTrainingClient:
        assert self._shared_model is not None, "Must call create_sampling_client first to load the model"
        if self._shared_model.lora_rank == 0:
            # Apply LoRA before loading saved LoRA weights
            self._get_or_create_model(self._shared_model.model_name, lora_rank=rank)
        client = LocalTrainingClient(self._shared_model, checkpoint_dir=self.checkpoint_dir)
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


def _td_to_tensor(td: TensorData, dtype=torch.float32) -> torch.Tensor:
    """Convert TensorData to a 1D torch tensor."""
    data = td.data
    return torch.tensor(data, dtype=dtype)
