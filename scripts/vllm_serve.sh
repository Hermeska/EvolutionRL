#!/bin/bash
# Launch the local vLLM sampling server (spl_vllm env) for the E-SPL vLLM backend.
# Used both by the training chain and for standalone GPU testing.
#
# Runtime LoRA updating is enabled so the trainer can hot-load the current-policy
# adapter each step via POST /v1/load_lora_adapter.
#
# Usage (manual test on a GPU node):
#   scripts/vllm_serve.sh            # foreground; Ctrl-C to stop
# Override knobs via env, e.g. VLLM_GPU_UTIL=0.5 VLLM_MAX_LEN=32768.
set -eo pipefail

VLLM_ENV=${VLLM_ENV:-/ibex/user/khangea/conda-environments/spl_vllm}
CONDA_SH=${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.35}     # fraction of the 80GB A100 for vLLM (rest is HF training)
VLLM_MAX_LEN=${VLLM_MAX_LEN:-32768}      # Qwen3-8B native ctx; raise + YaRN for long reflection prompts
LORA_RANK=${LORA_RANK:-32}
HF_HOME=${HF_HOME:-/ibex/user/khangea/espl/hf_cache}
export HF_HOME

# Source conda without tripping set -u (conda init isn't clean).
set +u
# shellcheck disable=SC1091
source "${CONDA_SH}"
conda activate "${VLLM_ENV}"
set -u

# Required so /v1/load_lora_adapter works at runtime.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
# Use the native torch top-k/top-p sampler instead of FlashInfer, which would
# JIT-compile a kernel at startup and needs nvcc (absent on the GPU nodes).
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[vllm] serving ${MODEL_NAME} on ${VLLM_HOST}:${VLLM_PORT} (gpu_util=${VLLM_GPU_UTIL}, max_len=${VLLM_MAX_LEN})"
exec vllm serve "${MODEL_NAME}" \
    --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
    --served-model-name "${MODEL_NAME}" \
    --dtype bfloat16 \
    --enable-lora --max-lora-rank "${LORA_RANK}" --max-loras 2 \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --max-model-len "${VLLM_MAX_LEN}"
