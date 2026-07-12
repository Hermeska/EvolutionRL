#!/bin/bash
#SBATCH --job-name=espl-chain
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1  # override with --gres=gpu:a100:2 for split vLLM/training
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# Self-resubmitting E-SPL (Evolution+RL) training chain, local backend.
# Drives system_prompt_learning_rl.py with the HuggingFace local backend (plain
# `python`, not torchrun). With NUM_GPUS=2 and USE_VLLM=1, vLLM is pinned
# to GPU 0 and HF training/backward is pinned to GPU 1. SLURM log paths are NOT hardcoded
# (#SBATCH can't read env vars); route them per-run via -o/-e at sbatch time.
#
# Each job:
#   1. Exits immediately if the run is already COMPLETE (target reached).
#   2. Resumes from the latest checkpoint (resume_strategy=last, internal) or
#      starts fresh, and trains for whatever fits the 1-hour wall-time budget.
#   3. Submits another copy of itself (afterany) BEFORE training, so the chain
#      continues whether this job finishes cleanly or is killed by wall-time.
#   4. On clean completion (all epochs) writes a COMPLETED marker; the queued
#      successor then starts, sees COMPLETED, and exits — the chain self-ends.
#
# All large artifacts (weights cache, checkpoints, rollouts, W&B) go to
# /ibex/user/$USER — nothing lands in /tmp or $HOME. W&B is OFFLINE by default;
# sync later (optionally to another account) with `wandb sync`.
#
# Usage (start a fresh chain):
#   chmod +x scripts/slurm_train_chain.sh
#   OUT=/ibex/user/khangea/espl/local_qwen3_8b
#   mkdir -p "$OUT/slurm"
#   sbatch -o "$OUT/slurm/%j.out" -e "$OUT/slurm/%j.err" \
#          --export=ALL,EXPERIMENT_NAME=local_qwen3_8b scripts/slurm_train_chain.sh
#   squeue -u $USER | grep espl-chain          # see pending/running jobs
#   scancel --name=espl-chain                  # stop the whole chain
#
# Override any knob at sbatch time via --export, e.g. DATASET_SIZE=50.

set -eo pipefail

# ---------- knobs you might tune ----------
EXPERIMENT_NAME=${EXPERIMENT_NAME:-local_qwen3_8b}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
DATASET_PAIR=${DATASET_PAIR:-aimo_beyondaime}
N_EPOCHS=${N_EPOCHS:-20}
DATASET_SIZE=${DATASET_SIZE:-100}
TEST_DATASET_SIZE=${TEST_DATASET_SIZE:-30}
TEST_GROUP_SIZE=${TEST_GROUP_SIZE:-10}
BATCH_SIZE=${BATCH_SIZE:-10}
GROUP_SIZE=${GROUP_SIZE:-5}
NUM_PARALLEL=${NUM_PARALLEL:-3}
MAX_TOKENS=${MAX_TOKENS:-8192}
LORA_RANK=${LORA_RANK:-32}
RL_LOSS_FN=${RL_LOSS_FN:-ppo}
# save_every=1 -> every completed step is a resume point. Steps are heavy with
# the local sampler, so we must not risk losing an hour of work.
SAVE_EVERY=${SAVE_EVERY:-1}
# Held-out test eval is monitoring only; run it every EVAL_EVERY steps.
EVAL_EVERY=${EVAL_EVERY:-1}
# Max sequences the local backend decodes concurrently per generate() call
# (batches the group_size / test_group_size samples). Cap for KV-cache memory.
LOCAL_SAMPLE_MICROBATCH=${LOCAL_SAMPLE_MICROBATCH:-16}
export LOCAL_SAMPLE_MICROBATCH
# vLLM sampling backend: run a local vLLM server (spl_vllm env) for sampling
# throughput. USE_VLLM=1 to enable. LOCAL_SAMPLE_MICROBATCH is ignored then.
USE_VLLM=${USE_VLLM:-0}
VLLM_ENV=${VLLM_ENV:-/ibex/user/khangea/conda-environments/spl_vllm}
NUM_GPUS=${NUM_GPUS:-1}
JOB_MEM=${JOB_MEM:-96G}
JOB_CPUS=${JOB_CPUS:-12}
# Use a per-job port by default so a stale vLLM server on :8000 cannot pass
# the health check for a new job. Set VLLM_FIXED_PORT=1 to honor VLLM_PORT.
VLLM_FIXED_PORT=${VLLM_FIXED_PORT:-0}
if [ "${VLLM_FIXED_PORT}" != "1" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        VLLM_PORT=$((20000 + SLURM_JOB_ID % 30000))
    else
        VLLM_PORT=${VLLM_PORT:-8000}
    fi
else
    VLLM_PORT=${VLLM_PORT:-8000}
fi
VLLM_GPU_UTIL=${VLLM_GPU_UTIL:-0.90}   # safe for split-GPU mode; lower only when sharing one GPU
VLLM_MAX_LEN=${VLLM_MAX_LEN:-32768}
# Gradient checkpointing on the HF trainer (saves activation memory at large MAX_TOKENS).
USE_GRAD_CKPT=${USE_GRAD_CKPT:-0}

# shared memory (ACE-style long-term reflection)
ENABLE_SHARED_MEMORY=${ENABLE_SHARED_MEMORY:-True}
SM_MAX_BULLETS=${SM_MAX_BULLETS:-50}
SM_MAX_NEW=${SM_MAX_NEW:-3}
SM_IN_PROMPT=${SM_IN_PROMPT:-True}

PROJECT_DIR=${PROJECT_DIR:-/home/khangea/EvolutionRL}
CONDA_ENV=${CONDA_ENV:-/ibex/user/khangea/conda-environments/spl}
CONDA_SH=${CONDA_SH:-/ibex/user/khangea/miniforge/etc/profile.d/conda.sh}
SCRATCH_BASE=${SCRATCH_BASE:-/ibex/user/khangea/espl}
# Wall-time safety margin: stop training this many seconds before SLURM kills us,
# so the last checkpoint actually lands on disk.
SOFT_TIME_LIMIT_SEC=${SOFT_TIME_LIMIT_SEC:-3300}   # 55 min of a 1h job

# W&B: OFFLINE by default. Plots/metrics stay local; upload later, to any account.
WANDB_MODE=${WANDB_MODE:-offline}

# ---------- derived paths (one SCRATCH_BASE, get them all) ----------
RUN_ROOT=${RUN_ROOT:-${SCRATCH_BASE}/${EXPERIMENT_NAME}}
LOG_PATH=${RUN_ROOT}/tinker_logs        # checkpoints.jsonl + JSON metrics
CKPT_DIR=${RUN_ROOT}/checkpoints        # LoRA + optimizer state dirs
WORK_DIR=${RUN_ROOT}/work               # cwd -> the script's relative data/ tree lands here
DONE_MARKER=${RUN_ROOT}/COMPLETED

# ---------- bookkeeping ----------
mkdir -p "${LOG_PATH}" "${CKPT_DIR}" "${WORK_DIR}" "${RUN_ROOT}/slurm"

echo "=================================================================="
echo "[chain] SLURM_JOB_ID=${SLURM_JOB_ID}   started $(date)"
echo "[chain] host=$(hostname)   run_root=${RUN_ROOT}"
echo "[chain] experiment=${EXPERIMENT_NAME}   dataset_pair=${DATASET_PAIR}   wandb_mode=${WANDB_MODE}"
echo "[chain] num_gpus=${NUM_GPUS}   use_vllm=${USE_VLLM}   vllm_port=${VLLM_PORT}   max_tokens=${MAX_TOKENS}"
echo "[chain] successor resources: gres=gpu:a100:${NUM_GPUS} mem=${JOB_MEM} cpus=${JOB_CPUS}"
echo "=================================================================="

# ---------- early exit if the run is already complete ----------
if [ -f "${DONE_MARKER}" ]; then
    echo "[chain] ${DONE_MARKER} exists — training is done. Not submitting next job. Exiting."
    exit 0
fi

# ---------- caches / scratch redirection (keep $HOME and /tmp clean) ----------
export HF_HOME="${SCRATCH_BASE}/hf_cache"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_BASE}/torch_cache"
export TMPDIR="${SCRATCH_BASE}/tmp"
# Reduce HF-training allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCHINDUCTOR_CACHE_DIR}" "${TMPDIR}"

# ---------- W&B: one run for the whole chain ----------
# The code skips W&B unless WANDB_API_KEY is truthy, but never validates it
# offline — so a placeholder keeps the logger (and its plots) alive locally.
# wandb.init honours these env vars even though the code passes neither id nor resume.
export WANDB_MODE
export WANDB_DIR="${RUN_ROOT}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
if [ "${WANDB_MODE}" = "offline" ]; then
    export WANDB_API_KEY="${WANDB_API_KEY:-offline-placeholder}"
elif [ -z "${WANDB_API_KEY:-}" ]; then
    echo "[chain] ERROR: WANDB_MODE=${WANDB_MODE} but WANDB_API_KEY is not set." >&2
    exit 1
fi

# ---------- activate env ----------
# Source conda init directly. We deliberately skip ~/.bashrc because the system
# rc files and conda's init block are not `set -e` clean and would silently kill
# the script (empty stderr, exit 1).
set +e
if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    ACT_RC=$?
else
    export PATH="${CONDA_ENV}/bin:${PATH}"
    ACT_RC=0
fi
set -e
if [ "${ACT_RC}" -ne 0 ]; then
    echo "[chain] ERROR: conda activate ${CONDA_ENV} failed (rc=${ACT_RC})" >&2
    exit "${ACT_RC}"
fi
echo "[chain] python: $(which python)"
echo "[chain] pytorch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
nvidia-smi -L || true

# ---------- GPU placement ----------
# In split mode, vLLM gets physical GPU 0 and the trainer sees only physical GPU 1
# as cuda:0. This keeps HF backward memory independent from vLLM KV cache.
TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-}
VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-}
if [ "${USE_VLLM}" = "1" ] && [ "${NUM_GPUS}" -ge 2 ]; then
    IFS=',' read -r _VLLM_DEV _TRAIN_DEV _REST <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
    # Recompute every job from this job's Slurm allocation; exported values from
    # a previous chained job may refer to a different node/allocation.
    VLLM_CUDA_VISIBLE_DEVICES=${_VLLM_DEV:-0}
    TRAIN_CUDA_VISIBLE_DEVICES=${_TRAIN_DEV:-1}
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
    VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}
fi
if [ "${USE_VLLM}" = "1" ]; then
    echo "[chain] vLLM CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-<inherit>}"
fi
echo "[chain] train CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES:-<inherit>}"

# ---------- optional vLLM sampling server (own env) ----------
VLLM_PID=""
cleanup_vllm() {
    if [ -n "${VLLM_PID}" ]; then
        echo "[chain] stopping vLLM server (pid ${VLLM_PID})"
        kill "${VLLM_PID}" 2>/dev/null || true
        # Only kill the process launched by this job; do not kill unrelated vLLM servers.
    fi
}
if [ "${USE_VLLM}" = "1" ]; then
    trap cleanup_vllm EXIT
    VLLM_LOG="${RUN_ROOT}/slurm/vllm_${SLURM_JOB_ID}.log"
    echo "[chain] launching vLLM server -> ${VLLM_LOG}"
    CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}" \
        VLLM_ENV="${VLLM_ENV}" CONDA_SH="${CONDA_SH}" MODEL_NAME="${MODEL_NAME}" \
        VLLM_PORT="${VLLM_PORT}" VLLM_GPU_UTIL="${VLLM_GPU_UTIL}" \
        VLLM_MAX_LEN="${VLLM_MAX_LEN}" LORA_RANK="${LORA_RANK}" HF_HOME="${HF_HOME}" \
        bash "${PROJECT_DIR}/scripts/vllm_serve.sh" > "${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    # Wait for /health (server does model load + compile + cudagraph, ~2-4 min).
    echo "[chain] waiting for vLLM /health ..."
    for _ in $(seq 1 240); do   # up to ~20 min
        if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            if kill -0 "${VLLM_PID}" 2>/dev/null; then
                echo "[chain] vLLM server is healthy."
                break
            fi
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[chain] ERROR: vLLM server died during startup. See ${VLLM_LOG}" >&2
            tail -30 "${VLLM_LOG}" >&2 || true
            exit 1
        fi
        sleep 5
    done
    curl -sf "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1 || {
        echo "[chain] ERROR: vLLM server not healthy in time. See ${VLLM_LOG}" >&2
        exit 1
    }
fi

# ---------- record progress before training (crash-loop detection) ----------
CKPT_JSONL="${LOG_PATH}/checkpoints.jsonl"
last_step() {
    # Highest "step" recorded in checkpoints.jsonl, or -1 if none yet.
    python - "${CKPT_JSONL}" <<'PY'
import json, os, sys
p = sys.argv[1]
best = -1
if os.path.exists(p):
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line).get("step")
            except json.JSONDecodeError:
                continue
            if isinstance(s, int) and s > best:
                best = s
print(best)
PY
}
PREV_STEP=$(last_step)
echo "[chain] last checkpoint step before this run: ${PREV_STEP}"

# ---------- submit the NEXT job NOW, with afterany dependency on this one ----------
# Doing this BEFORE training means even if training crashes or hangs, the chain
# continues. Use afterany (not afterok) so wall-time kills don't break it. The
# successor exits at its own early-check once COMPLETED exists, so no scancel.
NEXT_JID=""
if [ -n "${SLURM_JOB_ID}" ]; then
    NEXT_JID=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
        --gres="gpu:a100:${NUM_GPUS}" --mem="${JOB_MEM}" --cpus-per-task="${JOB_CPUS}" \
        --output="${RUN_ROOT}/slurm/%j.out" --error="${RUN_ROOT}/slurm/%j.err" \
        --export=ALL,EXPERIMENT_NAME="${EXPERIMENT_NAME}",MODEL_NAME="${MODEL_NAME}",DATASET_PAIR="${DATASET_PAIR}",N_EPOCHS="${N_EPOCHS}",DATASET_SIZE="${DATASET_SIZE}",TEST_DATASET_SIZE="${TEST_DATASET_SIZE}",TEST_GROUP_SIZE="${TEST_GROUP_SIZE}",BATCH_SIZE="${BATCH_SIZE}",GROUP_SIZE="${GROUP_SIZE}",NUM_PARALLEL="${NUM_PARALLEL}",MAX_TOKENS="${MAX_TOKENS}",LORA_RANK="${LORA_RANK}",RL_LOSS_FN="${RL_LOSS_FN}",SAVE_EVERY="${SAVE_EVERY}",EVAL_EVERY="${EVAL_EVERY}",LOCAL_SAMPLE_MICROBATCH="${LOCAL_SAMPLE_MICROBATCH}",USE_VLLM="${USE_VLLM}",NUM_GPUS="${NUM_GPUS}",JOB_MEM="${JOB_MEM}",JOB_CPUS="${JOB_CPUS}",VLLM_ENV="${VLLM_ENV}",VLLM_FIXED_PORT="${VLLM_FIXED_PORT}",VLLM_PORT="${VLLM_PORT}",VLLM_GPU_UTIL="${VLLM_GPU_UTIL}",VLLM_MAX_LEN="${VLLM_MAX_LEN}",VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES}",TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}",USE_GRAD_CKPT="${USE_GRAD_CKPT}",ENABLE_SHARED_MEMORY="${ENABLE_SHARED_MEMORY}",SM_MAX_BULLETS="${SM_MAX_BULLETS}",SM_MAX_NEW="${SM_MAX_NEW}",SM_IN_PROMPT="${SM_IN_PROMPT}",PROJECT_DIR="${PROJECT_DIR}",CONDA_ENV="${CONDA_ENV}",CONDA_SH="${CONDA_SH}",SCRATCH_BASE="${SCRATCH_BASE}",SOFT_TIME_LIMIT_SEC="${SOFT_TIME_LIMIT_SEC}",WANDB_MODE="${WANDB_MODE}",WANDB_RUN_ID="${WANDB_RUN_ID}",WANDB_RESUME="${WANDB_RESUME}" \
        "${PROJECT_DIR}/scripts/slurm_train_chain.sh")
    echo "[chain] queued NEXT job (dependency=afterany:${SLURM_JOB_ID}): jobid=${NEXT_JID}"
fi

# ---------- run training under a soft time limit ----------
# `timeout` kills the process at SOFT_TIME_LIMIT_SEC so the last checkpoint gets
# written before SLURM's hard kill. Resume is internal (resume_strategy=last).
# The script writes its relative data/ tree under cwd -> run from WORK_DIR.
cd "${WORK_DIR}"
VLLM_ARGS=""
if [ "${USE_VLLM}" = "1" ]; then
    VLLM_ARGS="use_vllm=True vllm_port=${VLLM_PORT}"
fi
if [ "${USE_GRAD_CKPT}" = "1" ]; then
    VLLM_ARGS="${VLLM_ARGS} use_grad_checkpointing=True"
fi
echo "[chain] starting training (soft limit ${SOFT_TIME_LIMIT_SEC}s)..."
set +e
# shellcheck disable=SC2086  # VLLM_ARGS must word-split into separate chz kwargs
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
    timeout --preserve-status --kill-after=60 --signal=SIGTERM "${SOFT_TIME_LIMIT_SEC}" \
    python -u "${PROJECT_DIR}/tinker_cookbook/recipes/system_prompt_learning_rl.py" \
        ${VLLM_ARGS} \
        use_local_backend=True \
        model_name="${MODEL_NAME}" \
        local_model_name="${MODEL_NAME}" \
        local_lora_rank="${LORA_RANK}" \
        local_max_tokens="${MAX_TOKENS}" \
        local_batch_size="${BATCH_SIZE}" \
        local_group_size="${GROUP_SIZE}" \
        batch_size="${BATCH_SIZE}" \
        group_size="${GROUP_SIZE}" \
        num_parallel_programs="${NUM_PARALLEL}" \
        n_epochs="${N_EPOCHS}" \
        dataset_size="${DATASET_SIZE}" \
        test_dataset_size="${TEST_DATASET_SIZE}" \
        test_group_size="${TEST_GROUP_SIZE}" \
        dataset_pair="${DATASET_PAIR}" \
        enable_shared_memory="${ENABLE_SHARED_MEMORY}" \
        shared_memory_max_bullets="${SM_MAX_BULLETS}" \
        shared_memory_max_new_per_step="${SM_MAX_NEW}" \
        shared_memory_in_prompt="${SM_IN_PROMPT}" \
        rl_loss_fn="${RL_LOSS_FN}" \
        resume_strategy=last \
        save_every="${SAVE_EVERY}" \
        eval_every="${EVAL_EVERY}" \
        experiment_name="${EXPERIMENT_NAME}" \
        log_path="${LOG_PATH}" \
        local_checkpoint_dir="${CKPT_DIR}"
RC=$?
set -e
echo "[chain] training exited rc=${RC}   $(date)"

# ---------- on clean completion, mark done so the successor self-terminates ----------
if [ "${RC}" -eq 0 ]; then
    echo "[chain] training completed all epochs — writing ${DONE_MARKER}."
    touch "${DONE_MARKER}"
fi

# ---------- log progress; warn (don't auto-kill) on no progress ----------
NEW_STEP=$(last_step)
if [ "${NEW_STEP}" -le "${PREV_STEP}" ] && [ "${RC}" -ne 0 ]; then
    echo "[chain] WARNING: no new checkpoint this run (was ${PREV_STEP}, still ${NEW_STEP}, rc=${RC})."
    echo "[chain] A crash, or a single step exceeding the ${SOFT_TIME_LIMIT_SEC}s slice."
    echo "[chain] If you see this twice in a row: scancel --name=espl-chain and reduce"
    echo "[chain] config (dataset_size / max_tokens / group_size) so a step fits the hour."
else
    echo "[chain] progress: step ${PREV_STEP} -> ${NEW_STEP}"
fi

if [ -n "${NEXT_JID}" ]; then
    echo "[chain] next job ${NEXT_JID} is queued and will run after this one releases."
fi

echo "=================================================================="
echo "[chain] SLURM_JOB_ID=${SLURM_JOB_ID}   finished $(date)"
echo "=================================================================="
exit "${RC}"
