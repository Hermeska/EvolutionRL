#!/usr/bin/env bash
# Run E-SPL Evolution+RL with local LLM backend (no Tinker API needed)
# Model: Qwen/Qwen3-8B — requires ~16 GB VRAM (A100 40GB+)
# First run: downloads model from HuggingFace (~14 GB) and installs dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

# ── Virtual environment setup ─────────────────────────────────────────────────
if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating virtual environment at $VENV ..."
    python3 -m venv "$VENV"
fi

echo "Installing / updating dependencies ..."
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -e "$SCRIPT_DIR[wandb]"
"$VENV/bin/pip" install peft accelerate

# ── Launch training ───────────────────────────────────────────────────────────
cd "$SCRIPT_DIR/tinker_cookbook/recipes"

"$VENV/bin/python" system_prompt_learning_rl.py \
    use_local_backend=True \
    local_model_name=Qwen/Qwen3-8B \
    local_lora_rank=32 \
    local_max_tokens=8192 \
    local_batch_size=10 \
    local_group_size=5 \
    batch_size=10 \
    group_size=5 \
    num_parallel_programs=3 \
    n_epochs=20 \
    dataset_size=100 \
    test_dataset_size=30 \
    dataset_pair=aimo_beyondaime \
    enable_shared_memory=True \
    shared_memory_max_bullets=50 \
    shared_memory_max_new_per_step=3 \
    shared_memory_in_prompt=True \
    rl_loss_fn=ppo \
    experiment_name=local_qwen3_7b \
    "$@"
