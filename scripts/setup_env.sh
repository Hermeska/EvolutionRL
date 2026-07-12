#!/usr/bin/env bash
#
# setup_env.sh — one-time creation of the `spl` conda env for E-SPL on iBEX.
#
# Builds a dedicated env from environment.yml, then installs the two deps the
# lockfile omits (peft, accelerate) and overrides the pinned PyPI
# `tinker-cookbook` with this local editable checkout so the E-SPL recipe and
# local_backend resolve to the repo on disk.
#
# Run this ONCE on a login node (it only downloads/installs Python packages;
# it does NOT touch the GPU and does NOT submit any job).
#
# Usage:   scripts/setup_env.sh
# Example: ENV_NAME=spl scripts/setup_env.sh
#
set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-spl}"

command -v conda >/dev/null || { echo "error: conda not on PATH" >&2; exit 1; }

# conda's own init block is not `set -u` clean, so source it under relaxed flags.
CONDA_BASE="$(conda info --base)"
set +u
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
set -u

# A complete env has the lockfile's core deps; a half-built one is worse than none.
env_is_complete() {
    conda run -n "$ENV_NAME" python -c "import tinker, chz, datasets, transformers, torch" 2>/dev/null
}

env_exists() { conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; }

if env_exists && [[ "${REBUILD:-0}" != "1" ]] && env_is_complete; then
    echo "[setup] env '$ENV_NAME' already exists and looks complete — reusing it (REBUILD=1 to force)."
else
    if env_exists; then
        echo "[setup] env '$ENV_NAME' is incomplete or REBUILD=1 — removing and rebuilding ..."
        conda env remove -n "$ENV_NAME" -y
    fi
    echo "[setup] creating env '$ENV_NAME' from environment.yml ..."
    # environment.yml declares `name: spl`; -n keeps us explicit/overridable.
    conda env create -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
    env_is_complete || { echo "error: env build did not install the core deps (tinker/chz/datasets). Check the conda output above." >&2; exit 1; }
fi

set +u
conda activate "$ENV_NAME"
set -u

echo "[setup] installing deps omitted by the lockfile (peft, accelerate) ..."
# --no-deps is critical: every runtime dep of peft/accelerate (torch, transformers,
# huggingface_hub, safetensors, numpy, ...) is already pinned by environment.yml.
# Without it, pip upgrades torch/transformers and breaks the lockfile.
pip install --no-deps peft==0.19.1 accelerate==1.14.0

echo "[setup] installing local repo editable (overrides pinned PyPI tinker-cookbook) ..."
pip install -e "$PROJECT_DIR" --no-deps

echo "[setup] sanity check ..."
python - <<'PY'
import torch, transformers, peft, tinker
import tinker_cookbook
from tinker_cookbook.local_backend import LocalServiceClient  # noqa: F401
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("peft        ", peft.__version__)
print("tinker      ", getattr(tinker, "__version__", "?"))
print("cuda avail  ", torch.cuda.is_available())
print("local_backend import OK")
PY

echo "[setup] done. Activate with:  conda activate $ENV_NAME"
