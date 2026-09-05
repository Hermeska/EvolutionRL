"""Compile or run E-SPL on Aviflow GPUs from a public GitHub commit."""

import argparse
import importlib
import os
import re
from typing import Any, Dict

import aviflow
from aviflow import HTML, Metrics, Model, Output, pipeline, remote
from kfp.dsl import Dataset

# Aviflow 0.2.x defaults to an insecure PyPI fallback. Use the official
# CUDA 12.8 wheel index so the runtime stays compatible with CUDA 12.x workers.
_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

DEFAULT_BASE_COMPONENT = "pytorch-dl/pytorch-train-dl:0.2.4"
BASE_COMPONENT = os.environ.get(
    "ESPL_BASE_COMPONENT",
    DEFAULT_BASE_COMPONENT,
)
SOURCE_REPOSITORY = "https://github.com/Hermeska/EvolutionRL"
SOURCE_SHA = "1694862d56630a704f5e69f8222de0514ebb1c4e"

COMMON_RUNTIME_PACKAGES = [
    "urllib3>=1.26.4,<2",
    "accelerate==1.14.0",
    "tinker==0.6.3",
    (
        "https://files.pythonhosted.org/packages/3c/eb/"
        "77789ad6f1807328a61c205881580546af597f60334f1f96fd4f3bb6e929/"
        "chz-0.4.0-py3-none-any.whl"
        "#sha256=5db5ffe42f6be38f1c37e1b18f0d5559572ee8a8dc941116e67f1bd5396e2a9b"
    ),
    "datasets",
    "rich",
    "math-verify",
    "plotly==6.3.0",
]
VLLM_RUNTIME_PACKAGES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "peft==0.19.1",
    *COMMON_RUNTIME_PACKAGES,
    "transformers==4.57.6",
    "vllm==0.10.2",
]
DFLASH_RUNTIME_PACKAGES = [
    # Native DFlash support starts in vLLM 0.20.1. That release is coupled to
    # Torch 2.11, so it intentionally has a separate environment from the
    # stable vLLM 0.10.2 presets.
    "torch==2.11.0",
    "torchvision==0.26.0",
    "peft==0.19.1",
    *COMMON_RUNTIME_PACKAGES,
    "transformers==4.57.6",
    "vllm==0.20.1",
]
SGLANG_RUNTIME_PACKAGES = [
    # SGLang 0.5.x wheels bundle sgl_kernel variants for Hopper/Blackwell but
    # omit the SM80 common_ops library required by Aviflow's A100 workers.
    # The 0.4.9 stack still ships Ampere kernels and uses Torch 2.7.1/cu128.
    "torch==2.7.1",
    "torchvision==0.22.1",
    # PEFT 0.19 requires torchao >=0.16, while this A100-compatible SGLang
    # release pins torchao 0.9. PEFT 0.18 supports torchao >=0.4.
    "peft==0.18.0",
    *COMMON_RUNTIME_PACKAGES,
    (
        "https://files.pythonhosted.org/packages/f6/9e/"
        "22eb53c9c9ad5983bf1f68b0cebfe173e552fbf204456f27b7e68243e008/"
        "sgl_kernel-0.2.7-cp39-abi3-manylinux2014_x86_64.whl"
        "#sha256=5d5247a4505930290a574d7838fa49dc062fc50a53bb2e65b93744654bcd296f"
    ),
    (
        "https://wheels.vllm.ai/flashinfer-python/"
        "flashinfer_python-0.2.9rc2%2Bcu128-cp39-abi3-manylinux1_x86_64.whl"
    ),
    # Pin a Python 3.12-compatible wheel. Without the pin, the Aviflow resolver
    # can select the broken uvicorn 0.2.1 sdist, which omits README.md.
    (
        "https://files.pythonhosted.org/packages/61/14/"
        "33a3a1352cfa71812a3a21e8c9bfb83f60b0011f5e36f2b1399d51928209/"
        "uvicorn-0.34.0-py3-none-any.whl"
        "#sha256=023dc038422502fa28a09c7a30bf2b6991512da7dcdb8fd35fe57cfc154126f4"
    ),
    "cuda-python>=12.8,<13",
    "transformers==4.54.0",
    "sglang[srt]==0.4.9.post6",
]

DEFAULT_PARAMS: Dict[str, Any] = {
    "source_sha": SOURCE_SHA,
    "run_name": "qwen3-4b-vllm-evolution-3ep-8k",
    "model_name": "Qwen/Qwen3-4B",
    "dataset_pair": "aimo_beyondaime",
    "domain": "math",
    "dataset_size": 90,
    "dataset_seed": 42,
    "test_dataset_size": 20,
    "n_epochs": 3,
    "batch_size": 5,
    "group_size": 4,
    "num_parallel_programs": 2,
    "max_tokens": 8192,
    "vllm_max_model_len": 16384,
    "gpus": 2,
    "vllm_device": "cuda:1",
    "vllm_gpu_memory_utilization": 0.9,
    "vllm_speculative_model": "",
    "vllm_num_speculative_tokens": 15,
    "vllm_attention_backend": "",
    "renderer_name": "",
    "sampling_backend": "vllm",
    "sampling_temperature": 0.7,
    "top_p": 0.95,
    "eval_sampling_temperature": 0.0,
    "eval_top_p": 1.0,
    "test_group_size": 10,
    "solution_token_budget": 7000,
    "lora_rank": 32,
    "learning_rate": 5e-6,
    "train_mode": "evolution",
    "rl_loss_fn": "cispo",
    "training_microbatch_size": 1,
    "save_every": 1,
    "eval_every": 18,
    "enable_shared_memory": False,
    "crossover_prob": 0.5,
    "checkpoint_root": "",
    "hf_home": "",
}

BASELINE_PARAMS: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "run_name": "qwen3-4b-vllm-baseline-1ep-8k",
    "n_epochs": 1,
    "num_parallel_programs": 1,
    "train_mode": "baseline",
    "crossover_prob": 0.0,
}

SMALL_LONG_CONTEXT_PARAMS: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "run_name": "qwen3-1.7b-vllm-evolution-3ep-16k",
    "model_name": "Qwen/Qwen3-1.7B",
    "max_tokens": 16384,
    "vllm_max_model_len": 24576,
    "solution_token_budget": 14000,
}

SMALL_LONG_BASELINE_PARAMS: Dict[str, Any] = {
    **SMALL_LONG_CONTEXT_PARAMS,
    "run_name": "qwen3-1.7b-vllm-baseline-1ep-16k",
    "n_epochs": 1,
    "num_parallel_programs": 1,
    "train_mode": "baseline",
    "crossover_prob": 0.0,
}

BALANCED_LONG_CONTEXT_PARAMS: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "run_name": "qwen3-4b-vllm-evolution-3ep-12k",
    "max_tokens": 12288,
    "vllm_max_model_len": 20480,
    "solution_token_budget": 8000,
}

SGLANG_BALANCED_LONG_CONTEXT_PARAMS: Dict[str, Any] = {
    **BALANCED_LONG_CONTEXT_PARAMS,
    "run_name": "qwen3-4b-sglang-evolution-3ep-12k",
    "sampling_backend": "sglang",
}

FORMULA_EVOLUTION_PARAMS: Dict[str, Any] = {
    **DEFAULT_PARAMS,
    "run_name": "qwen3-4b-vllm-formula-evolution-3ep-1gpu-eval-t0",
    "dataset_pair": "ace_formula",
    "domain": "finance_formula",
    "dataset_size": 500,
    "test_dataset_size": 200,
    "n_epochs": 3,
    "batch_size": 10,
    "group_size": 4,
    "num_parallel_programs": 2,
    "max_tokens": 4096,
    "vllm_max_model_len": 8192,
    "gpus": 1,
    "vllm_device": "cuda:0",
    # Evolution-only still creates the compatibility training model on cuda:0.
    # Leave enough headroom for that copy plus CUDA/PyTorch allocations.
    "vllm_gpu_memory_utilization": 0.72,
    "solution_token_budget": 1024,
    "sampling_temperature": 0.7,
    "top_p": 0.95,
    "eval_sampling_temperature": 0.0,
    "eval_top_p": 1.0,
    "test_group_size": 1,
    "train_mode": "evolution",
    "eval_every": 50,
    "save_every": 10,
    "crossover_prob": 0.5,
}

FORMULA_BASELINE_PARAMS: Dict[str, Any] = {
    **FORMULA_EVOLUTION_PARAMS,
    "run_name": "qwen3-4b-vllm-formula-baseline",
    # The baseline is evaluated at step 0 before this single dummy batch. No
    # model weights or principles are updated in baseline mode.
    "dataset_size": 1,
    "n_epochs": 1,
    "batch_size": 1,
    "group_size": 1,
    "num_parallel_programs": 1,
    "sampling_temperature": 0.0,
    "top_p": 1.0,
    "train_mode": "baseline",
    "eval_every": 1,
    "save_every": 1,
    "crossover_prob": 0.0,
}

FORMULA_VLLM_STRESS_PARAMS: Dict[str, Any] = {
    **FORMULA_EVOLUTION_PARAMS,
    "run_name": "qwen3-4b-vllm-formula-cardinality-stress-50steps",
    "model_name": "Qwen/Qwen3-4B",
    "dataset_size": 50,
    "test_dataset_size": 5,
    "n_epochs": 1,
    "batch_size": 1,
    "group_size": 4,
    "num_parallel_programs": 2,
    "max_tokens": 512,
    "solution_token_budget": 256,
    "vllm_max_model_len": 4096,
    "eval_every": 100,
    "save_every": 10,
}

FORMULA_DFLASH_PARAMS: Dict[str, Any] = {
    **FORMULA_EVOLUTION_PARAMS,
    "run_name": "qwen3-4b-dflash-formula-evolution-3ep-1gpu",
    "vllm_speculative_model": "z-lab/Qwen3-4B-DFlash-b16",
    "vllm_num_speculative_tokens": 15,
    # DFlash uses non-causal attention in its drafter. FLASH_ATTN is the
    # supported backend on A100; FA3 is Hopper-only.
    "vllm_attention_backend": "FLASH_ATTN",
    # The published Qwen3-4B drafter was trained for thinking-disabled prompts.
    "renderer_name": "qwen3_disable_thinking",
    # Reserve memory for the additional 0.5B-parameter draft model.
    "vllm_gpu_memory_utilization": 0.64,
}

PRESETS = {
    "evolution": DEFAULT_PARAMS,
    "baseline": BASELINE_PARAMS,
    "small-long": SMALL_LONG_CONTEXT_PARAMS,
    "small-long-baseline": SMALL_LONG_BASELINE_PARAMS,
    "balanced-long": BALANCED_LONG_CONTEXT_PARAMS,
    "balanced-long-sglang": SGLANG_BALANCED_LONG_CONTEXT_PARAMS,
    "formula-evolution": FORMULA_EVOLUTION_PARAMS,
    "formula-baseline": FORMULA_BASELINE_PARAMS,
    "formula-vllm-stress": FORMULA_VLLM_STRESS_PARAMS,
    "formula-evolution-dflash": FORMULA_DFLASH_PARAMS,
}


def sanitize_pip_sources(pipeline_path: str) -> None:
    """Fix pip hosts and promote built-in Dataset outputs to Aviflow reports."""
    with open(pipeline_path, encoding="utf-8") as pipeline_file:
        pipeline_yaml = pipeline_file.read()

    replacements = {
        "--trusted-host https://download.pytorch.org/whl/cu128": (
            "--trusted-host download.pytorch.org"
        ),
        "--trusted-host https://pypi.org/simple": "--trusted-host pypi.org",
        "--trusted-host http://pypi.org/simple": "--trusted-host pypi.org",
    }
    for invalid_value, valid_value in replacements.items():
        pipeline_yaml = pipeline_yaml.replace(invalid_value, valid_value)

    report_schemas = {
        "convergence_chart": "system.Chart",
        "efficiency_chart": "system.Chart",
        "evolution_history": "system.Table",
        "performance_chart": "system.Chart",
        "rollout_samples": "system.Table",
        "step_metrics": "system.Table",
    }
    for artifact_name, schema_title in report_schemas.items():
        pattern = (
            rf"({artifact_name}:\n"
            rf"\s+artifactType:\n"
            rf"\s+schemaTitle:) system\.Dataset"
        )
        pipeline_yaml, replacement_count = re.subn(
            pattern,
            rf"\1 {schema_title}",
            pipeline_yaml,
        )
        if replacement_count != 1:
            raise RuntimeError(
                f"Expected one {artifact_name} schema in compiled YAML, "
                f"found {replacement_count}"
            )

    with open(pipeline_path, "w", encoding="utf-8") as pipeline_file:
        pipeline_file.write(pipeline_yaml)


def make_pipeline(
    run_name: str = DEFAULT_PARAMS["run_name"],
    sampling_backend: str = DEFAULT_PARAMS["sampling_backend"],
    requested_gpus: int = DEFAULT_PARAMS["gpus"],
    use_dflash: bool = False,
):
    """Resolve the GPU base image and construct the lightweight component."""

    if use_dflash:
        if sampling_backend != "vllm":
            raise ValueError("DFlash is only supported by the vLLM backend")
        runtime_packages = DFLASH_RUNTIME_PACKAGES
    elif sampling_backend == "vllm":
        runtime_packages = VLLM_RUNTIME_PACKAGES
    elif sampling_backend == "sglang":
        runtime_packages = SGLANG_RUNTIME_PACKAGES
    else:
        raise ValueError(f"Unsupported pipeline sampling backend: {sampling_backend}")

    @remote(
        runtime_env={
            "base_component": BASE_COMPONENT,
            "pip": runtime_packages,
        },
        cpus=8,
        memory_mb=32768,
        gpus=requested_gpus,
        enable_caching=False,
    )
    def train_espl(
        params: Dict[str, Any],
        trained_model: Output[Model],
        metrics: Output[Metrics],
        performance_chart: Output[Dataset],
        convergence_chart: Output[Dataset],
        efficiency_chart: Output[Dataset],
        step_metrics: Output[Dataset],
        rollout_samples: Output[Dataset],
        evolution_history: Output[Dataset],
        experiment_report: Output[HTML],
    ) -> int:
        import ctypes.util
        import json
        import os
        import pathlib
        import shutil
        import subprocess
        import sys
        import tarfile
        import textwrap
        import urllib.request
        import zipfile

        import torch

        cfg = dict(params)
        if str(cfg.get("sampling_backend", "")) == "sglang":
            if ctypes.util.find_library("numa") is None:
                apt_get = shutil.which("apt-get")
                if apt_get is None:
                    raise RuntimeError(
                        "SGLang requires libnuma.so.1, but apt-get is unavailable"
                    )
                print("Installing system dependency libnuma1 for SGLang ...")
                subprocess.run([apt_get, "update"], check=True)
                subprocess.run(
                    [apt_get, "install", "-y", "--no-install-recommends", "libnuma1"],
                    check=True,
                )
        print(
            "PyTorch runtime:",
            f"torch={torch.__version__}",
            f"cuda_build={torch.version.cuda}",
            f"cuda_available={torch.cuda.is_available()}",
        )
        if not torch.cuda.is_available():
            raise RuntimeError("Aviflow task started without a visible CUDA GPU")

        source_sha = str(cfg["source_sha"]).strip()
        if len(source_sha) != 40 or any(char not in "0123456789abcdef" for char in source_sha):
            raise ValueError("source_sha must be a full lowercase 40-character Git SHA")

        source_dir = pathlib.Path("/tmp/evolutionrl-source")
        archive_path = pathlib.Path("/tmp/evolutionrl-source.zip")
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True)
        archive_url = (
            "https://github.com/Hermeska/EvolutionRL/archive/"
            f"{source_sha}.zip"
        )
        urllib.request.urlretrieve(archive_url, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(source_dir)

        recipes = list(
            source_dir.glob(
                "*/tinker_cookbook/recipes/system_prompt_learning_rl.py"
            )
        )
        if len(recipes) != 1:
            raise RuntimeError(
                f"Expected one E-SPL recipe in {archive_url}, found {len(recipes)}"
            )
        recipe = recipes[0]
        project_root = recipe.parents[2]

        # Source archives do not contain .git, and minimal GPU images may not
        # contain the git executable either. Make code-state logging treat both
        # cases as "not in a Git repository" instead of aborting training.
        code_state_path = project_root / "tinker_cookbook" / "utils" / "code_state.py"
        code_state_source = code_state_path.read_text(encoding="utf-8")
        old_handler = "except subprocess.CalledProcessError:"
        new_handler = "except (subprocess.CalledProcessError, FileNotFoundError):"
        if old_handler in code_state_source:
            code_state_path.write_text(
                code_state_source.replace(old_handler, new_handler, 1),
                encoding="utf-8",
            )

        # FlashInfer's cu128 AOT wheel requires GLIBC_2.32, but the Aviflow
        # base component uses an older glibc. Triton attention plus PyTorch
        # sampling avoid loading that incompatible shared object and support
        # Qwen3 on A100/SM80.
        sglang_sampling_path = (
            project_root / "tinker_cookbook" / "local_backend" / "sglang_sampling.py"
        )
        if str(cfg.get("sampling_backend", "")) == "sglang":
            sglang_sampling_source = sglang_sampling_path.read_text(encoding="utf-8")
            attention_marker = '            "--attention-backend",\n'
            if attention_marker not in sglang_sampling_source:
                dtype_marker = '            "--dtype",\n            dtype,\n'
                if dtype_marker not in sglang_sampling_source:
                    raise RuntimeError(
                        "Could not configure SGLang Triton attention backend"
                    )
                sglang_sampling_path.write_text(
                    sglang_sampling_source.replace(
                        dtype_marker,
                        dtype_marker
                        + '            "--attention-backend",\n'
                        + '            "triton",\n'
                        + '            "--sampling-backend",\n'
                        + '            "pytorch",\n',
                        1,
                    ),
                    encoding="utf-8",
                )

        # The Qwen tokenizer uses EOS as padding. Transformers cannot infer a
        # reliable mask in that case, so pass the all-ones mask explicitly for
        # each unpadded sampling prompt.
        local_client_path = project_root / "tinker_cookbook" / "local_backend" / "client.py"
        local_client_source = local_client_path.read_text(encoding="utf-8")
        generate_input = "                    input_ids=input_ids,\n"
        generate_input_with_mask = (
            generate_input
            + "                    attention_mask=torch.ones_like(input_ids),\n"
        )
        if (
            generate_input in local_client_source
            and generate_input_with_mask not in local_client_source
        ):
            local_client_path.write_text(
                local_client_source.replace(
                    generate_input,
                    generate_input_with_mask,
                    1,
                ),
                encoding="utf-8",
            )

        run_name = str(cfg["run_name"])
        persistent_root = str(cfg.get("checkpoint_root", "")).strip()
        run_root = (
            pathlib.Path(persistent_root)
            if persistent_root
            else pathlib.Path("/tmp") / "espl" / run_name
        )
        checkpoint_dir = run_root / "checkpoints"
        log_dir = run_root / "logs"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if str(cfg.get("hf_home", "")).strip():
            env["HF_HOME"] = str(cfg["hf_home"]).strip()

        model_name = str(cfg["model_name"])
        selected_sampling_backend = str(cfg.get("sampling_backend", "vllm"))
        command = [
            sys.executable,
            "-u",
            str(recipe),
            "use_local_backend=True",
            f"model_name={model_name}",
            f"local_model_name={model_name}",
            "local_num_gpus=1",
            "local_dtype=bfloat16",
            "local_attention_backend=sdpa",
            f"local_sampling_backend={selected_sampling_backend}",
            f"local_training_microbatch_size={cfg['training_microbatch_size']}",
            f"local_lora_rank={cfg['lora_rank']}",
            f"local_max_tokens={cfg['max_tokens']}",
            f"solution_token_budget={cfg['solution_token_budget']}",
            f"learning_rate={cfg['learning_rate']}",
            f"batch_size={cfg['batch_size']}",
            f"group_size={cfg['group_size']}",
            f"num_parallel_programs={cfg['num_parallel_programs']}",
            f"n_epochs={cfg['n_epochs']}",
            f"dataset_size={cfg['dataset_size']}",
            f"dataset_seed={cfg['dataset_seed']}",
            f"test_dataset_size={cfg['test_dataset_size']}",
            f"dataset_pair={cfg['dataset_pair']}",
            f"domain={cfg['domain']}",
            f"enable_shared_memory={cfg['enable_shared_memory']}",
            f"train_mode={cfg['train_mode']}",
            f"rl_loss_fn={cfg['rl_loss_fn']}",
            f"crossover_prob={cfg['crossover_prob']}",
            "resume_strategy=last",
            f"save_every={cfg['save_every']}",
            f"eval_every={cfg['eval_every']}",
            f"sampling_temperature={cfg['sampling_temperature']}",
            f"top_p={cfg['top_p']}",
            f"eval_sampling_temperature={cfg['eval_sampling_temperature']}",
            f"eval_top_p={cfg['eval_top_p']}",
            f"test_group_size={cfg['test_group_size']}",
            f"experiment_name={run_name}",
            f"log_path={log_dir}",
            f"local_checkpoint_dir={checkpoint_dir}",
        ]
        renderer_name = str(cfg.get("renderer_name", ""))
        if renderer_name:
            command.append(f"renderer_name={renderer_name}")
        if selected_sampling_backend == "vllm":
            command.extend([
                f"local_vllm_device={cfg['vllm_device']}",
                f"local_vllm_max_model_len={cfg['vllm_max_model_len']}",
                (
                    "local_vllm_gpu_memory_utilization="
                    f"{cfg['vllm_gpu_memory_utilization']}"
                ),
                f"local_vllm_max_lora_rank={cfg['lora_rank']}",
            ])
            speculative_model = str(cfg.get("vllm_speculative_model", ""))
            if speculative_model:
                command.extend([
                    f"local_vllm_speculative_model={speculative_model}",
                    (
                        "local_vllm_num_speculative_tokens="
                        f"{cfg.get('vllm_num_speculative_tokens', 15)}"
                    ),
                    (
                        "local_vllm_attention_backend="
                        f"{cfg.get('vllm_attention_backend', '')}"
                    ),
                ])
        elif selected_sampling_backend == "sglang":
            if "rl" in str(cfg["train_mode"]):
                raise ValueError(
                    "The SGLang preset currently supports baseline/evolution-only "
                    "runs, not dynamic LoRA updates"
                )
            command.extend([
                "local_sglang_device=cuda:1",
                f"local_sglang_max_model_len={cfg['vllm_max_model_len']}",
                "local_sglang_gpu_memory_utilization=0.88",
            ])
        else:
            raise ValueError(
                f"Unsupported sampling backend: {selected_sampling_backend}"
            )
        subprocess.run(command, cwd=run_root, env=env, check=True)

        stats_path = (
            run_root
            / "data"
            / str(cfg["domain"])
            / "train"
            / run_name
            / "stats.json"
        )
        completed_steps = []
        stats = {}
        if stats_path.exists():
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            completed_steps = [
                value
                for _, value in sorted(
                    stats.items(),
                    key=lambda item: int(item[0].removeprefix("step_")),
                )
                if value.get("complete")
            ]

        metrics.log_metric("completed_steps", float(len(completed_steps)))
        full_eval_rewards = []
        if completed_steps:
            rollout = completed_steps[-1].get("rollout", {})
            if "Agg_avg_reward" in rollout:
                metrics.log_metric(
                    "final_reward",
                    float(rollout["Agg_avg_reward"]),
                )
            pass_keys = sorted(
                key for key in rollout if key.startswith("Agg_Pass@")
            )
            if pass_keys:
                metrics.log_metric(
                    "final_pass_at_k",
                    float(rollout[pass_keys[-1]]),
                )

            full_eval_rewards = [
                float(step["test"]["eval_avg_reward"])
                for step in completed_steps
                if step.get("test", {}).get("full_eval")
                and "eval_avg_reward" in step["test"]
            ]
            if full_eval_rewards:
                metrics.log_metric("best_eval_reward", max(full_eval_rewards))
                metrics.log_metric("final_eval_reward", full_eval_rewards[-1])

        metric_rows = []
        metrics_file = log_dir / "metrics.jsonl"
        if metrics_file.exists():
            with metrics_file.open(encoding="utf-8") as metric_stream:
                for global_step, line in enumerate(metric_stream):
                    row = json.loads(line)
                    metric_rows.append({"global_step": global_step, **row})

        performance = None
        convergence = None
        efficiency = None
        success_rates = []
        if metric_rows:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            steps = [row["global_step"] for row in metric_rows]
            success_rates = [
                float(row.get("rollout/success_rate", 0.0)) for row in metric_rows
            ]
            metrics.log_metric("best_success_rate", max(success_rates))
            metrics.log_metric("final_success_rate", success_rates[-1])
            metrics.log_metric(
                "final_success_rate_ma5",
                float(metric_rows[-1].get("convergence/success_rate_ma5", 0.0)),
            )
            metrics.log_metric(
                "final_convergence_slope5",
                float(metric_rows[-1].get("convergence/success_rate_slope5", 0.0)),
            )
            performance = go.Figure()
            for key, label in [
                ("rollout/Agg_avg_reward", "Train reward"),
                ("rollout/Agg_Pass@4", "Train Pass@4"),
            ]:
                performance.add_trace(go.Scatter(
                    x=steps,
                    y=[row.get(key) for row in metric_rows],
                    mode="lines+markers",
                    name=label,
                ))
            full_eval_rows = [
                row for row in metric_rows if row.get("testing/full_eval") is True
            ]
            performance.add_trace(go.Scatter(
                x=[row["global_step"] for row in full_eval_rows],
                y=[row.get("testing/eval_avg_reward") for row in full_eval_rows],
                mode="lines+markers",
                name="Full eval reward",
                line={"width": 4},
            ))
            performance.update_layout(
                title="E-SPL learning curves",
                xaxis_title="Global training step",
                yaxis_title="Reward / pass rate",
                yaxis={"range": [0, 1]},
                hovermode="x unified",
            )
            performance_path = pathlib.Path(performance_chart.path)
            performance_path.parent.mkdir(parents=True, exist_ok=True)
            performance.write_json(performance_path)
            performance_chart.metadata["title"] = "E-SPL learning curves"

            convergence = make_subplots(specs=[[{"secondary_y": True}]])
            convergence.add_trace(go.Scatter(
                x=steps,
                y=[row.get("rollout/success_rate", 0) for row in metric_rows],
                mode="lines+markers",
                name="Success rate",
            ), secondary_y=False)
            convergence.add_trace(go.Scatter(
                x=steps,
                y=[row.get("convergence/success_rate_ma5", 0) for row in metric_rows],
                mode="lines",
                name="Success rate MA(5)",
                line={"width": 4},
            ), secondary_y=False)
            convergence.add_trace(go.Scatter(
                x=steps,
                y=[row.get("convergence/success_rate_slope5", 0) for row in metric_rows],
                mode="lines+markers",
                name="Convergence slope (5 steps)",
            ), secondary_y=True)
            convergence.update_layout(
                title="Evolution convergence and success rate",
                xaxis_title="Global training step",
                hovermode="x unified",
            )
            convergence.update_yaxes(title_text="Success rate", range=[0, 1], secondary_y=False)
            convergence.update_yaxes(title_text="Slope", secondary_y=True)
            convergence_path = pathlib.Path(convergence_chart.path)
            convergence_path.parent.mkdir(parents=True, exist_ok=True)
            convergence.write_json(convergence_path)
            convergence_chart.metadata["title"] = "Evolution convergence and success rate"

            efficiency = make_subplots(specs=[[{"secondary_y": True}]])
            efficiency.add_trace(go.Bar(
                x=steps,
                y=[row.get("time/total") for row in metric_rows],
                name="Step time (s)",
            ), secondary_y=False)
            efficiency.add_trace(go.Scatter(
                x=steps,
                y=[row.get("rl/training_datums", 0) for row in metric_rows],
                mode="lines+markers",
                name="RL datums",
            ), secondary_y=True)
            efficiency.add_trace(go.Scatter(
                x=steps,
                y=[row.get("rollout/truncation_rate", 0) for row in metric_rows],
                mode="lines+markers",
                name="Train truncation rate",
            ), secondary_y=True)
            efficiency.update_layout(
                title="Step time and useful RL signal",
                xaxis_title="Global training step",
                hovermode="x unified",
            )
            efficiency.update_yaxes(title_text="Seconds", secondary_y=False)
            efficiency.update_yaxes(title_text="Training datums", secondary_y=True)
            efficiency_path = pathlib.Path(efficiency_chart.path)
            efficiency_path.parent.mkdir(parents=True, exist_ok=True)
            efficiency.write_json(efficiency_path)
            efficiency_chart.metadata["title"] = "Step time and useful RL signal"

            columns = [
                "step", "epoch", "batch", "train_reward", "train_pass_at_4",
                "eval_reward", "full_eval", "rl_datums", "step_seconds",
                "train_truncation_rate", "eval_truncation_rate",
                "success_rate_ma5", "success_rate_slope5",
            ]
            table_rows = []
            for row, stats_step in zip(metric_rows, completed_steps):
                table_rows.append([
                    row["global_step"],
                    stats_step.get("epoch", -1),
                    stats_step.get("batch", -1),
                    float(row.get("rollout/Agg_avg_reward", 0.0)),
                    float(row.get("rollout/Agg_Pass@4", 0.0)),
                    float(row.get("testing/eval_avg_reward", 0.0)),
                    str(bool(row.get("testing/full_eval", False))),
                    int(row.get("rl/training_datums", 0)),
                    float(row.get("time/total", 0.0)),
                    float(row.get("rollout/truncation_rate", 0.0)),
                    float(row.get("testing/truncation_rate", 0.0)),
                    float(row.get("convergence/success_rate_ma5", 0.0)),
                    float(row.get("convergence/success_rate_slope5", 0.0)),
                ])
            import csv

            table_path = pathlib.Path(step_metrics.path)
            table_path.parent.mkdir(parents=True, exist_ok=True)
            with table_path.open("w", encoding="utf-8", newline="") as table_file:
                writer = csv.writer(table_file, quoting=csv.QUOTE_ALL)
                writer.writerow(columns)
                writer.writerows(table_rows)
            step_metrics.metadata["cols"] = len(columns)
            step_metrics.metadata["rows"] = len(table_rows) + 1

        import csv

        experiment_dir = (
            run_root / "data" / str(cfg["domain"]) / "train" / run_name
        )
        rollout_columns = [
            "step", "program_slot", "program_id", "reward", "problem",
            "query_prompt", "llm_response", "principles", "groundtruth",
        ]
        rollout_rows = []
        for step_dir in sorted(
            experiment_dir.glob("step_*"),
            key=lambda path: int(path.name.removeprefix("step_")),
        ):
            step_number = int(step_dir.name.removeprefix("step_"))
            step_stats = stats.get(f"step_{step_number}", {}) if stats_path.exists() else {}
            program_ids = step_stats.get("sampled_program_ids", [])
            for program_dir in sorted(step_dir.glob("sampled_program_*")):
                program_slot = int(program_dir.name.removeprefix("sampled_program_"))
                principles_path = program_dir / "principles_to_mutate.json"
                principles = (
                    principles_path.read_text(encoding="utf-8")
                    if principles_path.exists() else "{}"
                )
                rollout_path = program_dir / "rollout.jsonl"
                if not rollout_path.exists():
                    continue
                with rollout_path.open(encoding="utf-8") as rollout_stream:
                    for line in rollout_stream:
                        sample = json.loads(line)
                        trajectory = sample.get("trajectories", [{}])[0].get("trajectory", [])
                        query_prompt = next(
                            (message.get("content", "") for message in trajectory if message.get("role") == "user"),
                            "",
                        )
                        response = next(
                            (message.get("content", "") for message in trajectory if message.get("role") == "assistant"),
                            "",
                        )
                        rollout_rows.append([
                            step_number,
                            program_slot,
                            program_ids[program_slot] if program_slot < len(program_ids) else -1,
                            float(sample.get("reward", 0.0)),
                            sample.get("problem", ""),
                            query_prompt,
                            response,
                            principles,
                            str(sample.get("groundtruth", "")),
                        ])

        rollout_path = pathlib.Path(rollout_samples.path)
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        with rollout_path.open("w", encoding="utf-8", newline="") as rollout_file:
            writer = csv.writer(rollout_file, quoting=csv.QUOTE_ALL)
            writer.writerow(rollout_columns)
            writer.writerows(rollout_rows)
        rollout_samples.metadata["cols"] = len(rollout_columns)
        rollout_samples.metadata["rows"] = len(rollout_rows) + 1

        programs_by_id = {}
        for pool_path in sorted(experiment_dir.glob("step_*/evolution_pool.json")):
            pool_state = json.loads(pool_path.read_text(encoding="utf-8"))
            for program in pool_state.get("evolution_pool", []):
                programs_by_id[int(program["program_id"])] = program
        evolution_columns = [
            "program_id", "created_step", "kind", "mutation_parent",
            "crossover_parents", "rating_mu", "rating_sigma", "past_scores",
            "principles", "children",
        ]
        evolution_rows = []
        for program_id, program in sorted(programs_by_id.items()):
            crossover_parents = program.get("parents_list") or []
            mutation_parent = int(program.get("self_modified_from", -1))
            kind = "crossover" if crossover_parents else (
                "mutation" if mutation_parent >= 0 else "root"
            )
            rating = program.get("rating", {})
            evolution_rows.append([
                program_id,
                int(program.get("timestep") if program.get("timestep") is not None else -1),
                kind,
                mutation_parent,
                json.dumps(crossover_parents),
                float(rating.get("mu", 0.0)),
                float(rating.get("sigma", 0.0)),
                json.dumps(program.get("past_score_history", [])),
                json.dumps(program.get("principles", {}), ensure_ascii=False),
                json.dumps(program.get("children_list", [])),
            ])
        evolution_path = pathlib.Path(evolution_history.path)
        evolution_path.parent.mkdir(parents=True, exist_ok=True)
        with evolution_path.open("w", encoding="utf-8", newline="") as evolution_file:
            writer = csv.writer(evolution_file, quoting=csv.QUOTE_ALL)
            writer.writerow(evolution_columns)
            writer.writerows(evolution_rows)
        evolution_history.metadata["cols"] = len(evolution_columns)
        evolution_history.metadata["rows"] = len(evolution_rows) + 1

        # Publish a self-contained report that Aviflow can render directly in
        # the run UI. Plotly JS is embedded once, so the report has no CDN or
        # external-network dependency.
        import html
        import plotly.io as pio

        def metric_card(label: str, value: str, hint: str = "") -> str:
            return (
                '<section class="metric-card">'
                f'<div class="metric-label">{html.escape(label)}</div>'
                f'<div class="metric-value">{html.escape(value)}</div>'
                f'<div class="metric-hint">{html.escape(hint)}</div>'
                '</section>'
            )

        completed_count = len(completed_steps)
        best_eval = max(full_eval_rewards) if full_eval_rewards else 0.0
        final_eval = full_eval_rewards[-1] if full_eval_rewards else 0.0
        best_train = max(success_rates) if success_rates else 0.0
        final_train = success_rates[-1] if success_rates else 0.0
        cards = "".join([
            metric_card("Completed steps", str(completed_count), f"of {cfg['n_epochs']} epochs"),
            metric_card("Best full-eval success", f"{best_eval:.1%}", "fixed evaluation split"),
            metric_card("Final full-eval success", f"{final_eval:.1%}", "latest full checkpoint"),
            metric_card("Best train success", f"{best_train:.1%}", f"final {final_train:.1%}"),
        ])

        chart_blocks = []
        plotly_embedded = False
        for figure, section_title in [
            (performance, "Learning and evaluation"),
            (convergence, "Convergence"),
            (efficiency, "Runtime and truncation"),
        ]:
            if figure is None:
                continue
            chart_html = pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs="inline" if not plotly_embedded else False,
                config={"displaylogo": False, "responsive": True},
                default_width="100%",
                default_height="440px",
            )
            plotly_embedded = True
            chart_blocks.append(
                f'<section class="chart-section"><h2>{html.escape(section_title)}</h2>{chart_html}</section>'
            )

        top_programs = sorted(
            evolution_rows,
            key=lambda row: (float(row[5]), -float(row[6])),
            reverse=True,
        )[:10]
        program_rows = "".join(
            "<tr>"
            f"<td>{int(row[0])}</td>"
            f"<td>{html.escape(str(row[2]))}</td>"
            f"<td>{float(row[5]):.2f}</td>"
            f"<td>{float(row[6]):.2f}</td>"
            f"<td>{html.escape(str(row[7]))}</td>"
            "</tr>"
            for row in top_programs
        )
        if not program_rows:
            program_rows = '<tr><td colspan="5">No evolved programs were produced.</td></tr>'

        report_html = textwrap.dedent(f"""\
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>{html.escape(run_name)} report</title>
              <style>
                :root {{ color-scheme: light; --ink:#172033; --muted:#667085; --line:#e4e7ec; --surface:#fff; --page:#f6f8fb; --accent:#3568e8; }}
                * {{ box-sizing:border-box; }}
                body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.5 Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
                main {{ max-width:1280px; margin:0 auto; padding:28px; }}
                header {{ margin-bottom:22px; }}
                h1 {{ margin:0 0 6px; font-size:28px; font-weight:700; }}
                h2 {{ margin:0 0 12px; font-size:18px; }}
                .subtitle {{ color:var(--muted); }}
                .metrics {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-bottom:18px; }}
                .metric-card,.chart-section,.table-section {{ background:var(--surface); border:1px solid var(--line); border-radius:14px; box-shadow:0 4px 18px rgba(16,24,40,.05); }}
                .metric-card {{ padding:18px; }}
                .metric-label,.metric-hint {{ color:var(--muted); }}
                .metric-value {{ margin:5px 0 2px; color:var(--accent); font-size:28px; font-weight:700; font-variant-numeric:tabular-nums; }}
                .chart-section,.table-section {{ padding:18px; margin-bottom:18px; overflow:hidden; }}
                table {{ width:100%; border-collapse:collapse; }}
                th,td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
                th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
                tr:last-child td {{ border-bottom:0; }}
                .config {{ display:flex; flex-wrap:wrap; gap:8px 16px; margin-top:10px; color:var(--muted); }}
                .config strong {{ color:var(--ink); }}
                @media (max-width:800px) {{ .metrics {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} main {{ padding:18px; }} }}
                @media (max-width:480px) {{ .metrics {{ grid-template-columns:1fr; }} .table-wrap {{ overflow-x:auto; }} }}
              </style>
            </head>
            <body>
            <main>
              <header>
                <h1>{html.escape(run_name)}</h1>
                <div class="subtitle">EvolutionRL experiment report</div>
                <div class="config">
                  <span><strong>Model:</strong> {html.escape(model_name)}</span>
                  <span><strong>Dataset:</strong> {html.escape(str(cfg['dataset_pair']))}</span>
                  <span><strong>Mode:</strong> {html.escape(str(cfg['train_mode']))}</span>
                  <span><strong>Train sampling:</strong> T={float(cfg['sampling_temperature']):g}, top-p={float(cfg['top_p']):g}</span>
                  <span><strong>Eval sampling:</strong> T={float(cfg['eval_sampling_temperature']):g}, top-p={float(cfg['eval_top_p']):g}</span>
                  <span><strong>GPU:</strong> {int(cfg['gpus'])}</span>
                </div>
              </header>
              <div class="metrics">{cards}</div>
              {''.join(chart_blocks) if chart_blocks else '<section class="chart-section"><h2>Learning curves</h2><p>No step metrics were found.</p></section>'}
              <section class="table-section">
                <h2>Top evolved programs</h2>
                <div class="table-wrap"><table>
                  <thead><tr><th>Program</th><th>Kind</th><th>Rating μ</th><th>Rating σ</th><th>Past scores</th></tr></thead>
                  <tbody>{program_rows}</tbody>
                </table></div>
              </section>
            </main>
            </body>
            </html>""")
        report_path = pathlib.Path(experiment_report.path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_html, encoding="utf-8")
        experiment_report.metadata["title"] = "EvolutionRL experiment dashboard"
        experiment_report.metadata["run_name"] = run_name

        artifact_path = pathlib.Path(trained_model.path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists():
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()
        with tarfile.open(artifact_path, "w:gz") as archive:
            archive.add(run_root, arcname=run_name)
        trained_model.framework = "peft"
        trained_model.metadata["base_model"] = model_name
        trained_model.metadata["source_sha"] = source_sha
        trained_model.metadata["lora_rank"] = int(cfg["lora_rank"])

        return len(completed_steps)

    @pipeline(experiment_name="espl-training", run_name=run_name)
    def espl_training_pipeline(params: Dict[str, Any]):
        train_espl(params=params)

    return espl_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compile",
        metavar="PATH",
        help="Compile to YAML instead of starting a remote run.",
    )
    parser.add_argument("--namespace", default="students")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="evolution",
        help="Run the evolution experiment or the frozen Qwen baseline.",
    )
    args = parser.parse_args()

    preset_params = dict(PRESETS[args.preset])
    if (
        preset_params.get("vllm_speculative_model")
        and BASE_COMPONENT == DEFAULT_BASE_COMPONENT
    ):
        raise RuntimeError(
            "The DFlash vLLM wheel requires a newer glibc than "
            f"{DEFAULT_BASE_COMPONENT}. Set ESPL_BASE_COMPONENT to a current "
            "Aviflow CUDA component before compiling or launching this preset."
        )
    aviflow.init(args.namespace)
    training_pipeline = make_pipeline(
        run_name=str(preset_params["run_name"]),
        sampling_backend=str(preset_params["sampling_backend"]),
        requested_gpus=int(preset_params["gpus"]),
        use_dflash=bool(preset_params.get("vllm_speculative_model")),
    )
    parameters = {"params": preset_params}
    if args.compile:
        training_pipeline.compile(path=args.compile, parameters=parameters)
        sanitize_pip_sources(args.compile)
    else:
        training_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
