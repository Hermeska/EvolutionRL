"""Compile or run E-SPL on one Aviflow GPU from a public GitHub commit."""

import argparse
import importlib
import os
from typing import Any, Dict

import aviflow
from aviflow import Metrics, Model, Output, pipeline, remote

# Aviflow 0.2.x defaults to an insecure PyPI fallback. Use the official
# CUDA 12.8 wheel index so the runtime stays compatible with CUDA 12.x workers.
_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

BASE_COMPONENT = os.environ.get(
    "ESPL_BASE_COMPONENT",
    "pytorch-dl/pytorch-train-dl:0.2.4",
)
SOURCE_REPOSITORY = "https://github.com/Hermeska/EvolutionRL"
SOURCE_SHA = "08be53ed944418b30d5f19eb92af62aa36987725"

RUNTIME_PACKAGES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "urllib3>=1.26.4,<2",
    "transformers==4.57.6",
    "peft==0.19.1",
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
    "vllm==0.10.2",
]

DEFAULT_PARAMS: Dict[str, Any] = {
    "source_sha": SOURCE_SHA,
    "run_name": "qwen3-0.6b-vllm-3ep-8k",
    "model_name": "Qwen/Qwen3-0.6B",
    "dataset_pair": "aimo_beyondaime",
    "dataset_size": 20,
    "test_dataset_size": 10,
    "n_epochs": 3,
    "batch_size": 5,
    "group_size": 4,
    "num_parallel_programs": 3,
    "max_tokens": 8192,
    "lora_rank": 32,
    "learning_rate": 1e-5,
    "rl_loss_fn": "cispo",
    "training_microbatch_size": 1,
    "save_every": 1,
    "eval_every": 4,
    "enable_shared_memory": False,
    "crossover_prob": 0.0,
    "checkpoint_root": "",
    "hf_home": "",
}


def sanitize_pip_sources(pipeline_path: str) -> None:
    """Fix trusted-host values emitted as full URLs by the KFP compiler."""
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

    with open(pipeline_path, "w", encoding="utf-8") as pipeline_file:
        pipeline_file.write(pipeline_yaml)


def make_pipeline():
    """Resolve the GPU base image and construct the lightweight component."""

    @remote(
        runtime_env={
            "base_component": BASE_COMPONENT,
            "pip": RUNTIME_PACKAGES,
        },
        cpus=8,
        memory_mb=32768,
        gpus=2,
        enable_caching=False,
    )
    def train_espl(
        params: Dict[str, Any],
        trained_model: Output[Model],
        metrics: Output[Metrics],
    ) -> int:
        import json
        import os
        import pathlib
        import shutil
        import subprocess
        import sys
        import tarfile
        import urllib.request
        import zipfile

        import torch

        cfg = dict(params)
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
            "local_sampling_backend=vllm",
            "local_vllm_device=cuda:1",
            "local_vllm_max_model_len=16384",
            "local_vllm_gpu_memory_utilization=0.9",
            f"local_vllm_max_lora_rank={cfg['lora_rank']}",
            f"local_training_microbatch_size={cfg['training_microbatch_size']}",
            f"local_lora_rank={cfg['lora_rank']}",
            f"local_max_tokens={cfg['max_tokens']}",
            f"learning_rate={cfg['learning_rate']}",
            f"batch_size={cfg['batch_size']}",
            f"group_size={cfg['group_size']}",
            f"num_parallel_programs={cfg['num_parallel_programs']}",
            f"n_epochs={cfg['n_epochs']}",
            f"dataset_size={cfg['dataset_size']}",
            f"test_dataset_size={cfg['test_dataset_size']}",
            f"dataset_pair={cfg['dataset_pair']}",
            f"enable_shared_memory={cfg['enable_shared_memory']}",
            f"rl_loss_fn={cfg['rl_loss_fn']}",
            f"crossover_prob={cfg['crossover_prob']}",
            "resume_strategy=last",
            f"save_every={cfg['save_every']}",
            f"eval_every={cfg['eval_every']}",
            f"experiment_name={run_name}",
            f"log_path={log_dir}",
            f"local_checkpoint_dir={checkpoint_dir}",
        ]
        subprocess.run(command, cwd=run_root, env=env, check=True)

        stats_path = (
            run_root / "data" / "math" / "train" / run_name / "stats.json"
        )
        completed_steps = []
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

            eval_rewards = [
                float(step["test"]["eval_avg_reward"])
                for step in completed_steps
                if "eval_avg_reward" in step.get("test", {})
            ]
            if eval_rewards:
                metrics.log_metric("best_eval_reward", max(eval_rewards))
                metrics.log_metric("final_eval_reward", eval_rewards[-1])

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

    @pipeline(experiment_name="espl-training", run_name="qwen3-0.6b-vllm-3ep-8k")
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
    args = parser.parse_args()

    aviflow.init(args.namespace)
    training_pipeline = make_pipeline()
    parameters = {"params": dict(DEFAULT_PARAMS)}
    if args.compile:
        training_pipeline.compile(path=args.compile, parameters=parameters)
        sanitize_pip_sources(args.compile)
    else:
        training_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
