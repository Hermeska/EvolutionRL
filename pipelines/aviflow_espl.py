"""Compile or run E-SPL on Aviflow GPUs from a public GitHub commit."""

import argparse
import importlib
import os
import re
from typing import Any, Dict

import aviflow
from aviflow import Metrics, Model, Output, pipeline, remote
from kfp.dsl import Dataset

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
SOURCE_SHA = "5b98de127889b442d176104921f60a4968a19e89"

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
    "plotly==6.3.0",
]

DEFAULT_PARAMS: Dict[str, Any] = {
    "source_sha": SOURCE_SHA,
    "run_name": "qwen3-4b-vllm-evolution-3ep-8k",
    "model_name": "Qwen/Qwen3-4B",
    "dataset_pair": "aimo_beyondaime",
    "dataset_size": 90,
    "dataset_seed": 42,
    "test_dataset_size": 20,
    "n_epochs": 3,
    "batch_size": 5,
    "group_size": 4,
    "num_parallel_programs": 2,
    "max_tokens": 8192,
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

PRESETS = {
    "evolution": DEFAULT_PARAMS,
    "baseline": BASELINE_PARAMS,
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


def make_pipeline(run_name: str = DEFAULT_PARAMS["run_name"]):
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
        performance_chart: Output[Dataset],
        convergence_chart: Output[Dataset],
        efficiency_chart: Output[Dataset],
        step_metrics: Output[Dataset],
        rollout_samples: Output[Dataset],
        evolution_history: Output[Dataset],
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
            f"enable_shared_memory={cfg['enable_shared_memory']}",
            f"train_mode={cfg['train_mode']}",
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

        experiment_dir = run_root / "data" / "math" / "train" / run_name
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

    aviflow.init(args.namespace)
    preset_params = dict(PRESETS[args.preset])
    training_pipeline = make_pipeline(run_name=str(preset_params["run_name"]))
    parameters = {"params": preset_params}
    if args.compile:
        training_pipeline.compile(path=args.compile, parameters=parameters)
        sanitize_pip_sources(args.compile)
    else:
        training_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
