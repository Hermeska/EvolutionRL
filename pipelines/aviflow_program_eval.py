"""Compile or run paired evaluation of evolved E-SPL programs in Aviflow."""

import argparse
import importlib
import os
import re
from typing import Any, Dict

import aviflow
from aviflow import Metrics, Model, Output, pipeline, remote
from kfp.dsl import Dataset


_aviflow_remote_module = importlib.import_module("aviflow.remote")
_aviflow_remote_module.PYPI_INDEX_URLS = [
    "https://download.pytorch.org/whl/cu128",
    "https://pypi.org/simple",
]

BASE_COMPONENT = os.environ.get(
    "ESPL_BASE_COMPONENT",
    "pytorch-dl/pytorch-train-dl:0.2.4",
)
SOURCE_SHA = "18d57675f1963b4250475396dcb70c916ef66a67"

RUNTIME_PACKAGES = [
    "torch==2.8.0",
    "torchvision==0.23.0",
    "urllib3>=1.26.4,<2",
    "transformers==4.57.6",
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
    "run_name": "qwen3-4b-paired-program-eval-20x10",
    "model_name": "Qwen/Qwen3-4B",
    "dataset": "BeyondAIME",
    "dataset_start": 0,
    "dataset_size": 20,
    "group_size": 10,
    "max_tokens": 8192,
    "solution_token_budget": 7000,
    "max_model_len": 16384,
    "temperature": 0.7,
    "top_p": 0.95,
    "seed": 42,
    "gpu_memory_utilization": 0.9,
    "hf_home": "",
}


def sanitize_pipeline(path: str) -> None:
    with open(path, encoding="utf-8") as stream:
        source = stream.read()
    source = source.replace(
        "--trusted-host https://download.pytorch.org/whl/cu128",
        "--trusted-host download.pytorch.org",
    ).replace(
        "--trusted-host https://pypi.org/simple",
        "--trusted-host pypi.org",
    )
    schemas = {
        "comparison_chart": "system.Chart",
        "efficiency_chart": "system.Chart",
        "summary_table": "system.Table",
        "per_question_table": "system.Table",
        "responses_table": "system.Table",
    }
    for artifact_name, schema_title in schemas.items():
        pattern = (
            rf"({artifact_name}:\n"
            rf"\s+artifactType:\n"
            rf"\s+schemaTitle:) system\.Dataset"
        )
        source, count = re.subn(pattern, rf"\1 {schema_title}", source)
        if count != 1:
            raise RuntimeError(
                f"Expected one {artifact_name} schema, found {count}"
            )
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(source)


def make_pipeline():
    @remote(
        runtime_env={"base_component": BASE_COMPONENT, "pip": RUNTIME_PACKAGES},
        cpus=8,
        memory_mb=32768,
        gpus=1,
        enable_caching=False,
    )
    def evaluate_programs(
        params: Dict[str, Any],
        metrics: Output[Metrics],
        comparison_chart: Output[Dataset],
        efficiency_chart: Output[Dataset],
        summary_table: Output[Dataset],
        per_question_table: Output[Dataset],
        responses_table: Output[Dataset],
        evaluation_bundle: Output[Model],
    ) -> int:
        import csv
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
            raise RuntimeError("Aviflow evaluation started without a CUDA GPU")

        source_sha = str(cfg["source_sha"]).strip()
        if len(source_sha) != 40 or any(
            character not in "0123456789abcdef" for character in source_sha
        ):
            raise ValueError("source_sha must be a full lowercase Git SHA")

        source_dir = pathlib.Path("/tmp/evolutionrl-eval-source")
        archive_path = pathlib.Path("/tmp/evolutionrl-eval-source.zip")
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
            source_dir.glob("*/tinker_cookbook/recipes/evaluate_system_prompts.py")
        )
        program_files = list(
            source_dir.glob("*/pipelines/selected_evolution_programs.json")
        )
        if len(recipes) != 1 or len(program_files) != 1:
            raise RuntimeError(
                f"Expected one evaluator and one program file in {archive_url}"
            )
        recipe = recipes[0]
        programs_path = program_files[0]
        project_root = recipe.parents[2]

        run_root = pathlib.Path("/tmp") / "espl-eval" / str(cfg["run_name"])
        output_dir = run_root / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if str(cfg.get("hf_home", "")).strip():
            env["HF_HOME"] = str(cfg["hf_home"]).strip()

        command = [
            sys.executable,
            "-u",
            str(recipe),
            "--programs",
            str(programs_path),
            "--output-dir",
            str(output_dir),
            "--model",
            str(cfg["model_name"]),
            "--dataset",
            str(cfg["dataset"]),
            "--dataset-start",
            str(int(cfg["dataset_start"])),
            "--dataset-size",
            str(int(cfg["dataset_size"])),
            "--group-size",
            str(int(cfg["group_size"])),
            "--max-tokens",
            str(int(cfg["max_tokens"])),
            "--solution-token-budget",
            str(int(cfg["solution_token_budget"])),
            "--max-model-len",
            str(int(cfg["max_model_len"])),
            "--temperature",
            str(float(cfg["temperature"])),
            "--top-p",
            str(float(cfg["top_p"])),
            "--seed",
            str(int(cfg["seed"])),
            "--gpu-memory-utilization",
            str(float(cfg["gpu_memory_utilization"])),
        ]
        subprocess.run(command, cwd=run_root, env=env, check=True)

        summary_path = output_dir / "summary.csv"
        per_question_path = output_dir / "per_question.csv"
        responses_path = output_dir / "responses.csv"
        with summary_path.open(newline="", encoding="utf-8") as stream:
            summary_rows = list(csv.DictReader(stream))
        if not summary_rows:
            raise RuntimeError("Evaluator produced an empty summary")

        root_row = next(row for row in summary_rows if row["label"] == "root")
        metrics.log_metric("root_success_rate", float(root_row["success_rate"]))
        metrics.log_metric("root_pass_at_4", float(root_row["pass_at_4"]))
        metrics.log_metric("root_pass_at_10", float(root_row["pass_at_10"]))
        metrics.log_metric("root_truncation_rate", float(root_row["truncation_rate"]))
        best_row = max(summary_rows, key=lambda row: float(row["success_rate"]))
        metrics.log_metric("best_program_id", float(best_row["program_id"]))
        metrics.log_metric("best_success_rate", float(best_row["success_rate"]))
        metrics.log_metric("best_delta_vs_root", float(best_row["delta_vs_root"]))
        for row in summary_rows:
            safe_label = row["label"].replace("-", "_")
            metrics.log_metric(
                f"{safe_label}_success_rate", float(row["success_rate"])
            )
            metrics.log_metric(
                f"{safe_label}_delta_vs_root", float(row["delta_vs_root"])
            )

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        labels = [row["label"] for row in summary_rows]
        success = [float(row["success_rate"]) for row in summary_rows]
        deltas = [float(row["delta_vs_root"]) for row in summary_rows]
        ci_low = [float(row["delta_ci_low"]) for row in summary_rows]
        ci_high = [float(row["delta_ci_high"]) for row in summary_rows]
        comparison = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Quality metrics", "Paired delta vs root (95% bootstrap CI)"),
        )
        for key, label in [
            ("success_rate", "Success rate"),
            ("pass_at_4", "Pass@4"),
            ("pass_at_10", "Pass@10"),
        ]:
            comparison.add_trace(
                go.Bar(
                    x=labels,
                    y=[float(row[key]) for row in summary_rows],
                    name=label,
                ),
                row=1,
                col=1,
            )
        comparison.add_trace(
            go.Scatter(
                x=labels,
                y=deltas,
                mode="markers",
                marker={"size": 12},
                name="Delta vs root",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [high - delta for high, delta in zip(ci_high, deltas)],
                    "arrayminus": [delta - low for delta, low in zip(deltas, ci_low)],
                },
            ),
            row=1,
            col=2,
        )
        comparison.add_hline(y=0, line_dash="dash", row=1, col=2)
        comparison.update_yaxes(range=[0, 1], title_text="Rate", row=1, col=1)
        comparison.update_yaxes(title_text="Success-rate delta", row=1, col=2)
        comparison.update_layout(
            title="Paired E-SPL program evaluation",
            barmode="group",
            hovermode="x unified",
        )
        comparison_path = pathlib.Path(comparison_chart.path)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.write_json(comparison_path)
        comparison_chart.metadata["title"] = "Paired program quality comparison"

        efficiency = make_subplots(specs=[[{"secondary_y": True}]])
        efficiency.add_trace(
            go.Bar(
                x=labels,
                y=[float(row["truncation_rate"]) for row in summary_rows],
                name="Truncation rate",
            ),
            secondary_y=False,
        )
        efficiency.add_trace(
            go.Scatter(
                x=labels,
                y=[float(row["mean_output_tokens"]) for row in summary_rows],
                mode="lines+markers",
                name="Mean output tokens",
            ),
            secondary_y=True,
        )
        efficiency.update_yaxes(
            range=[0, 1], title_text="Truncation rate", secondary_y=False
        )
        efficiency.update_yaxes(title_text="Output tokens", secondary_y=True)
        efficiency.update_layout(title="Generation length and truncation")
        efficiency_path = pathlib.Path(efficiency_chart.path)
        efficiency_path.parent.mkdir(parents=True, exist_ok=True)
        efficiency.write_json(efficiency_path)
        efficiency_chart.metadata["title"] = "Generation efficiency comparison"

        def copy_table(source: pathlib.Path, output, rows: int) -> None:
            target = pathlib.Path(output.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            with source.open(newline="", encoding="utf-8") as stream:
                columns = len(next(csv.reader(stream)))
            output.metadata["rows"] = rows + 1
            output.metadata["cols"] = columns

        copy_table(summary_path, summary_table, len(summary_rows))
        copy_table(
            per_question_path,
            per_question_table,
            len(summary_rows) * int(cfg["dataset_size"]),
        )
        copy_table(
            responses_path,
            responses_table,
            len(summary_rows) * int(cfg["dataset_size"]) * int(cfg["group_size"]),
        )

        bundle_path = pathlib.Path(evaluation_bundle.path)
        bundle_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "w:gz") as archive:
            archive.add(output_dir, arcname=str(cfg["run_name"]))
        evaluation_bundle.framework = "vllm"
        evaluation_bundle.metadata["base_model"] = str(cfg["model_name"])
        evaluation_bundle.metadata["source_sha"] = source_sha
        evaluation_bundle.metadata["program_count"] = len(summary_rows)
        evaluation_bundle.metadata["response_count"] = (
            len(summary_rows)
            * int(cfg["dataset_size"])
            * int(cfg["group_size"])
        )
        return len(summary_rows)

    @pipeline(
        experiment_name="espl-program-evaluation",
        run_name="qwen3-4b-paired-program-eval-20x10",
    )
    def program_evaluation_pipeline(params: Dict[str, Any]):
        evaluate_programs(params=params)

    return program_evaluation_pipeline


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
    evaluation_pipeline = make_pipeline()
    parameters = {"params": dict(DEFAULT_PARAMS)}
    if args.compile:
        evaluation_pipeline.compile(path=args.compile, parameters=parameters)
        sanitize_pipeline(args.compile)
    else:
        evaluation_pipeline.remote(parameters=parameters)


if __name__ == "__main__":
    main()
