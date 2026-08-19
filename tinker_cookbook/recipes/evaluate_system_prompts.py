"""Paired evaluation of evolved system-prompt principles on a fixed dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path

import numpy as np
import tinker

from tinker_cookbook import model_info, renderers
from tinker_cookbook.local_backend import LocalServiceClient
from tinker_cookbook.tokenizer_utils import get_tokenizer

from system_prompt_learning_rl import (
    PROBLEM_WITH_PRINCIPLE_TEMPLATE,
    load_data,
    pass_at_k,
    verify_func,
)


logger = logging.getLogger(__name__)


def bootstrap_interval(
    deltas: list[float], seed: int, samples: int = 10_000
) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [deltas[generator.randrange(len(deltas))] for _ in deltas]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return (
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    )


def format_principles(principles: dict[str, str]) -> str:
    return "\n".join(f"[{key}]. {value}" for key, value in principles.items())


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    programs = json.loads(Path(args.programs).read_text(encoding="utf-8"))
    if not programs or programs[0]["label"] != "root":
        raise ValueError("The first selected program must be the root baseline")

    all_test_data = load_data(args.dataset)
    test_data = all_test_data[args.dataset_start : args.dataset_start + args.dataset_size]
    if len(test_data) != args.dataset_size:
        raise ValueError(
            f"Requested {args.dataset_size} questions at offset {args.dataset_start}, "
            f"found {len(test_data)}"
        )

    tokenizer = get_tokenizer(args.model)
    renderer_name = model_info.get_recommended_renderer_name(args.model)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Renderer: %s", renderer_name)

    service_client = LocalServiceClient(
        num_gpus=1,
        dtype="bfloat16",
        sampling_backend="vllm",
        vllm_device="cuda:0",
        vllm_max_model_len=args.max_model_len,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        vllm_max_lora_rank=32,
    )
    sampler = service_client.create_sampling_client(base_model=args.model)
    if not hasattr(sampler, "sample_seeded"):
        raise RuntimeError("The configured sampler does not support paired seeded requests")
    sampling_params = tinker.types.SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=renderer.get_stop_sequences(),
    )

    prepared_jobs = []
    for program in programs:
        principles_text = format_principles(program["principles"]) or "None"
        for question_index, item in enumerate(test_data):
            query_prompt = PROBLEM_WITH_PRINCIPLE_TEMPLATE.format(
                problem=item["problem"],
                solution_token_budget=args.solution_token_budget,
                principles=principles_text,
            )
            model_input = renderer.build_generation_prompt(
                [{"role": "user", "content": query_prompt}]
            )
            prepared_jobs.append(
                {
                    "program": program,
                    "question_index": question_index,
                    "item": item,
                    "query_prompt": query_prompt,
                    "model_input": model_input,
                    "prompt_tokens": len(model_input.to_ints()),
                    "seed": args.seed + question_index * 1009,
                }
            )

    for job in prepared_jobs:
        job["future"] = sampler.sample_seeded(
            prompt=job["model_input"],
            num_samples=args.group_size,
            sampling_params=sampling_params,
            seed=job["seed"],
        )

    response_rows = []
    per_question_rows = []
    question_scores: dict[str, list[float]] = {}
    summary_inputs: dict[str, dict[str, list[float]]] = {}
    for job_number, job in enumerate(prepared_jobs, start=1):
        program = job["program"]
        label = program["label"]
        result = job["future"].result()
        rewards = []
        truncations = []
        output_lengths = []
        for sample_index, sequence in enumerate(result.sequences):
            parsed_message, _ = renderer.parse_response(sequence.tokens)
            response = parsed_message["content"]
            reward = verify_func(response, job["item"]["groundtruth"])
            truncated = str(sequence.stop_reason).lower() in {"length", "max_tokens"}
            rewards.append(reward)
            truncations.append(float(truncated))
            output_lengths.append(len(sequence.tokens))
            response_rows.append(
                {
                    "program_id": program["program_id"],
                    "label": label,
                    "kind": program["kind"],
                    "rating_mu": program["rating_mu"],
                    "question_index": job["question_index"],
                    "sample_index": sample_index,
                    "seed": job["seed"],
                    "reward": reward,
                    "truncated": truncated,
                    "prompt_tokens": job["prompt_tokens"],
                    "output_tokens": len(sequence.tokens),
                    "problem": job["item"]["problem"],
                    "groundtruth": job["item"]["groundtruth"],
                    "query_prompt": job["query_prompt"],
                    "response": response,
                    "principles": json.dumps(program["principles"], ensure_ascii=False),
                }
            )
        correct = int(sum(rewards))
        question_success = float(np.mean(rewards))
        question_scores.setdefault(label, []).append(question_success)
        per_question_rows.append(
            {
                "program_id": program["program_id"],
                "label": label,
                "question_index": job["question_index"],
                "correct": correct,
                "samples": args.group_size,
                "success_rate": question_success,
                "pass_at_4": pass_at_k(args.group_size, correct, min(4, args.group_size)),
                "pass_at_10": pass_at_k(args.group_size, correct, args.group_size),
                "truncation_rate": float(np.mean(truncations)),
                "mean_output_tokens": float(np.mean(output_lengths)),
                "problem": job["item"]["problem"],
            }
        )
        bucket = summary_inputs.setdefault(
            label, {"rewards": [], "truncations": [], "output_tokens": []}
        )
        bucket["rewards"].extend(rewards)
        bucket["truncations"].extend(truncations)
        bucket["output_tokens"].extend(output_lengths)
        logger.info(
            "Completed request %s/%s: %s q%s",
            job_number,
            len(prepared_jobs),
            label,
            job["question_index"],
        )

    root_scores = question_scores["root"]
    summary_rows = []
    for program in programs:
        label = program["label"]
        bucket = summary_inputs[label]
        per_question = [row for row in per_question_rows if row["label"] == label]
        deltas = [score - root for score, root in zip(question_scores[label], root_scores)]
        ci_low, ci_high = bootstrap_interval(
            deltas, seed=args.seed + int(program["program_id"])
        )
        wins = sum(delta > 0 for delta in deltas)
        losses = sum(delta < 0 for delta in deltas)
        summary_rows.append(
            {
                "program_id": program["program_id"],
                "label": label,
                "kind": program["kind"],
                "rating_mu": program["rating_mu"],
                "principle_count": len(program["principles"]),
                "success_rate": float(np.mean(bucket["rewards"])),
                "pass_at_4": float(np.mean([row["pass_at_4"] for row in per_question])),
                "pass_at_10": float(np.mean([row["pass_at_10"] for row in per_question])),
                "truncation_rate": float(np.mean(bucket["truncations"])),
                "mean_output_tokens": float(np.mean(bucket["output_tokens"])),
                "delta_vs_root": float(np.mean(deltas)),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "question_wins": wins,
                "question_ties": len(deltas) - wins - losses,
                "question_losses": losses,
            }
        )

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_dir / "responses.csv", response_rows)
    write_csv(output_dir / "per_question.csv", per_question_rows)
    write_csv(output_dir / "summary.csv", summary_rows)
    (output_dir / "results.json").write_text(
        json.dumps(
            {
                "config": vars(args),
                "programs": programs,
                "summary": summary_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--programs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset", default="BeyondAIME")
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--solution-token-budget", type=int, default=7000)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate(parse_args())
