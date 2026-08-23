# E-SPL training in Aviflow

This pipeline runs EvolutionRL directly from its public GitHub repository.
Publishing a custom Aviflow component, pushing to the component-library Stash
repository, TeamCity and a Jira key are not required for this path.

## How it runs

1. Aviflow resolves the existing GPU base component
   `pytorch-dl/pytorch-train-dl:0.2.4`.
2. The lightweight task installs the Python runtime dependencies.
3. On the GPU worker it downloads the archive for the pinned public commit:
   `Hermeska/EvolutionRL@5ab4c866bc722092d8437df8f1cc468df006a4a4`.
4. It starts Hugging Face + PEFT training on GPU 0 and a dedicated vLLM
   sampling worker on GPU 1.
5. It publishes checkpoints and logs as a `Model` artifact and final values as
   Aviflow metrics.

Tinker Cloud is not used. Qwen3-4B is loaded once by the trainer and once by
vLLM. The trainer saves a fresh PEFT adapter before every rollout step; vLLM
loads that immutable step adapter and batches nearby sampling requests.

`chz` is pinned to its official PyPI wheel because the internal PyPI endpoint
for that package currently returns a malformed gzip response. Only packages
imported by this E-SPL recipe are installed; the broader upstream optional
dependency set is intentionally omitted to keep task startup short.

## Compile

```bash
cd ~/EvolutionRL
source ~/.venvs/evolutionrl-aviflow/bin/activate
avito login
python pipelines/aviflow_espl.py --compile pipelines/espl_train.yaml
```

The compile step needs access to Aviflow Component Registry so it can resolve
the base component image. It does not need access to the custom
`evolutionrl-components` library.

## Start the evolution-only run

```bash
python pipelines/aviflow_espl.py
```

## Start the frozen Qwen baseline

```bash
python pipelines/aviflow_espl.py --preset baseline --namespace students
```

## Start the smaller-model, longer-reasoning experiment

```bash
python pipelines/aviflow_espl.py --preset small-long --namespace students
```

This preset runs evolution-only training for three epochs with Qwen3-1.7B,
16,384 maximum completion tokens, a 14,000-token soft solution budget and a
24,576-token vLLM context window. The dataset, seed, batch/group sizes, LoRA
rank, evaluation schedule and crossover probability match the main evolution
experiment so its results can be compared with Qwen3-4B at 8K.

## Start the frozen Qwen3-1.7B 16K baseline

```bash
python pipelines/aviflow_espl.py --preset small-long-baseline --namespace students
```

This preset performs one frozen-model pass over the same 90/20 task split with
Qwen3-1.7B and a 16,384-token completion limit. It runs no optimizer updates,
mutation or crossover, making it directly comparable with the 1.7B/16K
evolution experiment.

## Start the balanced Qwen3-4B three-epoch experiment

```bash
python pipelines/aviflow_espl.py --preset balanced-long --namespace students
```

This preset keeps Qwen3-4B and runs three evolution epochs. It raises the hard
completion limit to 12,288 tokens while keeping an 8,000-token soft solution
budget and a 20,480-token vLLM context window. This gives truncated answers
room to finish without encouraging every solution to consume a full 16K
completion.

## Start the SGLang Qwen3-4B three-epoch experiment

```bash
python pipelines/aviflow_espl.py --preset balanced-long-sglang --namespace students
```

This preset matches `balanced-long` (Qwen3-4B, three evolution-only epochs,
12,288 hard completion tokens, 8,000-token soft budget, 20,480-token context)
but replaces the dedicated vLLM sampler on GPU 1 with SGLang. It uses SGLang's
continuous batching and prefix/radix cache. Dynamic LoRA updates are intentionally
disabled for this backend; use vLLM for `evolution_rl` runs until SGLang adapter
hot-reloading is implemented.

DFlash2 is not enabled by this preset. It requires a draft checkpoint trained
for the exact target model, and no public DFlash2 checkpoint is currently wired
for `Qwen/Qwen3-4B`. Plain SGLang keeps this comparison valid: the model,
sampling distribution and experiment parameters remain unchanged, so the run
measures the serving backend rather than a second model.

The baseline uses the same Qwen3-4B model, dataset pair, seed, renderer,
sampling temperature, top-p, group size, token limits and vLLM backend. It runs
one pass over all 90 train questions with one unchanged root prompt and performs
full evaluation on the same 20 BeyondAIME questions. It does not run optimizer
updates, mutation or crossover. The zero-initialized LoRA adapter is only the
local backend's transport format and does not change the base-model outputs.

## Compare evolved programs with the root baseline

```bash
python pipelines/aviflow_program_eval.py --namespace students
```

This inference-only run evaluates root and Programs 30, 44, 46 and 47 on the
same first 20 BeyondAIME questions. Each program receives ten samples per
question with identical request-level seeds. Aviflow publishes all 1,000
responses, per-question results, aggregate Pass@K and success rates, bootstrap
confidence intervals against root, and generation-length/truncation charts.
Only one GPU is requested because vLLM loads the unchanged Qwen3-4B model once;
there is no trainer replica, optimizer, LoRA update, mutation or crossover.

Default configuration:

```text
model                 Qwen/Qwen3-4B
GPU                   2 (GPU 0 training, GPU 1 vLLM sampling)
CPU / RAM             8 CPU / 32 GB
dataset               90 train / 20 test (seed 42)
epochs                3
LoRA rank             32
RL loss               CISPO
learning rate         5e-6
train mode             evolution only (no RL optimizer updates)
max completion tokens 8192
soft response budget  7000 (1192-token completion reserve)
training microbatch    1
shared memory          disabled
crossover              probability 0.5
full eval              every 18 steps and at the final step
```

After completion, Aviflow publishes two interactive Plotly charts and a step
metrics table: learning curves, step time versus useful RL datums, and the
underlying per-step values. Scalar summary metrics use full evaluation runs;
quick one-question evaluations no longer overwrite `final_eval_reward`.
The efficiency report also includes train and eval truncation rates, measured
as the fraction of generations stopped by the hard token limit.
Additional Aviflow outputs expose every rollout's query, response, principles,
reward and program id, plus the full mutation/crossover program history. A
dedicated convergence chart shows success rate, its 5-step moving average and
the recent slope.

The source is pinned by the `source_sha` pipeline parameter. Use a full commit
SHA for any later source version; do not replace it with a moving branch name.

## Possible first-run blockers

- GitHub or Hugging Face egress may be restricted from the GPU worker.
- The Hugging Face datasets and Qwen model are downloaded on the first run.
- The selected base component may have a dependency conflict with the pinned
  runtime packages. In that case select another published PyTorch GPU base
  through `ESPL_BASE_COMPONENT`.
- The `students` namespace may preempt opportunistic jobs.

For longer runs, set `checkpoint_root` and `hf_home` to mounted persistent
storage. Without this, restart data and the Hugging Face cache only live on the
worker.
