# E-SPL training in Aviflow

This pipeline runs EvolutionRL directly from its public GitHub repository.
Publishing a custom Aviflow component, pushing to the component-library Stash
repository, TeamCity and a Jira key are not required for this path.

## How it runs

1. Aviflow resolves the existing GPU base component
   `pytorch-dl/pytorch-train-dl:0.2.4`.
2. The lightweight task installs the Python runtime dependencies.
3. On the GPU worker it downloads the archive for the pinned public commit:
   `Hermeska/EvolutionRL@05074825fec8e06577ec63746083de1f2cf89f05`.
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

## Start the smoke run

```bash
python pipelines/aviflow_espl.py
```

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
max completion tokens 4096
training microbatch    1
shared memory          disabled
crossover              disabled
full eval              every 18 steps and at the final step
```

After completion, Aviflow publishes two interactive Plotly charts and a step
metrics table: learning curves, step time versus useful RL datums, and the
underlying per-step values. Scalar summary metrics use full evaluation runs;
quick one-question evaluations no longer overwrite `final_eval_reward`.

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
