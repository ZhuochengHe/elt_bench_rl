# ELT-Bench RL Environment for Tinker

This repository provides an execution-grounded reinforcement-learning environment for ELT-Bench tasks. An agent inspects a task workspace, configures extraction and loading, builds dbt transformations, uses execution feedback, and submits a pipeline. The environment grades the resulting warehouse state and returns outcome and verified milestone rewards.

The harness uses [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) for model sampling, tool-use rollouts, optimization, and checkpointing. The pinned [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) repository is included as a Git submodule and supplies the benchmark tasks, task-generation code, evaluation metadata, SWE-agent tools, and environment setup scripts.

## Quick start

Clone the repository with its benchmark dependency:

```bash
git clone --recurse-submodules https://github.com/ZhuochengHe/elt_bench_rl.git
cd elt_bench_rl
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Download Snowflake ground truth before grading or training. Files are stored at `runs/benchmark-assets/evaluation/gt` and discovered automatically:

```bash
bash scripts/fetch_ground_truth.sh
```

Follow [Setup](docs/setup.md) for the credential-free local integration flow and the credentialed Snowflake rollout with a Tinker optimization step.

## Documentation

- [Setup and data preparation](docs/setup.md)
- [ELT-Bench upstream repository](repo/README.md)

## Repository layout

```text
eltbench/       Environment, tools, warehouse adapters, rewards, and training CLI
tests/          Unit and integration tests
scripts/        Environment and benchmark-data helper scripts
docker/         Agent execution image
docs/           Setup guide
repo/           Pinned ELT-Bench Git submodule
```

Local credentials, downloaded benchmark data, generated workspaces, logs, and training runs are intentionally excluded from this repository. See `.gitignore` and [Setup](docs/setup.md).

## Attribution

ELT-Bench is maintained by its upstream authors and is included as a pinned submodule under its own license. Review the upstream license and benchmark dataset terms before redistribution. This repository does not relicense upstream code or benchmark data.
