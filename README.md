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

Complete the Install section in [Setup](docs/setup.md) first. For the Snowflake command, also complete its Airbyte and Snowflake checklist.

## Acceptance workflows

### Credential-free local integration

This runs the dbt-to-warehouse grader path with a local fixture:

```bash
bash scripts/fetch_ground_truth.sh local_postgres
bash scripts/start_local_services.sh
docker build -f docker/Dockerfile.elt-swe -t elt-swe:local .
uv run --active python tests/manual/rollout_local.py all
```

### Credentialed official rollout and Tinker optimization step

After completing the Snowflake checklist in [Setup](docs/setup.md), this command runs the official `address` task against Snowflake with four rollouts and requests one Tinker optimizer step using execution-derived rewards:

```bash
bash scripts/fetch_ground_truth.sh snowflake
bash scripts/fetch_local_assets.sh
bash scripts/start_local_services.sh
bash scripts/seed_resumable.sh address
docker build -f docker/Dockerfile.elt-swe -t elt-swe:local .
docker network connect elt-docker_elt_network airbyte-abctl-control-plane
uv run --active python -m eltbench.train \
  --destination snowflake \
  --task_name address \
  --model_name Qwen/Qwen3-8B \
  --check_loading \
  --batch_size 1 --group_size 4 --max_steps 1 --max_turns 40
```

Inspect the run log to confirm the optimizer step completed. If all rollouts have identical rewards, the update may be skipped; use a different task or collect another rollout group.

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
