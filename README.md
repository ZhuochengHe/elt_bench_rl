# ELT-Bench RL Environment for Tinker

This repository provides an execution-grounded reinforcement-learning environment for ELT-Bench tasks. An agent inspects a task workspace, configures extraction and loading, builds dbt transformations, uses execution feedback, and submits a pipeline. The environment grades the resulting warehouse state and returns outcome and verified milestone rewards.

The harness uses [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) for model sampling, tool-use rollouts, optimization, and checkpointing. The pinned [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) repository is included as a Git submodule and supplies the benchmark tasks, task-generation code, evaluation metadata, SWE-agent tools, and environment setup scripts.

## Quick start

Clone the repository with its benchmark dependency:

```bash
git clone --recurse-submodules <repository-url>
cd eltbench-tinker
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the benchmark ground truth, then run the credential-free tests:

```bash
python -m pip install gdown
bash scripts/fetch_ground_truth.sh
export ELT_BENCH_GT_DIR="$PWD/runs/benchmark-assets/evaluation/gt"
pytest -q
```

For local integration tests and training, prepare the local services and benchmark data described in [Setup](docs/setup.md). Set `TINKER_API_KEY` in your shell to run training; do not put credentials in source control.

```bash
python tests/manual/rollout_local.py all
python -m eltbench.train --destination local_postgres \
  --model_name Qwen/Qwen3-8B --batch_size 1 --group_size 4 --max_turns 40
```

For a credentialed warehouse rollout, configure the selected destination as described in [Setup](docs/setup.md), then run, for example:

```bash
python -m eltbench.train --destination snowflake \
  --model_name Qwen/Qwen3-8B --check_loading
```

## Documentation

- [Setup and data preparation](docs/setup.md)
- [Implementation and reward design](docs/design.md)
- [ELT-Bench upstream repository](repo/README.md)

## Repository layout

```text
eltbench/       Environment, tools, warehouse adapters, rewards, and training CLI
tests/          Unit and integration tests
scripts/        Environment and benchmark-data helper scripts
docker/         Agent execution image
docs/           User-facing setup and design documentation
repo/           Pinned ELT-Bench Git submodule
```

Local credentials, downloaded benchmark data, generated workspaces, logs, and training runs are intentionally excluded from this repository. See `.gitignore` and [Setup](docs/setup.md).

## Attribution

ELT-Bench is maintained by its upstream authors and is included as a pinned submodule under its own license. Review the upstream license and benchmark dataset terms before redistribution. This repository does not relicense upstream code or benchmark data.
