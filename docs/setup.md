# Setup

This guide covers dependency installation, local validation, benchmark data preparation, and credentialed rollouts. The local PostgreSQL mode is intended for credential-free development; source data is prepared locally and this mode does not exercise live extraction through Airbyte.

## Prerequisites

- Linux, macOS, or WSL2
- [uv](https://docs.astral.sh/uv/)
- Docker Engine with Compose
- `unzip`, AWS CLI, and `psql` for local source setup
- Git and Git LFS-compatible network access for the benchmark's external assets
- A Tinker API key for model sampling and optimization

Clone the repository including the pinned ELT-Bench submodule, then install Python dependencies:

```bash
git clone --recurse-submodules https://github.com/ZhuochengHe/elt_bench_rl.git
cd elt_bench_rl
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If the repository was cloned without submodules, initialize the pinned dependency with:

```bash
git submodule update --init --recursive
```

## Credential-free checks

The benchmark's ground-truth CSVs are distributed separately from the ELT-Bench Git repository. Download them into the fixed `runs/benchmark-assets/evaluation/gt` location before running tests or training; the application discovers this path automatically:

```bash
bash scripts/fetch_ground_truth.sh
```

The script downloads the upstream ELT-Bench ground-truth archive. `gdown` is installed with the project dependencies.

Run the unit and isolated integration suite:

```bash
pytest -q
```

The local end-to-end rollout also requires ELT-Bench source data, local services, and the agent image. Download and extract the upstream source-data archives with the project helper. The archives are cached under `runs/benchmark-assets/upstream`; extracted data stays in the submodule's expected local data directories and is not committed:

```bash
bash scripts/fetch_local_assets.sh
```

Start the services using the upstream Compose file plus the repository-maintained local overlay:

```bash
bash scripts/start_local_services.sh
```

The overlay pins LocalStack to a compatible release and applies local container defaults without modifying the submodule. Seed the PostgreSQL and S3 fixtures required by a task:

```bash
bash scripts/seed_resumable.sh address
```

The helper accepts task names as positional arguments. Without arguments it checks all available source definitions. It sets the S3-compatible storage checksum option required by the local seed scripts. For MongoDB-backed sources, use `python scripts/mongo_seed_resumable.py --path repo/setup`. These helpers require Docker, the AWS CLI, and `psql`.

Build the agent image from the project root:

```bash
docker build -f docker/Dockerfile.elt-swe -t elt-swe:local .
```

Run the full local dbt-to-grader integration probe:

```bash
python tests/manual/rollout_local.py all
```

Run `python tests/manual/rollout_local_sources.py` when validating local source materialization. These manual checks require the local database and Docker environment; the regular pytest suite skips external-service checks when prerequisites are unavailable.

## Local RL training

Export the Tinker credential in the shell that runs training:

```bash
export TINKER_API_KEY="..."
python -m eltbench.train \
  --destination local_postgres \
  --model_name Qwen/Qwen3-8B \
  --batch_size 1 --group_size 4 --max_turns 40
```

Do not put API keys or warehouse credentials in `.env` files that might be committed. Keep credentials in your shell environment or a local secret manager. Training artifacts are written under `runs/` by default and are ignored by Git.

## Credentialed warehouse rollouts

The harness supports the official ELT-Bench destinations that have upstream connector support. Set up Airbyte and the selected warehouse using the upstream instructions in [`repo/README.md`](../repo/README.md). Download and seed only the data required for the chosen task when possible. If Airbyte runs in its own local Kubernetes network, connect its control-plane container to the ELT-Bench network created by the helper:

```bash
docker network connect elt-docker_elt_network airbyte-abctl-control-plane
```

Keep warehouse credentials outside the submodule so its tracked templates remain untouched. For Snowflake, copy the template into the local secrets directory and fill in the required values in the copied file:

```bash
mkdir -p .secrets/destination
cp repo/setup/destination/snowflake_credential.json \
  .secrets/destination/snowflake_credential.json
export ELT_BENCH_CREDENTIAL_DIR="$PWD/.secrets/destination"
```

The credential directory can contain the matching `<destination>_credential.json` for Databricks or Redshift. `.secrets/` is ignored by Git. Do not commit populated credential files or modify the tracked submodule templates.

The upstream input generator reads Airbyte source credentials from `repo/setup/airbyte/airbyte_credential.json`. Populate that file locally using the Airbyte setup instructions in the upstream README. Because this template is tracked by the upstream repository, its local edit will mark the submodule as dirty; do not commit or push from inside the submodule. The main repository records only the pinned upstream commit. Generated task workspaces may contain connector configuration needed by the agent; keep `runs/` and generated inputs local, and do not share rollout artifacts containing credentials.

Example Snowflake rollout:

```bash
python -m eltbench.train \
  --destination snowflake \
  --model_name Qwen/Qwen3-8B \
  --check_loading \
  --batch_size 1 --group_size 1 --max_turns 40
```

Start with a small batch and group size: each concurrent rollout starts a Docker container and uses an isolated warehouse namespace. Credentialed runs can create billable cloud resources. Review the generated Terraform plan and clean up resources after testing.

## Configuration

Local PostgreSQL connection settings can be overridden with `ELT_PG_HOST`, `ELT_PG_PORT`, `ELT_PG_USER`, `ELT_PG_PASSWORD`, `ELT_PG_CONTAINER_HOST`, and `ELT_PG_CONTAINER_PORT`. The default values target the upstream local Compose setup. Ground truth is read from `runs/benchmark-assets/evaluation/gt`. `ELT_BENCH_CREDENTIAL_DIR` optionally points to local warehouse credentials. The `TINKER_API_KEY` is required only for Tinker model calls.

Use `python -m eltbench.train --help` to inspect training options. Keep run output, generated task workspaces, downloaded assets, and credentials outside source control.
