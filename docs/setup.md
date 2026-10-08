# Setup

This guide prepares the local integration environment and one official Snowflake rollout. The local integration uses a credential-free fixture; the Snowflake run exercises Airbyte and Tinker and may create billable warehouse resources.

## Install

Requirements: Linux, macOS, or WSL2; Git; [uv](https://docs.astral.sh/uv/); Docker Engine with Compose. The Snowflake flow additionally needs `abctl`, AWS CLI, `psql`, `unzip`, a Snowflake account, and a Tinker API key.

```bash
git clone --recurse-submodules https://github.com/ZhuochengHe/elt_bench_rl.git
cd elt_bench_rl
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Credential-free local integration

Run these from the repository root:

```bash
bash scripts/fetch_ground_truth.sh local_postgres
bash scripts/start_local_services.sh
docker build -f docker/Dockerfile.elt-swe -t elt-swe:local .
uv run --active python tests/manual/rollout_local.py all
```

This exercises dbt materialization, warehouse grading, and reward computation without warehouse credentials, model sampling, or Airbyte extraction.

## Snowflake rollout and one Tinker training step

Install the Airbyte platform version used by the upstream Terraform configuration:

```bash
abctl local install --chart-version 1.5.0
```

The upstream Terraform configuration pins provider version `0.6.5`. Fill in Airbyte source credentials in `repo/setup/airbyte/airbyte_credential.json`. Copy `repo/setup/destination/snowflake_credential.json` to `.secrets/destination/snowflake_credential.json` and fill in its required values. Then set the credential locations and Tinker key in your shell:

```bash
mkdir -p .secrets/destination
cp repo/setup/destination/snowflake_credential.json .secrets/destination/
export ELT_BENCH_CREDENTIAL_DIR="$PWD/.secrets/destination"
export TINKER_API_KEY="..."
```

Prepare the local source used by the `address` task, start the services, build the agent image if needed, and attach Airbyte to the benchmark network:

```bash
bash scripts/fetch_ground_truth.sh snowflake
bash scripts/fetch_local_assets.sh
bash scripts/start_local_services.sh
bash scripts/seed_resumable.sh address
docker build -f docker/Dockerfile.elt-swe -t elt-swe:local .
docker network connect elt-docker_elt_network airbyte-abctl-control-plane
```

Run one small training job on the prepared `address` task. It samples four rollouts and requests one optimizer step; the GRPO group size must be greater than one:

```bash
uv run --active python -m eltbench.train \
  --destination snowflake \
  --task_name address \
  --model_name Qwen/Qwen3-8B \
  --check_loading \
  --batch_size 1 --group_size 4 --max_steps 1 --max_turns 40
```

Check the run log for a completed training step and execution-derived reward metrics. If all four rollouts receive the same reward, Cookbook may skip the optimizer update; use a different task or collect another rollout group. Review and clean up Terraform-created resources after the rollout.

For other official destinations, use the matching credential file and download the corresponding ground-truth variant, for example `bash scripts/fetch_ground_truth.sh databricks`.
