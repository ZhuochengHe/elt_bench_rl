# Setup

This guide prepares the local integration environment and one official Snowflake rollout. The local integration uses a credential-free fixture; the Snowflake run exercises Airbyte and Tinker and may create billable warehouse resources.

## Install

Requirements: Linux, macOS, or WSL2; Git; [uv](https://docs.astral.sh/uv/); Docker Engine with Compose. The Snowflake flow additionally needs `abctl`, `kubectl`, AWS CLI, `psql`, `unzip`, a Snowflake account, and a Tinker API key.

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

Complete this one-time checklist before a Snowflake rollout:

- [ ] Install Airbyte **1.5.0**. The benchmark Terraform provider is pinned to **0.6.5** in the upstream configuration.
- [ ] Register the declarative connector image version in Airbyte's database using the command below. The `elt_snowflake.yaml` manifest uses version `6.33.4`, which requires the corresponding major-version `6` row.
- [ ] Follow the upstream [Airbyte setup instructions](https://github.com/uiuc-kang-lab/ELT-Bench#setup-airbyte): import `repo/setup/elt_snowflake.yaml`, publish it (confirm the warning prompt), and record the Workspace ID and API Definition ID.
- [ ] Follow the upstream [Snowflake destination setup](https://github.com/uiuc-kang-lab/ELT-Bench#snowflake): replace the sample role, user, warehouse, schema, and password values in `repo/setup/destination/setup.sql`, then run it in a Snowflake worksheet. As `ACCOUNTADMIN`, grant `CREATE DATABASE ON ACCOUNT` to the created role (for the default role: `GRANT CREATE DATABASE ON ACCOUNT TO ROLE AIRBYTE_ROLE;`).
- [ ] Fill in `repo/setup/airbyte/airbyte_credential.json` with Airbyte credentials and the two IDs. Copy `repo/setup/destination/snowflake_credential.json` to `.secrets/destination/snowflake_credential.json` and fill in the matching account, user, password, role, and warehouse values.

Install the pinned Airbyte version:

```bash
abctl local install --chart-version 1.5.0
```

Register the declarative connector image version before publishing the source manifest:

```bash
export KUBECONFIG="$HOME/.airbyte/abctl/abctl.kubeconfig"
IMAGE_SHA="$(docker buildx imagetools inspect airbyte/source-declarative-manifest:6.33.4 | awk '/Digest:/ {print $2; exit}')"
kubectl --context kind-airbyte-abctl -n airbyte-abctl exec airbyte-db-0 -- \
  psql -U airbyte -d db-airbyte -c "INSERT INTO declarative_manifest_image_version \
  (major_version, image_version, image_sha, created_at, updated_at) \
  VALUES (6, '6.33.4', '$IMAGE_SHA', now(), now()) \
  ON CONFLICT (major_version) DO NOTHING;"
kubectl --context kind-airbyte-abctl -n airbyte-abctl \
  rollout restart deployment/airbyte-abctl-server
```

Then set the credential locations and Tinker key in your shell:

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
