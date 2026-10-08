# Implementation and Reward Design

## Task and rollout state

An ELT-Bench task provides source definitions, a destination configuration, source schemas, documentation, a requested data model, and ground-truth output tables. The harness builds an isolated workspace from the pinned benchmark repository and presents it with the benchmark's SWE-agent tool interface. The agent can inspect and edit files, run shell commands, execute SQL, and submit.

Each rollout has its own workspace, execution container, and destination namespace. The workspace configuration and grader use the same namespace. This isolation prevents concurrent samples from overwriting one another's state. A rollout ends on submission or when the configured turn, tool, token, or time budget is exhausted. The warehouse state is persistent within a rollout and reset between rollouts.

## Correctness and outcome reward

The outcome grader reads the resulting warehouse relations and compares them with the task's ground truth using the benchmark's model names and sort keys. It measures row-count agreement and column values, including null and numeric comparison semantics. The reward reports a benchmark-aligned exact model score and a denser partial score for training.

For each model, the partial score combines row-count agreement with the fraction of ground-truth columns whose complete values match. Model scores are averaged. When loading is evaluated, the configured loading score gates transformation reward and contributes a separate loading component. This discourages a model from earning a transformation score on missing or incorrectly loaded inputs.

## Verified process milestones

Process rewards are awarded only for observable artifacts or externally verified state, not for command count, successful exit status alone, or repeated queries.

| Milestone | Evidence | Maximum reward |
| --- | --- | ---: |
| Terraform resources | Rollout-local Terraform state contains a complete source/destination/connection graph, and Airbyte API responses confirm the linked active resources and target warehouse | 0.05 |
| Sync and loading | Airbyte reports a successful sync for the verified connection and the warehouse passes the loading checks | 0.10 |
| dbt materialization | dbt run results report success for a model that the warehouse grader can observe | 0.05 |
| Documentation use | The trajectory reads task-relevant documentation before work begins, with a smaller award for reading it after edits but before dbt execution | Up to 0.30 |

Terraform and Airbyte milestones fail closed when state or API evidence is unavailable. Locally preloaded source data is not counted as agent-achieved extraction/loading progress. Documentation reward is independent of downstream success, so reading the task specification remains useful even when the rest of the trajectory fails.

## Reward aggregation

The episode reward combines the gated model partial score, optional loading score, verified infrastructure milestones, and documentation-use signal. The exact weights and aggregation logic live in `eltbench/reward.py` and `eltbench/milestones.py`; tests cover their boundaries. Reported exact model accuracy remains separate from dense training reward.

Budget termination is handled by the Tinker Cookbook agentic rollout configuration. Tool execution is sequential because all tools share the same workspace and container. Scoring failures are recorded as metrics and do not fabricate a positive reward.

## Reward-hacking controls

- Every rollout has a distinct warehouse namespace; an agent cannot benefit from another sample's tables.
- Grading reads the warehouse, not model claims or shell output.
- Terraform and sync rewards require state plus API verification.
- dbt reward requires both a successful model result and an observed graded relation.
- Documentation reward recognizes content reads, not directory listings or generic shell activity.
- Duplicate commands and repeated successful exits do not themselves increase reward.
- The execution container is frozen before final grading to prevent background writes during the grading window.

## Extensibility and efficiency

Destination-specific behavior is concentrated in the benchmark adapter and warehouse connector. Reward calculation consumes a shared table-reader contract, keeping task logic independent of a specific warehouse. A new destination can provide namespace reset, connection, and identifier semantics without rewriting the rollout loop or model grader.

The harness caches static benchmark metadata and source snapshots within the process, bounds rollout concurrency and execution time, and uses the benchmark's existing task generation and evaluation assets rather than duplicating them. Full-table comparisons remain necessary for trustworthy execution-derived reward; optimization should preserve those semantics.

## Validation

The test suite checks reward semantics, official benchmark integration, namespace isolation, destination adapters, and milestone verification. The manual local integration rollout exercises workspace generation, container tools, dbt execution, warehouse reads, and final scoring without Tinker credentials. A credentialed rollout additionally requires live Airbyte and warehouse credentials; local preloading does not validate that extraction path.
