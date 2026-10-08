"""Independent checks for process milestones in credentialed Airbyte rollouts.

Command output is not evidence: Terraform resource IDs are resolved against the
Airbyte API, sync jobs are read back from the API, and dbt success is paired with
the grader's observation of a materialized target relation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AIRBYTE_API = "http://localhost:8000/api/public/v1"
TERRAFORM_REWARD = 0.05
SYNC_REWARD = 0.10
DBT_REWARD = 0.05


@dataclass(frozen=True)
class VerifiedMilestones:
    terraform_resources: bool = False
    sync_validated: bool = False
    load_validation_score: float = 0.0
    dbt_materialized: bool = False

    @property
    def score(self) -> float:
        return (
            TERRAFORM_REWARD * self.terraform_resources
            + SYNC_REWARD * self.load_validation_score
            + DBT_REWARD * self.dbt_materialized
        )

    def metrics(self) -> dict[str, float]:
        return {
            "trajectory/stage_terraform_verified": float(self.terraform_resources),
            "trajectory/stage_sync_validated": float(self.sync_validated),
            "trajectory/stage_load_validation": self.load_validation_score,
            "trajectory/stage_dbt_materialized": float(self.dbt_materialized),
            "trajectory/stage_score": self.score,
        }


def _terraform_graph(workspace: Path):
    """Extract the complete source/destination/connection graph from tfstate."""
    state_path = Path(workspace) / "elt" / "terraform.tfstate"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or not isinstance(state.get("resources"), list):
        return None

    sources: set[str] = set()
    destinations: set[str] = set()
    connections: list[tuple[str, str, str]] = []
    for resource in state.get("resources", []):
        if not isinstance(resource, dict):
            continue
        kind = str(resource.get("type", ""))
        for instance in resource.get("instances", []):
            if not isinstance(instance, dict):
                continue
            attrs = instance.get("attributes") or {}
            if kind.startswith("airbyte_source_") and attrs.get("source_id"):
                sources.add(str(attrs["source_id"]))
            elif kind.startswith("airbyte_destination_") and attrs.get(
                "destination_id"
            ):
                destinations.add(str(attrs["destination_id"]))
            elif kind == "airbyte_connection":
                cid = attrs.get("connection_id")
                sid = attrs.get("source_id")
                did = attrs.get("destination_id")
                if cid and sid and did:
                    connections.append((str(cid), str(sid), str(did)))
    if not sources or not destinations or not connections:
        return None
    if {source for _, source, _ in connections} != sources or {
        destination for _, _, destination in connections
    } != destinations:
        return None
    return sources, destinations, connections


def _airbyte_token(credentials_path: Path) -> str | None:
    try:
        credentials = json.loads(credentials_path.read_text(encoding="utf-8"))
        body = urllib.parse.urlencode(
            {
                "client_id": credentials["client_id"],
                "client_secret": credentials["client_secret"],
                "grant_type": "client_credentials",
            }
        ).encode()
        request = urllib.request.Request(
            f"{os.environ.get('AIRBYTE_API', AIRBYTE_API)}/applications/token",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read()).get("access_token")
    except (OSError, KeyError, ValueError, urllib.error.URLError):
        return None


def _airbyte_get(path: str, token: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"{os.environ.get('AIRBYTE_API', AIRBYTE_API)}{path}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _verified_airbyte_connections(
    workspace: Path, target_database: str, credentials_path: Path
) -> tuple[bool, list[str]]:
    graph = _terraform_graph(workspace)
    if graph is None:
        return False, []
    source_ids, destination_ids, connections = graph
    token = _airbyte_token(credentials_path)
    if not token:
        return False, []
    sources = {
        source_id: _airbyte_get(f"/sources/{urllib.parse.quote(source_id)}", token)
        for source_id in source_ids
    }
    destinations = {
        destination_id: _airbyte_get(
            f"/destinations/{urllib.parse.quote(destination_id)}", token
        )
        for destination_id in destination_ids
    }
    if any(
        not isinstance(value, dict) or str(value.get("sourceId")) != key
        for key, value in sources.items()
    ):
        return False, []
    if any(
        not isinstance(value, dict)
        or str(value.get("destinationId")) != key
        or str((value.get("configuration") or {}).get("database", "")).casefold()
        != str(target_database).casefold()
        for key, value in destinations.items()
    ):
        return False, []
    workspace_ids = {str(value.get("workspaceId")) for value in sources.values()}
    workspace_ids.update(
        str(value.get("workspaceId")) for value in destinations.values()
    )
    connection_ids = []
    for connection_id, source_id, destination_id in connections:
        connection = _airbyte_get(
            f"/connections/{urllib.parse.quote(connection_id)}", token
        )
        if (
            not isinstance(connection, dict)
            or str(connection.get("connectionId")) != connection_id
            or str(connection.get("sourceId")) != source_id
            or str(connection.get("destinationId")) != destination_id
            or str(connection.get("status", "")).lower() != "active"
        ):
            return False, []
        workspace_ids.add(str(connection.get("workspaceId")))
        connection_ids.append(connection_id)
    if len(workspace_ids) != 1 or "None" in workspace_ids:
        return False, []
    return True, connection_ids


def _has_succeeded_sync(connection_id: str, credentials_path: Path) -> bool:
    token = _airbyte_token(credentials_path)
    if not token:
        return False
    query = urllib.parse.urlencode({"connectionId": connection_id, "limit": 100})
    result = _airbyte_get(f"/jobs?{query}", token)
    # Airbyte's public API returns newest jobs first; one page is enough for the
    # bounded rollout. Never infer success from the command text or CLI output.
    jobs = result.get("data", []) if isinstance(result, dict) else []
    return any(
        job.get("jobType") == "sync"
        and job.get("status") == "succeeded"
        and str(job.get("connectionId")) == connection_id
        for job in jobs
    )


def _dbt_run_matches(workspace: Path, graded_models) -> bool:
    """Require a successful dbt model result and a corresponding graded table."""
    existing = {m.model.casefold() for m in graded_models if m.got_rows > 0}
    if not existing:
        return False
    for run_results in Path(workspace).glob("**/target/run_results.json"):
        try:
            payload = json.loads(run_results.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for result in payload.get("results", []) if isinstance(payload, dict) else []:
            if not isinstance(result, dict):
                continue
            node = result.get("unique_id", "").split(".")
            if (
                result.get("status") == "success"
                and len(node) >= 3
                and node[0] == "model"
                and node[-1].casefold() in existing
            ):
                return True
    return False


def verify_milestones(
    *,
    workspace: Path,
    target_database: str,
    credentials_path: Path | None,
    graded_models,
    load_score: float,
    load_evaluated: bool,
) -> VerifiedMilestones:
    """Verify applicable milestones; infrastructure/API uncertainty fails closed.

    Local preloaded Stage 1 is intentionally not treated as agent achievement.
    dbt verification is backend-independent; Terraform/Airbyte checks need the
    credentialed public API and a rollout-local Terraform state.
    """
    terraform_ok = sync_ok = False
    validation_score = 0.0
    if credentials_path is not None and Path(credentials_path).is_file():
        terraform_ok, connection_ids = _verified_airbyte_connections(
            workspace, target_database, Path(credentials_path)
        )
        sync_ok = bool(
            terraform_ok
            and connection_ids
            and load_evaluated
            and load_score > 0
            and all(
                _has_succeeded_sync(cid, Path(credentials_path))
                for cid in connection_ids
            )
        )
        if sync_ok:
            validation_score = min(1.0, max(0.0, load_score))
    return VerifiedMilestones(
        terraform_resources=terraform_ok,
        sync_validated=sync_ok,
        load_validation_score=validation_score,
        dbt_materialized=_dbt_run_matches(workspace, graded_models),
    )
