import json
from pathlib import Path
from types import SimpleNamespace

from eltbench import milestones


def _write_state(
    root: Path, *, source="src-1", destination="dst-1", connection="conn-1"
):
    path = root / "elt" / "terraform.tfstate"
    path.parent.mkdir(parents=True)
    resources = []
    for kind, key, value in (
        ("airbyte_source_postgres", "source_id", source),
        ("airbyte_destination_snowflake", "destination_id", destination),
    ):
        resources.append({"type": kind, "instances": [{"attributes": {key: value}}]})
    resources.append(
        {
            "type": "airbyte_connection",
            "instances": [
                {
                    "attributes": {
                        "connection_id": connection,
                        "source_id": source,
                        "destination_id": destination,
                    }
                }
            ],
        }
    )
    path.write_text(json.dumps({"resources": resources}), encoding="utf-8")


def test_terraform_milestone_requires_live_linked_resources_and_target_db(
    tmp_path, monkeypatch
):
    _write_state(tmp_path)
    monkeypatch.setattr(milestones, "_airbyte_token", lambda _: "token")
    values = {
        "/connections/conn-1": {
            "connectionId": "conn-1",
            "sourceId": "src-1",
            "destinationId": "dst-1",
            "workspaceId": "w-1",
            "status": "active",
        },
        "/sources/src-1": {"sourceId": "src-1", "workspaceId": "w-1"},
        "/destinations/dst-1": {
            "destinationId": "dst-1",
            "workspaceId": "w-1",
            "configuration": {"database": "RL_DB"},
        },
    }
    monkeypatch.setattr(
        milestones, "_airbyte_get", lambda path, token: values.get(path)
    )
    ok, connections = milestones._verified_airbyte_connections(
        tmp_path, "rl_db", Path("app-creds.json")
    )
    assert ok and connections == ["conn-1"]
    ok, connections = milestones._verified_airbyte_connections(
        tmp_path, "some-other-db", Path("app-creds.json")
    )
    assert not ok and connections == []


def test_sync_milestone_requires_successful_sync_and_validated_load(
    monkeypatch, tmp_path
):
    _write_state(tmp_path)
    monkeypatch.setattr(
        milestones, "_verified_airbyte_connections", lambda *args: (True, ["conn-1"])
    )
    monkeypatch.setattr(milestones, "_airbyte_token", lambda _: "token")
    monkeypatch.setattr(
        milestones,
        "_airbyte_get",
        lambda path, token: {
            "data": [
                {"jobType": "sync", "status": "succeeded", "connectionId": "conn-1"}
            ]
        },
    )
    cred = Path("app-creds.json")
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == str(cred))
    model = SimpleNamespace(model="target", got_rows=4)
    got = milestones.verify_milestones(
        workspace=tmp_path,
        target_database="db",
        credentials_path=cred,
        graded_models=[model],
        load_score=0.5,
        load_evaluated=True,
    )
    assert got.terraform_resources
    assert got.sync_validated
    assert got.score == milestones.TERRAFORM_REWARD + milestones.SYNC_REWARD * 0.5

    # A successful job without a positive grader-verified load earns no sync stage.
    got = milestones.verify_milestones(
        workspace=tmp_path,
        target_database="db",
        credentials_path=cred,
        graded_models=[model],
        load_score=0,
        load_evaluated=True,
    )
    assert not got.sync_validated


def test_dbt_milestone_requires_success_artifact_and_materialized_graded_model(
    tmp_path,
):
    target = tmp_path / "elt" / "target"
    target.mkdir(parents=True)
    (target / "run_results.json").write_text(
        json.dumps(
            {
                "results": [
                    {"unique_id": "model.project.target", "status": "success"},
                ]
            }
        ),
        encoding="utf-8",
    )
    model = SimpleNamespace(model="target", got_rows=2)
    got = milestones.verify_milestones(
        workspace=tmp_path,
        target_database="db",
        credentials_path=None,
        graded_models=[model],
        load_score=0,
        load_evaluated=False,
    )
    assert got.dbt_materialized
    assert got.score == milestones.DBT_REWARD

    failed_model = SimpleNamespace(model="target", got_rows=0)
    got = milestones.verify_milestones(
        workspace=tmp_path,
        target_database="db",
        credentials_path=None,
        graded_models=[failed_model],
        load_score=0,
        load_evaluated=False,
    )
    assert not got.dbt_materialized
