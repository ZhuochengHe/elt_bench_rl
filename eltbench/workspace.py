"Build task workspaces and isolate warehouse state per rollout."

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from . import benchmark

WORKSPACE_IGNORE = shutil.ignore_patterns("*_credential.json")


SF_SCHEMA = "AIRBYTE_SCHEMA"


@dataclass(frozen=True)
class RolloutNamespace:
    destination: str
    task: str
    index: int

    eval_database: str

    eval_schema: str

    physical_schema: str = ""

    qualified: str = ""

    @property
    def label(self) -> str:
        return f"{self.task}[r{self.index}]"

    @property
    def target(self) -> str:
        return f"{self.eval_database}.{self.physical_schema or self.eval_schema}"


@dataclass
class RolloutWorkspace:
    root: Path
    namespace: RolloutNamespace

    destination_config: dict[str, Any] = field(default_factory=dict)

    harness_config: dict[str, Any] = field(default_factory=dict)


def attempt_tag(run_tag: str, attempt: int) -> str:
    return f"{run_tag}a{attempt}"


def plan_namespace(
    destination: str,
    task: str,
    base_config: dict[str, Any],
    index: int,
    run_tag: str,
) -> tuple[RolloutNamespace, dict[str, Any]]:
    base_db = str(base_config["database"])
    base_schema = str(base_config.get("schema") or SF_SCHEMA)

    slug = re.sub(r"[^0-9a-zA-Z_]", "_", str(task))

    if destination == "snowflake":
        #

        db = f"{base_db}_{slug}_{run_tag}_{index}".upper()
        schema = base_schema
        ns = RolloutNamespace(
            destination,
            task,
            index,
            eval_database=db,
            eval_schema=db,
            physical_schema=schema,
        )
    else:
        db = base_db
        schema = f"{base_schema}_{slug}_{run_tag}_{index}"
        ns = RolloutNamespace(
            destination,
            task,
            index,
            eval_database=db,
            eval_schema=schema,
            physical_schema=schema,
        )
    return ns, {**base_config, "database": db, "schema": schema}


@contextmanager
def _stage_lock(lock_path: Path, timeout: float = 300.0) -> Iterator[None]:
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Timed out waiting for the staging lock: {lock_path}"
                    )
                time.sleep(0.2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _credential_fingerprint(destination: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in (
        benchmark.credential_path_for(destination),
        benchmark.REPO / "setup" / "airbyte" / "airbyte_credential.json",
    ):
        digest.update(path.name.encode())
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def stage_inputs(destination: str, staging_root: Path) -> Path:
    staging_root = Path(staging_root)
    marker = staging_root / ".eltbench-staged"
    fingerprint = _credential_fingerprint(destination)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip().endswith(
        fingerprint
    ):
        return staging_root
    with _stage_lock(staging_root.parent / (staging_root.name + ".lock")):
        if marker.is_file() and marker.read_text(encoding="utf-8").strip().endswith(
            fingerprint
        ):
            return staging_root
        count, dest = benchmark.generate_inputs(
            destination,
            dest=staging_root,
            credential_path=benchmark.credential_path_for(destination),
        )
        marker.write_text(
            f"{destination} {count} {dest} {fingerprint}\n", encoding="utf-8"
        )
    return staging_root


def materialize_workspace(
    task: str,
    root: Path,
    *,
    destination: str,
    index: int,
    run_tag: str,
    staging_root: Path,
    base_destination_config: dict[str, Any] | None = None,
    harness_config: dict[str, Any] | None = None,
    credential_path: str | Path | None = None,
    prepare: bool = True,
) -> RolloutWorkspace:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    staged = stage_inputs(destination, Path(staging_root)) / task
    if not staged.is_dir():
        raise FileNotFoundError(
            f"Task {task} is missing from generated benchmark inputs: {staged}"
        )
    shutil.copytree(staged, root, dirs_exist_ok=True, ignore=WORKSPACE_IGNORE)

    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    fixture = benchmark.fixture_destination(destination)
    if base_destination_config is not None:
        base_config = dict(base_destination_config)
    else:
        try:
            base_config = dict(config[destination]["config"])
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{config_path} is missing {destination}.config (invalid task specification)."
            ) from exc

    namespace, block = plan_namespace(destination, task, base_config, index, run_tag)

    #

    harness_override = {
        k: v
        for k, v in (harness_config or {}).items()
        if k not in ("database", "schema")
    }
    harness_block = {**block, **harness_override}

    if fixture != destination:
        config.pop(fixture, None)
    config[destination] = {"config": block}
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    if prepare:
        namespace = replace_qualified(
            namespace,
            benchmark.prepare_destination(
                destination, root, credential_path, config_override=harness_block
            ),
        )

    if prepare and destination == benchmark.LOCAL_DESTINATION:
        try:
            block = _local_side_setup(
                task, namespace, block, harness_block, config_path
            )
        except Exception:
            try:
                drop_namespace(destination, harness_block, namespace)
            except Exception as cleanup_exc:  # noqa: BLE001
                print(
                    f"[workspace] Cleanup also failed after initialization error: {cleanup_exc}"
                )
            raise

    return RolloutWorkspace(
        root=root,
        namespace=namespace,
        destination_config=block,
        harness_config=harness_block,
    )


def _local_side_setup(
    task: str,
    namespace: RolloutNamespace,
    block: dict[str, Any],
    harness_block: dict[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    from .destinations import (
        provision_scoped_role,
        scoped_role_name,
    )

    attach_on = os.environ.get("ELT_ATTACH_SOURCES", "1") not in ("0", "false", "no")
    scope_on = os.environ.get("ELT_SCOPED_CREDENTIALS", "1") not in ("0", "false", "no")
    role = scoped_role_name(namespace.eval_schema) if scope_on else None

    source_db = task
    if scope_on:
        block = {
            **block,
            **provision_scoped_role(
                harness_block,
                target_db=namespace.eval_database,
                target_schema=namespace.eval_schema,
                source_db=source_db,
                source_tables=benchmark.source_tables(task),
            ),
        }
        creds = {k: block[k] for k in ("user", "password")}

        n = _rewrite_sql_blocks(config_path, creds)
        print(
            f"[workspace] Scoped credentials to {role} (write access to {namespace.eval_schema}; "
            f"read-only access to {source_db}; updated {n} config blocks)."
        )

    if attach_on:
        _attach_sources(task, namespace, harness_block, mapping_role=role)

        _attach_other_sources(task, namespace, harness_block, config_path, role)
        _validate_local_source_tables(task, namespace, harness_block)

    return block


def _attach_other_sources(
    task: str,
    namespace: RolloutNamespace,
    harness_block: dict[str, Any],
    config_path: Path,
    role: str | None,
) -> None:
    from . import localsources

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    written = localsources.materialize_non_postgres_sources(
        config,
        target_db=namespace.eval_database,
        target_schema=namespace.eval_schema,
        task=task,
        harness=harness_block,
    )
    if not written:
        print(
            f"[workspace] No non-PostgreSQL source tables were materialized for {task}."
        )
        return
    if role:
        import psycopg2

        conn = psycopg2.connect(
            host=harness_block["host"],
            port=int(harness_block["port"]),
            user=harness_block["user"],
            password=harness_block["password"],
            dbname=namespace.eval_database,
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"GRANT SELECT ON ALL TABLES IN SCHEMA "
                    f'"{namespace.eval_schema}" TO "{role}"'
                )
        finally:
            conn.close()
    expected = benchmark.source_table_manifest(task)
    bad = {
        t: (n, expected.get(t))
        for t, n in written.items()
        if expected.get(t) not in (None, n)
    }
    print(
        f"[workspace] Materialized {len(written)} non-PostgreSQL source tables: "
        f"{ {t: n for t, n in written.items()} }"
        + (
            f"  Warning: row counts differ from table.json: {bad}"
            if bad
            else "  (all row counts match)"
        )
    )


def _validate_local_source_tables(
    task: str, namespace: RolloutNamespace, harness_block: dict[str, Any]
) -> None:
    import psycopg2
    from psycopg2 import sql

    expected = benchmark.source_table_manifest(task)
    if not expected:
        raise RuntimeError(
            f"The source-table manifest is empty for task {task}; refusing rollout."
        )
    conn = psycopg2.connect(
        host=harness_block["host"],
        port=int(harness_block["port"]),
        user=harness_block["user"],
        password=harness_block["password"],
        dbname=namespace.eval_database,
    )
    problems: list[str] = []
    try:
        with conn.cursor() as cur:
            for table, want in sorted(expected.items()):
                try:
                    cur.execute(
                        sql.SQL("SELECT count(*) FROM {}.{}").format(
                            sql.Identifier(namespace.eval_schema), sql.Identifier(table)
                        )
                    )
                    got = int(cur.fetchone()[0])
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{table}: unreadable ({type(exc).__name__})")
                    conn.rollback()
                    continue
                if got != want:
                    problems.append(f"{table}: got {got} rows, expected {want}")
    finally:
        conn.close()
    if problems:
        raise RuntimeError(
            f"Local source data is not ready for task {task}; refusing rollout: "
            + "; ".join(problems)
        )


def _rewrite_sql_blocks(config_path: Path, creds: dict[str, str]) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    changed = 0
    for value in config.values():
        if not isinstance(value, dict) or not isinstance(value.get("config"), dict):
            continue
        cfg = value["config"]
        if all(k in cfg for k in ("host", "port", "user", "password")):
            cfg.update(creds)
            changed += 1
    if changed:
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    return changed


def _attach_sources(
    task: str,
    namespace: RolloutNamespace,
    harness_block: dict[str, Any],
    mapping_role: str | None = None,
) -> None:
    from .destinations import attach_source_tables

    tables = benchmark.source_tables(task)
    if not tables:
        return
    attached = attach_source_tables(
        harness_block,
        target_db=namespace.eval_database,
        target_schema=namespace.eval_schema,
        source_db=task,
        source_schema="public",
        tables=tables,
        mapping_role=mapping_role,
    )
    models = set(benchmark.task_models(task))
    clash = sorted(models & set(attached))
    if clash:
        print(
            f"[workspace] Task {task} has source tables with model-name conflicts {clash}; "
            "a dbt model with the same name will conflict."
        )
    print(
        f"[workspace] Mounted source tables in {namespace.eval_database}.{namespace.eval_schema}: "
        f"{attached}"
    )


def replace_qualified(namespace: RolloutNamespace, qualified: str) -> RolloutNamespace:
    return dataclasses.replace(namespace, qualified=qualified)


def drop_namespace(
    destination: str, config: dict[str, Any], namespace: RolloutNamespace
) -> None:
    if destination != benchmark.LOCAL_DESTINATION:
        return
    from .destinations import PostgresConnector

    connector = PostgresConnector(config)
    try:
        connector.drop_namespace(namespace.eval_database, namespace.eval_schema)
    finally:
        connector.close()
    if os.environ.get("ELT_SCOPED_CREDENTIALS", "1") not in ("0", "false", "no"):
        from .destinations import drop_scoped_role

        try:
            drop_scoped_role(
                config,
                target_db=namespace.eval_database,
                target_schema=namespace.eval_schema,
                source_db=namespace.task,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[workspace] Failed to remove the scoped role for {namespace.eval_schema}: {exc}"
            )


# ------------------------------------------------------------------ tag
def new_run_tag() -> str:
    return uuid.uuid4().hex[:6]


def keep_namespaces() -> bool:
    return os.environ.get("ELT_KEEP_NAMESPACES", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
