"Warehouse connection adapters used by the reward grader."

from __future__ import annotations

import os
import re
import uuid
from typing import Sequence

from . import benchmark

SF_LAYOUT_SCHEMA = "AIRBYTE_SCHEMA"


# ------------------------------------------------------------------ Postgres
class PostgresConnector(benchmark.db_connectors.RedshiftConnector):
    BOOTSTRAP_DATABASE = "postgres"

    def _connect(self):
        import psycopg2

        cfg = self.config

        conn = psycopg2.connect(
            host=cfg.get("host", "localhost"),
            port=int(cfg.get("port", 5432)),
            dbname=cfg.get("database", "postgres"),
            user=cfg.get("user") or cfg.get("username", "postgres"),
            password=cfg.get("password", ""),
            connect_timeout=int(cfg.get("connect_timeout", 15)),
        )

        conn.autocommit = True
        self._apply_session_settings(conn)
        return conn

    @staticmethod
    def _apply_session_settings(conn) -> None:
        for stmt in (
            "SET TIME ZONE 'UTC'",
            "SET lock_timeout = '20s'",
            "SET statement_timeout = '600s'",
        ):
            try:
                cur = conn.cursor()
                cur.execute(stmt)
                cur.close()
            except Exception:  # noqa: BLE001
                pass

    def _use_database(self, database: str) -> None:
        if not database:
            return
        want = str(database)
        if str(self.config.get("database", "")).lower() == want.lower():
            return
        try:
            if (
                self._conn is not None
                and str(self._conn.info.dbname).lower() == want.lower()
            ):
                self.config["database"] = want
                return
        except Exception:  # noqa: BLE001
            pass
        self.close()
        self.config["database"] = want
        self.conn()

    def use_database(self, database: str) -> None:
        self._use_database(database)

    def verify_schema(self, database: str, schema: str) -> bool:
        self._use_database(database)
        return super().verify_schema(database, schema)

    def list_tables(self, database: str, schema: str) -> list:
        self._use_database(database)
        return super().list_tables(database, schema)

    def fetch_table(self, database: str, schema: str, table: str):
        self._use_database(database)
        return super().fetch_table(database, schema, table)

    def table_size(self, database: str, schema: str, table: str) -> int:
        self._use_database(database)
        return super().table_size(database, schema, table)

    # ------------------------------------------------------------ DDL
    def execute(self, sql: str) -> None:
        cur = self.conn().cursor()
        try:
            cur.execute(sql)
        finally:
            cur.close()

    def scalar(self, sql: str):
        cur = self.conn().cursor()
        try:
            cur.execute(sql)
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            cur.close()

    def _create_database_if_missing(self, database: str) -> None:
        boot = dict(self.config)
        boot["database"] = self.config.get(
            "bootstrap_database", self.BOOTSTRAP_DATABASE
        )
        probe = PostgresConnector(boot)
        try:
            if not probe.scalar(
                f"SELECT 1 FROM pg_database WHERE datname = '{database}'"
            ):
                probe.execute(f"CREATE DATABASE {probe._quote(database)}")
        finally:
            probe.close()

    def reset_namespace(self, database: str, schema: str) -> None:
        self._create_database_if_missing(database)
        self._use_database(database)
        self.execute(f"DROP SCHEMA IF EXISTS {self._quote(schema)} CASCADE")
        self.execute(f"CREATE SCHEMA {self._quote(schema)}")

    def drop_namespace(self, database: str, schema: str) -> None:
        self._use_database(database)
        self.execute(f"DROP SCHEMA IF EXISTS {self._quote(schema)} CASCADE")


def reset_postgres_namespace(config: dict) -> str:
    connector = PostgresConnector(config)
    try:
        connector.reset_namespace(config["database"], config["schema"])
    finally:
        connector.close()
    return f"{config['database']}.{config['schema']}"


class TableReader:
    def __init__(self, connector, database: str, schema: str | None = None):
        self.connector = connector
        self.database = database

        self.schema = schema or SF_LAYOUT_SCHEMA
        self._is_snowflake = type(connector).__name__.startswith("Snowflake")

    def fetch_table(self, database: str, schema: str, table: str):
        if self._is_snowflake:
            return self.connector.fetch_table(self.database, self.database, table)
        return self.connector.fetch_table(database or self.database, schema, table)

    def list_tables(self, database: str, schema: str = "") -> list:
        if self._is_snowflake:
            want = (schema or self.schema).upper()
            df = self.connector.query(
                f"SELECT table_name FROM {self.database}.information_schema.tables "
                f"WHERE table_schema = '{want}' "
                "AND table_type = 'BASE TABLE' ORDER BY table_name"
            )
            return [str(x) for x in df.iloc[:, 0]]
        return self.connector.list_tables(database or self.database, schema)

    def verify_schema(self, database: str, schema: str = "") -> bool:
        if self._is_snowflake:
            want = (schema or self.schema).upper()
            try:
                df = self.connector.query(
                    f"SELECT schema_name FROM {self.database}.information_schema.schemata"
                )
                names = {str(x).upper() for x in df.iloc[:, 0]}
                return want in names
            except Exception:  # noqa: BLE001
                return False
        return bool(self.connector.verify_schema(database or self.database, schema))

    def reset_namespace(self, database: str, schema: str) -> None:
        fn = getattr(self.connector, "reset_namespace", None)
        if callable(fn):
            fn(database or self.database, schema)
            return
        raise NotImplementedError(
            f"{type(self.connector).__name__} does not implement reset_namespace; "
            "use the upstream warehouse.prepare_destination() for Snowflake/Databricks."
        )

    def close(self) -> None:
        try:
            self.connector.close()
        except Exception:  # noqa: BLE001
            pass


def make_connector(destination: str, config: dict):
    key = (destination or "").lower()
    if key in ("", benchmark.LOCAL_DESTINATION):
        return PostgresConnector(config)
    return benchmark.db_connectors.get_connector(key, config)


def make_reader(
    destination: str, config: dict, database: str, schema: str | None = None
):
    raw = make_connector(destination, config)
    if (destination or "").lower() == "snowflake":
        return TableReader(raw, database, schema)
    return raw


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def attach_source_tables(
    config: dict,
    *,
    target_db: str,
    target_schema: str,
    source_db: str,
    source_schema: str = "public",
    tables: Sequence[str] = (),
    mapping_role: str | None = None,
) -> list[str]:
    import psycopg2

    if not tables:
        return []

    self_host = os.environ.get("ELT_PG_SELF_HOST", "127.0.0.1")
    self_port = int(os.environ.get("ELT_PG_SELF_PORT", "5432"))
    server = f"elt_src_{re.sub(r'[^0-9a-zA-Z_]', '_', source_db)}"

    conn = psycopg2.connect(
        host=config["host"],
        port=config["port"],
        user=config["user"],
        password=config["password"],
        dbname=target_db,
    )
    conn.autocommit = True
    attached: list[str] = []
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgres_fdw")
            cur.execute(
                f"CREATE SERVER IF NOT EXISTS {server} FOREIGN DATA WRAPPER postgres_fdw "
                f"OPTIONS (host %s, port %s, dbname %s)",
                (self_host, str(self_port), source_db),
            )

            cur.execute(
                f"CREATE USER MAPPING IF NOT EXISTS FOR CURRENT_USER SERVER {server} "
                f"OPTIONS (user %s, password %s)",
                (config["user"], config["password"]),
            )
            if mapping_role:
                cur.execute(
                    f"DROP USER MAPPING IF EXISTS FOR {_quote_ident(mapping_role)} "
                    f"SERVER {server}"
                )
                cur.execute(
                    f"CREATE USER MAPPING FOR {_quote_ident(mapping_role)} SERVER {server} "
                    f"OPTIONS (user %s, password %s, password_required %s)",
                    (config["user"], config["password"], "false"),
                )
            for t in tables:
                cur.execute(f'DROP FOREIGN TABLE IF EXISTS "{target_schema}"."{t}"')
            cols = ", ".join('"' + t.replace('"', '""') + '"' for t in tables)
            cur.execute(
                f"IMPORT FOREIGN SCHEMA {_quote_ident(source_schema)} LIMIT TO ({cols}) "
                f"FROM SERVER {server} INTO {_quote_ident(target_schema)}"
            )
            cur.execute(
                "SELECT foreign_table_name FROM information_schema.foreign_tables "
                "WHERE foreign_table_schema = %s ORDER BY foreign_table_name",
                (target_schema,),
            )
            imported = [r[0] for r in cur.fetchall()]
            if mapping_role:
                cur.execute(
                    f"GRANT SELECT ON ALL TABLES IN SCHEMA "
                    f"{_quote_ident(target_schema)} TO {_quote_ident(mapping_role)}"
                )

                cur.execute(
                    f"ALTER DEFAULT PRIVILEGES IN SCHEMA {_quote_ident(target_schema)} "
                    f"GRANT SELECT ON TABLES TO {_quote_ident(mapping_role)}"
                )

                #   "Non-superuser cannot connect if the server does not request
                #    a password or use GSSAPI with delegated credentials"

            attached = imported
    finally:
        conn.close()
    return attached


def scoped_role_name(schema: str) -> str:
    return ("elt_r_" + re.sub(r"[^0-9a-zA-Z_]", "_", str(schema)))[:63]


def provision_scoped_role(
    config: dict,
    *,
    target_db: str,
    target_schema: str,
    password: str | None = None,
    source_db: str | None = None,
    source_schema: str = "public",
    source_tables: Sequence[str] = (),
) -> dict[str, str]:
    import psycopg2

    role = scoped_role_name(target_schema)
    pwd = password or uuid.uuid4().hex
    conn = psycopg2.connect(
        host=config["host"],
        port=int(config["port"]),
        user=config["user"],
        password=config["password"],
        dbname=target_db,
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if cur.fetchone():
                cur.execute(
                    f"ALTER ROLE {_quote_ident(role)} WITH LOGIN PASSWORD %s", (pwd,)
                )
            else:
                cur.execute(
                    f"CREATE ROLE {_quote_ident(role)} WITH LOGIN PASSWORD %s", (pwd,)
                )

            cur.execute(
                f"GRANT CONNECT ON DATABASE {_quote_ident(target_db)} "
                f"TO {_quote_ident(role)}"
            )
            cur.execute(
                f"ALTER SCHEMA {_quote_ident(target_schema)} "
                f"OWNER TO {_quote_ident(role)}"
            )
            cur.execute(
                f"GRANT USAGE, CREATE ON SCHEMA {_quote_ident(target_schema)} "
                f"TO {_quote_ident(role)}"
            )

            cur.execute(
                f"GRANT SELECT ON ALL TABLES IN SCHEMA "
                f"{_quote_ident(target_schema)} TO {_quote_ident(role)}"
            )
    finally:
        conn.close()

    if source_db and source_tables:
        src = psycopg2.connect(
            host=config["host"],
            port=int(config["port"]),
            user=config["user"],
            password=config["password"],
            dbname=source_db,
        )
        src.autocommit = True
        try:
            with src.cursor() as cur:
                cur.execute(
                    f"GRANT CONNECT ON DATABASE {_quote_ident(source_db)} "
                    f"TO {_quote_ident(role)}"
                )
                cur.execute(
                    f"GRANT USAGE ON SCHEMA {_quote_ident(source_schema)} "
                    f"TO {_quote_ident(role)}"
                )
                for t in source_tables:
                    try:
                        cur.execute(
                            f"GRANT SELECT ON TABLE "
                            f"{_quote_ident(source_schema)}.{_quote_ident(t)} "
                            f"TO {_quote_ident(role)}"
                        )
                    except Exception:  # noqa: BLE001
                        continue
        finally:
            src.close()
    return {"user": role, "password": pwd}


def drop_scoped_role(
    config: dict, *, target_db: str, target_schema: str, source_db: str | None = None
) -> None:
    import psycopg2

    role = scoped_role_name(target_schema)

    def connect(db):
        c = psycopg2.connect(
            host=config["host"],
            port=int(config["port"]),
            user=config["user"],
            password=config["password"],
            dbname=db,
        )
        c.autocommit = True
        return c

    for db in dict.fromkeys([target_db, source_db] if source_db else [target_db]):
        if not db:
            continue
        try:
            conn = connect(db)
        except Exception:  # noqa: BLE001
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(f"DROP OWNED BY {_quote_ident(role)}")
        except Exception:  # noqa: BLE001
            pass
        finally:
            conn.close()

    conn = connect(target_db)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
            if cur.fetchone():
                cur.execute(f"DROP ROLE {_quote_ident(role)}")
    finally:
        conn.close()
