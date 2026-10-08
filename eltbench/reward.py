"Compare warehouse state with benchmark outputs and compute rewards."

from __future__ import annotations

import datetime as _dt
import decimal
import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from . import benchmark

NA: frozenset[str] = frozenset({"", "NULL", "null", "None", "NaN", "nan", "N/A", "NA"})


RTOL = 1e-2
ATOL = 1e-9


INT_RTOL = 1e-8
INT_ATOL = 1e-9


HASH_DECIMALS = 6


PG_IDENT_LIMIT = 63


LOAD_GATE = 0.8
LOAD_WEIGHT = 0.25


def is_null(v: Any) -> bool:
    if v is None or (isinstance(v, str) and v in NA):
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def as_text(v: Any) -> str:
    return "" if v is None else str(v)


#:

#:   - `format="ISO8601"`：`"007"` / `"1e3"` / `"ABBREVIATION"` / `"123"` / `"1.5"`


def parse_temporal(v: Any):
    if isinstance(v, (int, float, decimal.Decimal, bool)):
        return None
    if not isinstance(v, (str, _dt.datetime, _dt.date, pd.Timestamp)):
        return None
    if isinstance(v, str) and not any(ch in v for ch in "-:/T "):
        return None
    try:
        ts = pd.to_datetime(v, errors="coerce", format="ISO8601", utc=True)
    except Exception:  # noqa: BLE001
        return None
    return None if (ts is None or pd.isna(ts)) else ts


def to_number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        n = pd.to_numeric(v, errors="coerce")
    except Exception:  # noqa: BLE001
        return None
    if n is None or pd.isna(n):
        return None
    f = float(n)
    return None if (math.isinf(f) or math.isnan(f)) else f


def is_integral_value(v: Any) -> bool:
    if v is None or isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return False
    if isinstance(v, decimal.Decimal):
        return v == v.to_integral_value()
    if isinstance(v, str):
        return bool(re.fullmatch(r"[+-]?\d+", v.strip()))
    return False


def values_match(a: Any, b: Any) -> bool:
    if is_null(a) and is_null(b):
        return True

    na, nb = to_number(a), to_number(b)
    if na is not None or nb is not None:
        if na is None or nb is None:
            return False
        if na == nb:
            return True
        if is_integral_value(a) or is_integral_value(b):
            return abs(na - nb) <= INT_ATOL + INT_RTOL * max(abs(na), abs(nb))
        return abs(na - nb) <= ATOL + RTOL * max(abs(na), abs(nb))

    ta, tb = parse_temporal(a), parse_temporal(b)
    if ta is not None or tb is not None:
        return ta is not None and tb is not None and ta == tb

    if is_null(a) or is_null(b):
        return False

    if ta is None and tb is None and (not isinstance(a, str) or not isinstance(b, str)):
        return as_text(a).strip().lower() == as_text(b).strip().lower()
    return as_text(a).strip().lower() == as_text(b).strip().lower()


def normalize_for_value(v: Any) -> str:
    if is_null(v):
        return "\x1e"
    n = to_number(v)
    if n is not None and math.isinf(n):
        return as_text(v).strip()
    if n is not None:
        if n == 0:
            n = 0.0
        s = f"{n:.{HASH_DECIMALS}f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return as_text(v).strip()


def sort_key_series(values: list[Any]) -> list:
    nums = [to_number(v) for v in values]
    non_null = [v for v in values if not is_null(v)]
    all_numeric = bool(non_null) and all(
        n is not None for v, n in zip(values, nums) if not is_null(v)
    )
    if all_numeric:
        return [(1, 0.0, "") if n is None else (0, n, "") for n in nums]
    return [
        (1, 0.0, "") if is_null(v) else (0, 0.0, as_text(v).strip().lower())
        for v in values
    ]


def normalize_sort_key(key: str) -> str | None:
    if not isinstance(key, str):
        return None
    s = key.strip()
    m = re.match(
        r"^\s*cast\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+as\s+[^)]*\)\s*$", s, re.I
    )
    if m:
        return m.group(1)
    s = re.sub(r"\s+(nulls\s+(first|last)|asc|desc)\s*$", "", s, flags=re.I)
    s = s.strip().strip('"').strip("`")
    return s if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", s) else None


def resolve_sort_keys(
    keys: list[str], columns: list[str]
) -> tuple[list[str], list[str]]:
    lower = {c.lower(): c for c in columns}
    ok, bad = [], []
    for k in keys:
        name = normalize_sort_key(k)
        if name is None:
            bad.append(k)
        elif name.lower() in lower:
            ok.append(lower[name.lower()])
        else:
            bad.append(k)
    return ok, bad


def sort_rows(
    rows: list[dict], keys: list[str], columns: list[str] | None = None
) -> list[dict]:
    if not rows:
        return []
    cols = (
        list(columns)
        if columns is not None
        else sorted({str(k) for r in rows for k in r})
    )
    lower = {str(c).lower(): c for c in cols}

    resolved = [lower[k.lower()] for k in keys if k.lower() in lower]
    order = resolved + [c for c in cols if c not in resolved]
    if not order:
        return list(rows)

    views = [{str(k).lower(): v for k, v in r.items()} for r in rows]

    def value_at(i: int, col: str):
        view = views[i]
        if col in view:
            return view[col]
        return view.get(str(col).lower())

    col_keys = [
        sort_key_series([value_at(i, c) for i in range(len(rows))]) for c in order
    ]
    decorated = [
        (tuple(col_keys[j][i] for j in range(len(order))), i, rows[i])
        for i in range(len(rows))
    ]
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [r for _k, _i, r in decorated]


def truncate_identifier(name: str, limit: int = PG_IDENT_LIMIT) -> str:
    raw = str(name).encode("utf-8")
    if len(raw) <= limit:
        return str(name)
    return raw[:limit].decode("utf-8", "ignore")


def content_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"").hexdigest()
    pairs = sorted((str(c).strip().lower(), c) for c in df.columns)
    names = [n for n, _ in pairs]
    cols = [c for _, c in pairs]
    lines = []
    for row in df[cols].itertuples(index=False, name=None):
        lines.append(
            "\x1f".join(f"{n}={normalize_for_value(v)}" for n, v in zip(names, row))
        )
    lines.sort()
    return hashlib.sha256("\x1f\x1f".join(lines).encode("utf-8", "replace")).hexdigest()


@dataclass
class ModelOutcome:
    model: str
    ok: bool
    missing: list[str] = field(default_factory=list)
    wrong: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    gt_rows: int = 0
    got_rows: int = 0

    columns_total: int = 0

    columns_matched: int = 0

    row_score: float = 0.0

    @property
    def column_score(self) -> float:
        if self.columns_total <= 0:
            return 0.0
        return self.columns_matched / self.columns_total

    @property
    def partial(self) -> float:
        return self.row_score * self.column_score


@dataclass
class TableOutcome:
    table: str

    ok: bool
    note: str = ""

    expected_rows: int | None = None
    got_rows: int = 0
    count_ok: bool = False

    content_ok: bool = False

    verified: bool = True


@dataclass
class RewardBreakdown:
    model_score: float = 0.0

    model_partial: float = 0.0

    load_score: float = 0.0
    gate: float = 1.0
    total: float = 0.0

    load_evaluated: bool = False

    load_count_score: float = 0.0

    load_content_score: float | None = None

    load_mode: str = "count"
    models: list[ModelOutcome] = field(default_factory=list)
    tables: list[TableOutcome] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:

        out = {
            "reward/episode_total": self.total,
            "reward/model_srdt": self.model_score,
            "reward/model_partial": self.model_partial,
            "reward/gate": self.gate,
            "reward/load_evaluated": float(self.load_evaluated),
        }
        if self.load_evaluated:
            out["reward/load"] = self.load_score
            out["reward/load_count"] = self.load_count_score
            if self.load_content_score is not None:
                out["reward/load_content"] = self.load_content_score
            out["reward/load_mode_content"] = float(self.load_mode == "content")

            out["reward/load_unverified"] = float(
                sum(1 for t in self.tables if not t.verified)
            )
        return out

    def summary(self) -> str:
        bad_m = [m.model for m in self.models if not m.ok]
        bad_t = [t.table for t in self.tables if not t.ok]
        s = (
            f"srdt={self.model_score:.3f} partial={self.model_partial:.3f} "
            f"load[{self.load_mode}]={self.load_score:.3f} "
            f"gate={self.gate:.2f} -> {self.total:.4f}"
        )
        if bad_m:
            s += f" | incorrect models={bad_m[:4]}"
        if bad_t:
            s += f" | incorrect tables ({len(bad_t)})={bad_t[:4]}"
        unverified = sum(1 for t in self.tables if not t.verified)
        if unverified:
            s += f" | unverified tables ({unverified})"
        return s


@lru_cache(maxsize=256)
def _read_gt_cached(path_str: str, mtime: float, size: int) -> pd.DataFrame:
    return pd.read_csv(path_str, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def read_gt(path: Path) -> pd.DataFrame:
    path = Path(path)
    stat = path.stat()
    return _read_gt_cached(str(path), stat.st_mtime, stat.st_size)


def load_sort_keys(task: str, repo_root: Path | None = None) -> dict[str, list[str]]:
    return benchmark.task_sort_keys(task)


def _norm_series(s: pd.Series) -> pd.Series:
    return s.map(normalize_for_value)


def _shared_sort_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    la = {str(c).lower() for c in a.columns}
    lb = {str(c).lower() for c in b.columns}
    return sorted(la & lb)


def _sort_frame(
    df: pd.DataFrame, keys: list[str], columns: list[str] | None = None
) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    records = df.to_dict("records")
    cols = (
        [str(c) for c in df.columns] if columns is None else [str(c) for c in columns]
    )
    ordered = sort_rows(records, keys, columns=cols)
    if not ordered:
        return df.reset_index(drop=True)
    return pd.DataFrame(ordered, columns=list(df.columns))


def load_score(
    connector,
    source_db: str,
    source_schema: str,
    target_db: str,
    target_schema: str,
    tables: list[str],
    manifest: dict[str, int] | None = None,
    *,
    want_content: bool = True,
    source_hashes: dict[str, str] | None = None,
) -> tuple[float, float, float | None, list[TableOutcome]]:
    manifest = manifest or {}
    out: list[TableOutcome] = []
    for t in tables:
        expected = manifest.get(t)
        try:
            got_df = connector.fetch_table(target_db, target_schema, t)
        except Exception as exc:  # noqa: BLE001
            out.append(
                TableOutcome(
                    t,
                    False,
                    f"Target read failed: {type(exc).__name__}",
                    expected_rows=expected,
                )
            )
            continue
        got_rows = 0 if got_df is None else len(got_df)
        count_ok = expected is not None and got_rows == expected
        if got_df is None or got_df.empty:
            out.append(
                TableOutcome(
                    t,
                    False,
                    "Target is empty",
                    expected_rows=expected,
                    got_rows=got_rows,
                    count_ok=count_ok,
                )
            )
            continue

        content_ok: bool | None = None
        if want_content:
            if source_hashes is not None and t not in source_hashes:
                #

                out.append(
                    TableOutcome(
                        t,
                        False,
                        "Source snapshot missing; content was not verified",
                        expected_rows=expected,
                        got_rows=got_rows,
                        count_ok=count_ok,
                        content_ok=False,
                        verified=False,
                    )
                )
                continue
            try:
                if source_hashes is not None:
                    want = source_hashes[t]
                else:
                    want = content_hash(
                        connector.fetch_table(source_db, source_schema, t)
                    )
                content_ok = want == content_hash(got_df)
            except Exception as exc:  # noqa: BLE001
                out.append(
                    TableOutcome(
                        t,
                        False,
                        f"Source read failed: {type(exc).__name__}",
                        expected_rows=expected,
                        got_rows=got_rows,
                        count_ok=count_ok,
                    )
                )
                continue

        ok = content_ok if content_ok is not None else count_ok
        note = (
            ""
            if ok
            else (
                "Content mismatch" if content_ok is not None else "Row-count mismatch"
            )
        )
        out.append(
            TableOutcome(
                t,
                ok,
                note,
                expected_rows=expected,
                got_rows=got_rows,
                count_ok=count_ok,
                content_ok=bool(content_ok),
            )
        )
    n = len(out)
    count_score = (sum(1 for o in out if o.count_ok) / n) if n else 0.0

    if want_content and n:
        verified = [o for o in out if o.verified]
        content_score = (
            sum(1 for o in verified if o.content_ok) / len(verified)
            if verified
            else 0.0
        )
    else:
        content_score = None
    effective = content_score if content_score is not None else count_score
    return effective, count_score, content_score, out


def model_outcome(
    connector,
    target_db: str,
    target_schema: str,
    model: str,
    gt_path: Path,
    sort_keys: list[str] | None = None,
) -> ModelOutcome:
    try:
        want = read_gt(gt_path)
    except Exception as exc:  # noqa: BLE001
        return ModelOutcome(
            model, False, missing=[f"Failed to read ground truth: {exc}"]
        )

    try:
        got = connector.fetch_table(target_db, target_schema, model)
    except Exception:  # noqa: BLE001
        return ModelOutcome(model, False, missing=list(want.columns), gt_rows=len(want))
    if got.empty:
        return ModelOutcome(model, False, missing=list(want.columns), gt_rows=len(want))

    keys = [k for k in (normalize_sort_key(x) for x in (sort_keys or [])) if k]

    shared_cols = _shared_sort_columns(want, got)
    want_s, got_s = (
        _sort_frame(want, keys, shared_cols),
        _sort_frame(got, keys, shared_cols),
    )

    overlap = min(len(want_s), len(got_s))
    if len(want_s) == 0:
        row_score = 1.0 if len(got_s) == 0 else 0.0
    else:
        row_score = overlap / max(len(want_s), len(got_s))

    got_lower = {str(c).lower(): c for c in got_s.columns}
    missing, wrong = [], []
    columns_matched = 0
    for gold in want.columns:
        actual = got_lower.get(str(gold).lower())
        if actual is None:
            actual = got_lower.get(str(gold)[:PG_IDENT_LIMIT].lower())
        if actual is None:
            missing.append(gold)
            continue
        w = _norm_series(want_s[gold]).tolist()
        g = _norm_series(got_s[actual]).tolist()

        matches = [values_match(x, y) for x, y in zip(w[:overlap], g[:overlap])]
        prefix_ok = bool(matches) and all(matches)
        if prefix_ok:
            columns_matched += 1
        if not (prefix_ok and len(w) == len(g)):
            wrong.append(gold)

    want_cols_lower = {str(g).lower() for g in want.columns}
    extra = [c for c in got.columns if str(c).lower() not in want_cols_lower]
    return ModelOutcome(
        model,
        not missing and not wrong,
        missing=missing,
        wrong=wrong,
        extra=list(extra),
        gt_rows=len(want),
        got_rows=len(got),
        columns_total=len(want.columns),
        columns_matched=columns_matched,
        row_score=row_score,
    )


def model_score(
    connector,
    target_db: str,
    target_schema: str,
    gt_dir: Path,
    sort_key_map: dict[str, list[str]] | None = None,
) -> tuple[float, list[ModelOutcome]]:
    gt_dir = Path(gt_dir)
    if not gt_dir.is_dir():
        return 0.0, []
    out = [
        model_outcome(
            connector,
            target_db,
            target_schema,
            p.stem,
            p,
            (sort_key_map or {}).get(p.stem, []),
        )
        for p in sorted(gt_dir.glob("*.csv"))
    ]
    return ((sum(1 for o in out if o.ok) / len(out)) if out else 0.0), out


def model_partial_score(outcomes: list[ModelOutcome]) -> float:
    if not outcomes:
        return 0.0
    return sum(o.partial for o in outcomes) / len(outcomes)


def combine_scores(
    model_term: float,
    load_score: float,
    *,
    load_evaluated: bool,
    load_gate: float = LOAD_GATE,
    load_weight: float = LOAD_WEIGHT,
) -> tuple[float, float]:
    gate = (
        min(1.0, load_score / load_gate) if (load_evaluated and load_gate > 0) else 1.0
    )
    return model_term * gate + load_weight * load_score, gate


def compute_reward(
    connector,
    task: str,
    gt_dir: Path,
    target_db: str,
    target_schema: str,
    *,
    source_db: str | None = None,
    source_schema: str = "public",
    load_tables: list[str] | None = None,
    sort_key_map: dict[str, list[str]] | None = None,
    table_manifest: dict[str, int] | None = None,
    load_mode: str = "count",
    source_hashes: dict[str, str] | None = None,
    load_gate: float = LOAD_GATE,
    load_weight: float = LOAD_WEIGHT,
) -> RewardBreakdown:
    b = RewardBreakdown(load_mode="content" if load_mode == "content" else "count")
    if sort_key_map is None:
        sort_key_map = benchmark.task_sort_keys(task)
    if table_manifest is None:
        table_manifest = benchmark.source_table_manifest(task)
    b.model_score, b.models = model_score(
        connector, target_db, target_schema, gt_dir, sort_key_map
    )
    b.model_partial = model_partial_score(b.models)

    tables = list(
        load_tables if load_tables is not None else benchmark.source_tables(task)
    )
    if tables and source_db:
        want_content = b.load_mode == "content"
        b.load_score, b.load_count_score, b.load_content_score, b.tables = load_score(
            connector,
            source_db,
            source_schema,
            target_db,
            target_schema,
            tables,
            table_manifest,
            want_content=want_content,
            source_hashes=source_hashes,
        )
        b.load_evaluated = True

    b.total, b.gate = combine_scores(
        b.model_partial,
        b.load_score,
        load_evaluated=b.load_evaluated,
        load_gate=load_gate,
        load_weight=load_weight,
    )
    return b


_SOURCE_SNAPSHOT: dict[tuple[str, str, str], str] = {}
_SOURCE_SNAPSHOT_LOCK = threading.Lock()


class SourceSnapshot(dict):
    __slots__ = ("unhashed",)

    def __init__(self, *args, unhashed: frozenset[str] = frozenset(), **kwargs):
        super().__init__(*args, **kwargs)

        self.unhashed: frozenset[str] = unhashed


def source_snapshot(
    connector, source_db: str, source_schema: str, tables: list[str]
) -> SourceSnapshot:
    snapshot: dict[str, str] = {}
    missing = []
    with _SOURCE_SNAPSHOT_LOCK:
        for t in tables:
            key = (source_db, source_schema, t)
            if key in _SOURCE_SNAPSHOT:
                snapshot[t] = _SOURCE_SNAPSHOT[key]
            else:
                missing.append(t)
    unhashed: set[str] = set()
    for t in missing:
        try:
            digest = content_hash(connector.fetch_table(source_db, source_schema, t))
        except Exception:  # noqa: BLE001
            unhashed.add(t)
            continue
        with _SOURCE_SNAPSHOT_LOCK:
            _SOURCE_SNAPSHOT[(source_db, source_schema, t)] = digest
        snapshot[t] = digest
    return SourceSnapshot(snapshot, unhashed=frozenset(unhashed))
