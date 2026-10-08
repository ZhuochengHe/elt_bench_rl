"""Test execution-derived model and loading reward semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eltbench.reward import (
    combine_scores,
    content_hash,
    is_null,
    normalize_for_value,
    parse_temporal,
    sort_rows,
    to_number,
    values_match,
)


def test_null_handling():
    assert is_null(None) and is_null("") and is_null("NULL")
    assert values_match(None, "")
    assert not values_match(None, "x")


def test_database_nan_matches_csv_null():
    # SQL NULL read through pandas/psycopg2 becomes NaN; GT CSV uses empty fields.
    assert is_null(float("nan"))
    assert values_match("", float("nan"))
    assert normalize_for_value("") == normalize_for_value(float("nan"))


def test_temporal_precision_equivalence():
    assert values_match("2021-04-01 19:42:58.123000", "2021-04-01 19:42:58.123")
    assert values_match("2021-12-10 05:51:07", "2021-12-10 05:51:07.000")
    assert values_match("2020-11-15", "2020-11-15 00:00:00")
    assert not values_match("2021-04-01 19:42:58.123", "2021-04-01 19:42:58.124")
    assert not values_match("2020-11-15", "2020-11-16")


def test_parse_temporal_rejects_numbers_and_non_dates():
    for numeric in ("123", "007", "1000", "1e3", 123, 1.5, True):
        assert parse_temporal(numeric) is None, numeric
    assert parse_temporal("ABBREVIATION") is None
    assert parse_temporal("") is None
    assert parse_temporal(None) is None

    assert parse_temporal("2020-11-15") is not None
    assert parse_temporal("2021-04-01 19:42:58.123") is not None


def test_numeric_strings_are_not_parsed_as_years():
    for a, b in [("007", "7"), ("1000", "1000"), ("123", "123")]:
        assert values_match(a, b), f"{a} and {b} should be equivalent"


def test_numeric_tolerance_is_symmetric_for_decimals():

    pairs = [
        (100.5, 99.6),
        (99.6, 100.5),
        (-100.5, -99.6),
        (100.25, 100.75),
        (100.9, 100.0),
        (10.5, 10.4),
    ]
    for a, b in pairs:
        assert values_match(a, b), f"{a} and {b} should be equivalent within 1%"

    for a, b in pairs:
        assert values_match(a, b) == values_match(b, a), (
            f"Comparison is asymmetric for {a} and {b}"
        )


def test_integers_get_only_a_tiny_relative_tolerance():

    for a, b in [(100, 99), (99, 100), (20062, 20074), (729, 731), (-100, -99)]:
        assert not values_match(a, b), f"{a} and {b} should not be equivalent"

    assert values_match("100", "100.0000001")
    assert values_match(100, 100.0000001)
    assert values_match(20062, "20062.000000001")

    assert not values_match("100", "100.1")
    assert not values_match("100", "100.5")

    assert values_match("6.364309636956422e+17", "636430963695642200")

    assert values_match(97.86, "97.86000000000001")
    assert values_match(0.1 + 0.2, 0.3)


def test_numeric_tolerance_rejects_big_diff():
    assert not values_match(100, 110)
    assert not values_match(0, 0.005)


def test_format_differences_are_not_errors():
    for a, b in [
        ("007", "7"),
        ("01234", "1234"),
        ("00", "0"),
        ("1e3", "1000"),
        ("1.50", "1.5"),
        ("100", "100.0"),
        ("6.36e+17", "636000000000000000"),
    ]:
        assert values_match(a, b), (
            f"{a} and {b} differ only in formatting and should be equivalent"
        )


def test_real_differences_are_still_caught():
    for a, b in [
        (100, 99),
        (100, 101),
        ("100", "101"),
        (729, 731),
        (20062, 20074),
        (100, 110),
        (0, 1),
    ]:
        assert not values_match(a, b), f"{a} and {b} should not be equivalent"


def test_to_number_uses_library_and_rejects_non_numeric():
    assert to_number("1e3") == 1000.0
    assert to_number("007") == 7.0
    assert to_number("6.364309636956422e+17") is not None
    for bad in ("107,7", "1,000", "abc", "inf", "nan", "", None):
        assert to_number(bad) is None, bad
    assert to_number(True) is None
    assert to_number("1.5") == 1.5


def test_string_comparison_is_case_and_space_insensitive():
    assert values_match("  Armed Forces  ", "ARMED FORCES")


def test_sort_rows_is_true_lexicographic():
    rows = [{"a": "1", "b": "zzz"}, {"a": "1", "b": "aaa"}]
    got = [(r["a"], r["b"]) for r in sort_rows(rows, ["a", "b"])]
    assert got == [("1", "aaa"), ("1", "zzz")]
    rev = [(r["a"], r["b"]) for r in sort_rows(list(reversed(rows)), ["a", "b"])]
    assert rev == got, "Physical row order must not affect the result"


def test_sort_rows_numeric_column_sorts_numerically():
    rows = [{"n": "10"}, {"n": "9"}, {"n": "100"}]
    assert [r["n"] for r in sort_rows(rows, ["n"])] == ["9", "10", "100"]


def test_sort_rows_handles_null_last_and_missing_keys():
    rows = [{"a": "1", "b": "2"}, {"a": None, "b": "1"}, {"a": "1", "b": "1"}]
    out = [(r["a"], r["b"]) for r in sort_rows(rows, ["a", "b"])]
    assert out[0] == ("1", "1")
    assert out[-1][0] is None


def test_sort_rows_key_case_insensitive():
    rows = [{"Abbreviation": "b"}, {"ABBREVIATION": "a"}]
    out = sort_rows(rows, ["abbreviation"])
    assert [list(r.values())[0] for r in out] == ["a", "b"]


def test_content_hash_row_order_invariant():
    a = [("1", "x"), ("2", "y")]
    assert content_hash(pd.DataFrame(a, columns=["c1", "c2"])) == content_hash(
        pd.DataFrame(list(reversed(a)), columns=["c1", "c2"])
    )


def test_content_hash_detects_real_differences():
    base = [("1", "x"), ("2", "y")]
    for bad in (base + [("1", "x")], [("1", "x"), ("2", "z")], [("1", "x")]):
        assert content_hash(pd.DataFrame(base, columns=["a", "b"])) != content_hash(
            pd.DataFrame(bad, columns=["a", "b"])
        ), bad


def test_normalize_for_hash_type_equivalence():
    assert normalize_for_value(1) == normalize_for_value("1")
    assert normalize_for_value(1.0) == normalize_for_value("1")
    assert normalize_for_value(1.5) == normalize_for_value("1.50")

    assert normalize_for_value(None) == normalize_for_value("")

    assert normalize_for_value("007") == normalize_for_value("7")

    assert normalize_for_value(40.817924) == normalize_for_value(40.81792449951172)


def test_combine_weights():
    total, gate = combine_scores(1.0, 1.0, load_evaluated=True)
    assert gate == 1.0

    assert total == pytest.approx(1.0 + 0.25)


def test_combine_skips_gate_when_load_not_evaluated():
    total, gate = combine_scores(1.0, 0.0, load_evaluated=False)
    assert gate == 1.0
    assert total == pytest.approx(1.0)


def test_combine_gate_closes_when_load_evaluated_but_bad():
    total, gate = combine_scores(1.0, 0.0, load_evaluated=True)
    assert gate == 0.0
    assert total == pytest.approx(0.0)


def test_combine_gate_is_continuous_and_monotonic():
    low, _ = combine_scores(1.0, 0.79, load_evaluated=True)
    high, _ = combine_scores(0.0, 0.80, load_evaluated=True)
    assert low > high, (
        f"All-correct models at load=0.79 should beat all-incorrect models at 0.80; got {low} vs {high}"
    )


def test_combine_monotonic_in_both_signals():
    for load in (0.0, 0.25, 0.5, 0.75, 1.0):
        totals = [
            combine_scores(m, load, load_evaluated=True)[0]
            for m in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert totals == sorted(totals), f"Model reward is not monotonic at load={load}"


def test_combine_gate_closed_below_threshold():
    _total, gate = combine_scores(1.0, 0.4, load_evaluated=True)
    assert gate == pytest.approx(0.5)


def test_normalize_sort_key_handles_real_benchmark_forms():
    from eltbench.reward import normalize_sort_key

    assert (
        normalize_sort_key("cast(community_area_no as integer)") == "community_area_no"
    )
    assert normalize_sort_key("CAST(label_id AS INT)") == "label_id"
    assert normalize_sort_key("video_id NULLS FIRST") == "video_id"
    assert normalize_sort_key("DATE_DAY NULLS FIRST") == "DATE_DAY"
    assert normalize_sort_key("abbreviation") == "abbreviation"

    assert normalize_sort_key("weird func(x)") is None
    assert normalize_sort_key(None) is None


def test_real_sort_key_expressions_all_normalize():
    import json
    from pathlib import Path

    from eltbench.reward import normalize_sort_key

    path = Path(__file__).resolve().parents[1] / "repo/evaluation/sort_key.json"
    if not path.is_file():
        pytest.skip("sort_key.json is missing")
    data = json.loads(path.read_text())
    unresolvable = [
        (db, tbl, k)
        for db, tables in data.items()
        for tbl, keys in tables.items()
        for k in keys
        if normalize_sort_key(k) is None
    ]
    assert not unresolvable, (
        f"Could not normalize {len(unresolvable)} keys: {unresolvable[:5]}"
    )


def test_real_sort_keys_mostly_exist_in_ground_truth():
    import csv
    import json
    from pathlib import Path

    from eltbench.reward import normalize_sort_key

    root = Path(__file__).resolve().parents[1]
    sk = root / "repo/evaluation/sort_key.json"
    gt = root / "repo/evaluation/gt"
    if not sk.is_file() or not gt.is_dir():
        pytest.skip("Benchmark data is unavailable")

    data = json.loads(sk.read_text())
    bogus = []
    for db, tables in data.items():
        for tbl, keys in tables.items():
            csvp = gt / db / f"{tbl}.csv"
            if not csvp.is_file():
                continue
            with csvp.open(newline="", encoding="utf-8-sig") as fh:
                cols = {c.lower() for c in (csv.DictReader(fh).fieldnames or [])}
            for k in keys:
                col = normalize_sort_key(k)
                if col and col.lower() not in cols:
                    bogus.append((db, tbl, k))
    assert len(bogus) <= 2, (
        f"Too many declared keys are missing from ground truth: {bogus[:5]}"
    )


def test_resolve_sort_keys_reports_misses():
    from eltbench.reward import resolve_sort_keys

    hit, missed = resolve_sort_keys(
        ["cast(a as integer)", "b", "nonexistent"], ["a", "b", "c"]
    )
    assert hit == ["a", "b"]
    assert missed == ["nonexistent"]


def test_driver_types_are_compared_semantically():
    import datetime as dt
    import decimal

    # datetime / timestamptz / date
    assert values_match(
        dt.datetime(2021, 4, 1, 19, 42, 58, 123000), "2021-04-01 19:42:58.123"
    )
    assert values_match(
        dt.datetime(2021, 4, 1, 19, 42, 58, tzinfo=dt.timezone.utc),
        "2021-04-01 19:42:58",
    )
    assert values_match(dt.date(2020, 11, 15), "2020-11-15 00:00:00")
    assert not values_match(
        dt.datetime(2021, 4, 1, 19, 42, 58, 123000), "2021-04-01 19:42:58.124"
    )

    assert values_match(decimal.Decimal("1234"), 1234)
    assert values_match(decimal.Decimal("1.50"), "1.5")
    assert to_number(decimal.Decimal("1234.5")) == 1234.5


def test_long_identifiers_are_matched_by_truncation():
    from eltbench.reward import PG_IDENT_LIMIT, truncate_identifier

    long_name = "HAS_A_PROFIT_GREATER_THAN_98_PER_OF_THE_AVERAGE_PROFIT_OF_ALL_PRODUCTS_IN_THE_EAST_REGION"
    assert len(long_name) == 89
    assert len(truncate_identifier(long_name)) == PG_IDENT_LIMIT
    assert truncate_identifier("short") == "short"

    assert truncate_identifier("é" * 40).encode("utf-8").decode("utf-8")


def test_sort_rows_with_fixed_columns_ignores_extra_columns():
    gt = [{"k": "1", "v": "b"}, {"k": "1", "v": "a"}]
    gen = [
        {"k": "1", "v": "a", "AAA_EXTRA": "x"},
        {"k": "1", "v": "b", "AAA_EXTRA": "y"},
    ]

    a = [(r["k"], r["v"]) for r in sort_rows(gt, ["k"], columns=["k", "v"])]
    b = [(r["k"], r["v"]) for r in sort_rows(gen, ["k"], columns=["k", "v"])]
    assert a == b == [("1", "a"), ("1", "b")]

    gen_rev = list(reversed(gen))
    b2 = [(r["k"], r["v"]) for r in sort_rows(gen_rev, ["k"], columns=["k", "v"])]
    assert b2 == b


def test_model_outcome_sorts_both_frames_with_shared_columns(tmp_path):
    gt = pd.DataFrame(
        {
            "k": ["1", "1", "2", "2"],
            "b": ["A", "B", "A", "B"],
            "c": ["2", "1", "4", "3"],
        }
    )
    gt = gt.iloc[[1, 0, 3, 2]].reset_index(drop=True)

    got = gt[["k", "c", "b"]].copy()
    out = _grade(gt, got, tmp_path, sort_keys=["k"])
    assert out.ok is True, (
        f"Column order changed the result: missing={out.missing} wrong={out.wrong}"
    )
    assert out.partial == pytest.approx(1.0)

    got_extra = gt[["k", "c", "b"]].copy()
    got_extra["AAA_EXTRA"] = ["x", "y", "z", "w"]
    out2 = _grade(gt, got_extra, tmp_path, sort_keys=["k"])
    assert out2.ok is True, f"Extra columns changed the result: wrong={out2.wrong}"
    assert out2.partial == pytest.approx(1.0)


def test_content_hash_detects_column_rename():
    base = [("1", "x")]
    renamed = [("1", "x")]
    assert content_hash(
        pd.DataFrame(base, columns=["zip_code", "county"])
    ) != content_hash(pd.DataFrame(renamed, columns=["zzz", "yyy"]))

    assert content_hash(
        pd.DataFrame(base, columns=["ZIP_CODE", "COUNTY"])
    ) == content_hash(pd.DataFrame(base, columns=["zip_code", "county"]))


def test_normalize_for_hash_handles_neg_zero_and_huge_numbers():
    assert normalize_for_value(-0.0) == normalize_for_value(0)
    big = "1" + "0" * 400
    assert normalize_for_value(big) != normalize_for_value(big[:-1])


class _FrameConnector:
    def __init__(self, tables: dict):
        self.tables = tables

    def fetch_table(self, database, schema, table):  # noqa: ARG002
        if table not in self.tables:
            raise KeyError(table)
        return self.tables[table].copy()


def _grade(gt: pd.DataFrame, got: pd.DataFrame | None, tmp_path: Path, sort_keys=None):
    from eltbench.reward import model_outcome

    gt_path = tmp_path / "m.csv"
    gt.to_csv(gt_path, index=False)
    tables = {} if got is None else {"m": got}
    return model_outcome(
        _FrameConnector(tables), "db", "schema", "m", gt_path, sort_keys or []
    )


GT3 = pd.DataFrame({"k": ["1", "2", "3"], "a": ["x", "y", "z"], "b": ["1", "2", "3"]})


def test_partial_is_one_when_model_is_perfect(tmp_path):
    out = _grade(GT3, GT3, tmp_path)
    assert out.ok and out.partial == pytest.approx(1.0)
    assert out.row_score == pytest.approx(1.0)
    assert out.column_score == pytest.approx(1.0)


def test_partial_gives_credit_for_matched_columns(tmp_path):
    got = GT3.copy()
    got["a"] = ["WRONG", "WRONG", "WRONG"]
    out = _grade(GT3, got, tmp_path)
    assert out.ok is False
    assert out.columns_matched == 2 and out.columns_total == 3
    assert out.partial == pytest.approx(2 / 3)


def test_partial_rewards_row_count_proximity(tmp_path):
    got = GT3.iloc[:2].copy()
    out = _grade(GT3, got, tmp_path)
    assert out.ok is False
    assert out.row_score == pytest.approx(2 / 3)
    assert out.column_score == pytest.approx(1.0)
    assert out.partial == pytest.approx(2 / 3)


def test_partial_is_zero_for_missing_or_empty_table(tmp_path):
    for got in (None, GT3.iloc[0:0].copy()):
        out = _grade(GT3, got, tmp_path)
        assert out.ok is False and out.partial == 0.0


def test_partial_does_not_pay_for_constant_filler_columns(tmp_path):
    got = GT3.copy()
    got["a"] = ["0", "0", "0"]
    out = _grade(GT3, got, tmp_path)
    assert "a" in out.wrong
    assert out.columns_matched == 2
    assert out.partial == pytest.approx(2 / 3)


def test_model_partial_score_averages_models(tmp_path):
    from eltbench.reward import model_partial_score

    good = _grade(GT3, GT3, tmp_path)
    bad = _grade(GT3, GT3.iloc[0:0].copy(), tmp_path)
    assert model_partial_score([good, bad]) == pytest.approx(0.5)
    assert model_partial_score([]) == 0.0


def test_partial_never_exceeds_srdt(tmp_path):
    cases = [
        GT3,
        GT3.iloc[:2].copy(),
        GT3.assign(a="WRONG"),
        GT3.assign(k=["1", "1", "1"]),
    ]
    for got in cases:
        out = _grade(GT3, got, tmp_path)
        assert 0.0 <= out.partial <= 1.0
        assert out.partial >= float(out.ok) - 1e-12


class _DualConnector:
    def __init__(self, source: pd.DataFrame, target: pd.DataFrame):
        self.source, self.target = source, target

    def fetch_table(self, database, schema, table):  # noqa: ARG002
        return (self.source if database == "src" else self.target).copy()


def test_load_content_mode_catches_corrupted_copy():
    from eltbench.reward import load_score

    src = pd.DataFrame({"id": ["1", "2"], "v": ["a", "b"]})
    corrupt = pd.DataFrame({"id": ["1", "2"], "v": ["a", "ZZZ"]})
    conn = _DualConnector(src, corrupt)
    manifest = {"t": 2}

    effective, count, content, out = load_score(
        conn, "src", "public", "tgt", "public", ["t"], manifest, want_content=True
    )
    assert content == 0.0 and count == 1.0 and effective == 0.0
    assert out[0].ok is False and out[0].count_ok is True and out[0].content_ok is False

    effective2, count2, content2, out2 = load_score(
        conn, "src", "public", "tgt", "public", ["t"], manifest, want_content=False
    )
    assert content2 is None
    assert count2 == 1.0 and effective2 == 1.0
    assert out2[0].ok is True


def test_load_content_mode_equals_self_comparison_for_cloud_layout():
    from eltbench.reward import load_score

    same = pd.DataFrame({"id": ["1", "2"], "v": ["a", "b"]})

    class _SelfConnector:
        def fetch_table(self, database, schema, table):  # noqa: ARG002
            return same.copy()

    _eff, count, content, _out = load_score(
        _SelfConnector(),
        "address_ab12_0",
        "AIRBYTE_SCHEMA",
        "address_ab12_0",
        "AIRBYTE_SCHEMA",
        ["t"],
        {"t": 2},
        want_content=True,
    )
    assert content == 1.0 and count == 1.0


def test_source_snapshot_freezes_hashes_against_later_mutation():
    from eltbench.reward import load_score, source_snapshot

    src = pd.DataFrame({"id": ["1", "2"], "v": ["a", "b"]})
    target_ok = src.copy()

    class _MutatingSourceConnector:
        def __init__(self):
            self.source = src.copy()

        def fetch_table(self, database, schema, table):  # noqa: ARG002
            return (self.source if database == "snap_src" else target_ok).copy()

    conn = _MutatingSourceConnector()
    snapshot = source_snapshot(conn, "snap_src", "public", ["t"])
    conn.source = src.iloc[0:0]

    _eff, _count, content, out = load_score(
        conn,
        "snap_src",
        "public",
        "tgt",
        "public",
        ["t"],
        {"t": 2},
        want_content=True,
        source_hashes=snapshot,
    )
    assert content == 1.0 and out[0].ok is True

    _eff2, _c2, content2, _o2 = load_score(
        conn,
        "snap_src",
        "public",
        "tgt",
        "public",
        ["t"],
        {"t": 2},
        want_content=True,
        source_hashes=None,
    )
    assert content2 == 0.0


def test_snapshot_missing_table_is_unverified_and_never_read_live():
    from eltbench.reward import load_score, source_snapshot

    good = pd.DataFrame({"id": ["1", "2"], "v": ["a", "b"]})
    tampered = pd.DataFrame({"id": ["1", "2"], "v": ["a", "ZZZ"]})

    class _Connector:
        def __init__(self):
            self.snapshotting = True
            self.live_source_reads: list[str] = []

        def fetch_table(self, database, schema, table):  # noqa: ARG002
            if database == "snap_miss_src":
                if self.snapshotting:
                    if table == "t2":
                        raise RuntimeError("Simulated read failure during snapshot")
                    return good.copy()
                self.live_source_reads.append(table)
                return (tampered if table == "t2" else good).copy()
            return good.copy()

    conn = _Connector()
    snapshot = source_snapshot(conn, "snap_miss_src", "public", ["t1", "t2"])

    assert snapshot == {"t1": content_hash(good)}
    assert snapshot.unhashed == frozenset({"t2"})

    conn.snapshotting = False
    _eff, count, content, out = load_score(
        conn,
        "snap_miss_src",
        "public",
        "tgt",
        "public",
        ["t1", "t2"],
        {"t1": 2, "t2": 2},
        want_content=True,
        source_hashes=snapshot,
    )
    assert conn.live_source_reads == []
    by_table = {o.table: o for o in out}
    assert by_table["t2"].verified is False
    assert by_table["t2"].ok is False and by_table["t2"].content_ok is False
    assert by_table["t1"].verified is True and by_table["t1"].ok is True
    assert count == 1.0
    assert content == 1.0

    conn.live_source_reads.clear()
    _eff2, _count2, content2, _out2 = load_score(
        conn,
        "snap_miss_src",
        "public",
        "tgt",
        "public",
        ["t1", "t2"],
        {"t1": 2, "t2": 2},
        want_content=True,
        source_hashes=None,
    )
    assert conn.live_source_reads == ["t1", "t2"]
    assert content2 == 0.5


def test_snapshot_with_no_verified_table_scores_zero():
    from eltbench.reward import load_score, source_snapshot

    class _UnreadableSource:
        def fetch_table(self, database, schema, table):  # noqa: ARG002
            if database == "snap_all_unreadable_src":
                raise RuntimeError("Source database is unavailable")
            return pd.DataFrame({"id": ["1", "2"], "v": ["a", "b"]})

    snapshot = source_snapshot(
        _UnreadableSource(), "snap_all_unreadable_src", "public", ["t"]
    )
    assert snapshot == {} and snapshot.unhashed == frozenset({"t"})
    _eff, count, content, out = load_score(
        _UnreadableSource(),
        "snap_all_unreadable_src",
        "public",
        "tgt",
        "public",
        ["t"],
        {"t": 2},
        want_content=True,
        source_hashes=snapshot,
    )
    assert count == 1.0
    assert content == 0.0 and _eff == 0.0


def test_combine_uses_dense_model_term():
    perfect, _ = combine_scores(1.0, 0.0, load_evaluated=False)
    partial, _ = combine_scores(0.5, 0.0, load_evaluated=False)
    zero, _ = combine_scores(0.0, 0.0, load_evaluated=False)
    assert perfect > partial > zero
