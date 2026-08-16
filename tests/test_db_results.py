"""User-facing SQL result rows, columns, and cursor proxies."""

import copy
import os
import pickle
from dataclasses import dataclass
from fractions import Fraction

import pytest
import sqlalchemy
from clickhouse_read_support import CLICKHOUSE_PARAM, bulk_store
from postgres_support import POSTGRES_PARAM, postgres_database

from httk.store.db import (
    Database,
    ExpiredCursorRowError,
    MultipleResultsError,
    NoResultError,
    SqlStore,
)

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")


@dataclass(frozen=True)
class ResultRecord:
    name: str
    energy: Fraction


@dataclass(frozen=True)
class EnergyRecord:
    energy: Fraction


@dataclass(frozen=True)
class ParentRecord:
    child: EnergyRecord


@dataclass(frozen=True)
class ChildListRecord:
    values: list[int]


@dataclass(frozen=True)
class IntRecord:
    value: int


@pytest.fixture(scope="module", params=("sqlite", CLICKHOUSE_PARAM, POSTGRES_PARAM))
def store(request):
    records = tuple(ResultRecord(str(index), Fraction(index, 3)) for index in range(6))
    if request.param == "clickhousedb":
        with bulk_store(records) as value:
            yield value
        return
    if request.param == "postgresql":
        with postgres_database() as database:
            value = SqlStore(database, entry_records={})
            for record in records:
                value.save(record)
            yield value
        return
    with Database.sqlite() as database:
        value = SqlStore(database, entry_records={})
        for record in records:
            value.save(record)
        yield value


def _search(store: SqlStore):
    searcher = store.searcher()
    record = searcher.variable(ResultRecord)
    return searcher, record


def _profile_rows() -> int:
    return 100_000 if os.environ.get("HTTK_TEST_PROFILE", "").lower() == "extended" else 2_000


def test_result_rows_are_frozen_indexable_and_reiterable(store):
    searcher, record = _search(store)
    results = searcher.results(structure=record, energy=record.energy)
    searcher.add(record.name == "missing")
    assert len(results) == 6
    assert results[1].structure.name == "1"
    assert results[1]["energy"] == Fraction(1, 3)
    assert results[1][1] == Fraction(1, 3)
    assert [row.structure.name for row in results[1:3]] == ["1", "2"]
    assert [row.structure.name for row in results] == [str(index) for index in range(6)]
    assert [row.structure.name for row in results] == [str(index) for index in range(6)]
    assert [row.name for row in results.scalars("structure")] == [str(index) for index in range(6)]


def test_slices_keep_their_own_scope_for_rows_columns_and_one(store):
    searcher, record = _search(store)
    results = searcher.results(structure=record, energy=record.energy)
    view = results[1:5]
    nested = view[1:3]
    assert len(view) == 4
    assert nested[0].structure.name == "2"
    assert nested.first().structure.name == "2"
    assert [row.structure.name for row in nested] == ["2", "3"]
    assert list(view.column("energy")) == [Fraction(index, 3) for index in range(1, 5)]
    assert view[0:1].one().structure.name == "1"


def test_reference_exact_projection_is_pinned_in_outer_statement():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        child = EnergyRecord(Fraction(7, 11))
        store.save(child)
        store.save(ParentRecord(child))
        searcher = store.searcher()
        parent = searcher.variable(ParentRecord)
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            results = searcher.results(energy=parent.child.energy)
            assert list(results.column("energy")) == [Fraction(7, 11)]
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)
        assert len(statements) == 1


def test_child_projection_is_rejected_at_declaration():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        searcher = store.searcher()
        record = searcher.variable(ChildListRecord)
        with pytest.raises(TypeError, match="variable-length child"):
            searcher.results(values=record.values)


def test_integer_columns_are_rational_vectors():
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store.save(IntRecord(1))
        searcher = store.searcher()
        record = searcher.variable(IntRecord)
        assert searcher.results(value=record.value).column("value").to_fracvector().to_fractions() == [Fraction(1)]


def test_scalar_and_one_queries(store):
    searcher, record = _search(store)
    results = searcher.results(energy=record.energy)
    assert list(results.scalars()) == [Fraction(index, 3) for index in range(6)]
    assert list(results.scalars("energy")) == list(results.column("energy"))
    assert results.first().energy == 0
    with pytest.raises(MultipleResultsError, match="more than one"):
        results.one()

    empty_searcher, empty_record = _search(store)
    empty_searcher.add(empty_record.name == "missing")
    empty = empty_searcher.results(energy=empty_record.energy)
    assert empty.first() is None
    with pytest.raises(NoResultError, match="none"):
        empty.one()


def test_columns_are_exact_and_float_views_are_explicit(store):
    searcher, record = _search(store)
    results = searcher.results(energy=record.energy)
    column = results.column("energy")
    assert list(column) == [Fraction(index, 3) for index in range(6)]
    assert list(column.floats()) == [float(Fraction(index, 3)) for index in range(6)]
    assert column.to_fracvector().to_fractions() == [Fraction(index, 3) for index in range(6)]


def test_cursor_proxy_is_a_reused_unhashable_expiring_view(store):
    searcher, record = _search(store)
    results = searcher.results(structure=record, energy=record.energy)
    cursor = results.cursor()
    first = next(cursor)
    proxy = first.structure
    assert isinstance(proxy, ResultRecord)
    assert first.energy == 0
    with pytest.raises(TypeError):
        hash(proxy)
    with pytest.raises(TypeError):
        copy.copy(proxy)
    with pytest.raises(TypeError):
        pickle.dumps(proxy)
    second = next(cursor)
    with pytest.raises(ExpiredCursorRowError):
        _ = proxy.name
    assert second.structure is proxy
    with pytest.raises(ExpiredCursorRowError):
        _ = first.structure
    with pytest.raises(ExpiredCursorRowError):
        _ = first.energy
    with pytest.raises(ExpiredCursorRowError):
        _ = first.names
    with pytest.raises(ExpiredCursorRowError):
        len(first)
    with pytest.raises(TypeError):
        copy.copy(first)
    with pytest.raises(TypeError):
        pickle.dumps(first)


def test_column_iteration_does_not_hydrate_records(store, monkeypatch):
    import httk.store.db.results as results_module

    calls = 0
    original = results_module.RowHydrator.row

    def count(self, sid):
        nonlocal calls
        calls += 1
        return original(self, sid)

    monkeypatch.setattr(results_module.RowHydrator, "row", count)
    searcher, record = _search(store)
    assert list(searcher.results(energy=record.energy).column("energy"))
    assert calls == 0


def test_perf_smoke_statement_bound_and_cursor_proxy_count():
    rows = _profile_rows()
    with Database.sqlite() as database:
        value = SqlStore(database, entry_records={})
        with value.transaction():
            for index in range(rows):
                value.save(ResultRecord(str(index), Fraction(index, 7)))
        searcher = value.searcher()
        record = searcher.variable(ResultRecord)
        results = searcher.results(record=record, energy=record.energy)
        statements: list[str] = []

        def count_select(_conn, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        sqlalchemy.event.listen(database.engine, "before_cursor_execute", count_select)
        try:
            assert sum(row.energy.numerator for row in results) == sum(
                Fraction(index, 7).numerator for index in range(rows)
            )
        finally:
            sqlalchemy.event.remove(database.engine, "before_cursor_execute", count_select)
        assert len(statements) <= 1 + 2 * ((rows + 499) // 500) + 2

        cursor = results.cursor()
        proxies = {id(row.record) for row in cursor}
        assert len(proxies) == 1
        print(f"perf rows={rows} select_statements={len(statements)} cursor_proxies={len(proxies)}")
