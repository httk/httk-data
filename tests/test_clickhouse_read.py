"""P4 ClickHouse read-only surface over one bulk-populated fixture."""

import os
from dataclasses import dataclass
from fractions import Fraction

import pytest
import sqlalchemy
from test_clickhouse_bulk import _clickhouse_bulk_database
from test_db_stored_properties import (
    FIRST,
    SECOND,
    CalculationEntry,
    GenericCalculationFirst,
    GenericCalculationSecond,
)

from httk.data import PageOrder, UnsupportedQueryError
from httk.data.db import SqlStore, stored_property_sql_plan
from httk.data.db.clickhouse import ClickHouseUnsupportedQueryError
from httk.data.db.optimade import optimade_filter_searcher

pytestmark = pytest.mark.xdist_group("clickhouse_read_corpus")


@dataclass(frozen=True)
class ReadReference:
    doi: str


@dataclass(frozen=True)
class ReadRecord:
    title: str
    payload: bytes
    score: float
    tags: list[str]
    reference: ReadReference | None = None


READ_ROWS = (
    ReadRecord("50% Mg", b"\x00\xff", 1.0, ["common", "red"], ReadReference("10.1/a")),
    ReadRecord("empty", b"binary\x80", 2.0, [], None),
)


@pytest.fixture(scope="module")
def clickhouse_read_store():
    """Build once with ``bulk_ingest``; all tests below are read-only."""
    uri = os.environ.get("HTTK_TEST_CLICKHOUSE_URI")
    if not uri:
        pytest.skip("HTTK_TEST_CLICKHOUSE_URI is not set")
    with _clickhouse_bulk_database(uri) as database:
        store = SqlStore(
            database,
            entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)},
        )
        with store.bulk_ingest(finalize="deferred") as bulk:
            for record in READ_ROWS:
                bulk.save(record)
            bulk.save(FIRST)
            bulk.save(SECOND)
        yield store


def _read_search(store: SqlStore):
    searcher = store.searcher()
    record = searcher.variable(ReadRecord)
    searcher.output(record, "record")
    return searcher, record


def test_clickhouse_search_rows_bytes_and_synthetic_null_semantics(clickhouse_read_store: SqlStore) -> None:
    searcher, record = _read_search(clickhouse_read_store)
    searcher.add(record.title.contains("50%"))
    assert [item[0][0].payload for item in searcher] == [b"\x00\xff"]

    empty_search, empty_record = _read_search(clickhouse_read_store)
    empty_search.add(empty_record.reference == None)  # query DSL NULL predicate
    empty_search.add(~empty_record.tags.has_any("missing"))
    assert {item[0][0].title for item in empty_search} == {"empty"}


def test_clickhouse_results_paging_and_reopen_are_stable(clickhouse_read_store: SqlStore) -> None:
    searcher, record = _read_search(clickhouse_read_store)
    results = searcher.results(record=record, title=record.title, score=record.score)
    order = (PageOrder("title"),)
    first = results.page(size=1, order_by=order)
    labels = [row.title for row in first.rows]
    while first.next is not None:
        first = results.page(size=1, order_by=order, cursor=first.next)
        labels.extend(row.title for row in first.rows)
    assert labels == ["50% Mg", "empty"]

    reopened = SqlStore(clickhouse_read_store._database)
    fresh_search, fresh_record = _read_search(reopened)
    fresh_results = fresh_search.results(record=fresh_record, title=fresh_record.title, score=fresh_record.score)
    page = fresh_results.page(size=1, order_by=order)
    assert page.rows[0].title == "50% Mg"


def test_clickhouse_optimade_and_stored_property_read_paths(clickhouse_read_store: SqlStore) -> None:
    optimade = optimade_filter_searcher(clickhouse_read_store, ReadRecord, '_httk_custom_title CONTAINS "50%"')
    assert [item[0][0].title for item in optimade] == ["50% Mg"]

    plan = stored_property_sql_plan(clickhouse_read_store, CalculationEntry)
    assert [item[0][0].label for item in plan.filter_searchers('immutable_id = "one-third"')[0]] == ["first"]
    with pytest.raises(ClickHouseUnsupportedQueryError, match="beyond one immediate scope"):
        plan.filter_searchers('immutable_id = "nested"')
    with pytest.raises(UnsupportedQueryError):
        plan.filter_searchers('immutable_id = "composition"')


@pytest.mark.parametrize(
    ("left", "left_factor", "right", "right_factor"),
    (
        (Fraction(1, 3), 2, Fraction(2, 3), 1),
        (Fraction(1, 3), 1, Fraction(2, 3), 1),
        (Fraction(-7, 11), Fraction(5, 2), Fraction(-35, 22), 1),
        (Fraction(0), Fraction(1, 17), Fraction(0), Fraction(99, 1)),
    ),
)
def test_clickhouse_fraction_property_battery(
    clickhouse_read_store: SqlStore,
    left: Fraction,
    left_factor: Fraction | int,
    right: Fraction,
    right_factor: Fraction | int,
) -> None:
    with clickhouse_read_store._database.engine.connect() as connection:
        result = connection.execute(
            sqlalchemy.select(
                sqlalchemy.func.httk_fraction_scaled_equal(
                    f"{left.numerator}/{left.denominator}",
                    f"{left_factor.numerator}/{left_factor.denominator}"
                    if isinstance(left_factor, Fraction)
                    else left_factor,
                    f"{right.numerator}/{right.denominator}",
                    f"{right_factor.numerator}/{right_factor.denominator}"
                    if isinstance(right_factor, Fraction)
                    else right_factor,
                )
            )
        ).scalar_one()
    assert bool(result) is (left * left_factor == right * right_factor)


def test_clickhouse_fraction_overflow_and_zero_denominator_throw(clickhouse_read_store: SqlStore) -> None:
    with clickhouse_read_store._database.engine.connect() as connection:
        with pytest.raises(Exception, match="zero denominator"):
            connection.execute(
                sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal("1/0", 1, "1/1", 1))
            ).scalar_one()
        with pytest.raises(Exception, match="19 digits"):
            connection.execute(
                sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal("123456789012345678901/1", 1, "1/1", 1))
            ).scalar_one()


def test_clickhouse_fraction_boundary_parameter_and_excluded_row_battery(clickhouse_read_store: SqlStore) -> None:
    signed_nineteen = "-" + "9" * 19
    with clickhouse_read_store._database.engine.connect() as connection:
        boundary = connection.execute(
            sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal(signed_nineteen, 1, signed_nineteen, 1))
        ).scalar_one()
        assert bool(boundary)

        parameterized = sqlalchemy.select(
            sqlalchemy.func.httk_fraction_scaled_equal(
                sqlalchemy.bindparam("left"),
                sqlalchemy.bindparam("left_factor"),
                sqlalchemy.bindparam("right"),
                sqlalchemy.bindparam("right_factor"),
            )
        )
        assert connection.execute(
            parameterized,
            {"left": "1/3", "left_factor": "2", "right": "2/3", "right_factor": "1"},
        ).scalar_one()

        for oversized in ("1/" + "8" * 20, "-" + "7" * 20 + "/1"):
            with pytest.raises(Exception, match="19 digits"):
                connection.execute(
                    sqlalchemy.select(sqlalchemy.func.httk_fraction_scaled_equal(oversized, 1, "1/1", 1))
                ).scalar_one()

        guarded_fraction = sqlalchemy.func.httk_fraction_scaled_equal("1/" + "8" * 20, 1, "1/1", 1)
        guarded_sql = str(guarded_fraction.compile(dialect=connection.dialect, compile_kwargs={"literal_binds": True}))
        excluded = connection.execute(
            sqlalchemy.text(f"SELECT count() FROM numbers(2) WHERE number = 0 AND if(number = 0, 1, {guarded_sql}) = 1")
        ).scalar_one()
        assert excluded == 1

        with pytest.raises(Exception, match="19 digits"):
            connection.execute(
                sqlalchemy.text(
                    f"SELECT count() FROM numbers(2) WHERE number = 1 AND if(number = 0, 1, {guarded_sql}) = 1"
                )
            ).scalar_one()


def test_clickhouse_select_fetch_is_refused_without_synthesized_limit(clickhouse_read_store: SqlStore) -> None:
    statement = sqlalchemy.select(sqlalchemy.literal(1)).fetch(3).offset(7)
    with pytest.raises(ClickHouseUnsupportedQueryError, match="FETCH"):
        statement.compile(dialect=clickhouse_read_store._database.engine.dialect)
