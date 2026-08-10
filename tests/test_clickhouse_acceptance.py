"""Grouped ClickHouse acceptance surfaces built once through bulk_ingest."""

from dataclasses import replace

import pytest
from clickhouse_read_support import clickhouse_database
from test_clickhouse_read import READ_ROWS, ReadRecord
from test_db_entry_provider import ADA, BOOK_1, BOOK_2, BOOLE, CARA, Book, Writer
from test_db_paging import ROWS
from test_db_searcher import LABELS, RECORDS, TAGS, Label, Rec
from test_db_stored_federation import FederatedCalculation, FederationFirst, FederationSecond, _record
from test_db_stored_properties import (
    FIRST,
    SECOND,
    CalculationEntry,
    GenericCalculationFirst,
    GenericCalculationSecond,
)

from httk.data import PageOrder
from httk.data.db import SqlStore, stored_property_sql_plan
from httk.data.db.clickhouse import ClickHouseUnsupportedQueryError
from httk.data.db.entry_provider import StoreEntryProvider
from httk.data.db.stored_federation import StoredEntryFederation, StoredEntrySource
from httk.data.db.stored_properties import StoredPropertySqlPlan

pytestmark = pytest.mark.xdist_group("clickhouse_acceptance")
LITERAL_LABEL = Label("slash\\quote'_%")


@pytest.fixture(scope="module")
def clickhouse_corpus():
    uri = __import__("os").environ.get("HTTK_TEST_CLICKHOUSE_URI")
    if not uri:
        pytest.skip("HTTK_TEST_CLICKHOUSE_URI is not set")
    records = (
        *RECORDS,
        *TAGS,
        *LABELS,
        LITERAL_LABEL,
        *ROWS,
        *READ_ROWS,
        FIRST,
        SECOND,
        ADA,
        BOOLE,
        CARA,
        BOOK_1,
        BOOK_2,
    )
    with clickhouse_database() as database:
        store = SqlStore(
            database, entry_records={CalculationEntry: (GenericCalculationFirst, GenericCalculationSecond)}
        )
        with store.bulk_ingest(finalize="deferred") as bulk:
            for record in records:
                bulk.save(record)
        yield store


@pytest.fixture(scope="module")
def clickhouse_federation():
    with clickhouse_database() as first_database, clickhouse_database() as second_database:
        family_records = {FederatedCalculation: (FederationFirst, FederationSecond)}
        first_store = SqlStore(first_database, entry_records=family_records)
        second_store = SqlStore(second_database, entry_records=family_records)
        first_records = (_record("alpha-first"), _record("alpha-second", second=True))
        second_records = (_record("beta-first"),)
        with first_store.bulk_ingest(finalize="deferred") as first_bulk:
            for record in first_records:
                first_bulk.save(record)
        with second_store.bulk_ingest(finalize="deferred") as second_bulk:
            for record in second_records:
                second_bulk.save(record)
        yield StoredEntryFederation(
            (
                StoredEntrySource(first_store, FederatedCalculation, "alpha", "alpha:"),
                StoredEntrySource(second_store, FederatedCalculation, "beta", "beta:"),
            )
        )


def test_results_rows_projection_and_paging_surface(clickhouse_corpus: SqlStore) -> None:
    searcher = clickhouse_corpus.searcher()
    record = searcher.variable(ReadRecord)
    results = searcher.results(record=record, title=record.title, payload=record.payload)
    page = results.page(size=1, order_by=(PageOrder("title"),))
    assert page.rows[0].payload == b"\x00\xff"
    assert page.next is not None
    assert results.page(size=1, order_by=(PageOrder("title"),), cursor=page.next).rows[0].title == "empty"


def test_live_literal_like_and_scalar_binary_projection(clickhouse_corpus: SqlStore) -> None:
    searcher = clickhouse_corpus.searcher()
    label = searcher.variable(Label)
    searcher.output(label, "label")
    searcher.add(label.text.contains(LITERAL_LABEL.text))
    assert [row[0][0].text for row in searcher] == [LITERAL_LABEL.text]

    binary_search = clickhouse_corpus.searcher()
    record = binary_search.variable(ReadRecord)
    binary_search.output(record.title, "title")
    binary_search.output(record.payload, "payload")
    rows = list(
        binary_search.results(title=record.title, payload=record.payload)
        .page(size=2, order_by=(PageOrder("title"),))
        .rows
    )
    assert {row.title: row.payload for row in rows} == {"50% Mg": b"\x00\xff", "empty": b"binary\x80"}


def test_stored_properties_reject_only_returned_nested_correlation(clickhouse_corpus: SqlStore) -> None:
    plan = stored_property_sql_plan(clickhouse_corpus, CalculationEntry)
    assert [row[0][0].label for row in plan.filter_searchers('immutable_id = "one-third"')[0]] == ["first"]
    for literal in (
        "nested",
        "filtered-count-nested",
        "filtered-count-value-nested",
        "filtered-distinct-count-nested",
        "boolean-nested",
        "boolean-combinators-nested",
    ):
        with pytest.raises(ClickHouseUnsupportedQueryError, match="beyond one immediate scope"):
            plan.filter_searchers(f'immutable_id = "{literal}"')
    assert list(plan.filter_searchers('immutable_id = "filtered-count-single"')[0])
    plan.filter_searchers('immutable_id = "unused-nested"')

    def nested_sort(context):
        return context.count(context.scope("parts").scope("ratios"))

    nested_backings = tuple(
        replace(
            backing,
            projections={
                **backing.projections,
                "immutable_id": replace(backing.projections["immutable_id"], sort=nested_sort),
            },
        )
        for backing in plan._backings
    )
    nested_plan = StoredPropertySqlPlan(
        plan.store,
        plan.family,
        plan.layout,
        plan.entry_type,
        plan.definition,
        nested_backings,
    )
    with pytest.raises(ClickHouseUnsupportedQueryError, match="beyond one immediate scope"):
        nested_plan.candidate_searchers(sort=(("immutable_id", False),))


def test_entry_provider_and_optimade_serving_end_to_end(clickhouse_corpus: SqlStore) -> None:
    provider = StoreEntryProvider(clickhouse_corpus, {"books": Book, "writers": Writer})
    assert [row["__id"] for row in provider.records("books")] == ["books-1", "books-2"]
    pytest.importorskip("httk.serve.optimade")
    from httk.serve.optimade import adapter_from_providers
    from httk.serve.optimade.backend import execute_query
    from httk.serve.optimade.filter import parse_optimade_filter

    adapter = adapter_from_providers([provider])
    rows = list(
        execute_query(
            adapter,
            ["books"],
            ["id", "_httk_custom_title"],
            [],
            100,
            0,
            parse_optimade_filter("_httk_custom_pages > 200"),
        )
    )
    assert [row.values["id"] for row in rows] == ["books-1"]


def test_federation_surface_is_bulk_populated(clickhouse_federation: StoredEntryFederation) -> None:
    page = clickhouse_federation.query(sort=(("immutable_id", False),), limit=10)
    assert [row["immutable_id"] for row in page.rows] == ["alpha-first", "alpha-second", "beta-first"]


def test_query_behavior_surface_handles_synthetic_nulls(clickhouse_corpus: SqlStore) -> None:
    searcher = clickhouse_corpus.searcher()
    record = searcher.variable(Rec)
    searcher.output(record, "record")
    searcher.add(~record.symbols.has_any("missing"))
    assert {row[0][0].formula for row in searcher} == {item.formula for item in RECORDS}


@pytest.mark.parametrize("operation", ("save", "transaction", "ensure_tables"))
def test_clickhouse_d11_matrix(clickhouse_corpus: SqlStore, operation: str) -> None:
    with pytest.raises(RuntimeError, match="clickhousedb bulk-fenced"):
        if operation == "save":
            clickhouse_corpus.save(object())
        elif operation == "transaction":
            with clickhouse_corpus.transaction():
                pass
        else:
            clickhouse_corpus.ensure_tables(Label)
