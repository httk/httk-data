"""Latest-only (``only_latest``) query rewriting for the SQL searcher stack.

Mirrors :mod:`test_as_of`: the ``store_factory`` fixture exercises sqlite and
duckdb, ``store._clock`` controls the store timestamp, and ``_values`` reads a
single variable output. The Mongo store is a sibling packet's concern, so the
SQL-only cases skip it.
"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo

from httk.store import FederatedStore
from httk.store.backend.sql import Backend, SqlStore, StoreEntryProvider


@dataclass(frozen=True)
class LatestWidget:
    value: int


@dataclass(frozen=True)
class LatestNote:
    text: str


@dataclass(frozen=True)
class LatestHolder:
    label: str
    note: LatestNote | None = None


@dataclass(frozen=True)
class LatestPair:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    value: int


def _skip_non_sql(store: object) -> None:
    if not isinstance(store, SqlStore):
        pytest.skip("only_latest is a SQL-layer feature; the Mongo store is a sibling packet's concern")


def _values(searcher, variable) -> list[int]:
    searcher.output(variable, "record")
    return sorted(row[0][0].value for row in searcher)


def test_only_latest_returns_only_the_latest_of_a_lineage(store_factory) -> None:
    store = store_factory()
    _skip_non_sql(store)
    first = LatestWidget(1)
    store.save(first)
    store.replace(first, LatestWidget(2))
    store.save(LatestWidget(99))  # an un-replaced, independent lineage

    plain = store.searcher()
    assert _values(plain, plain.variable(LatestWidget)) == [1, 2, 99]

    latest = store.searcher(only_latest=True)
    assert _values(latest, latest.variable(LatestWidget)) == [2, 99]


def test_only_latest_chained_replacement_keeps_only_the_last(store_factory) -> None:
    store = store_factory()
    _skip_non_sql(store)
    first = LatestWidget(1)
    store.save(first)
    b = store.replace(first, LatestWidget(2))
    store.replace(store.fetch(LatestWidget, b), LatestWidget(3))

    latest = store.searcher(only_latest=True)
    assert _values(latest, latest.variable(LatestWidget)) == [3]


def test_only_latest_with_as_of_returns_the_latest_as_of_t(store_factory) -> None:
    store = store_factory()
    _skip_non_sql(store)
    first = LatestWidget(1)
    store._clock = lambda: 1_000_000
    store.save(first)
    store._clock = lambda: 3_000_000
    store.replace(first, LatestWidget(2))

    # Between the two writes only the first row exists, so it is the latest
    # as of T even though it was later replaced.
    historic = store.searcher(as_of=2_000_000, only_latest=True)
    assert _values(historic, historic.variable(LatestWidget)) == [1]

    # After the replacement the second row is the latest.
    current = store.searcher(as_of=3_000_000, only_latest=True)
    assert _values(current, current.variable(LatestWidget)) == [2]


def test_only_latest_without_timestamps() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={}, store_timestamps=False)
        assert store.store_timestamps is False
        first = LatestWidget(1)
        store.save(first)
        store.replace(first, LatestWidget(2))

        latest = store.searcher(only_latest=True)
        assert _values(latest, latest.variable(LatestWidget)) == [2]


def test_only_latest_self_join_filters_each_variable_independently() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        # Two lineages, each replaced once.
        a = LatestPair(1)
        store.save(a)
        store.replace(a, LatestPair(2))
        c = LatestPair(10)
        store.save(c)
        store.replace(c, LatestPair(20))

        # Two variables of the same class each get their own NOT EXISTS: a join
        # equating them keeps only the two latest-of-lineage rows paired with
        # themselves, never with a replaced sibling.
        searcher = store.searcher(only_latest=True)
        left = searcher.variable(LatestPair)
        right = searcher.variable(LatestPair)
        searcher.add(left.value == right.value)
        assert searcher.count() == 2  # (2, 2) and (20, 20), not the replaced 1/10

        # And each variable in isolation yields exactly the two latest rows.
        solo = store.searcher(only_latest=True)
        assert _values(solo, solo.variable(LatestPair)) == [2, 20]


def test_only_latest_does_not_filter_reference_variables(store_factory) -> None:
    store = store_factory()
    _skip_non_sql(store)
    old_note = LatestNote("first")
    store.save(old_note)
    store.save(LatestHolder("h", old_note))
    # Replace the referenced note; the holder still pins the replaced row's sid.
    store.replace(old_note, LatestNote("second"))

    searcher = store.searcher(only_latest=True)
    holder = searcher.variable(LatestHolder)
    searcher.output(holder.note.text, "note_text")
    # The holder is the latest of its own lineage, and its reference join is
    # UNFILTERED, so it resolves the replaced (old) note.
    assert [row[0][0] for row in searcher] == ["first"]

    # A predicate over the reference likewise still matches the replaced row.
    filtered = store.searcher(only_latest=True)
    holder = filtered.variable(LatestHolder)
    filtered.add(holder.note.text == "first")
    filtered.output(holder, "record")
    assert [row[0][0].label for row in filtered] == ["h"]


def test_variable_logical_id_in_filter_and_output(store_factory) -> None:
    store = store_factory()
    _skip_non_sql(store)
    first = LatestWidget(1)
    lineage = store.save(first)  # a fresh row's logical_id is its own sid
    store.replace(first, LatestWidget(2))
    store.save(LatestWidget(99))  # a separate lineage

    searcher = store.searcher()
    variable = searcher.variable(LatestWidget)
    searcher.add(variable.logical_id == lineage)
    searcher.output(variable.value, "value")
    searcher.output(variable.logical_id, "lid")
    rows = [(row[0][0], row[0][1]) for row in searcher]
    assert sorted(value for value, _lid in rows) == [1, 2]
    assert {lid for _value, lid in rows} == {lineage}


def test_entry_provider_only_latest_serves_only_latest_rows() -> None:
    with Backend.sqlite() as database:
        store = SqlStore(database, entry_records={})
        first = LatestWidget(1)
        store.save(first)
        store.replace(first, LatestWidget(2))
        store.save(LatestWidget(99))

        plain = StoreEntryProvider(store, {"widgets": LatestWidget})
        assert sorted(row["_httk_custom_value"] for row in plain.records("widgets")) == [1, 2, 99]

        latest = StoreEntryProvider(store, {"widgets": LatestWidget}, only_latest=True)
        assert sorted(row["_httk_custom_value"] for row in latest.records("widgets")) == [2, 99]


def test_federated_store_only_latest_fan_out() -> None:
    with Backend.sqlite() as first_database, Backend.sqlite() as second_database:
        stores = []
        for database, base in ((first_database, 1), (second_database, 10)):
            store = SqlStore(database, entry_records={})
            first = LatestWidget(base)
            store.save(first)
            store.replace(first, LatestWidget(base + 1))
            stores.append(store)
        federation = FederatedStore({"first": stores[0], "second": stores[1]})

        plain = federation.searcher()
        variable = plain.variable(LatestWidget)
        assert sorted(row.record.value for row in plain.results(record=variable)) == [1, 2, 10, 11]

        latest = federation.searcher(only_latest=True)
        variable = latest.variable(LatestWidget)
        assert sorted(row.record.value for row in latest.results(record=variable)) == [2, 11]
