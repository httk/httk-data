from dataclasses import dataclass
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from httk.core.storage import StorageInfo
from test_db_stored_federation import FederatedCalculation, FederationFirst

from httk.store import FederatedSourceError, FederatedStore
from httk.store.db import Database, SqlStore, StoredEntryFederation, StoredEntrySource


@dataclass(frozen=True)
class AsOfRecord:
    value: int


@dataclass(frozen=True)
class AsOfPair:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="none")

    value: int


def _values(searcher, variable):
    searcher.output(variable, "record")
    return [row[0][0].value for row in searcher]


def test_searcher_as_of_forms_and_boundary(store_factory) -> None:
    store = store_factory()
    store._clock = lambda: 1_000_000
    store.save(AsOfRecord(1))
    store._clock = lambda: 3_000_000
    store.save(AsOfRecord(2))

    for cutoff in (
        2_000_000,
        datetime(1970, 1, 1, 0, 0, 0, 2000, tzinfo=UTC),
        "1970-01-01T00:00:00.002000Z",
    ):
        searcher = store.searcher(as_of=cutoff)
        assert _values(searcher, searcher.variable(AsOfRecord)) == [1]

    exact = store.searcher(as_of=3_000_000)
    assert _values(exact, exact.variable(AsOfRecord)) == [1, 2]
    before = store.searcher(as_of=2_999_000)
    assert _values(before, before.variable(AsOfRecord)) == [1]
    current = store.searcher()
    assert _values(current, current.variable(AsOfRecord)) == [1, 2]


def test_searcher_as_of_constrains_each_variable_and_disabled_capability() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={})
        store._clock = lambda: 1_000_000
        store.save(AsOfPair(1))
        store._clock = lambda: 3_000_000
        store.save(AsOfPair(1))

        searcher = store.searcher(as_of=2_000_000)
        left = searcher.variable(AsOfPair)
        right = searcher.variable(AsOfPair)
        searcher.add(left.value == right.value)
        assert searcher.count() == 1

        assert store.store_timestamps is True
        assert store.store_timestamp_resolution == 1000

    with Database.sqlite() as database:
        disabled = SqlStore(database, entry_records={}, store_timestamps=False)
        assert disabled.store_timestamps is False
        assert disabled.store_timestamp_resolution is None
        with pytest.raises(ValueError, match="as_of|store_timestamps"):
            disabled.searcher(as_of=1)


def test_federated_store_forwards_as_of_and_rejects_disabled_child() -> None:
    with Database.sqlite() as first_database, Database.sqlite() as second_database:
        stores = []
        for database, timestamp, value in (
            (first_database, 1_000_000, 1),
            (second_database, 3_000_000, 2),
        ):
            store = SqlStore(database, entry_records={})
            store._clock = lambda timestamp=timestamp: timestamp
            store.save(AsOfRecord(value))
            stores.append(store)
        federation = FederatedStore({"first": stores[0], "second": stores[1]})
        searcher = federation.searcher(as_of=2_000_000)
        variable = searcher.variable(AsOfRecord)
        assert [row.record.value for row in searcher.results(record=variable)] == [1]

    with Database.sqlite() as first_database, Database.sqlite() as second_database:
        enabled = SqlStore(first_database, entry_records={})
        disabled = SqlStore(second_database, entry_records={}, store_timestamps=False)
        federation = FederatedStore({"enabled": enabled, "disabled": disabled})
        with pytest.raises(FederatedSourceError, match="disabled"):
            federation.searcher(as_of=1).variable(AsOfRecord)


def test_stored_entry_federation_degrades_per_source_for_as_of() -> None:
    with Database.sqlite() as enabled_database, Database.sqlite() as disabled_database:
        enabled = SqlStore(enabled_database, entry_records={FederatedCalculation: FederationFirst})
        enabled._clock = lambda: 1_000_000
        enabled.save(FederationFirst("enabled-old", None))
        enabled._clock = lambda: 3_000_000
        enabled.save(FederationFirst("enabled-new", None))

        disabled = SqlStore(
            disabled_database,
            entry_records={FederatedCalculation: FederationFirst},
            store_timestamps=False,
        )
        disabled.save(FederationFirst("disabled-current", None))

        federation = StoredEntryFederation(
            (
                StoredEntrySource(enabled, FederatedCalculation, "enabled", "enabled:"),
                StoredEntrySource(disabled, FederatedCalculation, "disabled", "disabled:"),
            )
        )
        page = federation.query(as_of=2_000_000, sort=(("immutable_id", False),))
        assert [row["immutable_id"] for row in page.rows] == ["disabled-current", "enabled-old"]


def test_stored_property_cutoff_is_per_query_not_plan_state() -> None:
    with Database.sqlite() as database:
        store = SqlStore(database, entry_records={FederatedCalculation: FederationFirst})
        store._clock = lambda: 1_000_000
        store.save(FederationFirst("old", None))
        store._clock = lambda: 3_000_000
        store.save(FederationFirst("new", None))

        plan = store.stored_property_plan(FederatedCalculation)
        historic = plan.filter_searchers('immutable_id = "old"', as_of=2_000_000)
        assert [row[0][0].label for row in historic[0]] == ["old"]
        assert plan.records().__next__().get("immutable_id") == "old"
        current = plan.candidate_searchers(as_of=None)[0]
        assert current.searcher.count() == 2
