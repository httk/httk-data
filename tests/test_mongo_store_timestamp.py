"""Server-gated MongoStore timestamp parity coverage."""

import datetime
from dataclasses import dataclass

import pytest
from test_db_stored_federation import FederatedCalculation, FederationFirst

from httk.store.backend.sql import StoredEntryFederation, StoredEntrySource
from httk.store.backend.sql.schema import resolve_schema
from httk.store.backend.mongo import MongoStore, StoreClockRegressionError
from httk.store.backend.mongo.mapping import collection_name_for
from httk.store.storage_layout import StorageLayoutUpgradeRequiredError


@dataclass(frozen=True)
class MongoTimestampRecord:
    value: int


@dataclass(frozen=True)
class MongoAsOfLeaf:
    code: str


@dataclass(frozen=True)
class MongoAsOfBranch:
    label: str
    leaf: MongoAsOfLeaf


def test_mongo_timestamp_save_dedup_query_and_sort(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    store._clock = lambda: 1_000_000
    sid = store.save(MongoTimestampRecord(1))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoTimestampRecord))]
    before = collection.find_one({"_id": sid}, {"store_timestamp": 1})
    assert before["store_timestamp"] == 1000

    store._clock = lambda: 2_000_000
    assert store.save(MongoTimestampRecord(1)) == sid
    assert collection.find_one({"_id": sid}, {"store_timestamp": 1}) == before
    store.save(MongoTimestampRecord(2))

    query = store.searcher()
    variable = query.variable(MongoTimestampRecord)
    query.output(variable, "record")
    query.add(variable.store_timestamp <= 1_000_499)
    query.add_sort(variable.store_timestamp)
    assert [row[0][0].value for row in query] == [1]

    scalar = store.searcher()
    scalar_variable = scalar.variable(MongoTimestampRecord)
    scalar.output(scalar_variable.store_timestamp, "stamp")
    assert [row[0][0] for row in scalar] == [1_000_000, 2_000_000]

    for operand in (
        1_000_499,
        datetime.datetime(1970, 1, 1, 0, 0, 0, 1000, tzinfo=datetime.UTC),
        "1970-01-01T00:00:00.001000Z",
    ):
        candidate = store.searcher()
        item = candidate.variable(MongoTimestampRecord)
        candidate.output(item, "record")
        candidate.add(item.store_timestamp <= operand)
        assert list(candidate)
    with pytest.raises(ValueError, match="timezone-aware"):
        _ = variable.store_timestamp <= datetime.datetime(1970, 1, 1)  # noqa: DTZ001


def test_mongo_as_of_reference_lookup_and_pagination(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    leaf = MongoAsOfLeaf("visible")
    store._clock = lambda: 1_000_000
    store.save(leaf)
    store.save(MongoAsOfBranch("old-1", leaf))
    store._clock = lambda: 1_500_000
    store.save(MongoAsOfBranch("old-2", leaf))
    store._clock = lambda: 3_000_000
    store.save(MongoAsOfBranch("new", leaf))

    searcher = store.searcher(as_of=2_000_000)
    branch = searcher.variable(MongoAsOfBranch)
    searcher.add(branch.leaf.code == "visible")
    searcher.add_sort(branch.label)
    searcher.set_limit(1)
    searcher.add_offset(1)
    searcher.output(branch, "record")

    assert searcher.count() == 2
    assert [row[0][0].label for row in searcher] == ["old-2"]


def test_mongo_timestamp_layout_guard_and_clock_regression(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    assert (
        mongo_test_database.database["_httk_store_metadata"].find_one({"_id": "layout"})["store_timestamps"]
        == "v1:1000"
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
        MongoStore(mongo_test_database, store_timestamps=False)
    mongo_test_database.database["_httk_store_metadata"].update_one(
        {"_id": "layout"}, {"$set": {"store_timestamps": "v1:01000"}}
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError, match="store_timestamps"):
        MongoStore(mongo_test_database, entry_records={})
    mongo_test_database.database["_httk_store_metadata"].update_one(
        {"_id": "layout"}, {"$set": {"store_timestamps": "v1:1000"}}
    )

    store._clock = lambda: 10_000
    store.save(MongoTimestampRecord(1))
    reopened = MongoStore(mongo_test_database, entry_records={})
    reopened._clock = lambda: 9_000
    with pytest.raises(StoreClockRegressionError, match="ns"):
        reopened.save(MongoTimestampRecord(2))


def test_mongo_timestamp_fsck_future_and_clamp(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    store._clock = lambda: 1_000_000_000
    sid = store.save(MongoTimestampRecord(1))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoTimestampRecord))]
    collection.update_one({"_id": sid}, {"$set": {"store_timestamp": 10_000_000}})
    store = MongoStore(mongo_test_database, entry_records={})
    store._clock = lambda: 1_000_000_000
    report = store.fsck(repair=False, collect_garbage=False, known_types=(MongoTimestampRecord,))
    assert report.violations and "store_timestamp" in report.violations[0]
    repaired = store.fsck(
        repair=True,
        collect_garbage=False,
        clamp_future_timestamps=True,
        known_types=(MongoTimestampRecord,),
    )
    assert "clamped" in repaired.violations[0]
    assert store.fsck(repair=False, collect_garbage=False, known_types=(MongoTimestampRecord,)).violations == ()
    store._clock = lambda: 1_500_000_000
    store.save(MongoTimestampRecord(2))


def test_mongo_timestamp_transaction_pins_and_rollback_keeps_mark(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={})
    if not store._database.supports_transactions:
        pytest.skip("MongoDB transactions are unavailable")
    store._clock = lambda: 1_000_000
    with store.transaction():
        store.save(MongoTimestampRecord(1))
        store._clock = lambda: 2_000_000
        with store.transaction():
            store.save(MongoTimestampRecord(2))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoTimestampRecord))]
    assert [item["store_timestamp"] for item in collection.find({}, {"store_timestamp": 1}).sort("_id", 1)] == [
        1000,
        1000,
    ]

    store._clock = lambda: 3_000_000
    with pytest.raises(RuntimeError), store.transaction():
        store.save(MongoTimestampRecord(3))
        raise RuntimeError("rollback")
    store._clock = lambda: 1_500_000
    store.save(MongoTimestampRecord(4))


def test_mongo_timestamp_hwm_ignores_unrelated_collection(mongo_test_database):
    MongoStore(mongo_test_database, entry_records={})
    mongo_test_database.database["unrelated"].insert_one({"store_timestamp": 10_000_000})
    reopened = MongoStore(mongo_test_database, entry_records={})
    reopened._clock = lambda: 1_000_000
    reopened.save(MongoTimestampRecord(1))


def test_mongo_timestamp_nondivisor_future_limit(mongo_test_database):
    store = MongoStore(mongo_test_database, entry_records={}, store_timestamp_resolution=3)
    store._clock = lambda: 1
    sid = store.save(MongoTimestampRecord(1))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoTimestampRecord))]
    collection.update_one({"_id": sid}, {"$set": {"store_timestamp": 666_666_667}})
    assert store.fsck(repair=False, collect_garbage=False, known_types=(MongoTimestampRecord,)).violations == ()
    collection.update_one({"_id": sid}, {"$set": {"store_timestamp": 666_666_668}})
    report = store.fsck(repair=False, collect_garbage=False, known_types=(MongoTimestampRecord,))
    assert report.violations and "store_timestamp" in report.violations[0]


def test_mongo_carried_federation_timestamp_skips_lookup(mongo_test_database, monkeypatch):
    store = MongoStore(mongo_test_database, entry_records={FederatedCalculation: FederationFirst})
    store._clock = lambda: 1_000_000
    record = FederationFirst("carried", None)
    sid = store.save(record)
    collection = mongo_test_database.database[collection_name_for(resolve_schema(FederationFirst))]
    calls = 0
    original_find_one = collection.find_one

    def count_find_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_find_one(*args, **kwargs)

    monkeypatch.setattr(collection, "find_one", count_find_one)
    store.fetch(FederationFirst, sid)
    fetch_calls = calls
    calls = 0
    federation = StoredEntryFederation((StoredEntrySource(store, FederatedCalculation, "source", "source:"),))
    page = federation.query(sort=(("_httk_store_timestamp", False),), limit=1)
    assert calls == fetch_calls
    assert page.rows[0]["_httk_store_timestamp"] == 1_000_000
