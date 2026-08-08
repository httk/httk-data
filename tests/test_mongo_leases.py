"""Live coverage for the MongoStore writer/fsck lease handshake."""

import threading
import time
from dataclasses import dataclass

import pytest

from httk.data.db.schema import resolve_schema
from httk.data.mongo import MongoStore, StoreLockedError
from httk.data.mongo.leases import LeaseLostError, LeaseTiming, acquire_fsck, acquire_writer
from httk.data.mongo.mapping import METADATA_COLLECTION, collection_name_for


@dataclass(frozen=True)
class LeaseRecord:
    value: str


_FAST = LeaseTiming(refresh_interval=0.04, stale_multiplier=2, poll_initial=0.005, poll_max=0.01)


def test_writer_is_blocked_by_fresh_and_stale_fsck_leases(mongo_test_database) -> None:
    store = MongoStore(mongo_test_database, entry_records={})
    metadata = mongo_test_database.database[METADATA_COLLECTION]
    metadata.insert_one({"_id": "lease/fsck", "owner": "test", "heartbeat": None})
    with pytest.raises(StoreLockedError):
        store.save(LeaseRecord("fresh"))
    metadata.update_one({"_id": "lease/fsck"}, [{"$set": {"heartbeat": "$$NOW"}}])
    time.sleep(_FAST.stale_after + 0.02)
    with pytest.raises(StoreLockedError):
        store.save(LeaseRecord("stale"))


def test_fsck_waits_for_registered_writer_and_is_singleton(mongo_test_database) -> None:
    database = mongo_test_database.database
    MongoStore(mongo_test_database, entry_records={})
    writer = acquire_writer(database, timing=_FAST)
    acquired = threading.Event()
    result: list[object] = []

    def acquire() -> None:
        try:
            result.append(acquire_fsck(database, timing=_FAST))
            acquired.set()
        except BaseException as error:  # pragma: no cover - assertion below reports it.
            result.append(error)

    thread = threading.Thread(target=acquire)
    thread.start()
    time.sleep(0.03)
    assert not acquired.is_set()
    writer.release()
    thread.join(timeout=2)
    assert acquired.is_set()
    fsck = result[0]
    assert not isinstance(fsck, BaseException)
    try:
        with pytest.raises(StoreLockedError):
            acquire_fsck(database, timing=_FAST)
        with pytest.raises(StoreLockedError):
            acquire_writer(database, timing=_FAST)
    finally:
        fsck.release()  # type: ignore[union-attr]


def test_force_fsck_can_ignore_stale_writer_residue(mongo_test_database) -> None:
    database = mongo_test_database.database
    MongoStore(mongo_test_database, entry_records={})
    writer = acquire_writer(database, timing=_FAST)
    time.sleep(_FAST.stale_after + 0.02)
    with pytest.raises(StoreLockedError, match="stale writer"):
        acquire_fsck(database, timing=_FAST)
    fsck = acquire_fsck(database, force=True, timing=_FAST)
    try:
        assert database[METADATA_COLLECTION].find_one({"_id": writer.identifier}) is not None
    finally:
        fsck.release()
        writer.release()


def test_force_cannot_replace_fresh_fsck_and_displaced_owner_detects_loss(mongo_test_database) -> None:
    database = mongo_test_database.database
    MongoStore(mongo_test_database, entry_records={})
    fsck = acquire_fsck(database, timing=_FAST)
    try:
        with pytest.raises(StoreLockedError, match="fresh"):
            acquire_fsck(database, force=True, timing=_FAST)
        database[METADATA_COLLECTION].update_one({"_id": "lease/fsck"}, {"$set": {"owner": "replacement"}})
        with pytest.raises(LeaseLostError, match="lost"):
            fsck.refresh_heartbeat(force=True)
    finally:
        database[METADATA_COLLECTION].delete_one({"_id": "lease/fsck"})


def test_generation_change_clears_identity_caches(mongo_test_database) -> None:
    store = MongoStore(mongo_test_database, entry_records={})
    first = LeaseRecord("first")
    sid = store.save(first)
    assert store.fetch(LeaseRecord, sid) is first
    mongo_test_database.database[METADATA_COLLECTION].update_one({"_id": "layout"}, {"$inc": {"generation": 1}})
    store.save(LeaseRecord("second"))
    assert (LeaseRecord, sid) not in store._identity._instances
    assert mongo_test_database.database[collection_name_for(resolve_schema(LeaseRecord))].count_documents({}) == 2
