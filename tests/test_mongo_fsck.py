"""Live integrity-repair and mark/sweep coverage for :mod:`httk.store.mongo`."""

import os
import random
from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo, content_id

from httk.store.db.schema import resolve_schema
from httk.store.mongo import MongoDatabase, MongoStore
from httk.store.mongo.mapping import (
    COUNTERS_COLLECTION,
    METADATA_COLLECTION,
    collection_name_for,
    counter_next,
    entry_dispatch_table_name,
)


@dataclass(frozen=True)
class FsckLeaf:
    value: str


@dataclass(frozen=True)
class FsckHolder:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    label: str
    leaves: list[FsckLeaf]


@dataclass(frozen=True)
class FsckReference:
    leaf: FsckLeaf
    leaves: list[FsckLeaf]


@dataclass(frozen=True)
class FsckChain:
    value: str
    next: "FsckChain | None"


@dataclass(frozen=True)
class FsckPrivateRoot:
    leaf: FsckLeaf


class FsckFamily:
    """Test-only entry family."""


@dataclass(frozen=True)
class FsckEntryA:
    value: str


@dataclass(frozen=True)
class FsckEntryB:
    value: str


@dataclass(frozen=True)
class FsckEntryHolder:
    entries: list[FsckEntryB]


register_entry_family(name="test-mongo-fsck-family", family=f"{__name__}:FsckFamily")
register_entry_record(name="test-mongo-fsck-a", family="test-mongo-fsck-family", record=f"{__name__}:FsckEntryA")
register_entry_record(name="test-mongo-fsck-b", family="test-mongo-fsck-family", record=f"{__name__}:FsckEntryB")


def _store(database) -> MongoStore:
    return MongoStore(database, entry_records={FsckFamily: (FsckEntryA, FsckEntryB)})


def _insert_orphan_leaf(database, leaf: FsckLeaf) -> int:
    """Insert a valid dependency document simulating a parent-write crash."""
    schema = resolve_schema(FsckLeaf)
    sid = counter_next(database.database, schema.table_name)
    database.database[collection_name_for(schema)].insert_one(
        {
            "_id": sid,
            "content_id": content_id(leaf),
            "_httk_role": "dep",
            "store_timestamp": 0,
            "f": {"value": leaf.value},
        }
    )
    return sid


def test_fsck_sweeps_orphans_preserves_reachable_and_bumps_generation(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    live = FsckLeaf("live")
    orphan = FsckLeaf("orphan")
    store.save(FsckHolder("same", [live]))
    orphan_sid = _insert_orphan_leaf(mongo_test_database, orphan)
    main_sid = store.save(FsckLeaf("main"))
    leaves = mongo_test_database.database[collection_name_for(resolve_schema(FsckLeaf))]
    orphan_document = leaves.find_one({"content_id": content_id(orphan)})
    assert orphan_document is not None and orphan_document["_httk_role"] == "dep"
    before = mongo_test_database.database[METADATA_COLLECTION].find_one({"_id": "layout"})["generation"]
    counters_before = list(mongo_test_database.database[COUNTERS_COLLECTION].find({}))

    summary = store.fsck()

    assert summary.generation == before + 1
    assert leaves.find_one({"content_id": content_id(orphan)}) is None
    assert leaves.find_one({"content_id": content_id(live)}) is not None
    assert leaves.find_one({"_id": main_sid}) is not None  # unreferenced mains are roots.
    assert list(mongo_test_database.database[COUNTERS_COLLECTION].find({})) == counters_before
    assert summary.collections[collection_name_for(resolve_schema(FsckLeaf))].deleted >= 1
    assert store.save(FsckLeaf("after-fsck")) > max(main_sid, orphan_sid)


def test_fsck_marks_reference_child_and_shared_dependency_graphs(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    shared = FsckLeaf("shared")
    child_only = FsckLeaf("child-only")
    first = store.save(FsckReference(shared, [child_only, shared]))
    second = store.save(FsckReference(shared, []))
    chain = FsckChain("root", FsckChain("middle", FsckChain("leaf", None)))
    chain_sid = store.save(chain)

    store.fsck()

    reopened_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    reopened = _store(reopened_database)
    try:
        assert reopened.fetch(FsckReference, first) == FsckReference(shared, [child_only, shared])
        assert reopened.fetch(FsckReference, second) == FsckReference(shared, [])
        assert reopened.fetch(FsckChain, chain_sid) == chain
        leaf_collection = mongo_test_database.database[collection_name_for(resolve_schema(FsckLeaf))]
        assert (
            leaf_collection.count_documents({"content_id": {"$in": [content_id(shared), content_id(child_only)]}}) == 2
        )
    finally:
        reopened_database.dispose()


def test_fsck_repairs_only_main_family_dispatches_and_reports_conflicts(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    main = FsckEntryA("main")
    main_sid = store.save(main)
    dependency = FsckEntryB("dependency")
    store.save(FsckEntryHolder([dependency]))
    dispatch_name = entry_dispatch_table_name(store.entry_layout[0].name)
    dispatch = mongo_test_database.database[dispatch_name]
    dispatch.delete_one({"_id": content_id(main)})

    repaired = store.fsck()

    assert dispatch.find_one({"_id": content_id(main)}) == {
        "_id": content_id(main),
        "record": "test-mongo-fsck-a",
        "sid": main_sid,
    }
    assert dispatch.find_one({"_id": content_id(dependency)}) is None
    assert repaired.collections[dispatch_name].repaired == 1

    dispatch.insert_one({"_id": "0" * 64, "record": "test-mongo-fsck-a", "sid": 987654321})
    reported = store.fsck()
    assert dispatch.find_one({"_id": "0" * 64}) is not None
    assert reported.collections[dispatch_name].conflicts >= 1
    repaired_conflict = store.fsck(repair_conflicts=True)
    assert dispatch.find_one({"_id": "0" * 64}) is None
    assert repaired_conflict.collections[dispatch_name].deleted >= 1


def test_fsck_generation_invalidates_another_store_cache(mongo_test_database) -> None:
    first = _store(mongo_test_database)
    second_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    second = _store(second_database)
    try:
        leaf = FsckLeaf("cached")
        sid = first.save(leaf)
        assert second.fetch(FsckLeaf, sid) == leaf
        first.fsck(collect_garbage=False)
        second.save(FsckLeaf("next"))
        assert (FsckLeaf, sid) not in second._identity._instances
    finally:
        second_database.dispose()


def test_fsck_reports_and_preserves_unattributed_collection(mongo_test_database) -> None:
    store = _store(mongo_test_database)
    foreign = mongo_test_database.database["foreign_fsck_data"]
    foreign.insert_one({"_id": 1, "_httk_role": "dep", "f": {}})

    summary = store.fsck()

    assert foreign.find_one({"_id": 1}) is not None
    assert any("foreign_fsck_data" in violation and "sweep" in violation for violation in summary.violations)


def test_fsck_aborts_sweep_until_reopened_private_type_is_supplied(mongo_test_database) -> None:
    first = _store(mongo_test_database)
    live = FsckLeaf("private-live")
    root = FsckPrivateRoot(live)
    root_sid = first.save(root)
    leaves = mongo_test_database.database[collection_name_for(resolve_schema(FsckLeaf))]
    orphan = FsckLeaf("private-orphan")
    orphan_sid = _insert_orphan_leaf(mongo_test_database, orphan)
    reopened_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    reopened = _store(reopened_database)
    try:
        blocked = reopened.fsck()
        assert leaves.find_one({"_id": orphan_sid}) is not None
        assert any(
            "fsck_private_root" in violation and "blocks fsck sweep" in violation for violation in blocked.violations
        )
        assert any("sweep aborted" in violation and "known_types" in violation for violation in blocked.violations)

        proceeded = reopened.fsck(known_types=(FsckPrivateRoot,))
        assert leaves.find_one({"_id": orphan_sid}) is None
        assert leaves.find_one({"content_id": content_id(live)}) is not None
        assert reopened.fetch(FsckPrivateRoot, root_sid) == root
        assert all("sweep aborted" not in violation for violation in proceeded.violations)
    finally:
        reopened_database.dispose()


@pytest.mark.parametrize("rounds", [6, pytest.param(24, marks=pytest.mark.extended)])
def test_fsck_randomized_graphs_retain_reachable_documents(mongo_test_database, rounds: int) -> None:
    """Seeded normal/extended randomized graph coverage without a new dependency."""
    seed = 20260808
    rng = random.Random(seed)
    store = _store(mongo_test_database)
    reopened_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    reopened = _store(reopened_database)
    leaves = mongo_test_database.database[collection_name_for(resolve_schema(FsckLeaf))]
    try:
        for round_number in range(rounds):
            live = [FsckLeaf(f"{round_number}-live-{index}") for index in range(rng.randrange(1, 6))]
            orphan = [FsckLeaf(f"{round_number}-orphan-{index}") for index in range(rng.randrange(1, 5))]
            label = f"round-{round_number}"
            store.save(FsckHolder(label, live))
            for leaf in orphan:
                _insert_orphan_leaf(mongo_test_database, leaf)
            summary = store.fsck()
            assert summary.generation == round_number + 1, f"seed={seed}, round={round_number}"
            for leaf in live:
                assert leaves.find_one({"content_id": content_id(leaf)}) is not None, (
                    f"seed={seed}, round={round_number}"
                )
            for leaf in orphan:
                assert leaves.find_one({"content_id": content_id(leaf)}) is None, f"seed={seed}, round={round_number}"
            # A fresh store must hydrate the surviving graph rather than serving a cache hit.
            sid = store.save(FsckHolder(label, live))
            assert reopened.fetch(FsckHolder, sid) == FsckHolder(label, live)
    finally:
        reopened_database.dispose()
