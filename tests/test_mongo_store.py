"""MongoStore-specific degraded-mode guarantees."""

import os
from dataclasses import dataclass
from threading import Barrier, Thread
from typing import Annotated, ClassVar

import pytest
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import Skip, StorageInfo, content_id, stored_property

from httk.data.db.schema import resolve_schema
from httk.data.mongo import MongoDatabase, MongoStore, RecordTooLargeError
from httk.data.mongo.mapping import collection_name_for, entry_dispatch_table_name
from httk.data.store_common import EntryDispatchIntegrityError


@dataclass(frozen=True)
class MongoRoleDependency:
    value: str


@dataclass(frozen=True)
class MongoRoleContainer:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    label: str
    children: list[MongoRoleDependency]


@dataclass(frozen=True)
class MongoByValueDependency:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    value: str


@dataclass(frozen=True)
class MongoByValueParent:
    children: list[MongoByValueDependency]


@dataclass(frozen=True)
class MongoOptionalByValue:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    label: str
    note: str | None


@dataclass(frozen=True)
class MongoOptionalDerivedByValue:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(dedup="by_value")

    label: str
    include: Annotated[bool, Skip()] = False

    @stored_property
    def marker(self) -> str | None:
        return "present" if self.include else None


@dataclass(frozen=True)
class MongoDerivedRecord:
    values: list[str]

    @stored_property
    def count(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class MongoRequiredNone:
    value: str


class MongoDispatchFamily:
    """Test-only multi-record family."""


@dataclass(frozen=True)
class MongoDispatchA:
    value: str


@dataclass(frozen=True)
class MongoDispatchB:
    value: str


register_entry_family(name="test-mongo-store-family", family=f"{__name__}:MongoDispatchFamily")
register_entry_record(name="test-mongo-store-a", family="test-mongo-store-family", record=f"{__name__}:MongoDispatchA")
register_entry_record(name="test-mongo-store-b", family="test-mongo-store-family", record=f"{__name__}:MongoDispatchB")


def _store(database, *, entry_records=None):
    return MongoStore(database, entry_records=entry_records or {})


def test_role_marker_lifecycle_and_promotion(mongo_test_database):
    store = _store(mongo_test_database)
    dependency = MongoRoleDependency("dep")
    store.save(MongoRoleContainer("same", [dependency]))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoRoleDependency))]
    assert collection.find_one({"content_id": content_id(dependency)})["_httk_role"] == "dep"
    store.save(dependency)
    assert collection.find_one({"content_id": content_id(dependency)})["_httk_role"] == "main"


def test_by_value_top_level_hit_promotes_dependency_role(mongo_test_database):
    store = _store(mongo_test_database)
    dependency = MongoByValueDependency("dep")
    store.save(MongoByValueParent([dependency]))
    collection = mongo_test_database.database[collection_name_for(resolve_schema(MongoByValueDependency))]
    sid = store.sid_of(dependency)
    assert sid is not None
    assert collection.find_one({"_id": sid})["_httk_role"] == "dep"
    assert store.save(dependency) == sid
    assert collection.find_one({"_id": sid})["_httk_role"] == "main"


def test_by_value_optional_parent_none_is_not_a_wildcard(mongo_test_database):
    store = _store(mongo_test_database)
    different = store.save(MongoOptionalByValue("same", "different"))
    missing = store.save(MongoOptionalByValue("same", None))
    assert missing != different


def test_by_value_optional_derived_none_is_not_a_wildcard(mongo_test_database):
    store = _store(mongo_test_database)
    present = store.save(MongoOptionalDerivedByValue("same", True))
    missing = store.save(MongoOptionalDerivedByValue("same", False))
    assert missing != present


def test_derived_property_is_encoded_in_raw_document(mongo_test_database):
    store = _store(mongo_test_database)
    record = MongoDerivedRecord(["a", "b"])
    sid = store.save(record)
    document = mongo_test_database.database[collection_name_for(resolve_schema(MongoDerivedRecord))].find_one(
        {"_id": sid}
    )
    assert document["f"]["count"] == 2


def test_non_optional_none_is_rejected(mongo_test_database):
    with pytest.raises(ValueError, match="MongoRequiredNone.value"):
        _store(mongo_test_database).save(MongoRequiredNone(None))


def test_concurrent_fetches_use_independent_hydration_contexts(mongo_test_database):
    store = _store(mongo_test_database)
    record = MongoRoleContainer("hydration", [MongoRoleDependency("child")])
    sid = store.save(record)
    reader_database = MongoDatabase.connect(
        os.environ["HTTK_TEST_MONGODB_URI"], database=mongo_test_database.database.name, transactions="never"
    )
    reader = _store(reader_database)
    barrier = Barrier(2)
    results: list[MongoRoleContainer] = []
    errors: list[BaseException] = []

    def fetch() -> None:
        try:
            barrier.wait()
            results.append(reader.fetch(MongoRoleContainer, sid))
        except BaseException as error:
            errors.append(error)

    try:
        threads = [Thread(target=fetch), Thread(target=fetch)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        reader_database.dispose()
    assert not errors
    assert results == [record, record]


def test_dispatch_missing_is_detected_and_resaved(mongo_test_database):
    store = _store(mongo_test_database, entry_records={MongoDispatchFamily: (MongoDispatchA, MongoDispatchB)})
    record = MongoDispatchA("value")
    sid = store.save(record)
    key = content_id(record)
    dispatch = mongo_test_database.database[entry_dispatch_table_name(store.entry_layout[0].name)]
    dispatch.delete_one({"_id": key})
    with pytest.raises(EntryDispatchIntegrityError):
        store.fetch_entry(MongoDispatchFamily, key)
    assert store.save(record) == sid
    assert dispatch.find_one({"_id": key}) == {"_id": key, "record": "test-mongo-store-a", "sid": sid}


def test_content_id_race_returns_one_sid(mongo_test_database):
    uri = os.environ["HTTK_TEST_MONGODB_URI"]
    first = _store(mongo_test_database)
    second_db = MongoDatabase.connect(uri, database=mongo_test_database.database.name, transactions="never")
    second = _store(second_db)
    try:
        barrier = Barrier(2)
        results: list[int] = []
        errors: list[BaseException] = []

        def save(store):
            try:
                barrier.wait()
                results.append(store.save(MongoRoleDependency("race")))
            except BaseException as error:
                errors.append(error)

        threads = [Thread(target=save, args=(first,)), Thread(target=save, args=(second,))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        assert results[0] == results[1]
        assert (
            mongo_test_database.database[collection_name_for(resolve_schema(MongoRoleDependency))].count_documents({})
            == 1
        )
    finally:
        second_db.dispose()


def test_oversized_record_is_rejected_before_parent_insert(mongo_test_database):
    @dataclass(frozen=True)
    class HugeRecord:
        payload: list[str]

    store = _store(mongo_test_database)
    with pytest.raises(RecordTooLargeError, match="HugeRecord"):
        store.save(HugeRecord(["x" * (17 * 1024 * 1024)]))
    assert mongo_test_database.database[collection_name_for(resolve_schema(HugeRecord))].count_documents({}) == 0


@pytest.mark.parametrize("transactions", ["auto", "never"])
def test_by_value_hit_compensation_depends_on_mode(mongo_test_database, transactions):
    database = mongo_test_database
    owned_database = None
    if transactions == "never":
        name = f"httk_test_degraded_{id(mongo_test_database)}"
        database = MongoDatabase.connect(os.environ["HTTK_TEST_MONGODB_URI"], database=name, transactions="never")
        owned_database = database
    try:
        store = _store(database)
        first = MongoRoleDependency("first")
        second = MongoRoleDependency("second")
        assert store.save(MongoRoleContainer("same", [first])) == store.save(MongoRoleContainer("same", [second]))
        dependency_collection = database.database[collection_name_for(resolve_schema(MongoRoleDependency))]
        expected = 1 if transactions == "auto" else 2
        assert (
            dependency_collection.count_documents({"content_id": {"$in": [content_id(first), content_id(second)]}})
            == expected
        )
    finally:
        if owned_database is not None:
            owned_database.client.drop_database(owned_database.database.name)
            owned_database.dispose()


def test_record_document_shape(mongo_test_database):
    store = _store(mongo_test_database)
    record = MongoRoleDependency("shape")
    sid = store.save(record)
    document = mongo_test_database.database[collection_name_for(resolve_schema(MongoRoleDependency))].find_one(
        {"_id": sid}
    )
    assert set(document) == {"_id", "content_id", "_httk_role", "f"}
