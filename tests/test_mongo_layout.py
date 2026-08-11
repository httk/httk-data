"""Live stamp, trust, and collection-preparation coverage for MongoStore."""

from dataclasses import dataclass
from typing import ClassVar

import pytest
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import StorageInfo

from httk.store.mongo import MongoStore
from httk.store.mongo.mapping import METADATA_COLLECTION
from httk.store.storage_layout import StorageLayoutUpgradeRequiredError


class MongoLayoutFamily:
    """Test family for the Mongo layout tests."""


class MongoOtherFamily:
    """Second test family for declaration mismatch coverage."""


@dataclass(frozen=True)
class MongoLayoutRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_layout_record")

    value: str


@dataclass(frozen=True)
class MongoOtherRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_other_record")

    value: str


register_entry_family(name="test-mongo-layout-family", family=f"{__name__}:MongoLayoutFamily")
register_entry_record(
    name="test-mongo-layout-record", family="test-mongo-layout-family", record=f"{__name__}:MongoLayoutRecord"
)
register_entry_family(name="test-mongo-other-family", family=f"{__name__}:MongoOtherFamily")
register_entry_record(
    name="test-mongo-other-record", family="test-mongo-other-family", record=f"{__name__}:MongoOtherRecord"
)


def test_first_open_stamps_six_keys_and_reopen_trusts(mongo_test_database) -> None:
    """The single layout document is canonical and byte-stable."""
    store = MongoStore(mongo_test_database, entry_records={})
    document = mongo_test_database.database[METADATA_COLLECTION].find_one({"_id": "layout"})
    assert document is not None
    assert set(document) == {
        "_id",
        "protocol",
        "entry_declaration",
        "document_layout",
        "generation",
        "store_timestamps",
    }
    assert document["protocol"] == "v2.1.0"
    assert document["document_layout"] == "mongo-v2"
    assert MongoStore(mongo_test_database).layout == store.layout


def test_old_protocol_stamp_is_refused_on_reopen(mongo_test_database) -> None:
    """A Mongo store stamped by the previous format cannot be adopted."""
    MongoStore(mongo_test_database, entry_records={})
    mongo_test_database.database[METADATA_COLLECTION].update_one(
        {"_id": "layout"},
        {"$set": {"protocol": "v2.0.3", "document_layout": "mongo-v1"}},
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database)
    assert error.value.diff["protocol"]["actual"] == {
        "protocol": "v2.0.3",
        "document_layout": "mongo-v1",
    }


def test_missing_entry_records_is_rejected_on_first_open(mongo_test_database) -> None:
    """An empty uninitialized database needs an explicit declaration."""
    with pytest.raises(TypeError, match="entry_records"):
        MongoStore(mongo_test_database)


def test_supplied_declaration_mismatch_has_structured_diff(mongo_test_database) -> None:
    """Reopen declarations are compared as canonical JSON bytes."""
    MongoStore(mongo_test_database, entry_records={MongoLayoutFamily: MongoLayoutRecord})
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database, entry_records={MongoOtherFamily: MongoOtherRecord})
    assert "declaration" in error.value.diff
    assert error.value.diff["declaration"]["expected"] != error.value.diff["declaration"]["actual"]


def test_unversioned_database_is_refused_with_reserved_and_unversioned_entries(mongo_test_database) -> None:
    """Existing collections without the marker cannot be adopted."""
    mongo_test_database.database.create_collection("ordinary_collection")
    mongo_test_database.database.create_collection("_httk_foreign")
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database, entry_records={})
    schema = error.value.diff["schema"]
    assert schema["ordinary_collection"]["unversioned"] is True
    assert schema["_httk_foreign"]["reserved"] is True


def test_ensure_collections_is_idempotent_and_installs_validator(mongo_test_database) -> None:
    """Collection setup is observable and safe to repeat."""
    store = MongoStore(mongo_test_database, entry_records={})
    store.ensure_collections(MongoLayoutRecord)
    store.ensure_collections(MongoLayoutRecord)
    options = mongo_test_database.database["mongo_layout_record"].options()
    assert "$jsonSchema" in options["validator"]
    assert {index["name"] for index in mongo_test_database.database["mongo_layout_record"].list_indexes()} >= {
        "_id_",
        "uq_mongo_layout_record_content_id",
        "ix_mongo_layout_record__httk_role",
    }
