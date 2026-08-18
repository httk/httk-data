"""Live stamp, trust, and collection-preparation coverage for MongoStore."""

import json
from dataclasses import dataclass
from typing import Annotated, ClassVar

import pytest
from httk.core.register import register_entry_family, register_entry_record
from httk.core.storage import IdentitySkip, StorageInfo
from schema_override_support import schema_override

from httk.store import EntryFamilyDeclaration, EntryLayoutBindingError, EntryRecordDeclaration
from httk.store.backend.mongo import MongoStore
from httk.store.backend.mongo.mapping import METADATA_COLLECTION
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


class LocalMongoFamily:
    """Application-owned Mongo family which is deliberately not registered."""


@dataclass(frozen=True)
class LocalMongoRecord:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="local_mongo_record")

    value: str


LOCAL_MONGO_LAYOUT = EntryFamilyDeclaration(
    name="test-local-mongo-family",
    family=LocalMongoFamily,
    records=(EntryRecordDeclaration(name="test-local-mongo-record", record=LocalMongoRecord),),
)
register_entry_family(name="test-mongo-other-family", family=f"{__name__}:MongoOtherFamily")
register_entry_record(
    name="test-mongo-other-record", family="test-mongo-other-family", record=f"{__name__}:MongoOtherRecord"
)


class MongoRefFamily:
    """Registered Mongo family whose record references another storable class."""


@dataclass(frozen=True)
class MongoRefChild:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_ref_child")

    tag: str


@dataclass(frozen=True)
class MongoRefParent:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(storage_name="mongo_ref_parent")

    child: MongoRefChild
    value: str


register_entry_family(name="test-mongo-ref-family", family=f"{__name__}:MongoRefFamily")
register_entry_record(name="test-mongo-ref-record", family="test-mongo-ref-family", record=f"{__name__}:MongoRefParent")


class MongoUpgradeFamily:
    """Application-owned Mongo family used by the additive-upgrade tests."""


@dataclass(frozen=True)
class MongoRecOld:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="mongo_upgrade_rec", identity_name="mongo-upgrade-rec"
    )

    value: str


@dataclass(frozen=True)
class MongoRecNew:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="mongo_upgrade_rec", identity_name="mongo-upgrade-rec"
    )

    value: str
    note: Annotated[str | None, IdentitySkip()] = None


@dataclass(frozen=True)
class MongoRecNewRequired:
    __httk_storage__: ClassVar[StorageInfo] = StorageInfo(
        storage_name="mongo_upgrade_rec", identity_name="mongo-upgrade-rec"
    )

    value: str
    tag: Annotated[str, IdentitySkip()] = "x"


def _upgrade_decl(record: type) -> EntryFamilyDeclaration:
    return EntryFamilyDeclaration(
        name="test-mongo-upgrade-family",
        family=MongoUpgradeFamily,
        records=(EntryRecordDeclaration(name="test-mongo-upgrade-record", record=record),),
    )


def test_first_open_stamps_seven_keys_and_reopen_trusts(mongo_test_database) -> None:
    """The single layout document is canonical and byte-stable."""
    store = MongoStore(mongo_test_database, entry_records={})
    document = mongo_test_database.database[METADATA_COLLECTION].find_one({"_id": "layout"})
    assert document is not None
    assert set(document) == {
        "_id",
        "protocol",
        "entry_declaration",
        "entry_schemas",
        "document_layout",
        "generation",
        "store_timestamps",
    }
    assert document["protocol"] == "v2.2.0"
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
    entry_declaration = error.value.diff["declaration"]["entry_declaration"]
    assert entry_declaration["expected"] != entry_declaration["actual"]


def test_application_owned_declaration_rebinds_without_registration(mongo_test_database) -> None:
    """A local family is rebound explicitly rather than imported through discovery."""
    store = MongoStore(mongo_test_database, entry_families=(LOCAL_MONGO_LAYOUT,))
    assert store.entry_layout[0].family is LocalMongoFamily
    assert store.entry_layout[0].records == (LocalMongoRecord,)
    assert MongoStore(mongo_test_database, entry_families=(LOCAL_MONGO_LAYOUT,)).layout == store.layout
    with pytest.raises(EntryLayoutBindingError, match="entry_families"):
        MongoStore(mongo_test_database)


def test_unversioned_database_is_refused_with_reserved_and_unversioned_entries(mongo_test_database) -> None:
    """Existing collections without the marker cannot be adopted."""
    mongo_test_database.database.create_collection("ordinary_collection")
    mongo_test_database.database.create_collection("_httk_foreign")
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database, entry_records={})
    schema = error.value.diff["schema"]
    assert schema["ordinary_collection"]["unversioned"] is True
    assert schema["_httk_foreign"]["reserved"] is True


def test_reopen_with_unchanged_schema_succeeds(mongo_test_database) -> None:
    """A store whose record classes are unchanged reopens on the trusted fingerprint."""
    MongoStore(mongo_test_database, entry_records={MongoRefFamily: MongoRefParent})
    reopened = MongoStore(mongo_test_database)
    assert {family.name for family in reopened.entry_layout} == {"test-mongo-ref-family"}


def test_reopen_with_changed_referenced_schema_is_rejected(mongo_test_database) -> None:
    """A real resolution change on a referenced class is rejected, naming its collection."""
    MongoStore(mongo_test_database, entry_records={MongoRefFamily: MongoRefParent})
    # A real resolution-path change: the referenced child's dedup policy differs
    # from what was stamped, moving its fingerprint without hand-editing JSON.
    with (
        schema_override(MongoRefChild, StorageInfo(storage_name="mongo_ref_child", dedup="by_value")),
        pytest.raises(StorageLayoutUpgradeRequiredError) as error,
    ):
        MongoStore(mongo_test_database)
    # The offending table is the referenced child, not the declared parent.
    assert set(error.value.diff["schema"]) == {"mongo_ref_child"}


def test_reopen_with_hand_edited_fingerprint_names_the_table(mongo_test_database) -> None:
    """A hand-edited stored fingerprint produces a per-table schema diff."""
    MongoStore(mongo_test_database, entry_records={MongoLayoutFamily: MongoLayoutRecord})
    document = mongo_test_database.database[METADATA_COLLECTION].find_one({"_id": "layout"})
    assert document is not None
    stored = json.loads(document["entry_schemas"])
    # Simulate the record's stored column type having changed since creation.
    stored["tables"]["mongo_layout_record"]["fields"]["value"]["columns"][0]["kind"] = "int"
    mongo_test_database.database[METADATA_COLLECTION].update_one(
        {"_id": "layout"},
        {"$set": {"entry_schemas": json.dumps(stored, sort_keys=True, separators=(",", ":"))}},
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database)
    # Only the changed table is named, not the unrelated ones.
    assert set(error.value.diff["schema"]) == {"mongo_layout_record"}


def test_reopen_with_corrupt_fingerprint_reports_sentinel(mongo_test_database) -> None:
    """A stored fingerprint that is not parseable yields the single sentinel diff."""
    MongoStore(mongo_test_database, entry_records={MongoLayoutFamily: MongoLayoutRecord})
    mongo_test_database.database[METADATA_COLLECTION].update_one(
        {"_id": "layout"}, {"$set": {"entry_schemas": "not a fingerprint"}}
    )
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database)
    assert set(error.value.diff["schema"]) == {"<fingerprint>"}


def test_additive_reopen_without_upgrade_raises_with_hint(mongo_test_database) -> None:
    """A purely additive Mongo mismatch points at upgrade=True instead of applying it."""
    MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecOld),))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecNew),))
    assert "upgrade=True" in str(error.value)
    assert error.value.hint is not None
    assert set(error.value.diff["schema"]) == {"mongo_upgrade_rec"}


def test_additive_reopen_with_upgrade_restamps_and_reads_old_rows(mongo_test_database) -> None:
    """Documents are schemaless: upgrade re-stamps the fingerprint and old rows read with the new field None."""
    store = MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecOld),))
    sid = store.save(MongoRecOld("kept"))

    upgraded = MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecNew),), upgrade=True)
    assert upgraded.fetch(MongoRecNew, sid) == MongoRecNew("kept", None)
    new_sid = upgraded.save(MongoRecNew("fresh", "annotated"))
    assert upgraded.fetch(MongoRecNew, new_sid) == MongoRecNew("fresh", "annotated")

    # A plain reopen now trusts the re-stamped fingerprint.
    reopened = MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecNew),))
    assert reopened.fetch(MongoRecNew, sid) == MongoRecNew("kept", None)


def test_additive_upgrade_rejects_non_nullable_added_field(mongo_test_database) -> None:
    """A non-additive change still raises under upgrade=True, without re-stamping."""
    MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecOld),))
    with pytest.raises(StorageLayoutUpgradeRequiredError) as error:
        MongoStore(mongo_test_database, entry_families=(_upgrade_decl(MongoRecNewRequired),), upgrade=True)
    assert set(error.value.diff["schema"]) == {"mongo_upgrade_rec"}


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
